"""
This file implements the main training loop, tensorboard statistics tracking, etc...
"""

import copy
from contextlib import contextmanager
import importlib
import json
import math
import random
import sys
import time
import typing
from collections import defaultdict
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torchrl.data.replay_buffers import PrioritizedSampler

from config_files import config_copy
from trackmania_rl import buffer_management, utilities
from trackmania_rl.agents import iqn as iqn
from trackmania_rl.agents.iqn import make_untrained_iqn_network
from trackmania_rl.analysis_metrics import (
    distribution_curves,
    highest_prio_transitions,
    loss_distribution,
    race_time_left_curves,
    tau_curves,
)
from trackmania_rl.buffer_utilities import make_buffers, resize_buffers
from trackmania_rl.gen_v3_stage import StageController
from trackmania_rl.map_reference_times import reference_times


def _is_finite_scalar(value: typing.Any) -> bool:
    """Return whether a scalar-like value is finite, failing closed on errors."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, RuntimeError):
        return False


@contextmanager
def _guard_non_finite_optimizer_step(optimizer, scaler):
    """Block the underlying optimizer step unless this batch is fully finite.

    ``Trainer.train_on_batch`` owns the backward/clip/step sequence, so the
    learner cannot validate the returned values after the fact.  This scoped
    guard observes the loss passed to ``GradScaler.scale`` and the exact value
    returned by ``clip_grad_norm_`` before allowing the optimizer's real
    ``step`` method to run.
    """

    state = {
        "loss_finite": None,
        "grad_norm": None,
        "blocked": False,
        "reasons": [],
    }
    original_optimizer_step = optimizer.step
    original_scaler_scale = scaler.scale
    original_clip_grad_norm = torch.nn.utils.clip_grad_norm_

    def guarded_scale(outputs):
        state["loss_finite"] = _is_finite_scalar(outputs)
        return original_scaler_scale(outputs)

    def guarded_clip_grad_norm(*args, **kwargs):
        grad_norm = original_clip_grad_norm(*args, **kwargs)
        state["grad_norm"] = float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
        return grad_norm

    def guarded_optimizer_step(*args, **kwargs):
        reasons = []
        if state["loss_finite"] is not True:
            reasons.append("loss_non_finite")
        if state["grad_norm"] is None or not _is_finite_scalar(state["grad_norm"]):
            reasons.append("gradient_norm_non_finite")
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter_group in optimizer.param_groups
            for parameter in parameter_group["params"]
        )
        if not gradients_finite:
            reasons.append("gradient_values_non_finite")
        if reasons:
            state["blocked"] = True
            state["reasons"] = reasons
            return None
        return original_optimizer_step(*args, **kwargs)

    optimizer.step = guarded_optimizer_step
    scaler.scale = guarded_scale
    torch.nn.utils.clip_grad_norm_ = guarded_clip_grad_norm
    try:
        yield state
    finally:
        optimizer.step = original_optimizer_step
        scaler.scale = original_scaler_scale
        torch.nn.utils.clip_grad_norm_ = original_clip_grad_norm


def _non_finite_batch_reason(loss, grad_norm, guard_state) -> str | None:
    """Return a stable terminal reason for a rejected learner batch."""

    reasons = list(guard_state.get("reasons", [])) if guard_state.get("blocked") else []
    if not _is_finite_scalar(loss):
        reasons.append("loss_non_finite")
    if not _is_finite_scalar(grad_norm):
        reasons.append("gradient_norm_non_finite")
    return ", ".join(dict.fromkeys(reasons)) or None


def learner_process_fn(
    rollout_queues,
    uncompiled_shared_network,
    shared_network_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tensorboard_base_dir: Path,
    stop_event: mp.Event,
):
    layout_version = "lay_mono"
    SummaryWriter(log_dir=str(tensorboard_base_dir / layout_version)).add_custom_scalars(
        {
            layout_version: {
                # "eval_agg_ratio": [
                #     "Multiline",
                #     [
                #         "eval_agg_ratio_trained_author",
                #         "eval_agg_ratio_blind_author",
                #     ],
                # ],
                # "eval_ratio_trained_author": [
                #     "Multiline",
                #     [
                #         "eval_ratio_trained_author",
                #     ],
                # ],
                # "eval_ratio_blind_author": [
                #     "Multiline",
                #     [
                #         "eval_ratio_blind_author",
                #     ],
                # ],
                "eval_race_time_robust": [
                    "Multiline",
                    [
                        "eval_race_time_robust",
                    ],
                ],
                "explo_race_time_finished": [
                    "Multiline",
                    [
                        "explo_race_time_finished",
                    ],
                ],
                "loss": ["Multiline", ["loss$", "loss_test$"]],
                "avg_Q": [
                    "Multiline",
                    ["avg_Q"],
                ],
                "single_zone_reached": [
                    "Multiline",
                    [
                        "single_zone_reached",
                    ],
                ],
                "grad_norm_history": [
                    "Multiline",
                    [
                        "grad_norm_history_d9",
                        "grad_norm_history_d98",
                    ],
                ],
                "priorities": [
                    "Multiline",
                    [
                        "priorities",
                    ],
                ],
            },
        }
    )

    # ========================================================
    # Create new stuff
    # ========================================================

    online_network, uncompiled_online_network = make_untrained_iqn_network(config_copy.use_jit, is_inference=False)
    target_network, _ = make_untrained_iqn_network(config_copy.use_jit, is_inference=False)

    print(online_network)
    utilities.count_parameters(online_network)

    accumulated_stats: defaultdict[str, typing.Any] = defaultdict(int)
    accumulated_stats["alltime_min_ms"] = {}
    accumulated_stats["rolling_mean_ms"] = {}
    previous_alltime_min = None
    time_last_save = time.perf_counter()
    queue_check_order = list(range(len(rollout_queues)))
    rollout_queue_readers = [q._reader for q in rollout_queues]
    time_waited_for_workers_since_last_tensorboard_write = 0
    time_training_since_last_tensorboard_write = 0
    time_testing_since_last_tensorboard_write = 0

    # ========================================================
    # Load existing stuff
    # ========================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utilities.load_model_checkpoint(
        online_network,
        save_dir / "weights1.torch",
        map_location=device,
        allowed_missing_keys=utilities.NON_PERSISTENT_NORMALIZATION_KEYS,
        allowed_unexpected_keys=utilities.NON_PERSISTENT_NORMALIZATION_KEYS,
    )
    utilities.load_model_checkpoint(
        target_network,
        save_dir / "weights2.torch",
        map_location=device,
        allowed_missing_keys=utilities.NON_PERSISTENT_NORMALIZATION_KEYS,
        allowed_unexpected_keys=utilities.NON_PERSISTENT_NORMALIZATION_KEYS,
    )
    print(" =====================     Learner weights loaded !     ============================")

    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

    # noinspection PyBroadException
    try:
        accumulated_stats = joblib.load(save_dir / "accumulated_stats.joblib")
        shared_steps.value = accumulated_stats["cumul_number_frames_played"]
        print(" =====================      Learner stats loaded !      ============================")
    except:
        print(" Learner could not load stats")

    if "rolling_mean_ms" not in accumulated_stats.keys():
        # Temporary to preserve compatibility with old runs that doesn't have this feature. To be removed later.
        accumulated_stats["rolling_mean_ms"] = {}

    stage_baseline_dir = save_dir / "stage_baseline" if str(config_copy.run_name).startswith("gen_v3") else None
    if stage_baseline_dir is not None and stage_baseline_dir.exists():
        stage_start_frames = utilities.load_stage_baseline_start_frames(stage_baseline_dir)
    else:
        stage_start_frames = shared_steps.value
    stage_controller = StageController(stage_start_frames) if stage_baseline_dir is not None else None

    def save_stage_report(decision: str, reason: str | None) -> None:
        assert stage_controller is not None
        if decision == "rollback" and stage_baseline_dir is not None:
            utilities.restore_stage_baseline(stage_baseline_dir, save_dir)
        report = {
            "stage": getattr(config_copy, "gen_v3_stage_name", "unknown"),
            "decision": decision,
            "reason": reason,
            "stage_start_frames": stage_controller.stage_start_frames,
            "stage_frames": stage_controller.stage_frames(accumulated_stats["cumul_number_frames_played"]),
            "baseline_new_map_finish_rate": stage_controller.baseline_new_map_finish_rate,
            "metrics": stage_controller.last_metrics,
        }
        (save_dir / "gen_v3_stage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    accumulated_stats["cumul_number_single_memories_should_have_been_used"] = accumulated_stats["cumul_number_single_memories_used"]
    transitions_learned_last_save = accumulated_stats["cumul_number_single_memories_used"]
    neural_net_reset_counter = 0
    single_reset_flag = config_copy.single_reset_flag

    optimizer1 = torch.optim.RAdam(
        online_network.parameters(),
        lr=utilities.from_exponential_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_frames_played"]),
        eps=config_copy.adam_epsilon,
        betas=(config_copy.adam_beta1, config_copy.adam_beta2),
    )
    # optimizer1 = torch_optimizer.Lookahead(optimizer1, k=5, alpha=0.5)

    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    memory_size, memory_size_start_learn = utilities.from_staircase_schedule(
        config_copy.memory_size_schedule, accumulated_stats["cumul_number_frames_played"]
    )
    buffer, buffer_test = make_buffers(memory_size)
    offset_cumul_number_single_memories_used = memory_size_start_learn * config_copy.number_times_single_memory_is_used_before_discard

    # noinspection PyBroadException
    try:
        optimizer1.load_state_dict(torch.load(f=save_dir / "optimizer1.torch", map_location=device, weights_only=False))
        scaler.load_state_dict(torch.load(f=save_dir / "scaler.torch", map_location=device, weights_only=False))
        print(" =========================     Optimizer loaded !     ================================")
    except:
        print(" Could not load optimizer")

    if stage_controller is not None:
        stage_baseline_dir = utilities.create_stage_baseline(
            stage_baseline_dir,
            online_network,
            target_network,
            optimizer1,
            scaler,
            accumulated_stats,
        )
        print(f"Stage baseline ready at {stage_baseline_dir}")

    tensorboard_suffix = utilities.from_staircase_schedule(
        config_copy.tensorboard_suffix_schedule,
        accumulated_stats["cumul_number_frames_played"],
    )
    tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name + tensorboard_suffix)))

    loss_history = []
    loss_test_history = []
    train_on_batch_duration_history = []
    grad_norm_history = []
    layer_grad_norm_history = defaultdict(list)

    # ========================================================
    # Make the trainer
    # ========================================================
    trainer = iqn.Trainer(
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer1,
        scaler=scaler,
        batch_size=config_copy.batch_size,
        iqn_n=config_copy.iqn_n,
    )

    inferer = iqn.Inferer(
        inference_network=online_network,
        iqn_k=config_copy.iqn_k,
        tau_epsilon_boltzmann=config_copy.tau_epsilon_boltzmann,
    )

    while True:  # Trainer loop
        before_wait_time = time.perf_counter()
        wait(rollout_queue_readers)
        time_waited = time.perf_counter() - before_wait_time
        if time_waited > 1:
            print(f"Warning: learner waited {time_waited:.2f} seconds for workers to provide memories")
        time_waited_for_workers_since_last_tensorboard_write += time_waited
        for idx in queue_check_order:
            if not rollout_queues[idx].empty():
                (
                    rollout_results,
                    end_race_stats,
                    fill_buffer,
                    is_explo,
                    map_name,
                    map_status,
                    rollout_duration,
                    loop_number,
                    rollout_valid,
                ) = rollout_queues[idx].get()
                queue_check_order.append(queue_check_order.pop(queue_check_order.index(idx)))
                break

        importlib.reload(config_copy)

        episode_metrics = end_race_stats.get("episode_metrics", {})
        if stage_controller is not None and episode_metrics:
            episode_record = {
                "map": map_name,
                "is_evaluation": not is_explo,
                "cumulative_frames": accumulated_stats["cumul_number_frames_played"],
                "metrics": episode_metrics,
            }
            with (save_dir / "gen_v3_episode_metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(episode_record, sort_keys=True) + "\n")
            stage_decision = stage_controller.record_episode(
                map_name,
                episode_metrics,
                accumulated_stats["cumul_number_frames_played"],
                is_evaluation=not is_explo,
            )
            if stage_decision is not None:
                if stage_decision == "accepted":
                    utilities.save_checkpoint(save_dir, online_network, target_network, optimizer1, scaler)
                    joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")
                save_stage_report(stage_decision, stage_controller.reason)
                stop_event.set()

        if stop_event.is_set():
            break

        if not rollout_valid:
            print(f"Skipping crashed rollout for {map_name}; infrastructure failure recorded")
            continue

        new_tensorboard_suffix = utilities.from_staircase_schedule(
            config_copy.tensorboard_suffix_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        if new_tensorboard_suffix != tensorboard_suffix:
            tensorboard_suffix = new_tensorboard_suffix
            tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name + tensorboard_suffix)))

        (
            new_memory_size,
            new_memory_size_start_learn,
        ) = utilities.from_staircase_schedule(
            config_copy.memory_size_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        if new_memory_size != memory_size:
            buffer, buffer_test = resize_buffers(buffer, buffer_test, new_memory_size)
            offset_cumul_number_single_memories_used += (
                new_memory_size_start_learn - memory_size_start_learn
            ) * config_copy.number_times_single_memory_is_used_before_discard
            memory_size_start_learn = new_memory_size_start_learn
            memory_size = new_memory_size
        # ===============================================
        #   VERY BASIC TRAINING ANNEALING
        # ===============================================

        # LR and weight_decay calculation
        learning_rate = utilities.from_exponential_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_frames_played"])
        weight_decay = config_copy.weight_decay_lr_ratio * learning_rate
        engineered_speedslide_reward = utilities.from_linear_schedule(
            config_copy.engineered_speedslide_reward_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        engineered_neoslide_reward = utilities.from_linear_schedule(
            config_copy.engineered_neoslide_reward_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        engineered_kamikaze_reward = utilities.from_linear_schedule(
            config_copy.engineered_kamikaze_reward_schedule, accumulated_stats["cumul_number_frames_played"]
        )
        engineered_close_to_vcp_reward = utilities.from_linear_schedule(
            config_copy.engineered_close_to_vcp_reward_schedule, accumulated_stats["cumul_number_frames_played"]
        )
        gamma = utilities.from_linear_schedule(config_copy.gamma_schedule, accumulated_stats["cumul_number_frames_played"])

        # ===============================================
        #   RELOAD
        # ===============================================

        for param_group in optimizer1.param_groups:
            param_group["lr"] = learning_rate
            param_group["epsilon"] = config_copy.adam_epsilon
            param_group["betas"] = (config_copy.adam_beta1, config_copy.adam_beta2)

        if isinstance(buffer._sampler, PrioritizedSampler):
            buffer._sampler._alpha = config_copy.prio_alpha
            buffer._sampler._beta = config_copy.prio_beta
            buffer._sampler._eps = config_copy.prio_epsilon

        if config_copy.plot_race_time_left_curves and not is_explo and (loop_number // 5) % 17 == 0:
            race_time_left_curves(rollout_results, inferer, save_dir, map_name)
            tau_curves(rollout_results, inferer, save_dir, map_name)
            distribution_curves(buffer, save_dir, online_network, target_network)
            loss_distribution(buffer, save_dir, online_network, target_network)
            # patrick_curves(rollout_results, trainer, save_dir, map_name)

        accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])

        if stage_controller is not None and stage_controller.budget_exhausted(accumulated_stats["cumul_number_frames_played"]):
            save_stage_report(stage_controller.decision or "budget_exhausted", stage_controller.reason)
            stop_event.set()
            break

        # ===============================================
        #   WRITE SINGLE RACE RESULTS TO TENSORBOARD
        # ===============================================
        race_stats_to_write = {
            f"race_time_ratio_{map_name}": end_race_stats["race_time_for_ratio"] / (rollout_duration * 1000),
            f"explo_race_time_{map_status}_{map_name}" if is_explo else f"eval_race_time_{map_status}_{map_name}": end_race_stats[
                "race_time"
            ]
            / 1000,
            f"explo_race_finished_{map_status}_{map_name}" if is_explo else f"eval_race_finished_{map_status}_{map_name}": end_race_stats[
                "race_finished"
            ],
            f"mean_action_gap_{map_name}": -(
                np.array(rollout_results["q_values"]) - np.array(rollout_results["q_values"]).max(axis=1, initial=None).reshape(-1, 1)
            ).mean(),
            f"single_zone_reached_{map_status}_{map_name}": rollout_results["furthest_zone_idx"],
            "instrumentation__answer_normal_step": end_race_stats["instrumentation__answer_normal_step"],
            "instrumentation__answer_action_step": end_race_stats["instrumentation__answer_action_step"],
            "instrumentation__between_run_steps": end_race_stats["instrumentation__between_run_steps"],
            "instrumentation__grab_frame": end_race_stats["instrumentation__grab_frame"],
            "instrumentation__convert_frame": end_race_stats["instrumentation__convert_frame"],
            "instrumentation__grab_floats": end_race_stats["instrumentation__grab_floats"],
            "instrumentation__exploration_policy": end_race_stats["instrumentation__exploration_policy"],
            "instrumentation__request_inputs_and_speed": end_race_stats["instrumentation__request_inputs_and_speed"],
            "tmi_protection_cutoff": end_race_stats["tmi_protection_cutoff"],
            "worker_time_in_rollout_percentage": rollout_results["worker_time_in_rollout_percentage"],
        }
        print("Race time ratio  ", race_stats_to_write[f"race_time_ratio_{map_name}"])

        if not is_explo:
            race_stats_to_write[f"avg_Q_{map_status}_{map_name}"] = np.mean(rollout_results["q_values"])

        if end_race_stats["race_finished"]:
            race_stats_to_write[f"{'explo' if is_explo else 'eval'}_race_time_finished_{map_status}_{map_name}"] = (
                end_race_stats["race_time"] / 1000
            )
            if not is_explo:
                accumulated_stats["rolling_mean_ms"][map_name] = (
                    accumulated_stats["rolling_mean_ms"].get(map_name, config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms)
                    * 0.9
                    + end_race_stats["race_time"] * 0.1
                )
        if (
            (not is_explo)
            and end_race_stats["race_finished"]
            and end_race_stats["race_time"] < 1.02 * accumulated_stats["rolling_mean_ms"][map_name]
        ):
            race_stats_to_write[f"eval_race_time_robust_{map_status}_{map_name}"] = end_race_stats["race_time"] / 1000
            if map_name in reference_times:
                for reference_time_name in ["author", "gold"]:
                    if reference_time_name in reference_times[map_name]:
                        reference_time = reference_times[map_name][reference_time_name]
                        race_stats_to_write[f"eval_ratio_{map_status}_{reference_time_name}_{map_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                        )
                        race_stats_to_write[f"eval_agg_ratio_{map_status}_{reference_time_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                        )

        for i in [0]:
            race_stats_to_write[f"q_value_{i}_starting_frame_{map_name}"] = end_race_stats[f"q_value_{i}_starting_frame"]
        if not is_explo:
            for i, split_time in enumerate(
                [
                    (e - s) / 1000
                    for s, e in zip(
                        end_race_stats["cp_time_ms"][:-1],
                        end_race_stats["cp_time_ms"][1:],
                    )
                ]
            ):
                race_stats_to_write[f"split_{map_name}_{i}"] = split_time

        walltime_tb = time.time()
        for tag, value in race_stats_to_write.items():
            tensorboard_writer.add_scalar(
                tag=tag,
                scalar_value=value,
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

        # ===============================================
        #   SAVE STUFF IF THIS WAS A GOOD RACE
        # ===============================================

        if end_race_stats["race_time"] < accumulated_stats["alltime_min_ms"].get(map_name, 99999999999):
            # This is a new alltime_minimum
            accumulated_stats["alltime_min_ms"][map_name] = end_race_stats["race_time"]
            if accumulated_stats["cumul_number_frames_played"] > config_copy.frames_before_save_best_runs:
                name = f"{map_name}_{end_race_stats['race_time'] // 1000}_{end_race_stats['race_time'] % 1000:03d}"
                utilities.save_run(
                    base_dir,
                    save_dir / "best_runs" / name,
                    rollout_results,
                    f"{name}.inputs",
                    inputs_only=False,
                )
                utilities.save_checkpoint(
                    save_dir / "best_runs",
                    online_network,
                    target_network,
                    optimizer1,
                    scaler,
                )

        if end_race_stats["race_time"] < config_copy.threshold_to_save_all_runs_ms:
            race_time_ms = end_race_stats["race_time"]
            name = f"{map_name}_{race_time_ms // 1000}_{race_time_ms % 1000:03d}_{datetime.now().strftime('%m%d_%H%M%S')}_{accumulated_stats['cumul_number_frames_played']}_{'explo' if is_explo else 'eval'}"
            utilities.save_run(
                base_dir,
                save_dir / "good_runs",
                rollout_results,
                f"{name}.inputs",
                inputs_only=True,
            )

        # ===============================================
        #   FILL BUFFER WITH (S, A, R, S') transitions
        # ===============================================
        terminal_non_finite_batch = False
        if fill_buffer:
            (
                buffer,
                buffer_test,
                number_memories_added_train,
                number_memories_added_test,
            ) = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
                buffer,
                buffer_test,
                rollout_results,
                config_copy.n_steps,
                gamma,
                config_copy.discard_non_greedy_actions_in_nsteps,
                engineered_speedslide_reward,
                engineered_neoslide_reward,
                engineered_kamikaze_reward,
                engineered_close_to_vcp_reward,
            )

            accumulated_stats["cumul_number_memories_generated"] += number_memories_added_train + number_memories_added_test
            shared_steps.value = accumulated_stats["cumul_number_frames_played"]
            neural_net_reset_counter += number_memories_added_train
            accumulated_stats["cumul_number_single_memories_should_have_been_used"] += (
                config_copy.number_times_single_memory_is_used_before_discard * number_memories_added_train
            )
            print(f" NMG={accumulated_stats['cumul_number_memories_generated']:<8}")

            # ===============================================
            #   PERIODIC RESET ?
            # ===============================================

            if neural_net_reset_counter >= config_copy.reset_every_n_frames_generated or single_reset_flag != config_copy.single_reset_flag:
                neural_net_reset_counter = 0
                single_reset_flag = config_copy.single_reset_flag
                accumulated_stats["cumul_number_single_memories_should_have_been_used"] += config_copy.additional_transition_after_reset

                _, untrained_iqn_network = make_untrained_iqn_network(config_copy.use_jit, False)
                utilities.soft_copy_param(online_network, untrained_iqn_network, config_copy.overall_reset_mul_factor)

                with torch.no_grad():
                    online_network.A_head[2].weight = utilities.linear_combination(
                        online_network.A_head[2].weight,
                        untrained_iqn_network.A_head[2].weight,
                        config_copy.last_layer_reset_factor,
                    )
                    online_network.A_head[2].bias = utilities.linear_combination(
                        online_network.A_head[2].bias,
                        untrained_iqn_network.A_head[2].bias,
                        config_copy.last_layer_reset_factor,
                    )
                    online_network.V_head[2].weight = utilities.linear_combination(
                        online_network.V_head[2].weight,
                        untrained_iqn_network.V_head[2].weight,
                        config_copy.last_layer_reset_factor,
                    )
                    online_network.V_head[2].bias = utilities.linear_combination(
                        online_network.V_head[2].bias,
                        untrained_iqn_network.V_head[2].bias,
                        config_copy.last_layer_reset_factor,
                    )

            # ===============================================
            #   LEARN ON BATCH
            # ===============================================

            if not online_network.training:
                online_network.train()

            training_batches_this_rollout = 0
            while (
                len(buffer) >= memory_size_start_learn
                and accumulated_stats["cumul_number_single_memories_used"] + offset_cumul_number_single_memories_used
                <= accumulated_stats["cumul_number_single_memories_should_have_been_used"]
                and training_batches_this_rollout < config_copy.max_training_batches_per_rollout
            ):
                if (random.random() < config_copy.buffer_test_ratio and len(buffer_test) > 0) or len(buffer) == 0:
                    test_start_time = time.perf_counter()
                    loss, _ = trainer.train_on_batch(buffer_test, do_learn=False)
                    time_testing_since_last_tensorboard_write += time.perf_counter() - test_start_time
                    loss_test_history.append(loss)
                    print(f"BT   {loss=:<8.2e}")
                else:
                    train_start_time = time.perf_counter()
                    with _guard_non_finite_optimizer_step(optimizer1, scaler) as batch_guard:
                        loss, grad_norm = trainer.train_on_batch(buffer, do_learn=True)
                    non_finite_reason = _non_finite_batch_reason(loss, grad_norm, batch_guard)
                    if non_finite_reason is not None:
                        terminal_reason = f"non_finite_learner_batch: {non_finite_reason}"
                        print(f"Terminal learner guard: {terminal_reason}")
                        if stage_controller is not None:
                            save_stage_report("non_finite_batch", terminal_reason)
                        stop_event.set()
                        terminal_non_finite_batch = True
                        break
                    train_on_batch_duration_history.append(time.perf_counter() - train_start_time)
                    time_training_since_last_tensorboard_write += train_on_batch_duration_history[-1]
                    accumulated_stats["cumul_number_single_memories_used"] += (
                        4 * config_copy.batch_size
                        if (len(buffer) < buffer._storage.max_size and buffer._storage.max_size > 200_000)
                        else config_copy.batch_size
                    )  # do fewer batches while memory is not full
                    loss_history.append(loss)
                    if not math.isinf(grad_norm):
                        grad_norm_history.append(grad_norm)
                        # utilities.log_gradient_norms(online_network, layer_grad_norm_history) #~1ms overhead per batch

                    accumulated_stats["cumul_number_batches_done"] += 1
                    training_batches_this_rollout += 1
                    print(f"B    {loss=:<8.2e} {grad_norm=:<8.2e} {train_on_batch_duration_history[-1]*1000:<8.1f}")

                    utilities.custom_weight_decay(online_network, 1 - weight_decay)
                    if accumulated_stats["cumul_number_batches_done"] % config_copy.send_shared_network_every_n_batches == 0:
                        with shared_network_lock:
                            uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

                    # ===============================================
                    #   UPDATE TARGET NETWORK
                    # ===============================================
                    if (
                        accumulated_stats["cumul_number_single_memories_used"]
                        >= accumulated_stats["cumul_number_single_memories_used_next_target_network_update"]
                    ):
                        accumulated_stats["cumul_number_target_network_updates"] += 1
                        accumulated_stats["cumul_number_single_memories_used_next_target_network_update"] += (
                            config_copy.number_memories_trained_on_between_target_network_updates
                        )
                        # print("UPDATE")
                        utilities.soft_copy_param(target_network, online_network, config_copy.soft_update_tau)
            if terminal_non_finite_batch:
                break
            sys.stdout.flush()

        if stop_event.is_set():
            break

        # ===============================================
        #   WRITE AGGREGATED STATISTICS TO TENSORBOARD EVERY 5 MINUTES
        # ===============================================
        save_frequency_s = 5 * 60
        if time.perf_counter() - time_last_save >= save_frequency_s:
            accumulated_stats["cumul_training_hours"] += (time.perf_counter() - time_last_save) / 3600
            time_since_last_save = time.perf_counter() - time_last_save
            waited_percentage = time_waited_for_workers_since_last_tensorboard_write / time_since_last_save
            trained_percentage = time_training_since_last_tensorboard_write / time_since_last_save
            tested_percentage = time_testing_since_last_tensorboard_write / time_since_last_save
            time_waited_for_workers_since_last_tensorboard_write = 0
            time_training_since_last_tensorboard_write = 0
            time_testing_since_last_tensorboard_write = 0
            transitions_learned_per_second = (
                accumulated_stats["cumul_number_single_memories_used"] - transitions_learned_last_save
            ) / time_since_last_save
            time_last_save = time.perf_counter()
            transitions_learned_last_save = accumulated_stats["cumul_number_single_memories_used"]

            # ===============================================
            #   COLLECT VARIOUS STATISTICS
            # ===============================================
            step_stats = {
                "gamma": gamma,
                "n_steps": config_copy.n_steps,
                "epsilon": utilities.from_exponential_schedule(config_copy.epsilon_schedule, shared_steps.value),
                "epsilon_boltzmann": utilities.from_exponential_schedule(config_copy.epsilon_boltzmann_schedule, shared_steps.value),
                "tau_epsilon_boltzmann": config_copy.tau_epsilon_boltzmann,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "discard_non_greedy_actions_in_nsteps": config_copy.discard_non_greedy_actions_in_nsteps,
                "memory_size": len(buffer),
                "number_times_single_memory_is_used_before_discard": config_copy.number_times_single_memory_is_used_before_discard,
                "learner_percentage_waiting_for_workers": waited_percentage,
                "learner_percentage_training": trained_percentage,
                "learner_percentage_testing": tested_percentage,
                "transitions_learned_per_second": transitions_learned_per_second,
            }
            if len(loss_history) > 0 and len(loss_test_history) > 0:
                step_stats.update(
                    {
                        "loss": np.mean(loss_history),
                        "loss_test": np.mean(loss_test_history),
                        "train_on_batch_duration": np.median(train_on_batch_duration_history),
                        "grad_norm_history_q1": np.quantile(grad_norm_history, 0.25),
                        "grad_norm_history_median": np.quantile(grad_norm_history, 0.5),
                        "grad_norm_history_q3": np.quantile(grad_norm_history, 0.75),
                        "grad_norm_history_d9": np.quantile(grad_norm_history, 0.9),
                        "grad_norm_history_d98": np.quantile(grad_norm_history, 0.98),
                        "grad_norm_history_max": np.max(grad_norm_history),
                    }
                )
                for key, val in layer_grad_norm_history.items():
                    step_stats.update(
                        {
                            f"{key}_median": np.quantile(val, 0.5),
                            f"{key}_q3": np.quantile(val, 0.75),
                            f"{key}_d9": np.quantile(val, 0.9),
                            f"{key}_c98": np.quantile(val, 0.98),
                            f"{key}_max": np.max(val),
                        }
                    )
            if isinstance(buffer._sampler, PrioritizedSampler):
                all_priorities = np.array([buffer._sampler._sum_tree.at(i) for i in range(len(buffer))])
                step_stats.update(
                    {
                        "priorities_min": np.min(all_priorities),
                        "priorities_q1": np.quantile(all_priorities, 0.1),
                        "priorities_mean": np.mean(all_priorities),
                        "priorities_median": np.quantile(all_priorities, 0.5),
                        "priorities_q3": np.quantile(all_priorities, 0.75),
                        "priorities_d9": np.quantile(all_priorities, 0.9),
                        "priorities_c98": np.quantile(all_priorities, 0.98),
                        "priorities_max": np.max(all_priorities),
                    }
                )
            for key, value in accumulated_stats.items():
                if key not in ["alltime_min_ms", "rolling_mean_ms"]:
                    step_stats[key] = value
            for key, value in accumulated_stats["alltime_min_ms"].items():
                step_stats[f"alltime_min_ms_{map_name}"] = value

            loss_history = []
            loss_test_history = []
            train_on_batch_duration_history = []
            grad_norm_history = []
            layer_grad_norm_history = defaultdict(list)

            # ===============================================
            #   COLLECT IQN SPREAD
            # ===============================================

            if online_network.training:
                online_network.eval()
            tau = torch.linspace(0.05, 0.95, config_copy.iqn_k)[:, None].to(device)
            per_quantile_output = inferer.infer_network(rollout_results["frames"][0], rollout_results["state_float"][0], tau)
            for i, std in enumerate(list(per_quantile_output.std(axis=0))):
                step_stats[f"std_within_iqn_quantiles_for_action{i}"] = std

            # ===============================================
            #   WRITE TO TENSORBOARD
            # ===============================================

            walltime_tb = time.time()
            for name, param in online_network.named_parameters():
                tensorboard_writer.add_scalar(
                    tag=f"layer_{name}_L2",
                    scalar_value=np.sqrt((param**2).mean().detach().cpu().item()),
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )
            assert len(optimizer1.param_groups) == 1
            try:
                for p, (name, _) in zip(
                    optimizer1.param_groups[0]["params"],
                    online_network.named_parameters(),
                ):
                    state = optimizer1.state[p]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    mod_lr = 1 / (exp_avg_sq.sqrt() + 1e-4)
                    tensorboard_writer.add_scalar(
                        tag=f"lr_ratio_{name}_L2",
                        scalar_value=np.sqrt((mod_lr**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
                    tensorboard_writer.add_scalar(
                        tag=f"exp_avg_{name}_L2",
                        scalar_value=np.sqrt((exp_avg**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
                    tensorboard_writer.add_scalar(
                        tag=f"exp_avg_sq_{name}_L2",
                        scalar_value=np.sqrt((exp_avg_sq**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
            except:
                pass

            for k, v in step_stats.items():
                tensorboard_writer.add_scalar(
                    tag=k,
                    scalar_value=v,
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )

            previous_alltime_min = previous_alltime_min or copy.deepcopy(accumulated_stats["alltime_min_ms"])

            tensorboard_writer.add_text(
                "times_summary",
                f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')} "
                + " ".join(
                    [
                        f"{'**' if v < previous_alltime_min.get(k, 99999999) else ''}{k}: {v / 1000:.2f}{'**' if v < previous_alltime_min.get(k, 99999999) else ''}"
                        for k, v in accumulated_stats["alltime_min_ms"].items()
                    ]
                ),
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

            previous_alltime_min = copy.deepcopy(accumulated_stats["alltime_min_ms"])

            # ===============================================
            #   BUFFER STATS
            # ===============================================

            state_floats = np.array([experience.state_float for experience in buffer._storage])
            mean_in_buffer = state_floats.mean(axis=0)
            std_in_buffer = state_floats.std(axis=0)

            print("Raw mean in buffer  :", mean_in_buffer.round(1))
            print("Raw std in buffer   :", std_in_buffer.round(1))
            print("")
            print(
                "Corr mean in buffer :",
                ((mean_in_buffer - config_copy.float_inputs_mean) / config_copy.float_inputs_std).round(1),
            )
            print(
                "Corr std in buffer  :",
                (std_in_buffer / config_copy.float_inputs_std).round(1),
            )
            print("")

            # ===============================================
            #   HIGH PRIORITY TRANSITIONS
            # ===============================================
            if config_copy.make_highest_prio_figures and isinstance(buffer._sampler, PrioritizedSampler):
                highest_prio_transitions(buffer, save_dir)

            # ===============================================
            #   SAVE
            # ===============================================
            utilities.save_checkpoint(save_dir, online_network, target_network, optimizer1, scaler)
            joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")
