"""
Various neural network & scheduling utilities.
"""

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Collection, List, Tuple

import random
import joblib
import numpy as np
import torch
from prettytable import PrettyTable

from trackmania_rl import run_to_video


# These buffers are intentionally registered with persistent=False in
# IQN_Network, so older checkpoints correctly omit them.  They are rebuilt
# from the active configuration when the model is constructed.  No other
# checkpoint mismatch is implicitly compatible.
NON_PERSISTENT_NORMALIZATION_KEYS = frozenset({"float_inputs_mean", "float_inputs_std"})
STAGE_BASELINE_FILENAMES = (
    "weights1.torch",
    "weights2.torch",
    "optimizer1.torch",
    "scaler.torch",
    "accumulated_stats.joblib",
)


def init_kaiming(layer, neg_slope=0, nonlinearity="leaky_relu"):
    torch.nn.init.kaiming_normal_(layer.weight, a=neg_slope, mode="fan_out", nonlinearity=nonlinearity)
    torch.nn.init.zeros_(layer.bias)


def init_xavier(layer, gain=1.0):
    torch.nn.init.xavier_normal_(layer.weight, gain=gain)
    torch.nn.init.zeros_(layer.bias)


def init_orthogonal(layer, gain=1.0):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.zeros_(layer.bias)


def init_uniform(layer, a, b):
    torch.nn.init.uniform_(layer.weight, a=a, b=b)
    torch.nn.init.zeros_(layer.bias)


def init_normal(layer, mean, std):
    torch.nn.init.normal_(layer.weight, mean=mean, std=std)
    torch.nn.init.zeros_(layer.bias)


def log_gradient_norms(model, layer_grad_norm_history):
    l2_norms = []
    linf_norms = []
    param_names = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            l2_norms.append(torch.norm(grad))
            linf_norms.append(torch.max(grad))
            param_names.append(name)

    l2_norms_cpu = torch.stack(l2_norms).cpu().numpy()
    linf_norms_cpu = torch.stack(linf_norms).cpu().numpy()

    for name, l2_norm, linf_norm in zip(param_names, l2_norms_cpu, linf_norms_cpu):
        layer_grad_norm_history[f"L2_grad_norm_{name}"].append(l2_norm)
        layer_grad_norm_history[f"Linf_grad_norm_{name}"].append(linf_norm)


def linear_combination(a, b, alpha):
    assert a.shape == b.shape
    a.mul_(1 - alpha)
    a.add_(alpha * b)
    return a


# From https://github.com/pfnet/pfrl/blob/2ad3d51a7a971f3fe7f2711f024be11642990d61/pfrl/utils/copy_param.py#L37
def soft_copy_param(target_link, source_link, tau):
    """Soft-copy parameters of a link to another link."""
    target_dict = target_link.state_dict()
    source_dict = source_link.state_dict()
    for k, target_value in target_dict.items():
        source_value = source_dict[k]
        if source_value.dtype in [torch.float32, torch.float64, torch.float16]:
            linear_combination(target_value, source_value, tau)
        else:
            # Scalar type
            # Some modules such as BN has scalar value `num_batches_tracked`
            target_dict[k] = source_value
            assert False, "Soft scalar update should not happen"


def custom_weight_decay(target_link, decay_factor):
    target_dict = target_link.state_dict()
    for k, target_value in target_dict.items():
        target_value.mul_(decay_factor)


def from_exponential_schedule(schedule: List[Tuple[int, float]], current_step: int):
    """
    Calculate the current scheduled value, with exponential interpolation between fixed setpoints at given steps.
    If current step is larger than the largest scheduled step, return the value prescribed by the largest scheduled step.

    Args:
        - schedule:         a list of (step, value) tuples. Must contain a value for step 0.
        - current_step:     an int representing... the current step

    Returns:
        value: the value defined by the schedule and current_step
    """
    schedule = sorted(schedule, key=lambda p: p[0])  # Sort by step in case it was not defined in sorted order
    assert schedule[0][0] == 0
    schedule_end_index = next((idx for idx, p in enumerate(schedule) if p[0] > current_step), -1)  # Returns -1 if none is found
    if schedule_end_index == -1:
        return schedule[-1][1]
    else:
        assert schedule_end_index >= 1
        schedule_end_step = schedule[schedule_end_index][0]
        schedule_begin_step = schedule[schedule_end_index - 1][0]
        annealing_period = schedule_end_step - schedule_begin_step
        end_value = schedule[schedule_end_index][1]
        begin_value = schedule[schedule_end_index - 1][1]
        ratio = begin_value / end_value
        assert annealing_period > 0
        return begin_value * math.exp(-math.log(ratio) * (current_step - schedule_begin_step) / annealing_period)


def from_linear_schedule(schedule, current_step):
    """
    Calculate the current scheduled value, with linear interpolation between fixed setpoints at given steps.
    If current step is larger than the largest scheduled step, return the value prescribed by the largest scheduled step.

    Args:
        - schedule:         a list of (step, value) tuples. Must contain a value for step 0.
        - current_step:     an int representing... the current step

    Returns:
        value: the value defined by the schedule and current_step
    """
    schedule = sorted(schedule, key=lambda p: p[0])  # Sort by step in case it was not defined in sorted order
    assert schedule[0][0] == 0
    return np.interp([current_step], [p[0] for p in schedule], [p[1] for p in schedule])[0]


def from_staircase_schedule(schedule, current_step):
    """
    Calculate the current scheduled value, with no interpolation between steps.

    Args:
        - schedule:         a list of (step, value) tuples. Must contain a value for step 0.
        - current_step:     an int representing... the current step

    Returns:
        value: the value defined by the schedule and current_step
    """
    schedule = sorted(schedule, key=lambda p: p[0])  # Sort by step in case it was not defined in sorted order
    assert schedule[0][0] == 0
    return next((p for p in reversed(schedule) if p[0] <= current_step))[1]


def count_parameters(model):
    # from https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def save_run(
    base_dir: Path,
    run_dir: Path,
    rollout_results: dict,
    inputs_filename: str,
    inputs_only: bool,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    run_to_video.write_actions_in_tmi_format(rollout_results["actions"], run_dir / inputs_filename)
    if not inputs_only:
        shutil.copy(base_dir / "config_files" / "config_copy.py", run_dir / "config.bak.py")
        joblib.dump(rollout_results["q_values"], run_dir / "q_values.joblib")


def save_checkpoint(
    checkpoint_dir: Path,
    online_network: torch.nn.Module,
    target_network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(online_network.state_dict(), checkpoint_dir / "weights1.torch")
    torch.save(target_network.state_dict(), checkpoint_dir / "weights2.torch")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer1.torch")
    torch.save(scaler.state_dict(), checkpoint_dir / "scaler.torch")


def create_stage_baseline(
    baseline_dir: Path,
    online_network: torch.nn.Module,
    target_network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    accumulated_stats: Mapping,
) -> Path:
    """Create an immutable-on-restart stage baseline if one does not exist."""

    baseline_dir = Path(baseline_dir)
    required = [baseline_dir / name for name in STAGE_BASELINE_FILENAMES]
    if baseline_dir.exists():
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Incomplete stage baseline at {baseline_dir}: missing {missing}")
        return baseline_dir

    baseline_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{baseline_dir.name}.", dir=str(baseline_dir.parent)))
    committed = False
    try:
        save_checkpoint(staging_dir, online_network, target_network, optimizer, scaler)
        joblib.dump(accumulated_stats, staging_dir / "accumulated_stats.joblib")
        (staging_dir / "stage_baseline_manifest.json").write_text(
            json.dumps({"schema_version": 1, "files": list(STAGE_BASELINE_FILENAMES)}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_dir, baseline_dir)
        committed = True
    finally:
        if not committed and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
    return baseline_dir


def restore_stage_baseline(baseline_dir: Path, destination_dir: Path) -> None:
    """Restore the model/learner files from an existing stage baseline."""

    baseline_dir = Path(baseline_dir)
    destination_dir = Path(destination_dir)
    missing = [name for name in STAGE_BASELINE_FILENAMES if not (baseline_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Stage baseline is incomplete: missing {missing}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in STAGE_BASELINE_FILENAMES:
        shutil.copy2(baseline_dir / name, destination_dir / name)


def load_stage_baseline_start_frames(baseline_dir: Path) -> int:
    """Read and validate the immutable frame counter at stage creation."""

    stats_path = Path(baseline_dir) / "accumulated_stats.joblib"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Stage baseline stats not found: {stats_path}")
    stats = joblib.load(stats_path)
    try:
        stage_start_frames = int(stats["cumul_number_frames_played"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Stage baseline has no valid cumul_number_frames_played: {stats_path}") from error
    if stage_start_frames < 0:
        raise RuntimeError(f"Stage baseline frame counter cannot be negative: {stage_start_frames}")
    return stage_start_frames


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    map_location,
    *,
    allowed_missing_keys: Collection[str] = (),
    allowed_unexpected_keys: Collection[str] = (),
) -> None:
    """Load and validate a model state dict.

    Model checkpoints must match the current model unless a caller explicitly
    allows an exact missing or unexpected key for a known compatibility case.
    ``strict=False`` is used only to collect both categories of incompatibility;
    every disallowed key is then reported in the raised error.

    ``weights_only=False`` preserves loading of checkpoints written by older
    versions of this project.  The fallback is limited to PyTorch versions
    whose ``torch.load`` does not yet accept that keyword.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    try:
        state_dict = torch.load(f=checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        state_dict = torch.load(f=checkpoint_path, map_location=map_location)

    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Model checkpoint {checkpoint_path} must contain a state dict, "
            f"but loaded {type(state_dict).__name__}."
        )

    allowed_missing_keys = set(allowed_missing_keys)
    allowed_unexpected_keys = set(allowed_unexpected_keys)
    try:
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as error:
        raise RuntimeError(f"Model checkpoint mismatch for {checkpoint_path}: {error}") from error
    missing_keys = sorted(set(incompatible_keys.missing_keys) - allowed_missing_keys)
    unexpected_keys = sorted(set(incompatible_keys.unexpected_keys) - allowed_unexpected_keys)

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Model checkpoint mismatch for {checkpoint_path}: "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}."
        )


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
