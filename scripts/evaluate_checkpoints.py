"""
Defensible, Controlled Zero-Shot Benchmark Script for Project Linesight
Compares Foundational (3.18M) vs Late Saturated (10.64M) checkpoints across unseen tracks.
Computes unified distance completion %, termination classification, action distribution,
detailed weight hashing, greedy exploration parameters, and first-frame diagnostics.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from multiprocessing import Lock
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure repo root is on sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from config_files import config_copy
from config_files.inputs_list import inputs
from trackmania_rl.agents import iqn
from trackmania_rl.map_loader import load_next_map_zone_centers
from trackmania_rl import utilities
from trackmania_rl.tmi_interaction import game_instance_manager

TRACKS = {
    "A01": {
        "name": "A01-Race",
        "map_path": r'"Official Maps\A01-Race.Challenge.Gbx"',
        "cl_file": "A01-Race_0.5m_cl2.npy",
        "difficulty": "White / Easy",
    },
    "A03": {
        "name": "A03-Race",
        "map_path": r'"Official Maps\A03-Race.Challenge.Gbx"',
        "cl_file": "A03-Race_0.5m_cl2.npy",
        "difficulty": "White / Easy",
    },
    "A05": {
        "name": "A05-Race",
        "map_path": r'"Official Maps\A05-Race.Challenge.Gbx"',
        "cl_file": "A05-Race_0.5m_cl2.npy",
        "difficulty": "White / Easy",
    },
    "A07": {
        "name": "A07-Race",
        "map_path": r'"Official Maps\A07-Race.Challenge.Gbx"',
        "cl_file": "A07-Race_0.5m_cl2.npy",
        "difficulty": "White / Technical",
    },
    "A11": {
        "name": "A11-Race",
        "map_path": r'"Official Maps\A11-Race.Challenge.Gbx"',
        "cl_file": "A11-Race_0.5m_cl2.npy",
        "difficulty": "White / Medium",
    },
    "hock": {
        "name": "ESL-Hockolicious",
        "map_path": "ESL-Hockolicious.Challenge.Gbx",
        "cl_file": "ESL-Hockolicious_0.5m_cl2.npy",
        "difficulty": "Control / Trained",
    },
}

CHECKPOINTS = {
    "foundational_3.18M": BASE_DIR / "save" / "hocko_v1_foundational_checkpoint" / "weights1.torch",
    "saturated_10.64M": BASE_DIR / "save" / "hocko_v1" / "weights1.torch",
}


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_state_dict_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        h.update(k.encode("utf-8"))
        h.update(state_dict[k].cpu().numpy().tobytes())
    return h.hexdigest()


def evaluate_run(
    tmi: game_instance_manager.GameInstanceManager,
    inferer: iqn.Inferer,
    map_path: str,
    zone_centers: np.ndarray,
    ablate_vision: bool = False,
    trace_first_frame: bool = False,
) -> Dict:
    total_zones = len(zone_centers)
    finish_zone_idx = total_zones - config_copy.n_zone_centers_extrapolate_after_end_of_map
    finish_dist_m = finish_zone_idx * 0.5
    total_extrap_dist_m = total_zones * 0.5

    first_frame_info = {}
    step_counter = [0]

    def policy_wrapper(img, float_inputs):
        if ablate_vision:
            img = np.zeros_like(img)

        # On the very first frame of the rollout, inspect float inputs and raw Q-values
        if step_counter[0] == 0:
            with torch.no_grad():
                q_vals = inferer.infer_network(img, float_inputs).mean(axis=0)
                first_frame_info["frame0_q_values"] = [round(float(q), 4) for q in q_vals]
                first_frame_info["frame0_action"] = int(np.argmax(q_vals))
                first_frame_info["frame0_max_q"] = round(float(np.max(q_vals)), 4)
                first_frame_info["frame0_q_spread"] = round(float(np.max(q_vals) - np.min(q_vals)), 4)
                first_frame_info["frame0_floats_min"] = round(float(np.min(float_inputs)), 4)
                first_frame_info["frame0_floats_max"] = round(float(np.max(float_inputs)), 4)
                first_frame_info["frame0_floats_mean"] = round(float(np.mean(float_inputs)), 4)
                first_frame_info["frame0_floats_sample"] = [round(float(v), 3) for v in float_inputs[:8]]

        step_counter[0] += 1
        return inferer.get_exploration_action(img, float_inputs)

    rollout_results, end_race_stats = tmi.rollout(
        exploration_policy=policy_wrapper,
        map_path=map_path,
        zone_centers=zone_centers,
        update_network=lambda: None,
    )

    furthest_vcp = int(rollout_results.get("furthest_zone_idx", 0))
    dist_m = furthest_vcp * 0.5
    reached_finish_zone = bool(furthest_vcp >= finish_zone_idx)
    finished = bool(end_race_stats.get("race_finished", False))

    race_time_ms = float(end_race_stats.get("race_time", 0.0))
    race_time_for_ratio_ms = float(end_race_stats.get("race_time_for_ratio", 0.0))

    if finished:
        term_cause = "Finished"
        actual_time_s = race_time_ms / 1000.0
        completion_pct = 100.0
    elif tmi.last_rollout_crashed:
        term_cause = "Crash_Disrupt"
        actual_time_s = race_time_for_ratio_ms / 1000.0
        completion_pct = min(100.0, (furthest_vcp / finish_zone_idx) * 100.0)
    elif race_time_for_ratio_ms >= 290_000:
        term_cause = "Timeout_300s"
        actual_time_s = race_time_for_ratio_ms / 1000.0
        completion_pct = min(100.0, (furthest_vcp / finish_zone_idx) * 100.0)
    else:
        term_cause = "Stuck_Cutoff_2s"
        actual_time_s = race_time_for_ratio_ms / 1000.0
        completion_pct = min(100.0, (furthest_vcp / finish_zone_idx) * 100.0)

    # Action distribution
    raw_actions = rollout_results.get("actions", [])
    action_counts = Counter(raw_actions)
    total_actions = max(1, len(raw_actions))

    steer_left_pct = sum(action_counts[i] for i, a in enumerate(inputs) if a["left"]) / total_actions * 100.0
    steer_right_pct = sum(action_counts[i] for i, a in enumerate(inputs) if a["right"]) / total_actions * 100.0
    throttle_pct = sum(action_counts[i] for i, a in enumerate(inputs) if a["accelerate"]) / total_actions * 100.0
    brake_pct = sum(action_counts[i] for i, a in enumerate(inputs) if a["brake"]) / total_actions * 100.0

    res = {
        "furthest_vcp": furthest_vcp,
        "finish_vcp": finish_zone_idx,
        "total_vcps": total_zones,
        "distance_m": dist_m,
        "finish_dist_m": finish_dist_m,
        "total_extrap_dist_m": total_extrap_dist_m,
        "completion_pct": completion_pct,
        "reached_finish_zone": reached_finish_zone,
        "finished": finished,
        "actual_elapsed_s": actual_time_s,
        "term_cause": term_cause,
        "total_actions": total_actions,
        "steer_left_pct": steer_left_pct,
        "steer_right_pct": steer_right_pct,
        "throttle_pct": throttle_pct,
        "brake_pct": brake_pct,
        "is_explo": inferer.is_explo,
        "epsilon": inferer.epsilon,
        "epsilon_boltzmann": inferer.epsilon_boltzmann,
    }
    res.update(first_frame_info)
    return res


def run_benchmark_for_checkpoint(
    ckpt_key: str,
    tracks: List[str],
    runs_per_track: int,
    ablate_vision: bool,
    port: int,
    csv_file: Path,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = CHECKPOINTS[ckpt_key]
    if not ckpt_path.exists():
        print(f"[!] Checkpoint not found: {ckpt_path}")
        return

    file_hash = compute_file_sha256(ckpt_path)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict_hash = compute_state_dict_sha256(state_dict)

    print(f"\n=======================================================")
    print(f" CHECKPOINT PROCESS: [{ckpt_key}]")
    print(f" Path: {ckpt_path}")
    print(f" File SHA256:       {file_hash}")
    print(f" StateDict SHA256:  {state_dict_hash}")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"=======================================================")

    # Initialize network
    inference_network, _ = iqn.make_untrained_iqn_network(config_copy.use_jit, is_inference=True)
    inference_network.eval()
    utilities.load_model_checkpoint(
        inference_network,
        ckpt_path,
        map_location=device,
        allowed_missing_keys=utilities.NON_PERSISTENT_NORMALIZATION_KEYS,
        allowed_unexpected_keys=utilities.NON_PERSISTENT_NORMALIZATION_KEYS,
    )

    inferer = iqn.Inferer(inference_network, config_copy.iqn_k, config_copy.tau_epsilon_boltzmann)
    # Strictly zero out exploration to ensure pure greedy deterministic evaluation
    inferer.is_explo = False
    inferer.epsilon = 0.0
    inferer.epsilon_boltzmann = 0.0

    # Initialize TMI Game Instance
    spawning_lock = Lock()
    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=spawning_lock,
        running_speed=config_copy.running_speed,
        run_steps_per_action=config_copy.tm_engine_step_per_action,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        tmi_port=port,
    )

    ckpt_results = []
    try:
        for track_key in tracks:
            meta = TRACKS[track_key]
            map_path = meta["map_path"]
            cl_file = meta["cl_file"]

            zone_centers = load_next_map_zone_centers(cl_file, BASE_DIR)
            finish_vcp = len(zone_centers) - config_copy.n_zone_centers_extrapolate_after_end_of_map
            print(f"\n>>> Evaluating Track: [{track_key}] ({meta['name']}) - {meta['difficulty']}")
            print(f"    Finish Distance: {finish_vcp * 0.5:.1f}m ({finish_vcp} VCPs) | Total Array: {len(zone_centers) * 0.5:.1f}m ({len(zone_centers)} VCPs)")

            # Warm-up / sync rollout to cleanly load track and capture start_state before measuring
            print(f"    [Sync] Checking track state and pre-synchronizing start state...")
            warm_policy = lambda img, f: inferer.get_exploration_action(img, f)
            tmi.rollout(exploration_policy=warm_policy, map_path=map_path, zone_centers=zone_centers, update_network=lambda: None)
            print(f"    [Sync] Track synchronized. Beginning {runs_per_track} measured runs.")

            track_runs = []
            for run_idx in range(1, runs_per_track + 1):
                res = evaluate_run(
                    tmi=tmi,
                    inferer=inferer,
                    map_path=map_path,
                    zone_centers=zone_centers,
                    ablate_vision=ablate_vision,
                    trace_first_frame=(run_idx == 1),
                )
                res["checkpoint"] = ckpt_key
                res["file_sha256"] = file_hash
                res["state_dict_sha256"] = state_dict_hash
                res["track"] = track_key
                res["track_name"] = meta["name"]
                res["run_idx"] = run_idx
                res["vision_ablated"] = ablate_vision

                track_runs.append(res)
                ckpt_results.append(res)

                # Format log
                status_str = f"Time: {res['actual_elapsed_s']:.2f}s" if res["finished"] else f"Term: {res['term_cause']} ({res['actual_elapsed_s']:.2f}s)"
                print(
                    f"    Run {run_idx:>2}/{runs_per_track}: "
                    f"Dist: {res['distance_m']:>6.1f}m / {res['finish_dist_m']:.1f}m ({res['completion_pct']:>5.1f}%) | "
                    f"FinZone: {str(res['reached_finish_zone']):<5} | FinFlag: {str(res['finished']):<5} | "
                    f"{status_str:<26} | "
                    f"Thr: {res['throttle_pct']:>4.1f}% | Brk: {res['brake_pct']:>4.1f}% | "
                    f"L: {res['steer_left_pct']:>4.1f}% | R: {res['steer_right_pct']:>4.1f}% | "
                    f"F0 Act: {res.get('frame0_action', -1)} (Q-spread: {res.get('frame0_q_spread', 0.0):.2f})"
                )

                if run_idx == 1:
                    print(f"       [F0 Diagnostics] Floats Sample: {res.get('frame0_floats_sample')} | Q-Vals: {res.get('frame0_q_values')}")

            # Summary per track
            pcts = [r["completion_pct"] for r in track_runs]
            fin_count = sum(1 for r in track_runs if r["finished"])
            fin_zone_count = sum(1 for r in track_runs if r["reached_finish_zone"])
            mean_pct = np.mean(pcts)
            std_pct = np.std(pcts)
            causes = Counter([r["term_cause"] for r in track_runs])
            times = [r["actual_elapsed_s"] for r in track_runs if r["finished"]]
            time_str = f"Mean Time: {np.mean(times):.2f}s, Median: {np.median(times):.2f}s" if times else "No finishes"

            print(f"  --> SUMMARY [{track_key}]: Mean: {mean_pct:.1f}% ± {std_pct:.1f}% | Finishes: {fin_count}/{runs_per_track} | Reached FinZone: {fin_zone_count}/{runs_per_track} | {time_str} | Causes: {dict(causes)}")

    finally:
        # Append or write to CSV
        if ckpt_results:
            file_exists = csv_file.exists()
            fieldnames = [k for k in ckpt_results[0].keys() if k not in ["frame0_floats_sample", "frame0_q_values"]]
            with open(csv_file, "a" if file_exists else "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerows(ckpt_results)
            print(f"\n[OK] Results appended to: {csv_file}")

        os.system("taskkill /F /IM TmForever.exe 2>nul")


def main():
    parser = argparse.ArgumentParser(description="Controlled Zero-Shot Benchmark")
    parser.add_argument("--runs_per_track", type=int, default=10, help="Number of greedy runs per track (N>=10)")
    parser.add_argument("--checkpoints", nargs="+", default=["foundational_3.18M", "saturated_10.64M"], choices=list(CHECKPOINTS.keys()))
    parser.add_argument("--tracks", nargs="+", default=list(TRACKS.keys()), choices=list(TRACKS.keys()))
    parser.add_argument("--ablate_vision", action="store_true", help="Zero out camera image to test centerline reliance")
    parser.add_argument("--port", type=int, default=8478, help="TMInterface port")
    parser.add_argument("--output_csv", type=str, default="", help="Explicit CSV path")
    parser.add_argument("--single_ckpt", type=str, default="", help="Internal flag to run single checkpoint in current process")
    args = parser.parse_args()

    out_dir = BASE_DIR / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output_csv:
        csv_file = Path(args.output_csv)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = out_dir / f"benchmark_fixed_{timestamp}.csv"

    # If single_ckpt is specified, run directly in this process
    if args.single_ckpt:
        run_benchmark_for_checkpoint(
            ckpt_key=args.single_ckpt,
            tracks=args.tracks,
            runs_per_track=args.runs_per_track,
            ablate_vision=args.ablate_vision,
            port=args.port,
            csv_file=csv_file,
        )
        return

    # Otherwise, launch each checkpoint in a completely separate Python process
    print(f"================================================================")
    print(f" PROJECT LINESIGHT — MULTI-PROCESS EVALUATION BENCHMARK")
    print(f" Output CSV: {csv_file}")
    print(f" Checkpoints to evaluate in separate processes: {args.checkpoints}")
    print(f" Tracks: {args.tracks}")
    print(f" Runs per track: {args.runs_per_track}")
    print(f"================================================================\n")

    for ckpt in args.checkpoints:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single_ckpt", ckpt,
            "--runs_per_track", str(args.runs_per_track),
            "--tracks", *args.tracks,
            "--port", str(args.port),
            "--output_csv", str(csv_file),
        ]
        if args.ablate_vision:
            cmd.append("--ablate_vision")

        print(f"\n>>> Spawning clean Python process for [{ckpt}]...")
        subprocess.run(cmd, check=True)
        time.sleep(2)  # Allow port and game process cleanup

    print(f"\n[OK] All checkpoint processes completed. Full dataset saved at: {csv_file}")


if __name__ == "__main__":
    main()
