"""Describe what a residual TD3 actor does, on a fixed set of recorded states.

The deterministic actor is evaluated on the observations of recorded runs (``runXX_rl.npz``)
at the steps where the residual was active. Reported per run folder and pooled:
- mean |a| per channel and the share of saturated outputs (|a| > 0.9);
- correlation of the steering residual with sin(heading error) and with the lateral error
  (a negative heading correlation = the policy steers against the heading error: extra damping);
- mean target-speed residual (m/s) within 50 m of a jump vs elsewhere.

Usage:
    python scripts/analyze_residual_policy.py --checkpoint eval_results/residual_rl/td3_v2/final.pt \
        --runs "*_dev_td3v2" "*_rl_train_td3v2" --json eval_results/residual_rl/td3_v2_behaviour.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trackmania_rl.residual_action import ResidualConfig
from trackmania_rl.residual_observation import JUMP_DISTANCE_SCALE_M, OBSERVATION_NAMES
from trackmania_rl.residual_td3 import TD3Agent

LIVE_DIR = ROOT / "eval_results" / "hybrid_live"
NEAR_JUMP_M = 50.0
SATURATED = 0.9
COLUMN = {name: i for i, name in enumerate(OBSERVATION_NAMES)}


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 3)


def describe(observations: np.ndarray, actions: np.ndarray, target_scale_mps: float) -> dict:
    """Behaviour statistics of actions taken on the given (active-step) observations."""

    near_jump = (observations[:, COLUMN["in_jump"]] > 0.5) | (
        observations[:, COLUMN["distance_to_next_jump"]] * JUMP_DISTANCE_SCALE_M < NEAR_JUMP_M
    )
    target = actions[:, 1] * target_scale_mps
    return {
        "steps": int(len(actions)),
        "mean_abs_action": [round(float(v), 3) for v in np.abs(actions).mean(axis=0)] if len(actions) else None,
        "saturated_share": round(float((np.abs(actions) > SATURATED).any(axis=1).mean()), 3) if len(actions) else None,
        "corr_steer_vs_sin_heading_error": _corr(actions[:, 0], observations[:, COLUMN["sin_heading_error"]]),
        "corr_steer_vs_lateral_error": _corr(actions[:, 0], observations[:, COLUMN["lateral_error"]]),
        "mean_target_delta_near_jump_mps": round(float(target[near_jump].mean()), 2) if near_jump.any() else None,
        "mean_target_delta_elsewhere_mps": round(float(target[~near_jump].mean()), 2) if (~near_jump).any() else None,
    }


def load_states(folder: Path) -> np.ndarray:
    chunks = []
    for path in sorted(folder.glob("run*_rl.npz")):
        data = np.load(path)
        chunks.append(data["observations"][data["residual_active"].astype(bool)])
    return np.concatenate(chunks) if chunks else np.zeros((0, len(OBSERVATION_NAMES)), np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="run-folder globs under eval_results/hybrid_live")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    agent = TD3Agent.load(args.checkpoint)
    scale = ResidualConfig().max_target_speed_delta_mps
    folders = sorted({f for pattern in args.runs for f in LIVE_DIR.glob(pattern) if f.is_dir()})
    if not folders:
        parser.error(f"no run folders match {args.runs}")
    per_run, all_obs = {}, []
    for folder in folders:
        obs = load_states(folder)
        if len(obs):
            per_run[folder.name] = describe(obs, np.stack([agent.act(o) for o in obs]), scale)
            all_obs.append(obs)
    pooled_obs = np.concatenate(all_obs)
    pooled = describe(pooled_obs, np.stack([agent.act(o) for o in pooled_obs]), scale)
    print(json.dumps({"pooled": pooled}, indent=1))
    for name, stats in per_run.items():
        print(f"{name:<36} |a| {stats['mean_abs_action']} heading r {stats['corr_steer_vs_sin_heading_error']} "
              f"near-jump {stats['mean_target_delta_near_jump_mps']} m/s, elsewhere {stats['mean_target_delta_elsewhere_mps']} m/s")
    if args.json:
        args.json.write_text(json.dumps({"checkpoint": str(args.checkpoint), "runs": args.runs, "pooled": pooled,
                                         "per_run": per_run}, indent=1) + "\n")


if __name__ == "__main__":
    main()
