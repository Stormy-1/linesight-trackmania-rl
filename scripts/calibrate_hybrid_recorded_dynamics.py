"""Calibrate one-step kinematic residuals on disjoint saved telemetry traces."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_dynamics import (
    summarize_transition_observations,
    transition_observations,
)
from trackmania_rl.hybrid_rollout import load_rollout_csv

DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/recorded_transition_dynamics.json")
DEFAULT_SOURCE_SPLITS: dict[str, tuple[Path, ...]] = {
    "fit": (
        Path("eval_results/hybrid_rollouts/ESL-Hockolicious.Challenge_1792369534482_59880.csv"),
        Path("eval_results/hybrid_rollouts/ESL-Hockolicious.Challenge_1792369556798_59880.csv"),
        Path("eval_results/hybrid_rollouts/A01-Race.Challenge_1793657851614_45728.csv"),
    ),
    "heldout": (
        Path("eval_results/hybrid_rollouts/A01-Race.Challenge_1793658034903_19052.csv"),
        Path("eval_results/hybrid_rollouts/A01-Race.Challenge_1793658041712_19052.csv"),
        Path("eval_results/hybrid_rollouts/A01-Race.Challenge_1793658048487_19052.csv"),
        Path("eval_results/hybrid_rollouts_crossmap/A01-Race.Challenge_1792431937486_29380.csv"),
        Path("eval_results/hybrid_rollouts_crossmap/A01-Race.Challenge_1792431954646_29380.csv"),
        Path("eval_results/hybrid_rollouts_crossmap/A07-Race.Challenge_1792431962040_29380.csv"),
        Path("eval_results/hybrid_rollouts_crossmap/A07-Race.Challenge_1792431966943_29380.csv"),
    ),
}


def _relative_path(path: Path) -> str:
    return path.as_posix()


def _merge_observations(observations: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not observations:
        raise ValueError("at least one observation set is required")
    keys = tuple(observations[0])
    return {key: np.concatenate([np.asarray(item[key]) for item in observations]) for key in keys}


def calibrate_transition_split(
    source_paths: Mapping[str, Iterable[str | Path]],
    *,
    moving_speed_threshold_mps: float = 1.0,
    drag_model_mps2_per_mps: float = 0.05,
) -> dict[str, Any]:
    """Load disjoint source paths and summarize each named split."""

    normalized: dict[str, tuple[Path, ...]] = {
        split: tuple(Path(path) for path in paths) for split, paths in source_paths.items()
    }
    if not normalized or any(not paths for paths in normalized.values()):
        raise ValueError("each dynamics split must contain at least one source path")
    ownership: dict[Path, str] = {}
    for split, paths in normalized.items():
        for path in paths:
            if path in ownership:
                raise ValueError(f"source path appears in multiple splits: {path}")
            ownership[path] = split

    results: dict[str, Any] = {}
    for split, paths in normalized.items():
        trace_reports = []
        observation_sets = []
        for path in paths:
            samples = load_rollout_csv(path)
            observations = transition_observations(
                samples,
                moving_speed_threshold_mps=moving_speed_threshold_mps,
                drag_model_mps2_per_mps=drag_model_mps2_per_mps,
            )
            observation_sets.append(observations)
            trace_reports.append(
                {
                    "source": _relative_path(path),
                    "sample_count": len(samples),
                    "transition_count": len(observations["dt_s"]),
                    "metrics": summarize_transition_observations(
                        observations,
                        moving_speed_threshold_mps=moving_speed_threshold_mps,
                        drag_model_mps2_per_mps=drag_model_mps2_per_mps,
                    ),
                }
            )
        aggregate = summarize_transition_observations(
            _merge_observations(observation_sets),
            moving_speed_threshold_mps=moving_speed_threshold_mps,
            drag_model_mps2_per_mps=drag_model_mps2_per_mps,
        )
        results[split] = {
            **aggregate,
            "trace_count": len(trace_reports),
            "traces": trace_reports,
        }
    return results


def _finite_metric_values(value: Any) -> bool:
    if isinstance(value, (bool, int)):
        return True
    if isinstance(value, float):
        return bool(np.isfinite(value))
    if isinstance(value, dict):
        return all(_finite_metric_values(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_metric_values(item) for item in value)
    return isinstance(value, str)


def build_report() -> dict[str, Any]:
    splits = calibrate_transition_split(DEFAULT_SOURCE_SPLITS)
    fit = splits["fit"]
    heldout = splits["heldout"]
    fit_moving_speed = fit["displacement_rate"]["moving_p50_mps"]
    heldout_moving_speed = heldout["displacement_rate"]["moving_p50_mps"]
    fit_residual = fit["position_prediction_residual"]["moving_p90_m"]
    heldout_residual = heldout["position_prediction_residual"]["moving_p90_m"]
    return {
        "scope": "offline_recorded_transition_dynamics_calibration",
        "trackmania_launched": False,
        "training_started": False,
        "checkpoints_loaded": False,
        "real_physics_validated": False,
        "controller_defaults_modified": False,
        "promotion_allowed": False,
        "source_paths": {split: [_relative_path(path) for path in paths] for split, paths in DEFAULT_SOURCE_SPLITS.items()},
        "splits": splits,
        "fit_to_heldout": {
            "moving_p50_displacement_rate_delta_mps": float(heldout_moving_speed - fit_moving_speed),
            "moving_p90_position_residual_delta_m": float(heldout_residual - fit_residual),
            "fit_moving_p50_displacement_rate_mps": float(fit_moving_speed),
            "heldout_moving_p50_displacement_rate_mps": float(heldout_moving_speed),
            "fit_moving_p90_position_residual_m": float(fit_residual),
            "heldout_moving_p90_position_residual_m": float(heldout_residual),
        },
        "simulator_reference": {
            "dt_s": 0.1,
            "linear_drag_mps2_per_mps": 0.05,
            "action_response_parameters_estimated": False,
        },
        "identifiability": {
            "telemetry_contains_actions": False,
            "throttle_response_identifiable": False,
            "brake_response_identifiable": False,
            "steering_response_identifiable": False,
            "allowed_conclusion": "one-step residual and telemetry consistency only",
        },
        "finite_metrics": _finite_metric_values(splits) and _finite_metric_values(
            {
                "fit_to_heldout": {
                    "moving_p50_displacement_rate_delta_mps": float(heldout_moving_speed - fit_moving_speed),
                    "moving_p90_position_residual_delta_m": float(heldout_residual - fit_residual),
                }
            }
        ),
        "limitations": [
            "Saved telemetry has no throttle, brake, or steering action columns.",
            "The one-step predictor is observational and cannot identify causal action response.",
            "Heading, speed, and position include TrackMania telemetry and slip/jump effects not modeled by the stub.",
            "The result does not calibrate or promote an MPC parameter or live controller setting.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fit_transitions": report["splits"]["fit"]["transition_count"],
                "heldout_transitions": report["splits"]["heldout"]["transition_count"],
                "finite_metrics": report["finite_metrics"],
                "promotion_allowed": report["promotion_allowed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
