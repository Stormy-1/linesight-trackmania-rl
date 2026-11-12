"""Profile observable short-horizon risk features on saved centerlines.

The analysis runs the unchanged conservative controller in the deterministic
simulator and records map-relative predictive features without applying them.
It is a planner-readiness diagnostic for upcoming geometry and speed margin,
not a controller change or a claim about real TrackMania dynamics.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_controller import ConservativeCenterlineController
from trackmania_rl.hybrid_planner import PredictiveRiskFeatures, predictive_risk_features
from trackmania_rl.hybrid_simulator import ClosedLoopConfig, simulate_closed_loop
from trackmania_rl.hybrid_state import build_centerline_geometry

DEFAULT_MAPS_DIR = Path("maps")
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_predictive_features.json")
LOOKAHEAD_DISTANCES_M = (5.0, 10.0, 20.0, 30.0)


def _summary(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return {"min": None, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _feature_record(features: PredictiveRiskFeatures) -> dict[str, float]:
    return {
        "max_abs_lookahead_curvature": features.max_abs_lookahead_curvature,
        "min_lookahead_target_speed_mps": features.min_lookahead_target_speed_mps,
        "current_speed_mps": features.current_speed_mps,
        "speed_margin_to_min_target_mps": features.speed_margin_to_min_target_mps,
        "max_abs_tangent_change_rad": features.max_abs_tangent_change_rad,
        "lateral_acceleration_demand_mps2": features.lateral_acceleration_demand_mps2,
        "lateral_acceleration_margin_mps2": features.lateral_acceleration_margin_mps2,
        "estimated_braking_distance_m": features.estimated_braking_distance_m,
        "one_step_forward_alignment_margin_rad": features.one_step_forward_alignment_margin_rad,
    }


def _profile_summary(records: list[dict[str, Any]], result) -> dict[str, Any]:
    speed_margins = [float(item["speed_margin_to_min_target_mps"]) for item in records]
    lateral_margins = [float(item["lateral_acceleration_margin_mps2"]) for item in records]
    alignment_margins = [float(item["one_step_forward_alignment_margin_rad"]) for item in records]
    return {
        "sample_count": len(records),
        "completion": {
            "finished": bool(result.finished),
            "progress_complete": bool(result.progress_complete),
            "termination_reason": result.termination_reason,
            "elapsed_s": float(result.elapsed_s),
        },
        "risk_event_rates": {
            "speed_above_minimum_lookahead_target": (
                sum(value < 0.0 for value in speed_margins) / len(speed_margins) if speed_margins else 0.0
            ),
            "lateral_demand_above_configured_limit": (
                sum(value < 0.0 for value in lateral_margins) / len(lateral_margins) if lateral_margins else 0.0
            ),
            "one_step_forward_alignment_gap": (
                sum(value < 0.0 for value in alignment_margins) / len(alignment_margins)
                if alignment_margins
                else 0.0
            ),
        },
        "feature_distributions": {
            field: _summary(float(item[field]) for item in records)
            for field in (
                "max_abs_lookahead_curvature",
                "min_lookahead_target_speed_mps",
                "current_speed_mps",
                "speed_margin_to_min_target_mps",
                "max_abs_tangent_change_rad",
                "lateral_acceleration_demand_mps2",
                "lateral_acceleration_margin_mps2",
                "estimated_braking_distance_m",
                "one_step_forward_alignment_margin_rad",
            )
        },
        "first_observable_risk_steps": {
            "speed_target_excess": next(
                (int(item["step_index"]) for item in records if item["speed_margin_to_min_target_mps"] < 0.0),
                None,
            ),
            "lateral_acceleration_excess": next(
                (int(item["step_index"]) for item in records if item["lateral_acceleration_margin_mps2"] < 0.0),
                None,
            ),
            "one_step_alignment_gap": next(
                (int(item["step_index"]) for item in records if item["one_step_forward_alignment_margin_rad"] < 0.0),
                None,
            ),
        },
    }


def analyze_centerline(path: Path, *, max_steps: int = 10000, search_radius: int = 64) -> dict[str, Any]:
    if max_steps <= 0 or search_radius < 0:
        raise ValueError("max_steps must be positive and search_radius must be non-negative")
    geometry = build_centerline_geometry(np.load(path, allow_pickle=False))
    result = simulate_closed_loop(
        geometry,
        initial_position=geometry.points[0],
        initial_heading=geometry.segment_tangents[0],
        initial_speed_mps=18.0,
        controller=ConservativeCenterlineController(),
        config=ClosedLoopConfig(max_steps=max_steps, search_radius=search_radius),
    )
    records: list[dict[str, Any]] = []
    for step in result.steps:
        features = predictive_risk_features(
            step.state,
            geometry,
            lookahead_distances_m=LOOKAHEAD_DISTANCES_M,
        )
        records.append(
            {
                "step_index": step.step_index,
                "timestamp_s": float(step.timestamp_s),
                "progress_m": float(step.state.progress_m),
                "recovery": bool(step.command.recovery),
                **_feature_record(features),
            }
        )
    return {
        "map": path.name,
        "track_length_m": float(geometry.cumulative_lengths[-1]),
        "lookahead_distances_m": list(LOOKAHEAD_DISTANCES_M),
        "controller": "ConservativeCenterlineController defaults",
        "simulator": {
            "dt_s": 0.1,
            "max_steps": max_steps,
            "search_radius_segments": search_radius,
            "initial_speed_mps": 18.0,
        },
        "summary": _profile_summary(records, result),
        "records": records,
    }


def build_report(maps_dir: Path = DEFAULT_MAPS_DIR, *, max_steps: int = 10000, search_radius: int = 64) -> dict[str, Any]:
    paths = sorted(maps_dir.glob("*.npy"))
    if not paths:
        raise ValueError(f"no centerline .npy files found in {maps_dir}")
    maps = [analyze_centerline(path, max_steps=max_steps, search_radius=search_radius) for path in paths]
    return {
        "scope": "offline_hybrid_predictive_map_features",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "controller_defaults_modified": False,
        "future_disturbance_information_used": False,
        "lookahead_distances_m": list(LOOKAHEAD_DISTANCES_M),
        "maps": maps,
        "interpretation": {
            "diagnostic_only": True,
            "features_are_observable": True,
            "actions_modified": False,
            "next_action": "compare feature warnings with recorded real transitions before planner integration",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--search-radius", type=int, default=64)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.maps_dir, max_steps=args.max_steps, search_radius=args.search_radius)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "maps": len(report["maps"])}, separators=(",", ":")))


if __name__ == "__main__":
    main()
