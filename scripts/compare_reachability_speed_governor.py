"""Compare the isolated reachability speed governor with controller defaults.

This is an offline-only deterministic-simulator experiment over the saved A01,
A07, and Hockolicious centerlines.  It does not launch TrackMania, training,
Azure, or checkpoint loading, and it never mutates the default controller
configuration.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineController,
    ConservativeCenterlineSpeedGovernor,
    ConservativeCenterlineSpeedGovernorConfig,
    ConservativeControllerConfig,
)
from trackmania_rl.hybrid_simulator import ClosedLoopConfig, ClosedLoopResult, simulate_closed_loop
from trackmania_rl.hybrid_state import build_centerline_geometry

DEFAULT_MAPS_DIR = Path("maps")
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/reachability_speed_governor_centerline_comparison.json")
DEFAULT_CENTERLINES = {
    "A01": "A01-Race_0.5m_cl2.npy",
    "A07": "A07-Race_0.5m_cl2.npy",
    "Hockolicious": "ESL-Hockolicious_0.5m_cl2.npy",
}


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _mean_abs(values: list[float]) -> float:
    return _mean([abs(value) for value in values])


def _metrics(result: ClosedLoopResult) -> dict[str, Any]:
    states = [step.state for step in result.steps]
    commands = [step.command for step in result.steps]
    sample_count = max(len(states), 1)
    actual_speeds = [float(state.speed_mps) for state in states]
    targets = [float(command.target_speed_mps) for command in commands]
    speed_gaps = [actual - target for actual, target in zip(actual_speeds, targets)]
    lateral = [float(state.lateral_error_m) for state in states]
    heading = [float(state.heading_error_rad) for state in states]
    elevation = [float(state.elevation_error_m) for state in states]
    horizontal = [float(state.horizontal_distance_to_centerline_m) for state in states]
    return {
        "completion": {
            "finished": result.finished,
            "strict_3d_finished": result.strict_3d_finished,
            "elevation_tolerant_finished": result.elevation_tolerant_finished,
            "progress_complete": result.progress_complete,
            "termination_reason": result.termination_reason,
            "steps": len(result.steps),
            "elapsed_s": result.elapsed_s,
        },
        "progress": {
            "start_m": float(states[0].progress_m) if states else 0.0,
            "final_m": float(result.final_state.progress_m),
            "gain_m": result.progress_gain_m,
            "track_length_m": result.track_length_m,
            "fraction": float(result.final_state.progress_m / max(result.track_length_m, 1e-9)),
            "nonmonotonic_progress_steps": result.nonmonotonic_progress_steps,
        },
        "speed_target_behavior": {
            "mean_actual_speed_mps": _mean(actual_speeds),
            "mean_target_speed_mps": _mean(targets),
            "target_min_mps": min(targets) if targets else None,
            "target_max_mps": max(targets) if targets else None,
            "mean_actual_minus_target_mps": _mean(speed_gaps),
            "mean_abs_actual_minus_target_mps": _mean_abs(speed_gaps),
            "fraction_actual_above_target": float(sum(gap > 0.0 for gap in speed_gaps) / sample_count),
        },
        "horizontal_error": {
            "mean_m": _mean(horizontal),
            "max_m": max(horizontal, default=0.0),
            "final_m": float(result.final_state.horizontal_distance_to_centerline_m),
        },
        "lateral_error": {
            "mean_abs_m": _mean_abs(lateral),
            "max_abs_m": max((abs(value) for value in lateral), default=0.0),
            "final_abs_m": abs(float(result.final_state.lateral_error_m)),
            "final_signed_m": float(result.final_state.lateral_error_m),
        },
        "heading_error": {
            "mean_abs_rad": _mean_abs(heading),
            "max_abs_rad": max((abs(value) for value in heading), default=0.0),
            "final_abs_rad": abs(float(result.final_state.heading_error_rad)),
            "final_signed_rad": float(result.final_state.heading_error_rad),
        },
        "elevation_error": {
            "mean_abs_m": _mean_abs(elevation),
            "max_abs_m": max((abs(value) for value in elevation), default=0.0),
            "final_abs_m": abs(float(result.final_state.elevation_error_m)),
            "final_signed_m": float(result.final_state.elevation_error_m),
        },
        "recovery": {
            "samples": result.recovery_samples,
            "rate": float(result.recovery_samples / sample_count),
            "first_divergence_event": result.first_divergence_event,
            "first_recovery_event": result.first_recovery_event,
            "divergence_recovery": result.divergence_recovery,
        },
        "finish_diagnostic": result.finish_diagnostic,
        "section_metrics": result.section_metrics,
    }


def _run(
    geometry: Any,
    controller: ConservativeCenterlineController,
    simulator_config: ClosedLoopConfig,
) -> dict[str, Any]:
    result = simulate_closed_loop(
        geometry,
        initial_position=geometry.points[0],
        initial_heading=geometry.segment_tangents[0],
        initial_speed_mps=18.0,
        controller=controller,
        config=simulator_config,
    )
    return _metrics(result)


def _delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    scalar_fields = {
        "progress": ("final_m", "gain_m", "fraction", "nonmonotonic_progress_steps"),
        "speed_target_behavior": (
            "mean_actual_speed_mps",
            "mean_target_speed_mps",
            "mean_actual_minus_target_mps",
            "mean_abs_actual_minus_target_mps",
            "fraction_actual_above_target",
        ),
        "horizontal_error": ("mean_m", "max_m", "final_m"),
        "lateral_error": ("mean_abs_m", "max_abs_m", "final_abs_m"),
        "heading_error": ("mean_abs_rad", "max_abs_rad", "final_abs_rad"),
        "elevation_error": ("mean_abs_m", "max_abs_m", "final_abs_m"),
        "recovery": ("samples", "rate"),
    }
    result: dict[str, Any] = {}
    for group, fields in scalar_fields.items():
        result[group] = {
            field: float(candidate[group][field]) - float(reference[group][field]) for field in fields
        }
    result["completion"] = {
        field: int(bool(candidate["completion"][field])) - int(bool(reference["completion"][field]))
        for field in ("finished", "strict_3d_finished", "elevation_tolerant_finished", "progress_complete")
    }
    return result


def _resolve_centerlines(maps_dir: Path) -> list[tuple[str, Path]]:
    resolved: list[tuple[str, Path]] = []
    for label, filename in DEFAULT_CENTERLINES.items():
        path = maps_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"saved centerline for {label} was not found: {path}")
        resolved.append((label, path))
    return resolved


def compare_centerline(
    label: str,
    path: Path,
    *,
    governor_config: ConservativeCenterlineSpeedGovernorConfig,
    max_steps: int = 10000,
    search_radius: int = 64,
) -> dict[str, Any]:
    """Run default and governed controllers against one saved centerline."""

    if max_steps <= 0 or search_radius < 0:
        raise ValueError("max_steps must be positive and search_radius must be non-negative")
    geometry = build_centerline_geometry(np.load(path))
    simulator_config = ClosedLoopConfig(max_steps=max_steps, search_radius=search_radius)
    defaults = _run(geometry, ConservativeCenterlineController(), simulator_config)
    governed = _run(geometry, ConservativeCenterlineSpeedGovernor(governor_config), simulator_config)
    return {
        "name": label,
        "centerline": path.name,
        "length_m": float(geometry.cumulative_lengths[-1]),
        "reference": {"profile": "live_defaults_reference", **defaults},
        "governed": {"profile": "reachability_speed_governor", **governed},
        "comparison_deltas_governed_minus_reference": _delta(governed, defaults),
    }


def build_report(
    maps_dir: Path = DEFAULT_MAPS_DIR,
    *,
    governor_config: ConservativeCenterlineSpeedGovernorConfig | None = None,
    max_steps: int = 10000,
    search_radius: int = 64,
) -> dict[str, Any]:
    """Build the deterministic three-centerline comparison report."""

    defaults_before = ConservativeControllerConfig()
    active_governor_config = governor_config or ConservativeCenterlineSpeedGovernorConfig()
    centerlines = [
        compare_centerline(
            label,
            path,
            governor_config=active_governor_config,
            max_steps=max_steps,
            search_radius=search_radius,
        )
        for label, path in _resolve_centerlines(maps_dir)
    ]
    defaults_after = ConservativeControllerConfig()
    return {
        "scope": "offline_reachability_speed_governor_saved_centerline_comparison",
        "controller_defaults_modified": defaults_before != defaults_after,
        "trackmania_launched": False,
        "training_started": False,
        "azure_started": False,
        "checkpoints_loaded": False,
        "real_physics_validated": False,
        "simulator": {
            "dt_s": 0.1,
            "max_steps": max_steps,
            "search_radius_segments": search_radius,
            "map_match_horizontal_only": True,
            "initial_speed_mps": 18.0,
        },
        "profiles": {
            "reference": {
                "name": "live_defaults_reference",
                "config": {key: getattr(defaults_before, key) for key in defaults_before.__dataclass_fields__},
            },
            "governed": {
                "name": "reachability_speed_governor",
                "config": {
                    key: getattr(active_governor_config, key)
                    for key in active_governor_config.__dataclass_fields__
                },
            },
        },
        "centerlines": centerlines,
        "summary": {
            "centerline_count": len(centerlines),
            "reference_finished": sum(item["reference"]["completion"]["finished"] for item in centerlines),
            "governed_finished": sum(item["governed"]["completion"]["finished"] for item in centerlines),
            "reference_progress_complete": sum(item["reference"]["completion"]["progress_complete"] for item in centerlines),
            "governed_progress_complete": sum(item["governed"]["completion"]["progress_complete"] for item in centerlines),
        },
        "limitations": [
            "The deterministic simulator is a bounded kinematic model, not TrackMania physics.",
            "Saved centerlines are evaluated from their first point with an idealized initial pose and 18 m/s speed.",
            "The governor uses centerline curvature and configured acceleration limits; it does not model tire load, jumps, collisions, or surface physics.",
            "Non-monotonic progress counts are available from simulator state transitions; no live telemetry is involved.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--search-radius", type=int, default=64)
    parser.add_argument("--control-horizon-s", type=float, default=1.0)
    parser.add_argument("--reachable-acceleration-mps2", type=float, default=10.0)
    parser.add_argument("--max-lateral-acceleration-mps2", type=float, default=12.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = ConservativeCenterlineSpeedGovernorConfig(
        control_horizon_s=args.control_horizon_s,
        reachable_acceleration_mps2=args.reachable_acceleration_mps2,
        max_lateral_acceleration_mps2=args.max_lateral_acceleration_mps2,
    )
    report = build_report(
        args.maps_dir,
        governor_config=config,
        max_steps=args.max_steps,
        search_radius=args.search_radius,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "centerlines": len(report["centerlines"]), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
