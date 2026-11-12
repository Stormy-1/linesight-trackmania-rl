"""Sweep the isolated speed-governor limits on saved centerlines.

This experiment is deliberately offline-only.  It reads the three checked-in
centerline arrays, runs the deterministic closed-loop simulator, and writes a
JSON report.  It does not import TrackMania/TMInterface, start training, load
checkpoints, or contact Azure or any other external service.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineSpeedGovernor,
    ConservativeCenterlineSpeedGovernorConfig,
    ConservativeControllerConfig,
)
from trackmania_rl.hybrid_simulator import ClosedLoopConfig, ClosedLoopResult, simulate_closed_loop
from trackmania_rl.hybrid_state import build_centerline_geometry

DEFAULT_MAPS_DIR = Path("maps")
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/reachability_speed_governor_parameter_sweep.json")
DEFAULT_CENTERLINES = {
    "A01": "A01-Race_0.5m_cl2.npy",
    "A07": "A07-Race_0.5m_cl2.npy",
    "Hockolicious": "ESL-Hockolicious_0.5m_cl2.npy",
}

# The default grid is intentionally small enough to run as a repeatable local
# check while still covering shorter/longer lookahead horizons and
# lower/higher lateral acceleration limits.
DEFAULT_CONTROL_HORIZONS_S = (0.25, 0.75, 1.25)
DEFAULT_MAX_LATERAL_ACCELERATIONS_MPS2 = (6.0, 12.0, 18.0)
MAX_GRID_CONFIGURATIONS = 16

# Viability is based on the simulator's finish gates, rather than a hidden
# tuning score.  A partial result made progress to every centerline end but
# failed at least one strict finish distance/elevation gate.
VIABILITY_RULES = {
    "viable": "every saved centerline reaches the strict 3D finish gate",
    "partial": "every saved centerline reaches the progress-complete gate, but at least one misses the strict finish gate",
    "non_viable": "at least one saved centerline does not reach the progress-complete gate",
}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _mean_abs(values: Iterable[float]) -> float:
    return _mean(abs(value) for value in values)


def _metrics(result: ClosedLoopResult) -> dict[str, Any]:
    """Convert one simulator result into JSON-safe diagnostics."""

    states = [step.state for step in result.steps]
    sample_count = max(len(states), 1)
    horizontal = [float(state.horizontal_distance_to_centerline_m) for state in states]
    lateral = [float(state.lateral_error_m) for state in states]
    heading = [float(state.heading_error_rad) for state in states]
    elevation = [float(state.elevation_error_m) for state in states]
    nonmonotonic = int(result.nonmonotonic_progress_steps)
    metrics = {
        "completion": {
            "finished": bool(result.finished),
            "strict_3d_finished": bool(result.strict_3d_finished),
            "elevation_tolerant_finished": bool(result.elevation_tolerant_finished),
            "progress_complete": bool(result.progress_complete),
            "termination_reason": result.termination_reason,
            "steps": len(result.steps),
            "elapsed_s": float(result.elapsed_s),
        },
        "progress": {
            "start_m": float(states[0].progress_m) if states else 0.0,
            "final_m": float(result.final_state.progress_m),
            "gain_m": float(result.progress_gain_m),
            "track_length_m": float(result.track_length_m),
            "fraction": float(result.final_state.progress_m / max(result.track_length_m, 1e-9)),
            "nonmonotonic_progress_steps": nonmonotonic,
        },
        # Keep this direct alias easy to discover for consumers that do not
        # need to know the nested progress metric naming convention.
        "non_monotonic_progress_steps": nonmonotonic,
        "elapsed_s": float(result.elapsed_s),
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
            "samples": int(result.recovery_samples),
            "rate": float(result.recovery_samples / sample_count),
            "first_divergence_event": result.first_divergence_event,
            "first_recovery_event": result.first_recovery_event,
            "divergence_recovery": result.divergence_recovery,
        },
    }
    return metrics


def _validate_grid(values: Sequence[float], *, name: str, positive: bool) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    normalized = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain only finite values")
    if positive and any(value <= 0.0 for value in normalized):
        raise ValueError(f"{name} must contain only positive values")
    if not positive and any(value < 0.0 for value in normalized):
        raise ValueError(f"{name} must contain only non-negative values")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate values")
    return normalized


def _validate_grid_size(control_horizons_s: Sequence[float], lateral_accelerations: Sequence[float]) -> None:
    count = len(control_horizons_s) * len(lateral_accelerations)
    if count > MAX_GRID_CONFIGURATIONS:
        raise ValueError(
            f"parameter grid has {count} configurations; maximum is {MAX_GRID_CONFIGURATIONS}"
        )


def _configuration_id(control_horizon_s: float, max_lateral_acceleration_mps2: float) -> str:
    def token(value: float) -> str:
        return format(value, ".12g").replace("-", "m").replace(".", "p")

    return f"horizon_{token(control_horizon_s)}s_lateral_{token(max_lateral_acceleration_mps2)}mps2"


def _resolve_centerlines(maps_dir: Path) -> list[tuple[str, Path]]:
    resolved: list[tuple[str, Path]] = []
    for label, filename in DEFAULT_CENTERLINES.items():
        path = maps_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"saved centerline for {label} was not found: {path}")
        resolved.append((label, path))
    return resolved


def _run(geometry: Any, config: ConservativeCenterlineSpeedGovernorConfig, simulator_config: ClosedLoopConfig) -> dict[str, Any]:
    result = simulate_closed_loop(
        geometry,
        initial_position=geometry.points[0],
        initial_heading=geometry.segment_tangents[0],
        initial_speed_mps=18.0,
        controller=ConservativeCenterlineSpeedGovernor(config),
        config=simulator_config,
    )
    return _metrics(result)


def _aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot aggregate an empty result set")
    completion = {
        "centerline_count": len(results),
        "finished_count": sum(bool(item["completion"]["finished"]) for item in results),
        "progress_complete_count": sum(bool(item["completion"]["progress_complete"]) for item in results),
        "strict_3d_finished_fraction": _mean(bool(item["completion"]["strict_3d_finished"]) for item in results),
        "progress_complete_fraction": _mean(bool(item["completion"]["progress_complete"]) for item in results),
    }
    completion["all_finished"] = completion["finished_count"] == completion["centerline_count"]
    completion["all_progress_complete"] = completion["progress_complete_count"] == completion["centerline_count"]
    if completion["all_finished"]:
        viability = "viable"
    elif completion["all_progress_complete"]:
        viability = "partial"
    else:
        viability = "non_viable"

    return {
        "viability": viability,
        "completion": completion,
        "progress": {
            "mean_final_m": _mean(item["progress"]["final_m"] for item in results),
            "mean_gain_m": _mean(item["progress"]["gain_m"] for item in results),
            "mean_fraction": _mean(item["progress"]["fraction"] for item in results),
            "min_fraction": min(item["progress"]["fraction"] for item in results),
            "nonmonotonic_progress_steps": sum(item["progress"]["nonmonotonic_progress_steps"] for item in results),
        },
        "elapsed_s": {
            "mean": _mean(item["elapsed_s"] for item in results),
            "total": sum(float(item["elapsed_s"]) for item in results),
            "max": max(float(item["elapsed_s"]) for item in results),
        },
        "horizontal_error": {
            "mean_m": _mean(item["horizontal_error"]["mean_m"] for item in results),
            "max_m": max(item["horizontal_error"]["max_m"] for item in results),
            "mean_final_m": _mean(item["horizontal_error"]["final_m"] for item in results),
        },
        "lateral_error": {
            "mean_abs_m": _mean(item["lateral_error"]["mean_abs_m"] for item in results),
            "max_abs_m": max(item["lateral_error"]["max_abs_m"] for item in results),
            "mean_final_abs_m": _mean(item["lateral_error"]["final_abs_m"] for item in results),
        },
        "heading_error": {
            "mean_abs_rad": _mean(item["heading_error"]["mean_abs_rad"] for item in results),
            "max_abs_rad": max(item["heading_error"]["max_abs_rad"] for item in results),
            "mean_final_abs_rad": _mean(item["heading_error"]["final_abs_rad"] for item in results),
        },
        "elevation_error": {
            "mean_abs_m": _mean(item["elevation_error"]["mean_abs_m"] for item in results),
            "max_abs_m": max(item["elevation_error"]["max_abs_m"] for item in results),
            "mean_final_abs_m": _mean(item["elevation_error"]["final_abs_m"] for item in results),
        },
        "recovery": {
            "samples": sum(item["recovery"]["samples"] for item in results),
            "rate_mean": _mean(item["recovery"]["rate"] for item in results),
            "runs_with_recovery": sum(item["recovery"]["samples"] > 0 for item in results),
        },
    }


_PARETO_OBJECTIVES = (
    ("completion", "strict_3d_finished_fraction", True),
    ("progress", "mean_fraction", True),
    ("elapsed_s", "mean", False),
    ("horizontal_error", "mean_m", False),
    ("lateral_error", "mean_abs_m", False),
    ("heading_error", "mean_abs_rad", False),
    ("elevation_error", "mean_abs_m", False),
    ("recovery", "rate_mean", False),
    ("progress", "nonmonotonic_progress_steps", False),
)


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    at_least_as_good = True
    strictly_better = False
    for group, field, maximize in _PARETO_OBJECTIVES:
        left_value = float(left["aggregate"][group][field])
        right_value = float(right["aggregate"][group][field])
        if maximize:
            at_least_as_good &= left_value >= right_value
            strictly_better |= left_value > right_value
        else:
            at_least_as_good &= left_value <= right_value
            strictly_better |= left_value < right_value
    return at_least_as_good and strictly_better


def _classify_pareto(configurations: list[dict[str, Any]]) -> None:
    for candidate in configurations:
        dominators = [other["configuration_id"] for other in configurations if other is not candidate and _dominates(other, candidate)]
        candidate["pareto"] = {
            "classification": "dominated" if dominators else "pareto_optimal",
            "dominated_by": dominators,
            "dominance_count": len(dominators),
            "objectives": [
                {"group": group, "field": field, "direction": "maximize" if maximize else "minimize"}
                for group, field, maximize in _PARETO_OBJECTIVES
            ],
        }


def build_report(
    maps_dir: Path = DEFAULT_MAPS_DIR,
    *,
    control_horizons_s: Sequence[float] = DEFAULT_CONTROL_HORIZONS_S,
    max_lateral_accelerations_mps2: Sequence[float] = DEFAULT_MAX_LATERAL_ACCELERATIONS_MPS2,
    max_steps: int = 3000,
    search_radius: int = 64,
) -> dict[str, Any]:
    """Build a deterministic report for every point in the bounded grid."""

    horizons = _validate_grid(control_horizons_s, name="control_horizons_s", positive=False)
    lateral_accelerations = _validate_grid(
        max_lateral_accelerations_mps2,
        name="max_lateral_accelerations_mps2",
        positive=True,
    )
    _validate_grid_size(horizons, lateral_accelerations)
    if max_steps <= 0 or search_radius < 0:
        raise ValueError("max_steps must be positive and search_radius must be non-negative")

    defaults_before = ConservativeControllerConfig()
    centerline_paths = _resolve_centerlines(maps_dir)
    geometries = [(label, path, build_centerline_geometry(np.load(path, allow_pickle=False))) for label, path in centerline_paths]
    simulator_config = ClosedLoopConfig(max_steps=max_steps, search_radius=search_radius)
    configurations: list[dict[str, Any]] = []
    for control_horizon_s, max_lateral_acceleration_mps2 in itertools.product(horizons, lateral_accelerations):
        governor_config = ConservativeCenterlineSpeedGovernorConfig(
            control_horizon_s=control_horizon_s,
            max_lateral_acceleration_mps2=max_lateral_acceleration_mps2,
        )
        results = []
        for label, path, geometry in geometries:
            metrics = _run(geometry, governor_config, simulator_config)
            results.append(
                {
                    "name": label,
                    "centerline": path.name,
                    "length_m": float(geometry.cumulative_lengths[-1]),
                    **metrics,
                    "viability": "viable" if metrics["completion"]["finished"] else (
                        "partial" if metrics["completion"]["progress_complete"] else "non_viable"
                    ),
                }
            )
        aggregate = _aggregate(results)
        configurations.append(
            {
                "configuration_id": _configuration_id(control_horizon_s, max_lateral_acceleration_mps2),
                "control_horizon_s": control_horizon_s,
                "max_lateral_acceleration_mps2": max_lateral_acceleration_mps2,
                "centerlines": results,
                "aggregate": aggregate,
                "viability": aggregate["viability"],
            }
        )
    _classify_pareto(configurations)

    return {
        "scope": "offline_reachability_speed_governor_parameter_sweep",
        "offline_only": True,
        "controller_defaults_modified": defaults_before != ConservativeControllerConfig(),
        "trackmania_launched": False,
        "training_started": False,
        "azure_started": False,
        "checkpoints_loaded": False,
        "real_physics_validated": False,
        "simulator": {
            "dt_s": simulator_config.dt_s,
            "max_steps": max_steps,
            "search_radius_segments": search_radius,
            "map_match_horizontal_only": simulator_config.map_match_horizontal_only,
            "initial_speed_mps": 18.0,
        },
        "parameter_grid": {
            "control_horizon_s": list(horizons),
            "max_lateral_acceleration_mps2": list(lateral_accelerations),
            "configuration_count": len(configurations),
            "bounded_max_configurations": MAX_GRID_CONFIGURATIONS,
        },
        "viability_rules": VIABILITY_RULES,
        "configurations": configurations,
        "summary": {
            "centerline_count": len(geometries),
            "configuration_count": len(configurations),
            "viable_configurations": sum(item["viability"] == "viable" for item in configurations),
            "partial_configurations": sum(item["viability"] == "partial" for item in configurations),
            "non_viable_configurations": sum(item["viability"] == "non_viable" for item in configurations),
            "pareto_optimal_configurations": [
                item["configuration_id"]
                for item in configurations
                if item["pareto"]["classification"] == "pareto_optimal"
            ],
        },
        "limitations": [
            "The deterministic simulator is a bounded kinematic model, not TrackMania physics.",
            "Saved centerlines are evaluated from their first point with an idealized initial pose and 18 m/s speed.",
            "The sweep changes only control_horizon_s and max_lateral_acceleration_mps2; reachable acceleration and inherited controller defaults stay unchanged.",
            "Elapsed time is simulator time (steps multiplied by dt), not wall-clock runtime.",
            "Pareto status uses aggregate strict-finish/progress, elapsed, error, recovery, and non-monotonic-progress objectives listed on each configuration.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--search-radius", type=int, default=64)
    parser.add_argument("--control-horizon-s", type=float, nargs="+", default=DEFAULT_CONTROL_HORIZONS_S)
    parser.add_argument(
        "--max-lateral-acceleration-mps2",
        type=float,
        nargs="+",
        default=DEFAULT_MAX_LATERAL_ACCELERATIONS_MPS2,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        args.maps_dir,
        control_horizons_s=args.control_horizon_s,
        max_lateral_accelerations_mps2=args.max_lateral_acceleration_mps2,
        max_steps=args.max_steps,
        search_radius=args.search_radius,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "configurations": report["summary"]["configuration_count"],
                "summary": report["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
