"""Compare the isolated recovery-aware controller with conservative defaults.

The experiment uses only the deterministic offline simulator.  It covers a
nominal synthetic start, an off-center synthetic recovery start, and the saved
A01, A07, and Hockolicious centerlines.  No TrackMania, training, Azure,
checkpoint, or external-service code is imported or started.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineController,
    ConservativeControllerConfig,
    RecoveryAwareCenterlineController,
    RecoveryAwareCenterlineControllerConfig,
)
from trackmania_rl.hybrid_simulator import (
    ClosedLoopConfig,
    ClosedLoopResult,
    simulate_closed_loop,
    synthetic_centerline_geometry,
)
from trackmania_rl.hybrid_state import (
    CenterlineGeometry,
    build_centerline_geometry,
)

DEFAULT_MAPS_DIR = Path("maps")
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/recovery_aware_centerline_comparison.json")
DEFAULT_CENTERLINES = {
    "A01": "A01-Race_0.5m_cl2.npy",
    "A07": "A07-Race_0.5m_cl2.npy",
    "Hockolicious": "ESL-Hockolicious_0.5m_cl2.npy",
}


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _mean_abs(values: list[float]) -> float:
    return _mean([abs(value) for value in values])


def _progress_metrics(result: ClosedLoopResult, states: list[Any]) -> dict[str, Any]:
    progress = [float(state.progress_m) for state in states]
    progress.append(float(result.final_state.progress_m))
    backward_steps = [max(previous - current, 0.0) for previous, current in pairwise(progress)]
    count = sum(distance > 0.0 for distance in backward_steps)
    distance = float(sum(backward_steps))
    return {
        "start_m": progress[0] if progress else 0.0,
        "final_m": float(result.final_state.progress_m),
        "gain_m": result.progress_gain_m,
        "track_length_m": result.track_length_m,
        "fraction": float(result.final_state.progress_m / max(result.track_length_m, 1e-9)),
        "nonmonotonic_progress_steps": result.nonmonotonic_progress_steps,
        "non_monotonic_progress_steps": result.nonmonotonic_progress_steps,
        "nonmonotonic_progress_distance_m": distance,
        "non_monotonic_progress_distance_m": distance,
        "backward_progress_steps": count,
    }


def _metrics(result: ClosedLoopResult) -> dict[str, Any]:
    states = [step.state for step in result.steps]
    sample_count = max(len(states), 1)
    lateral = [float(state.lateral_error_m) for state in states]
    horizontal = [float(state.horizontal_distance_to_centerline_m) for state in states]
    heading = [float(state.heading_error_rad) for state in states]
    elevation = [float(state.elevation_error_m) for state in states]
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
        "progress": _progress_metrics(result, states),
        "lateral_error": {
            "mean_abs_m": _mean_abs(lateral),
            "max_abs_m": max((abs(value) for value in lateral), default=0.0),
            "final_abs_m": abs(float(result.final_state.lateral_error_m)),
            "final_signed_m": float(result.final_state.lateral_error_m),
        },
        "horizontal_error": {
            "mean_m": _mean(horizontal),
            "max_m": max(horizontal, default=0.0),
            "final_m": float(result.final_state.horizontal_distance_to_centerline_m),
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


def _delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Return candidate-minus-reference numeric deltas for each metric group."""

    fields = {
        "progress": (
            "final_m",
            "gain_m",
            "fraction",
            "nonmonotonic_progress_steps",
            "non_monotonic_progress_steps",
            "nonmonotonic_progress_distance_m",
            "non_monotonic_progress_distance_m",
            "backward_progress_steps",
        ),
        "lateral_error": ("mean_abs_m", "max_abs_m", "final_abs_m"),
        "horizontal_error": ("mean_m", "max_m", "final_m"),
        "heading_error": ("mean_abs_rad", "max_abs_rad", "final_abs_rad"),
        "elevation_error": ("mean_abs_m", "max_abs_m", "final_abs_m"),
        "recovery": ("samples", "rate"),
    }
    result: dict[str, Any] = {
        group: {field: float(candidate[group][field]) - float(reference[group][field]) for field in group_fields}
        for group, group_fields in fields.items()
    }
    result["completion"] = {
        field: int(bool(candidate["completion"][field])) - int(bool(reference["completion"][field]))
        for field in ("finished", "strict_3d_finished", "elevation_tolerant_finished", "progress_complete")
    }
    return result


def _run(
    geometry: CenterlineGeometry,
    *,
    initial_position: np.ndarray,
    initial_heading: np.ndarray,
    controller: ConservativeCenterlineController,
    simulator_config: ClosedLoopConfig,
) -> dict[str, Any]:
    result = simulate_closed_loop(
        geometry,
        initial_position=initial_position,
        initial_heading=initial_heading,
        initial_speed_mps=18.0,
        controller=controller,
        config=simulator_config,
    )
    return _metrics(result)


def compare_geometry(
    name: str,
    geometry: CenterlineGeometry,
    *,
    initial_position: np.ndarray,
    initial_heading: np.ndarray,
    recovery_config: RecoveryAwareCenterlineControllerConfig,
    max_steps: int,
    search_radius: int,
) -> dict[str, Any]:
    """Compare both controllers for one geometry and initial pose."""

    if max_steps <= 0 or search_radius < 0:
        raise ValueError("max_steps must be positive and search_radius must be non-negative")
    simulator_config = ClosedLoopConfig(max_steps=max_steps, search_radius=search_radius)
    reference = _run(
        geometry,
        initial_position=initial_position,
        initial_heading=initial_heading,
        controller=ConservativeCenterlineController(),
        simulator_config=simulator_config,
    )
    candidate = _run(
        geometry,
        initial_position=initial_position,
        initial_heading=initial_heading,
        controller=RecoveryAwareCenterlineController(recovery_config),
        simulator_config=simulator_config,
    )
    return {
        "name": name,
        "length_m": float(geometry.cumulative_lengths[-1]),
        "reference": {"profile": "live_defaults_reference", **reference},
        "recovery_aware": {"profile": "recovery_aware_centerline", **candidate},
        "comparison_deltas_recovery_aware_minus_reference": _delta(candidate, reference),
    }


def _synthetic_cases() -> list[tuple[str, CenterlineGeometry, np.ndarray, np.ndarray]]:
    geometry = synthetic_centerline_geometry(length_m=120.0, spacing_m=2.0, amplitude_m=6.0, wavelength_m=70.0)
    return [
        (
            "synthetic_nominal_centerline_start",
            geometry,
            geometry.points[0],
            geometry.segment_tangents[0],
        ),
        (
            "synthetic_off_center_recovery_start",
            geometry,
            np.asarray([0.0, 0.0, 12.0]),
            np.asarray([np.cos(0.3), 0.0, np.sin(0.3)]),
        ),
    ]


def _resolve_centerlines(maps_dir: Path) -> list[tuple[str, Path]]:
    resolved: list[tuple[str, Path]] = []
    for label, filename in DEFAULT_CENTERLINES.items():
        path = maps_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"saved centerline for {label} was not found: {path}")
        resolved.append((label, path))
    return resolved


def build_report(
    maps_dir: Path = DEFAULT_MAPS_DIR,
    *,
    recovery_config: RecoveryAwareCenterlineControllerConfig | None = None,
    max_steps: int = 10000,
    search_radius: int = 64,
) -> dict[str, Any]:
    """Build synthetic and saved-centerline comparisons."""

    defaults_before = ConservativeControllerConfig()
    active_config = recovery_config or RecoveryAwareCenterlineControllerConfig()
    synthetic = [
        compare_geometry(
            name,
            geometry,
            initial_position=position,
            initial_heading=heading,
            recovery_config=active_config,
            max_steps=max_steps,
            search_radius=search_radius,
        )
        for name, geometry, position, heading in _synthetic_cases()
    ]
    centerlines = []
    for label, path in _resolve_centerlines(maps_dir):
        geometry = build_centerline_geometry(np.load(path, allow_pickle=False))
        centerlines.append(
            compare_geometry(
                label,
                geometry,
                initial_position=geometry.points[0],
                initial_heading=geometry.segment_tangents[0],
                recovery_config=active_config,
                max_steps=max_steps,
                search_radius=search_radius,
            )
        )
    defaults_after = ConservativeControllerConfig()
    return {
        "scope": "offline_recovery_aware_centerline_controller_comparison",
        "offline_only": True,
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
            "recovery_aware": {
                "name": "recovery_aware_centerline",
                "config": {key: getattr(active_config, key) for key in active_config.__dataclass_fields__},
            },
        },
        "synthetic_scenarios": synthetic,
        "centerlines": centerlines,
        "summary": {
            "synthetic_scenario_count": len(synthetic),
            "centerline_count": len(centerlines),
            "reference_finished": sum(item["reference"]["completion"]["finished"] for item in synthetic + centerlines),
            "recovery_aware_finished": sum(item["recovery_aware"]["completion"]["finished"] for item in synthetic + centerlines),
            "reference_progress_complete": sum(item["reference"]["completion"]["progress_complete"] for item in synthetic + centerlines),
            "recovery_aware_progress_complete": sum(item["recovery_aware"]["completion"]["progress_complete"] for item in synthetic + centerlines),
        },
        "limitations": [
            "The deterministic simulator is a bounded kinematic model, not TrackMania physics.",
            "Synthetic and saved-centerline runs start from idealized poses; no live telemetry is involved.",
            "Horizontal and elevation errors use the existing map-matching state convention and do not model collisions, jumps, or wheel contact.",
            "The comparison does not validate TrackMania control, TMInterface behavior, training, Azure execution, or checkpoints.",
        ],
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
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "synthetic_scenarios": report["summary"]["synthetic_scenario_count"],
                "centerlines": report["summary"]["centerline_count"],
                "summary": report["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
