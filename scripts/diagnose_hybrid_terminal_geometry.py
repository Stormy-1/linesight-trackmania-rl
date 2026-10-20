"""Explain offline hybrid strict-finish failures at the terminal geometry.

This report compares the endpoint of each repaired, recorded rollout with the
same saved centerline and with a fresh run of the deterministic closed-loop
simulator.  It is deliberately offline: it does not launch TrackMania, train,
load checkpoints, use live control, or call external services.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Support both ``python scripts/...`` from the workspace root and module use
# without installing the project or consulting anything outside the workspace.
if __package__ in (None, ""):
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

from trackmania_rl.hybrid_rollout import RolloutSample, load_rollout_csv
from trackmania_rl.hybrid_simulator import ClosedLoopConfig, simulate_closed_loop
from trackmania_rl.hybrid_state import (
    CenterlineGeometry,
    TrackRelativeState,
    build_centerline_geometry,
    make_track_relative_state,
)


DEFAULT_TELEMETRY_DIR = Path("eval_results/hybrid_rollouts_crossmap")
DEFAULT_MAPS_DIR = Path("maps")
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_terminal_geometry_diagnostic.json")
DEFAULT_MAX_STEPS = ClosedLoopConfig().max_steps
DEFAULT_SEARCH_RADIUS = ClosedLoopConfig().search_radius
MAP_INPUTS = {
    "A01": ("A01-Race.repaired.csv", "A01-Race_0.5m_cl2.npy"),
    "A07": ("A07-Race.repaired.csv", "A07-Race_0.5m_cl2.npy"),
}

ZERO_VERTICAL_MOTION_TOLERANCE_M = 1e-12


def _load_geometry(path: Path) -> CenterlineGeometry:
    return build_centerline_geometry(np.load(path, allow_pickle=False))


def _match_samples(
    samples: list[RolloutSample],
    geometry: CenterlineGeometry,
    *,
    search_radius: int | None,
    horizontal_only: bool,
) -> list[TrackRelativeState]:
    """Map-match a repaired telemetry sequence with the normal state helper."""

    states: list[TrackRelativeState] = []
    previous_segment: int | None = None
    for sample in samples:
        state = make_track_relative_state(
            sample.position,
            sample.heading,
            sample.speed_mps,
            geometry,
            previous_segment=previous_segment,
            search_radius=search_radius,
            horizontal_only=horizontal_only,
        )
        states.append(state)
        previous_segment = state.segment_index
    return states


def _errors(state: TrackRelativeState) -> dict[str, float]:
    return {
        "horizontal_m": float(state.horizontal_distance_to_centerline_m),
        "distance_3d_m": float(state.distance_to_centerline_m),
        "elevation_m": float(state.elevation_error_m),
        "absolute_elevation_m": abs(float(state.elevation_error_m)),
    }


def _gate_outcome(
    state: TrackRelativeState,
    geometry: CenterlineGeometry,
    config: ClosedLoopConfig,
) -> dict[str, Any]:
    track_length = float(geometry.cumulative_lengths[-1])
    progress_limit = track_length - float(config.finish_progress_tolerance_m)
    errors = _errors(state)
    progress_complete = bool(state.progress_m >= progress_limit)
    strict_finish = bool(progress_complete and errors["distance_3d_m"] <= config.finish_distance_limit_m)
    elevation_tolerant_finish = bool(
        progress_complete
        and errors["horizontal_m"] <= config.finish_distance_limit_m
        and errors["absolute_elevation_m"] <= config.finish_elevation_tolerance_m
    )
    return {
        "projected_progress_completion": progress_complete,
        "progress_complete": progress_complete,
        "progress_m": float(state.progress_m),
        "progress_gate_m": progress_limit,
        "strict_finish": strict_finish,
        "strict_3d_finish": strict_finish,
        "elevation_tolerant_finish": elevation_tolerant_finish,
        "errors": errors,
        "gate_limits": {
            "finish_progress_tolerance_m": float(config.finish_progress_tolerance_m),
            "finish_distance_limit_m": float(config.finish_distance_limit_m),
            "finish_elevation_tolerance_m": float(config.finish_elevation_tolerance_m),
        },
    }


def _terminal_decomposition(
    result: Any,
    initial_position: np.ndarray,
    config: ClosedLoopConfig,
) -> dict[str, Any]:
    final_position = np.asarray(result.final_position, dtype=np.float64)
    initial_y = float(initial_position[1])
    final_y = float(final_position[1])
    vertical_displacement = final_y - initial_y
    errors = _errors(result.final_state)
    reconstructed = float(np.hypot(errors["horizontal_m"], errors["absolute_elevation_m"]))
    strict_failure = not bool(result.strict_3d_finished)
    horizontal_within_limit = errors["horizontal_m"] <= config.finish_distance_limit_m
    elevation_component_greater = errors["absolute_elevation_m"] > errors["horizontal_m"]
    elevation_dominated = bool(
        strict_failure
        and bool(result.progress_complete)
        and horizontal_within_limit
        and elevation_component_greater
    )
    return {
        "initial_position_y_m": initial_y,
        "final_position_y_m": final_y,
        "vertical_displacement_m": vertical_displacement,
        "zero_vertical_motion": abs(vertical_displacement) <= ZERO_VERTICAL_MOTION_TOLERANCE_M,
        "zero_vertical_motion_tolerance_m": ZERO_VERTICAL_MOTION_TOLERANCE_M,
        "final_errors": errors,
        "terminal_3d_error_decomposition": {
            "horizontal_component_m": errors["horizontal_m"],
            "elevation_component_m": errors["absolute_elevation_m"],
            "reported_3d_error_m": errors["distance_3d_m"],
            "reconstructed_3d_error_m": reconstructed,
            "pythagorean_residual_m": errors["distance_3d_m"] - reconstructed,
        },
        "strict_failure_analysis": {
            "strict_failure": strict_failure,
            "projected_progress_complete": bool(result.progress_complete),
            "horizontal_within_strict_distance_limit": horizontal_within_limit,
            "elevation_component_greater_than_horizontal": elevation_component_greater,
            "elevation_dominated": elevation_dominated,
            "comparison": (
                "elevation component > horizontal component, progress complete, and "
                "horizontal component is within the strict distance limit"
            ),
        },
    }


def _diagnosis(
    recorded: dict[str, Any],
    simulator: dict[str, Any],
) -> str:
    recorded_gate = recorded["default_finish_gate"]
    analysis = simulator["strict_failure_analysis"]
    if analysis["elevation_dominated"]:
        simulator_finding = (
            "the simulator's strict terminal failure is elevation-dominated by the numeric comparison "
            "of its terminal components"
        )
    elif simulator["strict_finish"]:
        simulator_finding = "the simulator reaches the strict finish gate"
    else:
        simulator_finding = "the simulator does not reach the strict finish gate"
    return (
        f"Recorded telemetry ends at {recorded['end_projected_progress_m']:.3f} m projected progress "
        f"with default strict={recorded_gate['strict_finish']} and "
        f"elevation-tolerant={recorded_gate['elevation_tolerant_finish']}; {simulator_finding}. "
        "The recorded endpoint is telemetry evidence, while the simulator comparison describes a "
        "limitation or behavior of the bounded kinematic model."
    )


def _map_report(
    label: str,
    telemetry_path: Path,
    map_path: Path,
    *,
    max_steps: int,
    search_radius: int | None,
    default_config: ClosedLoopConfig,
) -> dict[str, Any]:
    samples = load_rollout_csv(telemetry_path)
    geometry = _load_geometry(map_path)
    states = _match_samples(
        samples,
        geometry,
        search_radius=search_radius,
        horizontal_only=default_config.map_match_horizontal_only,
    )
    start_state = states[0]
    end_state = states[-1]
    recorded_gate = _gate_outcome(end_state, geometry, default_config)

    effective_config = ClosedLoopConfig(
        max_steps=max_steps,
        search_radius=search_radius,
    )
    result = simulate_closed_loop(
        geometry,
        initial_position=geometry.points[0],
        initial_heading=geometry.segment_tangents[0],
        initial_speed_mps=0.0,
        config=effective_config,
    )
    simulator = {
        "initial_position_y_m": float(geometry.points[0, 1]),
        "final_position_y_m": float(result.final_position[1]),
        "vertical_displacement_m": float(result.final_position[1] - geometry.points[0, 1]),
        "projected_progress_completion": bool(result.progress_complete),
        "progress_complete": bool(result.progress_complete),
        "strict_finish": bool(result.strict_3d_finished),
        "strict_3d_finish": bool(result.strict_3d_finished),
        "elevation_tolerant_finish": bool(result.elevation_tolerant_finished),
        "termination_reason": str(result.termination_reason),
        "steps": int(len(result.steps)),
        "final_progress_m": float(result.final_state.progress_m),
        "track_length_m": float(result.track_length_m),
    }
    simulator.update(_terminal_decomposition(result, geometry.points[0], effective_config))
    simulator["final_horizontal_error_m"] = simulator["final_errors"]["horizontal_m"]
    simulator["final_3d_error_m"] = simulator["final_errors"]["distance_3d_m"]
    simulator["final_elevation_error_m"] = simulator["final_errors"]["elevation_m"]
    simulator["terminal_strict_failure_elevation_dominated"] = simulator["strict_failure_analysis"][
        "elevation_dominated"
    ]

    report = {
        "map": label,
        "telemetry_file": str(telemetry_path),
        "centerline_file": str(map_path),
        "telemetry_samples": len(samples),
        "telemetry_start_projected_progress_m": float(start_state.progress_m),
        "telemetry_end_projected_progress_m": float(end_state.progress_m),
        "start_projected_progress_m": float(start_state.progress_m),
        "end_projected_progress_m": float(end_state.progress_m),
        "track_length_m": float(geometry.cumulative_lengths[-1]),
        "final_recorded_pose_errors": _errors(end_state),
        "centerline_elevation": {
            "start_m": float(geometry.points[0, 1]),
            "end_m": float(geometry.points[-1, 1]),
            "minimum_m": float(np.min(geometry.points[:, 1])),
            "maximum_m": float(np.max(geometry.points[:, 1])),
            "span_m": float(np.max(geometry.points[:, 1]) - np.min(geometry.points[:, 1])),
        },
        "default_finish_gate": recorded_gate,
        "simulator": simulator,
    }
    report["final_recorded_horizontal_error_m"] = report["final_recorded_pose_errors"]["horizontal_m"]
    report["final_recorded_3d_error_m"] = report["final_recorded_pose_errors"]["distance_3d_m"]
    report["final_recorded_elevation_error_m"] = report["final_recorded_pose_errors"]["elevation_m"]
    report["diagnosis"] = _diagnosis(report, simulator)
    return report


def build_report(
    telemetry_dir: Path = DEFAULT_TELEMETRY_DIR,
    maps_dir: Path = DEFAULT_MAPS_DIR,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    search_radius: int | None = DEFAULT_SEARCH_RADIUS,
) -> dict[str, Any]:
    """Build the terminal geometry report for the repaired A01/A07 traces."""

    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if search_radius is not None and (
        not isinstance(search_radius, int) or isinstance(search_radius, bool) or search_radius < 0
    ):
        raise ValueError("search_radius must be a non-negative integer or None")

    telemetry_dir = Path(telemetry_dir)
    maps_dir = Path(maps_dir)
    defaults_before = ClosedLoopConfig()
    maps = [
        _map_report(
            label,
            telemetry_dir / telemetry_filename,
            maps_dir / map_filename,
            max_steps=max_steps,
            search_radius=search_radius,
            default_config=defaults_before,
        )
        for label, (telemetry_filename, map_filename) in MAP_INPUTS.items()
    ]
    defaults_after = ClosedLoopConfig()
    unchanged_defaults = defaults_before == defaults_after
    if not unchanged_defaults:
        raise AssertionError("terminal geometry diagnostic modified ClosedLoopConfig defaults")

    elevation_dominated_count = sum(
        bool(item["simulator"]["strict_failure_analysis"]["elevation_dominated"]) for item in maps
    )
    conclusion = (
        "Recorded telemetry provides the endpoint progress and pose-error evidence for A01/A07. "
        f"The default simulator reports elevation-dominated strict failures for "
        f"{elevation_dominated_count} of {len(maps)} map(s) under the explicit component comparisons; "
        "that result identifies a bounded flat-vertical-motion simulator-model limitation, "
        "not a claim that the recorded telemetry itself is simulated."
    )
    return {
        "scope": "offline_hybrid_terminal_geometry_diagnostic",
        "offline_only": True,
        "safety": {
            "offline_only": True,
            "no_trackmania": True,
            "no_training": True,
            "no_checkpoints": True,
            "no_live_control": True,
            "no_external_services": True,
            "unchanged_defaults": unchanged_defaults,
            "trackmania_launched": False,
            "training_started": False,
            "checkpoints_loaded": False,
            "live_control_used": False,
        },
        "default_closed_loop_config": asdict(defaults_before),
        "effective_simulator_config": asdict(
            ClosedLoopConfig(max_steps=max_steps, search_radius=search_radius)
        ),
        "strict_gate_defaults_unchanged": {
            "finish_distance_limit_m": float(defaults_before.finish_distance_limit_m),
            "finish_elevation_tolerance_m": float(defaults_before.finish_elevation_tolerance_m),
            "finish_progress_tolerance_m": float(defaults_before.finish_progress_tolerance_m),
        },
        "maps": maps,
        "conclusion": conclusion,
        "limitations": [
            "The deterministic simulator is a bounded kinematic model, not TrackMania physics.",
            "Projected progress is map matching and does not by itself prove a finish.",
            "Recorded endpoint measurements and simulator terminal measurements are separate evidence sources.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-dir", type=Path, default=DEFAULT_TELEMETRY_DIR)
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--search-radius", type=int, default=DEFAULT_SEARCH_RADIUS)
    args = parser.parse_args()
    report = build_report(
        args.telemetry_dir,
        args.maps_dir,
        max_steps=args.max_steps,
        search_radius=args.search_radius,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "maps": len(report["maps"])}, indent=2))


if __name__ == "__main__":
    main()
