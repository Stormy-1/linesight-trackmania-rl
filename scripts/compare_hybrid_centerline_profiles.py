"""Compare an offline speed-profile candidate with live controller defaults.

The comparison runs only the deterministic, bounded hybrid simulator against
saved centerlines.  It does not import TrackMania/TMInterface, start training,
load checkpoints, or alter :class:`ConservativeControllerConfig` defaults.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.evaluate_hybrid_speed_candidates import derive_candidates, load_calibration
except ModuleNotFoundError:  # Running this file directly puts ``scripts`` first on sys.path.
    from evaluate_hybrid_speed_candidates import derive_candidates, load_calibration
from trackmania_rl.hybrid_controller import ConservativeCenterlineController, ConservativeControllerConfig
from trackmania_rl.hybrid_simulator import ClosedLoopConfig, ClosedLoopResult, simulate_closed_loop
from trackmania_rl.hybrid_state import build_centerline_geometry

DEFAULT_MAPS_DIR = Path("maps")
DEFAULT_CALIBRATION = Path("eval_results/hybrid_rollouts/speed_profile_calibration.json")
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/captured_p25_guardrail_centerline_comparison.json")
SELECTED_PROFILE = "captured_p25_guardrail"


def _mean_abs(values: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float64)))) if values else 0.0


def _metrics(result: ClosedLoopResult) -> dict[str, Any]:
    states = [step.state for step in result.steps]
    sample_count = max(len(states), 1)
    lateral = [float(state.lateral_error_m) for state in states]
    heading = [float(state.heading_error_rad) for state in states]
    return {
        "completion": {
            "finished": result.finished,
            "strict_3d_finished": result.strict_3d_finished,
            "elevation_tolerant_finished": result.elevation_tolerant_finished,
            "progress_complete": result.progress_complete,
            "termination_reason": result.termination_reason,
            "elapsed_s": result.elapsed_s,
        },
        "progress": {
            "progress_m": result.final_state.progress_m,
            "progress_gain_m": result.progress_gain_m,
            "track_length_m": result.track_length_m,
            "progress_fraction": float(result.final_state.progress_m / max(result.track_length_m, 1e-9)),
        },
        "finish_diagnostic": result.finish_diagnostic,
        "horizontal_error": {
            "mean_m": result.mean_horizontal_distance_to_centerline_m,
            "max_m": result.max_horizontal_distance_to_centerline_m,
            "final_m": result.final_state.horizontal_distance_to_centerline_m,
        },
        "elevation_error": {
            "mean_abs_m": result.mean_abs_elevation_error_m,
            "max_abs_m": result.max_abs_elevation_error_m,
            "final_abs_m": abs(float(result.final_state.elevation_error_m)),
            "final_signed_m": result.final_state.elevation_error_m,
        },
        "lateral_error": {
            "mean_abs_m": _mean_abs(lateral),
            "max_abs_m": result.max_abs_lateral_error_m,
            "final_abs_m": abs(float(result.final_state.lateral_error_m)),
            "final_signed_m": result.final_state.lateral_error_m,
        },
        "heading_error": {
            "mean_abs_rad": _mean_abs(heading),
            "max_abs_rad": result.max_abs_heading_error_rad,
            "final_abs_rad": abs(float(result.final_state.heading_error_rad)),
            "final_signed_rad": result.final_state.heading_error_rad,
        },
        "recovery": {
            "samples": result.recovery_samples,
            "rate": float(result.recovery_samples / sample_count),
            "first_divergence_event": result.first_divergence_event,
            "first_recovery_event": result.first_recovery_event,
            "divergence_recovery": result.divergence_recovery,
        },
        "section_metrics": result.section_metrics,
    }


def _delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Return candidate-minus-reference deltas for comparable map metrics."""

    scalar_paths = (
        ("progress", "progress_m"),
        ("progress", "progress_gain_m"),
        ("progress", "progress_fraction"),
        ("completion", "elapsed_s"),
        ("horizontal_error", "mean_m"),
        ("horizontal_error", "max_m"),
        ("horizontal_error", "final_m"),
        ("elevation_error", "mean_abs_m"),
        ("elevation_error", "max_abs_m"),
        ("elevation_error", "final_abs_m"),
        ("lateral_error", "mean_abs_m"),
        ("lateral_error", "max_abs_m"),
        ("lateral_error", "final_abs_m"),
        ("heading_error", "mean_abs_rad"),
        ("heading_error", "max_abs_rad"),
        ("heading_error", "final_abs_rad"),
        ("recovery", "samples"),
        ("recovery", "rate"),
    )
    result: dict[str, Any] = {}
    for group, field in scalar_paths:
        result.setdefault(group, {})[field] = float(candidate[group][field]) - float(reference[group][field])
    result["completion"] = {
        "finished_delta": int(bool(candidate["completion"]["finished"]))
        - int(bool(reference["completion"]["finished"])),
        "strict_3d_finished_delta": int(bool(candidate["completion"]["strict_3d_finished"]))
        - int(bool(reference["completion"]["strict_3d_finished"])),
        "elevation_tolerant_finished_delta": int(bool(candidate["completion"]["elevation_tolerant_finished"]))
        - int(bool(reference["completion"]["elevation_tolerant_finished"])),
        "progress_complete_delta": int(bool(candidate["completion"]["progress_complete"]))
        - int(bool(reference["completion"]["progress_complete"])),
    }
    return result


def _run_profile(
    geometry: Any,
    config: ConservativeControllerConfig,
    simulator_config: ClosedLoopConfig,
) -> dict[str, Any]:
    result = simulate_closed_loop(
        geometry,
        initial_position=geometry.points[0],
        initial_heading=geometry.segment_tangents[0],
        initial_speed_mps=18.0,
        controller=ConservativeCenterlineController(config),
        config=simulator_config,
    )
    return _metrics(result)


def _candidate_config(calibration_report: Mapping[str, Any]) -> ConservativeControllerConfig:
    for candidate in derive_candidates(calibration_report):
        if candidate.name == SELECTED_PROFILE:
            return candidate.config
    raise ValueError(f"selected profile {SELECTED_PROFILE!r} was not derived")


def analyze_centerline(
    path: Path,
    *,
    candidate_config: ConservativeControllerConfig,
    reference_config: ConservativeControllerConfig | None = None,
    max_steps: int = 10000,
    search_radius: int = 64,
) -> dict[str, Any]:
    """Run both profiles on one saved centerline and compare their metrics."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if search_radius < 0:
        raise ValueError("search_radius must be non-negative")
    geometry = build_centerline_geometry(np.load(path))
    simulator_config = ClosedLoopConfig(max_steps=max_steps, search_radius=search_radius)
    reference = _run_profile(geometry, reference_config or ConservativeControllerConfig(), simulator_config)
    candidate = _run_profile(geometry, candidate_config, simulator_config)
    return {
        "centerline": path.name,
        "length_m": float(geometry.cumulative_lengths[-1]),
        "reference": {"profile": "live_defaults_reference", **reference},
        "candidate": {"profile": SELECTED_PROFILE, **candidate},
        "comparison_deltas_candidate_minus_reference": _delta(candidate, reference),
    }


def build_report(
    maps_dir: Path = DEFAULT_MAPS_DIR,
    *,
    calibration_report: Mapping[str, Any],
    max_steps: int = 10000,
    search_radius: int = 64,
) -> dict[str, Any]:
    """Build a deterministic all-saved-centerline comparison report."""

    paths = sorted(maps_dir.glob("*.npy"))
    if not paths:
        raise ValueError(f"no .npy centerlines found in {maps_dir}")
    defaults_before = ConservativeControllerConfig()
    candidate_config = _candidate_config(calibration_report)
    centerlines = [
        analyze_centerline(
            path,
            candidate_config=candidate_config,
            reference_config=defaults_before,
            max_steps=max_steps,
            search_radius=search_radius,
        )
        for path in paths
    ]
    defaults_after = ConservativeControllerConfig()
    return {
        "scope": "offline_saved_centerline_speed_profile_comparison",
        "selected_profile": SELECTED_PROFILE,
        "calibration_input_scope": calibration_report.get("scope"),
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
        },
        "profiles": {
            "reference": {
                "name": "live_defaults_reference",
                "config": {
                    key: getattr(defaults_before, key)
                    for key in defaults_before.__dataclass_fields__
                },
            },
            "candidate": {
                "name": SELECTED_PROFILE,
                "config": {
                    key: getattr(candidate_config, key)
                    for key in candidate_config.__dataclass_fields__
                },
            },
        },
        "centerlines": centerlines,
        "summary": {
            "centerline_count": len(centerlines),
            "reference_strict_finished": sum(item["reference"]["completion"]["strict_3d_finished"] for item in centerlines),
            "candidate_strict_finished": sum(item["candidate"]["completion"]["strict_3d_finished"] for item in centerlines),
            "reference_elevation_tolerant_finished": sum(item["reference"]["completion"]["elevation_tolerant_finished"] for item in centerlines),
            "candidate_elevation_tolerant_finished": sum(item["candidate"]["completion"]["elevation_tolerant_finished"] for item in centerlines),
        },
        "limitations": [
            "The deterministic simulator is a bounded kinematic model, not TrackMania physics.",
            "Saved centerlines are evaluated from their first point with an idealized initial pose and 18 m/s speed.",
            "Candidate parameters are derived from captured telemetry percentiles and are descriptive only; defaults remain unchanged.",
            "The report does not validate collisions, jumps, TMInterface behavior, live control, training, Azure execution, or checkpoints.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--search-radius", type=int, default=64)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        args.maps_dir,
        calibration_report=load_calibration(args.calibration),
        max_steps=args.max_steps,
        search_radius=args.search_radius,
    )
    report["calibration_input"] = str(args.calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "centerlines": report["summary"]["centerline_count"],
                "candidate_strict_finished": report["summary"]["candidate_strict_finished"],
                "reference_strict_finished": report["summary"]["reference_strict_finished"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
