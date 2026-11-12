"""Run the offline controller stub against every saved centerline.

This is a comparative software-in-the-loop diagnostic. It is intentionally not
TrackMania evidence: the vehicle model has no game physics, collision model, or
TMInterface integration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_simulator import ClosedLoopConfig, simulate_closed_loop
from trackmania_rl.hybrid_state import build_centerline_geometry


def analyze_centerline(path: Path, *, max_steps: int = 10000) -> dict[str, Any]:
    points = np.load(path)
    geometry = build_centerline_geometry(points)
    result = simulate_closed_loop(
        geometry,
        initial_position=geometry.points[0],
        initial_heading=geometry.segment_tangents[0],
        initial_speed_mps=18.0,
        # Saved centerlines are densely sampled (roughly 0.5 m).  A local
        # temporal search prevents nearest-point matching from jumping to a
        # visually similar section elsewhere on a looping map.
        config=ClosedLoopConfig(max_steps=max_steps, search_radius=64),
    )
    total_length = float(geometry.cumulative_lengths[-1])
    return {
        "centerline": path.name,
        "length_m": total_length,
        "finished": result.finished,
        "strict_3d_finish": result.strict_3d_finished,
        "elevation_tolerant_finish": result.elevation_tolerant_finished,
        "elevation_tolerant_finished": result.elevation_tolerant_finished,
        "termination_reason": result.termination_reason,
        "steps": len(result.steps),
        "elapsed_s": result.elapsed_s,
        "progress_m": result.final_state.progress_m,
        "progress_fraction": float(result.final_state.progress_m / total_length),
        "progress_complete": result.progress_complete,
        "recovery_samples": result.recovery_samples,
        "recovery_rate": float(result.recovery_samples / len(result.steps)),
        "max_abs_lateral_error_m": result.max_abs_lateral_error_m,
        "mean_horizontal_distance_to_centerline_m": result.mean_horizontal_distance_to_centerline_m,
        "max_horizontal_distance_to_centerline_m": result.max_horizontal_distance_to_centerline_m,
        "mean_abs_elevation_error_m": result.mean_abs_elevation_error_m,
        "max_abs_elevation_error_m": result.max_abs_elevation_error_m,
        "max_abs_heading_error_rad": result.max_abs_heading_error_rad,
        "final_lateral_error_m": result.final_state.lateral_error_m,
        "final_horizontal_distance_to_centerline_m": result.final_state.horizontal_distance_to_centerline_m,
        "final_elevation_error_m": result.final_state.elevation_error_m,
        "final_heading_error_rad": result.final_state.heading_error_rad,
        "finish_diagnostic": result.finish_diagnostic,
        "divergence_recovery": result.divergence_recovery,
        "first_divergence_event": result.first_divergence_event,
        "first_recovery_event": result.first_recovery_event,
        "section_metrics": result.section_metrics,
    }


def build_report(maps_dir: Path, *, max_steps: int = 10000) -> dict[str, Any]:
    paths = sorted(maps_dir.glob("*.npy"))
    if not paths:
        raise ValueError(f"no .npy centerlines found in {maps_dir}")
    return {
        "scope": "offline_saved_centerline_software_in_the_loop_diagnostic",
        "trackmania_launched": False,
        "training_started": False,
        "checkpoints_loaded": False,
        "real_physics_validated": False,
        "max_steps": max_steps,
        "map_match_search_radius_segments": 64,
        "centerlines": [analyze_centerline(path, max_steps=max_steps) for path in paths],
        "limitations": [
            "The vehicle model is a bounded kinematic stub.",
            "A finished stub run is not a TrackMania completion claim.",
            "Real telemetry is still required to validate map matching and controller behavior.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-dir", type=Path, default=Path("maps"))
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("eval_results/hybrid_centerline_validation.json"))
    args = parser.parse_args()
    report = build_report(args.maps_dir, max_steps=args.max_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "centerlines": len(report["centerlines"]),
                "stub_finished": sum(item["finished"] for item in report["centerlines"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
