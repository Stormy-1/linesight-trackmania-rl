"""Run deterministic closed-loop controller scenarios and write a report.

This is a software-in-the-loop check only.  The vehicle model is deliberately
simple and the output must not be interpreted as TrackMania completion evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_simulator import (
    ClosedLoopConfig,
    simulate_closed_loop,
    synthetic_centerline_geometry,
)


def _run_scenario(name: str, *, initial_position: np.ndarray, initial_heading: np.ndarray, initial_speed_mps: float) -> dict[str, Any]:
    geometry = synthetic_centerline_geometry(
        length_m=120.0,
        spacing_m=2.0,
        amplitude_m=6.0,
        wavelength_m=70.0,
    )
    result = simulate_closed_loop(
        geometry,
        initial_position=initial_position,
        initial_heading=initial_heading,
        initial_speed_mps=initial_speed_mps,
        config=ClosedLoopConfig(dt_s=0.1, max_steps=1600),
    )
    return {
        "name": name,
        "finished": result.finished,
        "termination_reason": result.termination_reason,
        "elapsed_s": result.elapsed_s,
        "steps": len(result.steps),
        "progress_gain_m": result.progress_gain_m,
        "recovery_samples": result.recovery_samples,
        "max_abs_lateral_error_m": result.max_abs_lateral_error_m,
        "max_abs_heading_error_rad": result.max_abs_heading_error_rad,
        "final_progress_m": result.final_state.progress_m,
        "final_lateral_error_m": result.final_state.lateral_error_m,
        "final_heading_error_rad": result.final_state.heading_error_rad,
        "final_speed_mps": result.final_speed_mps,
    }


def build_report() -> dict[str, Any]:
    geometry = synthetic_centerline_geometry(
        length_m=120.0,
        spacing_m=2.0,
        amplitude_m=6.0,
        wavelength_m=70.0,
    )
    initial_heading = geometry.segment_tangents[0]
    scenarios = [
        _run_scenario(
            "nominal_centerline_start",
            initial_position=geometry.points[0],
            initial_heading=initial_heading,
            initial_speed_mps=18.0,
        ),
        _run_scenario(
            "off_center_recovery_start",
            initial_position=np.asarray([0.0, 0.0, 12.0]),
            initial_heading=np.asarray([np.cos(0.3), 0.0, np.sin(0.3)]),
            initial_speed_mps=16.0,
        ),
    ]
    return {
        "scope": "deterministic_software_in_the_loop_controller_validation",
        "trackmania_launched": False,
        "training_started": False,
        "checkpoints_loaded": False,
        "real_physics_validated": False,
        "centerline": {
            "length_m": float(geometry.cumulative_lengths[-1]),
            "points": len(geometry.points),
        },
        "scenarios": scenarios,
        "limitations": [
            "The vehicle model is a bounded kinematic stub.",
            "This does not validate TrackMania physics, jumps, collisions, wheel contact, or TMInterface.",
            "A recorded telemetry rollout is still required for real map-matching validation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("eval_results/hybrid_controller_offline_validation.json"))
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scenarios": len(report["scenarios"]), "all_finished": all(item["finished"] for item in report["scenarios"])}, indent=2))


if __name__ == "__main__":
    main()
