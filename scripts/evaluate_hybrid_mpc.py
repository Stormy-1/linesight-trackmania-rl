"""Compare nominal, planned, and bounded short-horizon MPC controllers offline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineController,
    PlannedCenterlineController,
    PlannedCenterlineControllerConfig,
)
from trackmania_rl.hybrid_mpc import ShortHorizonMPCConfig, ShortHorizonMPCController
from trackmania_rl.hybrid_simulator import (
    ClosedLoopConfig,
    ClosedLoopResult,
    simulate_closed_loop,
    synthetic_centerline_geometry,
)

DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/local_trajectory_mpc.json")


def _mean_abs(values: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float64)))) if values else 0.0


def _scenario_inputs(geometry: Any) -> tuple[tuple[str, np.ndarray, np.ndarray, float], ...]:
    return (
        ("nominal_centerline_start", geometry.points[0], geometry.segment_tangents[0], 18.0),
        (
            "off_center_recovery_start",
            np.asarray([0.0, 0.0, 12.0]),
            np.asarray([np.cos(0.3), 0.0, np.sin(0.3)]),
            16.0,
        ),
    )


def _scenario_metrics(result: ClosedLoopResult) -> dict[str, Any]:
    states = [step.state for step in result.steps]
    commands = [step.command for step in result.steps]
    target_gaps = [float(state.speed_mps - command.target_speed_mps) for state, command in zip(states, commands)]
    return {
        "finished": result.finished,
        "termination_reason": result.termination_reason,
        "elapsed_s": result.elapsed_s,
        "steps": len(result.steps),
        "progress_gain_m": result.progress_gain_m,
        "progress_fraction": float(result.final_state.progress_m / max(result.track_length_m, 1e-9)),
        "recovery_samples": result.recovery_samples,
        "mean_abs_lateral_error_m": _mean_abs([float(state.lateral_error_m) for state in states]),
        "max_abs_lateral_error_m": result.max_abs_lateral_error_m,
        "mean_abs_heading_error_rad": _mean_abs([float(state.heading_error_rad) for state in states]),
        "max_abs_heading_error_rad": result.max_abs_heading_error_rad,
        "mean_actual_minus_target_speed_mps": float(np.mean(target_gaps)) if target_gaps else 0.0,
        "fraction_actual_above_target": float(sum(gap > 0.0 for gap in target_gaps) / max(len(target_gaps), 1)),
    }


def _run_profile(name: str, controller: Any, geometry: Any) -> list[dict[str, Any]]:
    scenarios = []
    for scenario_name, position, heading, speed in _scenario_inputs(geometry):
        result = simulate_closed_loop(
            geometry,
            initial_position=position,
            initial_heading=heading,
            initial_speed_mps=speed,
            controller=controller,
            config=ClosedLoopConfig(dt_s=0.1, max_steps=1600),
        )
        metrics = _scenario_metrics(result)
        if isinstance(controller, ShortHorizonMPCController):
            decisions = [controller.decision(step.state, geometry) for step in result.steps]
            metrics["mpc"] = {
                "candidate_count": decisions[0].candidate_count if decisions else 0,
                "mean_cost": float(np.mean([decision.cost for decision in decisions])) if decisions else 0.0,
                "max_cost": max((decision.cost for decision in decisions), default=0.0),
                "all_costs_finite": all(np.isfinite(decision.cost) for decision in decisions),
            }
        scenarios.append({"name": scenario_name, **metrics})
    return scenarios


def _profile_summary(profile: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "completion_rate": float(sum(item["finished"] for item in profile) / max(len(profile), 1)),
        "mean_abs_lateral_error_m": float(np.mean([item["mean_abs_lateral_error_m"] for item in profile])),
        "mean_abs_heading_error_rad": float(np.mean([item["mean_abs_heading_error_rad"] for item in profile])),
        "mean_elapsed_s": float(np.mean([item["elapsed_s"] for item in profile])),
    }


def build_report() -> dict[str, Any]:
    geometry = synthetic_centerline_geometry(
        length_m=120.0,
        spacing_m=2.0,
        amplitude_m=6.0,
        wavelength_m=70.0,
    )
    mpc_config = ShortHorizonMPCConfig(horizon_steps=8)
    nominal = _run_profile("nominal_reference", ConservativeCenterlineController(), geometry)
    planned = _run_profile(
        "local_plan_reference",
        PlannedCenterlineController(PlannedCenterlineControllerConfig()),
        geometry,
    )
    mpc = _run_profile("short_horizon_mpc_candidate", ShortHorizonMPCController(mpc_config), geometry)
    repeated_mpc = _run_profile("short_horizon_mpc_candidate", ShortHorizonMPCController(mpc_config), geometry)
    nominal_summary = _profile_summary(nominal)
    planned_summary = _profile_summary(planned)
    mpc_summary = _profile_summary(mpc)
    return {
        "scope": "offline_bounded_short_horizon_mpc_comparison",
        "trackmania_launched": False,
        "training_started": False,
        "checkpoints_loaded": False,
        "real_physics_validated": False,
        "controller_defaults_modified": False,
        "candidate_status": "diagnostic_only",
        "centerline": {
            "length_m": float(geometry.cumulative_lengths[-1]),
            "points": len(geometry.points),
        },
        "mpc_config": asdict(mpc_config),
        "profiles": {
            "nominal_reference": nominal,
            "local_plan_reference": planned,
            "short_horizon_mpc_candidate": mpc,
        },
        "comparison": {
            "nominal_reference": nominal_summary,
            "local_plan_reference": planned_summary,
            "short_horizon_mpc_candidate": mpc_summary,
            "mpc_minus_nominal_mean_abs_lateral_error_m": (
                mpc_summary["mean_abs_lateral_error_m"] - nominal_summary["mean_abs_lateral_error_m"]
            ),
            "mpc_better_than_nominal_tracking": (
                mpc_summary["mean_abs_lateral_error_m"] < nominal_summary["mean_abs_lateral_error_m"]
            ),
        },
        "determinism": {
            "candidate_repeat_matches": mpc == repeated_mpc,
        },
        "limitations": [
            "The prediction model is a bounded kinematic approximation.",
            "The action family holds steering and longitudinal action constant across each horizon.",
            "The centerline is synthetic and does not validate TrackMania physics, jumps, collisions, or TMInterface.",
            "This result does not authorize live controller changes or residual-RL training.",
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
                "candidate_finished": all(
                    item["finished"] for item in report["profiles"]["short_horizon_mpc_candidate"]
                ),
                "candidate_repeat_matches": report["determinism"]["candidate_repeat_matches"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
