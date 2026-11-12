"""Compare the nominal controller with the isolated local-plan controller.

This is deterministic software-in-the-loop evidence only.  It does not launch
TrackMania, load checkpoints, train, or modify controller defaults.
"""

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
from trackmania_rl.hybrid_planner import LocalTrajectoryPlannerConfig, plan_local_trajectory
from trackmania_rl.hybrid_simulator import (
    ClosedLoopConfig,
    ClosedLoopResult,
    simulate_closed_loop,
    synthetic_centerline_geometry,
)

DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/local_trajectory_speed_planning.json")


def _mean_abs(values: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float64)))) if values else 0.0


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


def _run_controller(controller: Any, geometry: Any) -> list[dict[str, Any]]:
    simulator_config = ClosedLoopConfig(dt_s=0.1, max_steps=1600)
    scenarios = []
    for name, position, heading, speed in _scenario_inputs(geometry):
        result = simulate_closed_loop(
            geometry,
            initial_position=position,
            initial_heading=heading,
            initial_speed_mps=speed,
            controller=controller,
            config=simulator_config,
        )
        scenarios.append({"name": name, **_scenario_metrics(result)})
    return scenarios


def _planning_constraint_metrics(geometry: Any, planner_config: LocalTrajectoryPlannerConfig) -> dict[str, Any]:
    violations = 0
    fallback_violations = 0
    plan_count = 0
    for _, position, heading, speed in _scenario_inputs(geometry):
        result = simulate_closed_loop(
            geometry,
            initial_position=position,
            initial_heading=heading,
            initial_speed_mps=speed,
            controller=PlannedCenterlineController(
                PlannedCenterlineControllerConfig(planner_config=planner_config)
            ),
            config=ClosedLoopConfig(dt_s=0.1, max_steps=1600),
        )
        for step in result.steps:
            plan = plan_local_trajectory(step.state, geometry, planner_config)
            plan_count += 1
            distances = np.diff(plan.distances_m)
            speeds = plan.target_speeds_mps
            fallback = plan.fallback_target_speeds_mps
            for index, distance in enumerate(distances):
                if speeds[index + 1] ** 2 > speeds[index] ** 2 + 2.0 * planner_config.max_acceleration_mps2 * distance + 1e-8:
                    violations += 1
                if speeds[index] ** 2 > speeds[index + 1] ** 2 + 2.0 * planner_config.max_braking_mps2 * distance + 1e-8:
                    violations += 1
                if fallback[index + 1] ** 2 > fallback[index] ** 2 + 2.0 * planner_config.max_acceleration_mps2 * distance + 1e-8:
                    fallback_violations += 1
                if fallback[index] ** 2 > fallback[index + 1] ** 2 + 2.0 * planner_config.max_braking_mps2 * distance + 1e-8:
                    fallback_violations += 1
    return {
        "plans_checked": plan_count,
        "normal_constraint_violations": violations,
        "fallback_constraint_violations": fallback_violations,
    }


def build_report() -> dict[str, Any]:
    geometry = synthetic_centerline_geometry(
        length_m=120.0,
        spacing_m=2.0,
        amplitude_m=6.0,
        wavelength_m=70.0,
    )
    planner_config = LocalTrajectoryPlannerConfig()
    candidate_config = PlannedCenterlineControllerConfig(planner_config=planner_config)
    nominal = _run_controller(ConservativeCenterlineController(), geometry)
    candidate = _run_controller(PlannedCenterlineController(candidate_config), geometry)
    repeated_candidate = _run_controller(PlannedCenterlineController(candidate_config), geometry)
    determinism = candidate == repeated_candidate
    return {
        "scope": "offline_local_trajectory_and_speed_planning_comparison",
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
        "planner_config": asdict(planner_config),
        "profiles": {
            "nominal_reference": nominal,
            "planned_candidate": candidate,
        },
        "determinism": {
            "candidate_repeat_matches": determinism,
        },
        "planning_constraints": _planning_constraint_metrics(geometry, planner_config),
        "limitations": [
            "The vehicle model is a bounded kinematic stub.",
            "The centerline is synthetic and does not validate TrackMania map physics.",
            "Jumps, collisions, wheel contact, sliding, and TMInterface behavior are not modeled.",
            "This result does not authorize a live controller change or residual-RL training.",
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
                "candidate_finished": all(item["finished"] for item in report["profiles"]["planned_candidate"]),
                "candidate_repeat_matches": report["determinism"]["candidate_repeat_matches"],
                "constraint_violations": report["planning_constraints"]["normal_constraint_violations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
