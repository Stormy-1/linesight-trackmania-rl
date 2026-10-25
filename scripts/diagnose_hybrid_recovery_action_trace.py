"""Inspect deterministic recovery commands around boundary-study failures.

The report reruns selected held-out cases through the existing pure simulator
and records command/state diagnostics around the first backward-progress event.
It is analysis only: controller defaults, selection rules, and historical
metrics are unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineController,
    ConservativeControllerConfig,
    RecoveryAwareCenterlineController,
    RecoveryAwareCenterlineControllerConfig,
)
from trackmania_rl.hybrid_replay import (
    DisturbanceSpec,
    MidRolloutInjection,
    replay_midrollout_disturbance,
)
from trackmania_rl.hybrid_simulator import ClosedLoopConfig
from trackmania_rl.hybrid_state import build_centerline_geometry

DEFAULT_REPORTS = (
    Path("eval_results/hybrid_rollouts/hybrid_recovery_boundary_comparison_step25.json"),
    Path("eval_results/hybrid_rollouts/hybrid_recovery_boundary_comparison_step70.json"),
)
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_recovery_action_trace.json")
BACKWARD_PROGRESS_TOLERANCE_M = 1e-5


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"report {path} must contain a JSON object")
    return value


def _controller(profile: Mapping[str, Any]):
    config_values = profile.get("config")
    if not isinstance(config_values, Mapping):
        raise TypeError("profile config must be an object")
    role = str(profile.get("role", ""))
    if role == "recovery_parameter_candidate":
        return RecoveryAwareCenterlineController(RecoveryAwareCenterlineControllerConfig(**config_values))
    return ConservativeCenterlineController(ConservativeControllerConfig(**config_values))


def _first_backward_event(result, injection_step: int) -> dict[str, Any] | None:
    previous = result.steps[injection_step].state.progress_m
    for step in result.steps[injection_step + 1 :]:
        current = step.state.progress_m
        distance = float(previous - current)
        if distance > BACKWARD_PROGRESS_TOLERANCE_M:
            return {
                "step_index": step.step_index,
                "timestamp_s": step.timestamp_s,
                "distance_m": distance,
                "previous_progress_m": float(previous),
                "current_progress_m": float(current),
                "state": {
                    "lateral_error_m": float(step.state.lateral_error_m),
                    "heading_error_rad": float(step.state.heading_error_rad),
                    "horizontal_distance_to_centerline_m": float(
                        step.state.horizontal_distance_to_centerline_m
                    ),
                    "speed_mps": float(step.state.speed_mps),
                },
                "command": {
                    "steering": float(step.command.steering),
                    "throttle": float(step.command.throttle),
                    "brake": float(step.command.brake),
                    "target_speed_mps": float(step.command.target_speed_mps),
                    "recovery": bool(step.command.recovery),
                },
            }
        previous = current
    return None


def _trace_summary(result, injection_step: int) -> dict[str, Any]:
    suffix = result.steps[injection_step:]
    recovery_steps = [step for step in suffix if step.command.recovery]
    steering = [abs(float(step.command.steering)) for step in recovery_steps]
    brake = [float(step.command.brake) for step in recovery_steps]
    speed = [float(step.state.speed_mps) for step in recovery_steps]
    target_speed = [float(step.command.target_speed_mps) for step in recovery_steps]
    return {
        "recovery_sample_count": len(recovery_steps),
        "recovery_full_steering_count": sum(value >= 1.0 - 1e-12 for value in steering),
        "recovery_full_brake_count": sum(value >= 1.0 - 1e-12 for value in brake),
        "max_abs_recovery_steering": max(steering, default=0.0),
        "mean_recovery_brake": float(np.mean(brake)) if brake else None,
        "max_recovery_brake": max(brake, default=0.0),
        "max_recovery_speed_mps": max(speed, default=0.0),
        "min_recovery_speed_mps": min(speed, default=0.0),
        "mean_recovery_target_speed_mps": float(np.mean(target_speed)) if target_speed else None,
        "first_backward_event": _first_backward_event(result, injection_step),
    }


def _disturbances(report: Mapping[str, Any]) -> dict[str, DisturbanceSpec]:
    values = list(report.get("held_out_only_disturbances", [])) + list(report.get("disturbances", []))
    result: dict[str, DisturbanceSpec] = {}
    for value in values:
        if isinstance(value, Mapping) and isinstance(value.get("name"), str):
            result[value["name"]] = DisturbanceSpec(
                name=value["name"],
                lateral_offset_m=float(value.get("lateral_offset_m", 0.0)),
                heading_offset_rad=float(value.get("heading_offset_rad", 0.0)),
                speed_delta_mps=float(value.get("speed_delta_mps", 0.0)),
            )
    return result


def _run_case(report: Mapping[str, Any], map_name: str, profile_name: str, case: Mapping[str, Any]):
    map_entry = report["maps"][map_name]
    centerline = np.load(Path(map_entry["centerline_file"]), allow_pickle=False)
    geometry = build_centerline_geometry(centerline)
    anchor = next(
        anchor for anchor in map_entry["source_anchors"] if anchor["anchor_id"] == case["anchor_id"]
    )
    disturbance = _disturbances(report)[case["disturbance"]]
    simulator = report["simulator"]
    config = ClosedLoopConfig(
        dt_s=float(simulator["dt_s"]),
        max_steps=int(simulator["max_steps"]),
        search_radius=int(simulator["search_radius_segments"]),
    )
    result = replay_midrollout_disturbance(
        geometry,
        initial_position=np.asarray(anchor["position"], dtype=np.float64),
        initial_heading=np.asarray(anchor["heading"], dtype=np.float64),
        initial_speed_mps=float(anchor["speed_mps"]),
        injection=MidRolloutInjection(
            int(simulator["injection_step"]),
            disturbance,
        ),
        controller=_controller(report["profiles"][profile_name]),
        config=config,
    )
    return result


def build_action_trace_report(report_paths: Sequence[Path] = DEFAULT_REPORTS) -> dict[str, Any]:
    if not report_paths:
        raise ValueError("at least one report is required")
    records: list[dict[str, Any]] = []
    for path in report_paths:
        report = _load(path)
        injection_step = int(report["simulator"]["injection_step"])
        for map_name in sorted(report["maps"]):
            item = report["maps"][map_name]
            for profile_name in sorted(item["cases_by_profile"]):
                for case in item["cases_by_profile"][profile_name]:
                    if case.get("split") != "held_out" or case.get("status") != "ok":
                        continue
                    if not case["disturbance"].startswith("boundary_"):
                        continue
                    outcome = _run_case(report, map_name, profile_name, case)
                    trace = _trace_summary(outcome.result, injection_step)
                    records.append(
                        {
                            "report": str(path),
                            "injection_step": injection_step,
                            "map": map_name,
                            "profile": profile_name,
                            "case_key": case["case_key"],
                            "disturbance": case["disturbance"],
                            "historical_metrics": {
                                "recovery_success": case["metrics"]["recovery_success"],
                                "progress_continuity": case["metrics"]["progress_continuity"],
                                "backward_progress_distance_m": case["metrics"][
                                    "backward_progress_distance_m"
                                ],
                            },
                            "trace": trace,
                        }
                    )
    return {
        "scope": "offline_hybrid_recovery_action_trace_diagnostic",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "independent_confirmation": False,
        "controller_defaults_modified": False,
        "records": records,
        "counts": {
            "reports": len(report_paths),
            "records": len(records),
            "backward_progress_records": sum(
                record["historical_metrics"]["progress_continuity"] is False for record in records
            ),
            "full_steering_records": sum(
                record["trace"]["recovery_full_steering_count"] > 0 for record in records
            ),
            "full_brake_records": sum(record["trace"]["recovery_full_brake_count"] > 0 for record in records),
        },
        "interpretation": {
            "historical_metrics_unchanged": True,
            "next_action": "use command traces to scope a recovery-action experiment; do not promote defaults",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_action_trace_report(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": report["counts"]}, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
