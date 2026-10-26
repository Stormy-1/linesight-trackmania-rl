"""Evaluate state-triggered pre-rotation and speed-margin shaping offline.

The diagnostic activates a bounded coordinated action when both lateral and
heading error indicate an emerging recovery risk. It preserves the base
controller's correction direction, but applies full steering in that direction,
zero throttle, and full braking. The trigger is intentionally evaluated
independently of the historical recovery classifier so the experiment can test
earlier intervention. No controller defaults, historical metrics, or live code
are changed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineController,
    ConservativeControllerConfig,
    ControllerCommand,
    RecoveryAwareCenterlineController,
    RecoveryAwareCenterlineControllerConfig,
)
from trackmania_rl.hybrid_replay import MidRolloutInjection, replay_midrollout_disturbance
from trackmania_rl.hybrid_simulator import ClosedLoopConfig
from trackmania_rl.hybrid_state import build_centerline_geometry

try:
    from scripts.diagnose_hybrid_recovery_action_trace import DEFAULT_REPORTS, _disturbances, _load
except ModuleNotFoundError:  # Direct script execution puts scripts first on sys.path.
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from scripts.diagnose_hybrid_recovery_action_trace import DEFAULT_REPORTS, _disturbances, _load


DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_state_triggered_recovery.json")
BASE_PROFILES = (
    "live_defaults_reference",
    "captured_p25_guardrail",
    "recovery_h_2p2_l_0p12_brake_0p25",
)
TRIGGER_GRID = (
    (6.0, math.radians(35.0)),
    (6.0, math.radians(45.0)),
    (6.0, math.radians(55.0)),
    (8.0, math.radians(45.0)),
)
TRIGGER_BURST_STEPS = 3


def _trigger_name(lateral_limit_m: float, heading_limit_rad: float) -> str:
    return f"lateral_{lateral_limit_m:g}m_heading_{math.degrees(heading_limit_rad):g}deg"


def _base_controller(profile: Mapping[str, Any]):
    config = profile.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("profile config must be an object")
    if profile.get("role") == "recovery_parameter_candidate":
        return RecoveryAwareCenterlineController(RecoveryAwareCenterlineControllerConfig(**config))
    return ConservativeCenterlineController(ConservativeControllerConfig(**config))


class _StateTriggeredRecoveryController:
    """Apply coordinated authority before the historical recovery flag."""

    def __init__(self, base_controller, lateral_limit_m: float, heading_limit_rad: float, burst_steps: int = TRIGGER_BURST_STEPS):
        if lateral_limit_m <= 0.0 or heading_limit_rad <= 0.0:
            raise ValueError("trigger limits must be positive")
        if not isinstance(burst_steps, int) or burst_steps <= 0:
            raise ValueError("burst_steps must be a positive integer")
        self.base_controller = base_controller
        self.lateral_limit_m = lateral_limit_m
        self.heading_limit_rad = heading_limit_rad
        self.burst_steps = burst_steps
        self.triggered_commands = 0
        self.triggered_before_recovery = 0
        self.trigger_events = 0
        self._burst_remaining = 0
        self._risk_latched = False

    def command(self, state, geometry) -> ControllerCommand:
        command = self.base_controller.command(state, geometry)
        triggered = (
            abs(float(state.lateral_error_m)) >= self.lateral_limit_m
            and abs(float(state.heading_error_rad)) >= self.heading_limit_rad
            and abs(command.steering) > 1e-12
        )
        if not triggered:
            self._risk_latched = False
        if triggered and not self._risk_latched and self._burst_remaining == 0:
            self._risk_latched = True
            self._burst_remaining = self.burst_steps
            self.trigger_events += 1
        if self._burst_remaining == 0:
            return command
        self._burst_remaining -= 1
        self.triggered_commands += 1
        if not command.recovery:
            self.triggered_before_recovery += 1
        return replace(command, steering=float(np.sign(command.steering)), throttle=0.0, brake=1.0)


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [record["metrics"] for record in records]

    def rate(key: str, values: Sequence[Mapping[str, Any]] | None = None) -> float | None:
        items = list(values if values is not None else metrics)
        return sum(bool(item[key]) for item in items) / len(items) if items else None

    latency = [float(item["recovery_latency_s"]) for item in metrics if item["recovery_latency_s"] is not None]
    backward = [float(item["backward_progress_distance_m"]) for item in metrics]
    return {
        "case_count": len(metrics),
        "recovery_success_rate": rate("recovery_success", [item for item in metrics if item["recovery_activated"]]),
        "progress_continuity_rate": rate("progress_continuity"),
        "stable_rejoin_rate": rate("stable_rejoin"),
        "projected_progress_completion_rate": rate("progress_complete"),
        "mean_recovery_latency_s": sum(latency) / len(latency) if latency else None,
        "mean_backward_progress_distance_m": sum(backward) / len(backward) if backward else None,
        "max_backward_progress_distance_m": max(backward) if backward else None,
        "total_redivergence_count": sum(int(item["redivergence_count"]) for item in metrics),
        "mean_triggered_command_count": (
            sum(int(record["triggered_command_count"]) for record in records) / len(records) if records else None
        ),
        "mean_trigger_event_count": (
            sum(int(record["trigger_event_count"]) for record in records) / len(records) if records else None
        ),
        "mean_triggered_before_recovery_count": (
            sum(int(record["triggered_before_recovery_count"]) for record in records) / len(records)
            if records
            else None
        ),
    }


def _run_case(
    report: Mapping[str, Any],
    map_name: str,
    profile_name: str,
    case: Mapping[str, Any],
    lateral_limit_m: float,
    heading_limit_rad: float,
):
    map_entry = report["maps"][map_name]
    geometry = build_centerline_geometry(np.load(Path(map_entry["centerline_file"]), allow_pickle=False))
    anchor = next(item for item in map_entry["source_anchors"] if item["anchor_id"] == case["anchor_id"])
    simulator = report["simulator"]
    config = ClosedLoopConfig(
        dt_s=float(simulator["dt_s"]),
        max_steps=int(simulator["max_steps"]),
        search_radius=int(simulator["search_radius_segments"]),
    )
    controller = _StateTriggeredRecoveryController(
        _base_controller(report["profiles"][profile_name]), lateral_limit_m, heading_limit_rad
    )
    outcome = replay_midrollout_disturbance(
        geometry,
        initial_position=np.asarray(anchor["position"], dtype=np.float64),
        initial_heading=np.asarray(anchor["heading"], dtype=np.float64),
        initial_speed_mps=float(anchor["speed_mps"]),
        injection=MidRolloutInjection(int(simulator["injection_step"]), _disturbances(report)[case["disturbance"]]),
        controller=cast(Any, controller),
        config=config,
    )
    metrics = outcome.metrics["post_rollout"]
    return {
        "metrics": {
            key: metrics.get(key)
            for key in (
                "recovery_activated",
                "recovery_success",
                "recovery_latency_s",
                "stable_rejoin",
                "redivergence_count",
                "progress_continuity",
                "backward_progress_distance_m",
                "progress_complete",
            )
        },
        "triggered_command_count": controller.triggered_commands,
        "trigger_event_count": controller.trigger_events,
        "triggered_before_recovery_count": controller.triggered_before_recovery,
    }


def build_state_triggered_report(report_paths: Sequence[Path] = DEFAULT_REPORTS) -> dict[str, Any]:
    if not report_paths:
        raise ValueError("at least one report is required")
    records: list[dict[str, Any]] = []
    for report_path in report_paths:
        report = _load(report_path)
        injection_step = int(report["simulator"]["injection_step"])
        for profile_name in BASE_PROFILES:
            if profile_name not in report["profiles"]:
                raise ValueError(f"missing profile {profile_name}")
            for map_name, map_entry in sorted(report["maps"].items()):
                for case in map_entry["cases_by_profile"][profile_name]:
                    if case.get("split") != "held_out" or case.get("status") != "ok":
                        continue
                    for lateral_limit_m, heading_limit_rad in TRIGGER_GRID:
                        outcome = _run_case(
                            report,
                            map_name,
                            profile_name,
                            case,
                            lateral_limit_m,
                            heading_limit_rad,
                        )
                        records.append(
                            {
                                "report": str(report_path),
                                "injection_step": injection_step,
                                "map": map_name,
                                "profile": profile_name,
                                "case_key": case["case_key"],
                                "disturbance": case["disturbance"],
                                "trigger": _trigger_name(lateral_limit_m, heading_limit_rad),
                                "lateral_trigger_limit_m": lateral_limit_m,
                                "heading_trigger_limit_rad": heading_limit_rad,
                                **outcome,
                            }
                        )
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["profile"], int(record["injection_step"]), str(record["trigger"]))].append(record)
    aggregates = {
        f"{profile}:step{step}:{trigger}": {
            "profile": profile,
            "injection_step": step,
            "trigger": trigger,
            "metrics": _aggregate(items),
        }
        for (profile, step, trigger), items in sorted(groups.items())
    }
    return {
        "scope": "offline_hybrid_state_triggered_recovery",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "independent_confirmation": False,
        "controller_defaults_modified": False,
        "experiment": {
            "source_reports": [str(path) for path in report_paths],
            "base_profiles": list(BASE_PROFILES),
            "trigger_grid": [
                {"lateral_limit_m": lateral, "heading_limit_rad": heading, "name": _trigger_name(lateral, heading)}
                for lateral, heading in TRIGGER_GRID
            ],
            "trigger_burst_steps": TRIGGER_BURST_STEPS,
            "steering_policy": "preserve base correction sign, force magnitude 1.0",
            "speed_policy": "zero throttle and force brake 1.0",
            "historical_recovery_classifier_unchanged": True,
            "historical_metrics_unchanged": True,
        },
        "records": records,
        "aggregates": aggregates,
        "interpretation": {
            "diagnostic_only": True,
            "next_action": "inspect state-triggered timing trade-offs; do not promote any trigger",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_state_triggered_report(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(report["records"])}, separators=(",", ":")))


if __name__ == "__main__":
    main()
