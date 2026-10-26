"""Evaluate early recovery-phase speed shaping in the deterministic simulator.

This diagnostic forces full braking for a bounded number of the first recovery
commands, while preserving the base controller's steering, target speed, and
recovery classification. It is intended to test whether braking earlier in a
recovery trajectory changes the severe boundary outcome. It does not modify
controller defaults, historical metrics, or live-control code.
"""

from __future__ import annotations

import argparse
import json
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

DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_early_recovery_shaping.json")
BASE_PROFILES = (
    "live_defaults_reference",
    "captured_p25_guardrail",
    "recovery_h_2p2_l_0p12_brake_0p25",
)
EARLY_BRAKE_STEPS = (0, 1, 3, 5, 8)


def _base_controller(profile: Mapping[str, Any]):
    config = profile.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("profile config must be an object")
    if profile.get("role") == "recovery_parameter_candidate":
        return RecoveryAwareCenterlineController(RecoveryAwareCenterlineControllerConfig(**config))
    return ConservativeCenterlineController(ConservativeControllerConfig(**config))


class _EarlyBrakeController:
    """Apply full brake to the first N recovery commands only."""

    def __init__(self, base_controller, forced_recovery_steps: int):
        if not isinstance(forced_recovery_steps, int) or forced_recovery_steps < 0:
            raise ValueError("forced_recovery_steps must be a non-negative integer")
        self.base_controller = base_controller
        self.forced_recovery_steps = forced_recovery_steps
        self.recovery_commands_seen = 0
        self.forced_commands = 0

    def command(self, state, geometry) -> ControllerCommand:
        command = self.base_controller.command(state, geometry)
        if not command.recovery:
            return command
        recovery_index = self.recovery_commands_seen
        self.recovery_commands_seen += 1
        if recovery_index < self.forced_recovery_steps:
            self.forced_commands += 1
            return replace(command, throttle=0.0, brake=1.0)
        return command


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
        "mean_forced_brake_command_count": (
            sum(int(record["forced_brake_command_count"]) for record in records) / len(records) if records else None
        ),
    }


def _run_case(report: Mapping[str, Any], map_name: str, profile_name: str, case: Mapping[str, Any], forced_steps: int):
    map_entry = report["maps"][map_name]
    geometry = build_centerline_geometry(np.load(Path(map_entry["centerline_file"]), allow_pickle=False))
    anchor = next(item for item in map_entry["source_anchors"] if item["anchor_id"] == case["anchor_id"])
    simulator = report["simulator"]
    config = ClosedLoopConfig(
        dt_s=float(simulator["dt_s"]),
        max_steps=int(simulator["max_steps"]),
        search_radius=int(simulator["search_radius_segments"]),
    )
    controller = _EarlyBrakeController(_base_controller(report["profiles"][profile_name]), forced_steps)
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
        "forced_brake_command_count": controller.forced_commands,
    }


def build_early_shaping_report(report_paths: Sequence[Path] = DEFAULT_REPORTS) -> dict[str, Any]:
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
                    for forced_steps in EARLY_BRAKE_STEPS:
                        outcome = _run_case(report, map_name, profile_name, case, forced_steps)
                        records.append(
                            {
                                "report": str(report_path),
                                "injection_step": injection_step,
                                "map": map_name,
                                "profile": profile_name,
                                "case_key": case["case_key"],
                                "disturbance": case["disturbance"],
                                "forced_recovery_brake_steps": forced_steps,
                                **outcome,
                            }
                        )
    groups: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["profile"], int(record["injection_step"]), int(record["forced_recovery_brake_steps"]))].append(record)
    aggregates = {
        f"{profile}:step{step}:forced{forced}": {
            "profile": profile,
            "injection_step": step,
            "forced_recovery_brake_steps": forced,
            "metrics": _aggregate(items),
        }
        for (profile, step, forced), items in sorted(groups.items())
    }
    return {
        "scope": "offline_hybrid_early_recovery_shaping",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "independent_confirmation": False,
        "controller_defaults_modified": False,
        "experiment": {
            "source_reports": [str(path) for path in report_paths],
            "base_profiles": list(BASE_PROFILES),
            "forced_recovery_brake_steps": list(EARLY_BRAKE_STEPS),
            "steering_unchanged": True,
            "recovery_classifier_unchanged": True,
            "historical_metrics_unchanged": True,
        },
        "records": records,
        "aggregates": aggregates,
        "interpretation": {
            "diagnostic_only": True,
            "next_action": "inspect whether an early speed margin changes continuity without stable-rejoin loss; do not promote",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_early_shaping_report(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(report["records"])}, separators=(",", ":")))


if __name__ == "__main__":
    main()
