"""Evaluate a state-based one-step continuity viability guard.

The guard predicts the next along-track displacement under maximum braking and
maximum yaw correction toward the tangent. It intervenes only when that proxy
is negative, using the same maximum-authority action. This is a principled
state-trigger diagnostic rather than a fixed threshold or future-disturbance
oracle; it does not change controller defaults or replay metrics.
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


DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_viability_guard.json")
BASE_PROFILES = (
    "live_defaults_reference",
    "captured_p25_guardrail",
    "recovery_h_2p2_l_0p12_brake_0p25",
)
BOUNDARY_PREFIX = "boundary_"
BACKWARD_TOLERANCE_M = 1e-5


def predicted_max_authority_displacement_m(state, config: ClosedLoopConfig) -> float:
    """Estimate one next-step along-track displacement under max authority."""

    speed = float(state.speed_mps)
    heading = abs(float(state.heading_error_rad))
    next_speed = max(
        0.0,
        speed
        - config.max_braking_mps2 * config.dt_s
        - config.linear_drag_mps2_per_mps * speed * config.dt_s,
    )
    next_heading = max(0.0, heading - config.max_yaw_rate_rad_s * config.dt_s)
    return float(next_speed * math.cos(next_heading) * config.dt_s)


def _base_controller(profile: Mapping[str, Any]):
    config = profile.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("profile config must be an object")
    if profile.get("role") == "recovery_parameter_candidate":
        return RecoveryAwareCenterlineController(RecoveryAwareCenterlineControllerConfig(**config))
    return ConservativeCenterlineController(ConservativeControllerConfig(**config))


class _ViabilityGuardController:
    """Apply maximum authority only when the one-step proxy is reverse-aligned."""

    def __init__(self, base_controller, config: ClosedLoopConfig):
        self.base_controller = base_controller
        self.config = config
        self.guard_commands = 0
        self.predicted_displacements: list[float] = []

    def command(self, state, geometry) -> ControllerCommand:
        command = self.base_controller.command(state, geometry)
        predicted = predicted_max_authority_displacement_m(state, self.config)
        self.predicted_displacements.append(predicted)
        if predicted >= -BACKWARD_TOLERANCE_M or abs(float(state.heading_error_rad)) <= 1e-12:
            return command
        self.guard_commands += 1
        steering = -float(np.sign(float(state.heading_error_rad)))
        return replace(command, steering=steering, throttle=0.0, brake=1.0)


def _run_case(report: Mapping[str, Any], map_name: str, profile_name: str, case: Mapping[str, Any]):
    map_entry = report["maps"][map_name]
    geometry = build_centerline_geometry(np.load(Path(map_entry["centerline_file"]), allow_pickle=False))
    anchor = next(item for item in map_entry["source_anchors"] if item["anchor_id"] == case["anchor_id"])
    simulator = report["simulator"]
    config = ClosedLoopConfig(
        dt_s=float(simulator["dt_s"]),
        max_steps=int(simulator["max_steps"]),
        search_radius=int(simulator["search_radius_segments"]),
    )
    injection_step = int(simulator["injection_step"])
    controller = _ViabilityGuardController(_base_controller(report["profiles"][profile_name]), config)
    outcome = replay_midrollout_disturbance(
        geometry,
        initial_position=np.asarray(anchor["position"], dtype=np.float64),
        initial_heading=np.asarray(anchor["heading"], dtype=np.float64),
        initial_speed_mps=float(anchor["speed_mps"]),
        injection=MidRolloutInjection(injection_step, _disturbances(report)[case["disturbance"]]),
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
                "progress_continuity",
                "backward_progress_distance_m",
                "progress_complete",
            )
        },
        "guard_command_count": controller.guard_commands,
        "minimum_predicted_displacement_m": min(controller.predicted_displacements, default=None),
    }


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [record["metrics"] for record in records]

    def rate(key: str) -> float | None:
        return sum(bool(item[key]) for item in metrics) / len(metrics) if metrics else None

    backward = [float(item["backward_progress_distance_m"]) for item in metrics]
    return {
        "case_count": len(metrics),
        "recovery_success_rate": (
            sum(bool(item["recovery_success"]) for item in metrics if item["recovery_activated"])
            / sum(bool(item["recovery_activated"]) for item in metrics)
            if any(item["recovery_activated"] for item in metrics)
            else None
        ),
        "progress_continuity_rate": rate("progress_continuity"),
        "stable_rejoin_rate": rate("stable_rejoin"),
        "projected_progress_completion_rate": rate("progress_complete"),
        "mean_backward_progress_distance_m": sum(backward) / len(backward) if backward else None,
        "max_backward_progress_distance_m": max(backward) if backward else None,
        "mean_guard_command_count": (
            sum(int(record["guard_command_count"]) for record in records) / len(records) if records else None
        ),
        "total_guard_command_count": sum(int(record["guard_command_count"]) for record in records),
    }


def build_viability_guard_report(report_paths: Sequence[Path] = DEFAULT_REPORTS) -> dict[str, Any]:
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
                    if (
                        case.get("split") != "held_out"
                        or case.get("status") != "ok"
                        or not str(case.get("disturbance", "")).startswith(BOUNDARY_PREFIX)
                    ):
                        continue
                    outcome = _run_case(report, map_name, profile_name, case)
                    records.append(
                        {
                            "report": str(report_path),
                            "injection_step": injection_step,
                            "map": map_name,
                            "profile": profile_name,
                            "case_key": case["case_key"],
                            "disturbance": case["disturbance"],
                            **outcome,
                        }
                    )
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["profile"]), int(record["injection_step"]))].append(record)
    aggregates = {
        f"{profile}:step{step}": {
            "profile": profile,
            "injection_step": step,
            "metrics": _aggregate(items),
        }
        for (profile, step), items in sorted(groups.items())
    }
    return {
        "scope": "offline_hybrid_one_step_viability_guard",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "independent_confirmation": False,
        "controller_defaults_modified": False,
        "experiment": {
            "source_reports": [str(path) for path in report_paths],
            "base_profiles": list(BASE_PROFILES),
            "scope": "held-out boundary disturbances only",
            "trigger": "predicted maximum-authority next-step displacement < 0",
            "action": "steering opposite signed heading error, zero throttle, full brake",
            "future_disturbance_information_used": False,
            "historical_metrics_unchanged": True,
        },
        "records": records,
        "aggregates": aggregates,
        "interpretation": {
            "diagnostic_only": True,
            "next_action": "compare against authority-gap envelope; do not promote a reactive guard",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_viability_guard_report(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(report["records"])}, separators=(",", ":")))


if __name__ == "__main__":
    main()
