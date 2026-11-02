"""Measure an oracle upper bound for pre-disturbance heading shaping.

The wrapper knows the heading direction of the deliberately injected future
disturbance and steers against it for a bounded number of commands before the
injection. This is intentionally an oracle diagnostic: it answers whether
preserving continuity is physically reachable with lead time, without claiming
that the future disturbance is observable or that the policy is deployable.
It leaves controller defaults, replay metrics, and live-control code unchanged.
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


DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_predictive_prerotation.json")
BASE_PROFILES = (
    "live_defaults_reference",
    "captured_p25_guardrail",
    "recovery_h_2p2_l_0p12_brake_0p25",
)
LEAD_STEPS = (0, 1, 2, 3, 4, 6)
STEERING_LEVELS = (0.25, 0.5, 1.0)
BRAKE_LEVELS = (0.0, 1.0)
BOUNDARY_PREFIX = "boundary_"


def _base_controller(profile: Mapping[str, Any]):
    config = profile.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("profile config must be an object")
    if profile.get("role") == "recovery_parameter_candidate":
        return RecoveryAwareCenterlineController(RecoveryAwareCenterlineControllerConfig(**config))
    return ConservativeCenterlineController(ConservativeControllerConfig(**config))


class _OraclePreRotationController:
    """Apply known-future-opposition steering before one scheduled injection."""

    def __init__(
        self,
        base_controller,
        *,
        injection_step: int,
        heading_offset_rad: float,
        lead_steps: int,
        steering_level: float,
        brake_level: float,
    ):
        values = (heading_offset_rad, steering_level, brake_level)
        if not isinstance(injection_step, int) or injection_step < 0:
            raise ValueError("injection_step must be a non-negative integer")
        if not isinstance(lead_steps, int) or lead_steps < 0:
            raise ValueError("lead_steps must be a non-negative integer")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("oracle pre-rotation values must be finite")
        if not 0.0 <= steering_level <= 1.0 or not 0.0 <= brake_level <= 1.0:
            raise ValueError("steering_level and brake_level must be in [0, 1]")
        if abs(heading_offset_rad) <= 1e-12 and lead_steps > 0:
            raise ValueError("heading_offset_rad must be non-zero for pre-rotation")
        self.base_controller = base_controller
        self.injection_step = injection_step
        self.heading_offset_rad = heading_offset_rad
        self.lead_steps = lead_steps
        self.steering_level = steering_level
        self.brake_level = brake_level
        self.commands_seen = 0
        self.pre_rotation_commands = 0

    def command(self, state, geometry) -> ControllerCommand:
        command = self.base_controller.command(state, geometry)
        step_index = self.commands_seen
        self.commands_seen += 1
        start_step = max(0, self.injection_step - self.lead_steps)
        if not start_step <= step_index < self.injection_step:
            return command
        # A negative future heading offset needs positive yaw beforehand, and
        # vice versa. The magnitude is intentionally swept, not promoted.
        steering = -float(np.sign(self.heading_offset_rad)) * self.steering_level
        self.pre_rotation_commands += 1
        return replace(command, steering=steering, throttle=0.0, brake=self.brake_level)


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
        "mean_pre_rotation_command_count": (
            sum(int(record["pre_rotation_command_count"]) for record in records) / len(records) if records else None
        ),
    }


def _run_case(
    report: Mapping[str, Any],
    map_name: str,
    profile_name: str,
    case: Mapping[str, Any],
    lead_steps: int,
    steering_level: float,
    brake_level: float,
):
    map_entry = report["maps"][map_name]
    geometry = build_centerline_geometry(np.load(Path(map_entry["centerline_file"]), allow_pickle=False))
    anchor = next(item for item in map_entry["source_anchors"] if item["anchor_id"] == case["anchor_id"])
    simulator = report["simulator"]
    injection_step = int(simulator["injection_step"])
    disturbances = _disturbances(report)
    disturbance = disturbances[case["disturbance"]]
    controller = _OraclePreRotationController(
        _base_controller(report["profiles"][profile_name]),
        injection_step=injection_step,
        heading_offset_rad=disturbance.heading_offset_rad,
        lead_steps=lead_steps,
        steering_level=steering_level,
        brake_level=brake_level,
    )
    config = ClosedLoopConfig(
        dt_s=float(simulator["dt_s"]),
        max_steps=int(simulator["max_steps"]),
        search_radius=int(simulator["search_radius_segments"]),
    )
    outcome = replay_midrollout_disturbance(
        geometry,
        initial_position=np.asarray(anchor["position"], dtype=np.float64),
        initial_heading=np.asarray(anchor["heading"], dtype=np.float64),
        initial_speed_mps=float(anchor["speed_mps"]),
        injection=MidRolloutInjection(injection_step, disturbance),
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
        "pre_rotation_command_count": controller.pre_rotation_commands,
        "pre_injection_state": outcome.metrics["pre_injection"],
        "post_injection_state": outcome.metrics["post_injection"],
    }


def build_predictive_prerotation_report(report_paths: Sequence[Path] = DEFAULT_REPORTS) -> dict[str, Any]:
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
                    for lead_steps in LEAD_STEPS:
                        for steering_level in STEERING_LEVELS:
                            for brake_level in BRAKE_LEVELS:
                                outcome = _run_case(
                                    report,
                                    map_name,
                                    profile_name,
                                    case,
                                    lead_steps,
                                    steering_level,
                                    brake_level,
                                )
                                records.append(
                                    {
                                        "report": str(report_path),
                                        "injection_step": injection_step,
                                        "map": map_name,
                                        "profile": profile_name,
                                        "case_key": case["case_key"],
                                        "disturbance": case["disturbance"],
                                        "lead_steps": lead_steps,
                                        "steering_level": steering_level,
                                        "brake_level": brake_level,
                                        "oracle_future_disturbance_known": True,
                                        "deployable_policy": False,
                                        **outcome,
                                    }
                                )
    groups: dict[tuple[str, int, int, float, float], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[
            (
                str(record["profile"]),
                int(record["injection_step"]),
                int(record["lead_steps"]),
                float(record["steering_level"]),
                float(record["brake_level"]),
            )
        ].append(record)
    aggregates = {
        f"{profile}:step{step}:lead{lead}:steer{steering:g}:brake{brake:g}": {
            "profile": profile,
            "injection_step": step,
            "lead_steps": lead,
            "steering_level": steering,
            "brake_level": brake,
            "metrics": _aggregate(items),
        }
        for (profile, step, lead, steering, brake), items in sorted(groups.items())
    }
    return {
        "scope": "offline_hybrid_predictive_prerotation_oracle",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "independent_confirmation": False,
        "controller_defaults_modified": False,
        "experiment": {
            "source_reports": [str(path) for path in report_paths],
            "base_profiles": list(BASE_PROFILES),
            "lead_steps": list(LEAD_STEPS),
            "steering_levels": list(STEERING_LEVELS),
            "brake_levels": list(BRAKE_LEVELS),
            "scope": "held-out boundary disturbances only",
            "future_disturbance_direction_used": True,
            "post_injection_controller_unchanged": True,
            "historical_metrics_unchanged": True,
        },
        "records": records,
        "aggregates": aggregates,
        "interpretation": {
            "diagnostic_only": True,
            "oracle_upper_bound": True,
            "next_action": "use only continuity feasibility and cost envelope; do not promote the oracle policy",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_predictive_prerotation_report(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(report["records"])}, separators=(",", ":")))


if __name__ == "__main__":
    main()
