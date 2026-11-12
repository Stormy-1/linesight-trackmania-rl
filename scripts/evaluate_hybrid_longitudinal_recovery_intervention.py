"""Evaluate bounded full-brake interventions during severe recovery states.

The study reuses the existing boundary reports and deterministic simulator. It
does not change recovery metrics, candidate selection, controller defaults, or
live-control code. The intervention is an isolated wrapper that preserves the
base steering command while replacing the recovery brake with full braking
under explicit heading/along-track triggers.
"""

from __future__ import annotations

import argparse
import json
import math
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
from trackmania_rl.hybrid_replay import DisturbanceSpec, MidRolloutInjection, replay_midrollout_disturbance
from trackmania_rl.hybrid_simulator import ClosedLoopConfig
from trackmania_rl.hybrid_state import build_centerline_geometry

DEFAULT_REPORTS = (
    Path("eval_results/hybrid_rollouts/hybrid_recovery_boundary_comparison_step25.json"),
    Path("eval_results/hybrid_rollouts/hybrid_recovery_boundary_comparison_step70.json"),
)
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_longitudinal_recovery_intervention.json")

BASE_PROFILES = (
    "live_defaults_reference",
    "captured_p25_guardrail",
    "recovery_h_2p2_l_0p12_brake_0p25",
)
INTERVENTIONS = {
    "baseline": {"kind": "none"},
    "heading_90_full_brake": {"kind": "heading", "threshold_rad": math.radians(90.0)},
    "heading_105_full_brake": {"kind": "heading", "threshold_rad": math.radians(105.0)},
    "along_tangent_0_full_brake": {"kind": "along_tangent", "threshold_mps": 0.0},
    "along_tangent_minus_5_full_brake": {"kind": "along_tangent", "threshold_mps": -5.0},
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"report {path} must contain a JSON object")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


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


def _base_controller(profile: Mapping[str, Any]):
    config_values = profile.get("config")
    if not isinstance(config_values, Mapping):
        raise TypeError("profile config must be an object")
    role = str(profile.get("role", ""))
    if role == "recovery_parameter_candidate":
        return RecoveryAwareCenterlineController(RecoveryAwareCenterlineControllerConfig(**config_values))
    return ConservativeCenterlineController(ConservativeControllerConfig(**config_values))


class _LongitudinalInterventionController:
    """Wrap one base controller without changing its steering or classification."""

    def __init__(self, base_controller, intervention: str):
        if intervention not in INTERVENTIONS:
            raise ValueError(f"unknown intervention: {intervention}")
        self.base_controller = base_controller
        self.intervention = intervention

    def command(self, state, geometry) -> ControllerCommand:
        command = self.base_controller.command(state, geometry)
        if not command.recovery or self.intervention == "baseline":
            return command
        policy = INTERVENTIONS[self.intervention]
        kind = policy["kind"]
        triggered = False
        if kind == "heading":
            triggered = abs(float(state.heading_error_rad)) >= float(policy["threshold_rad"])
        elif kind == "along_tangent":
            along_tangent_speed = float(state.speed_mps) * math.cos(float(state.heading_error_rad))
            triggered = along_tangent_speed <= float(policy["threshold_mps"])
        if triggered:
            return replace(command, throttle=0.0, brake=1.0)
        return command


def _case_inputs(report: Mapping[str, Any], profile_name: str) -> list[tuple[str, str, Mapping[str, Any]]]:
    cases: list[tuple[str, str, Mapping[str, Any]]] = []
    for map_name, map_entry in sorted(report["maps"].items()):
        profile_cases = map_entry["cases_by_profile"][profile_name]
        for case in profile_cases:
            if case.get("split") == "held_out" and case.get("status") == "ok":
                cases.append((map_name, str(case["anchor_id"]), case))
    if not cases:
        raise ValueError(f"no held-out cases found for profile {profile_name}")
    return cases


def _run_case(report: Mapping[str, Any], map_name: str, anchor_id: str, case: Mapping[str, Any], profile_name: str, intervention: str):
    map_entry = report["maps"][map_name]
    centerline = np.load(Path(map_entry["centerline_file"]), allow_pickle=False)
    geometry = build_centerline_geometry(centerline)
    anchor = next(anchor for anchor in map_entry["source_anchors"] if anchor["anchor_id"] == anchor_id)
    disturbance = _disturbances(report)[case["disturbance"]]
    simulator = report["simulator"]
    config = ClosedLoopConfig(
        dt_s=float(simulator["dt_s"]),
        max_steps=int(simulator["max_steps"]),
        search_radius=int(simulator["search_radius_segments"]),
    )
    controller = _LongitudinalInterventionController(_base_controller(report["profiles"][profile_name]), intervention)
    outcome = replay_midrollout_disturbance(
        geometry,
        initial_position=np.asarray(anchor["position"], dtype=np.float64),
        initial_heading=np.asarray(anchor["heading"], dtype=np.float64),
        initial_speed_mps=float(anchor["speed_mps"]),
        injection=MidRolloutInjection(int(simulator["injection_step"]), disturbance),
        controller=cast(Any, controller),
        config=config,
    )
    metrics = outcome.metrics["post_rollout"]
    return {
        "map": map_name,
        "anchor_id": anchor_id,
        "case_key": str(case["case_key"]),
        "disturbance": str(case["disturbance"]),
        "profile": profile_name,
        "intervention": intervention,
        "metrics": {
            key: _plain(metrics.get(key))
            for key in (
                "recovery_activated",
                "recovery_completed",
                "recovery_success",
                "stable_rejoin",
                "recovery_latency_s",
                "redivergence_count",
                "progress_continuity",
                "backward_progress_distance_m",
                "progress_complete",
                "progress_fraction",
                "mean_abs_lateral_error_m",
                "max_abs_lateral_error_m",
                "final_abs_lateral_error_m",
                "elapsed_s",
            )
        },
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [record["metrics"] for record in records]
    activated = [item for item in metrics if item["recovery_activated"]]
    stable_latency = [float(item["recovery_latency_s"]) for item in metrics if item["recovery_latency_s"] is not None]
    def rate(key: str, items: Sequence[Mapping[str, Any]] | None = None) -> float | None:
        values = list(items if items is not None else metrics)
        return sum(bool(item[key]) for item in values) / len(values) if values else None
    return {
        "case_count": len(metrics),
        "recovery_activated_count": sum(bool(item["recovery_activated"]) for item in metrics),
        "recovery_success_rate": rate("recovery_success", activated),
        "progress_continuity_rate": rate("progress_continuity"),
        "stable_rejoin_rate": rate("stable_rejoin"),
        "projected_progress_completion_rate": rate("progress_complete"),
        "mean_recovery_latency_s": _mean(stable_latency),
        "total_redivergence_count": sum(int(item["redivergence_count"]) for item in metrics),
        "mean_backward_progress_distance_m": _mean([float(item["backward_progress_distance_m"]) for item in metrics]),
        "max_backward_progress_distance_m": max(
            (float(item["backward_progress_distance_m"]) for item in metrics), default=None
        ),
        "mean_abs_lateral_error_m": _mean([float(item["mean_abs_lateral_error_m"]) for item in metrics]),
        "mean_elapsed_s": _mean([float(item["elapsed_s"]) for item in metrics]),
    }


def _delta_summary(deltas: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    continuity = [int(item["continuity_delta"]) for item in deltas]
    stable = [int(item["stable_rejoin_delta"]) for item in deltas]
    backward = [float(item["backward_distance_delta_m"]) for item in deltas]
    latency = [float(item["recovery_latency_delta_s"]) for item in deltas if item["recovery_latency_delta_s"] is not None]
    return {
        "paired_case_count": len(deltas),
        "continuity_improved_count": sum(value > 0 for value in continuity),
        "continuity_worsened_count": sum(value < 0 for value in continuity),
        "continuity_unchanged_count": sum(value == 0 for value in continuity),
        "stable_rejoin_improved_count": sum(value > 0 for value in stable),
        "stable_rejoin_worsened_count": sum(value < 0 for value in stable),
        "mean_backward_distance_delta_m": _mean(backward),
        "min_backward_distance_delta_m": min(backward) if backward else None,
        "max_backward_distance_delta_m": max(backward) if backward else None,
        "mean_recovery_latency_delta_s": _mean(latency),
    }


def build_intervention_report(report_paths: Sequence[Path] = DEFAULT_REPORTS) -> dict[str, Any]:
    if not report_paths:
        raise ValueError("at least one report is required")
    records: list[dict[str, Any]] = []
    for report_path in report_paths:
        report = _load(report_path)
        injection_step = int(report["simulator"]["injection_step"])
        for profile_name in BASE_PROFILES:
            if profile_name not in report["profiles"]:
                raise ValueError(f"report {report_path} is missing profile {profile_name}")
            for map_name, anchor_id, case in _case_inputs(report, profile_name):
                for intervention in INTERVENTIONS:
                    item = _run_case(report, map_name, anchor_id, case, profile_name, intervention)
                    item["report"] = str(report_path)
                    item["injection_step"] = injection_step
                    records.append(item)

    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["profile"], record["intervention"], int(record["injection_step"]))].append(record)
    aggregates = {
        f"{profile}:{intervention}:step{step}": {
            "profile": profile,
            "intervention": intervention,
            "injection_step": step,
            "metrics": _aggregate(items),
        }
        for (profile, intervention, step), items in sorted(grouped.items())
    }

    baseline_lookup = {
        (record["report"], record["profile"], record["map"], record["case_key"]): record
        for record in records
        if record["intervention"] == "baseline"
    }
    paired_deltas = []
    for record in records:
        if record["intervention"] == "baseline":
            continue
        baseline = baseline_lookup[(record["report"], record["profile"], record["map"], record["case_key"])]
        paired_deltas.append(
            {
                "report": record["report"],
                "injection_step": record["injection_step"],
                "profile": record["profile"],
                "map": record["map"],
                "case_key": record["case_key"],
                "disturbance": record["disturbance"],
                "intervention": record["intervention"],
                "continuity_delta": int(bool(record["metrics"]["progress_continuity"]))
                - int(bool(baseline["metrics"]["progress_continuity"])),
                "stable_rejoin_delta": int(bool(record["metrics"]["stable_rejoin"]))
                - int(bool(baseline["metrics"]["stable_rejoin"])),
                "backward_distance_delta_m": float(record["metrics"]["backward_progress_distance_m"])
                - float(baseline["metrics"]["backward_progress_distance_m"]),
                "recovery_latency_delta_s": (
                    float(record["metrics"]["recovery_latency_s"])
                    - float(baseline["metrics"]["recovery_latency_s"])
                    if record["metrics"]["recovery_latency_s"] is not None
                    and baseline["metrics"]["recovery_latency_s"] is not None
                    else None
                ),
            }
        )

    delta_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    severe_delta_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for delta in paired_deltas:
        delta_groups[str(delta["intervention"])].append(delta)
        if "heading_minus_120" in str(delta["disturbance"]):
            severe_delta_groups[str(delta["intervention"])].append(delta)
    delta_summary = {
        intervention: {
            "all_cases": _delta_summary(items),
            "severe_minus_120_heading": _delta_summary(severe_delta_groups[intervention]),
        }
        for intervention, items in sorted(delta_groups.items())
    }

    return {
        "scope": "offline_hybrid_longitudinal_recovery_intervention",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "independent_confirmation": False,
        "safety_flags": {
            "trackmania_launched": False,
            "training_started": False,
            "live_control_used": False,
            "azure_started": False,
            "checkpoints_loaded": False,
            "external_services_used": False,
            "controller_defaults_modified": False,
        },
        "experiment": {
            "source_reports": [str(path) for path in report_paths],
            "base_profiles": list(BASE_PROFILES),
            "interventions": _plain(INTERVENTIONS),
            "steering_unchanged": True,
            "recovery_classifier_unchanged": True,
            "historical_metrics_unchanged": True,
        },
        "records": records,
        "aggregates": aggregates,
        "paired_deltas_vs_baseline": paired_deltas,
        "intervention_delta_summary": delta_summary,
        "interpretation": {
            "diagnostic_only": True,
            "intervention_scope": "full brake only while the existing recovery command is active",
            "next_action": "inspect whether longitudinal intervention reduces backward motion without stable-rejoin loss; do not promote",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_intervention_report(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "records": len(report["records"]), "aggregate_count": len(report["aggregates"])},
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
