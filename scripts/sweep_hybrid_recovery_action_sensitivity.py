"""Bounded offline sensitivity sweep for the recovery steering action.

This diagnostic evaluates a small recovery-gain grid against the previously
defined heading-opposition boundary family at steps 25 and 70. It writes only a
compact summary, never changes controller defaults, and never treats a result
as promotion evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from scripts.compare_hybrid_disturbed_replay import DEFAULT_ANCHOR_PROGRESS_M, DISTURBANCES
    from scripts.compare_hybrid_midrollout_recovery import build_report
    from scripts.evaluate_hybrid_recovery_boundary_schedule import BOUNDARY_HELD_OUT_DISTURBANCES
    from scripts.evaluate_hybrid_speed_candidates import load_calibration
except ModuleNotFoundError:  # Direct script execution puts scripts first on sys.path.
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from scripts.compare_hybrid_disturbed_replay import DEFAULT_ANCHOR_PROGRESS_M, DISTURBANCES
    from scripts.compare_hybrid_midrollout_recovery import build_report
    from scripts.evaluate_hybrid_recovery_boundary_schedule import BOUNDARY_HELD_OUT_DISTURBANCES
    from scripts.evaluate_hybrid_speed_candidates import load_calibration


DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_recovery_action_sensitivity.json")
DEFAULT_STEPS = (25, 70)
HEADING_GRID = (0.8, 1.4, 2.2, 3.0)
LATERAL_GRID = (0.06, 0.12, 0.18, 0.24)
BRAKE_GRID = (0.25,)
BRAKE_SLICE = (0.0, 0.25, 0.5, 0.75)


def _candidate_summary(report: Mapping[str, Any], profile: str) -> dict[str, Any]:
    def _finite_or_zero(item: Mapping[str, Any], key: str) -> float:
        value = item.get(key)
        return float(value) if value is not None else 0.0

    aggregate = report["aggregates"][profile]["held_out"]
    severe_parts = [
        item["metrics"]
        for item in report["disturbance_aggregates"]
        if item["profile"] == profile
        and item["split"] == "held_out"
        and item["disturbance"] == "boundary_lateral_plus_8m_heading_minus_120deg_speed_minus_4mps"
    ]
    severe_case_count = sum(int(item.get("case_count", 0)) for item in severe_parts)
    severe_success_observations = sum(
        int(item.get("recovery_success_observation_count", 0)) for item in severe_parts
    )
    severe_successes = sum(int(item.get("recovery_success_count", 0)) for item in severe_parts)
    severe_stable_rejoins = sum(int(item.get("stable_rejoin_count", 0)) for item in severe_parts)
    severe_stable_observations = sum(int(item.get("stable_rejoin_observation_count", 0)) for item in severe_parts)
    severe_aggregate = {
        "case_count": severe_case_count,
        "recovery_success_rate": (
            severe_successes / severe_success_observations if severe_success_observations else None
        ),
        "recovery_failure_count": sum(int(item.get("recovery_failure_count", 0)) for item in severe_parts),
        "progress_continuity_rate": (
            sum(_finite_or_zero(item, "progress_continuity_rate") * int(item.get("case_count", 0)) for item in severe_parts)
            / severe_case_count
            if severe_case_count
            else None
        ),
        "stable_rejoin_rate": severe_stable_rejoins / severe_case_count if severe_case_count else None,
        "mean_recovery_latency_s": (
            sum(_finite_or_zero(item, "mean_recovery_latency_s") * int(item.get("stable_rejoin_observation_count", 0)) for item in severe_parts)
            / severe_stable_observations
            if severe_stable_observations
            else None
        ),
        "total_redivergence_count": sum(int(item.get("total_redivergence_count", 0)) for item in severe_parts),
    }
    return {
        "profile": profile,
        "held_out": {
            key: aggregate.get(key)
            for key in (
                "case_count",
                "recovery_activated_count",
                "recovery_success_rate",
                "recovery_failure_count",
                "progress_continuity_rate",
                "stable_rejoin_rate",
                "mean_recovery_latency_s",
                "total_redivergence_count",
                "projected_progress_completion_rate",
            )
        },
        "severe_minus_120_heading": {
            key: severe_aggregate.get(key)
            for key in (
                "case_count",
                "recovery_success_rate",
                "recovery_failure_count",
                "progress_continuity_rate",
                "stable_rejoin_rate",
                "mean_recovery_latency_s",
                "total_redivergence_count",
            )
        },
    }


def _run_grid(
    calibration: Mapping[str, Any],
    *,
    steps: Sequence[int],
    heading: Sequence[float],
    lateral: Sequence[float],
    brake: Sequence[float],
    max_steps: int,
    search_radius: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for step in steps:
        report = build_report(
            calibration_report=calibration,
            max_steps=max_steps,
            search_radius=search_radius,
            injection_step=step,
            anchor_progress_m=DEFAULT_ANCHOR_PROGRESS_M,
            disturbances=DISTURBANCES,
            held_out_disturbances=BOUNDARY_HELD_OUT_DISTURBANCES,
            recovery_heading_gains=heading,
            recovery_lateral_gains=lateral,
            recovery_brake_biases=brake,
        )
        for candidate in report["candidates"]:
            profile = candidate["configuration_id"]
            records.append(
                {
                    "injection_step": step,
                    "configuration_id": profile,
                    "selection_eligible": profile in report["selection"]["eligible_candidates"],
                    "selection_selected": report["selection"]["selected_candidate"] == profile,
                    "summary": _candidate_summary(report, profile),
                }
            )
    return records


def build_sensitivity_report(
    calibration: Mapping[str, Any],
    *,
    steps: Sequence[int] = DEFAULT_STEPS,
    max_steps: int = 10000,
    search_radius: int = 64,
) -> dict[str, Any]:
    if not steps or len(set(steps)) != len(steps) or any(step < 1 for step in steps):
        raise ValueError("steps must contain unique positive integers")
    heading_records = _run_grid(
        calibration,
        steps=steps,
        heading=HEADING_GRID,
        lateral=LATERAL_GRID,
        brake=BRAKE_GRID,
        max_steps=max_steps,
        search_radius=search_radius,
    )
    brake_records = _run_grid(
        calibration,
        steps=steps,
        heading=(2.2,),
        lateral=(0.12,),
        brake=BRAKE_SLICE,
        max_steps=max_steps,
        search_radius=search_radius,
    )
    return {
        "scope": "offline_hybrid_recovery_action_sensitivity",
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
            "injection_steps": list(steps),
            "boundary_family": [item.name for item in BOUNDARY_HELD_OUT_DISTURBANCES],
            "heading_grid": list(HEADING_GRID),
            "lateral_grid": list(LATERAL_GRID),
            "brake_grid": list(BRAKE_GRID),
            "brake_slice": list(BRAKE_SLICE),
            "selection_rule_unchanged": True,
        },
        "heading_lateral_records": heading_records,
        "brake_records": brake_records,
        "interpretation": {
            "diagnostic_only": True,
            "next_action": "inspect gain trade-offs; do not promote any sensitivity winner",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("eval_results/hybrid_rollouts/speed_profile_calibration.json"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--search-radius", type=int, default=64)
    args = parser.parse_args()
    report = build_sensitivity_report(
        load_calibration(args.calibration),
        steps=args.steps,
        max_steps=args.max_steps,
        search_radius=args.search_radius,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "heading_records": len(report["heading_lateral_records"]), "brake_records": len(report["brake_records"])}, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
