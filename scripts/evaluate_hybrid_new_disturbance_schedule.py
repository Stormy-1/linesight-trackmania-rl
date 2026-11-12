"""Evaluate a new held-out disturbance family at a disjoint timing schedule.

This is an offline diagnostic extension of the existing paired mid-rollout
report.  It deliberately keeps the historical common disturbances, recovery
grid, source-anchor split, and diagnostic-only selection rule unchanged while
introducing a new held-out family with speed deficits and opposing yaw/lateral
signs.  The timing points are disjoint from both existing schedules.

No TrackMania, training, checkpoint, live-control, or external-service action is
performed by this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_replay import DisturbanceSpec

try:
    from scripts.compare_hybrid_disturbed_replay import DEFAULT_ANCHOR_PROGRESS_M, DISTURBANCES
    from scripts.compare_hybrid_midrollout_recovery import build_report
    from scripts.evaluate_hybrid_speed_candidates import load_calibration
    from scripts.sweep_hybrid_midrollout_injection_schedule import build_schedule_report
except ModuleNotFoundError:  # Direct script execution puts scripts first on sys.path.
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from scripts.compare_hybrid_disturbed_replay import DEFAULT_ANCHOR_PROGRESS_M, DISTURBANCES
    from scripts.compare_hybrid_midrollout_recovery import build_report
    from scripts.evaluate_hybrid_speed_candidates import load_calibration
    from scripts.sweep_hybrid_midrollout_injection_schedule import build_schedule_report


DEFAULT_OUTPUT_DIR = Path("eval_results/hybrid_rollouts")
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "hybrid_new_disturbance_schedule.json"
DEFAULT_EXISTING_SCHEDULES = (
    DEFAULT_OUTPUT_DIR / "hybrid_midrollout_injection_schedule.json",
    DEFAULT_OUTPUT_DIR / "hybrid_midrollout_heldout_injection_schedule.json",
)
DEFAULT_STEPS = (15, 30, 60)

# This family is intentionally different from the earlier held-out family:
# lateral and heading signs oppose one another, and the speed perturbation is a
# deficit rather than the earlier +12 m/s overspeed.  It is applied only to the
# source-region held-out anchors by build_report's existing split machinery.
NEW_HELD_OUT_DISTURBANCES: tuple[DisturbanceSpec, ...] = (
    DisturbanceSpec(
        "newfamily_lateral_plus_14m_heading_minus_70deg_speed_minus_8mps",
        lateral_offset_m=14.0,
        heading_offset_rad=float(np.deg2rad(-70.0)),
        speed_delta_mps=-8.0,
    ),
    DisturbanceSpec(
        "newfamily_lateral_minus_14m_heading_plus_70deg_speed_minus_8mps",
        lateral_offset_m=-14.0,
        heading_offset_rad=float(np.deg2rad(70.0)),
        speed_delta_mps=-8.0,
    ),
    DisturbanceSpec(
        "newfamily_lateral_plus_8m_heading_minus_120deg_speed_minus_4mps",
        lateral_offset_m=8.0,
        heading_offset_rad=float(np.deg2rad(-120.0)),
        speed_delta_mps=-4.0,
    ),
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"report {path} must contain a JSON object")
    return value


def _steps_from_schedule(path: Path) -> tuple[int, ...]:
    report = _load_object(path)
    schedule = report.get("schedule")
    if not isinstance(schedule, Mapping) and isinstance(report.get("held_out_schedule"), Mapping):
        schedule = report["held_out_schedule"]
    if not isinstance(schedule, Mapping) and isinstance(report.get("schedule_diagnostic"), Mapping):
        schedule = report["schedule_diagnostic"].get("schedule")
    if not isinstance(schedule, Mapping):
        raise TypeError(f"schedule {path} is missing schedule metadata")
    steps = schedule.get("injection_steps")
    if not isinstance(steps, list) or not all(
        isinstance(step, int) and not isinstance(step, bool) and step > 0 for step in steps
    ):
        raise ValueError(f"schedule {path} has invalid injection_steps")
    return tuple(steps)


def _family_payload() -> list[dict[str, Any]]:
    return [asdict(item) for item in NEW_HELD_OUT_DISTURBANCES]


def _validate_family(report_paths: Sequence[Path]) -> list[str]:
    expected = _family_payload()
    issues: list[str] = []
    for path in report_paths:
        report = _load_object(path)
        actual = report.get("held_out_only_disturbances")
        if actual != expected:
            issues.append(f"{path} does not contain the declared new held-out disturbance family")
        if report.get("promotion_allowed") is not False or report.get("independent_confirmation") is not False:
            issues.append(f"{path} is not marked diagnostic-only")
        validation = report.get("validation")
        if not isinstance(validation, Mapping) or validation.get("valid") is not True:
            issues.append(f"{path} is not valid")
    return issues


def build_new_disturbance_schedule_report(
    report_paths: Sequence[Path],
    *,
    existing_schedule_paths: Sequence[Path] = DEFAULT_EXISTING_SCHEDULES,
    expected_steps: Sequence[int] = DEFAULT_STEPS,
) -> dict[str, Any]:
    """Aggregate new-family reports and enforce schedule/family separation."""

    paths = tuple(report_paths)
    if len(paths) < 2:
        raise ValueError("at least two new-family reports are required")
    expected = tuple(expected_steps)
    if not expected or len(set(expected)) != len(expected) or any(
        not isinstance(step, int) or isinstance(step, bool) or step <= 0 for step in expected
    ):
        raise ValueError("expected_steps must contain unique positive integers")

    issues = _validate_family(paths)
    prior_steps: set[int] = set()
    for path in existing_schedule_paths:
        prior_steps.update(_steps_from_schedule(path))

    diagnostic = build_schedule_report(paths)
    actual_steps = tuple(diagnostic["schedule"]["injection_steps"])
    if actual_steps != tuple(sorted(expected)):
        issues.append(f"new-family steps {list(actual_steps)} do not match {sorted(expected)}")
    if prior_steps.intersection(actual_steps):
        issues.append("new-family timing points overlap an existing primary or held-out schedule")
    if diagnostic["validation"].get("valid") is not True:
        issues.append("new-family reports do not share consistent source assignments")

    stable = diagnostic["schedule"].get("stable_eligible_candidates", []) if not issues else []
    recommendation = diagnostic["schedule"].get("diagnostic_recommendation") if not issues else None
    return {
        "scope": "offline_new_held_out_disturbance_family_and_timing_schedule",
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
        "disturbance_family": {
            "family_id": "new_lateral_heading_opposition_with_speed_deficit",
            "held_out_only": True,
            "members": _family_payload(),
            "design_intent": (
                "exercise lateral/heading opposition and underspeed recovery, unlike the earlier "
                "+12 m/s same-sign held-out family"
            ),
        },
        "timing_schedule": {
            "injection_steps": list(actual_steps),
            "injection_times_s": diagnostic["schedule"].get("injection_times_s", []),
            "report_paths": [str(path) for path in paths],
            "prior_schedule_steps": sorted(prior_steps),
            "disjoint_from_prior_schedules": not bool(prior_steps.intersection(actual_steps)),
        },
        "schedule": {
            "stable_eligible_candidates": stable,
            "diagnostic_recommendation": recommendation,
            "candidate_eligibility_rule": diagnostic["schedule"].get("candidate_eligibility_rule"),
            "same_case_assignments_across_reports": diagnostic["schedule"].get(
                "same_case_assignments_across_reports"
            ),
        },
        "validation": {
            "valid": not issues,
            "issues": issues,
            "expected_steps_match": actual_steps == tuple(sorted(expected)),
            "family_matches_all_reports": not any("disturbance family" in issue for issue in issues),
            "disjoint_from_prior_schedules": not bool(prior_steps.intersection(actual_steps)),
        },
        "interpretation": {
            "new_family_is_not_independent_physics_confirmation": True,
            "selection_is_not_promotion": True,
            "next_action": (
                "retain no-promotion and inspect family-specific trajectory failures"
                if recommendation is None
                else "retain as diagnostic only and require untouched confirmation before promotion"
            ),
        },
        "schedule_diagnostic": diagnostic,
    }


def _build_report(
    calibration_report: Mapping[str, Any],
    *,
    injection_step: int,
    max_steps: int,
    search_radius: int,
) -> dict[str, Any]:
    report = build_report(
        calibration_report=calibration_report,
        max_steps=max_steps,
        search_radius=search_radius,
        injection_step=injection_step,
        anchor_progress_m=DEFAULT_ANCHOR_PROGRESS_M,
        disturbances=DISTURBANCES,
        held_out_disturbances=NEW_HELD_OUT_DISTURBANCES,
    )
    report["evaluation"]["new_held_out_disturbance_family"] = True
    report["evaluation"]["family_id"] = "new_lateral_heading_opposition_with_speed_deficit"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("eval_results/hybrid_rollouts/speed_profile_calibration.json"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--existing-schedule", nargs="+", type=Path, default=list(DEFAULT_EXISTING_SCHEDULES))
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--search-radius", type=int, default=64)
    args = parser.parse_args()

    if len(args.steps) < 2 or len(set(args.steps)) != len(args.steps) or any(step <= 0 for step in args.steps):
        raise ValueError("--steps must contain unique positive integers")

    calibration = load_calibration(args.calibration)
    report_paths: list[Path] = []
    for step in sorted(args.steps):
        report = _build_report(
            calibration,
            injection_step=step,
            max_steps=args.max_steps,
            search_radius=args.search_radius,
        )
        report["calibration_input"] = str(args.calibration)
        path = args.output_dir / f"hybrid_new_disturbance_recovery_comparison_step{step:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report_paths.append(path)

    summary = build_new_disturbance_schedule_report(
        report_paths,
        existing_schedule_paths=args.existing_schedule,
        expected_steps=args.steps,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(args.summary),
                "reports": [str(path) for path in report_paths],
                "validation": summary["validation"],
                "schedule": summary["schedule"],
            },
            allow_nan=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
