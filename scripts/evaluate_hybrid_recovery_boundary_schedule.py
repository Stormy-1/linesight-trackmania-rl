"""Offline boundary study for a new held-out recovery disturbance family.

The study varies only heading opposition for a fixed +8 m lateral offset and
-4 m/s speed deficit.  It is a diagnostic boundary experiment, not a new
controller default or a promotion path.  The timing points 25 and 70 are
disjoint from every earlier schedule in this workspace.
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
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "hybrid_recovery_boundary_schedule.json"
DEFAULT_EXISTING_SCHEDULES = (
    DEFAULT_OUTPUT_DIR / "hybrid_midrollout_injection_schedule.json",
    DEFAULT_OUTPUT_DIR / "hybrid_midrollout_heldout_injection_schedule.json",
    DEFAULT_OUTPUT_DIR / "hybrid_new_disturbance_schedule.json",
)
DEFAULT_STEPS = (25, 70)

BOUNDARY_HELD_OUT_DISTURBANCES: tuple[DisturbanceSpec, ...] = tuple(
    DisturbanceSpec(
        f"boundary_lateral_plus_8m_heading_minus_{angle:g}deg_speed_minus_4mps",
        lateral_offset_m=8.0,
        heading_offset_rad=float(np.deg2rad(-angle)),
        speed_delta_mps=-4.0,
    )
    for angle in (70.0, 90.0, 105.0, 120.0)
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"report {path} must contain a JSON object")
    return value


def _schedule_steps(path: Path) -> tuple[int, ...]:
    report = _load_object(path)
    schedule = report.get("schedule")
    if (not isinstance(schedule, Mapping) or "injection_steps" not in schedule) and isinstance(
        report.get("held_out_schedule"), Mapping
    ):
        schedule = report["held_out_schedule"]
    if (not isinstance(schedule, Mapping) or "injection_steps" not in schedule) and isinstance(
        report.get("schedule_diagnostic"), Mapping
    ):
        schedule = report["schedule_diagnostic"].get("schedule")
    if not isinstance(schedule, Mapping):
        raise TypeError(f"schedule {path} is missing schedule metadata")
    steps = schedule.get("injection_steps")
    if not isinstance(steps, list) or not all(isinstance(step, int) and not isinstance(step, bool) for step in steps):
        raise ValueError(f"schedule {path} has invalid injection_steps")
    return tuple(steps)


def _family_payload() -> list[dict[str, Any]]:
    return [asdict(item) for item in BOUNDARY_HELD_OUT_DISTURBANCES]


def build_boundary_schedule_report(
    report_paths: Sequence[Path],
    *,
    existing_schedule_paths: Sequence[Path] = DEFAULT_EXISTING_SCHEDULES,
    expected_steps: Sequence[int] = DEFAULT_STEPS,
) -> dict[str, Any]:
    paths = tuple(report_paths)
    if len(paths) < 2:
        raise ValueError("at least two boundary reports are required")
    expected = tuple(expected_steps)
    if not expected or len(set(expected)) != len(expected) or any(
        not isinstance(step, int) or isinstance(step, bool) or step <= 0 for step in expected
    ):
        raise ValueError("expected_steps must contain unique positive integers")

    issues: list[str] = []
    for path in paths:
        report = _load_object(path)
        if report.get("held_out_only_disturbances") != _family_payload():
            issues.append(f"{path} does not contain the declared boundary family")
        if report.get("promotion_allowed") is not False or report.get("independent_confirmation") is not False:
            issues.append(f"{path} is not diagnostic-only")

    prior_steps = {step for path in existing_schedule_paths for step in _schedule_steps(path)}
    diagnostic = build_schedule_report(paths)
    actual_steps = tuple(diagnostic["schedule"]["injection_steps"])
    if actual_steps != tuple(sorted(expected)):
        issues.append(f"boundary steps {list(actual_steps)} do not match {sorted(expected)}")
    if prior_steps.intersection(actual_steps):
        issues.append("boundary timing points overlap an existing schedule")
    if diagnostic["validation"].get("valid") is not True:
        issues.append("boundary reports do not share consistent source assignments")

    valid = not issues
    return {
        "scope": "offline_hybrid_recovery_boundary_schedule",
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
        "boundary_family": {
            "family_id": "fixed_lateral_speed_deficit_heading_opposition_boundary",
            "fixed_lateral_offset_m": 8.0,
            "fixed_speed_delta_mps": -4.0,
            "heading_magnitudes_deg": [70.0, 90.0, 105.0, 120.0],
            "members": _family_payload(),
        },
        "timing_schedule": {
            "injection_steps": list(actual_steps),
            "injection_times_s": diagnostic["schedule"].get("injection_times_s", []),
            "prior_schedule_steps": sorted(prior_steps),
            "disjoint_from_prior_schedules": not bool(prior_steps.intersection(actual_steps)),
            "report_paths": [str(path) for path in paths],
        },
        "validation": {
            "valid": valid,
            "issues": issues,
            "family_matches_all_reports": not any("boundary family" in issue for issue in issues),
            "expected_steps_match": actual_steps == tuple(sorted(expected)),
            "disjoint_from_prior_schedules": not bool(prior_steps.intersection(actual_steps)),
        },
        "schedule": {
            "stable_eligible_candidates": diagnostic["schedule"].get("stable_eligible_candidates", []) if valid else [],
            "diagnostic_recommendation": diagnostic["schedule"].get("diagnostic_recommendation") if valid else None,
            "same_case_assignments_across_reports": diagnostic["schedule"].get(
                "same_case_assignments_across_reports"
            ),
        },
        "interpretation": {
            "boundary_study_is_not_independent_physics_confirmation": True,
            "selection_is_not_promotion": True,
            "next_action": "use the angle boundary only to scope a future recovery-action experiment",
        },
        "schedule_diagnostic": diagnostic,
    }


def _build_report(calibration_report: Mapping[str, Any], *, injection_step: int, max_steps: int, search_radius: int):
    report = build_report(
        calibration_report=calibration_report,
        max_steps=max_steps,
        search_radius=search_radius,
        injection_step=injection_step,
        anchor_progress_m=DEFAULT_ANCHOR_PROGRESS_M,
        disturbances=DISTURBANCES,
        held_out_disturbances=BOUNDARY_HELD_OUT_DISTURBANCES,
    )
    report["evaluation"]["boundary_family"] = True
    report["evaluation"]["family_id"] = "fixed_lateral_speed_deficit_heading_opposition_boundary"
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
    calibration = load_calibration(args.calibration)
    report_paths: list[Path] = []
    for step in sorted(args.steps):
        report = _build_report(
            calibration, injection_step=step, max_steps=args.max_steps, search_radius=args.search_radius
        )
        report["calibration_input"] = str(args.calibration)
        path = args.output_dir / f"hybrid_recovery_boundary_comparison_step{step:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report_paths.append(path)
    summary = build_boundary_schedule_report(
        report_paths, existing_schedule_paths=args.existing_schedule, expected_steps=args.steps
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"summary": str(args.summary), "reports": [str(path) for path in report_paths], "validation": summary["validation"], "schedule": summary["schedule"]},
            allow_nan=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
