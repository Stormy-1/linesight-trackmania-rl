"""Validate a separately held-out mid-rollout timing schedule.

The historical step-5/10/20 schedule remains the pre-registered development
requirement.  This wrapper validates a second schedule whose injection steps
are disjoint from that requirement, then reuses the existing schedule
aggregator to test candidate stability across the new timing points.  It does
not rerun the simulator and it never promotes live defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from scripts.sweep_hybrid_midrollout_injection_schedule import build_schedule_report
except ModuleNotFoundError:  # Direct script execution puts scripts first on sys.path.
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from scripts.sweep_hybrid_midrollout_injection_schedule import build_schedule_report

DEFAULT_PRIMARY_SCHEDULE = Path("eval_results/hybrid_rollouts/hybrid_midrollout_injection_schedule.json")
DEFAULT_HELD_OUT_REPORTS = (
    Path("eval_results/hybrid_rollouts/hybrid_midrollout_recovery_comparison_step35.json"),
    Path("eval_results/hybrid_rollouts/hybrid_midrollout_recovery_comparison_step50.json"),
    Path("eval_results/hybrid_rollouts/hybrid_midrollout_recovery_comparison_step80.json"),
)
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_midrollout_heldout_injection_schedule.json")
DEFAULT_EXPECTED_STEPS = (35, 50, 80)


def _load_object(path: Path) -> dict[str, Any]:
    if not isinstance(path, Path):
        raise TypeError("schedule paths must be Path values")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"schedule {path} must contain a JSON object")
    return value


def _schedule_steps(schedule: dict[str, Any], path: Path) -> tuple[int, ...]:
    section = schedule.get("schedule")
    if not isinstance(section, dict):
        raise TypeError(f"schedule {path} is missing schedule metadata")
    steps = section.get("injection_steps")
    if not isinstance(steps, list) or not all(isinstance(step, int) and not isinstance(step, bool) for step in steps):
        raise ValueError(f"schedule {path} has invalid injection_steps")
    if len(set(steps)) != len(steps):
        raise ValueError(f"schedule {path} has duplicate injection steps")
    return tuple(steps)


def _validate_primary_schedule(schedule: dict[str, Any], path: Path) -> tuple[int, ...]:
    if schedule.get("scope") != "offline_fixed_midrollout_injection_schedule_stability":
        raise ValueError(f"schedule {path} is not the fixed primary schedule")
    if schedule.get("promotion_allowed") is not False or schedule.get("independent_confirmation") is not False:
        raise ValueError(f"schedule {path} is not marked diagnostic-only")
    validation = schedule.get("validation")
    if not isinstance(validation, dict) or validation.get("valid") is not True:
        raise ValueError(f"schedule {path} is not valid")
    return _schedule_steps(schedule, path)


def build_held_out_schedule_report(
    held_out_report_paths: Sequence[Path] = DEFAULT_HELD_OUT_REPORTS,
    *,
    primary_schedule_path: Path = DEFAULT_PRIMARY_SCHEDULE,
    expected_steps: Sequence[int] = DEFAULT_EXPECTED_STEPS,
) -> dict[str, Any]:
    """Build a report for a timing schedule disjoint from the primary schedule."""

    held_out_paths = tuple(held_out_report_paths)
    if len(held_out_paths) < 2:
        raise ValueError("at least two held-out injection reports are required")
    expected = tuple(expected_steps)
    if not expected or len(set(expected)) != len(expected) or any(
        not isinstance(step, int) or isinstance(step, bool) or step < 1 for step in expected
    ):
        raise ValueError("expected_steps must contain unique positive integers")

    primary = _load_object(primary_schedule_path)
    primary_steps = _validate_primary_schedule(primary, primary_schedule_path)
    held_out = build_schedule_report(held_out_paths)
    held_out_steps = tuple(held_out["schedule"]["injection_steps"])

    issues: list[str] = []
    if held_out_steps != tuple(sorted(expected)):
        issues.append(
            f"held-out steps {list(held_out_steps)} do not match declared schedule {sorted(expected)}"
        )
    if not set(primary_steps).isdisjoint(held_out_steps):
        issues.append("held-out injection steps overlap the pre-registered primary schedule")
    if held_out["validation"].get("valid") is not True:
        issues.append("held-out schedule reports do not share consistent source assignments")

    stable_candidates = held_out["schedule"].get("stable_eligible_candidates", [])
    recommendation = held_out["schedule"].get("diagnostic_recommendation")
    if issues:
        stable_candidates = []
        recommendation = None

    return {
        "scope": "offline_separately_held_out_midrollout_injection_schedule",
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
        "primary_schedule": {
            "path": str(primary_schedule_path),
            "injection_steps": list(primary_steps),
            "schedule_role": "pre_registered_development_requirement",
        },
        "held_out_schedule": {
            "schedule_role": "separately_held_out_timing_schedule",
            "injection_steps": list(held_out_steps),
            "injection_times_s": held_out["schedule"].get("injection_times_s", []),
            "report_paths": [str(path) for path in held_out_paths],
            "same_case_assignments_across_reports": held_out["schedule"].get(
                "same_case_assignments_across_reports"
            ),
            "candidate_eligibility_rule": held_out["schedule"].get("candidate_eligibility_rule"),
            "stable_eligible_candidates": stable_candidates,
            "diagnostic_recommendation": recommendation,
        },
        "validation": {
            "valid": not issues,
            "issues": issues,
            "primary_and_held_out_steps_disjoint": set(primary_steps).isdisjoint(held_out_steps),
            "held_out_steps_match_declared_schedule": held_out_steps == tuple(sorted(expected)),
        },
        "interpretation": {
            "timing_holdout_is_not_independent_physics_confirmation": True,
            "selection_is_not_promotion": True,
            "next_action": (
                "retain the no-promotion decision and investigate failure timing/trajectory semantics"
                if recommendation is None
                else "retain as diagnostic only and require untouched confirmation before promotion"
            ),
        },
        "schedule_diagnostic": held_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-schedule", type=Path, default=DEFAULT_PRIMARY_SCHEDULE)
    parser.add_argument("--reports", nargs="+", type=Path, default=list(DEFAULT_HELD_OUT_REPORTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-steps", nargs="+", type=int, default=list(DEFAULT_EXPECTED_STEPS))
    args = parser.parse_args()

    report = build_held_out_schedule_report(
        args.reports,
        primary_schedule_path=args.primary_schedule,
        expected_steps=args.expected_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "validation": report["validation"],
                "held_out_schedule": report["held_out_schedule"],
            },
            allow_nan=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
