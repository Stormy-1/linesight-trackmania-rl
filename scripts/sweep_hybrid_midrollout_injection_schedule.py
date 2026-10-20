"""Combine fixed mid-rollout reports into a conservative schedule diagnostic.

This script does not rerun the simulator.  It checks a pre-declared set of
already-generated injection-step reports and asks whether one recovery
configuration is eligible at every schedule point under the historical
train-ranking/held-out-acceptance rule.  The result is diagnostic only and
cannot promote live defaults.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_REPORTS = (
    Path("eval_results/hybrid_rollouts/hybrid_midrollout_recovery_comparison_step05.json"),
    Path("eval_results/hybrid_rollouts/hybrid_midrollout_recovery_comparison.json"),
    Path("eval_results/hybrid_rollouts/hybrid_midrollout_recovery_comparison_step20.json"),
)
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_midrollout_injection_schedule.json")


def _finite(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _load_report(path: Path) -> dict[str, Any]:
    if not isinstance(path, Path):
        raise TypeError("report paths must be Path values")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"report {path} must contain a JSON object")
    return report


def _case_assignments(report: Mapping[str, Any]) -> Mapping[str, Any]:
    split = report.get("split")
    if not isinstance(split, Mapping) or not isinstance(split.get("assignments"), Mapping):
        raise ValueError("report is missing split.assignments")
    return split["assignments"]


def _report_summary(path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    validation = report.get("validation")
    simulator = report.get("simulator")
    selection = report.get("selection")
    aggregates = report.get("aggregates")
    if not isinstance(validation, Mapping):
        raise ValueError(f"report {path} is missing validation")
    if not isinstance(simulator, Mapping):
        raise ValueError(f"report {path} is missing simulator")
    if not isinstance(selection, Mapping):
        raise ValueError(f"report {path} is missing selection")
    if not isinstance(aggregates, Mapping):
        raise ValueError(f"report {path} is missing validation, simulator, selection, or aggregates")
    if validation.get("valid") is not True:
        raise ValueError(f"report {path} is not valid: {validation.get('issues', [])}")
    if report.get("promotion_allowed") is not False or report.get("independent_confirmation") is not False:
        raise ValueError(f"report {path} is not marked diagnostic-only")
    injection_step = simulator.get("injection_step")
    if not isinstance(injection_step, int) or isinstance(injection_step, bool) or injection_step < 1:
        raise ValueError(f"report {path} has an invalid injection step")
    eligible = selection.get("eligible_candidates")
    if not isinstance(eligible, list) or not all(isinstance(item, str) for item in eligible):
        raise ValueError(f"report {path} has invalid selection.eligible_candidates")
    baseline = aggregates.get("live_defaults_reference")
    if not isinstance(baseline, Mapping) or not isinstance(baseline.get("held_out"), Mapping):
        raise ValueError(f"report {path} is missing baseline held-out aggregate")
    candidate_ranks = {
        item["configuration_id"]: int(item["train_rank"])
        for item in report.get("candidates", [])
        if isinstance(item, Mapping) and isinstance(item.get("configuration_id"), str)
    }
    if not candidate_ranks:
        raise ValueError(f"report {path} has no candidate ranks")
    return {
        "path": str(path),
        "injection_step": injection_step,
        "injection_time_s": _finite(simulator.get("injection_time_s"), "injection_time_s"),
        "eligible_candidates": sorted(eligible),
        "candidate_ranks": candidate_ranks,
        "baseline_held_out": {
            key: baseline["held_out"].get(key)
            for key in (
                "case_count",
                "recovery_activated_count",
                "recovery_success_rate",
                "recovery_failure_count",
                "progress_continuity_rate",
                "projected_progress_completion_rate",
                "stable_rejoin_rate",
                "mean_recovery_latency_s",
                "total_redivergence_count",
                "finished_rate",
            )
        },
    }


def build_schedule_report(report_paths: Sequence[Path] = DEFAULT_REPORTS) -> dict[str, Any]:
    paths = tuple(report_paths)
    if len(paths) < 2:
        raise ValueError("at least two injection-step reports are required")
    reports = [_load_report(path) for path in paths]
    summaries = [_report_summary(path, report) for path, report in zip(paths, reports)]
    summaries.sort(key=lambda item: item["injection_step"])
    steps = [item["injection_step"] for item in summaries]
    if len(set(steps)) != len(steps):
        raise ValueError("injection-step reports must have unique steps")

    assignments = _case_assignments(reports[0])
    split_consistent = all(_case_assignments(report) == assignments for report in reports[1:])
    candidate_sets = [set(item["candidate_ranks"]) for item in summaries]
    common_candidates = set.intersection(*candidate_sets)
    eligible_sets = [set(item["eligible_candidates"]) for item in summaries]
    stable_eligible = sorted(set.intersection(*eligible_sets)) if eligible_sets else []

    schedule_ranking = []
    for candidate in sorted(common_candidates):
        ranks = [item["candidate_ranks"][candidate] for item in summaries]
        schedule_ranking.append(
            {
                "configuration_id": candidate,
                "train_rank_by_step": {
                    str(item["injection_step"]): item["candidate_ranks"][candidate] for item in summaries
                },
                "mean_train_rank": sum(ranks) / len(ranks),
                "worst_train_rank": max(ranks),
                "eligible_at_steps": [item["injection_step"] for item in summaries if candidate in item["eligible_candidates"]],
            }
        )
    schedule_ranking.sort(
        key=lambda item: (item["mean_train_rank"], item["worst_train_rank"], item["configuration_id"])
    )
    diagnostic_recommendation = (
        schedule_ranking[0]["configuration_id"]
        if split_consistent and stable_eligible and schedule_ranking
        else None
    )
    if diagnostic_recommendation not in stable_eligible:
        diagnostic_recommendation = None

    report = {
        "scope": "offline_fixed_midrollout_injection_schedule_stability",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "promotion_allowed": False,
        "independent_confirmation": False,
        "schedule": {
            "injection_steps": steps,
            "injection_times_s": [item["injection_time_s"] for item in summaries],
            "report_count": len(summaries),
            "same_case_assignments_across_reports": split_consistent,
            "candidate_eligibility_rule": "candidate must be eligible under the existing selection rule at every injection step",
            "diagnostic_recommendation": diagnostic_recommendation,
            "stable_eligible_candidates": stable_eligible,
        },
        "reports": summaries,
        "candidate_schedule_ranking": schedule_ranking,
        "validation": {
            "valid": split_consistent,
            "issues": [] if split_consistent else ["source case assignments differ across schedule reports"],
        },
        "interpretation": {
            "selection_is_not_promotion": True,
            "schedule_stability_is_not_independent_confirmation": True,
            "timing_sensitivity_is_expected": True,
            "next_action": (
                "do not promote; add a separately held-out injection schedule or new disturbance family"
                if diagnostic_recommendation is None
                else "retain as diagnostic only and require independent confirmation before any promotion"
            ),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_schedule_report(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "schedule": report["schedule"]}, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
