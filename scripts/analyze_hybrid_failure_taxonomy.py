"""Classify held-out hybrid-recovery outcomes from existing JSON reports.

This is a read-only report-level analysis.  It does not rerun the simulator,
change selection rules, modify controller defaults, or reinterpret the
historical success metric.  It adds mutually explicit diagnostic labels so
failure timing, backward progress, stable rejoin, and re-divergence patterns
can be compared across the existing schedules and the new held-out family.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_REPORTS = (
    Path("eval_results/hybrid_rollouts/hybrid_midrollout_recovery_comparison_step05.json"),
    Path("eval_results/hybrid_rollouts/hybrid_midrollout_recovery_comparison.json"),
    Path("eval_results/hybrid_rollouts/hybrid_midrollout_recovery_comparison_step20.json"),
    Path("eval_results/hybrid_rollouts/hybrid_midrollout_recovery_comparison_step35.json"),
    Path("eval_results/hybrid_rollouts/hybrid_midrollout_recovery_comparison_step50.json"),
    Path("eval_results/hybrid_rollouts/hybrid_midrollout_recovery_comparison_step80.json"),
    Path("eval_results/hybrid_rollouts/hybrid_new_disturbance_recovery_comparison_step15.json"),
    Path("eval_results/hybrid_rollouts/hybrid_new_disturbance_recovery_comparison_step30.json"),
    Path("eval_results/hybrid_rollouts/hybrid_new_disturbance_recovery_comparison_step60.json"),
)
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_failure_taxonomy.json")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"report {path} must contain a JSON object")
    return value


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _number(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def classify_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Return diagnostic labels without changing the historical metrics."""

    status = case.get("status")
    if status != "ok":
        return {
            "primary_outcome": f"invalid_or_incomplete:{status}",
            "outcome_labels": [f"status:{status}"],
        }

    metrics = case.get("metrics")
    if not isinstance(metrics, Mapping):
        return {
            "primary_outcome": "invalid_or_incomplete:missing_metrics",
            "outcome_labels": ["status:missing_metrics"],
        }
    recovery = metrics.get("recovery")
    if not isinstance(recovery, Mapping):
        recovery = {}

    activated = bool(metrics.get("recovery_activated"))
    completed = bool(metrics.get("recovery_completed"))
    stable = bool(recovery.get("stable_rejoin"))
    redivergence_count = int(recovery.get("redivergence_count") or 0)
    continuity = bool(metrics.get("progress_continuity"))
    success = metrics.get("recovery_success")

    labels: list[str] = []
    if not activated:
        labels.append("not_activated")
    if activated and not completed:
        labels.append("activated_without_completion")
    if activated and not stable:
        labels.append("no_stable_rejoin")
    if redivergence_count > 0:
        labels.append("redivergence_after_rejoin" if stable else "redivergence_before_stable_rejoin")
    if activated and not continuity:
        labels.append("backward_progress_after_injection")
    if success is True:
        labels.append("recovery_success")
    elif success is False:
        labels.append("recovery_failure")
    else:
        labels.append("no_recovery_observation")

    if not activated:
        primary = "not_activated"
    elif not completed:
        primary = "activated_without_completion"
    elif not stable:
        primary = "no_stable_rejoin"
    elif redivergence_count > 0:
        primary = "stable_rejoin_then_redivergence"
    elif not continuity:
        primary = "stable_rejoin_with_backward_progress"
    elif success is True:
        primary = "recovery_success"
    else:
        primary = "recovery_failure_other"
    return {"primary_outcome": primary, "outcome_labels": labels}


def _schedule_group(path: Path, injection_step: int) -> str:
    if "new_disturbance" in path.name:
        return "new_disturbance_family"
    if injection_step in {5, 10, 20}:
        return "primary_schedule"
    if injection_step in {35, 50, 80}:
        return "timing_holdout_schedule"
    return "other_schedule"


def _family_id(report: Mapping[str, Any]) -> str:
    evaluation = report.get("evaluation")
    if isinstance(evaluation, Mapping) and isinstance(evaluation.get("family_id"), str):
        return str(evaluation["family_id"])
    return "historical_disturbance_family"


def _case_record(
    *, path: Path, group: str, family_id: str, injection_step: int, map_name: str, profile: str, case: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = case.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    recovery = metrics.get("recovery")
    if not isinstance(recovery, Mapping):
        recovery = {}
    classification = classify_case(case)
    return {
        "report": str(path),
        "schedule_group": group,
        "family_id": family_id,
        "injection_step": injection_step,
        "map": map_name,
        "profile": profile,
        "case_key": case.get("case_key"),
        "disturbance": case.get("disturbance"),
        "split": case.get("split"),
        "status": case.get("status"),
        **classification,
        "recovery_activated": bool(metrics.get("recovery_activated")),
        "recovery_completed": bool(metrics.get("recovery_completed")),
        "recovery_success": metrics.get("recovery_success"),
        "progress_continuity": metrics.get("progress_continuity"),
        "stable_rejoin": bool(recovery.get("stable_rejoin")),
        "redivergence_count": int(recovery.get("redivergence_count") or 0),
        "backward_progress_distance_m": _number(metrics, "backward_progress_distance_m"),
        "recovery_latency_s": _number(recovery, "recovery_latency_s"),
        "termination_reason": metrics.get("termination_reason"),
    }


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = Counter(str(record["primary_outcome"]) for record in records)
    labels = Counter(label for record in records for label in record["outcome_labels"])
    backward = [float(record["backward_progress_distance_m"]) for record in records if record["backward_progress_distance_m"] is not None]
    latencies = [float(record["recovery_latency_s"]) for record in records if record["recovery_latency_s"] is not None]
    return {
        "case_count": len(records),
        "primary_outcomes": dict(sorted(primary.items())),
        "outcome_labels": dict(sorted(labels.items())),
        "status_counts": dict(sorted(Counter(str(record["status"]) for record in records).items())),
        "failure_case_count": sum(record["recovery_success"] is False for record in records),
        "backward_progress_case_count": sum(
            "backward_progress_after_injection" in record["outcome_labels"] for record in records
        ),
        "redivergence_case_count": sum(record["redivergence_count"] > 0 for record in records),
        "total_redivergence_count": sum(int(record["redivergence_count"]) for record in records),
        "mean_backward_progress_distance_m": _mean(backward),
        "mean_recovery_latency_s": _mean(latencies),
    }


def build_failure_taxonomy(report_paths: Sequence[Path] = DEFAULT_REPORTS) -> dict[str, Any]:
    """Build deterministic held-out case classifications from report files."""

    paths = tuple(report_paths)
    if not paths:
        raise ValueError("at least one report is required")

    report_summaries: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for path in paths:
        report = _load_object(path)
        simulator = report.get("simulator")
        maps = report.get("maps")
        if not isinstance(simulator, Mapping) or not isinstance(maps, Mapping):
            raise TypeError(f"report {path} is missing simulator or maps")
        injection_step = simulator.get("injection_step")
        if not isinstance(injection_step, int) or isinstance(injection_step, bool) or injection_step < 1:
            raise ValueError(f"report {path} has an invalid injection step")
        group = _schedule_group(path, injection_step)
        family_id = _family_id(report)
        profile_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for map_name in sorted(maps):
            item = maps[map_name]
            if not isinstance(item, Mapping):
                raise TypeError(f"report {path} has invalid map entry {map_name!r}")
            cases_by_profile = item.get("cases_by_profile")
            if not isinstance(cases_by_profile, Mapping):
                raise TypeError(f"report {path} map {map_name!r} is missing cases_by_profile")
            for profile in sorted(cases_by_profile):
                cases = cases_by_profile[profile]
                if not isinstance(cases, list):
                    raise TypeError(f"report {path} profile {profile!r} cases must be a list")
                for case in cases:
                    if not isinstance(case, Mapping):
                        raise TypeError(f"report {path} contains a non-object case")
                    if case.get("split") != "held_out":
                        continue
                    record = _case_record(
                        path=path,
                        group=group,
                        family_id=family_id,
                        injection_step=injection_step,
                        map_name=str(map_name),
                        profile=str(profile),
                        case=case,
                    )
                    profile_records[str(profile)].append(record)
                    all_records.append(record)

        report_summaries.append(
            {
                "path": str(path),
                "schedule_group": group,
                "family_id": family_id,
                "injection_step": injection_step,
                "injection_time_s": simulator.get("injection_time_s"),
                "profile_aggregates": {
                    profile: _aggregate_records(records)
                    for profile, records in sorted(profile_records.items())
                },
                "baseline_held_out": (
                    report.get("aggregates", {})
                    .get("live_defaults_reference", {})
                    .get("held_out", {})
                ),
            }
        )

    baseline = [record for record in all_records if record["profile"] == "live_defaults_reference"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in baseline:
        grouped[(str(record["schedule_group"]), str(record["disturbance"]))].append(record)
    baseline_by_group_and_disturbance = {}
    for (group, disturbance), records in sorted(grouped.items()):
        baseline_by_group_and_disturbance.setdefault(group, {})[disturbance] = _aggregate_records(records)

    return {
        "scope": "offline_hybrid_held_out_failure_taxonomy",
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
        "classification": {
            "unit": "held-out case/profile/report",
            "historical_success_metric_unchanged": True,
            "labels_are_diagnostic_only": True,
        },
        "reports": report_summaries,
        "baseline_by_schedule_group_and_disturbance": baseline_by_group_and_disturbance,
        "case_records": all_records,
        "counts": {
            "reports": len(report_summaries),
            "held_out_case_records": len(all_records),
            "baseline_case_records": len(baseline),
            "profiles": sorted({str(record["profile"]) for record in all_records}),
        },
        "interpretation": {
            "next_action": "use failure labels to choose the next untouched disturbance or trajectory experiment",
            "no_promotion_basis": "taxonomy does not alter candidate ranking or acceptance",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_failure_taxonomy(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "counts": report["counts"], "groups": sorted(report["baseline_by_schedule_group_and_disturbance"])},
            allow_nan=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
