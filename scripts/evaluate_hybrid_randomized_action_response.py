"""Evaluate randomized action responses on disjoint measured capture files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_action_dynamics import randomized_action_transition_records
from trackmania_rl.hybrid_action_telemetry import load_randomized_action_rollout_csv

SAFE_ACTION_INDICES = tuple(range(9))
DEFAULT_MINIMUM_RANDOMIZED_TRANSITIONS = 50
RESPONSE_FIELDS = (
    "speed_change_rate_mps2",
    "yaw_rate_rad_s",
    "lateral_displacement_rate_mps",
)


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_splits(fit_sources: tuple[Path, ...], heldout_sources: tuple[Path, ...]) -> None:
    if not fit_sources or not heldout_sources:
        raise ValueError("fit and held-out source lists must both be non-empty")
    fit_resolved = {path.resolve() for path in fit_sources}
    heldout_resolved = {path.resolve() for path in heldout_sources}
    if len(fit_resolved) != len(fit_sources) or len(heldout_resolved) != len(heldout_sources):
        raise ValueError("source paths must be unique within each split")
    if not fit_resolved.isdisjoint(heldout_resolved):
        raise ValueError("fit and held-out source paths must be disjoint")
    for path in (*fit_sources, *heldout_sources):
        if not path.is_file():
            raise FileNotFoundError(path)


def _validate_provenance(
    provenance: dict[str, Any],
    fit_sources: tuple[Path, ...],
    heldout_sources: tuple[Path, ...],
) -> None:
    if provenance.get("status") != "complete":
        raise ValueError("capture provenance must have status=complete")
    if provenance.get("training_enabled") is not False or provenance.get("checkpoint_writes_enabled") is not False:
        raise ValueError("capture provenance must record training and checkpoint writes as disabled")
    if float(provenance.get("intervention_probability", -1.0)) != 0.25:
        raise ValueError("capture provenance has an unexpected intervention probability")
    if tuple(provenance.get("safe_randomized_action_indices", ())) != SAFE_ACTION_INDICES:
        raise ValueError("capture provenance has an unexpected randomized action set")
    runs = provenance.get("runs")
    if not isinstance(runs, list) or len(runs) != 6:
        raise ValueError("capture provenance must contain exactly six measured runs")
    if len({run.get("seed") for run in runs}) != 6:
        raise ValueError("capture provenance seeds must be distinct")
    expected_paths = {
        "fit": {path.resolve() for path in fit_sources},
        "heldout": {path.resolve() for path in heldout_sources},
    }
    actual_paths: dict[str, set[Path]] = {"fit": set(), "heldout": set()}
    for run in runs:
        split = run.get("split")
        if split not in actual_paths:
            raise ValueError("capture provenance contains an invalid split")
        actual_paths[split].add(Path(run["telemetry_path"]).resolve())
    if actual_paths != expected_paths:
        raise ValueError("capture provenance paths do not match evaluator source splits")


def _collect_randomized_records(paths: Iterable[Path]) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    records: list[dict[str, float | int]] = []
    source_reports = []
    for source in paths:
        samples = load_randomized_action_rollout_csv(source)
        source_records = randomized_action_transition_records(samples)
        if not source_records:
            raise ValueError(f"source contains no randomized transitions: {source}")
        if not all(
            math.isfinite(float(record[field]))
            for record in source_records
            for field in ("dt_s", *RESPONSE_FIELDS)
        ):
            raise ValueError(f"source contains non-finite randomized transition metrics: {source}")
        counts = Counter(int(record["action_idx"]) for record in source_records)
        source_reports.append(
            {
                "path": source.as_posix(),
                "sha256": _fingerprint(source),
                "sample_count": len(samples),
                "randomized_transition_count": len(source_records),
                "randomized_action_counts": {str(index): int(counts[index]) for index in SAFE_ACTION_INDICES},
            }
        )
        records.extend(source_records)
    return records, {"sources": source_reports, "transition_count": len(records)}


def _quota_report(records: list[dict[str, float | int]], minimum: int) -> dict[str, Any]:
    counts = Counter(int(record["action_idx"]) for record in records)
    by_action = {str(index): int(counts[index]) for index in SAFE_ACTION_INDICES}
    return {
        "minimum_randomized_transitions_per_action": minimum,
        "randomized_action_counts": by_action,
        "all_action_quotas_met": all(count >= minimum for count in by_action.values()),
        "unexpected_randomized_actions": sorted(set(counts) - set(SAFE_ACTION_INDICES)),
    }


def _mean_response(records: list[dict[str, float | int]]) -> dict[str, float]:
    if not records:
        raise ValueError("cannot fit a response model without records")
    return {
        field: float(np.mean([float(record[field]) for record in records]))
        for field in RESPONSE_FIELDS
    }


def _fit_action_response_model(records: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    by_action: dict[str, list[dict[str, float | int]]] = {}
    for record in records:
        by_action.setdefault(str(int(record["action_idx"])), []).append(record)
    return {action: _mean_response(action_records) for action, action_records in sorted(by_action.items(), key=lambda item: int(item[0]))}


def _mse(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("MSE requires equally sized non-empty vectors")
    value = float(np.mean((np.asarray(actual, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)) ** 2))
    if not math.isfinite(value):
        raise ValueError("MSE must be finite")
    return value


def _score_models(
    fit_records: list[dict[str, float | int]],
    heldout_records: list[dict[str, float | int]],
) -> dict[str, Any]:
    action_model = _fit_action_response_model(fit_records)
    action_blind = _mean_response(fit_records)
    metrics: dict[str, dict[str, float]] = {}
    for field in RESPONSE_FIELDS:
        actual = [float(record[field]) for record in heldout_records]
        action_predicted = [action_model[str(int(record["action_idx"]))][field] for record in heldout_records]
        blind_predicted = [action_blind[field]] * len(heldout_records)
        metrics[field] = {
            "action_conditioned_mse": _mse(actual, action_predicted),
            "action_blind_mse": _mse(actual, blind_predicted),
        }
        metrics[field]["improvement"] = metrics[field]["action_blind_mse"] - metrics[field]["action_conditioned_mse"]
    primary = metrics["speed_change_rate_mps2"]
    return {
        "action_conditioned_mean_model": action_model,
        "action_blind_mean_model": action_blind,
        "heldout_metrics": metrics,
        "primary_metric": "speed_change_rate_mps2",
        "primary_action_conditioned_beats_blind": primary["action_conditioned_mse"] < primary["action_blind_mse"],
    }


def build_report(
    fit_sources: tuple[Path, ...],
    heldout_sources: tuple[Path, ...],
    *,
    minimum_randomized_transitions: int = DEFAULT_MINIMUM_RANDOMIZED_TRANSITIONS,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(minimum_randomized_transitions, bool) or minimum_randomized_transitions <= 0:
        raise ValueError("minimum_randomized_transitions must be positive")
    _validate_source_splits(fit_sources, heldout_sources)
    if provenance is not None:
        _validate_provenance(provenance, fit_sources, heldout_sources)
    fit_records, fit_report = _collect_randomized_records(fit_sources)
    heldout_records, heldout_report = _collect_randomized_records(heldout_sources)
    fit_quota = _quota_report(fit_records, minimum_randomized_transitions)
    heldout_quota = _quota_report(heldout_records, minimum_randomized_transitions)
    quotas_met = (
        fit_quota["all_action_quotas_met"]
        and heldout_quota["all_action_quotas_met"]
        and not fit_quota["unexpected_randomized_actions"]
        and not heldout_quota["unexpected_randomized_actions"]
    )
    score = _score_models(fit_records, heldout_records) if quotas_met else None
    identification_gate_passed = bool(quotas_met and score and score["primary_action_conditioned_beats_blind"])
    return {
        "scope": "offline_randomized_action_response_identification_gate",
        "fit_sources": fit_report,
        "heldout_sources": heldout_report,
        "source_splits_disjoint": True,
        "fit_quota": fit_quota,
        "heldout_quota": heldout_quota,
        "model_score": score,
        "provenance": provenance,
        "randomization_design": {
            "intervention_probability": 0.25,
            "safe_randomized_action_indices": list(SAFE_ACTION_INDICES),
            "assignment_state_independent": True,
        },
        "identification_gate_passed": identification_gate_passed,
        "causal_fitting_allowed": identification_gate_passed,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "decision": (
            "offline_identification_gate_passed"
            if identification_gate_passed
            else "retain_negative_or_underpowered_randomized_capture_evidence"
        ),
        "limitations": [
            "The response model is a one-step observational summary over randomized assignments, not a live controller.",
            "The simulator and telemetry do not establish TrackMania completion or real-physics controller safety.",
            "Promotion remains false regardless of the offline identification result.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", nargs="+", type=Path, required=True)
    parser.add_argument("--heldout", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--minimum-randomized-transitions", type=int, default=DEFAULT_MINIMUM_RANDOMIZED_TRANSITIONS)
    args = parser.parse_args()
    provenance = None
    if args.provenance is not None:
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    report = build_report(
        tuple(args.fit),
        tuple(args.heldout),
        minimum_randomized_transitions=args.minimum_randomized_transitions,
        provenance=provenance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "identification_gate_passed": report["identification_gate_passed"],
                "causal_fitting_allowed": report["causal_fitting_allowed"],
                "promotion_allowed": report["promotion_allowed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
