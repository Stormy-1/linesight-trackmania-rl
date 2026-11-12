"""Evaluate action-response consistency on disjoint measured capture runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from trackmania_rl.hybrid_action_dynamics import action_transition_metrics
from trackmania_rl.hybrid_action_telemetry import load_action_rollout_csv

CAPTURE_ROOT = Path("eval_results/hybrid_rollouts/action_transition_a01_20261103_retry3")
FIT_SOURCES = (
    CAPTURE_ROOT / "A01-Race.Challenge_1793720895263_37200.csv",
    CAPTURE_ROOT / "A01-Race.Challenge_1793720904196_37200.csv",
)
HELDOUT_SOURCES = (CAPTURE_ROOT / "A01-Race.Challenge_1793720908486_37200.csv",)
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/action_response_a01_20261103_holdout.json")


def _aggregate_by_action(source_paths: tuple[Path, ...]) -> dict[str, dict[str, float | int]]:
    aggregate: dict[str, dict[str, float | int]] = {}
    for source in source_paths:
        report = action_transition_metrics(load_action_rollout_csv(source))
        for action_idx, metrics in report["by_action"].items():
            action = aggregate.setdefault(action_idx, {"transition_count": 0, "weighted_speed_change_sum": 0.0})
            count = int(metrics["transition_count"])
            action["transition_count"] = int(action["transition_count"]) + count
            action["weighted_speed_change_sum"] = float(action["weighted_speed_change_sum"]) + count * float(
                metrics["mean_speed_change_rate_mps2"]
            )
    return {
        action_idx: {
            "transition_count": int(values["transition_count"]),
            "mean_speed_change_rate_mps2": float(
                values["weighted_speed_change_sum"] / max(int(values["transition_count"]), 1)
            ),
        }
        for action_idx, values in aggregate.items()
    }


def compare_action_response_splits(
    fit: dict[str, dict[str, float | int]],
    heldout: dict[str, dict[str, float | int]],
    *,
    minimum_transitions: int = 10,
) -> dict[str, Any]:
    """Compare sufficiently represented action-response signs across splits."""

    if not isinstance(minimum_transitions, int) or isinstance(minimum_transitions, bool) or minimum_transitions <= 0:
        raise ValueError("minimum_transitions must be a positive integer")
    common = []
    comparisons = []
    for action_idx in sorted(set(fit).intersection(heldout), key=int):
        fit_metrics = fit[action_idx]
        heldout_metrics = heldout[action_idx]
        fit_count = int(fit_metrics["transition_count"])
        heldout_count = int(heldout_metrics["transition_count"])
        fit_mean = float(fit_metrics["mean_speed_change_rate_mps2"])
        heldout_mean = float(heldout_metrics["mean_speed_change_rate_mps2"])
        if not math.isfinite(fit_mean) or not math.isfinite(heldout_mean):
            raise ValueError("action-response means must be finite")
        if fit_count < minimum_transitions or heldout_count < minimum_transitions:
            continue
        fit_sign = 1 if fit_mean > 0.0 else -1 if fit_mean < 0.0 else 0
        heldout_sign = 1 if heldout_mean > 0.0 else -1 if heldout_mean < 0.0 else 0
        common.append(action_idx)
        comparisons.append(
            {
                "action_idx": action_idx,
                "fit_transition_count": fit_count,
                "heldout_transition_count": heldout_count,
                "fit_mean_speed_change_rate_mps2": fit_mean,
                "heldout_mean_speed_change_rate_mps2": heldout_mean,
                "sign_agrees": fit_sign == heldout_sign,
            }
        )
    sign_agreement_count = sum(bool(item["sign_agrees"]) for item in comparisons)
    return {
        "minimum_transitions": minimum_transitions,
        "compared_actions": common,
        "comparisons": comparisons,
        "sign_agreement_count": sign_agreement_count,
        "sign_agreement_rate": float(sign_agreement_count / len(comparisons)) if comparisons else 0.0,
        "signs_replicate": bool(comparisons) and sign_agreement_count == len(comparisons),
        "causal_fitting_allowed": False,
    }


def build_report() -> dict[str, Any]:
    fit = _aggregate_by_action(FIT_SOURCES)
    heldout = _aggregate_by_action(HELDOUT_SOURCES)
    comparison = compare_action_response_splits(fit, heldout)
    return {
        "scope": "offline_action_response_holdout_consistency",
        "capture_name": "action_transition_a01_20261103_retry3",
        "fit_sources": [source.as_posix() for source in FIT_SOURCES],
        "heldout_sources": [source.as_posix() for source in HELDOUT_SOURCES],
        "source_splits_disjoint": {source.as_posix() for source in FIT_SOURCES}.isdisjoint(
            {source.as_posix() for source in HELDOUT_SOURCES}
        ),
        "fit_by_action": fit,
        "heldout_by_action": heldout,
        "comparison": comparison,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "decision": "descriptive_only_requires_randomized_action_diverse_capture",
        "limitations": [
            "Fit and held-out runs come from the same greedy foundational policy and map.",
            "The action distribution is on-policy and not randomized.",
            "Unobserved speed, map section, contact, and vehicle state confound action-conditioned means.",
            "Sign replication is a consistency check, not causal identification or a safe MPC parameter.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "compared_actions": report["comparison"]["compared_actions"],
                "sign_agreement_rate": report["comparison"]["sign_agreement_rate"],
                "causal_fitting_allowed": report["comparison"]["causal_fitting_allowed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
