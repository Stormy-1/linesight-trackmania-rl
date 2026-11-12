"""Analyze the named measured action-transition capture without live control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trackmania_rl.hybrid_action_dynamics import action_transition_metrics
from trackmania_rl.hybrid_action_telemetry import load_action_rollout_csv

DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/action_transition_a01_20261103_retry3.json")
CAPTURE_ROOT = Path("eval_results/hybrid_rollouts/action_transition_a01_20261103_retry3")
MEASURED_SOURCES = (
    CAPTURE_ROOT / "A01-Race.Challenge_1793720895263_37200.csv",
    CAPTURE_ROOT / "A01-Race.Challenge_1793720904196_37200.csv",
    CAPTURE_ROOT / "A01-Race.Challenge_1793720908486_37200.csv",
)
WARMUP_SOURCE = CAPTURE_ROOT / "A01-Race.Challenge_1793720880525_37200.csv"
BENCHMARK_SOURCE = Path("eval_results/hybrid_rollouts/action_transition_a01_20261103_retry3_benchmark.csv")


def _benchmark_rows() -> list[dict[str, str]]:
    import csv

    with BENCHMARK_SOURCE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_report() -> dict[str, Any]:
    benchmark_rows = _benchmark_rows()
    if len(benchmark_rows) != len(MEASURED_SOURCES):
        raise ValueError("benchmark row count does not match the measured capture source count")
    runs = []
    for run_index, (source, benchmark) in enumerate(zip(MEASURED_SOURCES, benchmark_rows), start=1):
        samples = load_action_rollout_csv(source)
        metrics = action_transition_metrics(samples)
        expected_actions = int(benchmark["total_actions"])
        if len(samples) != expected_actions:
            raise ValueError(f"telemetry/action count mismatch for run {run_index}: {len(samples)} != {expected_actions}")
        runs.append(
            {
                "run_idx": run_index,
                "source": source.as_posix(),
                "sample_count": len(samples),
                "benchmark_total_actions": expected_actions,
                "benchmark_completion_pct": float(benchmark["completion_pct"]),
                "benchmark_finished": benchmark["finished"].lower() == "true",
                "benchmark_term_cause": benchmark["term_cause"],
                "metrics": metrics,
            }
        )
    return {
        "scope": "offline_action_conditioned_transition_analysis",
        "capture_name": "action_transition_a01_20261103_retry3",
        "track": "A01-Race",
        "checkpoint": "foundational_3.18M",
        "training_started": False,
        "controller_defaults_modified": False,
        "actions_modified": False,
        "real_capture_verified": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "warmup_source_excluded": WARMUP_SOURCE.as_posix(),
        "measured_sources": [source.as_posix() for source in MEASURED_SOURCES],
        "benchmark_source": BENCHMARK_SOURCE.as_posix(),
        "runs": runs,
        "capture_summary": {
            "run_count": len(runs),
            "finish_count": sum(run["benchmark_finished"] for run in runs),
            "completion_percentages": [run["benchmark_completion_pct"] for run in runs],
            "all_action_counts_aligned": all(
                run["sample_count"] == run["benchmark_total_actions"] for run in runs
            ),
            "all_transition_metrics_finite": all(run["metrics"]["finite_metrics"] for run in runs),
        },
        "identifiability": {
            "action_columns_present": True,
            "action_rows_are_joined_at_decision_point": True,
            "causal_dynamics_identified": False,
            "allowed_conclusion": "observational action-conditioned transition statistics",
        },
        "limitations": [
            "The capture uses the foundational RL policy rather than the hybrid controller.",
            "Actions are observed on-policy and are not randomized interventions.",
            "Action-conditioned statistics do not establish causal dynamics or safe controller parameters.",
            "The synchronization trace was excluded because it is not one of the three measured benchmark runs.",
            "The run set finished 0/3 and remains failure evidence, not promotion evidence.",
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
                "runs": report["capture_summary"]["run_count"],
                "finish_count": report["capture_summary"]["finish_count"],
                "aligned": report["capture_summary"]["all_action_counts_aligned"],
                "promotion_allowed": report["promotion_allowed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
