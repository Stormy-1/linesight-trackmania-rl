import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report_robustness import paired_outcomes, sign_test_p, summarize_maps
from report_test_set import RunMetrics


def metrics(stem, finished, pct=100.0, time=30.0):
    return RunMetrics(
        stem=stem, tier="White", finished=finished, termination="Finished" if finished else "Stuck_Cutoff_2s",
        completion_pct=pct if not finished else 100.0, race_time_s=time if finished else None, replay_time_s=20.0,
        landings=0, bumps=0, walls=0, scrapes=0, recovery_steps=0,
    )


def test_summarize_maps_counts_finishes_and_pools_runs():
    runs = {"X01": [metrics("X01", True), metrics("X01", False, 50.0)], "X02": [metrics("X02", False, 10.0)] * 2}
    per_map, total = summarize_maps(runs)
    assert per_map["X01"]["finished"] == 1 and per_map["X01"]["runs"] == 2
    assert per_map["X02"]["finished"] == 0 and per_map["X02"]["mean_completion_pct"] == 10.0
    assert total["runs"] == 4 and total["finished"] == 1 and total["finish_rate"] == 0.25
    assert total["mean_completion_pct"] == pytest.approx((100 + 50 + 10 + 10) / 4, abs=0.05)
    assert total["maps_finished_at_least_once"] == 1


def test_paired_outcomes_counts_discordant_pairs_by_map_and_run():
    a = {"X01": [metrics("X01", True), metrics("X01", False)], "X02": [metrics("X02", False)]}
    b = {"X01": [metrics("X01", True), metrics("X01", True)], "X02": [metrics("X02", True)]}
    pairs = paired_outcomes(a, b)
    assert pairs == {"both": 1, "only_a": 0, "only_b": 2, "neither": 0}


def test_paired_outcomes_ignores_unmatched_runs():
    a = {"X01": [metrics("X01", True)] * 3}
    b = {"X01": [metrics("X01", False)], "X02": [metrics("X02", True)]}
    assert sum(paired_outcomes(a, b).values()) == 1


def test_sign_test_matches_exact_binomial():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 5) == 1.0
    assert sign_test_p(0, 6) == pytest.approx(2 / 64)
    assert sign_test_p(1, 9) == pytest.approx(2 * 11 / 1024)
