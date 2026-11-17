import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from reanalyze_clustered import (  # noqa: E402
    cluster_bootstrap,
    discordant_pairs,
    effective_sample_size,
    icc_binary,
    min_attainable_p,
    sign_flip_p,
    smoothed_rates,
)


def test_discordant_pairs_counts_each_direction():
    assert discordant_pairs([True, True, False, False], [True, False, True, True]) == (1, 2)


def test_sign_flip_ignores_zero_maps_and_matches_hand_count():
    # Diffs +2, +7, -7 (the v9_rl finish changes on A09, B07, C02): every sign pattern has |sum| >= 2.
    assert sign_flip_p([2, 7, -7, 0, 0, 0, 0, 0, 0, 0]) == 1.0


def test_sign_flip_all_same_sign_reaches_two_over_two_to_the_n():
    assert sign_flip_p([1.0] * 6) == pytest.approx(2 / 64)
    assert min_attainable_p(6) == pytest.approx(2 / 64)
    assert min_attainable_p(0) == 1.0


def test_sign_flip_no_change_is_one():
    assert sign_flip_p([0, 0, 0]) == 1.0


def test_cluster_bootstrap_of_constant_diffs_has_zero_width():
    mean, lo, hi = cluster_bootstrap([3.0] * 5, np.random.default_rng(0), 200)
    assert mean == lo == hi == 3.0


def test_icc_is_one_when_maps_are_all_or_nothing():
    groups = [[True] * 10, [False] * 10, [True] * 10, [False] * 10]
    assert icc_binary(groups) == pytest.approx(1.0)
    assert effective_sample_size(40, 10, 1.0) == pytest.approx(4.0)


def test_smoothed_rates_keep_extremes_inside_zero_one():
    rates = smoothed_rates([0, 10], 10)
    assert 0 < rates[0] < 0.1 and 0.9 < rates[1] < 1
