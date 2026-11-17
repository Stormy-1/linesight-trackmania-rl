"""Map-level reanalysis of the test-set robustness runs (v7 vs v8_rl vs v9_rl).

`report_robustness.py` pools 100 (map, run) pairs into one exact sign test. Runs on the same
map are not independent: a map that never finishes gives 10 identical failures in every version.
This script treats the map as the unit instead:

- per-map discordant pairs against the base (which maps carry any information at all);
- an exact sign-flip test over maps, on the per-map change in finishes and in mean completion;
- a cluster bootstrap over maps for the change in finish rate and mean completion;
- the intraclass correlation of finishing, the design effect and the effective sample size;
- a power simulation: the smallest odds-ratio effect the map-level test detects 80% of the time.

statsmodels is not installed in the project environment, so there is no mixed-effects model.
With 7 of 10 maps at 0% or 100% in every version, a random-intercept logistic model would be
identified by the same 3 maps anyway (see review/stats.md).

Usage:
    python scripts/reanalyze_clustered.py [--json review/clustered_reanalysis.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BASE_TAG = "v7_rob"
RL_TAGS = ("v8_rl_rob", "v9_rl_rob")
BOOTSTRAP_RESAMPLES = 10_000
POWER_SIMULATIONS = 2_000
ALPHA = 0.05
TARGET_POWER = 0.8
SEED = 20261117
ODDS_RATIO_GRID = (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0, 64.0)

MapOutcomes = dict[str, tuple[list[bool], list[float]]]


def load_outcomes(tag: str) -> MapOutcomes:
    """Per test map: the finished flag and completion % of every run, in run order."""

    from report_robustness import collect_runs

    runs = collect_runs("test", tag)
    if not runs:
        raise SystemExit(f"no runs found for tag {tag!r}")
    return {stem: ([r.finished for r in rs], [r.completion_pct for r in rs]) for stem, rs in runs.items()}


def discordant_pairs(base: list[bool], other: list[bool]) -> tuple[int, int]:
    """(only base finished, only other finished) over runs paired by index."""

    only_base = sum(1 for a, b in zip(base, other) if a and not b)
    only_other = sum(1 for a, b in zip(base, other) if b and not a)
    return only_base, only_other


def sign_flip_p(diffs: list[float]) -> float:
    """Exact two-sided sign-flip test of a zero mean, enumerating every sign pattern of the non-zero diffs."""

    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    hits = sum(
        1 for signs in itertools.product((1, -1), repeat=len(nonzero))
        if abs(sum(s * d for s, d in zip(signs, nonzero))) >= observed - 1e-12
    )
    return hits / 2 ** len(nonzero)


def min_attainable_p(n_informative_maps: int) -> float:
    """The smallest p the exact sign-flip test can return with this many non-zero maps."""

    return 1.0 if n_informative_maps == 0 else min(1.0, 2.0 / 2**n_informative_maps)


def cluster_bootstrap(diffs: list[float], rng: np.random.Generator, resamples: int) -> tuple[float, float, float]:
    """Mean of per-map diffs and its 95% percentile interval, resampling maps with replacement."""

    values = np.asarray(diffs, dtype=float)
    picks = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[picks].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def icc_binary(groups: list[list[bool]]) -> float:
    """One-way ANOVA intraclass correlation ICC(1) of a 0/1 outcome, equal group sizes."""

    k = len(groups[0])
    data = np.asarray(groups, dtype=float)
    grand = data.mean()
    ms_between = k * ((data.mean(axis=1) - grand) ** 2).sum() / (len(groups) - 1)
    ms_within = ((data - data.mean(axis=1, keepdims=True)) ** 2).sum() / (len(groups) * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    return 0.0 if denom == 0 else float((ms_between - ms_within) / denom)


def effective_sample_size(n_runs: int, runs_per_map: int, icc: float) -> float:
    return n_runs / (1.0 + (runs_per_map - 1) * icc)


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def map_level_power(base_rates: list[float], odds_ratio: float, runs_per_map: int,
                    rng: np.random.Generator, simulations: int) -> float:
    """Share of simulated studies where the map-level sign-flip test on finish counts reaches p < ALPHA.

    Every map's finish odds are multiplied by ``odds_ratio``. Base and RL runs are drawn independently,
    which ignores the shared noise of a pair (a mild understatement of power for a paired design).
    """

    shifted = [expit(logit(p) + math.log(odds_ratio)) for p in base_rates]
    hits = 0
    for _ in range(simulations):
        base = rng.binomial(runs_per_map, base_rates)
        treated = rng.binomial(runs_per_map, shifted)
        hits += sign_flip_p(list((treated - base).astype(float))) < ALPHA
    return hits / simulations


def smoothed_rates(finished_counts: list[int], runs_per_map: int) -> list[float]:
    """Jeffreys-smoothed per-map finish rates, so that 0/10 and 10/10 maps can still move."""

    return [(k + 0.5) / (runs_per_map + 1.0) for k in finished_counts]


def compare(base: MapOutcomes, other: MapOutcomes, rng: np.random.Generator) -> dict:
    maps = sorted(base.keys() & other.keys())
    per_map, finish_diffs, completion_diffs = {}, [], []
    for stem in maps:
        (b_fin, b_comp), (o_fin, o_comp) = base[stem], other[stem]
        only_base, only_other = discordant_pairs(b_fin, o_fin)
        finish_diffs.append(float(sum(o_fin) - sum(b_fin)))
        completion_diffs.append(float(np.mean(o_comp) - np.mean(b_comp)))
        per_map[stem] = {"base_finished": sum(b_fin), "other_finished": sum(o_fin),
                         "only_base": only_base, "only_other": only_other,
                         "completion_diff_pts": round(completion_diffs[-1], 2)}
    informative = [s for s in maps if per_map[s]["only_base"] + per_map[s]["only_other"] > 0]
    runs_per_map = len(base[maps[0]][0])
    finish_mean, finish_lo, finish_hi = cluster_bootstrap(finish_diffs, rng, BOOTSTRAP_RESAMPLES)
    comp_mean, comp_lo, comp_hi = cluster_bootstrap(completion_diffs, rng, BOOTSTRAP_RESAMPLES)
    return {
        "maps": len(maps),
        "per_map": per_map,
        "maps_with_discordant_pairs": informative,
        "map_sign_flip_p_finishes": sign_flip_p(finish_diffs),
        "map_sign_flip_p_completion": sign_flip_p(completion_diffs),
        "min_attainable_p_finishes": min_attainable_p(sum(1 for d in finish_diffs if d != 0)),
        "bootstrap_finish_rate_diff_pts": [round(100 * v / runs_per_map, 1) for v in (finish_mean, finish_lo, finish_hi)],
        "bootstrap_completion_diff_pts": [round(v, 1) for v in (comp_mean, comp_lo, comp_hi)],
    }


def power_analysis(base: MapOutcomes, rng: np.random.Generator) -> dict:
    counts = [sum(fin) for fin, _ in base.values()]
    runs_per_map = len(next(iter(base.values()))[0])
    rates = smoothed_rates(counts, runs_per_map)
    curve = {}
    for odds_ratio in ODDS_RATIO_GRID:
        power = map_level_power(rates, odds_ratio, runs_per_map, rng, POWER_SIMULATIONS)
        gain = 100 * (np.mean([expit(logit(p) + math.log(odds_ratio)) for p in rates]) - np.mean(rates))
        curve[str(odds_ratio)] = {"power": round(power, 3), "expected_finish_rate_gain_pts": round(float(gain), 1)}
    detectable = next((float(k) for k, v in curve.items() if v["power"] >= TARGET_POWER), None)
    return {"base_rates_smoothed": [round(r, 3) for r in rates], "curve": curve,
            "smallest_odds_ratio_with_80pct_power": detectable}


def clustering_summary(base: MapOutcomes) -> dict:
    groups = [fin for fin, _ in base.values()]
    icc = icc_binary(groups)
    n_runs = sum(len(g) for g in groups)
    return {"icc_finish": round(icc, 3), "design_effect": round(1 + (len(groups[0]) - 1) * icc, 2),
            "runs": n_runs, "effective_sample_size": round(effective_sample_size(n_runs, len(groups[0]), icc), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the full result here")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    base = load_outcomes(BASE_TAG)
    result = {"base": BASE_TAG, "seed": SEED, "clustering_v7": clustering_summary(base)}
    for tag in RL_TAGS:
        result[f"{tag}_vs_{BASE_TAG}"] = compare(base, load_outcomes(tag), rng)
    result["power_map_level"] = power_analysis(base, rng)
    print(json.dumps(result, indent=1))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
