"""Robustness report: finish rates over several perturbed runs per map, and a paired comparison.

A single deterministic run per map flips between finish and fail at knife-edge jumps
(HANDOVER §70, §82). The robustness evaluation drives every map N times with small seeded
steering noise (``--steer-noise``, ``--perturb-seed``; run k sees the same noise stream for
every version), so each version gets a finish *rate*, and two versions can be compared run by
run with an exact sign test on the discordant (map, run) pairs.

Usage:
    python scripts/report_robustness.py --tag v7_rob
    python scripts/report_robustness.py --tag v8_rl_rob --compare v7_rob [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from classify_hybrid_impacts import LIVE_DIR
from make_map_split import MANIFEST, load_split
from report_test_set import RunMetrics, campaign_maps, run_dir_name, run_metrics

RunsByMap = dict[str, list[RunMetrics]]


def load_all_runs(stem: str, tier: str, replay_time_s: float, name: str) -> list[RunMetrics]:
    """Every run in ``<name>/summary.json`` (run k is logged in ``runKK.csv``)."""

    run_dir = LIVE_DIR / name
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text())
    results = []
    for index, run in enumerate(summary["runs"]):
        with open(run_dir / f"run{run['run']:02d}.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        results.append(run_metrics(stem, tier, replay_time_s, summary, rows, run_index=index))
    return results


def collect_runs(split_key: str, tag: str) -> RunsByMap:
    manifest = json.loads(MANIFEST.read_text())
    runs = {}
    for stem in campaign_maps(load_split()[split_key], manifest):
        entry = manifest[stem]
        found = load_all_runs(stem, entry["group"], entry["author_time_s"], run_dir_name(stem, tag, split_key))
        if found:
            runs[stem] = found
    return runs


def summarize_maps(runs: RunsByMap) -> tuple[dict, dict]:
    per_map = {}
    for stem, results in runs.items():
        times = [r.race_time_s for r in results if r.finished]
        per_map[stem] = {
            "tier": results[0].tier,
            "runs": len(results),
            "finished": len(times),
            "mean_completion_pct": round(statistics.fmean(r.completion_pct for r in results), 1),
            "median_time_s": round(statistics.median(times), 2) if times else None,
        }
    pooled = [r for results in runs.values() for r in results]
    total = {
        "maps": len(runs),
        "runs": len(pooled),
        "finished": sum(r.finished for r in pooled),
        "finish_rate": round(sum(r.finished for r in pooled) / len(pooled), 3) if pooled else 0.0,
        "mean_completion_pct": round(statistics.fmean(r.completion_pct for r in pooled), 1) if pooled else 0.0,
        "maps_finished_at_least_once": sum(1 for m in per_map.values() if m["finished"] > 0),
        "walls": sum(r.walls for r in pooled),
        "recovery_steps": sum(r.recovery_steps for r in pooled),
    }
    return per_map, total


def paired_outcomes(runs_a: RunsByMap, runs_b: RunsByMap) -> dict[str, int]:
    """Outcome pairs over (map, run index) present in both versions."""

    counts = {"both": 0, "only_a": 0, "only_b": 0, "neither": 0}
    for stem in runs_a.keys() & runs_b.keys():
        for a, b in zip(runs_a[stem], runs_b[stem]):
            key = "both" if a.finished and b.finished else "only_a" if a.finished else "only_b" if b.finished else "neither"
            counts[key] += 1
    return counts


def sign_test_p(only_a: int, only_b: int) -> float:
    """Exact two-sided sign test (McNemar) on the discordant pairs."""

    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def format_report(tag: str, per_map: dict, total: dict) -> str:
    lines = [f"### {tag}", "| Map | Tier | Finished | Mean completion | Median time |", "|---|---|---|---|---|"]
    for stem, m in per_map.items():
        time = f"{m['median_time_s']:.2f} s" if m["median_time_s"] is not None else "-"
        lines.append(f"| {stem} | {m['tier']} | {m['finished']}/{m['runs']} | {m['mean_completion_pct']}% | {time} |")
    lines.append(
        f"\n**{total['finished']}/{total['runs']} runs finished** (rate {total['finish_rate']:.0%}), "
        f"{total['maps_finished_at_least_once']}/{total['maps']} maps finished at least once, "
        f"mean completion {total['mean_completion_pct']}%, {total['walls']} walls, {total['recovery_steps']} recovery steps."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="run names are <map>_<set>_<tag>")
    parser.add_argument("--set", default="test", choices=("test", "rl_train", "dev"))
    parser.add_argument("--compare", default=None, help="baseline tag for a paired comparison (A = baseline, B = --tag)")
    parser.add_argument("--json", type=Path, help="also write the per-map and pooled results here")
    args = parser.parse_args()

    runs = collect_runs(args.set, args.tag)
    if not runs:
        parser.error(f"no runs found for tag {args.tag!r} in set {args.set!r}")
    per_map, total = summarize_maps(runs)
    payload = {"tag": args.tag, "set": args.set, "per_map": per_map, "total": total}
    print(format_report(args.tag, per_map, total))
    if args.compare:
        base_runs = collect_runs(args.set, args.compare)
        base_per_map, base_total = summarize_maps(base_runs)
        print("\n" + format_report(args.compare, base_per_map, base_total))
        pairs = paired_outcomes(base_runs, runs)
        p = sign_test_p(pairs["only_a"], pairs["only_b"])
        print(
            f"\nPaired over (map, run): both finished {pairs['both']}, only {args.compare} {pairs['only_a']}, "
            f"only {args.tag} {pairs['only_b']}, neither {pairs['neither']}; exact sign test p = {p:.3g}"
        )
        payload["compare"] = {"baseline": args.compare, "baseline_per_map": base_per_map, "baseline_total": base_total,
                              "pairs": pairs, "sign_test_p": p}
    if args.json:
        args.json.write_text(json.dumps(payload, indent=1) + "\n")


if __name__ == "__main__":
    main()
