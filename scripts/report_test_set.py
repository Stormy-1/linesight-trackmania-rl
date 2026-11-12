"""Turn a set of live runs into the evaluation results table.

Metrics (HANDOVER §67):
- finish rate: maps finished / maps driven (the headline).
- completion: % of the official line covered (partial credit on failures).
- time ratio: our race time / the official replay's race time (finished maps only; lower is better).
- contacts (quality, not pass/fail): landings, bumps, wall hits (``classify_impacts``) and
  scrapes (``find_scrapes``).
- recovery steps: control steps spent in the unstick/recovery behaviour.

Usage:
    python scripts/report_test_set.py --tag v4            # runs <map>_test_v4 for the test set
    python scripts/report_test_set.py --tag v4 --set rl_train
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from classify_hybrid_impacts import LIVE_DIR, classify_impacts, find_scrapes
from make_map_split import MANIFEST, load_split


@dataclass(frozen=True)
class RunMetrics:
    stem: str
    tier: str
    finished: bool
    termination: str
    completion_pct: float
    race_time_s: float | None
    replay_time_s: float
    landings: int
    bumps: int
    walls: int
    scrapes: int
    recovery_steps: int

    @property
    def time_ratio(self) -> float | None:
        return round(self.race_time_s / self.replay_time_s, 2) if self.finished else None


def run_dir_name(stem: str, tag: str, split_key: str = "test") -> str:
    return f"{stem}_{split_key}_{tag}"


def run_metrics(stem: str, tier: str, replay_time_s: float, summary: dict, rows: list[dict]) -> RunMetrics:
    if summary.get("speed_caps") or summary.get("max_speed_override_mps") is not None:
        raise ValueError(f"{stem}: per-map tuning (speed cap/override) is not allowed in an evaluation run")
    run = summary["runs"][0]
    impacts = classify_impacts(rows)
    kinds = [i.kind for i in impacts]
    return RunMetrics(
        stem=stem,
        tier=tier,
        finished=bool(run["finished"]),
        termination=run["termination"],
        completion_pct=float(run["completion_pct"]),
        race_time_s=float(run["race_time_s"]) if run["finished"] else None,
        replay_time_s=replay_time_s,
        landings=kinds.count("landing"),
        bumps=kinds.count("bump"),
        walls=kinds.count("wall"),
        scrapes=len(find_scrapes(rows)),
        recovery_steps=int(run["recovery_steps"]),
    )


def load_run(stem: str, tier: str, replay_time_s: float, name: str) -> RunMetrics | None:
    run_dir = LIVE_DIR / name
    if not (run_dir / "summary.json").exists():
        return None
    summary = json.loads((run_dir / "summary.json").read_text())
    if not summary["runs"]:
        return None
    with open(run_dir / "run01.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return run_metrics(stem, tier, replay_time_s, summary, rows)


def aggregate(results: list[RunMetrics]) -> dict:
    finished = [r for r in results if r.finished]
    ratios = [r.time_ratio for r in finished]
    return {
        "maps": len(results),
        "finished": len(finished),
        "finish_rate": round(len(finished) / len(results), 3) if results else 0.0,
        "mean_completion_pct": round(statistics.fmean(r.completion_pct for r in results), 1) if results else 0.0,
        "median_time_ratio": round(statistics.median(ratios), 2) if ratios else None,
        "landings": sum(r.landings for r in results),
        "bumps": sum(r.bumps for r in results),
        "walls": sum(r.walls for r in results),
        "scrapes": sum(r.scrapes for r in results),
        "recovery_steps": sum(r.recovery_steps for r in results),
    }


def format_table(results: list[RunMetrics]) -> str:
    lines = [
        "| Map | Tier | Result | Replay | Ratio | Landings / bumps / walls | Scrapes | Recovery steps |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        result = f"{r.race_time_s:.2f} s" if r.finished else f"{r.completion_pct:.1f}% ({r.termination})"
        ratio = f"{r.time_ratio:.2f}" if r.finished else "-"
        lines.append(
            f"| {r.stem} | {r.tier} | {result} | {r.replay_time_s:.2f} s | {ratio} | "
            f"{r.landings} / {r.bumps} / {r.walls} | {r.scrapes} | {r.recovery_steps} |"
        )
    agg = aggregate(results)
    lines.append(
        f"\n**Finished {agg['finished']}/{agg['maps']}** (finish rate {agg['finish_rate']:.0%}), "
        f"mean completion {agg['mean_completion_pct']}%, median time ratio {agg['median_time_ratio']}, "
        f"contacts {agg['landings']} landings / {agg['bumps']} bumps / {agg['walls']} walls / "
        f"{agg['scrapes']} scrapes, {agg['recovery_steps']} recovery steps."
    )
    return "\n".join(lines)


def campaign_maps(stems: list[str], manifest: dict) -> list[str]:
    """Maps with an official replay (Hockolicious, a dev map, has none)."""

    return [s for s in stems if s in manifest]


def collect(split_key: str, tag: str) -> tuple[list[RunMetrics], list[str]]:
    manifest = json.loads(MANIFEST.read_text())
    results, missing = [], []
    for stem in campaign_maps(load_split()[split_key], manifest):
        entry = manifest[stem]
        metrics = load_run(stem, entry["group"], entry["author_time_s"], run_dir_name(stem, tag, split_key))
        if metrics is None:
            missing.append(stem)
        else:
            results.append(metrics)
    return results, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="run names are <map>_<set>_<tag>")
    parser.add_argument("--set", default="test", choices=("test", "rl_train", "dev"))
    parser.add_argument("--json", type=Path, help="also write per-map metrics + aggregate here")
    args = parser.parse_args()

    results, missing = collect(args.set, args.tag)
    print(format_table(results))
    if missing:
        print(f"\nNot driven yet: {' '.join(missing)}")
    if args.json:
        payload = {"tag": args.tag, "set": args.set, "aggregate": aggregate(results),
                   "maps": [r.__dict__ | {"time_ratio": r.time_ratio} for r in results]}
        args.json.write_text(json.dumps(payload, indent=1) + "\n")


if __name__ == "__main__":
    main()
