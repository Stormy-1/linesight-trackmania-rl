"""Does the block map explain the logged wall hits?

For every live run, compare how often a hazard block sits at car height in the
car's own grid cell (radius 0) and in a neighbouring cell (radius 1) at:
* wall-hit steps (from ``classify_hybrid_impacts``), versus
* every ordinary driving step (the base rate).

A useful map shows hazards far more often at wall hits than at ordinary steps.

Usage: python scripts/check_block_map_impacts.py [run names...]  (default: every *_pp_v* run)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from classify_hybrid_impacts import LIVE_DIR, classify_impacts

from trackmania_rl.hybrid_block_map import find_campaign_challenge, load_block_map

MAP_STEMS = {"A01": "A01-Race", "A03": "A03-Race", "A05": "A05-Race", "A07": "A07-Race", "A11": "A11-Race"}


def position(row: dict) -> np.ndarray:
    return np.array([float(row["x"]), float(row["y"]), float(row["z"])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*")
    args = parser.parse_args()
    runs = args.runs or sorted(p.name for p in LIVE_DIR.glob("*_pp_v*") if (p / "run01.csv").exists())

    counts = {"wall": Counter(), "base": Counter()}
    names = Counter()
    maps = {}
    for run in runs:
        stem = MAP_STEMS.get(run.split("_")[0])
        if stem is None:
            continue
        if stem not in maps:
            path = find_campaign_challenge(stem)
            if path is None:
                raise SystemExit(f"{stem}.Challenge.Gbx not found; set TMNF_GAME_DIR")
            maps[stem] = load_block_map(path)
        block_map = maps[stem]
        with open(LIVE_DIR / run / "run01.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        wall_times = {x.race_time_s for x in classify_impacts(rows) if x.kind == "wall"}
        for row in rows:
            group = "wall" if float(row["race_time_s"]) in wall_times else "base"
            counts[group]["steps"] += 1
            for radius in (0, 1):
                hits = block_map.hazards_near(position(row), radius_cells=radius)
                counts[group][f"r{radius}"] += int(bool(hits))
                if group == "wall" and radius == 1:
                    names.update({b.name for b in hits})

    print(f"runs: {len(runs)}")
    for group, c in counts.items():
        n = max(c["steps"], 1)
        print(f"{group:5s} steps={c['steps']:6d}  hazard in own cell {c['r0'] / n:6.1%}  within 1 cell {c['r1'] / n:6.1%}")
    print("hazard types near wall hits:", names.most_common(10))


if __name__ == "__main__":
    main()
