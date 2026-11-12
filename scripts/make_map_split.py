"""Split the Nations campaign maps into dev / held-out test / RL-train / offline-data sets.

The test set is the project's yardstick: ~10 maps that no rule, threshold or model is
ever tuned on. It is a seeded random draw, stratified by tier, and it must contain
both maps the current stack finishes and maps it fails (otherwise an improvement
can't show). Maps already used for tuning are dev maps and can't be drawn.

Usage:
    python scripts/make_map_split.py            # writes maps/map_split.json
    python scripts/make_map_split.py --check    # print the split, write nothing

The output file is meant to be committed and never regenerated: re-drawing the test
set after seeing results would defeat its purpose.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

MANIFEST = ROOT / "maps" / "official_manifest.json"
SPLIT_FILE = ROOT / "maps" / "map_split.json"
SEED = 20261112
STATUS_TAG = "v3b"  # the full-campaign pass (HANDOVER §65) gives "currently finishes"

# Tuned on: the five story maps, plus A02 and B09, which got per-map pace-note experiments.
DEV_MAPS = ("A01-Race", "A03-Race", "A05-Race", "A07-Race", "A11-Race", "A02-Race", "B09-Acrobatic")
DEV_EXTRA = ("Hockolicious",)  # not a campaign map; recorded line, no official replay
# Black (3 maps, 0 finish, two are 5-min endurance maps) stays offline data only: too costly live.
TEST_PER_TIER = {"White": 2, "Green": 3, "Blue": 3, "Red": 2}
RL_TRAIN_PER_TIER = {"White": 2, "Green": 2, "Blue": 1, "Red": 1}
MIN_FAILING_IN_TEST = 3
MIN_FINISHING_IN_TEST = 3
MAX_DRAWS = 1000


def drivable(manifest: dict) -> list[str]:
    """Maps with a usable official-replay centerline."""

    return sorted(k for k, e in manifest.items() if e.get("centerline") and not e.get("suspect"))


def _draw(rng: random.Random, pool: list[str], manifest: dict, per_tier: dict[str, int]) -> list[str]:
    picked = []
    for tier, count in per_tier.items():
        candidates = sorted(m for m in pool if manifest[m]["group"] == tier)
        if len(candidates) < count:
            raise ValueError(f"tier {tier}: need {count} maps, only {len(candidates)} available")
        picked.extend(sorted(rng.sample(candidates, count)))
    return picked


def make_split(manifest: dict, finished: dict[str, bool], seed: int = SEED) -> dict:
    """Pure function: same manifest + status + seed -> same split.

    ``finished`` maps a campaign stem to whether the current stack finishes it.
    The test draw is repeated (same RNG stream, so still deterministic) until it holds
    at least MIN_FAILING_IN_TEST failing and MIN_FINISHING_IN_TEST finishing maps.
    """

    unknown = [m for m in DEV_MAPS if m not in manifest]
    if unknown:
        raise ValueError(f"dev maps missing from manifest: {unknown}")
    maps = drivable(manifest)
    pool = [m for m in maps if m not in DEV_MAPS]
    rng = random.Random(seed)
    for attempt in range(1, MAX_DRAWS + 1):
        test = _draw(rng, pool, manifest, TEST_PER_TIER)
        failing = sum(not finished.get(m, False) for m in test)
        if failing >= MIN_FAILING_IN_TEST and len(test) - failing >= MIN_FINISHING_IN_TEST:
            break
    else:
        raise ValueError("no test draw met the finishing/failing mix")
    rl_train = _draw(rng, [m for m in pool if m not in test], manifest, RL_TRAIN_PER_TIER)
    return {
        "seed": seed,
        "test_draw_attempts": attempt,
        "status_source": f"campaign pass {STATUS_TAG}",
        "dev": list(DEV_MAPS) + list(DEV_EXTRA),
        "test": test,
        "rl_train": rl_train,
        "offline_train": [m for m in maps if m not in test],
        "test_finished_at_split": {m: bool(finished.get(m, False)) for m in test},
        "tiers": {m: manifest[m]["group"] for m in test + rl_train},
    }


def campaign_status(manifest: dict, tag: str = STATUS_TAG) -> dict[str, bool]:
    """Whether each map finished in the campaign pass (missing run = not finished)."""

    from run_campaign_batch import load_summary, run_name

    status = {}
    for stem in drivable(manifest):
        summary = load_summary(run_name(stem, tag))
        status[stem] = bool(summary and summary["runs"] and summary["runs"][0]["finished"])
    return status


def load_split(path: Path = SPLIT_FILE) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="print only, don't write")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split (don't, normally)")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    split = make_split(manifest, campaign_status(manifest))
    for key in ("dev", "test", "rl_train"):
        print(f"{key:9s} ({len(split[key])}): {' '.join(split[key])}")
    print(f"offline_train: {len(split['offline_train'])} maps; test finishing at split: "
          f"{sum(split['test_finished_at_split'].values())}/{len(split['test'])}")
    if args.check:
        return
    if SPLIT_FILE.exists() and not args.force:
        sys.exit(f"{SPLIT_FILE} exists; the test set is fixed. Use --force only if you mean it.")
    SPLIT_FILE.write_text(json.dumps(split, indent=1) + "\n")
    print(f"wrote {SPLIT_FILE}")


if __name__ == "__main__":
    main()
