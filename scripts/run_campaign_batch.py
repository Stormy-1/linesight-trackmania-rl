"""Drive every Nations campaign map live on its official line, one map per process.

Resumable: a map whose output folder already has a summary is skipped. If the game
can't be reached (``TMI_Disrupt``: the desktop isn't visible), the batch removes that
map's empty output and stops, so it doesn't burn through the list.

Usage:
    python scripts/run_campaign_batch.py                   # every map not driven yet
    python scripts/run_campaign_batch.py --maps A02-Race B07-Race
    python scripts/run_campaign_batch.py --report          # results table only
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from classify_hybrid_impacts import LIVE_DIR, classify_impacts

MANIFEST = ROOT / "maps" / "official_manifest.json"
RUNNER = ROOT / "scripts" / "run_hybrid_controller_live.py"
GROUP_ORDER = ("White", "Green", "Blue", "Red", "Black")
MIN_RACE_S = 300.0
RACE_TIME_FACTOR = 4.0  # allow 4x the author time before giving up


def run_name(stem: str, tag: str) -> str:
    """Output folder for a map; reuses the short legacy name (``A05_official_v3b``) if it exists."""

    legacy = f"{stem.split('-')[0]}_official_{tag}"
    if (LIVE_DIR / legacy / "summary.json").exists():
        return legacy
    return f"{stem}_official_{tag}"


def drivable_maps(manifest: dict) -> list[str]:
    usable = [k for k, e in manifest.items() if e.get("centerline") and not e.get("suspect")]
    return sorted(usable, key=lambda k: (GROUP_ORDER.index(manifest[k]["group"]), k.lower()))


def load_summary(name: str) -> dict | None:
    path = LIVE_DIR / name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def drive(stem: str, entry: dict, tag: str, speed: float) -> str:
    max_race_s = max(MIN_RACE_S, RACE_TIME_FACTOR * entry["author_time_s"])
    command = [
        sys.executable, str(RUNNER), "--map", stem, "--runs", "1", "--speed", str(speed),
        "--stuck-cutoff-s", "6", "--max-race-s", str(round(max_race_s)), "--name", run_name(stem, tag),
    ]
    print(f"\n=== {stem} (author {entry['author_time_s']} s, limit {max_race_s:.0f} s)", flush=True)
    subprocess.run(command, cwd=ROOT, check=False)
    summary = load_summary(run_name(stem, tag))
    if summary is None or not summary["runs"]:
        return "no summary"
    return summary["runs"][0]["termination"]


def report(manifest: dict, tag: str) -> None:
    rows, finished = [], 0
    for stem in drivable_maps(manifest):
        summary = load_summary(run_name(stem, tag))
        if summary is None:
            continue
        run = summary["runs"][0]
        with open(LIVE_DIR / run_name(stem, tag) / "run01.csv", newline="") as handle:
            impacts = classify_impacts(list(csv.DictReader(handle)))
        walls = sum(1 for i in impacts if i.kind == "wall")
        finished += int(run["finished"])
        result = f"{run['race_time_s']} s" if run["finished"] else f"{run['completion_pct']}% ({run['termination']})"
        rows.append(
            f"| {stem} | {manifest[stem]['group']} | {result} | {manifest[stem]['author_time_s']} s | "
            f"{len(impacts) - walls} / {walls} | {run['recovery_steps']} |"
        )
    print("| Map | Group | Result | Author | Landings / walls | Recovery steps |")
    print("|---|---|---|---|---|---|")
    print("\n".join(rows))
    print(f"\nFinished {finished} of {len(rows)} driven ({len(drivable_maps(manifest))} drivable maps).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", nargs="*", default=None)
    parser.add_argument("--tag", default="v3b", help="run-name suffix: <map>_official_<tag>")
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    if args.report:
        report(manifest, args.tag)
        return
    for stem in args.maps or drivable_maps(manifest):
        if load_summary(run_name(stem, args.tag)) is not None:
            continue
        termination = drive(stem, manifest[stem], args.tag, args.speed)
        print(f"--> {stem}: {termination}", flush=True)
        if termination in ("TMI_Disrupt", "no summary"):
            shutil.rmtree(LIVE_DIR / run_name(stem, args.tag), ignore_errors=True)
            print("Game unreachable (is the desktop visible?). Stopping; rerun to resume.")
            break
    report(manifest, args.tag)


if __name__ == "__main__":
    main()
