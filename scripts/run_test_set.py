"""Drive a fixed map set from maps/map_split.json live, one run per map, then print the table.

The same settings for every map: no per-map arguments of any kind (no speed caps,
no pace notes). The only map-dependent value is the time limit, derived from the
official replay time. Resumable: maps that already have a summary are skipped; if the
game can't be reached the batch stops so it doesn't burn through the list.

Usage (from the repo root):
    python scripts/run_test_set.py --tag v4                 # test set -> <map>_test_v4
    python scripts/run_test_set.py --tag v4 --set rl_train
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from classify_hybrid_impacts import LIVE_DIR
from make_map_split import MANIFEST, load_split
from report_test_set import collect, format_table, run_dir_name

RUNNER = ROOT / "scripts" / "run_hybrid_controller_live.py"
COMMON_ARGS = ("--runs", "1", "--speed", "10", "--stuck-cutoff-s", "6")
MIN_RACE_S = 300.0
RACE_TIME_FACTOR = 4.0  # allow 4x the official replay time before giving up


def max_race_s(replay_time_s: float) -> int:
    return round(max(MIN_RACE_S, RACE_TIME_FACTOR * replay_time_s))


def command_for(stem: str, replay_time_s: float, tag: str) -> list[str]:
    return [sys.executable, str(RUNNER), "--map", stem, *COMMON_ARGS,
            "--max-race-s", str(max_race_s(replay_time_s)), "--name", run_dir_name(stem, tag)]


def termination(name: str) -> str:
    path = LIVE_DIR / name / "summary.json"
    if not path.exists():
        return "no summary"
    runs = json.loads(path.read_text())["runs"]
    return runs[0]["termination"] if runs else "no summary"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--set", default="test", choices=("test", "rl_train"))
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    for stem in load_split()[args.set]:
        name = run_dir_name(stem, args.tag)
        if (LIVE_DIR / name / "summary.json").exists():
            continue
        print(f"\n=== {stem} -> {name}", flush=True)
        subprocess.run(command_for(stem, manifest[stem]["author_time_s"], args.tag), cwd=ROOT, check=False)
        outcome = termination(name)
        print(f"--> {stem}: {outcome}", flush=True)
        if outcome in ("TMI_Disrupt", "no summary"):
            shutil.rmtree(LIVE_DIR / name, ignore_errors=True)
            print("Game unreachable (is the desktop visible?). Stopping; rerun to resume.")
            break
    results, _ = collect(args.set, args.tag)
    print(format_table(results))


if __name__ == "__main__":
    main()
