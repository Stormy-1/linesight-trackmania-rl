"""Sort every ``impact_suspect`` in live hybrid logs into hard landings vs wall hits.

There is no collision telemetry, so ``impact_suspect`` (speed drop > 4 m/s in
50 ms) catches both.  A hard landing shows a falling car (vertical speed well
below zero just before) whose vertical speed snaps back toward zero at the
impact.  A bump (a lip or kerb in the road surface) leaves the car on the line
and undeflected: small lateral error, near-zero slip angle, unchanged heading.
Anything else with a horizontal speed loss is called a wall hit: a barrier
deflects the car or sits at the road edge.

Scrapes are the other blind spot of the proxy: grinding along a wall or a
platform edge loses speed steadily (B09 v4: 52 -> 27 m/s over 36 m, never
4 m/s in one step).  A scrape is a sustained deceleration under throttle that
climbing doesn't explain, at the road edge, and not just after a fall.

Usage: python scripts/classify_hybrid_impacts.py A01_pp_v3 A03_pp_v3 ...
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "eval_results" / "hybrid_live"

FALLING_VY_MPS = -6.0  # vertical speed before impact that counts as "was falling"
LANDING_VY_RECOVERY_MPS = 5.0  # vertical speed must jump up by this much at the impact
LOOKBACK_STEPS = 3
BUMP_MAX_LATERAL_M = 3.0  # barriers sit at the road edge, well beyond this
BUMP_MAX_DEFLECTION_RAD = 0.1  # slip angle or heading change around the impact
DEFLECTION_STEPS = 2
GRAVITY_MPS2 = 15.4
SCRAPE_DECEL_MPS2 = 12.0  # unexplained deceleration under throttle (drag alone is a few m/s^2)
SCRAPE_MIN_LATERAL_M = 3.0
SCRAPE_MIN_LOSS_MPS = 5.0
SCRAPE_MIN_SPEED_MPS = 10.0


@dataclass(frozen=True)
class Impact:
    race_time_s: float
    progress_m: float
    speed_mps: float
    speed_drop_mps: float
    vy_before_mps: float
    vy_after_mps: float
    lateral_error_m: float
    elevation_error_m: float
    forward_velocity_angle_rad: float
    kind: str


def _vertical_speeds(rows: list[dict]) -> list[float]:
    vys = [0.0]
    for prev, row in pairwise(rows):
        dt = max(float(row["race_time_s"]) - float(prev["race_time_s"]), 1e-3)
        vys.append((float(row["y"]) - float(prev["y"])) / dt)
    return vys


def _angle(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", "nan") else 0.0


def _is_bump(rows: list[dict], i: int) -> bool:
    window = rows[max(i - 1, 0) : i + DEFLECTION_STEPS + 1]
    slip = max(abs(_angle(r, "forward_velocity_angle_rad")) for r in window)
    turn = abs(_angle(window[-1], "heading_error_rad") - _angle(window[0], "heading_error_rad"))
    on_line = abs(float(rows[i]["lateral_error_m"])) < BUMP_MAX_LATERAL_M
    return on_line and max(slip, turn) < BUMP_MAX_DEFLECTION_RAD


@dataclass(frozen=True)
class Scrape:
    start_m: float
    end_m: float
    speed_loss_mps: float
    max_abs_lateral_m: float


def _unexplained_decel(prev: dict, row: dict) -> float:
    """Speed change per second minus what climbing explains (negative = losing speed)."""

    dt = float(row["race_time_s"]) - float(prev["race_time_s"])
    v0 = float(prev["speed_mps"])
    if dt <= 0.0 or v0 < SCRAPE_MIN_SPEED_MPS:
        return 0.0
    climb = GRAVITY_MPS2 * (float(row["y"]) - float(prev["y"])) / (v0 * dt)
    return (float(row["speed_mps"]) - v0) / dt + climb


def _is_scraping_step(prev: dict, row: dict, vy_before: float) -> bool:
    driving = float(prev["throttle"]) >= 0.5 and float(prev["brake"]) == 0.0 and prev["recovery"] == "False"
    gripping = abs(_angle(prev, "forward_velocity_angle_rad")) < 0.3
    return driving and gripping and vy_before > FALLING_VY_MPS and _unexplained_decel(prev, row) < -SCRAPE_DECEL_MPS2


def find_scrapes(rows: list[dict]) -> list[Scrape]:
    """Runs of scraping steps that cost real speed at the road edge."""

    vys = _vertical_speeds(rows)
    scrapes, run = [], []
    for i in range(1, len(rows) + 1):
        if i < len(rows):
            vy_before = min(vys[max(i - LOOKBACK_STEPS, 1) : i + 1] or [0.0])
            if _is_scraping_step(rows[i - 1], rows[i], vy_before):
                run.append(i)
                continue
        if run:
            first, last = rows[run[0] - 1], rows[run[-1]]
            loss = float(first["speed_mps"]) - float(last["speed_mps"])
            lateral = max(abs(float(rows[j]["lateral_error_m"])) for j in run)
            if loss >= SCRAPE_MIN_LOSS_MPS and lateral >= SCRAPE_MIN_LATERAL_M:
                scrapes.append(Scrape(float(first["progress_m"]), float(last["progress_m"]), round(loss, 1), lateral))
            run = []
    return scrapes


def classify_impacts(rows: list[dict]) -> list[Impact]:
    """Return one :class:`Impact` per logged ``impact_suspect`` row."""

    vys = _vertical_speeds(rows)
    impacts = []
    for i, row in enumerate(rows):
        if row["impact_suspect"] != "True":
            continue
        before = vys[max(i - LOOKBACK_STEPS, 1) : i] or [0.0]
        after = vys[i : i + 2]
        vy_before, vy_after = min(before), max(after)
        landing = vy_before < FALLING_VY_MPS and vy_after - vy_before > LANDING_VY_RECOVERY_MPS
        fva = row["forward_velocity_angle_rad"]
        impacts.append(
            Impact(
                race_time_s=float(row["race_time_s"]),
                progress_m=float(row["progress_m"]),
                speed_mps=float(row["speed_mps"]),
                speed_drop_mps=float(row["speed_drop_mps"]),
                vy_before_mps=round(vy_before, 1),
                vy_after_mps=round(vy_after, 1),
                lateral_error_m=float(row["lateral_error_m"]),
                elevation_error_m=float(row["elevation_error_m"]),
                forward_velocity_angle_rad=float(fva) if fva not in ("", "nan") else float("nan"),
                kind="landing" if landing else ("bump" if _is_bump(rows, i) else "wall"),
            )
        )
    return impacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="folder names under eval_results/hybrid_live")
    args = parser.parse_args()
    print("| Run | Result | Impacts | Landings | Bumps | Wall hits | Wall hits at (m: lat, fva) | Scrapes (m: loss) |")
    print("|---|---|---|---|---|---|---|---|")
    for name in args.runs:
        run_dir = LIVE_DIR / name
        with open(run_dir / "run01.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        impacts = classify_impacts(rows)
        scrapes = ", ".join(f"{x.start_m:.0f}-{x.end_m:.0f} (-{x.speed_loss_mps:.0f})" for x in find_scrapes(rows))
        summary = json.loads((run_dir / "summary.json").read_text())
        walls = [x for x in impacts if x.kind == "wall"]
        where = ", ".join(f"{w.progress_m:.0f} ({w.lateral_error_m:+.1f}, {w.forward_velocity_angle_rad:+.2f})" for w in walls)
        landings = sum(x.kind == "landing" for x in impacts)
        bumps = sum(x.kind == "bump" for x in impacts)
        print(f"| {name} | {_result(summary)} | {len(impacts)} | {landings} | {bumps} | {len(walls)} | {where} | {scrapes} |")


def _result(summary: dict) -> str:
    run = summary["runs"][0]
    if run["finished"]:
        return f"finished {run['race_time_s']} s"
    return f"{run['completion_pct']}%"


if __name__ == "__main__":
    main()
