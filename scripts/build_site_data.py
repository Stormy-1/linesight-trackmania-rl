"""Export the showcase website's data (site/data/*.json) from the logged results.

Every number on the site comes from a file this script reads: per-step run logs and summaries in
eval_results/hybrid_live/, robustness reports, model validations, residual-RL training logs, the
official-replay centerlines in maps/, and the era-1 TensorBoard logs (local, not in git).

    python scripts/build_site_data.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "eval_results" / "hybrid_live"
OUT = ROOT / "site" / "data"

TIERS = {"A": "White", "B": "Green", "C": "Blue", "D": "Red", "E": "Black"}
TEST_MAPS = ["A09-Race", "A15-Speed", "B04-Acrobatic", "B07-Race", "B13-Obstacle",
             "C02-Race", "C10-Acrobatic", "C13-Race", "D04-Race", "D07-Race"]
ROB_TAGS = {"base": "v7_rob", "td3_v2": "v8_rl_rob", "td3_v3": "v9_rl_rob"}
CAMPAIGN_POINTS = 160          # centerline points per map in the campaign grid
ERA1_RUNS = ("hocko_v1", "hocko_v1_2")   # this project's IQN run on Hockolicious, 0 -> 10.68M frames
ERA1_TAG = "eval_race_time_finished_trained_hock"


def _write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


def _summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _centerline_xz(stem: str, points: int | None = None) -> list[list[float]]:
    line = np.load(ROOT / "maps" / f"{stem}_official_0.5m_cl.npy")
    if points:
        idx = np.linspace(0, len(line) - 1, points).astype(int)
        line = line[idx]
    else:
        line = line[::4]   # 2 m spacing is plenty for drawing
    return [[round(float(x), 1), round(float(z), 1)] for x, _, z in line]


def _read_run(csv_path: Path, stride: int = 1) -> dict:
    """Per-step log -> column arrays the site animates (50 ms steps, optionally strided)."""
    cols = {k: [] for k in ("t", "x", "y", "z", "v", "steer", "target", "rec", "imp", "rs", "rv", "prog")}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i % stride:
                continue
            cols["t"].append(round(float(row["race_time_s"]), 2))
            cols["x"].append(round(float(row["x"]), 1))
            cols["y"].append(round(float(row["y"]), 1))
            cols["z"].append(round(float(row["z"]), 1))
            cols["v"].append(round(float(row["speed_mps"]), 1))
            cols["steer"].append(round(float(row["steering"]), 2))
            cols["target"].append(round(float(row["target_speed_mps"]), 1))
            cols["rec"].append(1 if row["recovery"] == "True" else 0)
            cols["imp"].append(1 if row["impact_suspect"] == "True" else 0)
            cols["rs"].append(round(float(row.get("residual_steering") or 0.0), 3))
            cols["rv"].append(round(float(row.get("residual_target_speed_mps") or 0.0), 2))
            cols["prog"].append(round(float(row["progress_m"]), 1))
    return cols


def _run_meta(summary: dict, k: int) -> dict:
    r = summary["runs"][k - 1]
    return {"run": k, "finished": bool(r["finished"]), "time": r["race_time_s"],
            "completion": r["completion_pct"], "termination": r["termination"],
            "impacts": r.get("impact_suspects"), "recovery": r.get("recovery_steps")}


def build_campaign() -> None:
    maps = []
    for d in sorted(LIVE.glob("*_official_v3b")):
        s = _summary(d)
        stem = s["centerline"].replace("_official_0.5m_cl.npy", "")
        r = s["runs"][0]
        maps.append({"map": stem, "tier": TIERS[stem[0]], "finished": bool(r["finished"]),
                     "time": r["race_time_s"], "completion": r["completion_pct"],
                     "cl": _centerline_xz(stem, CAMPAIGN_POINTS)})
    maps.sort(key=lambda m: m["map"].upper())
    finished = sum(m["finished"] for m in maps)
    assert len(maps) == 63 and finished == 27, (len(maps), finished)
    _write("campaign.json", {"maps": maps, "finished": finished, "total": len(maps),
                             "source": "eval_results/hybrid_live/*_official_v3b/summary.json"})


def _paired_run(map_name: str, base_finishes: bool) -> int:
    """First run index k where exactly one of base / td3_v3 finishes, in the wanted direction."""
    base = _summary(LIVE / f"{map_name}_test_{ROB_TAGS['base']}")["runs"]
    rl = _summary(LIVE / f"{map_name}_test_{ROB_TAGS['td3_v3']}")["runs"]
    for k, (b, r) in enumerate(zip(base, rl), start=1):
        if b["finished"] == base_finishes and r["finished"] != base_finishes:
            return k
    raise ValueError(f"no discordant pair on {map_name}")


def _duel(map_name: str, base_finishes: bool) -> dict:
    k = _paired_run(map_name, base_finishes)
    out = {"map": map_name, "tier": TIERS[map_name[0]], "pair": k, "cl": _centerline_xz(map_name)}
    for who in ("base", "td3_v3"):
        d = LIVE / f"{map_name}_test_{ROB_TAGS[who]}"
        out[who] = {**_run_meta(_summary(d), k), **_read_run(d / f"run{k:02d}.csv"),
                    "source": str(d.relative_to(ROOT) / f"run{k:02d}.csv").replace("\\", "/")}
    return out


def build_replays() -> None:
    hero_dir = LIVE / "B07-Race_test_v9_rl_rob"
    k = _paired_run("B07-Race", base_finishes=False)
    hero = {"map": "B07-Race", "label": "td3_v3", **_run_meta(_summary(hero_dir), k),
            **_read_run(hero_dir / f"run{k:02d}.csv"),
            "jumps": _summary(hero_dir)["jumps"],
            "source": f"eval_results/hybrid_live/B07-Race_test_v9_rl_rob/run{k:02d}.csv"}
    _write("replays.json", {"hero": hero,
                            "b07": _duel("B07-Race", base_finishes=False),
                            "c02": _duel("C02-Race", base_finishes=True)})


def build_paired() -> None:
    maps = []
    for m in TEST_MAPS:
        row = {"map": m, "tier": TIERS[m[0]]}
        for who, tag in ROB_TAGS.items():
            runs = _summary(LIVE / f"{m}_test_{tag}")["runs"]
            row[who] = [{"f": int(r["finished"]), "c": r["completion_pct"], "t": r["race_time_s"]} for r in runs]
        maps.append(row)
    rob = json.loads((ROOT / "eval_results" / "test_set_robustness_v7_v9rl.json").read_text())
    rob2 = json.loads((ROOT / "eval_results" / "test_set_robustness_v7_v8rl.json").read_text())
    _write("paired.json", {
        "maps": maps,
        "totals": {"base": rob["compare"]["baseline_total"], "td3_v3": rob["total"],
                   "td3_v2": rob2["total"]},
        "pairs_v3": rob["compare"]["pairs"], "sign_p_v3": rob["compare"]["sign_test_p"],
        "source": "eval_results/test_set_robustness_v7_v9rl.json, *_test_{v7,v8_rl,v9_rl}_rob/summary.json"})


def build_td3() -> None:
    gens = {}
    for g in ("td3_v1", "td3_v2", "td3_v3"):
        path = ROOT / "eval_results" / "residual_rl" / g / "train_log.jsonl"
        eps = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        gens[g] = [{"ep": e["episode"], "map": e["map"], "c": e["completion_pct"],
                    "ret": e["episode_return"], "a": e["mean_abs_residual"],
                    "fin": e["termination"] == "Finished"} for e in eps]
    _write("td3.json", {"gens": gens, "source": "eval_results/residual_rl/td3_v*/train_log.jsonl"})


def _era1_curve() -> list[list[float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not installed: era-1 curve skipped")
        return []
    points = []
    for run in ERA1_RUNS:
        acc = EventAccumulator(str(ROOT / "tensorboard" / run), size_guidance={"scalars": 0})
        acc.Reload()
        points += [[e.step, round(e.value, 2)] for e in acc.Scalars(ERA1_TAG)]
    points.sort()
    return points


def build_era1() -> None:
    bench = {}
    with open(ROOT / "eval_results" / "benchmark_20261014_190540.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["checkpoint"], row["track"])
            entry = bench.setdefault(key, {"runs": 0, "finished": 0})
            entry["runs"] += 1
            entry["finished"] += row["finished"] == "True"
    table = [{"checkpoint": c, "track": t, **v} for (c, t), v in sorted(bench.items())]
    _write("era1.json", {"curve": _era1_curve(), "curve_tag": ERA1_TAG, "benchmark": table,
                         "source": "tensorboard/hocko_v1*, eval_results/benchmark_20261014_190540.csv"})


def build_models() -> None:
    ev = ROOT / "eval_results"
    load = lambda p: json.loads((ev / p).read_text(encoding="utf-8"))  # noqa: E731
    _write("models.json", {"speed": load("speed_model/validation_v1.json"),
                           "grip": load("grip_model/grip_model_v1.json"),
                           "lidar": load("range_sensor/validation_v1.json"),
                           "gravity": load("gravity_measurement.json")})


def main() -> None:
    build_campaign()
    build_replays()
    build_paired()
    build_td3()
    build_era1()
    build_models()


if __name__ == "__main__":
    main()
