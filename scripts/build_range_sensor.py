"""Learn obstacle shapes from the pro ghosts, then check the virtual range sensor offline.

1. For each non-test map with an official replay, count where the ghost's centre passed
   inside each hazard block (``FootprintModel``). The sum over maps is the sensor's
   shape model, saved to ``eval_results/range_sensor/footprints_v1.npz``.
2. Validation on our own live runs (non-test maps), **leave-one-map-out**: the shapes
   used on a map are learned from every *other* map's ghost. At each driving step the
   sensor scans 24 rays around the car. Questions:
   * Does it see the obstacle we hit? Nearest range at the wall-hit step vs ordinary steps.
   * Does it warn in time? AUC of "nearest range" for "wall hit within the next 0.5 s",
     against the old 32 m-cell ``hazards_near`` flag and |lateral error|.
   * Is it wrong about free space? Share of ordinary steps where the car centre sits in
     "solid" (ideally ~0).
   The test-map runs are scored once at the end (they choose nothing).

    python scripts/build_range_sensor.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from build_speed_dataset import replay_path
from classify_hybrid_impacts import LIVE_DIR, classify_impacts
from fit_grip_model import map_stem
from make_map_split import MANIFEST, load_split

from trackmania_rl.grip_model import auc
from trackmania_rl.hybrid_block_map import find_campaign_challenge, load_block_map
from trackmania_rl.map_loader import gbx_to_raw_pos_list
from trackmania_rl.range_sensor import FootprintModel, RangeSensor

OUT_DIR = BASE_DIR / "eval_results" / "range_sensor"
SURROUND_ANGLES_RAD = tuple(np.deg2rad(np.arange(-180, 180, 15)))  # 24 rays, all round
WARNING_STEPS = 10  # 0.5 s at 50 ms
MIN_SPEED_MPS = 5.0
CONTACT_M = 2.0


def ghost_footprints(stems: list[str], manifest: dict) -> dict[str, FootprintModel]:
    per_map = {}
    for stem in stems:
        if stem not in manifest:
            continue
        challenge = find_campaign_challenge(stem)
        if challenge is None:
            continue
        model = FootprintModel()
        model.add_path(load_block_map(challenge), np.array(gbx_to_raw_pos_list(replay_path(stem, manifest[stem]["group"]))))
        per_map[stem] = model
    return per_map


def total(models: list[FootprintModel]) -> FootprintModel:
    result = FootprintModel()
    for model in models:
        result = result.merged(model)
    return result


def live_runs(split: dict, test: bool) -> dict[str, list[Path]]:
    known = sorted(set(split["dev"]) | set(split["test"]) | set(split["offline_train"]) | set(split["rl_train"]))
    runs: dict[str, list[Path]] = {}
    for run_dir in sorted(LIVE_DIR.iterdir()):
        summary_path = run_dir / "summary.json"
        if not (summary_path.is_file() and (run_dir / "run01.csv").is_file()):
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("controller") != "purepursuit":
            continue
        stem = map_stem(summary["track"], known)
        if (stem in split["test"]) == test and find_campaign_challenge(stem) is not None:
            runs.setdefault(stem, []).append(run_dir)
    return runs


def run_rows(sensor: RangeSensor, run_dir: Path) -> dict[str, np.ndarray]:
    with open(run_dir / "run01.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    wall_times = sorted(i.race_time_s for i in classify_impacts(rows) if i.kind == "wall")
    t = np.array([float(r["race_time_s"]) for r in rows])
    pos = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    if len(t) < 3 or np.any(np.diff(t) <= 0):
        return {}
    heading = np.arctan2(np.gradient(pos[:, 2], t), np.gradient(pos[:, 0], t))
    speed = np.array([float(r["speed_mps"]) for r in rows])
    driving = np.array([r["recovery"] != "True" and r.get("maneuver", "drive") == "drive" for r in rows]) & (speed > MIN_SPEED_MPS)
    nearest = np.full(len(rows), np.nan)
    inside = np.zeros(len(rows), dtype=bool)
    cell_flag = np.zeros(len(rows), dtype=bool)
    for i in np.where(driving)[0]:
        nearest[i] = sensor.scan(pos[i], heading[i], SURROUND_ANGLES_RAD).min()
        _, level, _ = sensor.block_map.cell_of(pos[i])
        inside[i] = bool(sensor.is_solid(np.array([pos[i, 0]]), np.array([pos[i, 2]]), level)[0])
        cell_flag[i] = bool(sensor.block_map.hazards_near(pos[i], radius_cells=0))
    hit_now = np.isin(t, wall_times)
    soon = np.zeros(len(rows), dtype=bool)
    for i in np.where(hit_now)[0]:
        soon[max(0, i - WARNING_STEPS) : i + 1] = True
    lateral = np.abs(np.array([float(r["lateral_error_m"]) for r in rows]))
    return {"nearest": nearest, "inside": inside, "cell": cell_flag, "hit_now": hit_now, "soon": soon, "lateral": lateral, "driving": driving}


def evaluate(runs: dict[str, list[Path]], per_map: dict[str, FootprintModel], all_maps: FootprintModel, leave_out: bool) -> dict:
    parts: dict[str, list] = {}
    for stem, dirs in runs.items():
        model = all_maps.merged(per_map[stem], sign=-1) if (leave_out and stem in per_map) else all_maps
        sensor = RangeSensor(load_block_map(find_campaign_challenge(stem)), model)
        for run_dir in dirs:
            for key, values in run_rows(sensor, run_dir).items():
                parts.setdefault(key, []).append(values)
    d = {key: np.concatenate(values) for key, values in parts.items()}
    ok = d["driving"] & np.isfinite(d["nearest"])
    ordinary = ok & ~d["soon"]
    hits = d["hit_now"] & np.isfinite(d["nearest"])
    soon = d["soon"][ok]
    return {
        "maps": len(runs),
        "driving_steps": int(ok.sum()),
        "wall_hits_scored": int(hits.sum()),
        "nearest_at_hit_median_m": float(np.median(d["nearest"][hits])) if hits.any() else None,
        "hits_within_contact_m": float(np.mean(d["nearest"][hits] <= CONTACT_M)) if hits.any() else None,
        "nearest_ordinary_median_m": float(np.median(d["nearest"][ordinary])),
        "ordinary_within_contact_m": float(np.mean(d["nearest"][ordinary] <= CONTACT_M)),
        "false_solid_rate": float(np.mean(d["inside"][ordinary])),
        "auc_warning_range": auc(soon, -d["nearest"][ok]),
        "auc_warning_cell_flag": auc(soon, d["cell"][ok].astype(float) + 1e-6 * np.random.default_rng(0).random(ok.sum())),
        "auc_warning_lateral": auc(soon, d["lateral"][ok]),
    }


def main() -> None:
    split = load_split()
    manifest = json.loads(MANIFEST.read_text())
    per_map = ghost_footprints(split["offline_train"], manifest)
    all_maps = total(list(per_map.values()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_maps.save(OUT_DIR / "footprints_v1.npz")
    print(f"shapes learned from {len(per_map)} ghosts, {len(all_maps.visits)} hazard types visited")

    report = {"train_lomo": evaluate(live_runs(split, test=False), per_map, all_maps, leave_out=True)}
    report["test_once"] = evaluate(live_runs(split, test=True), per_map, all_maps, leave_out=False)
    for name, metrics in report.items():
        print(name, json.dumps(metrics, indent=1))
    (OUT_DIR / "validation_v1.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
