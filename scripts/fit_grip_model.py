"""Fit the grip model (``trackmania_rl/grip_model.py``) from our own live run logs.

Uses every pure-pursuit run in ``eval_results/hybrid_live`` whose map is NOT in the
held-out test set. Validation: leave-one-map-out over those maps (AUC, log loss,
calibration). The test-map runs are scored once at the end, never used for choices.

    python scripts/fit_grip_model.py            # writes eval_results/grip_model/grip_model_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from trackmania_rl.grip_model import GripSamples, auc, fit_grip_model, log_loss, run_samples, save_grip_model
from trackmania_rl.hybrid_state import build_centerline_geometry
from trackmania_rl.hybrid_track_profile import TrackProfileConfig, smoothed_horizontal_curvature

RISKS = (0.1, 0.2, 0.3, 0.5)
REPORT_SPEEDS = (30, 40, 50, 60, 70)


def map_stem(track: str, known: list[str]) -> str:
    """Old runs use short names (``A01``, ``hock``); map them to the split's stems."""

    if track == "hock":
        return "Hockolicious"
    matches = [stem for stem in known if stem == track or stem.startswith(track + "-")]
    return matches[0] if matches else track


def collect(live_dir: Path, split: dict) -> dict[str, list[GripSamples]]:
    known = sorted(set(split["dev"]) | set(split["test"]) | set(split["offline_train"]) | set(split["rl_train"]))
    window = TrackProfileConfig().curvature_window_m
    by_map: dict[str, list[GripSamples]] = {}
    for run_dir in sorted(live_dir.iterdir()):
        summary_path = run_dir / "summary.json"
        csv_path = run_dir / "run01.csv"
        if not (summary_path.is_file() and csv_path.is_file()):
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("controller") != "purepursuit" or not summary.get("centerline"):
            continue
        geometry = build_centerline_geometry(np.load(BASE_DIR / "maps" / summary["centerline"]))
        curvature = smoothed_horizontal_curvature(geometry, window)
        samples = run_samples(csv_path, geometry.cumulative_lengths, curvature)
        by_map.setdefault(map_stem(summary["track"], known), []).append(samples)
    return by_map


def merge(parts: list[GripSamples]) -> GripSamples:
    return GripSamples(
        np.concatenate([p.demand_mps2 for p in parts]),
        np.concatenate([p.speed_mps for p in parts]),
        np.concatenate([p.wide for p in parts]),
    )


def leave_one_map_out(train: dict[str, GripSamples]) -> dict:
    labels, probs, demand, speed = [], [], [], []
    for held_out, samples in train.items():
        if not len(samples.wide):
            continue
        model = fit_grip_model(merge([s for name, s in train.items() if name != held_out]))
        labels.append(samples.wide)
        probs.append(model.probability(samples.demand_mps2, samples.speed_mps))
        demand.append(samples.demand_mps2)
        speed.append(samples.speed_mps)
    y, p = np.concatenate(labels), np.concatenate(probs)
    calibration = []
    for lo, hi in ((0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)):
        sel = (p >= lo) & (p < hi)
        if sel.any():
            calibration.append({"pred": [lo, hi], "n": int(sel.sum()), "mean_pred": float(p[sel].mean()), "observed": float(y[sel].mean())})
    return {
        "auc": auc(y, p),
        "log_loss": log_loss(y, p),
        "auc_demand_only": auc(y, np.concatenate(demand)),
        "auc_speed_only": auc(y, np.concatenate(speed)),
        "base_rate": float(y.mean()),
        "calibration": calibration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=BASE_DIR / "eval_results" / "grip_model" / "grip_model_v1.json")
    args = parser.parse_args()

    split = json.loads((BASE_DIR / "maps" / "map_split.json").read_text())
    by_map = {name: merge(parts) for name, parts in collect(BASE_DIR / "eval_results" / "hybrid_live", split).items()}
    test = set(split["test"])
    train = {name: s for name, s in by_map.items() if name not in test}
    held = {name: s for name, s in by_map.items() if name in test}
    all_train = merge(list(train.values()))
    print(f"train: {len(train)} maps, {len(all_train.wide)} samples, wide rate {all_train.wide.mean():.3f}")

    cv = leave_one_map_out(train)
    print(f"LOMO AUC {cv['auc']:.3f} (demand only {cv['auc_demand_only']:.3f}, speed only {cv['auc_speed_only']:.3f}), log loss {cv['log_loss']:.4f}")
    for row in cv["calibration"]:
        print(f"  pred {row['pred']}: n={row['n']:6d} mean {row['mean_pred']:.3f} observed {row['observed']:.3f}")

    model = fit_grip_model(all_train)
    print(f"model: logit = {model.intercept:.3f} + {model.demand_coef:.4f}*v^2|k| + {model.speed_coef:.4f}*v")
    limits = {str(r): {str(v): round(model.lateral_limit_mps2(v, r), 1) for v in REPORT_SPEEDS} for r in RISKS}
    for risk, row in limits.items():
        print(f"  a_lat_max at risk {risk}: {row}")

    test_score = None
    if held:
        test_samples = merge(list(held.values()))
        p = model.probability(test_samples.demand_mps2, test_samples.speed_mps)
        test_score = {"maps": sorted(held), "n": int(len(p)), "auc": auc(test_samples.wide, p), "log_loss": log_loss(test_samples.wide, p)}
        print(f"test maps (scored once): AUC {test_score['auc']:.3f} log loss {test_score['log_loss']:.4f} n={test_score['n']}")

    extra = {
        "train_maps": sorted(train),
        "n_samples": int(len(all_train.wide)),
        "lomo": cv,
        "lateral_limit_mps2": limits,
        "test_score_once": test_score,
    }
    save_grip_model(model, args.out, extra)
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
