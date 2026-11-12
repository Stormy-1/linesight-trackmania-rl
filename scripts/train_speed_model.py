"""Train and validate the learned speed model (Step 2) — offline, from the official replays.

Validation is by map (GroupKFold over maps), never by point: neighbouring points of one
map are near-copies, so a point-wise split would measure memorisation. Only
``offline_train`` maps are used for training and validation. The test maps are scored once
at the end, for the record (never used to choose anything).

Baselines on the same folds:
- hand rule: the current profile's corner limit 70 / (1 + 28 |k|) (what the live stack uses);
- linear: least squares on the same features.

Usage: python scripts/train_speed_model.py     (needs eval_results/speed_model/dataset.npz)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_speed_dataset import DATASET, OUT_DIR
from make_map_split import load_split
from trackmania_rl.hybrid_track_profile import TrackProfileConfig
from trackmania_rl.speed_model import save_speed_model, train_speed_model

MODEL_PATH = OUT_DIR / "speed_model_v1.pt"
REPORT_PATH = OUT_DIR / "validation_v1.json"
FOLDS = 5
SEED = 0


def hand_rule(features: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    config = TrackProfileConfig()
    k = features[:, names.index("abs_curvature")]
    return np.clip(config.max_speed_mps / (1.0 + config.curvature_gain * k), config.min_speed_mps, config.max_speed_mps)


def fit_linear(x: np.ndarray, y: np.ndarray):
    design = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return lambda q: np.column_stack([q, np.ones(len(q))]) @ coef


def scores(pred: np.ndarray, y: np.ndarray) -> dict:
    return {"mae_mps": round(float(np.mean(np.abs(pred - y))), 2), "r": round(float(np.corrcoef(pred, y)[0, 1]), 3)}


def map_folds(maps: list[str], folds: int, seed: int) -> list[list[str]]:
    order = np.random.default_rng(seed).permutation(sorted(maps))
    return [list(order[i::folds]) for i in range(folds)]


def main() -> None:
    data = np.load(DATASET)
    x, y, row_map = data["features"], data["target"], data["map"]
    names = tuple(str(n) for n in data["feature_names"])
    split = load_split()
    train_maps = [m for m in split["offline_train"] if m in set(row_map)]
    report = {"folds": [], "feature_names": list(names)}
    for i, held_out in enumerate(map_folds(train_maps, FOLDS, SEED)):
        val = np.isin(row_map, held_out)
        fit = np.isin(row_map, train_maps) & ~val
        model = train_speed_model(x[fit], y[fit], names, seed=SEED)
        fold = {
            "held_out_maps": held_out,
            "mlp": scores(model.predict(x[val]), y[val]),
            "linear": scores(fit_linear(x[fit], y[fit])(x[val]), y[val]),
            "hand_rule": scores(hand_rule(x[val], names), y[val]),
        }
        report["folds"].append(fold)
        print(f"fold {i}: mlp {fold['mlp']}  linear {fold['linear']}  hand {fold['hand_rule']}")
    for key in ("mlp", "linear", "hand_rule"):
        report[f"cv_{key}"] = {m: round(float(np.mean([f[key][m] for f in report["folds"]])), 3) for m in ("mae_mps", "r")}
        print(f"CV {key}: {report[f'cv_{key}']}")

    every_train = np.isin(row_map, train_maps)
    final = train_speed_model(x[every_train], y[every_train], names, seed=SEED)
    save_speed_model(final, MODEL_PATH)
    test = np.isin(row_map, split["test"])
    report["test_once"] = {"mlp": scores(final.predict(x[test]), y[test]), "hand_rule": scores(hand_rule(x[test], names), y[test])}
    print(f"test maps (scored once): {report['test_once']}")
    REPORT_PATH.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {MODEL_PATH} and {REPORT_PATH}")


if __name__ == "__main__":
    main()
