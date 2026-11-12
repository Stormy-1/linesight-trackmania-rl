"""Build the learned-speed-model dataset from the official replays (offline, no game needed).

For every drivable campaign map: track features per centerline point (geometry only,
``trackmania_rl.speed_features``) and the target, the pro's speed at that point.
Pro speed comes from the replay ghost's 100 ms positions (centred difference),
mapped onto the centerline by arclength: the centerline is a 0.5 m resampling of
the same ghost path, so arclength along both lines is the same quantity.

Rows for every map are stored with their map name; the training script decides
which maps it may use (``maps/map_split.json``: never the test maps).

Usage: python scripts/build_speed_dataset.py   ->  eval_results/speed_model/dataset.npz
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_map_split import MANIFEST, drivable
from trackmania_rl.hybrid_block_map import CAMPAIGN_SUBDIR, DEFAULT_GAME_DIR
from trackmania_rl.hybrid_state import build_centerline_geometry
from trackmania_rl.map_loader import gbx_to_raw_pos_list
from trackmania_rl.speed_features import FEATURE_NAMES, track_point_features

OUT_DIR = ROOT / "eval_results" / "speed_model"
DATASET = OUT_DIR / "dataset.npz"
SAMPLE_PERIOD_S = 0.1  # ghost record spacing in the Nations replays
ROW_STRIDE = 4  # keep every 4th 0.5 m point (2 m): neighbouring points are near-duplicates
MAX_LENGTH_MISMATCH = 0.02


def pro_speed_along_path(raw: np.ndarray, sample_period_s: float = SAMPLE_PERIOD_S) -> tuple[np.ndarray, np.ndarray]:
    """(arclength_m, speed_mps) at each ghost sample; speed by centred difference."""

    steps = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    arclength = np.concatenate(([0.0], np.cumsum(steps)))
    speed = np.empty(len(raw))
    speed[1:-1] = (steps[:-1] + steps[1:]) / (2.0 * sample_period_s)
    speed[0], speed[-1] = steps[0] / sample_period_s, steps[-1] / sample_period_s
    return arclength, speed


def pro_speed_on_centerline(raw: np.ndarray, centerline_progress: np.ndarray) -> np.ndarray:
    arclength, speed = pro_speed_along_path(raw)
    scale = arclength[-1] / centerline_progress[-1]
    if abs(scale - 1.0) > MAX_LENGTH_MISMATCH:
        raise ValueError(f"replay path and centerline lengths differ by {abs(scale - 1.0):.1%}")
    return np.interp(centerline_progress * scale, arclength, speed)


def replay_path(stem: str, group: str) -> Path:
    return DEFAULT_GAME_DIR / CAMPAIGN_SUBDIR / "Nations" / group / f"{stem}.Replay.gbx"


def map_rows(stem: str, entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with contextlib.redirect_stdout(io.StringIO()):
        raw = np.array(gbx_to_raw_pos_list(replay_path(stem, entry["group"])))
    geometry = build_centerline_geometry(np.load(ROOT / "maps" / entry["centerline"]))
    features = track_point_features(geometry)
    target = pro_speed_on_centerline(raw, geometry.cumulative_lengths)
    keep = slice(None, None, ROW_STRIDE)
    return features[keep], target[keep], geometry.cumulative_lengths[keep]


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    xs, ys, maps, progress = [], [], [], []
    for stem in drivable(manifest):
        try:
            x, y, s = map_rows(stem, manifest[stem])
        except (ValueError, AssertionError, OSError) as error:
            print(f"skip {stem}: {error}")
            continue
        xs.append(x), ys.append(y), progress.append(s), maps.append(np.full(len(y), stem))
        print(f"{stem}: {len(y)} rows, pro speed median {np.median(y):.1f} m/s, max {y.max():.1f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATASET, features=np.concatenate(xs), target=np.concatenate(ys), map=np.concatenate(maps),
        progress=np.concatenate(progress), feature_names=np.array(FEATURE_NAMES),
    )
    print(f"wrote {DATASET}: {sum(len(y) for y in ys)} rows from {len(ys)} maps")


if __name__ == "__main__":
    main()
