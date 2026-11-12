"""Measure in-game gravity from the official replay ghosts (offline, no game needed).

A car in free flight has a constant vertical acceleration -g. In the 100 ms ghost samples
that shows up as runs of near-identical second differences of y. We take the median over
all such samples on the non-test maps. As a check, it compares the jump detector's
reference speed (``v_h = sqrt(g / |y''|)`` on the centerline arc) with the pro's actual
horizontal speed over the same sections, for the old and the measured constant.

    python scripts/measure_gravity.py      # writes eval_results/gravity_measurement.json
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from build_speed_dataset import SAMPLE_PERIOD_S, replay_path
from make_map_split import MANIFEST, load_split

from trackmania_rl.hybrid_state import build_centerline_geometry
from trackmania_rl.hybrid_track_profile import TrackProfileConfig, detect_jumps
from trackmania_rl.map_loader import gbx_to_raw_pos_list

STEADY_SAMPLES = 5  # a free-flight window: this many second differences ...
STEADY_SPREAD_MPS2 = 2.0  # ... within this spread of each other ...
FALL_RANGE_MPS2 = (-35.0, -15.0)  # ... and their mean in this range
OLD_GRAVITY_MPS2 = 15.4


def free_fall_samples(y: np.ndarray, sample_period_s: float) -> np.ndarray:
    """Vertical accelerations at samples that sit inside a steady free-fall window."""

    accel = np.gradient(np.gradient(y, sample_period_s), sample_period_s)
    half = STEADY_SAMPLES // 2
    picked = []
    for i in range(half, len(accel) - half):
        window = accel[i - half : i + half + 1]
        if window.max() - window.min() < STEADY_SPREAD_MPS2 and FALL_RANGE_MPS2[0] < window.mean() < FALL_RANGE_MPS2[1]:
            picked.append(accel[i])
    return np.array(picked)


def detector_to_pro_ratios(raw: np.ndarray, centerline: np.ndarray, gravity_mps2: float) -> list[float]:
    """Jump detector reference speed / pro horizontal speed at each detected jump's middle."""

    geometry = build_centerline_geometry(centerline)
    pro_h = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(raw[:, 0]), np.diff(raw[:, 2])))))
    pro_speed = np.hypot(np.gradient(raw[:, 0], SAMPLE_PERIOD_S), np.gradient(raw[:, 2], SAMPLE_PERIOD_S))
    line_h = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(centerline[:, 0]), np.diff(centerline[:, 2])))))
    config = dataclasses.replace(TrackProfileConfig(), gravity_mps2=gravity_mps2)
    ratios = []
    for jump in detect_jumps(geometry, config):
        start, end = np.searchsorted(geometry.cumulative_lengths, [jump.start_m, jump.end_m])
        middle_h = 0.5 * (line_h[start] + line_h[min(end, len(line_h) - 1)]) * pro_h[-1] / line_h[-1]
        ratios.append(jump.reference_horizontal_speed_mps / float(np.interp(middle_h, pro_h, pro_speed)))
    return ratios


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    split = load_split()
    falls, ratios = [], {OLD_GRAVITY_MPS2: [], "measured": []}
    loaded = []
    for stem in split["offline_train"]:
        if stem not in manifest:
            continue
        raw = np.array(gbx_to_raw_pos_list(replay_path(stem, manifest[stem]["group"])))
        falls.append(free_fall_samples(raw[:, 1], SAMPLE_PERIOD_S))
        loaded.append((stem, raw))
    samples = np.concatenate(falls)
    gravity = float(-np.median(samples))
    for stem, raw in loaded:
        centerline = np.load(BASE_DIR / "maps" / f"{stem}_official_0.5m_cl.npy")
        ratios[OLD_GRAVITY_MPS2] += detector_to_pro_ratios(raw, centerline, OLD_GRAVITY_MPS2)
        ratios["measured"] += detector_to_pro_ratios(raw, centerline, gravity)
    result = {
        "gravity_mps2": round(gravity, 2),
        "n_samples": int(len(samples)),
        "n_maps": len(loaded),
        "iqr_mps2": [round(float(-q), 2) for q in np.percentile(samples, [75, 25])],
        "detector_over_pro_speed_median": {str(k): round(float(np.median(v)), 3) for k, v in ratios.items()},
        "n_jumps": {str(k): len(v) for k, v in ratios.items()},
    }
    print(json.dumps(result, indent=2))
    (BASE_DIR / "eval_results" / "gravity_measurement.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
