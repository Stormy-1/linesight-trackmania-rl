"""Per-point track features for the learned speed model (Step 2).

Everything here is computed from the centerline geometry alone, so the model can
run on any map that has a line. Each feature answers a question a driver asks:
how sharp is the road here, how sharp does it get ahead (and how far), what did
I just come out of (acceleration is limited), am I climbing or on a crest, is the
line a flight arc (a jump), and am I still accelerating from the start.
"""

from __future__ import annotations

import numpy as np

from trackmania_rl.hybrid_state import CenterlineGeometry
from trackmania_rl.hybrid_track_profile import (
    TrackProfileConfig,
    detect_jumps,
    smoothed_horizontal_curvature,
)

CURVATURE_WINDOW_M = 10.0
AHEAD_WINDOWS_M = ((0.0, 15.0), (15.0, 40.0), (40.0, 80.0), (80.0, 140.0), (140.0, 220.0))
BEHIND_WINDOWS_M = ((-40.0, 0.0), (-120.0, -40.0))
SLOPE_WINDOWS_M = ((-5.0, 5.0), (0.0, 30.0), (30.0, 80.0))
CREST_HALF_WINDOW_M = 10.0
JUMP_LOOKAHEAD_M = 200.0
START_CLIP_M = 300.0
# speed_model_v1 was trained with jump features from the old 15.4 m/s^2 gravity constant.
# Keep the features identical so the saved model sees what it was trained on (HANDOVER §70).
FEATURE_GRAVITY_MPS2 = 15.4

FEATURE_NAMES = (
    ("abs_curvature",)
    + tuple(f"max_abs_curvature_ahead_{int(a)}_{int(b)}" for a, b in AHEAD_WINDOWS_M)
    + tuple(f"max_abs_curvature_behind_{int(-b)}_{int(-a)}" for a, b in BEHIND_WINDOWS_M)
    + tuple(f"slope_{int(a)}_{int(b)}" for a, b in SLOPE_WINDOWS_M)
    + ("vertical_curvature", "min_vertical_curvature_ahead_50")
    + ("in_jump", "jump_speed", "distance_to_next_jump", "next_jump_speed")
    + ("distance_from_start",)
)


def _window_max(progress: np.ndarray, values: np.ndarray, lo_m: float, hi_m: float) -> np.ndarray:
    """max(values) over points with progress in [s + lo, s + hi] for every s (clipped at the ends)."""

    lo_idx = np.searchsorted(progress, progress + lo_m, side="left")
    hi_idx = np.searchsorted(progress, progress + hi_m, side="right")
    out = np.empty(len(progress))
    for i, (lo, hi) in enumerate(zip(lo_idx, hi_idx)):
        out[i] = values[lo:hi].max() if hi > lo else values[min(i, len(values) - 1)]
    return out


def _window_slope(progress: np.ndarray, y: np.ndarray, lo_m: float, hi_m: float) -> np.ndarray:
    a = np.clip(progress + lo_m, progress[0], progress[-1])
    b = np.clip(progress + hi_m, progress[0], progress[-1])
    rise = np.interp(b, progress, y) - np.interp(a, progress, y)
    return rise / np.maximum(b - a, 1e-6)


def _vertical_curvature(progress: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d2y/ds2 from a centred difference over +-CREST_HALF_WINDOW_M (crests < 0, dips > 0)."""

    h = CREST_HALF_WINDOW_M
    ahead = _window_slope(progress, y, 0.0, h)
    behind = _window_slope(progress, y, -h, 0.0)
    return (ahead - behind) / h


def _jump_features(progress: np.ndarray, geometry: CenterlineGeometry) -> np.ndarray:
    jumps = detect_jumps(geometry, TrackProfileConfig(gravity_mps2=FEATURE_GRAVITY_MPS2))
    in_jump = np.zeros(len(progress))
    jump_speed = np.zeros(len(progress))
    to_next = np.full(len(progress), JUMP_LOOKAHEAD_M)
    next_speed = np.zeros(len(progress))
    for jump in reversed(jumps):  # later jumps first, so the nearest one ahead wins
        inside = (progress >= jump.start_m) & (progress <= jump.end_m)
        in_jump[inside] = 1.0
        jump_speed[inside] = jump.reference_horizontal_speed_mps
        gap = jump.start_m - progress
        ahead = (gap > 0.0) & (gap < JUMP_LOOKAHEAD_M)
        to_next[ahead] = gap[ahead]
        next_speed[ahead] = jump.reference_horizontal_speed_mps
    return np.stack([in_jump, jump_speed, to_next, next_speed], axis=1)


def track_point_features(geometry: CenterlineGeometry) -> np.ndarray:
    """(N, len(FEATURE_NAMES)) feature matrix, one row per centerline point."""

    progress = geometry.cumulative_lengths
    y = geometry.points[:, 1]
    k = np.abs(smoothed_horizontal_curvature(geometry, CURVATURE_WINDOW_M))
    vertical = _vertical_curvature(progress, y)
    columns = [k]
    columns += [_window_max(progress, k, a, b) for a, b in AHEAD_WINDOWS_M]
    columns += [_window_max(progress, k, a, b) for a, b in BEHIND_WINDOWS_M]
    columns += [_window_slope(progress, y, a, b) for a, b in SLOPE_WINDOWS_M]
    columns += [vertical, -_window_max(progress, -vertical, 0.0, 50.0)]
    features = np.column_stack(columns + [_jump_features(progress, geometry), np.minimum(progress, START_CLIP_M)])
    assert features.shape[1] == len(FEATURE_NAMES)
    return features
