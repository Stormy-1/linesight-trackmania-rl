"""Per-map speed profile: smoothed corner limits, braking preview, and jump floors.

Built once per map from the centerline (which is a recorded reference run).

* Corner limits use **horizontal** curvature smoothed over a window.  The raw
  0.5 m point curvature is dominated by noise and vertical bumps (71% of
  Hockolicious' spikes), which pinned the live car at the 12 m/s minimum.
* A backward pass limits speed so the car can brake in time for the next corner.
* Jumps: where the reference flew, its centerline traces a ballistic arc
  ``y(h) = y0 + tan(theta) h - g h^2 / (2 v_h^2)``.  A long section with a
  steady negative ``d2y/dh2`` gives the reference horizontal speed
  ``v_h = sqrt(g / |d2y/dh2|)``.  The car must arrive near that speed or it
  falls short (A07 at 1577 m, Hockolicious at 967 m in the first live runs).
  In-game gravity was measured at ~15.4 m/s^2 from a live free fall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trackmania_rl.hybrid_state import CenterlineGeometry, curvature_target_speed_mps


@dataclass(frozen=True)
class TrackProfileConfig:
    curvature_window_m: float = 15.0
    max_speed_mps: float = 70.0
    min_speed_mps: float = 12.0
    curvature_gain: float = 28.0
    braking_decel_mps2: float = 15.0
    gravity_mps2: float = 15.4
    jump_fit_half_window_m: float = 6.0
    jump_min_length_m: float = 20.0
    jump_speed_range_mps: tuple[float, float] = (15.0, 110.0)
    jump_max_relative_std: float = 0.35
    jump_speed_fraction: float = 1.0
    jump_approach_m: float = 120.0
    # Pace notes: (start_m, end_m, max_speed_mps) caps that override jump floors.
    speed_caps: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class JumpSection:
    start_m: float
    end_m: float
    reference_horizontal_speed_mps: float
    required_speed_mps: float


@dataclass(frozen=True)
class TrackSpeedProfile:
    progress_m: np.ndarray
    horizontal_curvature: np.ndarray
    corner_limit_mps: np.ndarray
    target_speed_mps: np.ndarray
    jumps: tuple[JumpSection, ...]
    jump_locked: np.ndarray | None = None

    def target_at(self, progress_m: float) -> float:
        return float(np.interp(progress_m, self.progress_m, self.target_speed_mps))

    def is_jump_locked(self, progress_m: float) -> bool:
        """True where the target is a jump's required speed (a floor to reach, not a cap)."""

        if self.jump_locked is None:
            return False
        index = min(int(np.searchsorted(self.progress_m, progress_m)), len(self.progress_m) - 1)
        return bool(self.jump_locked[index])


def smoothed_horizontal_curvature(geometry: CenterlineGeometry, window_m: float) -> np.ndarray:
    """Signed horizontal curvature at each centerline point, averaged over ``window_m``."""

    progress = geometry.cumulative_lengths
    yaw = np.unwrap(np.arctan2(geometry.segment_tangents[:, 2], geometry.segment_tangents[:, 0]))
    yaw = np.append(yaw, yaw[-1])  # one value per point
    half = window_m / 2.0
    ahead = np.interp(np.minimum(progress + half, progress[-1]), progress, yaw)
    behind = np.interp(np.maximum(progress - half, 0.0), progress, yaw)
    span = np.minimum(progress + half, progress[-1]) - np.maximum(progress - half, 0.0)
    return (ahead - behind) / np.maximum(span, 1e-6)


def detect_jumps(geometry: CenterlineGeometry, config: TrackProfileConfig) -> tuple[JumpSection, ...]:
    points = geometry.points
    horizontal = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(points[:, 0]), np.diff(points[:, 2])))))
    y = points[:, 1]
    lo_idx = np.searchsorted(horizontal, horizontal - config.jump_fit_half_window_m, side="left")
    hi_idx = np.searchsorted(horizontal, horizontal + config.jump_fit_half_window_m, side="right")
    curvature_y = np.full(len(points), np.nan)
    for i, (lo, hi) in enumerate(zip(lo_idx, hi_idx)):
        if hi - lo > 8:
            curvature_y[i] = 2.0 * np.polyfit(horizontal[lo:hi] - horizontal[i], y[lo:hi], 2)[0]

    slow, fast = config.jump_speed_range_mps
    g = config.gravity_mps2
    in_band = (curvature_y > -g / slow**2) & (curvature_y < -g / fast**2)
    jumps = []
    i = 0
    while i < len(points):
        if not in_band[i]:
            i += 1
            continue
        j = i
        while j < len(points) and in_band[j]:
            j += 1
        section = curvature_y[i:j]
        long_enough = horizontal[j - 1] - horizontal[i] >= config.jump_min_length_m
        steady = np.std(section) / abs(np.mean(section)) < config.jump_max_relative_std
        if long_enough and steady:
            v_h = math.sqrt(g / abs(float(np.median(section))))
            jumps.append(
                JumpSection(
                    start_m=float(geometry.cumulative_lengths[i]),
                    end_m=float(geometry.cumulative_lengths[j - 1]),
                    reference_horizontal_speed_mps=round(v_h, 2),
                    required_speed_mps=round(min(config.jump_speed_fraction * v_h, config.max_speed_mps), 2),
                )
            )
        i = j
    return tuple(jumps)


def build_track_speed_profile(geometry: CenterlineGeometry, config: TrackProfileConfig | None = None) -> TrackSpeedProfile:
    config = config or TrackProfileConfig()
    progress = geometry.cumulative_lengths
    curvature = smoothed_horizontal_curvature(geometry, config.curvature_window_m)
    corner_limit = np.array(
        [
            curvature_target_speed_mps(
                k, max_speed_mps=config.max_speed_mps, min_speed_mps=config.min_speed_mps, curvature_gain=config.curvature_gain
            )
            for k in curvature
        ]
    )
    # Backward pass: never be faster than what can be braked down for what's ahead.
    target = corner_limit.copy()
    for i in range(len(target) - 2, -1, -1):
        ds = progress[i + 1] - progress[i]
        target[i] = min(target[i], math.sqrt(target[i + 1] ** 2 + 2.0 * config.braking_decel_mps2 * ds))

    # Jumps: match the reference speed (floor AND ceiling).  Too slow falls
    # short (A07 1577 m); too fast flies past the landing into the next corner
    # (A05 608 m: 43-46 m/s vs 38.5 reference, landed in the corner and hit the wall).
    jumps = detect_jumps(geometry, config)
    locked = np.zeros(len(target), dtype=bool)
    previous_end = -math.inf
    for jump in jumps:  # sorted by start; an approach never overrides the previous jump (A05 583 m)
        approach_start = max(jump.start_m - config.jump_approach_m, previous_end)
        previous_end = jump.end_m
        window = (progress > approach_start) & (progress <= jump.end_m)
        # Never above the corner limit: A07 v3b took a 50-80 m radius curve at 62 m/s
        # inside a 70 m/s approach and hit the wall (1261 m).  Curve points stay
        # unlocked so the braking pass still slows the car for them.
        target[window] = np.minimum(jump.required_speed_mps, corner_limit[window])
        locked |= window & (corner_limit >= jump.required_speed_mps)
    for start_m, end_m, cap_mps in config.speed_caps:
        capped = (progress >= start_m) & (progress <= end_m)
        before = progress < start_m  # pace notes outrank jump floors, so brake for them even in a locked window
        envelope = np.where(capped, cap_mps, np.inf)
        envelope[before] = np.sqrt(cap_mps**2 + 2.0 * config.braking_decel_mps2 * (start_m - progress[before]))
        lowered = envelope < target
        target = np.minimum(target, envelope)
        locked &= ~lowered
    for i in range(len(target) - 2, -1, -1):  # brake in time for a slower jump; never lower a locked window
        if not locked[i]:
            ds = progress[i + 1] - progress[i]
            target[i] = min(target[i], math.sqrt(target[i + 1] ** 2 + 2.0 * config.braking_decel_mps2 * ds))

    return TrackSpeedProfile(
        progress_m=progress,
        horizontal_curvature=curvature,
        corner_limit_mps=corner_limit,
        target_speed_mps=target,
        jumps=jumps,
        jump_locked=locked,
    )
