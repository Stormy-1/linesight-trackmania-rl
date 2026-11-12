"""Observation vector for the residual RL policy (Step 4).

The residual policy only corrects the classical stack, so it sees what a driver
needs to judge whether the base command is about to go wrong:

* **Track-relative state**: lateral / heading / elevation error, speed, the
  profile's target speed and the speed error, and the slip angle (nose vs velocity).
* **Base command**: the steering and throttle-minus-brake the pure-pursuit
  controller is about to send (the residual is added to these).
* **Lookahead geometry** (precomputed once per map, from the centerline alone,
  in the style of ``speed_features``): signed curvature here and ahead, slope
  ahead, the sharpest crest ahead, the profile target ahead, and the jump state.
* **Virtual lidar**: the 13 ``RangeSensor`` rays (left to right), as range / 60 m.
* **Airborne**: the fraction of wheels touching the ground (TMInterface wheel state).

Every feature is divided by a fixed, physical scale (no statistics fitted on any
map), so the same vector means the same thing on every map, including unseen ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trackmania_rl.hybrid_controller import ControllerCommand
from trackmania_rl.hybrid_state import CenterlineGeometry, TrackRelativeState
from trackmania_rl.hybrid_track_profile import TrackSpeedProfile, smoothed_horizontal_curvature
from trackmania_rl.range_sensor import DEFAULT_MAX_RANGE_M, DEFAULT_RAY_ANGLES_RAD
from trackmania_rl.speed_features import CURVATURE_WINDOW_M, _vertical_curvature, _window_max, _window_slope

# Fixed scales: each feature divided by its scale is roughly in [-1, 1] on normal driving.
LATERAL_SCALE_M = 10.0  # recovery starts at 10 m off the line
ELEVATION_SCALE_M = 8.0  # one block-map level
SPEED_SCALE_MPS = 70.0  # profile top speed
SPEED_ERROR_SCALE_MPS = 20.0
CURVATURE_SCALE = 0.05  # 1 / (20 m radius): a tight hairpin
SLOPE_SCALE = 0.5
VERTICAL_CURVATURE_SCALE = 0.01
JUMP_DISTANCE_SCALE_M = 200.0
JUMP_SPEED_SCALE_MPS = 70.0

CURVATURE_OFFSETS_M = (0.0, 10.0, 25.0, 50.0, 80.0, 120.0)
SLOPE_WINDOWS_M = ((0.0, 30.0), (30.0, 80.0))
CREST_LOOKAHEAD_M = 50.0
TARGET_OFFSETS_M = (20.0, 50.0, 100.0)
WHEEL_COUNT = 4

STATE_FEATURES = (
    "lateral_error",
    "sin_heading_error",
    "cos_heading_error",
    "elevation_error",
    "speed",
    "target_speed",
    "speed_error",
    "slip_angle",
    "base_steering",
    "base_throttle_minus_brake",
)
GEOMETRY_FEATURES = (
    tuple(f"curvature_{int(d)}" for d in CURVATURE_OFFSETS_M)
    + tuple(f"slope_{int(a)}_{int(b)}" for a, b in SLOPE_WINDOWS_M)
    + ("min_vertical_curvature_ahead_50",)
    + tuple(f"target_ahead_{int(d)}" for d in TARGET_OFFSETS_M)
    + ("in_jump", "distance_to_next_jump", "next_jump_speed")
)
RAY_FEATURES = tuple(f"ray_{round(float(np.rad2deg(a))):+d}" for a in DEFAULT_RAY_ANGLES_RAD)
CONTACT_FEATURES = ("ground_contact",)
OBSERVATION_NAMES = STATE_FEATURES + GEOMETRY_FEATURES + RAY_FEATURES + CONTACT_FEATURES
OBSERVATION_SIZE = len(OBSERVATION_NAMES)


@dataclass(frozen=True)
class TrackObservationTable:
    """Per-centerline-point geometry features, computed once per map."""

    progress_m: np.ndarray  # (N,)
    signed_curvature: np.ndarray  # (N,)
    slopes: np.ndarray  # (N, len(SLOPE_WINDOWS_M))
    min_vertical_curvature_ahead: np.ndarray  # (N,)
    jump_starts_m: np.ndarray
    jump_ends_m: np.ndarray
    jump_speeds_mps: np.ndarray


def build_track_observation_table(geometry: CenterlineGeometry, profile: TrackSpeedProfile) -> TrackObservationTable:
    progress = geometry.cumulative_lengths
    y = geometry.points[:, 1]
    vertical = _vertical_curvature(progress, y)
    jumps = profile.jumps
    return TrackObservationTable(
        progress_m=progress,
        signed_curvature=smoothed_horizontal_curvature(geometry, CURVATURE_WINDOW_M),
        slopes=np.stack([_window_slope(progress, y, a, b) for a, b in SLOPE_WINDOWS_M], axis=1),
        min_vertical_curvature_ahead=-_window_max(progress, -vertical, 0.0, CREST_LOOKAHEAD_M),
        jump_starts_m=np.array([j.start_m for j in jumps], dtype=np.float64),
        jump_ends_m=np.array([j.end_m for j in jumps], dtype=np.float64),
        jump_speeds_mps=np.array([j.required_speed_mps for j in jumps], dtype=np.float64),
    )


def _index_at(table: TrackObservationTable, progress_m: float) -> int:
    return min(int(np.searchsorted(table.progress_m, progress_m)), len(table.progress_m) - 1)


def _jump_features(table: TrackObservationTable, progress_m: float) -> tuple[float, float, float]:
    inside = (table.jump_starts_m <= progress_m) & (progress_m <= table.jump_ends_m)
    gaps = table.jump_starts_m - progress_m
    ahead = (gaps > 0.0) & (gaps < JUMP_DISTANCE_SCALE_M)
    if not ahead.any():
        return float(inside.any()), 1.0, 0.0
    nearest = int(np.argmin(np.where(ahead, gaps, np.inf)))
    return float(inside.any()), gaps[nearest] / JUMP_DISTANCE_SCALE_M, table.jump_speeds_mps[nearest] / JUMP_SPEED_SCALE_MPS


def geometry_features(table: TrackObservationTable, profile: TrackSpeedProfile, progress_m: float) -> np.ndarray:
    index = _index_at(table, progress_m)
    curvatures = [
        np.interp(progress_m + d, table.progress_m, table.signed_curvature) / CURVATURE_SCALE for d in CURVATURE_OFFSETS_M
    ]
    slopes = list(table.slopes[index] / SLOPE_SCALE)
    crest = table.min_vertical_curvature_ahead[index] / VERTICAL_CURVATURE_SCALE
    targets = [profile.target_at(progress_m + d) / SPEED_SCALE_MPS for d in TARGET_OFFSETS_M]
    return np.array([*curvatures, *slopes, crest, *targets, *_jump_features(table, progress_m)], dtype=np.float32)


def state_features(
    state: TrackRelativeState, base: ControllerCommand, slip_angle_rad: float
) -> np.ndarray:
    slip = 0.0 if math.isnan(slip_angle_rad) else slip_angle_rad
    return np.array(
        [
            state.lateral_error_m / LATERAL_SCALE_M,
            math.sin(state.heading_error_rad),
            math.cos(state.heading_error_rad),
            state.elevation_error_m / ELEVATION_SCALE_M,
            state.speed_mps / SPEED_SCALE_MPS,
            base.target_speed_mps / SPEED_SCALE_MPS,
            (base.target_speed_mps - state.speed_mps) / SPEED_ERROR_SCALE_MPS,
            slip,
            base.steering,
            base.throttle - base.brake,
        ],
        dtype=np.float32,
    )


def ray_features(ranges_m: np.ndarray | None) -> np.ndarray:
    """Rays as a fraction of the max range; with no sensor every ray reads 'open' (1.0)."""

    if ranges_m is None:
        return np.ones(len(RAY_FEATURES), dtype=np.float32)
    ranges = np.asarray(ranges_m, dtype=np.float32)
    if ranges.shape != (len(RAY_FEATURES),):
        raise ValueError(f"expected {len(RAY_FEATURES)} rays, got shape {ranges.shape}")
    return np.clip(ranges / DEFAULT_MAX_RANGE_M, 0.0, 1.0)


def contact_feature(wheel_contact: np.ndarray | None) -> np.ndarray:
    """Fraction of wheels on the ground (0 = airborne); unknown reads as grounded."""

    if wheel_contact is None:
        return np.ones(1, dtype=np.float32)
    contact = np.asarray(wheel_contact, dtype=np.float32).ravel()
    if contact.shape != (WHEEL_COUNT,):
        raise ValueError(f"expected {WHEEL_COUNT} wheel flags, got shape {contact.shape}")
    return np.array([contact.mean()], dtype=np.float32)


class ResidualObservationBuilder:
    """Assembles the fixed-size observation vector each step (see module docstring)."""

    def __init__(self, geometry: CenterlineGeometry, profile: TrackSpeedProfile):
        self.profile = profile
        self.table = build_track_observation_table(geometry, profile)

    def build(
        self,
        state: TrackRelativeState,
        base: ControllerCommand,
        *,
        slip_angle_rad: float,
        ranges_m: np.ndarray | None,
        wheel_contact: np.ndarray | None,
    ) -> np.ndarray:
        observation = np.concatenate(
            [
                state_features(state, base, slip_angle_rad),
                geometry_features(self.table, self.profile, state.progress_m),
                ray_features(ranges_m),
                contact_feature(wheel_contact),
            ]
        )
        assert observation.shape == (OBSERVATION_SIZE,)
        return observation
