"""Pure short-horizon planning features for the offline hybrid stack.

This module derives observable map-relative quantities without choosing an
action. It is deliberately separate from the default controller so predictive
features can be audited on saved rollouts before any policy integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trackmania_rl.hybrid_state import (
    CenterlineGeometry,
    TrackRelativeState,
    curvature_target_speed_mps,
    sample_centerline_lookahead,
)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _finite(value: object, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class PredictiveRiskFeatures:
    """Observable local features for predictive speed and safety logic."""

    lookahead_distances_m: np.ndarray
    lookahead_curvatures: np.ndarray
    lookahead_target_speeds_mps: np.ndarray
    max_abs_lookahead_curvature: float
    min_lookahead_target_speed_mps: float
    current_speed_mps: float
    speed_margin_to_min_target_mps: float
    max_abs_tangent_change_rad: float
    lateral_acceleration_demand_mps2: float
    lateral_acceleration_margin_mps2: float
    estimated_braking_distance_m: float
    one_step_forward_alignment_margin_rad: float

    def __post_init__(self) -> None:
        for value, name in (
            (self.max_abs_lookahead_curvature, "max_abs_lookahead_curvature"),
            (self.min_lookahead_target_speed_mps, "min_lookahead_target_speed_mps"),
            (self.current_speed_mps, "current_speed_mps"),
            (self.speed_margin_to_min_target_mps, "speed_margin_to_min_target_mps"),
            (self.max_abs_tangent_change_rad, "max_abs_tangent_change_rad"),
            (self.lateral_acceleration_demand_mps2, "lateral_acceleration_demand_mps2"),
            (self.lateral_acceleration_margin_mps2, "lateral_acceleration_margin_mps2"),
            (self.estimated_braking_distance_m, "estimated_braking_distance_m"),
            (self.one_step_forward_alignment_margin_rad, "one_step_forward_alignment_margin_rad"),
        ):
            _finite(value, name)
        distances = np.asarray(self.lookahead_distances_m, dtype=np.float64)
        curvatures = np.asarray(self.lookahead_curvatures, dtype=np.float64)
        targets = np.asarray(self.lookahead_target_speeds_mps, dtype=np.float64)
        if distances.ndim != 1 or len(distances) == 0 or not np.isfinite(distances).all() or np.any(distances < 0.0):
            raise ValueError("lookahead_distances_m must be finite and non-negative")
        if curvatures.shape != distances.shape or targets.shape != distances.shape:
            raise ValueError("lookahead arrays must have matching one-dimensional shapes")
        if not np.isfinite(curvatures).all() or not np.isfinite(targets).all():
            raise ValueError("lookahead arrays must be finite")


@dataclass(frozen=True)
class LocalTrajectoryPlannerConfig:
    """Configuration for a bounded, centerline-based local plan."""

    horizon_distances_m: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 70.0)
    max_speed_mps: float = 70.0
    min_speed_mps: float = 12.0
    curvature_gain: float = 28.0
    max_acceleration_mps2: float = 10.0
    max_braking_mps2: float = 18.0
    fallback_speed_scale: float = 0.75

    def __post_init__(self) -> None:
        distances = np.asarray(self.horizon_distances_m, dtype=np.float64)
        if distances.ndim != 1 or len(distances) == 0 or not np.isfinite(distances).all():
            raise ValueError("horizon_distances_m must be a non-empty finite sequence")
        if np.any(distances < 0.0) or np.any(np.diff(distances) < 0.0):
            raise ValueError("horizon_distances_m must be non-negative and nondecreasing")
        values = (
            self.max_speed_mps,
            self.min_speed_mps,
            self.curvature_gain,
            self.max_acceleration_mps2,
            self.max_braking_mps2,
            self.fallback_speed_scale,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("local planner configuration values must be finite")
        if self.max_speed_mps < self.min_speed_mps or self.min_speed_mps < 0.0:
            raise ValueError("speed bounds are invalid")
        if self.curvature_gain < 0.0 or self.max_acceleration_mps2 <= 0.0 or self.max_braking_mps2 <= 0.0:
            raise ValueError("speed-planning dynamics are invalid")
        if not 0.0 < self.fallback_speed_scale <= 1.0:
            raise ValueError("fallback_speed_scale must be in (0, 1]")


@dataclass(frozen=True)
class LocalTrajectoryPlan:
    """Short-horizon centerline trajectory and constrained speed profiles."""

    distances_m: np.ndarray
    points: np.ndarray
    tangents: np.ndarray
    curvatures: np.ndarray
    desired_lateral_offsets_m: np.ndarray
    target_speeds_mps: np.ndarray
    fallback_lateral_offsets_m: np.ndarray
    fallback_target_speeds_mps: np.ndarray

    def __post_init__(self) -> None:
        distances = np.asarray(self.distances_m, dtype=np.float64)
        points = np.asarray(self.points, dtype=np.float64)
        tangents = np.asarray(self.tangents, dtype=np.float64)
        curvatures = np.asarray(self.curvatures, dtype=np.float64)
        lateral = np.asarray(self.desired_lateral_offsets_m, dtype=np.float64)
        targets = np.asarray(self.target_speeds_mps, dtype=np.float64)
        fallback_lateral = np.asarray(self.fallback_lateral_offsets_m, dtype=np.float64)
        fallback_targets = np.asarray(self.fallback_target_speeds_mps, dtype=np.float64)
        count = len(distances)
        if distances.ndim != 1 or count == 0 or not np.isfinite(distances).all():
            raise ValueError("plan distances must be a non-empty finite vector")
        if np.any(distances < 0.0) or np.any(np.diff(distances) < 0.0):
            raise ValueError("plan distances must be non-negative and nondecreasing")
        if points.shape != (count, 3) or tangents.shape != (count, 3):
            raise ValueError("plan points and tangents must have shape (N, 3)")
        for values, name in (
            (curvatures, "curvatures"),
            (lateral, "desired_lateral_offsets_m"),
            (targets, "target_speeds_mps"),
            (fallback_lateral, "fallback_lateral_offsets_m"),
            (fallback_targets, "fallback_target_speeds_mps"),
        ):
            if values.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
        if not all(np.isfinite(values).all() for values in (points, tangents, curvatures, lateral, targets, fallback_lateral, fallback_targets)):
            raise ValueError("plan arrays must be finite")
        if np.any(targets < 0.0) or np.any(fallback_targets < 0.0):
            raise ValueError("plan speeds must be non-negative")
        if np.any(fallback_targets > targets + 1e-9):
            raise ValueError("fallback speeds must not exceed normal target speeds")


def _constrain_speed_profile(
    raw_speeds: np.ndarray,
    distances_m: np.ndarray,
    *,
    max_speed_mps: float,
    max_acceleration_mps2: float,
    max_braking_mps2: float,
) -> np.ndarray:
    """Project speed targets onto forward and braking reachability bounds."""

    profile = np.clip(np.asarray(raw_speeds, dtype=np.float64), 0.0, max_speed_mps)
    distances = np.asarray(distances_m, dtype=np.float64)
    for _ in range(max(1, len(profile) * 2)):
        previous = profile.copy()
        for index in range(1, len(profile)):
            distance = max(0.0, float(distances[index] - distances[index - 1]))
            reachable = math.sqrt(max(0.0, profile[index - 1] ** 2 + 2.0 * max_acceleration_mps2 * distance))
            profile[index] = min(profile[index], reachable)
        for index in range(len(profile) - 2, -1, -1):
            distance = max(0.0, float(distances[index + 1] - distances[index]))
            braking_limit = math.sqrt(max(0.0, profile[index + 1] ** 2 + 2.0 * max_braking_mps2 * distance))
            profile[index] = min(profile[index], braking_limit)
        if np.array_equal(profile, previous):
            break
    return profile


def plan_local_trajectory(
    state: TrackRelativeState,
    geometry: CenterlineGeometry,
    config: LocalTrajectoryPlannerConfig | None = None,
) -> LocalTrajectoryPlan:
    """Generate a centerline horizon with constrained normal and fallback speeds."""

    if not isinstance(state, TrackRelativeState):
        raise TypeError("state must be a TrackRelativeState")
    if not isinstance(geometry, CenterlineGeometry):
        raise TypeError("geometry must be a CenterlineGeometry")
    planner_config = config or LocalTrajectoryPlannerConfig()
    lookahead = sample_centerline_lookahead(geometry, state.progress_m, planner_config.horizon_distances_m)
    raw_speeds = np.asarray(
        [
            curvature_target_speed_mps(
                curvature,
                max_speed_mps=planner_config.max_speed_mps,
                min_speed_mps=planner_config.min_speed_mps,
                curvature_gain=planner_config.curvature_gain,
            )
            for curvature in lookahead.curvatures
        ],
        dtype=np.float64,
    )
    target_speeds = _constrain_speed_profile(
        raw_speeds,
        lookahead.distances_m,
        max_speed_mps=planner_config.max_speed_mps,
        max_acceleration_mps2=planner_config.max_acceleration_mps2,
        max_braking_mps2=planner_config.max_braking_mps2,
    )
    fallback_speeds = _constrain_speed_profile(
        raw_speeds * planner_config.fallback_speed_scale,
        lookahead.distances_m,
        max_speed_mps=planner_config.max_speed_mps,
        max_acceleration_mps2=planner_config.max_acceleration_mps2,
        max_braking_mps2=planner_config.max_braking_mps2,
    )
    lateral_offsets = np.zeros_like(lookahead.distances_m)
    return LocalTrajectoryPlan(
        distances_m=lookahead.distances_m.copy(),
        points=lookahead.points.copy(),
        tangents=lookahead.tangents.copy(),
        curvatures=lookahead.curvatures.copy(),
        desired_lateral_offsets_m=lateral_offsets.copy(),
        target_speeds_mps=target_speeds,
        fallback_lateral_offsets_m=lateral_offsets.copy(),
        fallback_target_speeds_mps=fallback_speeds,
    )


def predictive_risk_features(
    state: TrackRelativeState,
    geometry: CenterlineGeometry,
    *,
    lookahead_distances_m: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0),
    max_lateral_acceleration_mps2: float = 12.0,
    max_braking_mps2: float = 18.0,
    max_yaw_rate_rad_s: float = 1.8,
    dt_s: float = 0.1,
    max_speed_mps: float = 70.0,
    min_speed_mps: float = 12.0,
    curvature_gain: float = 28.0,
) -> PredictiveRiskFeatures:
    """Derive a deterministic local risk envelope without selecting actions."""

    if not isinstance(state, TrackRelativeState):
        raise TypeError("state must be a TrackRelativeState")
    if not isinstance(geometry, CenterlineGeometry):
        raise TypeError("geometry must be a CenterlineGeometry")
    distances = tuple(_finite(value, "lookahead distance") for value in lookahead_distances_m)
    if not distances or any(value < 0.0 for value in distances):
        raise ValueError("lookahead_distances_m must be non-empty and non-negative")
    lateral_limit = _finite(max_lateral_acceleration_mps2, "max_lateral_acceleration_mps2")
    braking = _finite(max_braking_mps2, "max_braking_mps2")
    yaw_rate = _finite(max_yaw_rate_rad_s, "max_yaw_rate_rad_s")
    dt = _finite(dt_s, "dt_s")
    current_speed = _finite(state.speed_mps, "state.speed_mps")
    heading_error = _finite(state.heading_error_rad, "state.heading_error_rad")
    configured_max_speed = _finite(max_speed_mps, "max_speed_mps")
    configured_min_speed = _finite(min_speed_mps, "min_speed_mps")
    configured_curvature_gain = _finite(curvature_gain, "curvature_gain")
    if lateral_limit <= 0.0 or braking <= 0.0 or yaw_rate <= 0.0 or dt <= 0.0 or current_speed < 0.0:
        raise ValueError("planning dynamics values are out of bounds")
    if configured_max_speed < configured_min_speed or configured_min_speed < 0.0 or configured_curvature_gain < 0.0:
        raise ValueError("speed-target values are out of bounds")

    lookahead = sample_centerline_lookahead(geometry, state.progress_m, distances)
    curvatures = np.asarray(lookahead.curvatures, dtype=np.float64)
    targets = np.asarray(
        [
            curvature_target_speed_mps(
                curvature,
                max_speed_mps=configured_max_speed,
                min_speed_mps=configured_min_speed,
                curvature_gain=configured_curvature_gain,
            )
            for curvature in curvatures
        ],
        dtype=np.float64,
    )
    current_yaw = math.atan2(float(state.tangent[2]), float(state.tangent[0]))
    tangent_changes = np.asarray(
        [
            abs(_wrap_angle(math.atan2(float(tangent[2]), float(tangent[0])) - current_yaw))
            for tangent in lookahead.tangents
        ],
        dtype=np.float64,
    )
    max_curvature = float(np.max(np.abs(curvatures)))
    min_target = float(np.min(targets))
    lateral_demand = current_speed**2 * max_curvature
    next_abs_heading = max(0.0, abs(heading_error) - yaw_rate * dt)
    alignment_margin = math.pi / 2.0 - next_abs_heading
    braking_distance = current_speed**2 / (2.0 * braking)
    return PredictiveRiskFeatures(
        lookahead_distances_m=np.asarray(distances, dtype=np.float64),
        lookahead_curvatures=curvatures,
        lookahead_target_speeds_mps=targets,
        max_abs_lookahead_curvature=max_curvature,
        min_lookahead_target_speed_mps=min_target,
        current_speed_mps=current_speed,
        speed_margin_to_min_target_mps=min_target - current_speed,
        max_abs_tangent_change_rad=float(np.max(tangent_changes)),
        lateral_acceleration_demand_mps2=lateral_demand,
        lateral_acceleration_margin_mps2=lateral_limit - lateral_demand,
        estimated_braking_distance_m=braking_distance,
        one_step_forward_alignment_margin_rad=alignment_margin,
    )
