"""Bounded short-horizon action evaluation for the offline hybrid stack.

The model is intentionally a small kinematic approximation.  It is useful for
testing the tracking interface and action-selection semantics, not for claiming
TrackMania physics fidelity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from trackmania_rl.hybrid_controller import ControllerCommand
from trackmania_rl.hybrid_planner import LocalTrajectoryPlan, LocalTrajectoryPlannerConfig, plan_local_trajectory
from trackmania_rl.hybrid_state import (
    CenterlineGeometry,
    TrackRelativeState,
    make_track_relative_state,
    recovery_required,
)


@dataclass(frozen=True)
class ShortHorizonMPCConfig:
    """Bounds and costs for constant-action short-horizon predictions."""

    horizon_steps: int = 8
    dt_s: float = 0.1
    max_speed_mps: float = 80.0
    max_acceleration_mps2: float = 10.0
    max_braking_mps2: float = 18.0
    linear_drag_mps2_per_mps: float = 0.05
    max_yaw_rate_rad_s: float = 1.8
    steering_candidates: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    longitudinal_candidates: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.0, 0.5),
        (0.0, 1.0),
    )
    lateral_error_weight: float = 3.0
    heading_error_weight: float = 2.0
    speed_error_weight: float = 0.1
    action_effort_weight: float = 0.05
    planner_config: LocalTrajectoryPlannerConfig = field(default_factory=LocalTrajectoryPlannerConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.horizon_steps, int) or isinstance(self.horizon_steps, bool) or self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be a positive integer")
        numeric_values = (
            self.dt_s,
            self.max_speed_mps,
            self.max_acceleration_mps2,
            self.max_braking_mps2,
            self.linear_drag_mps2_per_mps,
            self.max_yaw_rate_rad_s,
            self.lateral_error_weight,
            self.heading_error_weight,
            self.speed_error_weight,
            self.action_effort_weight,
        )
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("MPC configuration values must be finite")
        if (
            self.dt_s <= 0.0
            or self.max_speed_mps <= 0.0
            or self.max_acceleration_mps2 < 0.0
            or self.max_braking_mps2 < 0.0
            or self.linear_drag_mps2_per_mps < 0.0
            or self.max_yaw_rate_rad_s <= 0.0
        ):
            raise ValueError("MPC dynamics are out of bounds")
        if any(not math.isfinite(float(value)) or not -1.0 <= float(value) <= 1.0 for value in self.steering_candidates):
            raise ValueError("steering candidates must be finite and bounded")
        if not self.steering_candidates:
            raise ValueError("steering_candidates must be non-empty")
        if not self.longitudinal_candidates:
            raise ValueError("longitudinal_candidates must be non-empty")
        for throttle, brake in self.longitudinal_candidates:
            if (
                not math.isfinite(float(throttle))
                or not math.isfinite(float(brake))
                or not 0.0 <= float(throttle) <= 1.0
                or not 0.0 <= float(brake) <= 1.0
                or (throttle > 0.0 and brake > 0.0)
            ):
                raise ValueError("longitudinal candidates must contain bounded, non-conflicting actions")
        if any(float(value) < 0.0 for value in numeric_values[-4:]) or (
            self.lateral_error_weight == 0.0
            and self.heading_error_weight == 0.0
            and self.speed_error_weight == 0.0
        ):
            raise ValueError("MPC costs must be non-negative and include a tracking term")


@dataclass(frozen=True)
class MPCPrediction:
    """Finite track-relative summary of one predicted horizon sample."""

    progress_m: float
    lateral_error_m: float
    heading_error_rad: float
    speed_mps: float


@dataclass(frozen=True)
class MPCDecision:
    """Selected first action and diagnostics from one receding-horizon solve."""

    command: ControllerCommand
    cost: float
    candidate_count: int
    predicted_states: tuple[MPCPrediction, ...]


def _heading_from_yaw(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw), 0.0, math.sin(yaw)], dtype=np.float64)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _pose_from_state(state: TrackRelativeState) -> tuple[np.ndarray, float, float]:
    tangent = np.asarray([state.tangent[0], state.tangent[2]], dtype=np.float64)
    tangent /= np.linalg.norm(tangent)
    position = state.projection.copy()
    position[0] += tangent[1] * state.lateral_error_m
    position[2] -= tangent[0] * state.lateral_error_m
    position[1] += state.elevation_error_m
    yaw = math.atan2(float(tangent[1]), float(tangent[0])) + state.heading_error_rad
    return position, _wrap_angle(yaw), float(state.speed_mps)


def _advance_pose(
    position: np.ndarray,
    yaw: float,
    speed_mps: float,
    steering: float,
    throttle: float,
    brake: float,
    config: ShortHorizonMPCConfig,
) -> tuple[np.ndarray, float, float]:
    acceleration = (
        throttle * config.max_acceleration_mps2
        - brake * config.max_braking_mps2
        - config.linear_drag_mps2_per_mps * speed_mps
    )
    next_speed = float(np.clip(speed_mps + acceleration * config.dt_s, 0.0, config.max_speed_mps))
    next_yaw = _wrap_angle(yaw + steering * config.max_yaw_rate_rad_s * config.dt_s)
    next_position = position + _heading_from_yaw(next_yaw) * (next_speed * config.dt_s)
    return next_position, next_yaw, next_speed


class ShortHorizonMPCController:
    """Choose the first action of the lowest-cost constant action prediction."""

    def __init__(self, config: ShortHorizonMPCConfig | None = None):
        self.config = config or ShortHorizonMPCConfig()

    def _target_speed(self, plan: LocalTrajectoryPlan, initial_progress_m: float, state: TrackRelativeState) -> float:
        distance = max(0.0, state.progress_m - initial_progress_m)
        profile = plan.fallback_target_speeds_mps if recovery_required(state) else plan.target_speeds_mps
        return float(np.interp(distance, plan.distances_m, profile))

    def _evaluate_action(
        self,
        state: TrackRelativeState,
        geometry: CenterlineGeometry,
        plan: LocalTrajectoryPlan,
        steering: float,
        throttle: float,
        brake: float,
    ) -> tuple[float, tuple[MPCPrediction, ...]]:
        position, yaw, speed = _pose_from_state(state)
        predictions: list[MPCPrediction] = []
        cost = 0.0
        for _ in range(self.config.horizon_steps):
            predicted_state = make_track_relative_state(
                position,
                _heading_from_yaw(yaw),
                speed,
                geometry,
                horizontal_only=True,
            )
            target_speed = self._target_speed(plan, state.progress_m, predicted_state)
            cost += (
                self.config.lateral_error_weight * predicted_state.lateral_error_m**2
                + self.config.heading_error_weight * predicted_state.heading_error_rad**2
                + self.config.speed_error_weight * (predicted_state.speed_mps - target_speed) ** 2
                + self.config.action_effort_weight * (steering**2 + throttle**2 + brake**2)
            )
            predictions.append(
                MPCPrediction(
                    progress_m=predicted_state.progress_m,
                    lateral_error_m=predicted_state.lateral_error_m,
                    heading_error_rad=predicted_state.heading_error_rad,
                    speed_mps=predicted_state.speed_mps,
                )
            )
            position, yaw, speed = _advance_pose(
                position,
                yaw,
                speed,
                steering,
                throttle,
                brake,
                self.config,
            )
        return float(cost), tuple(predictions)

    def decision(self, state: TrackRelativeState, geometry: CenterlineGeometry) -> MPCDecision:
        if not isinstance(state, TrackRelativeState):
            raise TypeError("state must be a TrackRelativeState")
        if not isinstance(geometry, CenterlineGeometry):
            raise TypeError("geometry must be a CenterlineGeometry")
        plan = plan_local_trajectory(state, geometry, self.config.planner_config)
        recovery = recovery_required(state)
        target_speed = float(
            plan.fallback_target_speeds_mps[0] if recovery else plan.target_speeds_mps[0]
        )
        best: tuple[float, float, float, float, tuple[MPCPrediction, ...]] | None = None
        candidate_count = 0
        for steering in self.config.steering_candidates:
            for throttle, brake in self.config.longitudinal_candidates:
                candidate_count += 1
                cost, predictions = self._evaluate_action(
                    state,
                    geometry,
                    plan,
                    float(steering),
                    float(throttle),
                    float(brake),
                )
                if not math.isfinite(cost):
                    continue
                if best is None or cost < best[0]:
                    best = (cost, float(steering), float(throttle), float(brake), predictions)
        if best is None:
            raise RuntimeError("MPC produced no finite action candidate")
        cost, steering, throttle, brake, predictions = best
        return MPCDecision(
            command=ControllerCommand(steering, throttle, brake, target_speed, recovery),
            cost=cost,
            candidate_count=candidate_count,
            predicted_states=predictions,
        )

    def command(self, state: TrackRelativeState, geometry: CenterlineGeometry) -> ControllerCommand:
        """Return the first action selected by :meth:`decision`."""

        return self.decision(state, geometry).command
