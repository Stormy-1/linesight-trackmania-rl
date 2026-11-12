"""Conservative, deterministic centerline controller for offline validation.

The controller deliberately has no TrackMania or RL dependencies.  It consumes
the structured state produced by :mod:`trackmania_rl.hybrid_state` and emits a
bounded abstract action that can later be adapted to the game's input layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from trackmania_rl.hybrid_planner import LocalTrajectoryPlan, LocalTrajectoryPlannerConfig, plan_local_trajectory
from trackmania_rl.hybrid_state import (
    CenterlineGeometry,
    TrackRelativeState,
    curvature_target_speed_mps,
    recovery_required,
    sample_centerline_lookahead,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def governed_target_speed_mps(
    base_target_speed_mps: float,
    current_speed_mps: float,
    curvature: float,
    *,
    control_horizon_s: float,
    reachable_acceleration_mps2: float,
    max_lateral_acceleration_mps2: float,
) -> float:
    """Apply acceleration-reachability and lateral-acceleration speed caps.

    The returned value is an upper-bounded target, rather than a new speed
    profile.  In particular, a target below the current speed is retained so
    the existing brake command can request deceleration.  A zero-curvature
    sample has no lateral cap.
    """

    values = (
        base_target_speed_mps,
        current_speed_mps,
        curvature,
        control_horizon_s,
        reachable_acceleration_mps2,
        max_lateral_acceleration_mps2,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("speed-governor values must be finite")
    if (
        base_target_speed_mps < 0.0
        or current_speed_mps < 0.0
        or control_horizon_s < 0.0
        or reachable_acceleration_mps2 < 0.0
        or max_lateral_acceleration_mps2 <= 0.0
    ):
        raise ValueError("speed-governor values are out of bounds")

    reachable_speed = current_speed_mps + reachable_acceleration_mps2 * control_horizon_s
    if abs(curvature) <= 1e-12:
        lateral_speed = math.inf
    else:
        lateral_speed = math.sqrt(max_lateral_acceleration_mps2 / abs(curvature))
    return float(max(0.0, min(base_target_speed_mps, reachable_speed, lateral_speed)))


@dataclass(frozen=True)
class ControllerCommand:
    """Bounded abstract action plus diagnostics for later adapters."""

    steering: float
    throttle: float
    brake: float
    target_speed_mps: float
    recovery: bool


@dataclass(frozen=True)
class ConservativeControllerConfig:
    lookahead_m: float = 18.0
    lookahead_speed_gain_m_per_mps: float = 0.25
    heading_gain: float = 1.1
    lateral_gain: float = 0.08
    max_speed_mps: float = 70.0
    min_speed_mps: float = 12.0
    curvature_gain: float = 28.0
    throttle_speed_gain: float = 0.08
    recovery_lateral_limit_m: float = 8.0
    recovery_heading_limit_rad: float = math.radians(55.0)


class ConservativeCenterlineController:
    """Track a centerline with bounded steering and speed control."""

    def __init__(self, config: ConservativeControllerConfig | None = None):
        self.config = config or ConservativeControllerConfig()
        if self.config.lookahead_m <= 0 or self.config.max_speed_mps < self.config.min_speed_mps:
            raise ValueError("invalid conservative controller configuration")

    def _target_speed(self, state: TrackRelativeState, target_curvature: float) -> float:
        return curvature_target_speed_mps(
            target_curvature,
            max_speed_mps=self.config.max_speed_mps,
            min_speed_mps=self.config.min_speed_mps,
            curvature_gain=self.config.curvature_gain,
        )

    def command(self, state: TrackRelativeState, geometry: CenterlineGeometry) -> ControllerCommand:
        lookahead = self.config.lookahead_m + self.config.lookahead_speed_gain_m_per_mps * max(state.speed_mps, 0.0)
        target = sample_centerline_lookahead(geometry, state.progress_m, [lookahead])
        target_point = target.points[0]
        target_curvature = float(target.curvatures[0])
        target_speed = self._target_speed(state, target_curvature)
        recovery = recovery_required(
            state,
            lateral_limit_m=self.config.recovery_lateral_limit_m,
            heading_limit_rad=self.config.recovery_heading_limit_rad,
        )

        # Signed horizontal cross-track term: positive steering asks the car to
        # correct the signed lateral error under the state convention.
        target_offset = target_point - state.projection
        target_heading = math.atan2(float(target_offset[2]), float(target_offset[0]))
        current_heading = math.atan2(float(state.tangent[2]), float(state.tangent[0])) + state.heading_error_rad
        aim_error = (target_heading - current_heading + math.pi) % (2 * math.pi) - math.pi
        # ``lateral_error_m`` is negative when the vehicle is to the positive-Z
        # side of a +X tangent.  Positive yaw/steering follows positive Z, so
        # the lateral term must use the same signed convention as the heading
        # term to steer back toward the centerline.
        steering = self.config.heading_gain * aim_error + self.config.lateral_gain * state.lateral_error_m
        steering = _clamp(steering, -1.0, 1.0)

        speed_error = target_speed - state.speed_mps
        if recovery:
            throttle, brake = 0.0, _clamp(0.25 + max(-speed_error, 0.0) * 0.05, 0.0, 1.0)
        else:
            throttle = _clamp(speed_error * self.config.throttle_speed_gain, 0.0, 1.0)
            brake = _clamp(-speed_error * 0.06, 0.0, 1.0)
        return ControllerCommand(steering, throttle, brake, target_speed, recovery)


@dataclass(frozen=True)
class PlannedCenterlineControllerConfig(ConservativeControllerConfig):
    """Configuration for the isolated local-plan tracking experiment."""

    planner_config: LocalTrajectoryPlannerConfig = field(default_factory=LocalTrajectoryPlannerConfig)


class PlannedCenterlineController(ConservativeCenterlineController):
    """Track a planned centerline horizon without changing default behavior."""

    config: PlannedCenterlineControllerConfig

    def __init__(self, config: PlannedCenterlineControllerConfig | None = None):
        active_config = config or PlannedCenterlineControllerConfig()
        if (
            active_config.max_speed_mps != active_config.planner_config.max_speed_mps
            or active_config.min_speed_mps != active_config.planner_config.min_speed_mps
            or active_config.curvature_gain != active_config.planner_config.curvature_gain
        ):
            raise ValueError("controller and planner speed settings must match")
        super().__init__(active_config)

    def _target_index(self, state: TrackRelativeState, plan: LocalTrajectoryPlan) -> int:
        lookahead = self.config.lookahead_m + self.config.lookahead_speed_gain_m_per_mps * max(state.speed_mps, 0.0)
        index = int(np.searchsorted(plan.distances_m, lookahead, side="left"))
        return min(max(index, 0), len(plan.distances_m) - 1)

    def command(self, state: TrackRelativeState, geometry: CenterlineGeometry) -> ControllerCommand:
        plan = plan_local_trajectory(state, geometry, self.config.planner_config)
        target_index = self._target_index(state, plan)
        target_point = plan.points[target_index]
        recovery = recovery_required(
            state,
            lateral_limit_m=self.config.recovery_lateral_limit_m,
            heading_limit_rad=self.config.recovery_heading_limit_rad,
        )
        target_speed = float(
            plan.fallback_target_speeds_mps[target_index]
            if recovery
            else plan.target_speeds_mps[target_index]
        )

        target_offset = target_point - state.projection
        target_heading = math.atan2(float(target_offset[2]), float(target_offset[0]))
        current_heading = math.atan2(float(state.tangent[2]), float(state.tangent[0])) + state.heading_error_rad
        aim_error = (target_heading - current_heading + math.pi) % (2 * math.pi) - math.pi
        steering = self.config.heading_gain * aim_error + self.config.lateral_gain * state.lateral_error_m
        steering = _clamp(steering, -1.0, 1.0)

        speed_error = target_speed - state.speed_mps
        if recovery:
            throttle, brake = 0.0, _clamp(0.25 + max(-speed_error, 0.0) * 0.05, 0.0, 1.0)
        else:
            throttle = _clamp(speed_error * self.config.throttle_speed_gain, 0.0, 1.0)
            brake = _clamp(-speed_error * 0.06, 0.0, 1.0)
        return ControllerCommand(steering, throttle, brake, target_speed, recovery)


@dataclass(frozen=True)
class ConservativeCenterlineSpeedGovernorConfig(ConservativeControllerConfig):
    """Additional limits for the isolated reachable-speed governor variant."""

    control_horizon_s: float = 1.0
    reachable_acceleration_mps2: float = 10.0
    max_lateral_acceleration_mps2: float = 12.0

    @property
    def max_acceleration_mps2(self) -> float:
        """Compatibility spelling for the acceleration reachability limit."""

        return self.reachable_acceleration_mps2

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.control_horizon_s)) or self.control_horizon_s < 0.0:
            raise ValueError("control_horizon_s must be finite and non-negative")
        if not math.isfinite(float(self.reachable_acceleration_mps2)) or self.reachable_acceleration_mps2 < 0.0:
            raise ValueError("reachable_acceleration_mps2 must be finite and non-negative")
        if not math.isfinite(float(self.max_lateral_acceleration_mps2)) or self.max_lateral_acceleration_mps2 <= 0.0:
            raise ValueError("max_lateral_acceleration_mps2 must be finite and positive")


class ConservativeCenterlineSpeedGovernor(ConservativeCenterlineController):
    """Conservative controller variant with a bounded target-speed governor.

    Only target speed generation differs from the default controller.  The
    inherited command path intentionally keeps the existing steering,
    recovery classification, throttle, and brake logic unchanged.
    """

    config: ConservativeCenterlineSpeedGovernorConfig

    def __init__(self, config: ConservativeCenterlineSpeedGovernorConfig | None = None):
        super().__init__(config or ConservativeCenterlineSpeedGovernorConfig())

    def _target_speed(self, state: TrackRelativeState, target_curvature: float) -> float:
        base_target = super()._target_speed(state, target_curvature)
        return governed_target_speed_mps(
            base_target,
            max(state.speed_mps, 0.0),
            target_curvature,
            control_horizon_s=self.config.control_horizon_s,
            reachable_acceleration_mps2=self.config.reachable_acceleration_mps2,
            max_lateral_acceleration_mps2=self.config.max_lateral_acceleration_mps2,
        )


# Descriptive aliases keep the experiment discoverable without creating a
# second implementation or changing the default controller's public name.
ReachabilityConstrainedCenterlineController = ConservativeCenterlineSpeedGovernor
ReachabilityConstrainedControllerConfig = ConservativeCenterlineSpeedGovernorConfig
ConservativeCenterlineControllerSpeedGovernor = ConservativeCenterlineSpeedGovernor
ConservativeCenterlineControllerSpeedGovernorConfig = ConservativeCenterlineSpeedGovernorConfig
SpeedGovernorConfig = ConservativeCenterlineSpeedGovernorConfig


@dataclass(frozen=True)
class RecoveryAwareCenterlineControllerConfig(ConservativeControllerConfig):
    """Optional recovery-only limits for offline controller experiments.

    The inherited conservative configuration is intentionally unchanged.  The
    additional values are used only while :func:`recovery_required` is true;
    nominal tracking therefore follows the default controller exactly.
    """

    recovery_lookahead_m: float = 8.0
    recovery_lookahead_speed_gain_m_per_mps: float = 0.10
    recovery_heading_gain: float = 1.8
    recovery_lateral_gain: float = 0.18
    recovery_brake_bias: float = 0.35
    recovery_brake_speed_gain: float = 0.08

    def __post_init__(self) -> None:
        values = (
            self.recovery_lookahead_m,
            self.recovery_lookahead_speed_gain_m_per_mps,
            self.recovery_heading_gain,
            self.recovery_lateral_gain,
            self.recovery_brake_bias,
            self.recovery_brake_speed_gain,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("recovery controller values must be finite")
        if self.recovery_lookahead_m <= 0.0 or self.recovery_lookahead_speed_gain_m_per_mps < 0.0:
            raise ValueError("recovery lookahead values are out of bounds")
        if self.recovery_heading_gain < 0.0 or self.recovery_lateral_gain < 0.0:
            raise ValueError("recovery steering gains must be non-negative")
        if not 0.0 <= self.recovery_brake_bias <= 1.0 or self.recovery_brake_speed_gain < 0.0:
            raise ValueError("recovery braking values are out of bounds")


class RecoveryAwareCenterlineController(ConservativeCenterlineController):
    """Conservative centerline tracking with an isolated recovery path.

    The controller delegates all non-recovery states to the default
    implementation.  Recovery states use a nearer target and stronger, still
    bounded correction so that the vehicle prioritizes rejoining the centerline
    while applying a conservative brake command.
    """

    config: RecoveryAwareCenterlineControllerConfig

    def __init__(self, config: RecoveryAwareCenterlineControllerConfig | None = None):
        super().__init__(config or RecoveryAwareCenterlineControllerConfig())

    def command(self, state: TrackRelativeState, geometry: CenterlineGeometry) -> ControllerCommand:
        recovery = recovery_required(
            state,
            lateral_limit_m=self.config.recovery_lateral_limit_m,
            heading_limit_rad=self.config.recovery_heading_limit_rad,
        )
        if not recovery:
            # This preserves the default command path and its exact behavior for
            # every state that does not require recovery.
            return super().command(state, geometry)

        lookahead = self.config.recovery_lookahead_m + self.config.recovery_lookahead_speed_gain_m_per_mps * max(
            state.speed_mps, 0.0
        )
        target = sample_centerline_lookahead(geometry, state.progress_m, [lookahead])
        target_point = target.points[0]
        target_curvature = float(target.curvatures[0])
        target_speed = self._target_speed(state, target_curvature)

        target_offset = target_point - state.projection
        target_heading = math.atan2(float(target_offset[2]), float(target_offset[0]))
        current_heading = math.atan2(float(state.tangent[2]), float(state.tangent[0])) + state.heading_error_rad
        aim_error = (target_heading - current_heading + math.pi) % (2 * math.pi) - math.pi
        steering = self.config.recovery_heading_gain * aim_error + self.config.recovery_lateral_gain * state.lateral_error_m
        steering = _clamp(steering, -1.0, 1.0)

        speed_error = target_speed - state.speed_mps
        throttle = 0.0
        brake = _clamp(
            self.config.recovery_brake_bias + max(-speed_error, 0.0) * self.config.recovery_brake_speed_gain,
            0.0,
            1.0,
        )
        return ControllerCommand(steering, throttle, brake, target_speed, True)


# Descriptive aliases make the offline experiment easy to discover while
# retaining one implementation and leaving the default public class intact.
RecoveryAwareController = RecoveryAwareCenterlineController
RecoveryAwareControllerConfig = RecoveryAwareCenterlineControllerConfig
RecoveryCenterlineController = RecoveryAwareCenterlineController
RecoveryCenterlineControllerConfig = RecoveryAwareCenterlineControllerConfig


@dataclass(frozen=True)
class PurePursuitControllerConfig:
    """Live-tuned controller (2026-11-10), built from the first live failures.

    * Steering aims from the car's actual position (pure pursuit), so a large
      drift gets a proportionally large correction.  The older controllers aim
      from the nearest centerline point and correct drift only via a weak
      lateral gain.
    * Target speed comes from a per-map :class:`TrackSpeedProfile` (smoothed
      corners, braking preview, jump floors).
    * Recovery keeps the car moving.  Zero-throttle recovery stopped the car
      dead on A03, A05, A07 and Hockolicious.
    """

    lookahead_m: float = 10.0
    lookahead_time_s: float = 0.45
    steering_gain: float = 2.2
    speed_preview_m: float = 5.0
    throttle_speed_gain: float = 0.25
    brake_overspeed_mps: float = 3.0
    recovery_lateral_limit_m: float = 10.0
    recovery_heading_limit_rad: float = math.radians(55.0)
    recovery_lookahead_m: float = 6.0
    recovery_target_speed_mps: float = 15.0
    recovery_min_speed_mps: float = 8.0
    # Steering authority shrinks with speed: sustained full lock at ~50 m/s
    # started a 0.9 rad ground slide on A01 (v2, 480 m).  Full authority up
    # to the start speed, then linear down to the limit at the end speed.
    steer_limit_start_mps: float = 35.0
    steer_limit_end_mps: float = 60.0
    high_speed_steer_limit: float = 0.5

    def steering_limit(self, speed_mps: float) -> float:
        span = max(self.steer_limit_end_mps - self.steer_limit_start_mps, 1e-6)
        fraction = _clamp((speed_mps - self.steer_limit_start_mps) / span, 0.0, 1.0)
        return 1.0 - fraction * (1.0 - self.high_speed_steer_limit)


class PurePursuitController:
    """Pure-pursuit steering + profile speed tracking + moving recovery."""

    def __init__(self, profile, config: PurePursuitControllerConfig | None = None):
        self.profile = profile
        self.config = config or PurePursuitControllerConfig()

    @staticmethod
    def car_xz(state: TrackRelativeState) -> np.ndarray:
        # lateral_error = t_z*off_x - t_x*off_z, so the horizontal offset is lateral * (t_z, -t_x)
        tangent = np.array([state.tangent[0], state.tangent[2]])
        tangent /= max(np.linalg.norm(tangent), 1e-9)
        normal = np.array([tangent[1], -tangent[0]])
        return np.array([state.projection[0], state.projection[2]]) + state.lateral_error_m * normal

    def command(
        self, state: TrackRelativeState, geometry: CenterlineGeometry, target_speed_offset_mps: float = 0.0
    ) -> ControllerCommand:
        """``target_speed_offset_mps`` is the residual policy's target-speed delta (ignored in recovery)."""

        cfg = self.config
        speed = max(state.speed_mps, 0.0)
        recovery = recovery_required(
            state, lateral_limit_m=cfg.recovery_lateral_limit_m, heading_limit_rad=cfg.recovery_heading_limit_rad
        )
        lookahead = cfg.recovery_lookahead_m if recovery else cfg.lookahead_m + cfg.lookahead_time_s * speed
        target_point = sample_centerline_lookahead(geometry, state.progress_m, [lookahead]).points[0]

        car = self.car_xz(state)
        car_yaw = math.atan2(float(state.tangent[2]), float(state.tangent[0])) + state.heading_error_rad
        bearing = math.atan2(float(target_point[2] - car[1]), float(target_point[0] - car[0]))
        alpha = (bearing - car_yaw + math.pi) % (2 * math.pi) - math.pi
        limit = cfg.steering_limit(speed)
        steering = _clamp(cfg.steering_gain * alpha, -limit, limit)

        if recovery:
            target_speed = cfg.recovery_target_speed_mps
            if speed < cfg.recovery_min_speed_mps:
                return ControllerCommand(steering, 1.0, 0.0, target_speed, True)
        else:
            target_speed = self.profile.target_at(state.progress_m + cfg.speed_preview_m)
            if target_speed_offset_mps != 0.0:
                target_speed = max(0.0, target_speed + target_speed_offset_mps)

        speed_error = target_speed - speed
        if speed_error < -cfg.brake_overspeed_mps:
            return ControllerCommand(steering, 0.0, 1.0, target_speed, recovery)
        throttle = _clamp(speed_error * cfg.throttle_speed_gain, 0.0, 1.0)
        # A jump's required speed is a floor: proportional throttle left A01 v3b 3 m/s
        # under it, so the car landed short (420 m), bounced and slid into a wall.
        if not recovery and speed_error > 0.0 and self._jump_locked(state.progress_m + cfg.speed_preview_m):
            throttle = 1.0
        return ControllerCommand(steering, throttle, 0.0, target_speed, recovery)

    def _jump_locked(self, progress_m: float) -> bool:
        is_locked = getattr(self.profile, "is_jump_locked", None)
        return bool(is_locked(progress_m)) if is_locked else False
