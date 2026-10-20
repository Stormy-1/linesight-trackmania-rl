"""Deterministic closed-loop simulator for offline controller validation.

The simulator intentionally models only a small, bounded kinematic slice of a
car: speed changes from throttle/brake, yaw changes from steering, and position
advances along the current heading.  It is useful for validating the complete
state -> controller -> update loop without importing TrackMania, TMInterface,
RL code, or checkpoint artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np

from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineController,
    ControllerCommand,
)
from trackmania_rl.hybrid_diagnostics import (
    identify_divergence_recovery_events,
    summarize_progress_sections,
)
from trackmania_rl.hybrid_state import (
    CenterlineGeometry,
    TrackRelativeState,
    build_centerline_geometry,
    make_track_relative_state,
)


PoseMutator = Callable[
    [int, float, np.ndarray, np.ndarray, float, TrackRelativeState],
    tuple[np.ndarray, np.ndarray, float] | None,
]


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class ClosedLoopConfig:
    """Bounds and timestep for the simple deterministic vehicle model."""

    dt_s: float = 0.1
    max_speed_mps: float = 80.0
    max_acceleration_mps2: float = 10.0
    max_braking_mps2: float = 18.0
    linear_drag_mps2_per_mps: float = 0.05
    max_yaw_rate_rad_s: float = 1.8
    max_steps: int = 3000
    search_radius: int | None = None
    map_match_horizontal_only: bool = True
    finish_progress_tolerance_m: float = 1.0
    finish_distance_limit_m: float = 4.0
    finish_elevation_tolerance_m: float = 8.0
    section_bins: int = 10

    def __post_init__(self) -> None:
        numeric_values = (
            self.dt_s,
            self.max_speed_mps,
            self.max_acceleration_mps2,
            self.max_braking_mps2,
            self.linear_drag_mps2_per_mps,
            self.max_yaw_rate_rad_s,
            self.finish_progress_tolerance_m,
            self.finish_distance_limit_m,
            self.finish_elevation_tolerance_m,
        )
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("simulator configuration values must be finite")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be positive")
        if self.max_acceleration_mps2 < 0.0 or self.max_braking_mps2 < 0.0:
            raise ValueError("acceleration and braking bounds must be non-negative")
        if self.linear_drag_mps2_per_mps < 0.0 or self.max_yaw_rate_rad_s <= 0.0:
            raise ValueError("drag must be non-negative and yaw rate must be positive")
        if (
            not isinstance(self.max_steps, int)
            or isinstance(self.max_steps, bool)
            or self.max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")
        if self.search_radius is not None and self.search_radius < 0:
            raise ValueError("search_radius must be non-negative or None")
        if not isinstance(self.map_match_horizontal_only, bool):
            raise TypeError("map_match_horizontal_only must be a boolean")
        if self.finish_progress_tolerance_m < 0.0 or self.finish_distance_limit_m <= 0.0:
            raise ValueError("finish tolerances must be non-negative and positive")
        if self.finish_elevation_tolerance_m < 0.0:
            raise ValueError("finish_elevation_tolerance_m must be non-negative")
        if not isinstance(self.section_bins, int) or isinstance(self.section_bins, bool) or self.section_bins <= 0:
            raise ValueError("section_bins must be a positive integer")


@dataclass(frozen=True)
class SimulationStep:
    """One observed state and the command applied for the following timestep."""

    step_index: int
    timestamp_s: float
    state: TrackRelativeState
    command: ControllerCommand


@dataclass(frozen=True)
class ClosedLoopResult:
    """Terminal data and observations from one deterministic simulation run."""

    finished: bool
    elevation_tolerant_finished: bool
    termination_reason: str
    elapsed_s: float
    final_state: TrackRelativeState
    final_position: np.ndarray
    final_heading: np.ndarray
    final_speed_mps: float
    track_length_m: float
    section_bins: int
    finish_distance_limit_m: float
    finish_elevation_tolerance_m: float
    steps: tuple[SimulationStep, ...]

    @property
    def progress_gain_m(self) -> float:
        if not self.steps:
            return self.final_state.progress_m
        return self.final_state.progress_m - self.steps[0].state.progress_m

    @property
    def nonmonotonic_progress_steps(self) -> int:
        """Count observed backward progress transitions in the simulator trace."""

        progress = [step.state.progress_m for step in self.steps]
        progress.append(self.final_state.progress_m)
        return sum(current < previous for previous, current in pairwise(progress))

    @property
    def recovery_samples(self) -> int:
        return sum(step.command.recovery for step in self.steps)

    @property
    def max_abs_lateral_error_m(self) -> float:
        errors = [abs(step.state.lateral_error_m) for step in self.steps]
        errors.append(abs(self.final_state.lateral_error_m))
        return max(errors)

    @property
    def max_abs_heading_error_rad(self) -> float:
        errors = [abs(step.state.heading_error_rad) for step in self.steps]
        errors.append(abs(self.final_state.heading_error_rad))
        return max(errors)

    @property
    def max_horizontal_distance_to_centerline_m(self) -> float:
        errors = [step.state.horizontal_distance_to_centerline_m for step in self.steps]
        errors.append(self.final_state.horizontal_distance_to_centerline_m)
        return max(errors)

    @property
    def mean_horizontal_distance_to_centerline_m(self) -> float:
        errors = [step.state.horizontal_distance_to_centerline_m for step in self.steps]
        errors.append(self.final_state.horizontal_distance_to_centerline_m)
        return float(np.mean(np.asarray(errors, dtype=np.float64)))

    @property
    def max_abs_elevation_error_m(self) -> float:
        errors = [abs(step.state.elevation_error_m) for step in self.steps]
        errors.append(abs(self.final_state.elevation_error_m))
        return max(errors)

    @property
    def mean_abs_elevation_error_m(self) -> float:
        errors = [abs(step.state.elevation_error_m) for step in self.steps]
        errors.append(abs(self.final_state.elevation_error_m))
        return float(np.mean(np.asarray(errors, dtype=np.float64)))

    @property
    def strict_3d_finished(self) -> bool:
        """Whether the strict 3D finish gate accepted the run."""

        return self.finished

    @property
    def elevation_tolerant_finish(self) -> bool:
        """Whether the separate horizontal-plus-elevation-tolerance gate passed."""

        return self.elevation_tolerant_finished

    @property
    def progress_complete(self) -> bool:
        return self.final_state.progress_m >= self.track_length_m - 1.0

    @property
    def finish_diagnostic(self) -> dict[str, object]:
        return {
            "strict_3d_finish": self.strict_3d_finished,
            "elevation_tolerant_finish": self.elevation_tolerant_finished,
            "strict_distance_limit_m": self.finish_distance_limit_m,
            "tolerant_horizontal_distance_limit_m": self.finish_distance_limit_m,
            "tolerant_elevation_limit_m": self.finish_elevation_tolerance_m,
        }

    def _diagnostic_records(self) -> list[dict[str, object]]:
        return [
            {
                "step_index": step.step_index,
                "timestamp_s": step.timestamp_s,
                "progress_m": step.state.progress_m,
                "lateral_error_m": step.state.lateral_error_m,
                "heading_error_rad": step.state.heading_error_rad,
                "horizontal_distance_to_centerline_m": step.state.horizontal_distance_to_centerline_m,
                "elevation_error_m": step.state.elevation_error_m,
                "recovery": step.command.recovery,
            }
            for step in self.steps
        ]

    @property
    def divergence_recovery(self) -> dict[str, dict[str, object] | None]:
        return identify_divergence_recovery_events(self._diagnostic_records())

    @property
    def first_divergence_event(self) -> dict[str, object] | None:
        return self.divergence_recovery["first_divergence_event"]

    @property
    def first_divergence(self) -> dict[str, object] | None:
        return self.first_divergence_event

    @property
    def first_recovery_event(self) -> dict[str, object] | None:
        return self.divergence_recovery["first_recovery"]

    @property
    def first_recovery(self) -> dict[str, object] | None:
        return self.first_recovery_event

    @property
    def section_metrics(self) -> list[dict[str, object]]:
        if not self.steps:
            return []
        return summarize_progress_sections(self._diagnostic_records(), self.track_length_m, bins=self.section_bins)


def synthetic_centerline(
    *,
    length_m: float = 180.0,
    spacing_m: float = 2.0,
    amplitude_m: float = 7.0,
    wavelength_m: float = 72.0,
) -> np.ndarray:
    """Create a repeatable 3D centerline with gentle alternating bends."""

    values = (length_m, spacing_m, amplitude_m, wavelength_m)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("synthetic centerline parameters must be finite")
    if length_m <= 0.0 or spacing_m <= 0.0 or wavelength_m <= 0.0 or amplitude_m < 0.0:
        raise ValueError("synthetic centerline parameters are out of bounds")

    x = np.arange(0.0, length_m, spacing_m, dtype=np.float64)
    if len(x) == 0 or x[-1] < length_m:
        x = np.append(x, float(length_m))
    else:
        x[-1] = float(length_m)
    z = amplitude_m * np.sin(2.0 * math.pi * x / wavelength_m)
    return np.column_stack((x, np.zeros_like(x), z))


def synthetic_centerline_geometry(**kwargs: float) -> CenterlineGeometry:
    """Build :class:`CenterlineGeometry` from :func:`synthetic_centerline`."""

    return build_centerline_geometry(synthetic_centerline(**kwargs))


def _validate_pose(
    position: np.ndarray,
    heading: np.ndarray,
    speed_mps: float,
    config: ClosedLoopConfig,
) -> tuple[np.ndarray, float, float]:
    pose = np.asarray(position, dtype=np.float64)
    direction = np.asarray(heading, dtype=np.float64)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError("initial position must contain three finite values")
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("initial heading must contain three finite values")
    horizontal_norm = math.hypot(float(direction[0]), float(direction[2]))
    if horizontal_norm <= 1e-9:
        raise ValueError("initial heading must have a non-zero horizontal component")
    if not math.isfinite(float(speed_mps)) or speed_mps < 0.0 or speed_mps > config.max_speed_mps:
        raise ValueError("initial speed must be within the simulator speed bounds")
    yaw = math.atan2(float(direction[2]), float(direction[0]))
    return pose.copy(), yaw, float(speed_mps)


def _heading_from_yaw(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw), 0.0, math.sin(yaw)], dtype=np.float64)


def _advance_pose(
    position: np.ndarray,
    yaw: float,
    speed_mps: float,
    command: ControllerCommand,
    config: ClosedLoopConfig,
) -> tuple[np.ndarray, float, float]:
    """Apply one bounded kinematic update."""

    throttle = float(np.clip(command.throttle, 0.0, 1.0))
    brake = float(np.clip(command.brake, 0.0, 1.0))
    steering = float(np.clip(command.steering, -1.0, 1.0))
    acceleration = (
        throttle * config.max_acceleration_mps2
        - brake * config.max_braking_mps2
        - config.linear_drag_mps2_per_mps * speed_mps
    )
    next_speed = float(np.clip(speed_mps + acceleration * config.dt_s, 0.0, config.max_speed_mps))
    next_yaw = _wrap_angle(yaw + steering * config.max_yaw_rate_rad_s * config.dt_s)
    # Semi-implicit integration keeps the update bounded and deterministic.
    next_position = position + _heading_from_yaw(next_yaw) * (next_speed * config.dt_s)
    return next_position, next_yaw, next_speed


def simulate_closed_loop(
    geometry: CenterlineGeometry,
    *,
    initial_position: np.ndarray,
    initial_heading: np.ndarray,
    initial_speed_mps: float,
    controller: ConservativeCenterlineController | None = None,
    config: ClosedLoopConfig | None = None,
    _pose_mutator: PoseMutator | None = None,
) -> ClosedLoopResult:
    """Run the controller against a centerline using only offline state helpers.

    Each loop iteration projects the current pose, computes a controller command,
    and applies one bounded kinematic update.  The run terminates when the
    projected progress reaches the end of the centerline within the configured
    finish tolerances, or when ``max_steps`` is reached.

    The private pose hook receives independent pose/state arrays before the
    command. Returning ``None`` leaves the iteration entirely unchanged; a
    returned pose is validated and rematched with the existing search context.
    """

    active_config = config or ClosedLoopConfig()
    active_controller = controller if controller is not None else ConservativeCenterlineController()
    position, yaw, speed_mps = _validate_pose(
        initial_position, initial_heading, initial_speed_mps, active_config
    )
    previous_segment: int | None = None
    observations: list[SimulationStep] = []
    final_state: TrackRelativeState | None = None
    finished = False
    elevation_tolerant_finished = False
    termination_reason = "max_steps"

    for step_index in range(active_config.max_steps):
        heading = _heading_from_yaw(yaw)
        state = make_track_relative_state(
            position,
            heading,
            speed_mps,
            geometry,
            previous_segment=previous_segment,
            search_radius=active_config.search_radius if previous_segment is not None else None,
            horizontal_only=active_config.map_match_horizontal_only,
        )
        if _pose_mutator is not None:
            mutation = _pose_mutator(
                step_index,
                step_index * active_config.dt_s,
                position.copy(),
                heading.copy(),
                speed_mps,
                replace(state, projection=state.projection.copy(), tangent=state.tangent.copy()),
            )
            if mutation is not None:
                position, heading, speed_mps = mutation
                position, yaw, speed_mps = _validate_pose(position, heading, speed_mps, active_config)
                state = make_track_relative_state(
                    position,
                    _heading_from_yaw(yaw),
                    speed_mps,
                    geometry,
                    previous_segment=previous_segment,
                    search_radius=active_config.search_radius if previous_segment is not None else None,
                    horizontal_only=active_config.map_match_horizontal_only,
                )
        command = active_controller.command(state, geometry)
        observations.append(
            SimulationStep(step_index, step_index * active_config.dt_s, state, command)
        )
        position, yaw, speed_mps = _advance_pose(position, yaw, speed_mps, command, active_config)
        final_state = make_track_relative_state(
            position,
            _heading_from_yaw(yaw),
            speed_mps,
            geometry,
            previous_segment=state.segment_index,
            search_radius=active_config.search_radius,
            horizontal_only=active_config.map_match_horizontal_only,
        )
        previous_segment = final_state.segment_index
        finish_progress = geometry.cumulative_lengths[-1] - active_config.finish_progress_tolerance_m
        reached_finish_progress = final_state.progress_m >= finish_progress
        if (
            reached_finish_progress
            and final_state.horizontal_distance_to_centerline_m <= active_config.finish_distance_limit_m
            and abs(final_state.elevation_error_m) <= active_config.finish_elevation_tolerance_m
        ):
            elevation_tolerant_finished = True
        if (
            reached_finish_progress
            and final_state.distance_to_centerline_m <= active_config.finish_distance_limit_m
        ):
            finished = True
            termination_reason = "finished"
            break
        if reached_finish_progress:
            termination_reason = (
                "elevation_tolerant_finish_only"
                if elevation_tolerant_finished
                else "progress_complete_finish_gate_failed"
            )
            break

    if final_state is None:
        # max_steps is validated as positive, so this is defensive only.
        raise RuntimeError("closed-loop simulation produced no state")
    return ClosedLoopResult(
        finished=finished,
        elevation_tolerant_finished=elevation_tolerant_finished,
        termination_reason=termination_reason,
        elapsed_s=len(observations) * active_config.dt_s,
        final_state=final_state,
        final_position=position.copy(),
        final_heading=_heading_from_yaw(yaw),
        final_speed_mps=speed_mps,
        track_length_m=float(geometry.cumulative_lengths[-1]),
        section_bins=active_config.section_bins,
        finish_distance_limit_m=active_config.finish_distance_limit_m,
        finish_elevation_tolerance_m=active_config.finish_elevation_tolerance_m,
        steps=tuple(observations),
    )
