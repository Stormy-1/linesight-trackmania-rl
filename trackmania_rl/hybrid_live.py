"""Live adapter: lets a hybrid controller drive TrackMania through the rollout loop.

The game loop asks ``exploration_policy(frame, floats)`` for a discrete action
every 50 ms.  :class:`HybridLivePolicy` satisfies that interface.  Just before
each call the loop hands it the raw simulation state through
``observe_simulation_state``; the policy turns that into a
:class:`TrackRelativeState`, asks the controller for a continuous command, and
:class:`DiscreteActionAdapter` converts the command into one of the 12 digital
key combinations in ``config_files.inputs_list``.

Steering sign was measured on real A01 telemetry (action-aligned capture
2026-11-03): holding RIGHT increases the world heading angle ``atan2(z, x)``
and LEFT decreases it.  The controllers use the same convention (positive
steering = increase that angle), so positive steering maps to RIGHT.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from config_files.inputs_list import inputs as ACTION_INPUTS
from trackmania_rl.hybrid_controller import ControllerCommand
from trackmania_rl.hybrid_state import CenterlineGeometry, TrackRelativeState, build_centerline_geometry, make_track_relative_state


class Controller(Protocol):
    def command(self, state: TrackRelativeState, geometry: CenterlineGeometry) -> ControllerCommand: ...


def action_index(*, accelerate: bool, brake: bool, left: bool, right: bool) -> int:
    """Look up the discrete action index for a key combination."""

    wanted = {"accelerate": accelerate, "brake": brake, "left": left, "right": right}
    for index, keys in enumerate(ACTION_INPUTS):
        if keys == wanted:
            return index
    raise ValueError(f"no discrete action for {wanted}")


@dataclass(frozen=True)
class DiscreteActionAdapterConfig:
    steer_deadband: float = 0.05
    brake_threshold: float = 0.5


class DiscreteActionAdapter:
    """Sigma-delta modulation of continuous steering/throttle onto digital keys.

    Over several steps the fraction of steps a key is held matches the
    commanded magnitude, which is the closest a keyboard-only interface gets
    to analog input.
    """

    def __init__(self, config: DiscreteActionAdapterConfig | None = None):
        self.config = config or DiscreteActionAdapterConfig()
        self.reset()

    def reset(self) -> None:
        self._steer_acc = 0.0
        self._throttle_acc = 0.0

    def to_action_index(self, command: ControllerCommand) -> int:
        steering = float(np.clip(command.steering, -1.0, 1.0))
        if abs(steering) < self.config.steer_deadband:
            steering = 0.0
            self._steer_acc *= 0.5
        self._steer_acc += steering
        left = right = False
        if self._steer_acc >= 0.5:
            right = True
            self._steer_acc -= 1.0
        elif self._steer_acc <= -0.5:
            left = True
            self._steer_acc += 1.0

        if command.brake >= self.config.brake_threshold:
            self._throttle_acc = 0.0
            return action_index(accelerate=False, brake=True, left=left, right=right)
        self._throttle_acc += float(np.clip(command.throttle, 0.0, 1.0))
        accelerate = self._throttle_acc >= 0.5
        if accelerate:
            self._throttle_acc -= 1.0
        return action_index(accelerate=accelerate, brake=False, left=left, right=right)


LOG_FIELDS = (
    "race_time_s", "x", "y", "z", "speed_mps", "progress_m", "lateral_error_m", "heading_error_rad",
    "elevation_error_m", "heading_source", "forward_velocity_angle_rad", "steering", "throttle", "brake",
    "target_speed_mps", "recovery", "action_idx", "speed_drop_mps", "impact_suspect", "maneuver",
)


@dataclass(frozen=True)
class HybridLivePolicyConfig:
    # Velocity is a reliable heading only when the car actually moves horizontally.
    min_horizontal_speed_for_velocity_heading_mps: float = 5.0
    # Temporal map matching window (centerline points are ~0.5 m apart).
    search_radius_segments: int = 200
    # A drop larger than this between two 50 ms decisions (~80 m/s^2) is far
    # beyond braking and is logged as a suspected wall/obstacle impact.
    impact_speed_drop_mps: float = 4.0
    # Unstick: pinned against something (A11 at 273 m: full throttle, ~0 m/s).
    # Brake held at standstill reverses in TrackMania; steering is mirrored so
    # the nose swings back toward the track.
    unstick_enabled: bool = True
    stuck_speed_mps: float = 2.0
    stuck_time_s: float = 0.6
    reverse_time_s: float = 1.0
    unstick_grace_s: float = 2.0
    # Slip guard: when the nose and the velocity diverge, stop following the
    # line and steer gently with steering = -gain * angle.  Live evidence
    # (2026-11-10): full lock with the opposite sign (+angle) spun A01 at
    # 485 m and A07 at 1600 m to ~3.1 rad.  The unguarded v1/v2 runs, which
    # happened to steer mildly with -angle, held the slide (A01 v2 at -0.9 rad,
    # A07 v1 at +1.3 rad) and finished.  A one-step key-response statistic
    # suggested the other sign: it measures the transient tyre response, not
    # how the slide develops.  Hysteresis avoids flapping at the threshold.
    slip_guard_enabled: bool = True
    slip_enter_rad: float = 0.3
    slip_exit_rad: float = 0.15
    slip_steer_gain: float = 0.8
    slip_steer_limit: float = 0.5
    slip_min_speed_mps: float = 10.0


class HybridLivePolicy:
    """``exploration_policy`` that drives with a hybrid controller."""

    def __init__(
        self,
        centerline_points: np.ndarray,
        controller: Controller,
        *,
        adapter: DiscreteActionAdapter | None = None,
        config: HybridLivePolicyConfig | None = None,
    ):
        self.geometry = build_centerline_geometry(centerline_points)
        self.controller = controller
        self.adapter = adapter or DiscreteActionAdapter()
        self.config = config or HybridLivePolicyConfig()
        self._log_file = None
        self._log_writer = None
        self.reset()

    # --- lifecycle -------------------------------------------------------
    def reset(self, log_path: Path | None = None) -> None:
        self.close()
        self.adapter.reset()
        self._observation: dict | None = None
        self._previous_segment: int | None = None
        self._previous_speed: float | None = None
        self.steps = 0
        self.impact_suspects = 0
        self.recovery_steps = 0
        self.max_progress_m = 0.0
        self.reversals = 0
        self._stuck_since: float | None = None
        self._reverse_until = -math.inf
        self._slipping = False
        self.slip_steps = 0
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log_path, "w", newline="")  # noqa: SIM115 - lives across rollout steps, closed in close()
            self._log_writer = csv.DictWriter(self._log_file, fieldnames=LOG_FIELDS)
            self._log_writer.writeheader()

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
        self._log_file = None
        self._log_writer = None

    # --- rollout-loop interface ------------------------------------------
    def observe_simulation_state(
        self, *, race_time_s: float, position: np.ndarray, velocity: np.ndarray, orientation: np.ndarray
    ) -> None:
        self._observation = {
            "race_time_s": float(race_time_s),
            "position": np.asarray(position, dtype=np.float64),
            "velocity": np.asarray(velocity, dtype=np.float64),
            "orientation": np.asarray(orientation, dtype=np.float64),
        }

    def __call__(self, frame, floats) -> tuple[int, bool, float, np.ndarray]:
        if self._observation is None:
            raise RuntimeError("HybridLivePolicy needs observe_simulation_state() before each action")
        action, _ = self.decide(**self._observation)
        self._observation = None
        return action, True, 0.0, np.zeros(len(ACTION_INPUTS), dtype=np.float32)

    # --- core decision ---------------------------------------------------
    def heading(self, velocity: np.ndarray, orientation: np.ndarray) -> tuple[np.ndarray, str, float]:
        """Return (heading, source, angle between car-forward and velocity)."""

        forward = orientation[2]  # rows of the world->car matrix are car axes; z is forward
        horizontal_speed = math.hypot(velocity[0], velocity[2])
        angle = math.nan
        if horizontal_speed > 1e-6 and math.hypot(forward[0], forward[2]) > 1e-6:
            angle = math.atan2(
                forward[2] * velocity[0] - forward[0] * velocity[2], forward[0] * velocity[0] + forward[2] * velocity[2]
            )
        rolling_backward = forward[0] * velocity[0] + forward[2] * velocity[2] < 0.0
        if horizontal_speed >= self.config.min_horizontal_speed_for_velocity_heading_mps and not rolling_backward:
            return velocity / np.linalg.norm(velocity), "velocity", angle
        return forward, "forward_axis", angle

    def decide(
        self, *, race_time_s: float, position: np.ndarray, velocity: np.ndarray, orientation: np.ndarray
    ) -> tuple[int, ControllerCommand]:
        speed = float(np.linalg.norm(velocity))
        heading, source, forward_velocity_angle = self.heading(velocity, orientation)
        state = make_track_relative_state(
            position,
            heading,
            speed,
            self.geometry,
            previous_segment=self._previous_segment,
            search_radius=None if self._previous_segment is None else self.config.search_radius_segments,
            horizontal_only=True,
        )
        self._previous_segment = state.segment_index
        command = self.controller.command(state, self.geometry)
        slipping = self._update_slip(speed, forward_velocity_angle)
        if slipping:
            limit = self.config.slip_steer_limit
            counter = float(np.clip(-self.config.slip_steer_gain * forward_velocity_angle, -limit, limit))
            command = replace(command, steering=counter)
        action = self.adapter.to_action_index(command)
        maneuver = self._unstick(race_time_s, speed, command)
        if maneuver == "drive" and slipping:
            maneuver = "slip"
        if maneuver == "reverse":
            action = action_index(accelerate=False, brake=True, left=command.steering > 0.1, right=command.steering < -0.1)

        speed_drop = 0.0 if self._previous_speed is None else self._previous_speed - speed
        impact = speed_drop > self.config.impact_speed_drop_mps
        self._previous_speed = speed
        self.steps += 1
        self.impact_suspects += int(impact)
        self.recovery_steps += int(command.recovery)
        self.slip_steps += int(slipping)
        self.max_progress_m = max(self.max_progress_m, state.progress_m)

        if self._log_writer is not None:
            self._log_writer.writerow(
                {
                    "race_time_s": round(race_time_s, 3),
                    "x": round(float(position[0]), 3),
                    "y": round(float(position[1]), 3),
                    "z": round(float(position[2]), 3),
                    "speed_mps": round(speed, 3),
                    "progress_m": round(state.progress_m, 3),
                    "lateral_error_m": round(state.lateral_error_m, 3),
                    "heading_error_rad": round(state.heading_error_rad, 4),
                    "elevation_error_m": round(state.elevation_error_m, 3),
                    "heading_source": source,
                    "forward_velocity_angle_rad": round(forward_velocity_angle, 4),
                    "steering": round(command.steering, 4),
                    "throttle": round(command.throttle, 4),
                    "brake": round(command.brake, 4),
                    "target_speed_mps": round(command.target_speed_mps, 3),
                    "recovery": command.recovery,
                    "action_idx": action,
                    "speed_drop_mps": round(speed_drop, 3),
                    "impact_suspect": impact,
                    "maneuver": maneuver,
                }
            )
        return action, command

    def _update_slip(self, speed: float, angle: float) -> bool:
        cfg = self.config
        if not cfg.slip_guard_enabled or math.isnan(angle) or speed < cfg.slip_min_speed_mps:
            self._slipping = False
            return False
        threshold = cfg.slip_exit_rad if self._slipping else cfg.slip_enter_rad
        self._slipping = abs(angle) > threshold
        return self._slipping

    def _unstick(self, race_time_s: float, speed: float, command: ControllerCommand) -> str:
        cfg = self.config
        if not cfg.unstick_enabled:
            return "drive"
        if race_time_s < self._reverse_until:
            return "reverse"
        trying = command.throttle > 0.5 and command.brake == 0.0
        if race_time_s < cfg.unstick_grace_s or speed >= cfg.stuck_speed_mps or not trying:
            self._stuck_since = None
            return "drive"
        if self._stuck_since is None:
            self._stuck_since = race_time_s
        if race_time_s - self._stuck_since >= cfg.stuck_time_s:
            self._stuck_since = None
            self._reverse_until = race_time_s + cfg.reverse_time_s
            self.reversals += 1
            return "reverse"
        return "drive"

    def summary(self) -> dict:
        return {
            "steps": self.steps,
            "impact_suspects": self.impact_suspects,
            "recovery_steps": self.recovery_steps,
            "reversals": self.reversals,
            "slip_steps": self.slip_steps,
            "max_progress_m": round(self.max_progress_m, 2),
            "centerline_length_m": round(float(self.geometry.cumulative_lengths[-1]), 2),
        }
