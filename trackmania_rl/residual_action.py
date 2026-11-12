"""Residual action for Step 4: a small, clipped correction on the base command.

The classical stack stays in charge. The policy outputs two numbers in [-1, 1]:

* ``a[0]`` -> Δsteering = a[0] · ``max_steering_delta``, added to pure pursuit's
  steering and clipped to the controller's speed-dependent steering cap (a fitted
  vehicle limit the residual can't override).
* ``a[1]`` -> Δtarget speed = a[1] · ``max_target_speed_delta_mps``, added to the
  speed profile's target before the controller turns it into throttle/brake.

The residual is **shielded**: it is gated off whenever a safety layer owns the
car (recovery mode, the slip guard, the unstick reverse). Those keep their
hand-checked behaviour, and the policy only learns to refine nominal driving.
With a zero residual the command is bit-for-bit the base command.

Optional physics shield (HANDOVER §82; **off by default**: on the rl_train maps it cost more than it
saved, 1/6 finishes vs 3/6 without it, because a jump's "required" speed is the pro's speed, not a true
minimum: lowering it is fatal on a marginal jump (A06) and what got C05 over an overshot one):

* **Jump floor:** on a jump approach the profile's target is the speed the jump *requires*
  (ballistic flight, §70), a floor rather than a cap. The residual may raise it, never lower it.
  td3_v2 lowered it by 0.2–1.5 m/s before A06's marginal first jump and the car fell short.
* **Airborne:** with no wheel on the ground the residual is off. Inputs barely act in the air,
  and td3_v2 steering and braking mid-air on B11 landed the car 2.8 m off the line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

RESIDUAL_SIZE = 2


@dataclass(frozen=True)
class ResidualConfig:
    # Small on purpose: a quarter of full lock and 8 m/s of target speed.
    max_steering_delta: float = 0.25
    max_target_speed_delta_mps: float = 8.0
    jump_floor_shield: bool = False
    airborne_shield: bool = False


@dataclass(frozen=True)
class Residual:
    """A scaled residual in physical units."""

    steering_delta: float = 0.0
    target_speed_delta_mps: float = 0.0

    @property
    def is_zero(self) -> bool:
        return self.steering_delta == 0.0 and self.target_speed_delta_mps == 0.0


ZERO_RESIDUAL = Residual()


def scale_residual(action: np.ndarray, config: ResidualConfig) -> Residual:
    """Map a policy output in [-1, 1]^2 (clipped) to physical deltas."""

    a = np.asarray(action, dtype=np.float64).ravel()
    if a.shape != (RESIDUAL_SIZE,):
        raise ValueError(f"residual action must have {RESIDUAL_SIZE} values, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"residual action must be finite, got {a}")
    a = np.clip(a, -1.0, 1.0)
    return Residual(float(a[0]) * config.max_steering_delta, float(a[1]) * config.max_target_speed_delta_mps)


def shield_jump_floor(residual: Residual, action: np.ndarray, jump_locked: bool) -> tuple[Residual, np.ndarray]:
    """On a jump-locked approach the residual may not lower the target speed (it's a required speed)."""

    if not jump_locked or residual.target_speed_delta_mps >= 0.0:
        return residual, action
    shielded = np.array([action[0], 0.0], dtype=np.float32)
    return Residual(residual.steering_delta, 0.0), shielded


def apply_steering_residual(base_steering: float, delta: float, steering_limit: float) -> float:
    """base + delta, clipped to the controller's steering cap (unchanged when delta is 0)."""

    if delta == 0.0:
        return base_steering
    return float(np.clip(base_steering + delta, -steering_limit, steering_limit))


class ResidualPolicy(Protocol):
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return a residual action in [-1, 1]^2 for one observation."""
        ...


class ZeroResidualPolicy:
    """Always outputs zero: base+residual must equal the base stack exactly."""

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.zeros(RESIDUAL_SIZE, dtype=np.float32)
