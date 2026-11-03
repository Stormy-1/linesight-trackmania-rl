"""Recorded-transition metrics for validating approximate hybrid dynamics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from trackmania_rl.hybrid_rollout import RolloutSample


def _summary(values: np.ndarray, unit: str, *, prefix: str = "") -> dict[str, float | int]:
    finite_values = np.asarray(values, dtype=np.float64)
    if finite_values.ndim != 1 or not np.isfinite(finite_values).all():
        raise ValueError("transition metric values must be a finite vector")
    if len(finite_values) == 0:
        return {
            f"{prefix}count": 0,
            f"{prefix}mean_{unit}": 0.0,
            f"{prefix}p50_{unit}": 0.0,
            f"{prefix}p90_{unit}": 0.0,
            f"{prefix}p99_{unit}": 0.0,
            f"{prefix}max_{unit}": 0.0,
        }
    return {
        f"{prefix}count": len(finite_values),
        f"{prefix}mean_{unit}": float(np.mean(finite_values)),
        f"{prefix}p50_{unit}": float(np.percentile(finite_values, 50.0)),
        f"{prefix}p90_{unit}": float(np.percentile(finite_values, 90.0)),
        f"{prefix}p99_{unit}": float(np.percentile(finite_values, 99.0)),
        f"{prefix}max_{unit}": float(np.max(finite_values)),
    }


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def transition_observations(
    samples: Sequence[RolloutSample],
    *,
    moving_speed_threshold_mps: float = 1.0,
    drag_model_mps2_per_mps: float = 0.05,
) -> dict[str, np.ndarray]:
    """Convert adjacent telemetry samples into validated transition vectors."""

    if len(samples) < 2:
        raise ValueError("at least two samples are required for transition metrics")
    if not math.isfinite(float(moving_speed_threshold_mps)) or moving_speed_threshold_mps < 0.0:
        raise ValueError("moving_speed_threshold_mps must be finite and non-negative")
    if not math.isfinite(float(drag_model_mps2_per_mps)) or drag_model_mps2_per_mps < 0.0:
        raise ValueError("drag_model_mps2_per_mps must be finite and non-negative")

    values: dict[str, list[float | bool]] = {
        "dt_s": [],
        "moving": [],
        "displacement_rate_mps": [],
        "position_prediction_residual_m": [],
        "speed_alignment_error_mps": [],
        "speed_change_rate_mps2": [],
        "speed_model_residual_mps": [],
        "abs_yaw_rate_rad_s": [],
    }
    for current, following in pairwise(samples):
        dt_s = float(following.timestamp_s - current.timestamp_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("sample timestamps must be strictly increasing")
        current_position = np.asarray(current.position, dtype=np.float64)
        following_position = np.asarray(following.position, dtype=np.float64)
        if (
            current_position.shape != (3,)
            or following_position.shape != (3,)
            or not np.isfinite(current_position).all()
            or not np.isfinite(following_position).all()
        ):
            raise ValueError("sample positions must contain three finite values")
        position_delta = following_position - current_position
        heading = np.asarray(current.heading, dtype=np.float64)
        heading_norm = float(np.linalg.norm(heading))
        if heading.shape != (3,) or heading_norm <= 1e-9 or not np.isfinite(heading).all():
            raise ValueError("sample headings must be finite and non-zero")
        current_speed = float(current.speed_mps)
        following_speed = float(following.speed_mps)
        if not math.isfinite(current_speed) or not math.isfinite(following_speed):
            raise ValueError("sample speeds must be finite")
        heading /= heading_norm
        displacement_rate = float(np.linalg.norm(position_delta) / dt_s)
        expected_delta = heading * current_speed * dt_s
        current_yaw = math.atan2(float(heading[2]), float(heading[0]))
        following_heading = np.asarray(following.heading, dtype=np.float64)
        following_norm = float(np.linalg.norm(following_heading))
        if (
            following_heading.shape != (3,)
            or following_norm <= 1e-9
            or not np.isfinite(following_heading).all()
        ):
            raise ValueError("sample headings must be finite and non-zero")
        following_heading /= following_norm
        following_yaw = math.atan2(float(following_heading[2]), float(following_heading[0]))
        values["dt_s"].append(dt_s)
        values["moving"].append(abs(current_speed) > moving_speed_threshold_mps)
        values["displacement_rate_mps"].append(displacement_rate)
        values["position_prediction_residual_m"].append(float(np.linalg.norm(position_delta - expected_delta)))
        values["speed_alignment_error_mps"].append(displacement_rate - abs(current_speed))
        values["speed_change_rate_mps2"].append((following_speed - current_speed) / dt_s)
        drag_only_speed = current_speed - drag_model_mps2_per_mps * current_speed * dt_s
        values["speed_model_residual_mps"].append(following_speed - drag_only_speed)
        values["abs_yaw_rate_rad_s"].append(abs(_wrap_angle(following_yaw - current_yaw)) / dt_s)
    return {key: np.asarray(items, dtype=np.float64) for key, items in values.items() if key != "moving"} | {
        "moving": np.asarray(values["moving"], dtype=bool)
    }


def summarize_transition_observations(
    observations: Mapping[str, np.ndarray],
    *,
    moving_speed_threshold_mps: float = 1.0,
    drag_model_mps2_per_mps: float = 0.05,
) -> dict[str, Any]:
    """Summarize validated transition vectors with all/moving quantiles."""

    if not math.isfinite(float(moving_speed_threshold_mps)) or moving_speed_threshold_mps < 0.0:
        raise ValueError("moving_speed_threshold_mps must be finite and non-negative")
    if not math.isfinite(float(drag_model_mps2_per_mps)) or drag_model_mps2_per_mps < 0.0:
        raise ValueError("drag_model_mps2_per_mps must be finite and non-negative")
    required = (
        "dt_s",
        "moving",
        "displacement_rate_mps",
        "position_prediction_residual_m",
        "speed_alignment_error_mps",
        "speed_change_rate_mps2",
        "speed_model_residual_mps",
        "abs_yaw_rate_rad_s",
    )
    arrays = {key: np.asarray(observations[key]) for key in required}
    count = len(arrays["dt_s"])
    if any(array.ndim != 1 or len(array) != count for array in arrays.values()):
        raise ValueError("transition observation vectors must have matching one-dimensional shapes")
    if not np.asarray(arrays["moving"], dtype=bool).shape == (count,):
        raise ValueError("moving transition mask has an invalid shape")
    moving = np.asarray(arrays["moving"], dtype=bool)

    def metric(name: str, unit: str) -> dict[str, float | int]:
        array = np.asarray(arrays[name], dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("transition observation vectors must be finite")
        return {
            **_summary(array, unit),
            **_summary(array[moving], unit, prefix="moving_"),
        }

    return {
        "transition_count": int(count),
        "moving_transition_count": int(np.sum(moving)),
        "moving_fraction": float(np.mean(moving)) if count else 0.0,
        "moving_speed_threshold_mps": float(moving_speed_threshold_mps),
        "drag_model_mps2_per_mps": float(drag_model_mps2_per_mps),
        "timestamp": metric("dt_s", "s"),
        "displacement_rate": metric("displacement_rate_mps", "mps"),
        "position_prediction_residual": metric("position_prediction_residual_m", "m"),
        "speed_alignment_error": metric("speed_alignment_error_mps", "mps"),
        "speed_change_rate": metric("speed_change_rate_mps2", "mps2"),
        "speed_model_residual": metric("speed_model_residual_mps", "mps"),
        "observed_abs_yaw_rate": metric("abs_yaw_rate_rad_s", "rad_s"),
    }


def transition_metrics(
    samples: Sequence[RolloutSample],
    *,
    moving_speed_threshold_mps: float = 1.0,
    drag_model_mps2_per_mps: float = 0.05,
) -> dict[str, Any]:
    """Return deterministic one-step dynamics metrics for one rollout."""

    observations = transition_observations(
        samples,
        moving_speed_threshold_mps=moving_speed_threshold_mps,
        drag_model_mps2_per_mps=drag_model_mps2_per_mps,
    )
    return summarize_transition_observations(
        observations,
        moving_speed_threshold_mps=moving_speed_threshold_mps,
        drag_model_mps2_per_mps=drag_model_mps2_per_mps,
    )
