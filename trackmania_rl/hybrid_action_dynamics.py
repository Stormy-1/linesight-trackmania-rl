"""Observational action-conditioned transition metrics for captured telemetry."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from trackmania_rl.hybrid_action_telemetry import ActionTelemetrySample


def _summary(values: list[float], unit: str) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("action transition values must be finite")
    if len(array) == 0:
        return {
            "transition_count": 0,
            f"mean_{unit}": 0.0,
            f"p50_{unit}": 0.0,
            f"p90_{unit}": 0.0,
            f"p99_{unit}": 0.0,
        }
    return {
        "transition_count": len(array),
        f"mean_{unit}": float(np.mean(array)),
        f"p50_{unit}": float(np.percentile(array, 50.0)),
        f"p90_{unit}": float(np.percentile(array, 90.0)),
        f"p99_{unit}": float(np.percentile(array, 99.0)),
    }


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def action_transition_metrics(samples: Sequence[ActionTelemetrySample]) -> dict[str, Any]:
    """Summarize transitions using the action stored on the preceding sample."""

    if len(samples) < 2:
        raise ValueError("at least two action telemetry samples are required")
    actions = [int(sample.action_idx) for sample in samples]
    transitions_by_action: dict[int, list[tuple[float, float, float]]] = {}
    speed_changes: list[float] = []
    yaw_rates: list[float] = []
    lateral_rates: list[float] = []
    transition_records: list[dict[str, float | int]] = []
    for current, following in pairwise(samples):
        current_telemetry = current.telemetry
        following_telemetry = following.telemetry
        dt_s = float(following_telemetry.timestamp_s - current_telemetry.timestamp_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("action telemetry timestamps must be strictly increasing")
        current_heading = np.asarray(current_telemetry.heading, dtype=np.float64)
        following_heading = np.asarray(following_telemetry.heading, dtype=np.float64)
        current_position = np.asarray(current_telemetry.position, dtype=np.float64)
        following_position = np.asarray(following_telemetry.position, dtype=np.float64)
        if (
            current_heading.shape != (3,)
            or following_heading.shape != (3,)
            or current_position.shape != (3,)
            or following_position.shape != (3,)
            or not np.isfinite(current_heading).all()
            or not np.isfinite(following_heading).all()
            or not np.isfinite(current_position).all()
            or not np.isfinite(following_position).all()
        ):
            raise ValueError("action telemetry pose fields must be finite 3-vectors")
        current_heading /= max(float(np.linalg.norm(current_heading)), 1e-9)
        following_heading /= max(float(np.linalg.norm(following_heading)), 1e-9)
        current_speed = float(current_telemetry.speed_mps)
        following_speed = float(following_telemetry.speed_mps)
        if not math.isfinite(current_speed) or not math.isfinite(following_speed):
            raise ValueError("action telemetry speeds must be finite")
        speed_change = (following_speed - current_speed) / dt_s
        current_yaw = math.atan2(float(current_heading[2]), float(current_heading[0]))
        following_yaw = math.atan2(float(following_heading[2]), float(following_heading[0]))
        yaw_rate = _wrap_angle(following_yaw - current_yaw) / dt_s
        position_delta = following_position - current_position
        lateral_rate = float(current_heading[0] * position_delta[2] - current_heading[2] * position_delta[0]) / dt_s
        speed_changes.append(speed_change)
        yaw_rates.append(yaw_rate)
        lateral_rates.append(lateral_rate)
        transitions_by_action.setdefault(int(current.action_idx), []).append((speed_change, yaw_rate, lateral_rate))
        transition_records.append(
            {
                "action_idx": int(current.action_idx),
                "dt_s": dt_s,
                "speed_change_rate_mps2": speed_change,
                "yaw_rate_rad_s": yaw_rate,
                "lateral_displacement_rate_mps": lateral_rate,
            }
        )

    by_action: dict[str, dict[str, Any]] = {}
    for action_idx in sorted(set(actions) | set(transitions_by_action)):
        values = transitions_by_action.get(action_idx, [])
        by_action[str(action_idx)] = {
            "sample_count": actions.count(action_idx),
            "transition_count": len(values),
            "speed_change_rate": _summary([value[0] for value in values], "speed_change_rate_mps2"),
            "yaw_rate": _summary([value[1] for value in values], "yaw_rate_rad_s"),
            "lateral_displacement_rate": _summary([value[2] for value in values], "lateral_displacement_rate_mps"),
        }
        by_action[str(action_idx)]["mean_speed_change_rate_mps2"] = (
            float(np.mean([value[0] for value in values])) if values else 0.0
        )

    return {
        "sample_count": len(samples),
        "transition_count": len(transition_records),
        "action_counts": {str(action_idx): int(count) for action_idx, count in sorted(Counter(actions).items())},
        "greedy_fraction": float(sum(sample.action_was_greedy for sample in samples) / len(samples)),
        "overall": {
            "speed_change_rate": _summary(speed_changes, "speed_change_rate_mps2"),
            "yaw_rate": _summary(yaw_rates, "yaw_rate_rad_s"),
            "lateral_displacement_rate": _summary(lateral_rates, "lateral_displacement_rate_mps"),
        },
        "by_action": by_action,
        "finite_metrics": bool(
            all(
                np.isfinite(record[metric])
                for record in transition_records
                for metric in ("dt_s", "speed_change_rate_mps2", "yaw_rate_rad_s", "lateral_displacement_rate_mps")
            )
        ),
        "transition_records": transition_records,
    }


def randomized_action_transition_records(samples: Sequence[ActionTelemetrySample]) -> list[dict[str, float | int]]:
    """Return within-file transitions whose preceding action was randomized."""

    if any(sample.action_is_randomized is None for sample in samples):
        raise ValueError("randomized transition analysis requires intervention provenance on every sample")
    report = action_transition_metrics(samples)
    return [
        record
        for sample, record in zip(samples[:-1], report["transition_records"])
        if sample.action_is_randomized is True
    ]
