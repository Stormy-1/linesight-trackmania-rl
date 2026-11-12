"""Offline rollout ingestion and structured-state analysis.

The format handled here is intentionally small and explicit so a future
collector can write it without coupling the analysis to TrackMania objects.
No game, checkpoint, or learner code is imported by this module.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from trackmania_rl.hybrid_controller import ConservativeCenterlineController
from trackmania_rl.hybrid_diagnostics import (
    identify_divergence_recovery_events,
    summarize_progress_sections,
)
from trackmania_rl.hybrid_state import (
    STRUCTURED_STATE_SCHEMA_VERSION,
    CenterlineGeometry,
    make_track_relative_state,
    progress_rate_mps,
)


ROLLOUT_COLUMNS = (
    "timestamp_s",
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "heading_x",
    "heading_y",
    "heading_z",
    "speed_mps",
)


@dataclass(frozen=True)
class RolloutSample:
    timestamp_s: float
    position: np.ndarray
    heading: np.ndarray
    speed_mps: float


def derive_headings_from_positions(samples: list[RolloutSample]) -> list[RolloutSample]:
    """Derive forward headings from consecutive recorded positions.

    This is an offline repair for traces captured before the live hook used
    world-velocity direction.  It is appropriate for moving samples; a zero
    displacement keeps the original heading.
    """

    if not samples:
        raise ValueError("rollout must contain at least one sample")
    repaired: list[RolloutSample] = []
    for index, sample in enumerate(samples):
        if len(samples) == 1:
            delta = np.zeros(3, dtype=np.float64)
        elif index < len(samples) - 1:
            delta = samples[index + 1].position - sample.position
        else:
            delta = sample.position - samples[index - 1].position
        norm = float(np.linalg.norm(delta))
        heading = sample.heading if norm <= 1e-9 else delta / norm
        repaired.append(
            RolloutSample(
                timestamp_s=sample.timestamp_s,
                position=sample.position.copy(),
                heading=np.asarray(heading, dtype=np.float64),
                speed_mps=sample.speed_mps,
            )
        )
    return repaired


def _finite_float(row_number: int, row: Mapping[str, str], column: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {column} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"row {row_number}: {column} must be finite")
    return value


def parse_rollout_rows(rows: Iterable[Mapping[str, str]]) -> list[RolloutSample]:
    """Parse and validate explicit rollout rows.

    Timestamps must be strictly increasing.  Position and heading are kept in
    the same 3D coordinate convention as the centerline files.
    """

    parsed: list[RolloutSample] = []
    previous_timestamp: float | None = None
    for row_number, row in enumerate(rows, start=2):
        timestamp = _finite_float(row_number, row, "timestamp_s")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(f"row {row_number}: timestamps must be strictly increasing")
        sample = RolloutSample(
            timestamp_s=timestamp,
            position=np.asarray(
                [
                    _finite_float(row_number, row, "position_x_m"),
                    _finite_float(row_number, row, "position_y_m"),
                    _finite_float(row_number, row, "position_z_m"),
                ],
                dtype=np.float64,
            ),
            heading=np.asarray(
                [
                    _finite_float(row_number, row, "heading_x"),
                    _finite_float(row_number, row, "heading_y"),
                    _finite_float(row_number, row, "heading_z"),
                ],
                dtype=np.float64,
            ),
            speed_mps=_finite_float(row_number, row, "speed_mps"),
        )
        parsed.append(sample)
        previous_timestamp = timestamp
    if not parsed:
        raise ValueError("rollout must contain at least one sample")
    return parsed


def load_rollout_csv(path: str | Path) -> list[RolloutSample]:
    """Load a rollout CSV using :data:`ROLLOUT_COLUMNS`."""

    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in ROLLOUT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"rollout CSV is missing required columns: {', '.join(missing)}")
        return parse_rollout_rows(reader)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def analyze_rollout(
    samples: list[RolloutSample],
    geometry: CenterlineGeometry,
    *,
    controller: ConservativeCenterlineController | None = None,
    search_radius: int | None = 32,
    finish_progress_m: float | None = None,
    finish_distance_limit_m: float = 4.0,
    finish_elevation_tolerance_m: float = 8.0,
) -> dict[str, Any]:
    """Map-match samples and evaluate nominal controller diagnostics."""

    if not samples:
        raise ValueError("rollout must contain at least one sample")
    if search_radius is not None and search_radius < 0:
        raise ValueError("search_radius must be non-negative or None")
    if finish_progress_m is not None and (not math.isfinite(float(finish_progress_m)) or finish_progress_m <= 0):
        raise ValueError("finish_progress_m must be finite and positive or None")
    if not math.isfinite(float(finish_distance_limit_m)) or finish_distance_limit_m <= 0:
        raise ValueError("finish_distance_limit_m must be finite and positive")
    if not math.isfinite(float(finish_elevation_tolerance_m)) or finish_elevation_tolerance_m < 0:
        raise ValueError("finish_elevation_tolerance_m must be finite and non-negative")
    active_controller = controller or ConservativeCenterlineController()
    previous_segment: int | None = None
    previous_progress: float | None = None
    states: list[dict[str, Any]] = []
    raw_progress_deltas: list[float] = []

    for sample_index, sample in enumerate(samples):
        state = make_track_relative_state(
            sample.position,
            sample.heading,
            sample.speed_mps,
            geometry,
            previous_segment=previous_segment,
            search_radius=search_radius if previous_segment is not None else None,
        )
        if previous_progress is None:
            progress_rate = 0.0
            raw_delta = 0.0
        else:
            dt_s = sample.timestamp_s - samples[sample_index - 1].timestamp_s
            progress_rate = progress_rate_mps(state.progress_m, previous_progress, dt_s)
            raw_delta = state.progress_m - previous_progress
        command = active_controller.command(state, geometry)
        states.append(
            {
                "timestamp_s": sample.timestamp_s,
                "segment_index": state.segment_index,
                "progress_m": state.progress_m,
                "progress_rate_mps": progress_rate,
                "lateral_error_m": state.lateral_error_m,
                "distance_to_centerline_m": state.distance_to_centerline_m,
                "horizontal_distance_to_centerline_m": state.horizontal_distance_to_centerline_m,
                "elevation_error_m": state.elevation_error_m,
                "heading_error_rad": state.heading_error_rad,
                "speed_mps": state.speed_mps,
                "local_curvature": state.local_curvature,
                "target_speed_mps": command.target_speed_mps,
                "speed_error_mps": command.target_speed_mps - state.speed_mps,
                "steering": command.steering,
                "throttle": command.throttle,
                "brake": command.brake,
                "recovery": command.recovery,
                "step_index": sample_index,
            }
        )
        raw_progress_deltas.append(raw_delta)
        previous_segment = state.segment_index
        previous_progress = state.progress_m

    progress = [float(item["progress_m"]) for item in states]
    progress_rates = [float(item["progress_rate_mps"]) for item in states]
    lateral_errors = [abs(float(item["lateral_error_m"])) for item in states]
    horizontal_errors = [float(item["horizontal_distance_to_centerline_m"]) for item in states]
    elevation_errors = [abs(float(item["elevation_error_m"])) for item in states]
    heading_errors = [abs(float(item["heading_error_rad"])) for item in states]
    speed_errors = [abs(float(item["speed_error_mps"])) for item in states]
    recovery_count = sum(bool(item["recovery"]) for item in states)
    total_length = float(geometry.cumulative_lengths[-1])
    finish_progress = total_length - 1.0 if finish_progress_m is None else float(finish_progress_m)
    reached_finish_progress = progress[-1] >= finish_progress
    strict_3d_finish = bool(
        reached_finish_progress and float(states[-1]["distance_to_centerline_m"]) <= finish_distance_limit_m
    )
    elevation_tolerant_finish = bool(
        reached_finish_progress
        and horizontal_errors[-1] <= finish_distance_limit_m
        and elevation_errors[-1] <= finish_elevation_tolerance_m
    )
    event_records = [
        {
            "step_index": item["step_index"],
            "timestamp_s": item["timestamp_s"],
            "progress_m": item["progress_m"],
            "lateral_error_m": item["lateral_error_m"],
            "heading_error_rad": item["heading_error_rad"],
            "horizontal_distance_to_centerline_m": item["horizontal_distance_to_centerline_m"],
            "elevation_error_m": item["elevation_error_m"],
            "recovery": item["recovery"],
        }
        for item in states
    ]
    events = identify_divergence_recovery_events(event_records)
    return {
        "schema_version": STRUCTURED_STATE_SCHEMA_VERSION,
        "samples": len(states),
        "duration_s": float(samples[-1].timestamp_s - samples[0].timestamp_s),
        "start_progress_m": progress[0],
        "end_progress_m": progress[-1],
        "progress_gain_m": progress[-1] - progress[0],
        "nonmonotonic_progress_steps": sum(delta < 0.0 for delta in raw_progress_deltas),
        "mean_progress_rate_mps": _mean(progress_rates),
        "max_progress_rate_mps": max(progress_rates),
        "mean_abs_lateral_error_m": _mean(lateral_errors),
        "max_abs_lateral_error_m": max(lateral_errors),
        "mean_horizontal_distance_to_centerline_m": _mean(horizontal_errors),
        "max_horizontal_distance_to_centerline_m": max(horizontal_errors),
        "mean_abs_elevation_error_m": _mean(elevation_errors),
        "max_abs_elevation_error_m": max(elevation_errors),
        "mean_abs_heading_error_rad": _mean(heading_errors),
        "max_abs_heading_error_rad": max(heading_errors),
        "mean_abs_speed_error_mps": _mean(speed_errors),
        "recovery_samples": recovery_count,
        "recovery_rate": float(recovery_count / len(states)),
        "strict_3d_finish": strict_3d_finish,
        "elevation_tolerant_finish": elevation_tolerant_finish,
        "finish_diagnostic": {
            "strict_3d_finish": strict_3d_finish,
            "elevation_tolerant_finish": elevation_tolerant_finish,
            "progress_gate_m": finish_progress,
            "strict_distance_limit_m": finish_distance_limit_m,
            "tolerant_horizontal_distance_limit_m": finish_distance_limit_m,
            "tolerant_elevation_limit_m": finish_elevation_tolerance_m,
        },
        "divergence_recovery": events,
        "first_divergence_event": events["first_divergence_event"],
        "first_recovery_event": events["first_recovery"],
        "section_metrics": summarize_progress_sections(states, total_length, bins=10),
        "trajectory_data_available": True,
        "states": states,
    }
