"""Pure map-matching and structured-state helpers for hybrid driving.

This module is intentionally independent of TrackMania, TMInterface, and the
RL learner.  It converts a known centerline plus a vehicle pose into the
track-relative quantities needed by a nominal controller, MPC, or residual RL
policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


# Version 2 adds horizontal-distance and elevation-error diagnostics while
# keeping the original 3D distance and lateral fields available.
STRUCTURED_STATE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CenterlineGeometry:
    """Precomputed geometry for a 3D centerline."""

    points: np.ndarray
    segment_vectors: np.ndarray
    segment_lengths: np.ndarray
    segment_tangents: np.ndarray
    cumulative_lengths: np.ndarray
    point_curvature: np.ndarray


@dataclass(frozen=True)
class TrackRelativeState:
    """Vehicle state expressed in the local frame of a known track."""

    segment_index: int
    progress_m: float
    lateral_error_m: float
    distance_to_centerline_m: float
    heading_error_rad: float
    speed_mps: float
    local_curvature: float
    projection: np.ndarray
    tangent: np.ndarray
    horizontal_distance_to_centerline_m: float
    elevation_error_m: float


@dataclass(frozen=True)
class LookaheadGeometry:
    """Centerline samples at distances measured from the current progress."""

    distances_m: np.ndarray
    points: np.ndarray
    tangents: np.ndarray
    curvatures: np.ndarray


def _as_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 2:
        raise ValueError("centerline must have shape (N, 3) with N >= 2")
    if not np.isfinite(values).all():
        raise ValueError("centerline must contain only finite values")
    return values


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-9):
        raise ValueError("centerline contains a zero-length segment")
    return vectors / lengths[:, None]


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def build_centerline_geometry(points: np.ndarray) -> CenterlineGeometry:
    """Precompute segment lengths, tangents, progress, and curvature."""

    values = _as_points(points)
    segment_vectors = np.diff(values, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    segment_tangents = _normalize_vectors(segment_vectors)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    point_tangents = np.empty_like(values)
    point_tangents[0] = segment_tangents[0]
    point_tangents[-1] = segment_tangents[-1]
    point_tangents[1:-1] = _normalize_vectors(segment_tangents[:-1] + segment_tangents[1:])
    point_curvature = np.zeros(len(values), dtype=np.float64)
    for index in range(1, len(values) - 1):
        angle = math.acos(float(np.clip(np.dot(segment_tangents[index - 1], segment_tangents[index]), -1.0, 1.0)))
        local_length = 0.5 * (segment_lengths[index - 1] + segment_lengths[index])
        point_curvature[index] = angle / local_length

    return CenterlineGeometry(
        points=values,
        segment_vectors=segment_vectors,
        segment_lengths=segment_lengths,
        segment_tangents=segment_tangents,
        cumulative_lengths=cumulative_lengths,
        point_curvature=point_curvature,
    )


def project_to_centerline(
    position: np.ndarray,
    geometry: CenterlineGeometry,
    *,
    previous_segment: int | None = None,
    search_radius: int | None = None,
    horizontal_only: bool = False,
) -> tuple[int, float, np.ndarray, np.ndarray, float]:
    """Project a vehicle position onto the closest centerline segment.

    Returns ``(segment_index, segment_fraction, projection, tangent,
    distance_to_centerline)``.  A previous segment and radius can constrain the
    search for temporal map matching; without them the full centerline is used.
    """

    point = np.asarray(position, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("position must contain three finite values")

    if not isinstance(horizontal_only, bool):
        raise ValueError("horizontal_only must be a boolean")
    segment_count = len(geometry.segment_lengths)
    if previous_segment is None or search_radius is None:
        candidates = np.arange(segment_count)
    else:
        if not 0 <= previous_segment < segment_count:
            raise ValueError("previous_segment is outside the centerline")
        if search_radius < 0:
            raise ValueError("search_radius must be non-negative")
        start = max(0, previous_segment - search_radius)
        stop = min(segment_count, previous_segment + search_radius + 1)
        candidates = np.arange(start, stop)

    starts = geometry.points[candidates]
    vectors = geometry.segment_vectors[candidates]
    if horizontal_only:
        metric_starts = starts[:, [0, 2]]
        metric_vectors = vectors[:, [0, 2]]
        metric_lengths = np.linalg.norm(metric_vectors, axis=1)
        if np.any(metric_lengths <= 1e-9):
            raise ValueError("horizontal projection requires non-zero horizontal centerline segments")
        point_metric = point[[0, 2]]
    else:
        metric_starts = starts
        metric_vectors = vectors
        metric_lengths = geometry.segment_lengths[candidates]
        point_metric = point
    fractions = np.einsum("ij,ij->i", point_metric - metric_starts, metric_vectors) / (metric_lengths ** 2)
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = starts + vectors * fractions[:, None]
    distances = np.linalg.norm(point_metric - (metric_starts + metric_vectors * fractions[:, None]), axis=1)
    local_index = int(np.argmin(distances))
    segment_index = int(candidates[local_index])
    fraction = float(fractions[local_index])
    return (
        segment_index,
        fraction,
        projections[local_index],
        geometry.segment_tangents[segment_index],
        float(distances[local_index]),
    )


def heading_error_rad(heading: np.ndarray, tangent: np.ndarray) -> float:
    """Return signed horizontal heading error in ``[-pi, pi)``."""

    vehicle_heading = np.asarray(heading, dtype=np.float64)
    track_tangent = np.asarray(tangent, dtype=np.float64)
    if vehicle_heading.shape != (3,) or track_tangent.shape != (3,):
        raise ValueError("heading and tangent must have shape (3,)")
    vehicle_horizontal = np.array([vehicle_heading[0], vehicle_heading[2]], dtype=np.float64)
    track_horizontal = np.array([track_tangent[0], track_tangent[2]], dtype=np.float64)
    vehicle_norm = np.linalg.norm(vehicle_horizontal)
    track_norm = np.linalg.norm(track_horizontal)
    if vehicle_norm <= 1e-9 or track_norm <= 1e-9:
        raise ValueError("heading and tangent must have non-zero horizontal components")
    vehicle_horizontal /= vehicle_norm
    track_horizontal /= track_norm
    cross = track_horizontal[0] * vehicle_horizontal[1] - track_horizontal[1] * vehicle_horizontal[0]
    dot = np.dot(track_horizontal, vehicle_horizontal)
    return _wrap_angle(math.atan2(float(cross), float(dot)))


def make_track_relative_state(
    position: np.ndarray,
    heading: np.ndarray,
    speed_mps: float,
    geometry: CenterlineGeometry,
    *,
    previous_segment: int | None = None,
    search_radius: int | None = None,
    horizontal_only: bool = False,
) -> TrackRelativeState:
    """Build the structured state consumed by hybrid controllers."""

    if not math.isfinite(float(speed_mps)):
        raise ValueError("speed_mps must be finite")
    segment_index, fraction, projection, tangent, distance = project_to_centerline(
        position,
        geometry,
        previous_segment=previous_segment,
        search_radius=search_radius,
        horizontal_only=horizontal_only,
    )
    offset = np.asarray(position, dtype=np.float64) - projection
    distance = float(np.linalg.norm(offset))
    lateral_error = float(tangent[2] * offset[0] - tangent[0] * offset[2])
    horizontal_offset = np.asarray([offset[0], offset[2]], dtype=np.float64)
    horizontal_distance = float(np.linalg.norm(horizontal_offset))
    elevation_error = float(offset[1])
    progress = float(geometry.cumulative_lengths[segment_index] + fraction * geometry.segment_lengths[segment_index])
    curvature_index = min(segment_index + int(fraction >= 0.5), len(geometry.point_curvature) - 1)
    return TrackRelativeState(
        segment_index=segment_index,
        progress_m=progress,
        lateral_error_m=lateral_error,
        distance_to_centerline_m=distance,
        heading_error_rad=heading_error_rad(heading, tangent),
        speed_mps=float(speed_mps),
        local_curvature=float(geometry.point_curvature[curvature_index]),
        projection=projection.copy(),
        tangent=tangent.copy(),
        horizontal_distance_to_centerline_m=horizontal_distance,
        elevation_error_m=elevation_error,
    )


def _sample_at_progress(progress_m: float, geometry: CenterlineGeometry) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate a centerline point, tangent, and curvature at track progress."""

    if not math.isfinite(float(progress_m)):
        raise ValueError("progress_m must be finite")
    progress = float(np.clip(progress_m, 0.0, geometry.cumulative_lengths[-1]))
    segment = int(np.searchsorted(geometry.cumulative_lengths, progress, side="right") - 1)
    segment = min(max(segment, 0), len(geometry.segment_lengths) - 1)
    fraction = (progress - geometry.cumulative_lengths[segment]) / geometry.segment_lengths[segment]
    point = geometry.points[segment] + geometry.segment_vectors[segment] * fraction
    curvature_index = min(segment + int(fraction >= 0.5), len(geometry.point_curvature) - 1)
    return point, geometry.segment_tangents[segment].copy(), float(geometry.point_curvature[curvature_index])


def sample_centerline_lookahead(
    geometry: CenterlineGeometry,
    progress_m: float,
    distances_m: np.ndarray | list[float] | tuple[float, ...],
) -> LookaheadGeometry:
    """Return deterministic centerline samples for a local planner.

    Distances are non-negative offsets from ``progress_m`` and are clipped at
    the end of the centerline.  The input order is preserved so callers can
    use a fixed-size horizon without special-casing the finish.
    """

    offsets = np.asarray(distances_m, dtype=np.float64)
    if offsets.ndim != 1 or len(offsets) == 0 or not np.isfinite(offsets).all() or np.any(offsets < 0):
        raise ValueError("distances_m must be a non-empty finite 1D sequence of non-negative values")
    if not math.isfinite(float(progress_m)):
        raise ValueError("progress_m must be finite")
    sampled_distances = np.clip(float(progress_m) + offsets, 0.0, geometry.cumulative_lengths[-1])
    samples = [_sample_at_progress(value, geometry) for value in sampled_distances]
    return LookaheadGeometry(
        distances_m=offsets.copy(),
        points=np.stack([sample[0] for sample in samples]),
        tangents=np.stack([sample[1] for sample in samples]),
        curvatures=np.asarray([sample[2] for sample in samples], dtype=np.float64),
    )


def progress_rate_mps(current_progress_m: float, previous_progress_m: float, dt_s: float) -> float:
    """Compute forward progress rate, clamping reverse/noisy motion to zero."""

    if not all(math.isfinite(float(value)) for value in (current_progress_m, previous_progress_m, dt_s)) or dt_s <= 0:
        raise ValueError("progress values must be finite and dt_s must be positive")
    return max(0.0, float(current_progress_m - previous_progress_m) / float(dt_s))


def curvature_target_speed_mps(
    curvature: float,
    *,
    max_speed_mps: float = 70.0,
    min_speed_mps: float = 12.0,
    curvature_gain: float = 28.0,
) -> float:
    """Produce a conservative speed target from curvature magnitude."""

    values = (curvature, max_speed_mps, min_speed_mps, curvature_gain)
    if not all(math.isfinite(float(value)) for value in values) or max_speed_mps < min_speed_mps or min_speed_mps < 0 or curvature_gain < 0:
        raise ValueError("invalid curvature speed parameters")
    return float(np.clip(max_speed_mps / (1.0 + curvature_gain * abs(float(curvature))), min_speed_mps, max_speed_mps))


def recovery_required(
    state: TrackRelativeState,
    *,
    lateral_limit_m: float = 8.0,
    heading_limit_rad: float = math.radians(55.0),
) -> bool:
    """Flag states where a nominal controller should prioritize rejoining."""

    if lateral_limit_m <= 0 or heading_limit_rad <= 0:
        raise ValueError("recovery limits must be positive")
    return abs(state.lateral_error_m) > lateral_limit_m or abs(state.heading_error_rad) > heading_limit_rad
