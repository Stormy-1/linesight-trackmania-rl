"""Calibrate heading and map-projection conventions on saved real telemetry.

This is an offline diagnostic.  It compares the current 3D projection with
horizontal XZ map matching and compares the captured velocity heading with an
offline finite-difference heading.  It does not change the live telemetry
hook, controller defaults, checkpoints, or any action path.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_rollout import (
    RolloutSample,
    derive_headings_from_positions,
    load_rollout_csv,
)
from trackmania_rl.hybrid_state import (
    CenterlineGeometry,
    build_centerline_geometry,
    project_to_centerline,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = (
    (
        "real_retry",
        ROOT / "maps/A01-Race_0.5m_cl2.npy",
        ROOT / "eval_results/hybrid_rollouts/A01-Race.Challenge_1793657875306_45728.csv",
    ),
    (
        "real_repeat_1",
        ROOT / "maps/A01-Race_0.5m_cl2.npy",
        ROOT / "eval_results/hybrid_rollouts/A01-Race.Challenge_1793658034903_19052.csv",
    ),
    (
        "real_repeat_2",
        ROOT / "maps/A01-Race_0.5m_cl2.npy",
        ROOT / "eval_results/hybrid_rollouts/A01-Race.Challenge_1793658041712_19052.csv",
    ),
    (
        "real_repeat_3",
        ROOT / "maps/A01-Race_0.5m_cl2.npy",
        ROOT / "eval_results/hybrid_rollouts/A01-Race.Challenge_1793658048487_19052.csv",
    ),
)

STATIONARY_SPEED_EPSILON = 1e-6

# ``current_3d`` is the convention used by the existing structured-state
# analyzer.  The other entries are calibration candidates only.
PROJECTION_CANDIDATES: dict[str, tuple[tuple[int, ...] | None, tuple[int, int]]] = {
    "current_3d": (None, (0, 2)),
    "xz_horizontal": ((0, 2), (0, 2)),
    "xy": ((0, 1), (0, 1)),
    "yz": ((1, 2), (1, 2)),
}


def _finite_summary(values: Iterable[float | None]) -> dict[str, float | None]:
    finite_values = [float(value) for value in values if value is not None]
    array = np.asarray(finite_values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _heading_error_in_axes(
    heading: np.ndarray,
    tangent: np.ndarray,
    axes: tuple[int, int],
) -> float | None:
    vehicle = np.asarray(heading, dtype=np.float64)[list(axes)]
    track = np.asarray(tangent, dtype=np.float64)[list(axes)]
    vehicle_norm = float(np.linalg.norm(vehicle))
    track_norm = float(np.linalg.norm(track))
    if vehicle_norm <= 1e-9 or track_norm <= 1e-9:
        return None
    vehicle /= vehicle_norm
    track /= track_norm
    cross = track[0] * vehicle[1] - track[1] * vehicle[0]
    return _wrap_angle(math.atan2(float(cross), float(np.dot(track, vehicle))))


def _project_in_axes(
    position: np.ndarray,
    geometry: CenterlineGeometry,
    axes: tuple[int, ...] | None,
    *,
    previous_segment: int | None,
    search_radius: int | None,
) -> tuple[int, float, np.ndarray, np.ndarray, float]:
    """Project a pose using a candidate metric while retaining 3D output."""

    if axes is None:
        return project_to_centerline(
            position,
            geometry,
            previous_segment=previous_segment,
            search_radius=search_radius,
        )

    point = np.asarray(position, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("position must contain three finite values")
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

    starts = geometry.points[candidates][:, list(axes)]
    vectors = geometry.segment_vectors[candidates][:, list(axes)]
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-9):
        raise ValueError(f"projection axes {axes} contain a zero-length segment")
    metric_point = point[list(axes)]
    fractions = np.einsum("ij,ij->i", metric_point - starts, vectors) / (lengths**2)
    fractions = np.clip(fractions, 0.0, 1.0)
    metric_projections = starts + vectors * fractions[:, None]
    distances = np.linalg.norm(metric_point - metric_projections, axis=1)
    local_index = int(np.argmin(distances))
    segment_index = int(candidates[local_index])
    fraction = float(fractions[local_index])
    projection = geometry.points[segment_index] + geometry.segment_vectors[segment_index] * fraction
    return (
        segment_index,
        fraction,
        projection,
        geometry.segment_tangents[segment_index],
        float(distances[local_index]),
    )


def evaluate_samples(
    samples: list[RolloutSample],
    geometry: CenterlineGeometry,
    *,
    source_name: str,
    projection_name: str,
    search_radius: int | None = 32,
) -> dict[str, Any]:
    """Evaluate one heading source and one projection convention."""

    if source_name not in {"recorded", "finite_difference"}:
        raise ValueError("source_name must be recorded or finite_difference")
    if projection_name not in PROJECTION_CANDIDATES:
        raise ValueError(f"unknown projection candidate: {projection_name}")
    if not samples:
        raise ValueError("samples must not be empty")

    source_samples = samples if source_name == "recorded" else derive_headings_from_positions(samples)
    projection_axes, heading_axes = PROJECTION_CANDIDATES[projection_name]
    previous_segment: int | None = None
    progress: list[float] = []
    metric_distances: list[float] = []
    horizontal_distances: list[float] = []
    heading_errors: list[float | None] = []
    stationary_indices: list[int] = []

    for index, sample in enumerate(source_samples):
        segment, fraction, projection, tangent, metric_distance = _project_in_axes(
            sample.position,
            geometry,
            projection_axes,
            previous_segment=previous_segment,
            search_radius=search_radius if previous_segment is not None else None,
        )
        progress.append(float(geometry.cumulative_lengths[segment] + fraction * geometry.segment_lengths[segment]))
        metric_distances.append(metric_distance)
        horizontal_distances.append(float(np.linalg.norm((sample.position - projection)[[0, 2]])))
        heading_error = _heading_error_in_axes(sample.heading, tangent, heading_axes)
        heading_errors.append(abs(heading_error) if heading_error is not None else None)
        if sample.speed_mps <= STATIONARY_SPEED_EPSILON:
            stationary_indices.append(index)
        previous_segment = segment

    progress_array = np.asarray(progress, dtype=np.float64)
    progress_deltas = np.diff(progress_array)
    moving_indices = [
        index
        for index in range(len(samples))
        if index not in stationary_indices and heading_errors[index] is not None
    ]
    moving_heading_errors = [heading_errors[index] for index in moving_indices]
    elapsed = np.diff(np.asarray([sample.timestamp_s for sample in samples], dtype=np.float64))
    speeds = np.asarray([sample.speed_mps for sample in samples], dtype=np.float64)
    integrated_speed_distance = float(np.sum(speeds[:-1] * elapsed)) if len(samples) > 1 else 0.0
    return {
        "source": source_name,
        "projection": projection_name,
        "sample_count": len(samples),
        "stationary_sample_count": len(stationary_indices),
        "stationary_sample_indices": stationary_indices,
        "first_sample_abs_heading_error_rad": heading_errors[0],
        "first_moving_sample_index": moving_indices[0] if moving_indices else None,
        "first_moving_abs_heading_error_rad": (
            heading_errors[moving_indices[0]] if moving_indices else None
        ),
        "heading_error_all_rad": _finite_summary(heading_errors),
        "heading_error_moving_rad": _finite_summary(moving_heading_errors),
        "invalid_heading_sample_count": sum(value is None for value in heading_errors),
        "moving_heading_sample_count": len(moving_heading_errors),
        "metric_distance_m": _finite_summary(metric_distances),
        "horizontal_xz_distance_m": _finite_summary(horizontal_distances),
        "progress_start_m": float(progress_array[0]),
        "progress_end_m": float(progress_array[-1]),
        "progress_gain_m": float(progress_array[-1] - progress_array[0]),
        "backward_progress_steps": int(np.sum(progress_deltas < 0.0)),
        "max_backward_progress_m": float(max(0.0, -float(np.min(progress_deltas))))
        if len(progress_deltas)
        else 0.0,
        "integrated_speed_distance_m": integrated_speed_distance,
        "progress_to_integrated_speed_ratio": (
            float((progress_array[-1] - progress_array[0]) / integrated_speed_distance)
            if integrated_speed_distance > 1e-9
            else None
        ),
    }


def _source_projection_summary(
    case_reports: list[dict[str, Any]],
    *,
    source_name: str,
    projection_name: str,
) -> dict[str, Any]:
    selected = [
        report
        for report in case_reports
        if report["source"] == source_name and report["projection"] == projection_name
    ]
    if not selected:
        raise ValueError("no case reports matched summary selection")
    total_samples = sum(int(report["sample_count"]) for report in selected)
    total_stationary = sum(int(report["stationary_sample_count"]) for report in selected)
    valid_heading_counts = [
        int(report["sample_count"]) - int(report["invalid_heading_sample_count"])
        for report in selected
    ]
    valid_heading_total = sum(valid_heading_counts)
    weighted_heading = (
        sum(
            float(report["heading_error_all_rad"]["mean"]) * count
            for report, count in zip(selected, valid_heading_counts)
            if count and report["heading_error_all_rad"]["mean"] is not None
        )
        / valid_heading_total
        if valid_heading_total
        else None
    )
    moving_counts = [int(report["moving_heading_sample_count"]) for report in selected]
    moving_total = sum(moving_counts)
    moving_heading = (
        sum(
            float(report["heading_error_moving_rad"]["mean"]) * count
            for report, count in zip(selected, moving_counts)
            if count and report["heading_error_moving_rad"]["mean"] is not None
        )
        / moving_total
        if moving_total
        else None
    )
    horizontal_mean = sum(
        float(report["horizontal_xz_distance_m"]["mean"]) * int(report["sample_count"])
        for report in selected
    ) / total_samples
    return {
        "source": source_name,
        "projection": projection_name,
        "cases": len(selected),
        "samples": total_samples,
        "stationary_samples": total_stationary,
        "stationary_sample_rate": total_stationary / total_samples,
        "invalid_heading_samples": sum(int(report["invalid_heading_sample_count"]) for report in selected),
        "mean_abs_heading_error_all_rad": weighted_heading,
        "mean_abs_heading_error_moving_rad": moving_heading,
        "mean_horizontal_xz_distance_m": horizontal_mean,
        "total_backward_progress_steps": sum(int(report["backward_progress_steps"]) for report in selected),
        "mean_progress_gain_m": float(np.mean([report["progress_gain_m"] for report in selected])),
    }


def _heading_source_agreement(samples: list[RolloutSample]) -> dict[str, float | None]:
    finite_difference = derive_headings_from_positions(samples)
    angles: list[float] = []
    for recorded, repaired in zip(samples, finite_difference):
        if recorded.speed_mps <= STATIONARY_SPEED_EPSILON:
            continue
        angle = _heading_error_in_axes(recorded.heading, repaired.heading, (0, 2))
        if angle is not None:
            angles.append(abs(angle))
    summary = _finite_summary(angles)
    return {
        "samples_compared": len(angles),
        "mean_abs_xz_angle_rad": summary["mean"],
        "median_abs_xz_angle_rad": summary["median"],
        "p95_abs_xz_angle_rad": summary["p95"],
    }


def evaluate_case(
    name: str,
    centerline_path: Path,
    rollout_path: Path,
    *,
    search_radius: int | None = 32,
) -> dict[str, Any]:
    geometry = build_centerline_geometry(np.load(centerline_path, allow_pickle=False))
    samples = load_rollout_csv(rollout_path)
    reports = [
        evaluate_samples(
            samples,
            geometry,
            source_name=source,
            projection_name=projection,
            search_radius=search_radius,
        )
        for source in ("recorded", "finite_difference")
        for projection in PROJECTION_CANDIDATES
    ]
    return {
        "name": name,
        "centerline": str(centerline_path),
        "rollout": str(rollout_path),
        "samples": len(samples),
        "duration_s": float(samples[-1].timestamp_s - samples[0].timestamp_s),
        "heading_source_agreement": _heading_source_agreement(samples),
        "evaluations": reports,
    }


def build_report(
    cases: tuple[tuple[str, Path, Path], ...] = DEFAULT_CASES,
    *,
    search_radius: int | None = 32,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("at least one calibration case is required")
    case_reports = [
        evaluate_case(name, centerline, rollout, search_radius=search_radius)
        for name, centerline, rollout in cases
    ]
    flat_reports = [report for case in case_reports for report in case["evaluations"]]
    summaries = [
        _source_projection_summary(
            flat_reports,
            source_name=source,
            projection_name=projection,
        )
        for source in ("recorded", "finite_difference")
        for projection in PROJECTION_CANDIDATES
    ]
    current = next(
        item
        for item in summaries
        if item["source"] == "recorded" and item["projection"] == "current_3d"
    )
    xz = next(
        item
        for item in summaries
        if item["source"] == "recorded" and item["projection"] == "xz_horizontal"
    )
    return {
        "scope": "offline_real_heading_projection_calibration",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "real_capture_inputs": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "actions_modified": False,
        "controller_defaults_modified": False,
        "future_disturbance_information_used": False,
        "current_convention": {
            "projection": "3D nearest-segment projection",
            "heading_plane": "X/Z",
            "stationary_heading_fallback": "rotation matrix first row",
        },
        "cases": case_reports,
        "aggregate_summaries": summaries,
        "interpretation": {
            "stationary_heading_artifact": {
                "observed": all(
                    case["evaluations"][0]["first_sample_abs_heading_error_rad"] > math.radians(80.0)
                    for case in case_reports
                ),
                "meaning": "The zero-speed fallback heading should not be treated as a driving divergence.",
                "recommended_analysis_policy": "Exclude stationary samples from heading-event calibration or initialize from the first moving heading.",
            },
            "moving_heading_source": {
                "observation": "The captured moving heading is a close X/Z match to finite-difference motion.",
                "recommended_source": "Keep the captured world-velocity heading for moving samples; use a separate stationary policy.",
            },
            "projection": {
                "observation": "XZ horizontal matching reduced mean X/Z distance and did not increase aggregate backward-step count on these captures.",
                "current_mean_horizontal_xz_distance_m": current["mean_horizontal_xz_distance_m"],
                "xz_mean_horizontal_xz_distance_m": xz["mean_horizontal_xz_distance_m"],
                "current_backward_steps": current["total_backward_progress_steps"],
                "xz_backward_steps": xz["total_backward_progress_steps"],
                "recommended_next_calibration": "Treat X/Z horizontal projection as the leading diagnostic convention, but do not change live analysis or control defaults from this evidence alone.",
            },
            "limitations": [
                "The captures contain position, velocity-heading proxy, and speed but not the raw rotation matrix or action transitions.",
                "The finite-difference heading is a motion-direction comparison, not an independent vehicle-forward ground truth.",
                "The four traces are repeated runs on one map and are not held-out real-physics validation.",
            ],
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "eval_results/hybrid_rollouts/real_heading_projection_calibration.json")
    parser.add_argument("--search-radius", type=int, default=32)
    args = parser.parse_args()
    report = build_report(search_radius=args.search_radius)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"])}, separators=(",", ":")))


if __name__ == "__main__":
    main()
