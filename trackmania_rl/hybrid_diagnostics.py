"""Pure diagnostics shared by the offline simulator and rollout analyzer.

The helpers in this module deliberately operate on small mappings instead of
TrackMania objects.  They make the distinction between horizontal tracking and
elevation explicit, and keep event/section reporting deterministic for both
simulated and recorded trajectories.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def _finite_or_none(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def _record_event(record: Mapping[str, Any], *, kind: str, reason: str) -> dict[str, Any]:
    """Return a JSON-safe event snapshot with the useful error context."""

    event: dict[str, Any] = {
        "kind": kind,
        "reason": reason,
        "step_index": int(record["step_index"]),
        "timestamp_s": float(record["timestamp_s"]),
        "progress_m": float(record["progress_m"]),
    }
    for source, target in (
        ("lateral_error_m", "lateral_error_m"),
        ("heading_error_rad", "heading_error_rad"),
        ("horizontal_distance_to_centerline_m", "horizontal_distance_to_centerline_m"),
        ("elevation_error_m", "elevation_error_m"),
    ):
        if source in record:
            event[target] = float(record[source])
    return event


def identify_divergence_recovery_events(
    records: Iterable[Mapping[str, Any]],
    *,
    lateral_limit_m: float = 8.0,
    heading_limit_rad: float = math.radians(55.0),
) -> dict[str, dict[str, Any] | None]:
    """Find the first divergence, recovery activation, and recovery completion.

    A divergence is the first sample outside the configured horizontal tracking
    limits (or explicitly marked as a controller recovery sample).  A recovery
    activation is the first ``recovery=True`` sample.  A recovery completion is
    the first later sample where recovery is no longer requested.  The aliases
    ``first_divergence`` and ``first_recovery`` provide a compact report shape;
    ``first_recovery`` prefers completion when it exists and otherwise reports
    activation, so an unfinished recovery remains visible.
    """

    if lateral_limit_m <= 0.0 or heading_limit_rad <= 0.0:
        raise ValueError("event limits must be positive")

    first_divergence: dict[str, Any] | None = None
    first_activation: dict[str, Any] | None = None
    first_completion: dict[str, Any] | None = None
    was_recovering = False
    for raw_record in records:
        record = raw_record
        recovering = bool(record.get("recovery", False))
        lateral = abs(float(record.get("lateral_error_m", 0.0)))
        heading = abs(float(record.get("heading_error_rad", 0.0)))
        if first_divergence is None and (recovering or lateral > lateral_limit_m or heading > heading_limit_rad):
            reason = "lateral_error" if lateral > lateral_limit_m else "heading_error" if heading > heading_limit_rad else "controller_recovery"
            first_divergence = _record_event(record, kind="divergence", reason=reason)
        if recovering and first_activation is None:
            first_activation = _record_event(record, kind="recovery_activation", reason="controller_recovery")
        if was_recovering and not recovering and first_completion is None:
            first_completion = _record_event(record, kind="recovery_completion", reason="tracking_rejoined")
        was_recovering = recovering

    return {
        "first_divergence": first_divergence,
        "first_divergence_event": first_divergence,
        "first_recovery": first_completion or first_activation,
        "first_recovery_activation": first_activation,
        "first_recovery_activation_event": first_activation,
        "first_recovery_completion": first_completion,
        "first_recovery_completion_event": first_completion,
    }


def summarize_progress_sections(
    records: Iterable[Mapping[str, Any]],
    total_length_m: float,
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    """Aggregate tracking diagnostics into fixed, deterministic progress bins."""

    if not math.isfinite(float(total_length_m)) or total_length_m <= 0.0:
        raise ValueError("total_length_m must be finite and positive")
    if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
        raise ValueError("bins must be a positive integer")

    grouped: list[list[Mapping[str, Any]]] = [[] for _ in range(bins)]
    for record in records:
        progress = float(record["progress_m"])
        fraction = min(max(progress / total_length_m, 0.0), 1.0)
        grouped[min(int(fraction * bins), bins - 1)].append(record)

    sections: list[dict[str, Any]] = []
    for index, section_records in enumerate(grouped):
        start_fraction = index / bins
        end_fraction = (index + 1) / bins
        horizontal = [abs(float(item.get("horizontal_distance_to_centerline_m", item.get("distance_to_centerline_m", 0.0)))) for item in section_records]
        elevation = [abs(float(item.get("elevation_error_m", 0.0))) for item in section_records]
        lateral = [abs(float(item.get("lateral_error_m", 0.0))) for item in section_records]
        heading = [abs(float(item.get("heading_error_rad", 0.0))) for item in section_records]
        recovery_count = sum(bool(item.get("recovery", False)) for item in section_records)
        sections.append(
            {
                "bin_index": index,
                "progress_fraction_start": start_fraction,
                "progress_fraction_end": end_fraction,
                "progress_start_m": start_fraction * total_length_m,
                "progress_end_m": end_fraction * total_length_m,
                "samples": len(section_records),
                "mean_abs_lateral_error_m": _finite_or_none(lateral),
                "max_abs_lateral_error_m": _max_or_none(lateral),
                "mean_horizontal_distance_to_centerline_m": _finite_or_none(horizontal),
                "max_horizontal_distance_to_centerline_m": _max_or_none(horizontal),
                "mean_abs_elevation_error_m": _finite_or_none(elevation),
                "max_abs_elevation_error_m": _max_or_none(elevation),
                "mean_abs_heading_error_rad": _finite_or_none(heading),
                "max_abs_heading_error_rad": _max_or_none(heading),
                "recovery_samples": recovery_count,
                "recovery_rate": float(recovery_count / len(section_records)) if section_records else 0.0,
            }
        )
    return sections
