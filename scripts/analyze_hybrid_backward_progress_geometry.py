"""Classify backward-progress events by local along-track velocity.

This diagnostic consumes the existing recovery-action trace report.  It does
not change replay metrics or rerun a controller.  At each first backward
event it estimates the vehicle velocity component along the local centerline
tangent from speed and heading error, helping distinguish reverse-aligned
motion from a mostly lateral projection change.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

DEFAULT_INPUT = Path("eval_results/hybrid_rollouts/hybrid_recovery_action_trace.json")
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_backward_progress_geometry.json")
DEFAULT_DT_S = 0.1
DEFAULT_NEGATIVE_THRESHOLD_MPS = 1.0


def _finite(value: object, name: str) -> float:
    try:
        converted = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _classify(value: float, negative_threshold_mps: float) -> str:
    if value < -negative_threshold_mps:
        return "negative_along_tangent"
    if value > negative_threshold_mps:
        return "positive_along_tangent"
    return "near_zero_along_tangent"


def build_geometry_report(
    trace_report: Mapping[str, Any],
    *,
    dt_s: float = DEFAULT_DT_S,
    negative_threshold_mps: float = DEFAULT_NEGATIVE_THRESHOLD_MPS,
) -> dict[str, Any]:
    dt = _finite(dt_s, "dt_s")
    threshold = _finite(negative_threshold_mps, "negative_threshold_mps")
    if dt <= 0.0 or threshold <= 0.0:
        raise ValueError("dt_s and negative_threshold_mps must be positive")
    records = trace_report.get("records")
    if not isinstance(records, list):
        raise TypeError("trace report records must be a list")

    events: list[dict[str, Any]] = []
    by_disturbance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("trace records must be objects")
        historical = record.get("historical_metrics")
        trace = record.get("trace")
        if not isinstance(historical, Mapping) or not isinstance(trace, Mapping):
            raise TypeError("trace records must contain historical_metrics and trace objects")
        if bool(historical.get("progress_continuity", True)):
            continue
        event = trace.get("first_backward_event")
        if not isinstance(event, Mapping):
            raise TypeError("backward trace records must contain first_backward_event")
        state = event.get("state")
        if not isinstance(state, Mapping):
            raise TypeError("first_backward_event must contain a state object")
        speed = _finite(state.get("speed_mps"), "speed_mps")
        heading_error = _finite(state.get("heading_error_rad"), "heading_error_rad")
        backward_distance = _finite(historical.get("backward_progress_distance_m"), "backward_progress_distance_m")
        along_tangent_speed = speed * math.cos(heading_error)
        item = {
            "map": str(record.get("map", "")),
            "profile": str(record.get("profile", "")),
            "disturbance": str(record.get("disturbance", "")),
            "injection_step": int(record.get("injection_step", 0)),
            "case_key": str(record.get("case_key", "")),
            "first_backward_step": int(event.get("step_index", 0)),
            "speed_mps": speed,
            "heading_error_rad": heading_error,
            "estimated_along_tangent_speed_mps": along_tangent_speed,
            "estimated_one_step_along_tangent_displacement_m": along_tangent_speed * dt,
            "backward_progress_distance_m": backward_distance,
            "classification": _classify(along_tangent_speed, threshold),
        }
        events.append(item)
        by_disturbance[item["disturbance"]].append(item)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        values = [float(item["estimated_along_tangent_speed_mps"]) for item in items]
        distances = [float(item["backward_progress_distance_m"]) for item in items]
        classifications = {name: sum(item["classification"] == name for item in items) for name in (
            "negative_along_tangent",
            "near_zero_along_tangent",
            "positive_along_tangent",
        )}
        return {
            "event_count": len(items),
            "negative_along_tangent_count": classifications["negative_along_tangent"],
            "near_zero_along_tangent_count": classifications["near_zero_along_tangent"],
            "positive_along_tangent_count": classifications["positive_along_tangent"],
            "negative_or_near_zero_rate": (
                (classifications["negative_along_tangent"] + classifications["near_zero_along_tangent"]) / len(items)
                if items
                else None
            ),
            "mean_estimated_along_tangent_speed_mps": sum(values) / len(values) if values else None,
            "min_estimated_along_tangent_speed_mps": min(values) if values else None,
            "max_estimated_along_tangent_speed_mps": max(values) if values else None,
            "mean_backward_progress_distance_m": sum(distances) / len(distances) if distances else None,
            "max_backward_progress_distance_m": max(distances) if distances else None,
        }

    disturbance_summary = {
        disturbance: summarize(items)
        for disturbance, items in sorted(by_disturbance.items())
    }
    overall = summarize(events)
    return {
        "scope": "offline_hybrid_backward_progress_geometry_diagnostic",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "independent_confirmation": False,
        "controller_defaults_modified": False,
        "source_scope": "existing_hybrid_recovery_action_trace",
        "classification": {
            "dt_s": dt,
            "negative_threshold_mps": threshold,
            "interpretation": "speed times cosine of local heading error estimates motion along the centerline tangent",
        },
        "overall": overall,
        "by_disturbance": disturbance_summary,
        "events": events,
        "interpretation": {
            "diagnostic_only": True,
            "metrics_unchanged": True,
            "next_action": "use local velocity classification to prioritize recovery-action analysis; do not relax continuity",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--negative-threshold", type=float, default=DEFAULT_NEGATIVE_THRESHOLD_MPS)
    args = parser.parse_args()
    trace_report = json.loads(args.input.read_text(encoding="utf-8"))
    report = build_geometry_report(
        trace_report,
        dt_s=args.dt,
        negative_threshold_mps=args.negative_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "events": report["overall"]["event_count"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
