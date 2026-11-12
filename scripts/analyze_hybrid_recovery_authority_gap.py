"""Quantify the yaw-authority and lead-time gap before backward motion.

This report consumes the existing first-backward-event feasibility report and
computes counterfactual toy-dynamics quantities only. It estimates the minimum
constant yaw rate needed to reach a non-negative along-track projection after
one maximum-braking step, plus the number of maximum-authority steps required
to become forward-aligned. It does not alter replay metrics, controller
defaults, or selection semantics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

DEFAULT_INPUT = Path("eval_results/hybrid_rollouts/hybrid_recovery_feasibility.json")
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_recovery_authority_gap.json")
DEFAULT_MAX_YAW_RATE_RAD_S = 1.8
DEFAULT_DT_S = 0.1
DEFAULT_MAX_BRAKING_MPS2 = 18.0
DEFAULT_DRAG_COEFFICIENT = 0.05
BACKWARD_TOLERANCE_M = 1e-5


def _finite(value: object, name: str) -> float:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _next_speed(speed_mps: float, dt_s: float, max_braking_mps2: float, drag: float) -> float:
    return max(0.0, speed_mps - max_braking_mps2 * dt_s - drag * speed_mps * dt_s)


def _required_yaw_rate_for_forward_projection(
    heading_error_rad: float,
    *,
    dt_s: float,
) -> float:
    """Minimum constant yaw rate to reach <=90° absolute error in one step."""

    return max(0.0, abs(heading_error_rad) - math.pi / 2.0) / dt_s


def _required_lead_steps(
    heading_error_rad: float,
    *,
    dt_s: float,
    max_yaw_rate_rad_s: float,
) -> int:
    remaining = max(0.0, abs(heading_error_rad) - math.pi / 2.0)
    return math.ceil(remaining / (max_yaw_rate_rad_s * dt_s)) if remaining else 0


def build_authority_gap_report(
    feasibility_report: Mapping[str, Any],
    *,
    dt_s: float = DEFAULT_DT_S,
    max_yaw_rate_rad_s: float = DEFAULT_MAX_YAW_RATE_RAD_S,
    max_braking_mps2: float = DEFAULT_MAX_BRAKING_MPS2,
    drag: float = DEFAULT_DRAG_COEFFICIENT,
) -> dict[str, Any]:
    dt = _finite(dt_s, "dt_s")
    yaw_rate = _finite(max_yaw_rate_rad_s, "max_yaw_rate_rad_s")
    braking = _finite(max_braking_mps2, "max_braking_mps2")
    drag_value = _finite(drag, "drag")
    if dt <= 0.0 or yaw_rate <= 0.0 or braking < 0.0 or drag_value < 0.0:
        raise ValueError("invalid dynamics parameters")
    source_records = feasibility_report.get("records")
    if not isinstance(source_records, list):
        raise TypeError("feasibility report records must be a list")

    records: list[dict[str, Any]] = []
    for source in source_records:
        if not isinstance(source, Mapping):
            raise TypeError("feasibility records must be objects")
        previous = source.get("previous_state")
        if not isinstance(previous, Mapping):
            raise TypeError("feasibility record must contain previous_state")
        speed = _finite(previous.get("speed_mps"), "speed_mps")
        heading = _finite(previous.get("heading_error_rad"), "heading_error_rad")
        next_speed = _next_speed(speed, dt, braking, drag_value)
        required_rate = _required_yaw_rate_for_forward_projection(heading, dt_s=dt)
        lead_steps = _required_lead_steps(heading, dt_s=dt, max_yaw_rate_rad_s=yaw_rate)
        lead_time = lead_steps * dt
        records.append(
            {
                "report": source.get("report"),
                "injection_step": source.get("injection_step"),
                "map": source.get("map"),
                "profile": source.get("profile"),
                "case_key": source.get("case_key"),
                "disturbance": source.get("disturbance"),
                "actual_backward_distance_m": source.get("actual_backward_distance_m"),
                "speed_mps": speed,
                "heading_error_rad": heading,
                "next_max_braking_speed_mps": next_speed,
                "required_one_step_yaw_rate_rad_s": required_rate,
                "required_yaw_rate_multiple_of_default": required_rate / yaw_rate,
                "max_authority_lead_steps": lead_steps,
                "max_authority_lead_time_s": lead_time,
                "one_step_yaw_authority_gap_rad": max(0.0, abs(heading) - math.pi / 2.0 - yaw_rate * dt),
                "forward_projection_possible_after_one_step": required_rate <= yaw_rate,
            }
        )

    def aggregate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rates = [float(item["required_one_step_yaw_rate_rad_s"]) for item in items]
        multiples = [float(item["required_yaw_rate_multiple_of_default"]) for item in items]
        lead = [int(item["max_authority_lead_steps"]) for item in items]
        gaps = [float(item["one_step_yaw_authority_gap_rad"]) for item in items]
        return {
            "event_count": len(items),
            "one_step_forward_projection_possible_count": sum(
                bool(item["forward_projection_possible_after_one_step"]) for item in items
            ),
            "one_step_forward_projection_possible_rate": (
                sum(bool(item["forward_projection_possible_after_one_step"]) for item in items) / len(items)
                if items
                else None
            ),
            "mean_required_yaw_rate_rad_s": sum(rates) / len(rates) if rates else None,
            "max_required_yaw_rate_rad_s": max(rates) if rates else None,
            "mean_required_yaw_rate_multiple": sum(multiples) / len(multiples) if multiples else None,
            "max_required_yaw_rate_multiple": max(multiples) if multiples else None,
            "mean_max_authority_lead_steps": sum(lead) / len(lead) if lead else None,
            "max_max_authority_lead_steps": max(lead) if lead else None,
            "mean_one_step_yaw_authority_gap_rad": sum(gaps) / len(gaps) if gaps else None,
        }

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_step: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["disturbance"])].append(record)
        by_step[str(record["injection_step"])].append(record)

    return {
        "scope": "offline_hybrid_recovery_authority_gap",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "independent_confirmation": False,
        "controller_defaults_modified": False,
        "dynamics_proxy": {
            "dt_s": dt,
            "max_yaw_rate_rad_s": yaw_rate,
            "max_braking_mps2": braking,
            "linear_drag_coefficient": drag_value,
            "forward_projection_condition": "absolute heading error <= 90 degrees after one step",
        },
        "records": records,
        "overall": aggregate(records),
        "by_disturbance": {name: aggregate(items) for name, items in sorted(grouped.items())},
        "by_injection_step": {name: aggregate(items) for name, items in sorted(by_step.items())},
        "interpretation": {
            "diagnostic_only": True,
            "source_metrics_unchanged": True,
            "next_action": "use the required lead-time envelope to design predictive pre-rotation; do not promote reactive bursts",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    report = build_authority_gap_report(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(report["records"])}, separators=(",", ":")))


if __name__ == "__main__":
    main()
