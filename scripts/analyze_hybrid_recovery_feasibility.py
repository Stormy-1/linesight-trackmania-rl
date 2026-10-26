"""Estimate whether first backward events are avoidable in the toy dynamics.

The diagnostic replays the existing boundary cases and compares each first
backward transition with a conservative maximum-authority one-step proxy:
full braking, no throttle, and the maximum yaw correction toward the local
track tangent. It does not change replay metrics or claim real TrackMania
physics. The result helps separate controller design limits from an
unavoidable consequence of the simplified simulator's bounded dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trackmania_rl.hybrid_simulator import ClosedLoopConfig

try:
    from scripts.diagnose_hybrid_recovery_action_trace import DEFAULT_REPORTS, _load, _run_case
except ModuleNotFoundError:  # Direct script execution puts scripts first on sys.path.
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from scripts.diagnose_hybrid_recovery_action_trace import DEFAULT_REPORTS, _load, _run_case


DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_recovery_feasibility.json")
BACKWARD_PROGRESS_TOLERANCE_M = 1e-5


def maximum_authority_one_step(
    *,
    speed_mps: float,
    heading_error_rad: float,
    config: ClosedLoopConfig | None = None,
) -> dict[str, float | bool]:
    """Return a one-step maximum-brake/max-yaw feasibility proxy.

    The proxy assumes a locally constant centerline tangent. It applies the
    simulator's maximum braking and drag, then reduces absolute heading error
    by the maximum available yaw authority for one timestep. A negative
    projected displacement means that even this bounded one-step proxy remains
    reverse-aligned.
    """

    active = config or ClosedLoopConfig()
    speed = float(speed_mps)
    heading_error = float(heading_error_rad)
    values = (speed, heading_error)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("speed and heading error must be finite")
    if speed < 0.0:
        raise ValueError("speed must be non-negative")
    next_speed = max(
        0.0,
        speed - active.max_braking_mps2 * active.dt_s - active.linear_drag_mps2_per_mps * speed * active.dt_s,
    )
    next_abs_heading_error = max(0.0, abs(heading_error) - active.max_yaw_rate_rad_s * active.dt_s)
    displacement = next_speed * math.cos(next_abs_heading_error) * active.dt_s
    return {
        "next_speed_mps": float(next_speed),
        "next_abs_heading_error_rad": float(next_abs_heading_error),
        "estimated_along_tangent_displacement_m": float(displacement),
        "reverse_unavoidable_proxy": bool(displacement < -BACKWARD_PROGRESS_TOLERANCE_M),
    }


def _first_backward_transition(result, injection_step: int):
    previous = result.steps[injection_step]
    for current in result.steps[injection_step + 1 :]:
        distance = float(previous.state.progress_m - current.state.progress_m)
        if distance > BACKWARD_PROGRESS_TOLERANCE_M:
            return previous, current, distance
        previous = current
    return None


def _aggregate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    displacements = [float(item["feasibility"]["estimated_along_tangent_displacement_m"]) for item in items]
    actual = [float(item["actual_backward_distance_m"]) for item in items]
    return {
        "event_count": len(items),
        "reverse_unavoidable_proxy_count": sum(bool(item["feasibility"]["reverse_unavoidable_proxy"]) for item in items),
        "reverse_unavoidable_proxy_rate": (
            sum(bool(item["feasibility"]["reverse_unavoidable_proxy"]) for item in items) / len(items)
            if items
            else None
        ),
        "mean_estimated_authority_displacement_m": sum(displacements) / len(displacements) if displacements else None,
        "min_estimated_authority_displacement_m": min(displacements) if displacements else None,
        "max_estimated_authority_displacement_m": max(displacements) if displacements else None,
        "mean_actual_backward_distance_m": sum(actual) / len(actual) if actual else None,
        "max_actual_backward_distance_m": max(actual) if actual else None,
    }


def build_feasibility_report(report_paths: Sequence[Path] = DEFAULT_REPORTS) -> dict[str, Any]:
    if not report_paths:
        raise ValueError("at least one report is required")
    records: list[dict[str, Any]] = []
    config = ClosedLoopConfig()
    for report_path in report_paths:
        report = _load(report_path)
        injection_step = int(report["simulator"]["injection_step"])
        for map_name, map_entry in sorted(report["maps"].items()):
            for profile_name in sorted(map_entry["cases_by_profile"]):
                for case in map_entry["cases_by_profile"][profile_name]:
                    if case.get("split") != "held_out" or case.get("status") != "ok":
                        continue
                    if not str(case["disturbance"]).startswith("boundary_"):
                        continue
                    outcome = _run_case(report, map_name, profile_name, case)
                    transition = _first_backward_transition(outcome.result, injection_step)
                    if transition is None:
                        continue
                    previous, current, distance = transition
                    feasibility = maximum_authority_one_step(
                        speed_mps=previous.state.speed_mps,
                        heading_error_rad=previous.state.heading_error_rad,
                        config=config,
                    )
                    records.append(
                        {
                            "report": str(report_path),
                            "injection_step": injection_step,
                            "map": map_name,
                            "profile": profile_name,
                            "case_key": case["case_key"],
                            "disturbance": case["disturbance"],
                            "first_backward_step": current.step_index,
                            "actual_backward_distance_m": distance,
                            "previous_state": {
                                "progress_m": float(previous.state.progress_m),
                                "speed_mps": float(previous.state.speed_mps),
                                "heading_error_rad": float(previous.state.heading_error_rad),
                                "lateral_error_m": float(previous.state.lateral_error_m),
                            },
                            "current_state": {
                                "progress_m": float(current.state.progress_m),
                                "speed_mps": float(current.state.speed_mps),
                                "heading_error_rad": float(current.state.heading_error_rad),
                                "lateral_error_m": float(current.state.lateral_error_m),
                            },
                            "actual_command": {
                                "steering": float(previous.command.steering),
                                "throttle": float(previous.command.throttle),
                                "brake": float(previous.command.brake),
                                "recovery": bool(previous.command.recovery),
                            },
                            "feasibility": feasibility,
                        }
                    )

    by_disturbance: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_injection_step: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_disturbance[str(record["disturbance"])].append(record)
        by_injection_step[str(record["injection_step"])].append(record)
    return {
        "scope": "offline_hybrid_recovery_dynamic_feasibility_diagnostic",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "independent_confirmation": False,
        "controller_defaults_modified": False,
        "simulator_proxy": {
            "dt_s": config.dt_s,
            "max_braking_mps2": config.max_braking_mps2,
            "linear_drag_mps2_per_mps": config.linear_drag_mps2_per_mps,
            "max_yaw_rate_rad_s": config.max_yaw_rate_rad_s,
            "assumptions": [
                "full braking and zero throttle",
                "maximum yaw correction toward the local tangent",
                "locally constant centerline tangent for one step",
            ],
        },
        "records": records,
        "overall": _aggregate(records),
        "by_disturbance": {name: _aggregate(items) for name, items in sorted(by_disturbance.items())},
        "by_injection_step": {name: _aggregate(items) for name, items in sorted(by_injection_step.items())},
        "interpretation": {
            "diagnostic_only": True,
            "historical_metrics_unchanged": True,
            "next_action": "use feasibility evidence to decide whether earlier trajectory shaping is required; do not relax continuity",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_feasibility_report(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(report["records"])}, separators=(",", ":")))


if __name__ == "__main__":
    main()
