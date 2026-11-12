"""Evaluate bounded, evidence-derived speed profiles in the offline simulator.

The calibration input is a descriptive summary of captured telemetry.  This
script turns its section percentiles into experimental controller configs and
compares them with a read-only reference using the deterministic hybrid
simulator.  It does not launch TrackMania, training, Azure work, or load
checkpoints, and it never changes :class:`ConservativeControllerConfig`'s
defaults.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from trackmania_rl.hybrid_controller import ConservativeCenterlineController, ConservativeControllerConfig
from trackmania_rl.hybrid_simulator import ClosedLoopConfig, ClosedLoopResult, simulate_closed_loop, synthetic_centerline_geometry

DEFAULT_CALIBRATION = Path("eval_results/hybrid_rollouts/speed_profile_calibration.json")
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/speed_profile_candidate_evaluation.json")

# These bounds make the experiment deliberately smaller than the observed
# telemetry range.  They are also below the simulator's 80 m/s speed ceiling.
MAX_SPEED_BOUNDS = (50.0, 80.0)
MIN_SPEED_BOUNDS = (10.0, 18.0)
CURVATURE_GAIN_BOUNDS = (18.0, 40.0)


@dataclass(frozen=True)
class SpeedProfileCandidate:
    """A named experimental profile and the evidence used to derive it."""

    name: str
    description: str
    config: ConservativeControllerConfig
    evidence_basis: dict[str, float]


def _bounded(value: float, bounds: tuple[float, float]) -> float:
    if not math.isfinite(value):
        raise ValueError("candidate parameter must be finite")
    return round(float(np.clip(value, bounds[0], bounds[1])), 3)


def _section_values(calibration_report: Mapping[str, Any], field: str, percentile: str) -> list[float]:
    values: list[float] = []
    traces = calibration_report.get("traces")
    if not isinstance(traces, list):
        raise TypeError("calibration report must contain a traces list")
    for trace in traces:
        if not isinstance(trace, Mapping) or not isinstance(trace.get("sections"), list):
            raise TypeError("each calibration trace must contain sections")
        for section in trace["sections"]:
            try:
                value = float(section[field][percentile])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"calibration section is missing {field}.{percentile}") from error
            if not math.isfinite(value):
                raise ValueError(f"calibration section contains non-finite {field}.{percentile}")
            values.append(value)
    if not values:
        raise ValueError("calibration report contains no section percentile values")
    return values


def _evidence_summary(calibration_report: Mapping[str, Any]) -> dict[str, float]:
    actual_p25 = median(_section_values(calibration_report, "actual_speed_mps", "p25"))
    actual_p50 = median(_section_values(calibration_report, "actual_speed_mps", "p50"))
    actual_p75 = median(_section_values(calibration_report, "actual_speed_mps", "p75"))
    nominal_p25 = median(_section_values(calibration_report, "nominal_target_speed_mps", "p25"))
    nominal_p50 = median(_section_values(calibration_report, "nominal_target_speed_mps", "p50"))
    nominal_p75 = median(_section_values(calibration_report, "nominal_target_speed_mps", "p75"))
    curvature_p50 = median(_section_values(calibration_report, "absolute_curvature", "p50"))
    return {
        "observed_actual_speed_p25_mps": float(actual_p25),
        "observed_actual_speed_p50_mps": float(actual_p50),
        "observed_actual_speed_p75_mps": float(actual_p75),
        "observed_nominal_target_p25_mps": float(nominal_p25),
        "observed_nominal_target_p50_mps": float(nominal_p50),
        "observed_nominal_target_p75_mps": float(nominal_p75),
        "observed_absolute_curvature_p50": float(curvature_p50),
    }


def derive_candidates(calibration_report: Mapping[str, Any]) -> tuple[SpeedProfileCandidate, ...]:
    """Derive bounded profiles from captured section percentile evidence.

    The formulas intentionally use lower fractions of captured actual speeds;
    observed policy speed is evidence for scale, not a claim of safe target
    speed.  The returned configs are passed explicitly to a controller and do
    not mutate the live/default controller configuration.
    """

    evidence = _evidence_summary(calibration_report)
    base = ConservativeControllerConfig()
    curvature = evidence["observed_absolute_curvature_p50"]
    evidence_gain = _bounded(24.0 + 12.0 * curvature, CURVATURE_GAIN_BOUNDS)
    nominal_floor = _bounded(evidence["observed_nominal_target_p50_mps"], MIN_SPEED_BOUNDS)
    candidates = (
        SpeedProfileCandidate(
            name="captured_p25_guardrail",
            description="Lower-speed guardrail anchored to the captured actual-speed p25.",
            config=replace(
                base,
                max_speed_mps=_bounded(0.80 * evidence["observed_actual_speed_p25_mps"], MAX_SPEED_BOUNDS),
                min_speed_mps=nominal_floor,
                curvature_gain=_bounded(1.15 * evidence_gain, CURVATURE_GAIN_BOUNDS),
            ),
            evidence_basis=evidence,
        ),
        SpeedProfileCandidate(
            name="captured_p50_balanced",
            description="Balanced experimental cap anchored to the captured actual-speed p50.",
            config=replace(
                base,
                max_speed_mps=_bounded(0.95 * evidence["observed_actual_speed_p50_mps"], MAX_SPEED_BOUNDS),
                min_speed_mps=nominal_floor,
                curvature_gain=evidence_gain,
            ),
            evidence_basis=evidence,
        ),
        SpeedProfileCandidate(
            name="captured_p75_exploratory",
            description="Upper exploratory cap anchored to the captured actual-speed p75.",
            config=replace(
                base,
                max_speed_mps=_bounded(0.95 * evidence["observed_actual_speed_p75_mps"], MAX_SPEED_BOUNDS),
                min_speed_mps=_bounded(evidence["observed_nominal_target_p75_mps"], MIN_SPEED_BOUNDS),
                curvature_gain=_bounded(0.85 * evidence_gain, CURVATURE_GAIN_BOUNDS),
            ),
            evidence_basis=evidence,
        ),
    )
    return candidates


def _mean_abs(values: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float64)))) if values else 0.0


def _scenario_metrics(result: ClosedLoopResult) -> dict[str, Any]:
    states = [step.state for step in result.steps]
    commands = [step.command for step in result.steps]
    lateral = [float(state.lateral_error_m) for state in states]
    heading = [float(state.heading_error_rad) for state in states]
    actual_speeds = [float(state.speed_mps) for state in states]
    targets = [float(command.target_speed_mps) for command in commands]
    gaps = [actual - target for actual, target in zip(actual_speeds, targets)]
    sample_count = max(len(commands), 1)
    return {
        "completion": {
            "finished": result.finished,
            "elevation_tolerant_finished": result.elevation_tolerant_finished,
            "termination_reason": result.termination_reason,
            "elapsed_s": result.elapsed_s,
        },
        "progress": {
            "progress_gain_m": result.progress_gain_m,
            "final_progress_m": result.final_state.progress_m,
            "track_length_m": result.track_length_m,
            "progress_fraction": float(result.final_state.progress_m / max(result.track_length_m, 1e-9)),
        },
        "lateral_error": {
            "mean_abs_m": _mean_abs(lateral),
            "max_abs_m": result.max_abs_lateral_error_m,
            "final_abs_m": abs(float(result.final_state.lateral_error_m)),
        },
        "heading_error": {
            "mean_abs_rad": _mean_abs(heading),
            "max_abs_rad": result.max_abs_heading_error_rad,
            "final_abs_rad": abs(float(result.final_state.heading_error_rad)),
        },
        "recovery": {
            "samples": result.recovery_samples,
            "rate": float(result.recovery_samples / sample_count),
            "first_recovery": result.first_recovery_event,
        },
        "speed_target_behavior": {
            "mean_actual_speed_mps": float(np.mean(actual_speeds)) if actual_speeds else 0.0,
            "mean_target_speed_mps": float(np.mean(targets)) if targets else 0.0,
            "mean_actual_minus_target_mps": float(np.mean(gaps)) if gaps else 0.0,
            "mean_abs_actual_minus_target_mps": _mean_abs(gaps),
            "fraction_actual_above_target": float(sum(gap > 0.0 for gap in gaps) / sample_count),
            "target_min_mps": min(targets) if targets else None,
            "target_max_mps": max(targets) if targets else None,
        },
    }


def _aggregate_metrics(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    completions = [bool(item["completion"]["finished"]) for item in scenarios]
    def mean(path: tuple[str, str]) -> float:
        return float(np.mean([float(item[path[0]][path[1]]) for item in scenarios]))

    return {
        "completion": {
            "finished_scenarios": sum(completions),
            "scenario_count": len(scenarios),
            "completion_rate": float(sum(completions) / max(len(scenarios), 1)),
        },
        "progress": {
            "mean_progress_gain_m": mean(("progress", "progress_gain_m")),
            "mean_final_progress_m": mean(("progress", "final_progress_m")),
            "mean_progress_fraction": mean(("progress", "progress_fraction")),
        },
        "lateral_error": {
            "mean_of_scenario_mean_abs_m": mean(("lateral_error", "mean_abs_m")),
            "worst_max_abs_m": max(float(item["lateral_error"]["max_abs_m"]) for item in scenarios),
        },
        "heading_error": {
            "mean_of_scenario_mean_abs_rad": mean(("heading_error", "mean_abs_rad")),
            "worst_max_abs_rad": max(float(item["heading_error"]["max_abs_rad"]) for item in scenarios),
        },
        "recovery": {
            "total_samples": sum(int(item["recovery"]["samples"]) for item in scenarios),
            "mean_rate": mean(("recovery", "rate")),
            "scenarios_with_recovery": sum(item["recovery"]["samples"] > 0 for item in scenarios),
        },
        "speed_target_behavior": {
            "mean_actual_speed_mps": mean(("speed_target_behavior", "mean_actual_speed_mps")),
            "mean_target_speed_mps": mean(("speed_target_behavior", "mean_target_speed_mps")),
            "mean_actual_minus_target_mps": mean(("speed_target_behavior", "mean_actual_minus_target_mps")),
            "mean_abs_actual_minus_target_mps": mean(("speed_target_behavior", "mean_abs_actual_minus_target_mps")),
            "mean_fraction_actual_above_target": mean(("speed_target_behavior", "fraction_actual_above_target")),
        },
    }


def _run_profile(name: str, role: str, description: str, config: ConservativeControllerConfig, geometry: Any) -> dict[str, Any]:
    initial_heading = geometry.segment_tangents[0]
    scenario_inputs = (
        ("nominal_centerline_start", geometry.points[0], initial_heading, 18.0),
        ("off_center_recovery_start", np.asarray([0.0, 0.0, 12.0]), np.asarray([np.cos(0.3), 0.0, np.sin(0.3)]), 16.0),
    )
    simulator_config = ClosedLoopConfig(dt_s=0.1, max_steps=1600)
    scenarios = []
    for scenario_name, position, heading, speed in scenario_inputs:
        result = simulate_closed_loop(
            geometry,
            initial_position=position,
            initial_heading=heading,
            initial_speed_mps=speed,
            controller=ConservativeCenterlineController(config),
            config=simulator_config,
        )
        scenarios.append({"name": scenario_name, **_scenario_metrics(result)})
    return {
        "name": name,
        "role": role,
        "description": description,
        "config": asdict(config),
        "scenarios": scenarios,
        "summary": _aggregate_metrics(scenarios),
    }


def build_report(calibration_report: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the live-default reference and derived profiles offline."""

    candidates = derive_candidates(calibration_report)
    geometry = synthetic_centerline_geometry(length_m=120.0, spacing_m=2.0, amplitude_m=6.0, wavelength_m=70.0)
    profiles = [
        _run_profile(
            "live_defaults_reference",
            "reference",
            "Current ConservativeControllerConfig defaults; included for comparison only.",
            ConservativeControllerConfig(),
            geometry,
        )
    ]
    for candidate in candidates:
        profiles.append(_run_profile(candidate.name, "experimental_candidate", candidate.description, candidate.config, geometry))
        profiles[-1]["evidence_basis"] = candidate.evidence_basis

    return {
        "scope": "offline_bounded_speed_profile_candidate_evaluation",
        "controller_defaults_modified": False,
        "trackmania_launched": False,
        "training_started": False,
        "azure_started": False,
        "checkpoints_loaded": False,
        "candidate_profiles_are_descriptive_only": True,
        "centerline": {"length_m": float(geometry.cumulative_lengths[-1]), "points": len(geometry.points)},
        "evidence_summary": _evidence_summary(calibration_report),
        "profiles": profiles,
        "limitations": [
            "The calibration input contains section percentiles from captured RL telemetry, not a raw speed map.",
            "The simulator is a bounded kinematic model and does not represent TrackMania physics, jumps, collisions, or TMInterface.",
            "Results compare deterministic software-in-the-loop behavior only; they are not live controller or real-map validation.",
        ],
    }


def load_calibration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("calibration report must be a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(load_calibration(args.calibration))
    report["calibration_input"] = str(args.calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "profiles": len(report["profiles"]), "all_finished": all(profile["summary"]["completion"]["completion_rate"] == 1.0 for profile in report["profiles"])}, indent=2))


if __name__ == "__main__":
    main()
