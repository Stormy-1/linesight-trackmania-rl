"""Calibrate observable planner warnings against saved rollout divergence.

This report map-matches repaired A01/A07 telemetry, derives the pure predictive
features, and measures whether candidate warnings precede the first divergence.
It does not change actions, controller defaults, telemetry, or live-control
paths. The calibration is evidence about the saved traces only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_planner import PredictiveRiskFeatures, predictive_risk_features
from trackmania_rl.hybrid_rollout import load_rollout_csv
from trackmania_rl.hybrid_state import CenterlineGeometry, build_centerline_geometry, make_track_relative_state

DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_predictive_feature_calibration.json")
DEFAULT_HORIZONS_S = (0.5, 1.0, 2.0)
DEFAULT_CASES = (
    (
        "A01-Race",
        Path("maps/A01-Race_0.5m_cl2.npy"),
        Path("eval_results/hybrid_rollouts_crossmap/A01-Race.repaired.csv"),
    ),
    (
        "A07-Race",
        Path("maps/A07-Race_0.5m_cl2.npy"),
        Path("eval_results/hybrid_rollouts_crossmap/A07-Race.repaired.csv"),
    ),
)


@dataclass(frozen=True)
class FeatureObservation:
    sample_index: int
    timestamp_s: float
    progress_m: float
    lateral_error_m: float
    heading_error_rad: float
    recovery: bool
    features: PredictiveRiskFeatures


def _finite(value: object, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _match_observations(
    centerline: CenterlineGeometry,
    rollout_path: Path,
    *,
    search_radius: int | None = 32,
) -> tuple[list[FeatureObservation], dict[str, Any]]:
    samples = load_rollout_csv(rollout_path)
    if not samples:
        raise ValueError("rollout must contain samples")
    observations: list[FeatureObservation] = []
    previous_segment: int | None = None
    first_divergence: int | None = None
    for sample_index, sample in enumerate(samples):
        state = make_track_relative_state(
            sample.position,
            sample.heading,
            sample.speed_mps,
            centerline,
            previous_segment=previous_segment,
            search_radius=search_radius if previous_segment is not None else None,
        )
        features = predictive_risk_features(state, centerline)
        observation = FeatureObservation(
            sample_index=sample_index,
            timestamp_s=float(sample.timestamp_s),
            progress_m=float(state.progress_m),
            lateral_error_m=float(state.lateral_error_m),
            heading_error_rad=float(state.heading_error_rad),
            recovery=abs(float(state.lateral_error_m)) > 8.0 or abs(float(state.heading_error_rad)) > math.radians(55.0),
            features=features,
        )
        observations.append(observation)
        if first_divergence is None and observation.recovery:
            first_divergence = sample_index
        previous_segment = state.segment_index
    metadata = {
        "sample_count": len(observations),
        "duration_s": float(observations[-1].timestamp_s - observations[0].timestamp_s),
        "first_divergence_index": first_divergence,
        "first_divergence_timestamp_s": (
            observations[first_divergence].timestamp_s if first_divergence is not None else None
        ),
        "first_divergence_progress_m": (
            observations[first_divergence].progress_m if first_divergence is not None else None
        ),
    }
    return observations, metadata


def _feature_value(observation: FeatureObservation, name: str) -> float:
    if name == "speed_margin_to_min_target_mps":
        return float(observation.features.speed_margin_to_min_target_mps)
    if name == "lateral_acceleration_margin_mps2":
        return float(observation.features.lateral_acceleration_margin_mps2)
    if name == "max_abs_tangent_change_rad":
        return float(observation.features.max_abs_tangent_change_rad)
    if name == "one_step_forward_alignment_margin_rad":
        return float(observation.features.one_step_forward_alignment_margin_rad)
    raise ValueError(f"unknown feature {name}")


def _warning(observation: FeatureObservation, feature: str, threshold: float) -> bool:
    return _feature_value(observation, feature) < threshold


def _summary_for_threshold(
    observations: Sequence[FeatureObservation],
    *,
    feature: str,
    threshold: float,
    horizon_s: float,
) -> dict[str, Any]:
    if not observations:
        raise ValueError("observations must not be empty")
    if horizon_s <= 0.0 or not math.isfinite(horizon_s):
        raise ValueError("horizon_s must be finite and positive")
    first_divergence = next(
        (
            index
            for index, observation in enumerate(observations)
            if observation.recovery
        ),
        None,
    )
    warnings = [index for index, observation in enumerate(observations) if _warning(observation, feature, threshold)]
    if first_divergence is None:
        return {
            "feature": feature,
            "threshold": threshold,
            "horizon_s": horizon_s,
            "warning_count": len(warnings),
            "first_warning_index": warnings[0] if warnings else None,
            "first_warning_timestamp_s": observations[warnings[0]].timestamp_s if warnings else None,
            "divergence_observed": False,
            "warning_before_divergence": None,
            "warning_within_horizon": None,
            "lead_time_s": None,
            "warning_precision_before_divergence": None,
        }
    divergence_time = observations[first_divergence].timestamp_s
    pre_divergence_warnings = [index for index in warnings if index < first_divergence]
    horizon_warnings = [
        index
        for index in pre_divergence_warnings
        if divergence_time - observations[index].timestamp_s <= horizon_s
    ]
    first_warning = warnings[0] if warnings else None
    warning_precision = len(pre_divergence_warnings) / len(warnings) if warnings else None
    return {
        "feature": feature,
        "threshold": threshold,
        "horizon_s": horizon_s,
        "warning_count": len(warnings),
        "warning_rate": len(warnings) / len(observations),
        "first_warning_index": first_warning,
        "first_warning_timestamp_s": observations[first_warning].timestamp_s if first_warning is not None else None,
        "divergence_observed": True,
        "divergence_index": first_divergence,
        "divergence_timestamp_s": divergence_time,
        "warning_before_divergence": bool(pre_divergence_warnings),
        "warning_within_horizon": bool(horizon_warnings),
        "lead_time_s": (
            divergence_time - observations[pre_divergence_warnings[0]].timestamp_s
            if pre_divergence_warnings
            else None
        ),
        "warning_precision_before_divergence": warning_precision,
    }


def _aggregate_case_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = [item for item in summaries if item.get("divergence_observed")]
    warned = [item for item in observed if item.get("warning_before_divergence")]
    within = [item for item in observed if item.get("warning_within_horizon")]
    leads = [float(item["lead_time_s"]) for item in warned if item.get("lead_time_s") is not None]
    precisions = [
        float(item["warning_precision_before_divergence"])
        for item in summaries
        if item.get("warning_precision_before_divergence") is not None
    ]
    return {
        "case_count": len(summaries),
        "divergence_observed_count": len(observed),
        "warning_before_divergence_count": len(warned),
        "warning_before_divergence_rate": len(warned) / len(observed) if observed else None,
        "warning_within_horizon_count": len(within),
        "warning_within_horizon_rate": len(within) / len(observed) if observed else None,
        "mean_lead_time_s": sum(leads) / len(leads) if leads else None,
        "minimum_lead_time_s": min(leads) if leads else None,
        "mean_warning_precision_before_divergence": sum(precisions) / len(precisions) if precisions else None,
    }


def calibrate_case(
    name: str,
    centerline_path: Path,
    rollout_path: Path,
    *,
    search_radius: int | None = 32,
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
) -> dict[str, Any]:
    centerline = build_centerline_geometry(np.load(centerline_path, allow_pickle=False))
    observations, metadata = _match_observations(centerline, rollout_path, search_radius=search_radius)
    thresholds = {
        "speed_margin_to_min_target_mps": (0.0, 2.0, 5.0),
        "lateral_acceleration_margin_mps2": (0.0, 2.0, 5.0),
        "max_abs_tangent_change_rad": (0.15, 0.30, 0.50),
        "one_step_forward_alignment_margin_rad": (0.0, 0.25, 0.50),
    }
    summaries = [
        _summary_for_threshold(
            observations,
            feature=feature,
            threshold=threshold,
            horizon_s=float(horizon),
        )
        for feature, feature_thresholds in thresholds.items()
        for threshold in feature_thresholds
        for horizon in horizons_s
    ]
    return {
        "name": name,
        "centerline": str(centerline_path),
        "rollout": str(rollout_path),
        "metadata": metadata,
        "thresholds": thresholds,
        "summaries": summaries,
        "aggregate_by_feature_threshold_horizon": {
            f"{feature}:threshold{threshold:g}:horizon{horizon:g}": _aggregate_case_summaries(
                [
                    item
                    for item in summaries
                    if item["feature"] == feature
                    and item["threshold"] == float(threshold)
                    and item["horizon_s"] == float(horizon)
                ]
            )
            for feature in thresholds
            for threshold in thresholds[feature]
            for horizon in horizons_s
        },
    }


def build_report(
    cases: Sequence[tuple[str, Path, Path]] = DEFAULT_CASES,
    *,
    search_radius: int | None = 32,
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("at least one calibration case is required")
    if search_radius is not None and search_radius < 0:
        raise ValueError("search_radius must be non-negative or None")
    calibrated = [
        calibrate_case(name, centerline, rollout, search_radius=search_radius, horizons_s=horizons_s)
        for name, centerline, rollout in cases
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in calibrated:
        for summary in case["summaries"]:
            key = (
                f"{summary['feature']}:threshold{summary['threshold']:g}:"
                f"horizon{summary['horizon_s']:g}"
            )
            grouped.setdefault(key, []).append(summary)
    return {
        "scope": "offline_hybrid_predictive_feature_calibration",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "controller_defaults_modified": False,
        "actions_modified": False,
        "future_disturbance_information_used": False,
        "cases": calibrated,
        "aggregate_by_feature_threshold_horizon": {
            key: _aggregate_case_summaries(items)
            for key, items in sorted(grouped.items())
        },
        "calibration_policy": {
            "divergence_definition": "absolute lateral error > 8m or absolute heading error > 55 degrees",
            "warning_definition": "candidate feature value below candidate threshold",
            "horizon_s": [float(value) for value in horizons_s],
            "saved_trace_limitations": [
                "A01 and A07 are repaired telemetry traces, not independent randomized episodes",
                "warning labels describe observed divergence timing, not causal intervention benefit",
                "feature thresholds must not be promoted without real closed-loop validation",
            ],
        },
        "interpretation": {
            "diagnostic_only": True,
            "next_action": "use warning lead-time and false-positive cost to select a narrow real-test observer; do not integrate automatically",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-radius", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(search_radius=args.search_radius)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"])}, separators=(",", ":")))


if __name__ == "__main__":
    main()
