"""Evaluate the non-intervening predictive observer on saved telemetry.

The observer emits hysteresis-delimited feature warnings beside repaired A01/A07
telemetry. It never produces or modifies a command. This is the exact shadow
protocol intended for a later visible capture before any intervention is
considered.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.calibrate_hybrid_predictive_features import DEFAULT_CASES, _match_observations
except ModuleNotFoundError:  # Direct script execution puts scripts first on sys.path.
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from scripts.calibrate_hybrid_predictive_features import DEFAULT_CASES, _match_observations

from trackmania_rl.hybrid_observer import PredictiveObserverConfig, PredictiveShadowObserver
from trackmania_rl.hybrid_state import build_centerline_geometry

DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_shadow_observer.json")


def default_observer_configs() -> tuple[PredictiveObserverConfig, ...]:
    """Return diagnostic configurations; none is a promotion decision."""

    return (
        PredictiveObserverConfig(
            "speed_and_lateral_margin",
            speed_margin_threshold_mps=0.0,
            lateral_acceleration_margin_threshold_mps2=0.0,
        ),
        PredictiveObserverConfig(
            "tangent_change_only",
            tangent_change_threshold_rad=0.15,
        ),
        PredictiveObserverConfig(
            "forward_alignment_only",
            forward_alignment_margin_threshold_rad=0.0,
        ),
        PredictiveObserverConfig(
            "all_observable_signals",
            speed_margin_threshold_mps=0.0,
            lateral_acceleration_margin_threshold_mps2=0.0,
            tangent_change_threshold_rad=0.15,
            forward_alignment_margin_threshold_rad=0.0,
        ),
    )


def _event_metrics(observer_report: Mapping[str, Any], observations: Sequence[Any]) -> dict[str, Any]:
    divergence_index = next(
        (index for index, observation in enumerate(observations) if observation.recovery),
        None,
    )
    divergence_timestamp = observations[divergence_index].timestamp_s if divergence_index is not None else None
    events = list(observer_report["events"])
    pre_events = [event for event in events if divergence_index is None or event["start_index"] < divergence_index]
    # The compact observer report intentionally omits every step. Event bounds
    # are sufficient for this report; active durations are reconstructed from
    # event sample bounds when timestamps are available.
    active_before = sum(
        max(0, min(event["end_index"], divergence_index - 1) - event["start_index"] + 1)
        for event in events
        if divergence_index is not None and event["start_index"] < divergence_index
    )
    active_total = sum(event["end_index"] - event["start_index"] + 1 for event in events)
    return {
        "divergence_observed": divergence_index is not None,
        "divergence_index": divergence_index,
        "divergence_timestamp_s": divergence_timestamp,
        "event_count": len(events),
        "event_before_divergence": bool(pre_events),
        "first_event_start_index": events[0]["start_index"] if events else None,
        "first_event_lead_time_s": (
            divergence_timestamp - observations[events[0]["start_index"]].timestamp_s
            if events and divergence_timestamp is not None and events[0]["start_index"] < divergence_index
            else None
        ),
        "active_sample_count_from_events": active_total,
        "active_sample_count_before_divergence": active_before,
        "active_precision_before_divergence": active_before / active_total if active_total else None,
    }


def evaluate_case(
    name: str,
    centerline_path: Path,
    rollout_path: Path,
    *,
    configs: Sequence[PredictiveObserverConfig] = default_observer_configs(),
    search_radius: int | None = 32,
) -> dict[str, Any]:
    geometry = build_centerline_geometry(np.load(centerline_path, allow_pickle=False))
    observations, metadata = _match_observations(geometry, rollout_path, search_radius=search_radius)
    config_reports: list[dict[str, Any]] = []
    for config in configs:
        observer = PredictiveShadowObserver(config)
        for observation in observations:
            observer.observe(observation.sample_index, observation.timestamp_s, observation.features)
        observer_report = observer.report()
        observer_report.pop("config", None)
        observer_report["observer_name"] = config.name
        observer_report["event_metrics"] = _event_metrics(observer_report, observations)
        observer_report["signals_observed"] = sorted(
            {
                signal
                for event in observer_report["events"]
                for signal in event["signals"]
            }
        )
        config_reports.append(observer_report)
    return {
        "name": name,
        "centerline": str(centerline_path),
        "rollout": str(rollout_path),
        "telemetry": metadata,
        "observers": config_reports,
    }


def build_report(
    cases: Sequence[tuple[str, Path, Path]] = DEFAULT_CASES,
    *,
    configs: Sequence[PredictiveObserverConfig] = default_observer_configs(),
    search_radius: int | None = 32,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("at least one observer case is required")
    if not configs:
        raise ValueError("at least one observer config is required")
    calibrated = [
        evaluate_case(name, centerline, rollout, configs=configs, search_radius=search_radius)
        for name, centerline, rollout in cases
    ]
    return {
        "scope": "offline_hybrid_predictive_shadow_observer",
        "evidence_class": "diagnostic_development_validation",
        "offline_only": True,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "actions_modified": False,
        "controller_defaults_modified": False,
        "observer_only": True,
        "future_disturbance_information_used": False,
        "observer_configurations": [
            {
                "name": config.name,
                "speed_margin_threshold_mps": config.speed_margin_threshold_mps,
                "lateral_acceleration_margin_threshold_mps2": config.lateral_acceleration_margin_threshold_mps2,
                "tangent_change_threshold_rad": config.tangent_change_threshold_rad,
                "forward_alignment_margin_threshold_rad": config.forward_alignment_margin_threshold_rad,
                "activation_samples": config.activation_samples,
                "release_samples": config.release_samples,
            }
            for config in configs
        ],
        "cases": calibrated,
        "interpretation": {
            "diagnostic_only": True,
            "no_configuration_selected": True,
            "next_action": "run the same observer in shadow mode during authorized real capture; do not intervene",
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
