"""Sweep recovery-only controller parameters on held-out offline pose replays.

The experiment uses the recorded A01/A07 telemetry only to select deterministic
source anchors.  Each fixed disturbance is then replayed through the saved
centerline and the pure kinematic simulator.  No recorded transition, live
game, training process, checkpoint, or external service is used.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from scripts.compare_hybrid_disturbed_replay import (
        DISTURBANCES,
        HELD_OUT_DISTURBANCES,
        MAP_FILES,
        select_source_anchors,
    )
    from scripts.evaluate_hybrid_speed_candidates import derive_candidates, load_calibration
except ModuleNotFoundError:  # Running this file directly puts ``scripts`` first on sys.path.
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from scripts.compare_hybrid_disturbed_replay import (
        DISTURBANCES,
        HELD_OUT_DISTURBANCES,
        MAP_FILES,
        select_source_anchors,
    )
    from scripts.evaluate_hybrid_speed_candidates import derive_candidates, load_calibration

from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineController,
    ConservativeControllerConfig,
    RecoveryAwareCenterlineController,
    RecoveryAwareCenterlineControllerConfig,
)
from trackmania_rl.hybrid_replay import disturb_sample, replay_disturbed_pose
from trackmania_rl.hybrid_rollout import RolloutSample, load_rollout_csv
from trackmania_rl.hybrid_simulator import ClosedLoopConfig
from trackmania_rl.hybrid_state import CenterlineGeometry, build_centerline_geometry, make_track_relative_state


DEFAULT_TELEMETRY_DIR = Path("eval_results/hybrid_rollouts_crossmap")
DEFAULT_MAPS_DIR = Path("maps")
DEFAULT_CALIBRATION = Path("eval_results/hybrid_rollouts/speed_profile_calibration.json")
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_recovery_parameter_sweep_harder_holdout.json")

DEFAULT_RECOVERY_GRID: dict[str, tuple[float, ...]] = {
    "recovery_heading_gain": (1.4, 2.2),
    "recovery_lateral_gain": (0.12, 0.24),
    "recovery_brake_bias": (0.25, 0.45),
}
MAX_GRID_CONFIGURATIONS = 16
BASELINE_DEFAULT = "live_defaults_reference"
BASELINE_GUARDRAIL = "captured_p25_guardrail"
MINIMUM_INFORMATIVE_ACTIVATED_CASES = 4
MINIMUM_INFORMATIVE_BASELINE_FAILURES = 1


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _token(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _configuration_id(parameters: Mapping[str, float]) -> str:
    return "recovery_h_{}_l_{}_brake_{}".format(
        _token(parameters["recovery_heading_gain"]),
        _token(parameters["recovery_lateral_gain"]),
        _token(parameters["recovery_brake_bias"]),
    )


def _validate_grid(
    recovery_heading_gains: Sequence[float],
    recovery_lateral_gains: Sequence[float],
    recovery_brake_biases: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    values_by_name = {
        "recovery_heading_gain": recovery_heading_gains,
        "recovery_lateral_gain": recovery_lateral_gains,
        "recovery_brake_bias": recovery_brake_biases,
    }
    normalized: dict[str, tuple[float, ...]] = {}
    for name, values in values_by_name.items():
        if not values:
            raise ValueError(f"{name} must contain at least one value")
        converted = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in converted):
            raise ValueError(f"{name} must contain only finite values")
        if any(value < 0.0 for value in converted):
            raise ValueError(f"{name} must contain only non-negative values")
        if name == "recovery_brake_bias" and any(value > 1.0 for value in converted):
            raise ValueError("recovery_brake_bias must be between 0 and 1")
        if len(set(converted)) != len(converted):
            raise ValueError(f"{name} must not contain duplicate values")
        normalized[name] = converted
    count = math.prod(len(values) for values in normalized.values())
    if count > MAX_GRID_CONFIGURATIONS:
        raise ValueError(f"parameter grid has {count} configurations; maximum is {MAX_GRID_CONFIGURATIONS}")
    return (
        normalized["recovery_heading_gain"],
        normalized["recovery_lateral_gain"],
        normalized["recovery_brake_bias"],
    )


def _split_source_anchors(
    anchors_by_map: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Assign complete source anchors to train or held-out by map.

    Documented divergence anchors are the preferred held-out regions.  Small
    fixtures often omit those targets, so the highest-progress source anchor
    is the deterministic fallback.  The returned assignment is anchor-level;
    every disturbance generated from an anchor inherits its assignment.
    """

    anchor_assignments: dict[str, dict[str, str]] = {}
    train_anchor_ids_by_map: dict[str, list[str]] = {}
    held_out_anchor_ids_by_map: dict[str, list[str]] = {}

    for map_name, anchors in anchors_by_map.items():
        ordered_anchors = sorted(anchors, key=lambda anchor: str(anchor["anchor_id"]))
        if not ordered_anchors:
            raise ValueError(f"map {map_name!r} must have at least one source anchor")

        documented = [
            anchor
            for anchor in ordered_anchors
            if anchor.get("anchor_role") == "documented_divergence_region"
            or bool(anchor.get("documented_divergence_region", False))
        ]
        held_out = documented or [
            max(
                ordered_anchors,
                key=lambda anchor: (float(anchor["source_progress_m"]), str(anchor["anchor_id"])),
            )
        ]
        held_out_ids = {str(anchor["anchor_id"]) for anchor in held_out}
        train = [anchor for anchor in ordered_anchors if str(anchor["anchor_id"]) not in held_out_ids]
        if not train or not held_out:
            raise ValueError(
                f"map {map_name!r} must have at least one train anchor and one held-out anchor "
                "for the source-region split"
            )

        assignments = {
            str(anchor["anchor_id"]): ("held_out" if str(anchor["anchor_id"]) in held_out_ids else "train")
            for anchor in ordered_anchors
        }
        anchor_assignments[str(map_name)] = assignments
        train_anchor_ids_by_map[str(map_name)] = sorted(
            anchor_id for anchor_id, split in assignments.items() if split == "train"
        )
        held_out_anchor_ids_by_map[str(map_name)] = sorted(held_out_ids)

    train_anchor_ids = sorted(
        anchor_id for anchors in train_anchor_ids_by_map.values() for anchor_id in anchors
    )
    held_out_anchor_ids = sorted(
        anchor_id for anchors in held_out_anchor_ids_by_map.values() for anchor_id in anchors
    )
    return anchor_assignments, {
        "policy": (
            "deterministic source-region split: documented_divergence_region anchors are held out per map; "
            "when a map has none, its highest-progress source anchor is held out; all disturbance families "
            "for an anchor share its assignment; held_out_only disturbance families are generated only on "
            "held-out anchors; split is shared by every profile"
        ),
        "train_anchor_ids": train_anchor_ids,
        "held_out_anchor_ids": held_out_anchor_ids,
        "train_anchor_ids_by_map": train_anchor_ids_by_map,
        "held_out_anchor_ids_by_map": held_out_anchor_ids_by_map,
        "anchor_assignments": anchor_assignments,
    }


def _selected_guardrail(calibration_report: Mapping[str, Any]):
    for candidate in derive_candidates(calibration_report):
        if candidate.name == BASELINE_GUARDRAIL:
            return candidate
    raise ValueError("captured_p25_guardrail was not derived from calibration")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _aggregate(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("cannot aggregate an empty case set")
    metrics = [case["metrics"] for case in cases]
    successes = [metric["recovery_success"] for metric in metrics if metric["recovery_success"] is not None]
    backward_counts = [int(metric["backward_progress_event_count"]) for metric in metrics]
    section_groups: dict[int, list[Mapping[str, Any]]] = {}
    for case in cases:
        section_groups.setdefault(int(case["source_anchor"]["source_section_bin"]), []).append(case)

    def section_aggregate(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result = _aggregate_without_sections(group)
        result["source_section_bin"] = int(group[0]["source_anchor"]["source_section_bin"])
        return result

    aggregate = _aggregate_without_sections(cases)
    aggregate["section_aggregates"] = [section_aggregate(section_groups[key]) for key in sorted(section_groups)]
    aggregate["simulator_section_aggregates"] = _simulator_section_aggregates(metrics)
    return aggregate


def _aggregate_without_sections(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [case["metrics"] for case in cases]
    successes = [metric["recovery_success"] for metric in metrics if metric["recovery_success"] is not None]
    activated = [metric for metric in metrics if metric["recovery_activated"]]
    success_count = sum(bool(value) for value in successes)
    stable_rejoins = [metric for metric in metrics if metric["stable_rejoin"]]
    recovery_latencies_samples = [
        int(metric["recovery_latency_samples"])
        for metric in metrics
        if metric["recovery_latency_samples"] is not None
    ]
    recovery_latencies_s = [
        float(metric["recovery_latency_s"])
        for metric in metrics
        if metric["recovery_latency_s"] is not None
    ]
    redivergence_counts = [int(metric["redivergence_count"]) for metric in metrics]
    backward_counts = [int(metric["backward_progress_event_count"]) for metric in metrics]
    progress_continuity_rate = _mean(bool(metric["progress_continuity"]) for metric in metrics)
    projected_completion_rate = _mean(bool(metric["progress_complete"]) for metric in metrics)
    mean_progress_fraction = _mean(float(metric["progress_fraction"]) for metric in metrics)
    return {
        "case_count": len(cases),
        "recovery_activated_count": sum(bool(metric["recovery_activated"]) for metric in metrics),
        "recovery_not_activated_count": sum(not bool(metric["recovery_activated"]) for metric in metrics),
        "recovery_completed_count": sum(bool(metric["recovery_completed"]) for metric in metrics),
        "recovery_success_observation_count": len(successes),
        "recovery_success_count": success_count,
        "recovery_failure_count": len(successes) - success_count,
        "conditional_recovery_success_rate": float(success_count / len(successes)) if successes else None,
        # Keep the historical field for consumers, while making its conditional
        # denominator explicit above.  Non-activated cases do not pass this
        # all-case measure.
        "recovery_success_rate": float(success_count / len(successes)) if successes else None,
        "all_case_recovery_pass_rate": float(success_count / len(cases)) if cases else None,
        "recovery_activation_rate": float(len(activated) / len(cases)) if cases else None,
        "effective_recovery_sample_size": len(successes),
        "stable_rejoin_count": len(stable_rejoins),
        "stable_rejoin_rate": float(len(stable_rejoins) / len(cases)) if cases else None,
        "stable_rejoin_observation_count": len(recovery_latencies_s),
        "mean_recovery_latency_samples": _mean(recovery_latencies_samples),
        "mean_recovery_latency_s": _mean(recovery_latencies_s),
        "redivergence_case_count": sum(count > 0 for count in redivergence_counts),
        "total_redivergence_count": int(sum(redivergence_counts)),
        "mean_abs_lateral_error_m": _mean(float(metric["mean_abs_lateral_error_m"]) for metric in metrics),
        "mean_horizontal_error_m": _mean(float(metric["mean_horizontal_error_m"]) for metric in metrics),
        "mean_abs_heading_error_rad": _mean(float(metric["mean_abs_heading_error_rad"]) for metric in metrics),
        "progress_continuity_rate": progress_continuity_rate,
        "progress_continuity": progress_continuity_rate,
        "backward_progress_event_rate": _mean(count > 0 for count in backward_counts),
        "mean_backward_progress_event_count": _mean(backward_counts),
        "total_backward_progress_event_count": int(sum(backward_counts)),
        "mean_backward_progress_distance_m": _mean(float(metric["backward_progress_distance_m"]) for metric in metrics),
        "mean_projected_progress_fraction": mean_progress_fraction,
        "mean_final_progress_fraction": mean_progress_fraction,
        "projected_progress_completion_rate": projected_completion_rate,
        "progress_complete_rate": projected_completion_rate,
        "projected_progress_completion": projected_completion_rate,
        "finished_rate": _mean(bool(metric["finished"]) for metric in metrics),
    }


def _simulator_section_aggregates(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for metric in metrics:
        for section in metric.get("section_metrics", []):
            grouped.setdefault(int(section["bin_index"]), []).append(section)
    return [
        {
            "bin_index": index,
            "samples": sum(int(item["samples"]) for item in items),
            "mean_abs_lateral_error_m": _mean(
                float(item["mean_abs_lateral_error_m"]) for item in items if item["mean_abs_lateral_error_m"] is not None
            ),
            "mean_horizontal_distance_to_centerline_m": _mean(
                float(item["mean_horizontal_distance_to_centerline_m"])
                for item in items
                if item["mean_horizontal_distance_to_centerline_m"] is not None
            ),
            "mean_abs_heading_error_rad": _mean(
                float(item["mean_abs_heading_error_rad"]) for item in items if item["mean_abs_heading_error_rad"] is not None
            ),
            "recovery_samples": sum(int(item["recovery_samples"]) for item in items),
        }
        for index, items in sorted(grouped.items())
    ]


def _profile_aggregate(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"case_count": 0, "section_aggregates": [], "simulator_section_aggregates": []}
    return _aggregate(cases)


def _rank_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank from training aggregates only; ties use the configuration id."""

    def key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        aggregate = candidate["train_aggregate"]
        success = aggregate["recovery_success_rate"]
        return (
            -(float(success) if success is not None else 0.0),
            -float(aggregate["progress_continuity_rate"]),
            -float(aggregate["projected_progress_completion_rate"]),
            float(aggregate["mean_abs_lateral_error_m"]),
            float(aggregate["mean_horizontal_error_m"]),
            float(aggregate["mean_abs_heading_error_rad"]),
            str(candidate["configuration_id"]),
        )

    ranked = sorted(candidates, key=key)
    return [dict(candidate, train_rank=index) for index, candidate in enumerate(ranked, start=1)]


def _selection(ranked: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any]) -> dict[str, Any]:
    baseline_aggregate = baseline["held_out_aggregate"]
    baseline_success = float(baseline_aggregate["recovery_success_rate"] or 0.0)
    baseline_continuity = float(baseline_aggregate["progress_continuity_rate"])
    baseline_completion = float(baseline_aggregate["projected_progress_completion_rate"])
    eligible: list[str] = []
    for candidate in ranked:
        aggregate = candidate["held_out_aggregate"]
        success = float(aggregate["recovery_success_rate"] or 0.0)
        if (
            success > baseline_success
            and float(aggregate["progress_continuity_rate"]) >= baseline_continuity
            and float(aggregate["projected_progress_completion_rate"]) >= baseline_completion
        ):
            eligible.append(str(candidate["configuration_id"]))
    selected = eligible[0] if eligible else None
    return {
        "rule": "select the highest training-ranked candidate only when held-out recovery success is strictly better than the default baseline and held-out progress continuity and projected completion are not worse",
        "ranking_source": "train",
        "acceptance_source": "held_out",
        "baseline_profile": BASELINE_DEFAULT,
        "eligible_candidates": eligible,
        "selected_candidate": selected,
        "accepted": selected is not None,
        "rejected_if_none_pass": True,
        "baseline_held_out_metrics": {
            "recovery_success_rate": baseline_success,
            "progress_continuity_rate": baseline_continuity,
            "projected_progress_completion_rate": baseline_completion,
        },
    }


def _evaluation_status(held_out_aggregate: Mapping[str, Any]) -> dict[str, Any]:
    activated = int(held_out_aggregate["recovery_activated_count"])
    failures = int(held_out_aggregate["recovery_failure_count"])
    reasons = []
    if activated < MINIMUM_INFORMATIVE_ACTIVATED_CASES:
        reasons.append(
            f"only {activated} activated held-out cases; minimum is {MINIMUM_INFORMATIVE_ACTIVATED_CASES}"
        )
    if failures < MINIMUM_INFORMATIVE_BASELINE_FAILURES:
        reasons.append(
            f"baseline has {failures} conditional recovery failures; minimum is "
            f"{MINIMUM_INFORMATIVE_BASELINE_FAILURES}"
        )
    return {
        "evaluation_status": "informative" if not reasons else "uninformative_ceiling",
        "ceiling_reason": reasons or None,
        "minimum_activated_cases": MINIMUM_INFORMATIVE_ACTIVATED_CASES,
        "minimum_baseline_failures": MINIMUM_INFORMATIVE_BASELINE_FAILURES,
        "baseline_activated_case_count": activated,
        "baseline_failure_count": failures,
        "projected_completion_rate": float(held_out_aggregate["projected_progress_completion_rate"]),
        "strict_simulator_finish_rate": float(held_out_aggregate["finished_rate"]),
        "completion_gap_is_present": bool(
            float(held_out_aggregate["projected_progress_completion_rate"])
            > float(held_out_aggregate["finished_rate"])
        ),
    }


def _profile_definition(name: str, role: str, description: str, config: Any) -> dict[str, Any]:
    return {"name": name, "role": role, "description": description, "config": _json_value(asdict(config))}


def build_report(
    telemetry_dir: Path = DEFAULT_TELEMETRY_DIR,
    maps_dir: Path = DEFAULT_MAPS_DIR,
    *,
    calibration_report: Mapping[str, Any],
    max_steps: int = 10000,
    search_radius: int = 64,
    anchor_progress_m: Mapping[str, Iterable[float]] | None = None,
    disturbances: Iterable[Any] = DISTURBANCES,
    held_out_disturbances: Iterable[Any] = HELD_OUT_DISTURBANCES,
    recovery_heading_gains: Sequence[float] = DEFAULT_RECOVERY_GRID["recovery_heading_gain"],
    recovery_lateral_gains: Sequence[float] = DEFAULT_RECOVERY_GRID["recovery_lateral_gain"],
    recovery_brake_biases: Sequence[float] = DEFAULT_RECOVERY_GRID["recovery_brake_bias"],
) -> dict[str, Any]:
    """Build a deterministic bounded train/held-out recovery sweep report."""

    if max_steps <= 0 or search_radius < 0:
        raise ValueError("max_steps must be positive and search_radius must be non-negative")
    grid = _validate_grid(recovery_heading_gains, recovery_lateral_gains, recovery_brake_biases)
    disturbance_list = tuple(disturbances)
    if not disturbance_list:
        raise ValueError("disturbances must contain at least one disturbance")
    held_out_disturbance_list = tuple(held_out_disturbances)
    common_names = {str(disturbance.name) for disturbance in disturbance_list}
    held_out_names = {str(disturbance.name) for disturbance in held_out_disturbance_list}
    if common_names & held_out_names:
        overlap = sorted(common_names & held_out_names)
        raise ValueError(f"held_out_disturbances overlap common disturbances: {overlap}")

    defaults = ConservativeControllerConfig()
    guardrail = _selected_guardrail(calibration_report)
    recovery_base = RecoveryAwareCenterlineControllerConfig()
    if any(getattr(recovery_base, field) != getattr(defaults, field) for field in defaults.__dataclass_fields__):
        raise AssertionError("recovery controller nominal fields must match default controller fields")

    parameter_dicts = [
        {
            "recovery_heading_gain": heading,
            "recovery_lateral_gain": lateral,
            "recovery_brake_bias": brake,
        }
        for heading, lateral, brake in itertools.product(*grid)
    ]
    candidate_specs = []
    for parameters in parameter_dicts:
        config = replace(recovery_base, **parameters)
        candidate_specs.append(
            {
                "configuration_id": _configuration_id(parameters),
                "parameters": parameters,
                "config": config,
                "controller": RecoveryAwareCenterlineController(config),
            }
        )

    profiles: dict[str, dict[str, Any]] = {
        BASELINE_DEFAULT: _profile_definition(
            BASELINE_DEFAULT, "reference", "ConservativeCenterlineController with current defaults.", defaults
        ),
        BASELINE_GUARDRAIL: _profile_definition(
            BASELINE_GUARDRAIL, "derived_candidate", guardrail.description, guardrail.config
        ),
    }
    profiles[BASELINE_GUARDRAIL]["evidence_basis"] = _json_value(guardrail.evidence_basis)
    profiles[BASELINE_GUARDRAIL]["derived_from_calibration"] = True
    for candidate in candidate_specs:
        profiles[candidate["configuration_id"]] = _profile_definition(
            candidate["configuration_id"], "recovery_parameter_candidate", "Recovery-only parameter candidate.", candidate["config"]
        )
        profiles[candidate["configuration_id"]]["recovery_parameters"] = _json_value(candidate["parameters"])
        profiles[candidate["configuration_id"]]["nominal_fields_match_default"] = True

    simulator_config = ClosedLoopConfig(max_steps=max_steps, search_radius=search_radius)
    controllers: dict[str, Any] = {
        BASELINE_DEFAULT: ConservativeCenterlineController(defaults),
        BASELINE_GUARDRAIL: ConservativeCenterlineController(guardrail.config),
    }
    controllers.update({candidate["configuration_id"]: candidate["controller"] for candidate in candidate_specs})
    map_reports: dict[str, Any] = {}

    for map_name, (telemetry_filename, centerline_filename) in MAP_FILES.items():
        telemetry_path = telemetry_dir / telemetry_filename
        centerline_path = maps_dir / centerline_filename
        samples = load_rollout_csv(telemetry_path)
        geometry = build_centerline_geometry(np.load(centerline_path, allow_pickle=False))
        targets = None if anchor_progress_m is None else anchor_progress_m.get(map_name)
        anchors = select_source_anchors(map_name, samples, geometry, target_progress_m=targets)
        base_cases: list[dict[str, Any]] = []
        for anchor in anchors:
            source_sample: RolloutSample = samples[int(anchor["source_index"])]
            for disturbance, disturbance_scope in itertools.chain(
                ((item, "common") for item in disturbance_list),
                ((item, "held_out_only") for item in held_out_disturbance_list),
            ):
                pose = disturb_sample(
                    source_sample,
                    geometry,
                    disturbance,
                    source_index=int(anchor["source_index"]),
                    section_bins=simulator_config.section_bins,
                )
                if pose.speed_mps > simulator_config.max_speed_mps:
                    raise ValueError(
                        f"held-out disturbance {disturbance.name!r} at anchor {anchor['anchor_id']!r} "
                        f"produces an initial speed above ClosedLoopConfig.max_speed_mps"
                    )
                initial_state = make_track_relative_state(
                    pose.position, pose.heading, pose.speed_mps, geometry
                )
                case_key = f"{map_name}:{anchor['anchor_id']}:{disturbance.name}"
                base_cases.append(
                    {
                        "case_key": case_key,
                        "map": map_name,
                        "disturbance": {
                            "name": disturbance.name,
                            "scope": disturbance_scope,
                            "lateral_offset_m": float(disturbance.lateral_offset_m),
                            "heading_offset_rad": float(disturbance.heading_offset_rad),
                            "speed_delta_mps": float(disturbance.speed_delta_mps),
                        },
                        "source_anchor": dict(anchor),
                        "disturbed_pose": {
                            "progress_m": float(pose.progress_m),
                            "section_bin": int(pose.section_bin),
                            "position": np.asarray(pose.position).tolist(),
                            "heading": np.asarray(pose.heading).tolist(),
                            "speed_mps": float(pose.speed_mps),
                            "realized_initial_lateral_error_m": float(initial_state.lateral_error_m),
                            "realized_initial_heading_error_rad": float(initial_state.heading_error_rad),
                            "realized_initial_speed_mps": float(pose.speed_mps),
                        },
                        "disturbance_scope": disturbance_scope,
                        "pose": pose,
                    }
                )
        map_reports[map_name] = {
            "telemetry_file": str(telemetry_path),
            "centerline_file": str(centerline_path),
            "track_length_m": float(geometry.cumulative_lengths[-1]),
            "source_anchors": anchors,
            "geometry": geometry,
            "base_cases": base_cases,
        }

    anchor_assignments, split_metadata = _split_source_anchors(
        {map_name: map_report["source_anchors"] for map_name, map_report in map_reports.items()}
    )
    split_by_case: dict[str, str] = {}
    case_counts_by_map: dict[str, dict[str, int]] = {}
    for map_name, map_report in map_reports.items():
        assignments = anchor_assignments[map_name]
        counts = {"case_count": 0, "train_case_count": 0, "held_out_case_count": 0}
        for base_case in map_report["base_cases"]:
            anchor_id = str(base_case["source_anchor"]["anchor_id"])
            try:
                split = assignments[anchor_id]
            except KeyError as exc:
                raise ValueError(f"case {base_case['case_key']!r} has no source-anchor split assignment") from exc
            # Novel disturbance families are generated only on held-out source
            # anchors.  This keeps both the source region and the perturbation
            # family unseen during training-side ranking.
            if base_case["disturbance_scope"] == "held_out_only" and split != "held_out":
                continue
            split_by_case[base_case["case_key"]] = split
            counts["case_count"] += 1
            counts[f"{split}_case_count"] += 1
        case_counts_by_map[map_name] = counts
    split_metadata.update(
        {
            "case_count": len(split_by_case),
            "train_case_count": sum(value == "train" for value in split_by_case.values()),
            "held_out_case_count": sum(value == "held_out" for value in split_by_case.values()),
            "case_counts_by_map": case_counts_by_map,
            "assignments": split_by_case,
        }
    )
    cases_by_profile: dict[str, list[dict[str, Any]]] = {name: [] for name in controllers}
    for map_report in map_reports.values():
        geometry: CenterlineGeometry = map_report.pop("geometry")
        for base_case in map_report.pop("base_cases"):
            if base_case["case_key"] not in split_by_case:
                continue
            split = split_by_case[base_case["case_key"]]
            for profile_name, controller in controllers.items():
                outcome = replay_disturbed_pose(
                    geometry,
                    base_case["pose"],
                    controller=controller,
                    config=simulator_config,
                )
                cases_by_profile[profile_name].append(
                    {
                        "case_id": f"{base_case['case_key']}:{profile_name}",
                        "case_key": base_case["case_key"],
                        "split": split,
                        "map": base_case["map"],
                        "profile": profile_name,
                        "disturbance": base_case["disturbance"],
                        "source_anchor": base_case["source_anchor"],
                        "disturbed_pose": base_case["disturbed_pose"],
                        "metrics": _json_value(outcome.metrics),
                    }
                )

    aggregates: dict[str, dict[str, Any]] = {}
    for profile_name, cases in cases_by_profile.items():
        train_cases = [case for case in cases if case["split"] == "train"]
        held_out_cases = [case for case in cases if case["split"] == "held_out"]
        aggregates[profile_name] = {
            "train": _profile_aggregate(train_cases),
            "held_out": _profile_aggregate(held_out_cases),
        }

    candidate_rows = []
    for candidate in candidate_specs:
        name = candidate["configuration_id"]
        candidate_rows.append(
            {
                "configuration_id": name,
                "recovery_parameters": _json_value(candidate["parameters"]),
                "config": _json_value(asdict(candidate["config"])),
                "nominal_fields_match_default": True,
                "train_aggregate": aggregates[name]["train"],
                "held_out_aggregate": aggregates[name]["held_out"],
            }
        )
    ranked_candidates = _rank_candidates(candidate_rows)
    baseline_rows = {
        name: {
            "profile": profiles[name],
            "train_aggregate": aggregates[name]["train"],
            "held_out_aggregate": aggregates[name]["held_out"],
        }
        for name in (BASELINE_DEFAULT, BASELINE_GUARDRAIL)
    }
    selection = _selection(ranked_candidates, baseline_rows[BASELINE_DEFAULT])

    report = {
        "scope": "offline_a01_a07_source_region_and_disturbance_family_held_out_hybrid_recovery_parameter_sweep",
        "replay_interpretation": "pose replay plus deterministic simulator rollout; recorded transitions are not replayed",
        "offline_only": True,
        "safety_flags": {
            "trackmania_launched": False,
            "training_started": False,
            "live_control_used": False,
            "azure_started": False,
            "checkpoints_loaded": False,
            "external_services_used": False,
            "controller_defaults_modified": False,
        },
        "trackmania_launched": False,
        "training_started": False,
        "live_control_used": False,
        "azure_started": False,
        "checkpoints_loaded": False,
        "external_services_used": False,
        "controller_defaults_modified": False,
        "real_physics_validated": False,
        "calibration_input": None,
        "telemetry_files": {name: report["telemetry_file"] for name, report in map_reports.items()},
        "centerline_files": {name: report["centerline_file"] for name, report in map_reports.items()},
        "simulator": {
            "dt_s": simulator_config.dt_s,
            "max_steps": max_steps,
            "search_radius_segments": search_radius,
            "map_match_horizontal_only": simulator_config.map_match_horizontal_only,
            "section_bins": simulator_config.section_bins,
            "backward_progress_tolerance_m": 1e-5,
            "collision_model": False,
            "track_width_model": False,
            "off_track_termination": False,
            "vertical_dynamics": False,
            "jump_model": False,
            "surface_model": False,
            "slip_model": False,
            "recorded_transition_replay": False,
            "disturbance_injection_mode": "initial_pose_only",
        },
        "split": split_metadata,
        "disturbances": [
            {
                "name": disturbance.name,
                "lateral_offset_m": float(disturbance.lateral_offset_m),
                "heading_offset_rad": float(disturbance.heading_offset_rad),
                "speed_delta_mps": float(disturbance.speed_delta_mps),
            }
            for disturbance in disturbance_list
        ],
        "held_out_only_disturbances": [
            {
                "name": disturbance.name,
                "scope": "held_out_only",
                "lateral_offset_m": float(disturbance.lateral_offset_m),
                "heading_offset_rad": float(disturbance.heading_offset_rad),
                "speed_delta_mps": float(disturbance.speed_delta_mps),
            }
            for disturbance in held_out_disturbance_list
        ],
        "disturbance_novelty": {
            "split_unit": "source_anchor_plus_held_out_disturbance_family",
            "train_disturbance_families": sorted(common_names),
            "held_out_disturbance_families": sorted(
                common_names | held_out_names
            ),
            "held_out_only_disturbance_families": sorted(held_out_names),
            "unseen_family": bool(held_out_names),
            "source_overlap_check": "passed",
        },
        "parameter_grid": {
            "recovery_heading_gain": list(grid[0]),
            "recovery_lateral_gain": list(grid[1]),
            "recovery_brake_bias": list(grid[2]),
            "configuration_count": len(candidate_specs),
            "bounded_max_configurations": MAX_GRID_CONFIGURATIONS,
            "recovery_only": True,
        },
        "baseline_profiles": baseline_rows,
        "profiles": profiles,
        "train_aggregate_metrics": {name: value["train"] for name, value in aggregates.items()},
        "held_out_aggregate_metrics": {name: value["held_out"] for name, value in aggregates.items()},
        "candidate_configs": [
            {
                "configuration_id": candidate["configuration_id"],
                "recovery_parameters": _json_value(candidate["parameters"]),
                "config": _json_value(asdict(candidate["config"])),
                "nominal_fields_match_default": True,
            }
            for candidate in candidate_specs
        ],
        "candidates": ranked_candidates,
        "aggregates": aggregates,
        "maps": {
            name: {
                "telemetry_file": report["telemetry_file"],
                "centerline_file": report["centerline_file"],
                "track_length_m": report["track_length_m"],
                "source_anchors": report["source_anchors"],
                "cases_by_profile": {
                    profile_name: [case for case in cases if case["map"] == name]
                    for profile_name, cases in cases_by_profile.items()
                },
            }
            for name, report in map_reports.items()
        },
        "selection": selection,
        "evaluation": _evaluation_status(aggregates[BASELINE_DEFAULT]["held_out"]),
        "limitations": [
            "Training cases are used only for deterministic candidate ranking; acceptance is evaluated on held-out cases.",
            "The simulator is a bounded kinematic model and does not represent TrackMania physics, collisions, jumps, or TMInterface.",
            "The captured p25 guardrail is derived through derive_candidates from calibration evidence and is comparison-only.",
            "Metrics establish offline software-in-the-loop behavior only, not live-control or training performance.",
        ],
    }
    return _json_value(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-dir", type=Path, default=DEFAULT_TELEMETRY_DIR)
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--search-radius", type=int, default=64)
    args = parser.parse_args()
    report = build_report(
        args.telemetry_dir,
        args.maps_dir,
        calibration_report=load_calibration(args.calibration),
        max_steps=args.max_steps,
        search_radius=args.search_radius,
        disturbances=DISTURBANCES,
        held_out_disturbances=HELD_OUT_DISTURBANCES,
    )
    report["calibration_input"] = str(args.calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "candidates": len(report["candidates"]), "selection": report["selection"]}, indent=2))


if __name__ == "__main__":
    main()
