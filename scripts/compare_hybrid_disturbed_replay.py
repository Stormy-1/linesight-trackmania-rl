"""Compare three offline hybrid-controller profiles from disturbed telemetry poses.

This report is deliberately a *pose replay plus simulator rollout*: recorded
telemetry supplies deterministic source poses, saved centerlines supply map
geometry, and the pure closed-loop simulator supplies the subsequent trajectory.
It does not replay recorded transitions and it never starts TrackMania, training,
live control, checkpoint loading, or external services.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from scripts.evaluate_hybrid_speed_candidates import derive_candidates, load_calibration
except ModuleNotFoundError:  # Running this file directly puts ``scripts`` first on sys.path.
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from scripts.evaluate_hybrid_speed_candidates import derive_candidates, load_calibration

from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineController,
    ConservativeControllerConfig,
    RecoveryAwareCenterlineController,
    RecoveryAwareCenterlineControllerConfig,
)
from trackmania_rl.hybrid_replay import (
    DisturbanceSpec,
    disturb_sample,
    replay_disturbed_pose,
    select_source_samples,
)
from trackmania_rl.hybrid_rollout import RolloutSample, load_rollout_csv
from trackmania_rl.hybrid_simulator import ClosedLoopConfig
from trackmania_rl.hybrid_state import CenterlineGeometry, build_centerline_geometry, make_track_relative_state


DEFAULT_TELEMETRY_DIR = Path("eval_results/hybrid_rollouts_crossmap")
DEFAULT_MAPS_DIR = Path("maps")
DEFAULT_CALIBRATION = Path("eval_results/hybrid_rollouts/speed_profile_calibration.json")
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/disturbed_pose_replay_comparison.json")

MAP_FILES = {
    "A01-Race": ("A01-Race.repaired.csv", "A01-Race_0.5m_cl2.npy"),
    "A07-Race": ("A07-Race.repaired.csv", "A07-Race_0.5m_cl2.npy"),
}

# These include the documented divergence regions.  The helper accepts an
# override so fixture reports and future calibration runs can choose another
# small, deterministic anchor set without changing the report algorithm.
DEFAULT_ANCHOR_PROGRESS_M: dict[str, tuple[float, ...]] = {
    "A01-Race": (20.0, 80.0, 140.0),
    "A07-Race": (20.0, 250.0, 379.0),
}
DOCUMENTED_DIVERGENCE_PROGRESS_M = {"A01-Race": 80.0, "A07-Race": 379.0}

DISTURBANCES: tuple[DisturbanceSpec, ...] = (
    DisturbanceSpec("bilateral_lateral_minus_10m", lateral_offset_m=-10.0),
    DisturbanceSpec("bilateral_lateral_plus_10m", lateral_offset_m=10.0),
    DisturbanceSpec(
        "combined_lateral_plus_10m_heading_plus_35deg",
        lateral_offset_m=10.0,
        heading_offset_rad=float(np.deg2rad(35.0)),
    ),
)

# These disturbance families are deliberately absent from the training set in
# the recovery sweep.  They exercise lateral/heading recovery while adding a
# speed error, so the result tests both source-region and disturbance-family
# generalization without changing the historical common disturbance set.
HELD_OUT_DISTURBANCES: tuple[DisturbanceSpec, ...] = (
    DisturbanceSpec(
        "heldout_lateral_plus_20m_heading_plus_90deg_speed_plus_12mps",
        lateral_offset_m=20.0,
        heading_offset_rad=float(np.deg2rad(90.0)),
        speed_delta_mps=12.0,
    ),
    DisturbanceSpec(
        "heldout_lateral_minus_20m_heading_minus_90deg_speed_plus_12mps",
        lateral_offset_m=-20.0,
        heading_offset_rad=float(np.deg2rad(-90.0)),
        speed_delta_mps=12.0,
    ),
)

PROFILE_NAMES = (
    "live_defaults_reference",
    "captured_p25_guardrail",
    "recovery_aware_centerline",
)


def _json_value(value: Any) -> Any:
    """Convert numpy scalars/arrays and nested dataclass-like values to JSON values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _projected_progress(sample: RolloutSample, geometry: CenterlineGeometry) -> float:
    state = make_track_relative_state(sample.position, sample.heading, sample.speed_mps, geometry)
    return float(state.progress_m)


def select_source_anchors(
    map_name: str,
    samples: Iterable[RolloutSample],
    geometry: CenterlineGeometry,
    *,
    target_progress_m: Iterable[float] | None = None,
    count_per_target: int = 1,
    section_bins: int = 10,
) -> list[dict[str, Any]]:
    """Select stable nearest-progress source anchors and return JSON metadata.

    One anchor is selected for each requested progress target.  Duplicate source
    rows caused by targets beyond a short fixture/map are collapsed by source
    index, while preserving target order and deterministic nearest-progress tie
    resolution from :func:`select_source_samples`.
    """

    if map_name not in MAP_FILES and target_progress_m is None:
        raise ValueError(f"no default anchor targets are defined for {map_name!r}")
    targets = tuple(
        float(value)
        for value in (DEFAULT_ANCHOR_PROGRESS_M[map_name] if target_progress_m is None else target_progress_m)
    )
    if not targets:
        raise ValueError("target_progress_m must contain at least one target")
    if not isinstance(count_per_target, int) or isinstance(count_per_target, bool) or count_per_target <= 0:
        raise ValueError("count_per_target must be a positive integer")
    if not isinstance(section_bins, int) or isinstance(section_bins, bool) or section_bins <= 0:
        raise ValueError("section_bins must be a positive integer")

    sample_list = list(samples)
    selected: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    track_length = float(geometry.cumulative_lengths[-1])
    for target in targets:
        for source_index, sample in select_source_samples(
            sample_list, geometry, target, count=count_per_target
        ):
            if source_index in seen_indices:
                continue
            seen_indices.add(source_index)
            source_progress = _projected_progress(sample, geometry)
            section_fraction = min(max(source_progress / max(track_length, 1e-9), 0.0), 1.0)
            section_bin = min(int(section_fraction * section_bins), section_bins - 1)
            selected.append(
                {
                    "anchor_id": f"{map_name}:anchor_{len(selected):02d}",
                    "target_progress_m": target,
                    "anchor_role": (
                        "documented_divergence_region"
                        if target == DOCUMENTED_DIVERGENCE_PROGRESS_M.get(map_name)
                        else "fixed_progress_anchor"
                    ),
                    "source_index": source_index,
                    "source_progress_m": source_progress,
                    "source_distance_to_target_m": abs(source_progress - target),
                    "source_section_bin": section_bin,
                    "timestamp_s": float(sample.timestamp_s),
                    "position": np.asarray(sample.position, dtype=np.float64).tolist(),
                    "heading": np.asarray(sample.heading, dtype=np.float64).tolist(),
                    "speed_mps": float(sample.speed_mps),
                }
            )
    if not selected:
        raise ValueError("samples must contain at least one sample")
    return selected


def _find_sample(samples: list[RolloutSample], source_index: int) -> RolloutSample:
    if source_index < 0 or source_index >= len(samples):
        raise IndexError(f"source_index {source_index} is outside telemetry samples")
    return samples[source_index]


def _selected_guardrail(calibration_report: Mapping[str, Any]):
    for candidate in derive_candidates(calibration_report):
        if candidate.name == "captured_p25_guardrail":
            return candidate
    raise ValueError("captured_p25_guardrail was not derived from calibration")


def _profile_specs(calibration_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    guardrail = _selected_guardrail(calibration_report)
    reference_config = ConservativeControllerConfig()
    recovery_config = RecoveryAwareCenterlineControllerConfig()
    return [
        {
            "name": "live_defaults_reference",
            "role": "reference",
            "description": "ConservativeCenterlineController with current defaults.",
            "config": reference_config,
            "controller": ConservativeCenterlineController(),
        },
        {
            "name": "captured_p25_guardrail",
            "role": "derived_candidate",
            "description": guardrail.description,
            "config": guardrail.config,
            "controller": ConservativeCenterlineController(guardrail.config),
            "evidence_basis": guardrail.evidence_basis,
            "derived_from_calibration": True,
        },
        {
            "name": "recovery_aware_centerline",
            "role": "recovery_variant",
            "description": "RecoveryAwareCenterlineController with its isolated recovery path.",
            "config": recovery_config,
            "controller": RecoveryAwareCenterlineController(),
        },
    ]


def _case_aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("cannot aggregate an empty case list")

    def values(key: str) -> list[float]:
        return [float(case["metrics"][key]) for case in cases]

    successes = [case["metrics"]["recovery_success"] for case in cases if case["metrics"]["recovery_success"] is not None]
    backward_cases = [case for case in cases if int(case["metrics"]["backward_progress_event_count"]) > 0]
    finished = [bool(case["metrics"]["finished"]) for case in cases]
    progress_complete = [bool(case["metrics"]["progress_complete"]) for case in cases]
    elevation_tolerant_finished = [bool(case["metrics"]["elevation_tolerant_finished"]) for case in cases]
    return {
        "case_count": len(cases),
        "finished_count": sum(finished),
        "completion_rate": float(sum(finished) / len(cases)),
        "progress_complete_count": sum(progress_complete),
        "progress_complete_rate": float(sum(progress_complete) / len(cases)),
        "elevation_tolerant_finished_count": sum(elevation_tolerant_finished),
        "elevation_tolerant_completion_rate": float(sum(elevation_tolerant_finished) / len(cases)),
        "recovery_activated_count": sum(bool(case["metrics"]["recovery_activated"]) for case in cases),
        "recovery_completed_count": sum(bool(case["metrics"]["recovery_completed"]) for case in cases),
        "recovery_success_rate": float(sum(bool(value) for value in successes) / len(successes)) if successes else None,
        "mean_abs_lateral_error_m": float(np.mean(values("mean_abs_lateral_error_m"))),
        "mean_horizontal_error_m": float(np.mean(values("mean_horizontal_error_m"))),
        "mean_abs_heading_error_rad": float(np.mean(values("mean_abs_heading_error_rad"))),
        "mean_final_progress_fraction": float(np.mean(values("progress_fraction"))),
        "progress_continuity_rate": float(np.mean([bool(case["metrics"]["progress_continuity"]) for case in cases])),
        "backward_progress_event_rate": float(len(backward_cases) / len(cases)),
        "mean_backward_progress_event_count": float(np.mean(values("backward_progress_event_count"))),
        "mean_backward_progress_distance_m": float(np.mean(values("backward_progress_distance_m"))),
        "total_backward_progress_distance_m": float(sum(values("backward_progress_distance_m"))),
    }


def _section_aggregates(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for case in cases:
        source_bin = int(case["source_anchor"]["source_section_bin"])
        grouped.setdefault(source_bin, []).append(case)
    result = []
    for source_bin in sorted(grouped):
        aggregate = _case_aggregate(grouped[source_bin])
        result.append({"source_section_bin": source_bin, **aggregate})
    return result


def _profile_aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = _case_aggregate(cases)
    aggregate["section_aggregates"] = _section_aggregates(cases)
    return aggregate


def _winner(profile_aggregates: Mapping[str, Mapping[str, Any]], metric: str, *, lower_is_better: bool) -> dict[str, Any]:
    ordered = list(PROFILE_NAMES)
    winner = min(ordered, key=lambda name: (float(profile_aggregates[name][metric]) * (1 if lower_is_better else -1), ordered.index(name)))
    return {"profile": winner, "value": float(profile_aggregates[winner][metric]), "lower_is_better": lower_is_better}


def _summary(map_reports: Mapping[str, Any]) -> dict[str, Any]:
    combined: dict[str, list[dict[str, Any]]] = {name: [] for name in PROFILE_NAMES}
    for map_report in map_reports.values():
        for name in PROFILE_NAMES:
            combined[name].extend(map_report["cases_by_profile"][name])
    aggregates = {name: _profile_aggregate(cases) for name, cases in combined.items()}
    winners = {
        "completion_rate": _winner(aggregates, "completion_rate", lower_is_better=False),
        "progress_complete_rate": _winner(aggregates, "progress_complete_rate", lower_is_better=False),
        "mean_final_progress_fraction": _winner(aggregates, "mean_final_progress_fraction", lower_is_better=False),
        "mean_abs_lateral_error_m": _winner(aggregates, "mean_abs_lateral_error_m", lower_is_better=True),
        "mean_horizontal_error_m": _winner(aggregates, "mean_horizontal_error_m", lower_is_better=True),
        "mean_abs_heading_error_rad": _winner(aggregates, "mean_abs_heading_error_rad", lower_is_better=True),
        "backward_progress_event_rate": _winner(aggregates, "backward_progress_event_rate", lower_is_better=True),
    }
    return {"profile_aggregates": aggregates, "metric_winners": winners}


def build_report(
    telemetry_dir: Path = DEFAULT_TELEMETRY_DIR,
    maps_dir: Path = DEFAULT_MAPS_DIR,
    *,
    calibration_report: Mapping[str, Any],
    max_steps: int = 10000,
    search_radius: int = 64,
    anchor_progress_m: Mapping[str, Iterable[float]] | None = None,
    disturbances: Iterable[DisturbanceSpec] = DISTURBANCES,
) -> dict[str, Any]:
    """Build the deterministic A01/A07 disturbed-pose comparison report."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if search_radius < 0:
        raise ValueError("search_radius must be non-negative")
    disturbance_list = tuple(disturbances)
    if not disturbance_list:
        raise ValueError("disturbances must contain at least one disturbance")
    specs = _profile_specs(calibration_report)
    simulator_config = ClosedLoopConfig(max_steps=max_steps, search_radius=search_radius)
    map_reports: dict[str, Any] = {}
    telemetry_files: dict[str, str] = {}
    centerline_files: dict[str, str] = {}

    for map_name, (telemetry_filename, centerline_filename) in MAP_FILES.items():
        telemetry_path = telemetry_dir / telemetry_filename
        centerline_path = maps_dir / centerline_filename
        samples = load_rollout_csv(telemetry_path)
        geometry = build_centerline_geometry(np.load(centerline_path, allow_pickle=False))
        telemetry_files[map_name] = str(telemetry_path)
        centerline_files[map_name] = str(centerline_path)
        targets = None if anchor_progress_m is None else anchor_progress_m.get(map_name)
        anchors = select_source_anchors(map_name, samples, geometry, target_progress_m=targets)
        cases_by_profile: dict[str, list[dict[str, Any]]] = {name: [] for name in PROFILE_NAMES}
        cases: list[dict[str, Any]] = []
        for anchor in anchors:
            source_sample = _find_sample(samples, int(anchor["source_index"]))
            for disturbance in disturbance_list:
                pose = disturb_sample(
                    source_sample,
                    geometry,
                    disturbance,
                    source_index=int(anchor["source_index"]),
                    section_bins=simulator_config.section_bins,
                )
                for spec in specs:
                    outcome = replay_disturbed_pose(
                        geometry,
                        pose,
                        controller=spec["controller"],
                        config=simulator_config,
                    )
                    case = {
                        "case_id": f"{map_name}:{anchor['anchor_id']}:{disturbance.name}:{spec['name']}",
                        "map": map_name,
                        "profile": spec["name"],
                        "disturbance": {
                            "name": disturbance.name,
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
                        },
                        "metrics": _json_value(outcome.metrics),
                    }
                    cases.append(case)
                    cases_by_profile[spec["name"]].append(case)
        profile_aggregates = {
            name: _profile_aggregate(cases_by_profile[name]) for name in PROFILE_NAMES
        }
        map_reports[map_name] = {
            "telemetry_file": str(telemetry_path),
            "centerline_file": str(centerline_path),
            "track_length_m": float(geometry.cumulative_lengths[-1]),
            "source_anchors": anchors,
            "cases": cases,
            "cases_by_profile": cases_by_profile,
            "profile_aggregates": profile_aggregates,
        }

    profiles = {}
    for spec in specs:
        config = spec["config"]
        profile = {
            "name": spec["name"],
            "role": spec["role"],
            "description": spec["description"],
            "config": _json_value(asdict(config)),
        }
        if "evidence_basis" in spec:
            profile["evidence_basis"] = _json_value(spec["evidence_basis"])
            profile["derived_from_calibration"] = True
        profiles[spec["name"]] = profile

    report = {
        "scope": "offline_a01_a07_disturbed_pose_replay_comparison",
        "replay_interpretation": "pose replay plus simulator rollout; recorded transitions are not replayed",
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
        "azure_started": False,
        "checkpoints_loaded": False,
        "real_physics_validated": False,
        "telemetry_files": telemetry_files,
        "centerline_files": centerline_files,
        "calibration_input": None,
        "disturbances": [
            {
                "name": disturbance.name,
                "lateral_offset_m": float(disturbance.lateral_offset_m),
                "heading_offset_rad": float(disturbance.heading_offset_rad),
                "speed_delta_mps": float(disturbance.speed_delta_mps),
            }
            for disturbance in disturbance_list
        ],
        "simulator": {
            "dt_s": simulator_config.dt_s,
            "max_steps": max_steps,
            "search_radius_segments": search_radius,
            "map_match_horizontal_only": simulator_config.map_match_horizontal_only,
            "section_bins": simulator_config.section_bins,
            "backward_progress_tolerance_m": 1e-5,
        },
        "profiles": profiles,
        "maps": map_reports,
        "summary": _summary(map_reports),
        "limitations": [
            "This is pose replay plus deterministic simulator rollout, not replay of recorded transitions.",
            "The simulator is a bounded kinematic model and does not represent TrackMania physics, collisions, jumps, or TMInterface.",
            "Saved centerlines and recorded telemetry are used offline; these metrics do not establish live-control or training performance.",
            "The captured p25 guardrail is derived from calibration percentiles for comparison only; it does not change controller defaults.",
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
    )
    report["calibration_input"] = str(args.calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "maps": list(report["maps"]), "profiles": list(report["profiles"])}, indent=2))


if __name__ == "__main__":
    main()
