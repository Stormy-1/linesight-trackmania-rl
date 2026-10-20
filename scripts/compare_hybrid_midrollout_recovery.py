"""Paired, offline A01/A07 mid-rollout recovery development diagnostics.

The profiles, calibration-derived guardrail, anchor split and decision rules are
the historical recovery sweep's.  These already-inspected anchors/disturbances
are development validation, not independent confirmation or promotion evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from scripts import sweep_hybrid_recovery_parameters as historical
except (ImportError, ModuleNotFoundError):  # Direct script execution puts scripts first.
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from scripts import sweep_hybrid_recovery_parameters as historical

from scripts.compare_hybrid_disturbed_replay import (
    DISTURBANCES,
    HELD_OUT_DISTURBANCES,
    MAP_FILES,
    select_source_anchors,
)
from scripts.evaluate_hybrid_speed_candidates import load_calibration
from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineController,
    ConservativeControllerConfig,
    RecoveryAwareCenterlineController,
    RecoveryAwareCenterlineControllerConfig,
)
from trackmania_rl.hybrid_replay import (
    DisturbanceSpec,
    InjectionNotReached,
    MidRolloutInjection,
    replay_midrollout_disturbance,
    summarize_rollout_suffix,
)
from trackmania_rl.hybrid_rollout import load_rollout_csv
from trackmania_rl.hybrid_simulator import ClosedLoopConfig, simulate_closed_loop
from trackmania_rl.hybrid_state import build_centerline_geometry


BASELINE_DEFAULT = historical.BASELINE_DEFAULT
BASELINE_GUARDRAIL = historical.BASELINE_GUARDRAIL
DEFAULT_RECOVERY_GRID = historical.DEFAULT_RECOVERY_GRID
DEFAULT_TELEMETRY_DIR = historical.DEFAULT_TELEMETRY_DIR
DEFAULT_MAPS_DIR = historical.DEFAULT_MAPS_DIR
DEFAULT_CALIBRATION = historical.DEFAULT_CALIBRATION
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_midrollout_recovery_comparison.json")
DEFAULT_INJECTION_STEP = 10
SNAPSHOT_FIELDS = (
    "progress_m", "speed_mps", "lateral_error_m", "heading_error_rad", "segment_index",
)
PAIRED_METRICS = (
    "mean_abs_lateral_error_m", "mean_horizontal_error_m", "mean_abs_heading_error_rad",
    "max_abs_lateral_error_m", "progress_gain_m", "progress_fraction",
    "backward_progress_event_count", "backward_progress_distance_m", "elapsed_s",
    "recovery_latency_samples", "recovery_latency_s", "redivergence_count",
    "progress_continuity", "progress_complete", "finished", "stable_rejoin",
    "recovery_activated", "recovery_completed", "recovery_success",
)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _nonfinite_paths(value: Any, path: str = "") -> list[str]:
    value = _plain(value)
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, Mapping):
        return [found for key, item in value.items() for found in _nonfinite_paths(item, f"{path}.{key}")]
    if isinstance(value, list):
        return [found for index, item in enumerate(value) for found in _nonfinite_paths(item, f"{path}[{index}]")]
    return []


def _json_safe(value: Any) -> Any:
    """Invalid observations remain explicit, but NaN/Infinity never enter JSON."""
    value = _plain(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _validate_disturbances(common: Iterable[Any], held_out: Iterable[Any]):
    groups = (tuple(common), tuple(held_out))
    names: set[str] = set()
    signatures: dict[tuple[float, float, float], str] = {}
    for label, group in zip(("disturbances", "held_out_disturbances"), groups):
        if not group:
            raise ValueError(f"{label} must contain at least one disturbance")
        for item in group:
            if not isinstance(item, DisturbanceSpec):
                raise TypeError(f"{label} must contain DisturbanceSpec values")
            if item.name in names:
                raise ValueError(f"duplicate disturbance name: {item.name!r}")
            names.add(item.name)
            # Rotations by whole turns are the same numerical intervention too.
            signature = (
                float(item.lateral_offset_m),
                round((float(item.heading_offset_rad) + math.pi) % (2 * math.pi) - math.pi, 12),
                float(item.speed_delta_mps),
            )
            if not all(math.isfinite(value) for value in signature):
                raise ValueError("disturbance parameters must be finite")
            if signature in signatures:
                raise ValueError(
                    f"equivalent numeric disturbance signature: {item.name!r} and {signatures[signature]!r}"
                )
            signatures[signature] = item.name
    return groups


def _profile_specs(calibration_report, grid):
    defaults = ConservativeControllerConfig()
    recovery_base = RecoveryAwareCenterlineControllerConfig()
    if any(getattr(defaults, field) != getattr(recovery_base, field) for field in defaults.__dataclass_fields__):
        raise AssertionError("recovery controller nominal fields must match default controller fields")
    guardrail = historical._selected_guardrail(calibration_report)
    specs = {
        BASELINE_DEFAULT: (ConservativeCenterlineController, defaults),
        BASELINE_GUARDRAIL: (ConservativeCenterlineController, guardrail.config),
    }
    definitions = {
        BASELINE_DEFAULT: historical._profile_definition(
            BASELINE_DEFAULT, "reference", "Unmodified conservative defaults.", defaults
        ),
        BASELINE_GUARDRAIL: {
            **historical._profile_definition(
                BASELINE_GUARDRAIL, "derived_candidate", guardrail.description, guardrail.config
            ),
            "derived_from_calibration": True,
            "evidence_basis": _plain(guardrail.evidence_basis),
        },
    }
    for heading, lateral, brake in itertools.product(*grid):
        parameters = dict(recovery_heading_gain=heading, recovery_lateral_gain=lateral, recovery_brake_bias=brake)
        name = historical._configuration_id(parameters)
        if name in specs:
            raise ValueError("grid values produce duplicate configuration identifiers")
        config = replace(recovery_base, **parameters)
        specs[name] = (RecoveryAwareCenterlineController, config)
        definitions[name] = {
            **historical._profile_definition(name, "recovery_parameter_candidate", "Frozen recovery sweep configuration.", config),
            "recovery_parameters": parameters,
            "nominal_fields_match_default": True,
        }
    return specs, definitions


def _snapshot(state):
    return {field: _plain(getattr(state, field)) for field in SNAPSHOT_FIELDS}


def _terminal(result):
    return {
        "steps": len(result.steps), "elapsed_s": result.elapsed_s,
        "termination_reason": result.termination_reason,
        "final_state": _snapshot(result.final_state),
    }


def _prefix_diagnostics(result, injection_step, pre_injection, config):
    flags = [bool(step.command.recovery) for step in result.steps if step.step_index < injection_step]
    clean = None
    if isinstance(pre_injection, Mapping) and all(
        isinstance(pre_injection.get(field), (int, float, np.number))
        and math.isfinite(float(pre_injection[field]))
        for field in ("lateral_error_m", "heading_error_rad")
    ):
        clean = bool(
            abs(pre_injection["lateral_error_m"]) <= config.recovery_lateral_limit_m
            and abs(pre_injection["heading_error_rad"]) <= config.recovery_heading_limit_rad
        )
    return {
        "recovery_flags": flags,
        "recovery_sample_count": sum(flags),
        "any_recovery": any(flags),
        "clean_at_injection": clean,
        "clean_definition": "pre-injection lateral/heading within this profile's recovery thresholds",
        "lateral_limit_m": config.recovery_lateral_limit_m,
        "heading_limit_rad": config.recovery_heading_limit_rad,
    }


def _compare_prefix(control, disturbed, injection_step, pre_injection=None):
    """Compare observed states AND commands; never call a controller to inspect it."""
    left = [step for step in control.steps if step.step_index < injection_step]
    right = [step for step in disturbed.steps if step.step_index < injection_step]
    mismatches = []
    for a, b in zip(left, right):
        fields = [
            field for field in ("step_index", "timestamp_s", "state", "command")
            if _plain(asdict(a)[field]) != _plain(asdict(b)[field])
        ]
        if fields:
            mismatches.append({"step_index": a.step_index, "fields": fields})
    complete = len(left) == len(right) == injection_step
    boundary_equal = None
    if isinstance(pre_injection, Mapping) and len(control.steps) > injection_step:
        expected = _snapshot(control.steps[injection_step].state)
        boundary_equal = all(pre_injection.get(field) == value for field, value in expected.items())
    return {
        "comparison": "exact observed state, command, step index and timestamp equality within profile",
        "control_prefix_steps": len(left), "disturbed_prefix_steps": len(right),
        "compared_steps": min(len(left), len(right)), "expected_prefix_steps": injection_step,
        "complete": complete, "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "matches": complete and not mismatches,
        "pre_injection_snapshot_matches_control": boundary_equal,
    }


def _error(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def _validate_suffix(metrics):
    """Catch incomplete/malformed simulator summaries at their case boundary."""
    if not isinstance(metrics, Mapping):
        raise TypeError("suffix metrics must be a mapping")
    for field in PAIRED_METRICS:
        value = metrics[field]
        if value is not None and not isinstance(value, (bool, int, float, np.number)):
            raise ValueError(f"suffix metric {field!r} must be numeric, boolean or null")
    # Exercise exactly the schema consumed by the reused historical aggregate.
    # Nonfinite values are recorded separately, before they can enter ranking.
    if not _nonfinite_paths(metrics):
        historical._aggregate_without_sections([{"metrics": metrics}])


def _control(geometry, initial, controller_type, controller_config, config, injection_step):
    result = None
    record: dict[str, Any] = {"status": "ok", "suffix_metrics": None, "prefix": None}
    try:
        result = simulate_closed_loop(
            geometry, **initial, controller=controller_type(controller_config), config=config
        )
        record["terminal"] = _terminal(result)
        reached = len(result.steps) > injection_step
        snapshot = _snapshot(result.steps[injection_step].state) if reached else None
        record["pre_injection"] = snapshot
        record["prefix"] = _prefix_diagnostics(result, injection_step, snapshot, controller_config)
        if not reached:
            record["status"] = "injection_not_reached"
        else:
            record["suffix_metrics"] = summarize_rollout_suffix(result, injection_step, dt_s=config.dt_s)
            _validate_suffix(record["suffix_metrics"])
            nonfinite = _nonfinite_paths(record)
            if nonfinite:
                record.update(status="invalid", nonfinite_paths=nonfinite)
    except Exception as exc:  # A failed control is report evidence, never a dropped anchor.
        record.update(status="invalid" if isinstance(exc, (TypeError, ValueError)) else "error", error=_error(exc))
    return result, record


def _run_case(geometry, initial, controller_type, controller_config, config, injection_step,
              disturbance, control_result, control_record):
    case: dict[str, Any] = {
        "status": "ok", "metrics": None, "prefix_comparison": None, "prefix": None,
        "injection": None, "pre_injection": None, "post_injection": None,
        "injection_recovery": None, "total_elapsed_s": None,
        "paired_control_suffix_metrics": None, "paired_deltas": None,
    }
    result = None
    try:
        outcome = replay_midrollout_disturbance(
            geometry, **initial, injection=MidRolloutInjection(injection_step, disturbance),
            controller=controller_type(controller_config), config=config,
        )
        result = outcome.result
        metrics = outcome.metrics
        case.update(
            metrics=metrics["post_rollout"], injection=metrics["injection"],
            pre_injection=metrics["pre_injection"], post_injection=metrics["post_injection"],
            injection_recovery=metrics["recovery"], total_elapsed_s=metrics["total_elapsed_s"],
        )
        _validate_suffix(case["metrics"])
        for label in ("pre_injection", "post_injection"):
            if any(field not in case[label] for field in SNAPSHOT_FIELDS):
                raise ValueError(f"{label} lacks required realized snapshot fields")
        nonfinite = _nonfinite_paths(case)
        if nonfinite:
            case.update(status="invalid", nonfinite_paths=nonfinite)
    except InjectionNotReached as exc:
        result = exc.result
        case.update(status="injection_not_reached", error=_error(exc), unreached_step_index=exc.step_index)
    except Exception as exc:
        case.update(status="invalid" if isinstance(exc, (TypeError, ValueError)) else "error", error=_error(exc))

    if result is not None:
        case["terminal"] = _terminal(result)
        case["prefix"] = _prefix_diagnostics(result, injection_step, case["pre_injection"], controller_config)
        if control_result is not None:
            comparison = _compare_prefix(control_result, result, injection_step, case["pre_injection"])
            case["prefix_comparison"] = comparison
            if case["status"] == "ok" and (
                not comparison["matches"] or comparison["pre_injection_snapshot_matches_control"] is not True
            ):
                case["status"] = "prefix_mismatch"
    if control_record["status"] != "ok":
        case["control_failure_status"] = control_record["status"]
        if case["status"] == "ok":
            case["status"] = "control_invalid"
    elif case["status"] in ("ok", "prefix_mismatch") and case["metrics"] is not None:
        control_metrics = control_record["suffix_metrics"]
        # The full control suffix is stored once per anchor.  Per-case paired
        # scalar observations and deltas keep comparison convenient and small.
        case["paired_control_suffix_metrics"] = {field: control_metrics.get(field) for field in PAIRED_METRICS}
        case["paired_deltas"] = {
            field: (float(case["metrics"][field]) - float(control_metrics[field]))
            if case["metrics"].get(field) is not None and control_metrics.get(field) is not None else None
            for field in PAIRED_METRICS
        }
    return case


def _aggregate(cases, anchors, *, sections=False):
    valid = [
        {"metrics": case["metrics"], "source_anchor": anchors[case["anchor_id"]]}
        for case in cases if case["status"] == "ok"
    ]
    if valid:
        aggregate = historical._aggregate(valid) if sections else historical._aggregate_without_sections(valid)
        for item in [aggregate, *aggregate.get("section_aggregates", [])]:
            if item["stable_rejoin_observation_count"] == 0:
                item["mean_recovery_latency_samples"] = None
                item["mean_recovery_latency_s"] = None
    else:
        aggregate = {
            "case_count": 0, "recovery_activated_count": 0, "recovery_not_activated_count": 0,
            "recovery_completed_count": 0, "recovery_success_observation_count": 0,
            "recovery_success_count": 0, "recovery_failure_count": 0,
            "effective_recovery_sample_size": 0, "stable_rejoin_observation_count": 0,
            "recovery_success_rate": None, "conditional_recovery_success_rate": None,
            "all_case_recovery_pass_rate": None, "mean_recovery_latency_samples": None,
            "mean_recovery_latency_s": None,
        }
    aggregate.update(
        attempted_case_count=len(cases), valid_case_count=len(valid), invalid_case_count=len(cases) - len(valid),
        status_counts=dict(sorted(Counter(case["status"] for case in cases).items())),
        attempted_all_case_recovery_pass_rate=aggregate["recovery_success_count"] / len(cases) if cases else None,
    )
    return aggregate


def _blocked_selection(reasons):
    return {
        "ranking_source": "train", "acceptance_source": "held_out", "baseline_profile": BASELINE_DEFAULT,
        "eligible_candidates": [], "selected_candidate": None, "accepted": False,
        "rejected_if_none_pass": True, "blocked": True, "blocking_reasons": reasons,
        "promotion_allowed": False, "independent_confirmation": False,
        "interpretation": "diagnostic development validation only",
    }


def build_report(
    telemetry_dir: Path = DEFAULT_TELEMETRY_DIR,
    maps_dir: Path = DEFAULT_MAPS_DIR,
    *,
    calibration_report: Mapping[str, Any],
    max_steps: int = 10000,
    search_radius: int = 64,
    injection_step: int = DEFAULT_INJECTION_STEP,
    anchor_progress_m: Mapping[str, Iterable[float]] | None = None,
    disturbances: Iterable[DisturbanceSpec] = DISTURBANCES,
    held_out_disturbances: Iterable[DisturbanceSpec] = HELD_OUT_DISTURBANCES,
    recovery_heading_gains: Sequence[float] = DEFAULT_RECOVERY_GRID["recovery_heading_gain"],
    recovery_lateral_gains: Sequence[float] = DEFAULT_RECOVERY_GRID["recovery_lateral_gain"],
    recovery_brake_biases: Sequence[float] = DEFAULT_RECOVERY_GRID["recovery_brake_bias"],
) -> dict[str, Any]:
    """Run one fresh control per profile/anchor, then its fresh disturbed pairs.

    Fixture overrides retain the historical bounded grid API.  The default
    experiment freezes its eight configurations; this report performs no search
    beyond that declared grid.  A mid-rollout injection requires a nonempty prefix.
    """
    for name, value, minimum in (("max_steps", max_steps, 1), ("search_radius", search_radius, 0),
                                 ("injection_step", injection_step, 1)):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if injection_step >= max_steps:
        raise ValueError("injection_step must be less than max_steps")
    common, held_out = _validate_disturbances(disturbances, held_out_disturbances)
    grid = historical._validate_grid(recovery_heading_gains, recovery_lateral_gains, recovery_brake_biases)
    specs, profiles = _profile_specs(calibration_report, grid)
    config = ClosedLoopConfig(max_steps=max_steps, search_radius=search_radius)
    if anchor_progress_m is not None and (not anchor_progress_m or set(anchor_progress_m) - set(MAP_FILES)):
        raise ValueError("anchor_progress_m must be nonempty and contain only A01/A07 map names")

    loaded = {}
    maps = {}
    for map_name, (telemetry_filename, centerline_filename) in MAP_FILES.items():
        telemetry_path = Path(telemetry_dir) / telemetry_filename
        centerline_path = Path(maps_dir) / centerline_filename
        samples = load_rollout_csv(telemetry_path)
        geometry = build_centerline_geometry(np.load(centerline_path, allow_pickle=False))
        targets = None if anchor_progress_m is None else anchor_progress_m.get(map_name)
        anchors = select_source_anchors(map_name, samples, geometry, target_progress_m=targets)
        loaded[map_name] = geometry
        maps[map_name] = {
            "telemetry_file": str(telemetry_path), "centerline_file": str(centerline_path),
            "track_length_m": float(geometry.cumulative_lengths[-1]), "source_anchors": anchors,
            "cases_by_profile": {name: [] for name in specs},
            "controls_by_profile": {name: {} for name in specs},
        }

    # Split first.  No excluded train-anchor/held-out-family pose is constructed
    # or validated, including speed bounds that only make sense at injection.
    assignments, split = historical._split_source_anchors({name: item["source_anchors"] for name, item in maps.items()})
    manifest = {}
    anchors_by_id = {}
    for map_name, item in maps.items():
        geometry = loaded.pop(map_name)
        for anchor in item["source_anchors"]:
            anchor_id = anchor["anchor_id"]
            anchors_by_id[anchor_id] = anchor
            assigned = assignments[map_name][anchor_id]
            families = [(value, "common") for value in common]
            if assigned == "held_out":
                families.extend((value, "held_out_only") for value in held_out)
            for disturbance, scope in families:
                key = f"{map_name}:{anchor_id}:{disturbance.name}"
                if key in manifest:
                    raise ValueError(f"duplicate case key: {key}")
                manifest[key] = dict(map=map_name, anchor_id=anchor_id, split=assigned,
                                     disturbance=disturbance.name, disturbance_scope=scope)
            initial = dict(initial_position=np.asarray(anchor["position"], dtype=float),
                           initial_heading=np.asarray(anchor["heading"], dtype=float),
                           initial_speed_mps=float(anchor["speed_mps"]))
            for profile, (controller_type, controller_config) in specs.items():
                control_result, control_record = _control(
                    geometry, initial, controller_type, controller_config, config, injection_step
                )
                control_record["control_id"] = f"{map_name}:{anchor_id}:{profile}:control"
                item["controls_by_profile"][profile][anchor_id] = control_record
                for disturbance, scope in families:
                    key = f"{map_name}:{anchor_id}:{disturbance.name}"
                    case = _run_case(geometry, initial, controller_type, controller_config, config,
                                     injection_step, disturbance, control_result, control_record)
                    case.update(case_key=key, anchor_id=anchor_id, split=assigned,
                                disturbance=disturbance.name, disturbance_scope=scope,
                                paired_control_id=control_record["control_id"])
                    item["cases_by_profile"][profile].append(case)
                # Keep only summaries once this anchor's pairs have been checked.
                del control_result

    split["assignments"] = {key: value["split"] for key, value in manifest.items()}
    split["case_count"] = len(manifest)
    split["train_case_count"] = sum(value["split"] == "train" for value in manifest.values())
    split["held_out_case_count"] = len(manifest) - split["train_case_count"]
    split["case_counts_by_map"] = {
        name: {f"{part}_case_count": sum(value["map"] == name and value["split"] == part for value in manifest.values())
               for part in ("train", "held_out")}
        for name in maps
    }
    cases_by_profile = {
        name: [case for item in maps.values() for case in item["cases_by_profile"][name]] for name in specs
    }
    verified = all(
        len(cases) == len(manifest)
        and {case["case_key"]: case["split"] for case in cases} == split["assignments"]
        for cases in cases_by_profile.values()
    )
    all_cases = [case for cases in cases_by_profile.values() for case in cases]
    all_controls = [record for item in maps.values() for records in item["controls_by_profile"].values() for record in records.values()]
    bad_cases = sum(case["status"] != "ok" for case in all_cases)
    bad_controls = sum(record["status"] != "ok" for record in all_controls)
    issues = ([] if verified else ["case keys/assignments differ across profiles"])
    if bad_cases or bad_controls:
        issues.append(f"{bad_cases} invalid/unreached/error cases and {bad_controls} invalid controls")
    aggregates = {
        name: {part: _aggregate([case for case in cases if case["split"] == part], anchors_by_id, sections=True)
               for part in ("train", "held_out")}
        for name, cases in cases_by_profile.items()
    }
    disturbance_aggregates = []
    for map_name, item in maps.items():
        item["aggregates"] = {}
        for profile, cases in item["cases_by_profile"].items():
            item["aggregates"][profile] = {
                part: _aggregate([case for case in cases if case["split"] == part], anchors_by_id)
                for part in ("train", "held_out")
            }
            groups = sorted({(case["split"], case["disturbance"]) for case in cases})
            for part, disturbance in groups:
                group = [case for case in cases if (case["split"], case["disturbance"]) == (part, disturbance)]
                disturbance_aggregates.append(dict(map=map_name, profile=profile, split=part, disturbance=disturbance,
                                                   metrics=_aggregate(group, anchors_by_id)))

    candidate_names = [name for name in specs if name not in (BASELINE_DEFAULT, BASELINE_GUARDRAIL)]
    ranked = []
    if not issues:
        ranked = historical._rank_candidates([
            {"configuration_id": name, "train_aggregate": aggregates[name]["train"],
             "held_out_aggregate": aggregates[name]["held_out"]} for name in candidate_names
        ])
        selection = historical._selection(ranked, {"held_out_aggregate": aggregates[BASELINE_DEFAULT]["held_out"]})
        selection.update(blocked=False, blocking_reasons=[], promotion_allowed=False, independent_confirmation=False,
                         interpretation="diagnostic development validation only")
    else:
        selection = _blocked_selection(issues)

    report = {
        "scope": "offline_a01_a07_paired_midrollout_recovery",
        "evidence_class": "diagnostic_development_validation",
        "independent_confirmation": False, "promotion_allowed": False, "offline_only": True,
        "real_physics_validated": False,
        "safety_flags": {name: False for name in (
            "trackmania_launched", "training_started", "live_control_used", "azure_started",
            "checkpoints_loaded", "external_services_used", "controller_defaults_modified")},
        "calibration_input": None,
        "evaluation": {
            "status": "invalid" if issues else "diagnostic_development_validation",
            "previously_inspected_or_adapted_anchors_and_disturbances": True,
            "held_out_means": "excluded from this report's training ranking, not untouched confirmation",
            "independent_confirmation": False, "promotion_allowed": False,
        },
        "simulator": {
            "dt_s": config.dt_s, "max_steps": max_steps, "search_radius_segments": search_radius,
            "map_match_horizontal_only": config.map_match_horizontal_only, "section_bins": config.section_bins,
            "disturbance_injection_mode": "single_fixed_midrollout_step_before_command",
            "injection_step": injection_step, "injection_time_s": injection_step * config.dt_s,
        },
        "pairing": {
            "unit": "profile_plus_source_anchor", "fresh_controller_per_run": True,
            "one_unperturbed_control_per_profile_anchor": True,
            "cross_profile_prefix_equality_required": False,
            "interpretation": "Independent profiles may have different prefixes and injection states; each is paired only to its own control.",
            "metrics_window": "post-injection suffix only, identically summarized for the uninjected control",
            "paired_delta_direction": "disturbed minus its within-profile control; booleans are 0/1; missing observations yield null",
            "full_control_suffix_location": "maps[map].controls_by_profile[profile][anchor_id].suffix_metrics",
            "aggregate_denominator": "case_count counts valid suffix pairs; attempted/invalid counts include every scheduled case; any invalid case blocks selection",
        },
        "parameter_grid": {
            **{key: list(values) for key, values in zip(DEFAULT_RECOVERY_GRID, grid)},
            "configuration_count": len(candidate_names), "bounded_max_configurations": historical.MAX_GRID_CONFIGURATIONS,
            "recovery_only": True, "default_grid_frozen": grid == tuple(DEFAULT_RECOVERY_GRID.values()),
        },
        "profiles": profiles, "split": split,
        "disturbances": [asdict(item) for item in common],
        "held_out_only_disturbances": [asdict(item) for item in held_out],
        "case_manifest": manifest, "maps": maps, "aggregates": aggregates,
        "disturbance_aggregates": disturbance_aggregates,
        "candidates": [{"configuration_id": item["configuration_id"], "train_rank": item["train_rank"]} for item in ranked],
        "selection": selection,
        "validation": {"valid": not issues, "issues": issues, "same_case_keys_and_assignments_all_profiles": verified},
        "counts": {
            "profiles": len(profiles), "recovery_configurations": len(candidate_names),
            "cases_per_profile": len(manifest), "control_runs": len(all_controls),
            "disturbed_runs": len(all_cases), "valid_cases": len(all_cases) - bad_cases,
            "invalid_cases": bad_cases, "invalid_controls": bad_controls,
            "prefix_comparisons": sum(case["prefix_comparison"] is not None for case in all_cases),
            "matching_prefixes": sum(bool(case["prefix_comparison"] and case["prefix_comparison"]["matches"])
                                     for case in all_cases),
            "case_status_counts": dict(sorted(Counter(case["status"] for case in all_cases).items())),
        },
        "limitations": {
            "vertical_dynamics": {"modeled": False, "consequence": "No jumps, gravity or vertical recovery physics."},
            "collisions": {"modeled": False},
            "offtrack": {"modeled": False, "track_width_model": False, "off_track_termination": False},
            "projection_is_finish": False,
            "finish_interpretation": "Projected completion is distinct from the simulator's strict geometric finish gate; neither validates a game finish.",
            "split_unit": "initial_source_anchor",
            "entire_trajectories_disjoint": False,
            "split_interpretation": "Grouping initial anchors does not make their subsequent trajectories spatially disjoint.",
            "recorded_transition_replay": False,
            "live_control_or_training_claims": False,
        },
    }
    nonfinite = _nonfinite_paths(report)
    if nonfinite:
        report["validation"].update(valid=False, nonfinite_paths=nonfinite)
        issues.append("nonfinite observations serialized as null")
        report["evaluation"]["status"] = "invalid"
        report["selection"] = _blocked_selection(issues)
        report["candidates"] = []
    return _json_safe(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-dir", type=Path, default=DEFAULT_TELEMETRY_DIR)
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--search-radius", type=int, default=64)
    parser.add_argument("--injection-step", type=int, default=DEFAULT_INJECTION_STEP)
    args = parser.parse_args()
    report = build_report(args.telemetry_dir, args.maps_dir, calibration_report=load_calibration(args.calibration),
                          max_steps=args.max_steps, search_radius=args.search_radius, injection_step=args.injection_step)
    report["calibration_input"] = str(args.calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    baseline = report["aggregates"][BASELINE_DEFAULT]["held_out"]
    baseline_metrics = {field: baseline.get(field) for field in (
        "attempted_case_count", "invalid_case_count", "recovery_activated_count", "recovery_success_rate",
        "recovery_failure_count", "mean_recovery_latency_s", "progress_continuity_rate",
        "projected_progress_completion_rate", "finished_rate")}
    print(json.dumps({"output": str(args.output), "validation": report["validation"], "counts": report["counts"],
                      "baseline_held_out_metrics": baseline_metrics,
                      "selection": report["selection"]}, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
