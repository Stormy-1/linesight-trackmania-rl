"""Small offline report fixtures; all temporary files stay in the workspace."""

import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np

from scripts import compare_hybrid_midrollout_recovery as report_module
from scripts import sweep_hybrid_recovery_parameters as historical
from scripts.evaluate_hybrid_speed_candidates import derive_candidates
from trackmania_rl.hybrid_controller import ConservativeCenterlineController, ConservativeControllerConfig
from trackmania_rl.hybrid_replay import DisturbanceSpec, summarize_rollout_suffix
from trackmania_rl.hybrid_simulator import ClosedLoopConfig, simulate_closed_loop
from trackmania_rl.hybrid_state import build_centerline_geometry


WORKSPACE = Path(__file__).resolve().parents[1]


def _calibration():
    return {
        "scope": "offline_recorded_speed_and_section_calibration",
        "traces": [{"sections": [
            {"actual_speed_mps": {"p25": 60.0, "p50": 70.0, "p75": 80.0},
             "nominal_target_speed_mps": {"p25": 12.0, "p50": 14.0, "p75": 55.0},
             "absolute_curvature": {"p50": 0.25}},
            {"actual_speed_mps": {"p25": 50.0, "p50": 65.0, "p75": 75.0},
             "nominal_target_speed_mps": {"p25": 11.0, "p50": 13.0, "p75": 50.0},
             "absolute_curvature": {"p50": 0.10}},
        ]}],
    }


def _fixtures(root, *, length=600.0, positions=(10.0, 90.0), first_speed=16.0, lateral=0.0):
    telemetry, maps = root / "telemetry", root / "maps"
    telemetry.mkdir(exist_ok=True)
    maps.mkdir(exist_ok=True)
    points = np.asarray([[0.0, 0.0, 0.0], [length / 2, 0.0, 0.0], [length, 0.0, 0.0]])
    fields = ("timestamp_s", "position_x_m", "position_y_m", "position_z_m",
              "heading_x", "heading_y", "heading_z", "speed_mps")
    for telemetry_name, map_name in report_module.MAP_FILES.values():
        np.save(maps / map_name, points)
        with (telemetry / telemetry_name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows([
                (0.0, positions[0], 0.0, lateral, 1.0, 0.0, 0.0, first_speed),
                (1.0, positions[1], 0.0, lateral, 1.0, 0.0, 0.0, 16.0),
            ])
    return telemetry, maps


def _cases(report):
    return [case for item in report["maps"].values() for cases in item["cases_by_profile"].values() for case in cases]


class HybridMidrolloutReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="midrollout-test-", dir=WORKSPACE)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directories = _fixtures(self.root)

    def report(self, **overrides):
        kwargs: dict[str, Any] = dict(
            calibration_report=_calibration(), max_steps=14, search_radius=8, injection_step=3,
            anchor_progress_m={"A01-Race": (20.0, 80.0), "A07-Race": (20.0, 80.0)},
            recovery_heading_gains=(1.4,), recovery_lateral_gains=(0.12,), recovery_brake_biases=(0.25,),
        )
        kwargs.update(overrides)
        return report_module.build_report(*self.directories, **kwargs)

    def test_determinism_split_and_historical_decisions(self):
        first, second = self.report(), self.report()
        self.assertEqual(first, second)
        self.assertTrue(first["validation"]["valid"])
        self.assertTrue(first["validation"]["same_case_keys_and_assignments_all_profiles"])
        self.assertEqual(first["counts"]["cases_per_profile"], 16)
        self.assertEqual(first["counts"]["control_runs"], len(first["profiles"]) * 4)
        self.assertEqual(first["counts"]["disturbed_runs"], len(first["profiles"]) * 16)
        self.assertEqual(first["counts"]["invalid_cases"], 0)
        self.assertEqual(first["counts"]["matching_prefixes"], first["counts"]["disturbed_runs"])
        assignments, expected = historical._split_source_anchors({
            name: item["source_anchors"] for name, item in first["maps"].items()
        })
        self.assertEqual(first["split"]["anchor_assignments"], assignments)
        self.assertEqual(first["split"]["held_out_anchor_ids"], expected["held_out_anchor_ids"])
        self.assertTrue(set(first["split"]["train_anchor_ids"]).isdisjoint(first["split"]["held_out_anchor_ids"]))
        for item in first["maps"].values():
            reference = None
            for cases in item["cases_by_profile"].values():
                actual = {case["case_key"]: case["split"] for case in cases}
                if reference is not None:
                    self.assertEqual(actual, reference)
                reference = actual
                for case in cases:
                    if case["disturbance_scope"] == "held_out_only":
                        self.assertEqual(case["split"], "held_out")
                    self.assertEqual(first["case_manifest"][case["case_key"]]["split"], case["split"])
        ranked = historical._rank_candidates([
            {"configuration_id": row["configuration_id"],
             "train_aggregate": first["aggregates"][row["configuration_id"]]["train"],
             "held_out_aggregate": first["aggregates"][row["configuration_id"]]["held_out"]}
            for row in first["candidates"]
        ])
        self.assertEqual(first["candidates"], [
            {"configuration_id": row["configuration_id"], "train_rank": row["train_rank"]} for row in ranked
        ])
        selection = historical._selection(ranked, {"held_out_aggregate": first["aggregates"][historical.BASELINE_DEFAULT]["held_out"]})
        for key, value in selection.items():
            self.assertEqual(first["selection"][key], value)
        json.dumps(first, allow_nan=False)

    def test_default_profiles_remain_frozen_and_guardrail_is_derived(self):
        report = self.report(injection_step=10, **{
            "recovery_heading_gains": historical.DEFAULT_RECOVERY_GRID["recovery_heading_gain"],
            "recovery_lateral_gains": historical.DEFAULT_RECOVERY_GRID["recovery_lateral_gain"],
            "recovery_brake_biases": historical.DEFAULT_RECOVERY_GRID["recovery_brake_bias"],
        })
        self.assertTrue(report["parameter_grid"]["default_grid_frozen"])
        self.assertEqual(report["counts"]["profiles"], 10)
        self.assertEqual(report["counts"]["recovery_configurations"], 8)
        self.assertEqual(report["simulator"]["injection_time_s"], 1.0)
        defaults = asdict(ConservativeControllerConfig())
        for profile in report["profiles"].values():
            if profile["role"] == "recovery_parameter_candidate":
                self.assertTrue(all(profile["config"][key] == value for key, value in defaults.items()))
                self.assertEqual(set(profile["recovery_parameters"]), set(historical.DEFAULT_RECOVERY_GRID))
        guardrail = next(candidate for candidate in derive_candidates(_calibration()) if candidate.name == historical.BASELINE_GUARDRAIL)
        self.assertEqual(report["profiles"][historical.BASELINE_GUARDRAIL]["config"], asdict(guardrail.config))
        self.assertTrue(report["validation"]["valid"])

    def test_actual_prefix_fidelity_and_suffix_pair_metrics(self):
        report = self.report()
        for map_name, item in report["maps"].items():
            anchors = {anchor["anchor_id"]: anchor for anchor in item["source_anchors"]}
            for profile, cases in item["cases_by_profile"].items():
                for case in cases:
                    comparison = case["prefix_comparison"]
                    self.assertTrue(comparison["matches"] and comparison["complete"])
                    self.assertEqual(comparison["compared_steps"], 3)
                    self.assertEqual(comparison["mismatch_count"], 0)
                    self.assertTrue(comparison["pre_injection_snapshot_matches_control"])
                    self.assertEqual(case["prefix"]["recovery_flags"], [False] * 3)
                    self.assertTrue(case["prefix"]["clean_at_injection"])
                    self.assertEqual(case["pre_injection"]["lateral_error_m"], 0.0)
                    self.assertNotEqual(case["post_injection"]["lateral_error_m"], 0.0)
                    self.assertEqual(case["metrics"]["steps"], 11)
                    self.assertAlmostEqual(case["metrics"]["elapsed_s"], 1.1)
                    self.assertAlmostEqual(case["total_elapsed_s"], 1.4)
                    control = item["controls_by_profile"][profile][case["anchor_id"]]
                    self.assertEqual(case["paired_control_id"], control["control_id"])
                    for field, value in case["paired_control_suffix_metrics"].items():
                        self.assertEqual(value, control["suffix_metrics"].get(field))
                        observed, delta = case["metrics"].get(field), case["paired_deltas"][field]
                        if observed is None or value is None:
                            self.assertIsNone(delta)
                        else:
                            self.assertAlmostEqual(delta, float(observed) - float(value))
                    self.assertEqual(anchors[case["anchor_id"]]["position"][2], 0.0)
            anchor = item["source_anchors"][0]
            geometry = build_centerline_geometry(np.load(self.directories[1] / report_module.MAP_FILES[map_name][1], allow_pickle=False))
            control = simulate_closed_loop(
                geometry, initial_position=np.asarray(anchor["position"]), initial_heading=np.asarray(anchor["heading"]),
                initial_speed_mps=anchor["speed_mps"], controller=ConservativeCenterlineController(),
                config=ClosedLoopConfig(max_steps=14, search_radius=8),
            )
            expected = summarize_rollout_suffix(control, 3, dt_s=0.1)
            self.assertEqual(item["controls_by_profile"][historical.BASELINE_DEFAULT][anchor["anchor_id"]]["suffix_metrics"], expected)

    def test_recovery_denominators_and_missing_latency_are_explicit(self):
        report = self.report(max_steps=5)
        aggregates = [value for profile in report["aggregates"].values() for value in profile.values()]
        aggregates += [row["metrics"] for row in report["disturbance_aggregates"]]
        for item in report["maps"].values():
            aggregates += [value for profile in item["aggregates"].values() for value in profile.values()]
        for aggregate in aggregates:
            self.assertEqual(aggregate["valid_case_count"], aggregate["attempted_case_count"])
            self.assertEqual(aggregate["invalid_case_count"], 0)
            self.assertEqual(aggregate["recovery_activated_count"] + aggregate["recovery_not_activated_count"], aggregate["case_count"])
            self.assertEqual(aggregate["recovery_success_count"] + aggregate["recovery_failure_count"], aggregate["recovery_success_observation_count"])
            self.assertEqual(aggregate["stable_rejoin_observation_count"], 0)
            self.assertIsNone(aggregate["mean_recovery_latency_samples"])
            self.assertIsNone(aggregate["mean_recovery_latency_s"])
            for section in aggregate.get("section_aggregates", []):
                self.assertIsNone(section["mean_recovery_latency_s"])
        self.assertEqual({row["map"] for row in report["disturbance_aggregates"]}, set(report_module.MAP_FILES))

    def test_fresh_controllers_once_per_run_and_no_extra_commands(self):
        created = []

        class CountingController(ConservativeCenterlineController):
            def __init__(self, config=None):
                super().__init__(config)
                self.calls = 0
                created.append(self)

            def command(self, state, geometry):
                self.calls += 1
                return super().command(state, geometry)

        with patch.object(report_module, "ConservativeCenterlineController", CountingController):
            with patch.object(report_module, "simulate_closed_loop", wraps=simulate_closed_loop) as controls:
                report = self.report()
        self.assertEqual(controls.call_count, report["counts"]["control_runs"])
        expected_steps = []
        for item in report["maps"].values():
            for name in (historical.BASELINE_DEFAULT, historical.BASELINE_GUARDRAIL):
                expected_steps.extend(record["terminal"]["steps"] for record in item["controls_by_profile"][name].values())
                expected_steps.extend(case["terminal"]["steps"] for case in item["cases_by_profile"][name])
        self.assertEqual(len(created), len(expected_steps))
        self.assertEqual(sum(controller.calls for controller in created), sum(expected_steps))
        self.assertTrue(all(controller.calls == 14 for controller in created))

    def test_heldout_only_poses_are_not_validated_on_excluded_training_anchors(self):
        _fixtures(self.root, first_speed=79.0)
        original = report_module.replay_midrollout_disturbance
        seen = []

        def checked(geometry, **kwargs):
            if kwargs["injection"].disturbance.name.startswith("heldout_"):
                seen.append(kwargs["initial_speed_mps"])
            return original(geometry, **kwargs)

        with patch.object(report_module, "replay_midrollout_disturbance", side_effect=checked):
            report = self.report(max_steps=3, injection_step=1)
        self.assertTrue(seen)
        self.assertEqual(set(seen), {16.0})
        self.assertTrue(report["validation"]["valid"])
        self.assertEqual(report["counts"]["invalid_cases"], 0)

    def test_dirty_prefix_diagnostics_are_within_profile_not_cross_profile(self):
        _fixtures(self.root, lateral=12.0)
        report = self.report()
        self.assertTrue(report["validation"]["valid"])
        self.assertFalse(report["pairing"]["cross_profile_prefix_equality_required"])
        self.assertTrue(any(case["prefix"]["any_recovery"] for case in _cases(report)))
        self.assertTrue(any(case["prefix"]["clean_at_injection"] is False for case in _cases(report)))
        item = report["maps"]["A01-Race"]
        anchor_id = item["source_anchors"][0]["anchor_id"]
        candidate = report["candidates"][0]["configuration_id"]
        baseline = item["controls_by_profile"][historical.BASELINE_DEFAULT][anchor_id]
        recovery = item["controls_by_profile"][candidate][anchor_id]
        self.assertNotEqual(baseline["pre_injection"], recovery["pre_injection"])
        self.assertTrue(all(case["prefix_comparison"]["matches"] for case in _cases(report)))

    def test_invalid_bounds_and_empty_sets(self):
        cases = [
            ({"max_steps": 0}, "max_steps"), ({"max_steps": True}, "max_steps"),
            ({"search_radius": -1}, "search_radius"), ({"search_radius": 1.5}, "search_radius"),
            ({"injection_step": 0}, "injection_step"), ({"injection_step": True}, "injection_step"),
            ({"injection_step": 14}, "less than max_steps"), ({"injection_step": -2}, "injection_step"),
            ({"disturbances": ()}, "at least one"), ({"held_out_disturbances": ()}, "at least one"),
            ({"recovery_heading_gains": ()}, "at least one"),
            ({"recovery_heading_gains": (1.4, 1.4)}, "duplicate"),
            ({"recovery_heading_gains": (float("nan"),)}, "finite"),
            ({"recovery_heading_gains": tuple(range(17))}, "maximum"),
            ({"recovery_brake_biases": (1.1,)}, "between 0 and 1"),
            ({"anchor_progress_m": {}}, "anchor_progress_m"),
            ({"anchor_progress_m": {"unknown": (20.0,)}}, "A01/A07"),
            ({"anchor_progress_m": {"A01-Race": ()}}, "at least one target"),
            ({"anchor_progress_m": {"A01-Race": (20.0,)}}, "train anchor.*held-out anchor"),
        ]
        for override, message in cases:
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, message):
                self.report(**override)

    def test_duplicate_names_and_relabeled_numeric_signatures_rejected(self):
        cases = [
            ((DisturbanceSpec("a", 10), DisturbanceSpec("a", -10)), (DisturbanceSpec("b", 20),), "duplicate"),
            ((DisturbanceSpec("a", 10),), (DisturbanceSpec("a", 20),), "duplicate"),
            ((DisturbanceSpec("a", 10),), (DisturbanceSpec("b", 20), DisturbanceSpec("b", -20)), "duplicate"),
            ((DisturbanceSpec("a", 10),), (DisturbanceSpec("relabeled", 10),), "equivalent numeric"),
            ((DisturbanceSpec("a", 10),), (DisturbanceSpec("relabeled", 10, 2 * math.pi),), "equivalent numeric"),
        ]
        for common, heldout, message in cases:
            with self.subTest(common=common, heldout=heldout), self.assertRaisesRegex(ValueError, message):
                self.report(disturbances=common, held_out_disturbances=heldout)

    def test_unreached_injection_is_retained_and_blocks_selection(self):
        _fixtures(self.root, length=120.0, positions=(10.0, 119.0))
        with patch.object(historical, "_rank_candidates", side_effect=AssertionError("must not rank invalid report")):
            report = self.report(injection_step=10)
        self.assertFalse(report["validation"]["valid"])
        self.assertTrue(report["selection"]["blocked"])
        self.assertIsNone(report["selection"]["selected_candidate"])
        self.assertFalse(report["selection"]["accepted"])
        self.assertEqual(report["counts"]["cases_per_profile"], 16)
        self.assertEqual(report["counts"]["disturbed_runs"], 48)
        unreached = [case for case in _cases(report) if case["status"] == "injection_not_reached"]
        self.assertTrue(unreached)
        self.assertTrue(all(case["terminal"]["steps"] == 1 for case in unreached))
        self.assertTrue(all(case["unreached_step_index"] == 10 for case in unreached))
        self.assertTrue(all(case["split"] == "held_out" for case in unreached))
        for profile in report["aggregates"].values():
            self.assertEqual(profile["held_out"]["invalid_case_count"], profile["held_out"]["attempted_case_count"])
            self.assertIsNone(profile["held_out"]["mean_recovery_latency_s"])
        json.dumps(report, allow_nan=False)

    def test_actual_prefix_mismatch_blocks_selection(self):
        original = report_module.replay_midrollout_disturbance

        def changed_prefix(geometry, **kwargs):
            outcome = original(geometry, **kwargs)
            first = outcome.result.steps[0]
            changed = replace(first, command=replace(first.command, steering=first.command.steering + 0.01))
            result = replace(outcome.result, steps=(changed, *outcome.result.steps[1:]))
            return SimpleNamespace(result=result, metrics=outcome.metrics)

        with patch.object(report_module, "replay_midrollout_disturbance", side_effect=changed_prefix):
            report = self.report()
        self.assertTrue(report["selection"]["blocked"])
        self.assertTrue(all(case["status"] == "prefix_mismatch" for case in _cases(report)))
        self.assertTrue(all(case["prefix_comparison"]["first_mismatch"] == {"step_index": 0, "fields": ["command"]} for case in _cases(report)))
        self.assertEqual(report["counts"]["matching_prefixes"], 0)

    def test_invalid_injection_and_simulator_errors_are_not_dropped(self):
        report = self.report(held_out_disturbances=(DisturbanceSpec("speed_out_of_bounds", speed_delta_mps=100),))
        self.assertTrue(report["selection"]["blocked"])
        self.assertTrue(any(case["status"] == "invalid" for case in _cases(report)))
        self.assertTrue(report["validation"]["same_case_keys_and_assignments_all_profiles"])
        with patch.object(report_module, "replay_midrollout_disturbance", side_effect=RuntimeError("fixture simulation failure")):
            failed = self.report()
        self.assertEqual(failed["counts"]["invalid_cases"], failed["counts"]["disturbed_runs"])
        self.assertTrue(all(case["status"] == "error" and case["error"]["type"] == "RuntimeError" for case in _cases(failed)))
        self.assertTrue(failed["selection"]["blocked"])
        json.dumps(failed, allow_nan=False)

    def test_nonfinite_observations_serialize_as_null_and_block_selection(self):
        original = report_module.replay_midrollout_disturbance

        def nonfinite(geometry, **kwargs):
            outcome = original(geometry, **kwargs)
            metrics = dict(outcome.metrics)
            metrics["post_rollout"] = dict(metrics["post_rollout"], mean_abs_lateral_error_m=float("nan"))
            return SimpleNamespace(result=outcome.result, metrics=metrics)

        with patch.object(report_module, "replay_midrollout_disturbance", side_effect=nonfinite):
            report = self.report()
        self.assertTrue(report["selection"]["blocked"])
        self.assertFalse(report["validation"]["valid"])
        self.assertTrue(all(case["status"] == "invalid" for case in _cases(report)))
        self.assertTrue(all(case["metrics"]["mean_abs_lateral_error_m"] is None for case in _cases(report)))
        self.assertTrue(report["validation"]["nonfinite_paths"])
        json.dumps(report, allow_nan=False)

    def test_no_promotion_independent_confirmation_or_live_claims(self):
        report = self.report()
        self.assertEqual(report["evidence_class"], "diagnostic_development_validation")
        self.assertTrue(report["evaluation"]["previously_inspected_or_adapted_anchors_and_disturbances"])
        for value in (report, report["selection"], report["evaluation"]):
            self.assertIs(value["promotion_allowed"], False)
            self.assertIs(value["independent_confirmation"], False)
        self.assertTrue(report["offline_only"])
        self.assertFalse(report["real_physics_validated"])
        self.assertTrue(all(value is False for value in report["safety_flags"].values()))
        for key in ("vertical_dynamics", "collisions", "offtrack"):
            self.assertFalse(report["limitations"][key]["modeled"])
        self.assertFalse(report["limitations"]["projection_is_finish"])
        self.assertFalse(report["limitations"]["entire_trajectories_disjoint"])


if __name__ == "__main__":
    unittest.main()
