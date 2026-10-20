import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_hybrid_speed_candidates import derive_candidates
from scripts.sweep_hybrid_recovery_parameters import (
    BASELINE_DEFAULT,
    BASELINE_GUARDRAIL,
    HELD_OUT_DISTURBANCES,
    build_report,
)
from trackmania_rl.hybrid_controller import ConservativeControllerConfig


def _calibration_fixture():
    return {
        "scope": "offline_recorded_speed_and_section_calibration",
        "traces": [
            {
                "sections": [
                    {
                        "actual_speed_mps": {"p25": 60.0, "p50": 70.0, "p75": 80.0},
                        "nominal_target_speed_mps": {"p25": 12.0, "p50": 14.0, "p75": 55.0},
                        "absolute_curvature": {"p50": 0.25},
                    },
                    {
                        "actual_speed_mps": {"p25": 50.0, "p50": 65.0, "p75": 75.0},
                        "nominal_target_speed_mps": {"p25": 11.0, "p50": 13.0, "p75": 50.0},
                        "absolute_curvature": {"p50": 0.10},
                    },
                ]
            }
        ],
    }


class HybridRecoverySweepTests(unittest.TestCase):
    def _fixtures(self, directory: str) -> tuple[Path, Path]:
        root = Path(directory)
        telemetry = root / "telemetry"
        maps = root / "maps"
        telemetry.mkdir(exist_ok=True)
        maps.mkdir(exist_ok=True)
        points = np.asarray([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [80.0, 0.0, 0.0], [120.0, 0.0, 0.0]])
        for filename in ("A01-Race_0.5m_cl2.npy", "A07-Race_0.5m_cl2.npy"):
            np.save(maps / filename, points)
        fields = (
            "timestamp_s",
            "position_x_m",
            "position_y_m",
            "position_z_m",
            "heading_x",
            "heading_y",
            "heading_z",
            "speed_mps",
        )
        rows = (
            (0.0, 10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 16.0),
            (1.0, 50.0, 0.0, 0.0, 1.0, 0.0, 0.0, 16.0),
            (2.0, 90.0, 0.0, 0.0, 1.0, 0.0, 0.0, 16.0),
        )
        for filename in ("A01-Race.repaired.csv", "A07-Race.repaired.csv"):
            with (telemetry / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
        return telemetry, maps

    def _report(self, directory: str, *, anchor_progress_m=None):
        telemetry, maps = self._fixtures(directory)
        return build_report(
            telemetry,
            maps,
            calibration_report=_calibration_fixture(),
            max_steps=35,
            search_radius=8,
            anchor_progress_m=anchor_progress_m
            if anchor_progress_m is not None
            else {"A01-Race": (20.0, 80.0), "A07-Race": (20.0, 80.0)},
        )

    def test_split_and_training_ranking_are_deterministic(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            first = self._report(directory)
            # A second report uses the same fixture contents and must produce
            # the same split, ranking, and selection decisions.
            second = self._report(directory)

        self.assertEqual(first["split"], second["split"])
        self.assertEqual(
            [(item["configuration_id"], item["train_rank"]) for item in first["candidates"]],
            [(item["configuration_id"], item["train_rank"]) for item in second["candidates"]],
        )
        self.assertGreater(first["split"]["train_case_count"], 0)
        self.assertGreater(first["split"]["held_out_case_count"], 0)
        self.assertEqual(first["selection"], second["selection"])

    def test_documented_regions_are_held_out_as_complete_source_groups(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            report = self._report(
                directory,
                anchor_progress_m={"A01-Race": (20.0, 80.0), "A07-Race": (20.0, 379.0)},
            )

        split = report["split"]
        self.assertIn("documented_divergence_region", split["policy"])
        for map_name, map_report in report["maps"].items():
            documented = {
                anchor["anchor_id"]
                for anchor in map_report["source_anchors"]
                if anchor["anchor_role"] == "documented_divergence_region"
            }
            self.assertTrue(documented)
            self.assertEqual(set(split["held_out_anchor_ids_by_map"][map_name]), documented)
            self.assertTrue(split["train_anchor_ids_by_map"][map_name])
            self.assertTrue(split["held_out_anchor_ids_by_map"][map_name])

            by_anchor = {}
            for case in map_report["cases_by_profile"][BASELINE_DEFAULT]:
                by_anchor.setdefault(case["source_anchor"]["anchor_id"], set()).add(case["split"])
            self.assertTrue(by_anchor)
            self.assertTrue(all(len(assignments) == 1 for assignments in by_anchor.values()))
            self.assertEqual(
                {anchor_id for anchor_id, assignments in by_anchor.items() if assignments == {"held_out"}},
                documented,
            )
            for profile_name, cases in map_report["cases_by_profile"].items():
                profile_by_anchor = {}
                for case in cases:
                    profile_by_anchor.setdefault(case["source_anchor"]["anchor_id"], set()).add(case["split"])
                self.assertEqual(profile_by_anchor, by_anchor, msg=profile_name)

    def test_fixture_without_documented_region_uses_highest_progress_fallback(self):
        targets = {"A01-Race": (20.0, 60.0), "A07-Race": (20.0, 60.0)}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            report = self._report(directory, anchor_progress_m=targets)

        split = report["split"]
        self.assertIn("highest-progress source anchor", split["policy"])
        for map_name, map_report in report["maps"].items():
            highest = max(
                map_report["source_anchors"],
                key=lambda anchor: (anchor["source_progress_m"], anchor["anchor_id"]),
            )["anchor_id"]
            self.assertEqual(split["held_out_anchor_ids_by_map"][map_name], [highest])
            self.assertEqual(
                set(split["held_out_anchor_ids_by_map"][map_name])
                | set(split["train_anchor_ids_by_map"][map_name]),
                {anchor["anchor_id"] for anchor in map_report["source_anchors"]},
            )

    def test_split_requires_train_and_held_out_anchor_per_map(self):
        targets = {"A01-Race": (20.0,), "A07-Race": (20.0,)}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            with self.assertRaisesRegex(ValueError, "map 'A01-Race'.*train anchor.*held-out anchor"):
                self._report(directory, anchor_progress_m=targets)

    def test_candidates_change_only_recovery_fields(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            report = self._report(directory)
        defaults = ConservativeControllerConfig()
        recovery_fields = {"recovery_heading_gain", "recovery_lateral_gain", "recovery_brake_bias"}
        for item in report["candidate_configs"]:
            config = item["config"]
            self.assertTrue(item["nominal_fields_match_default"])
            for field in defaults.__dataclass_fields__:
                self.assertEqual(config[field], getattr(defaults, field))
            self.assertEqual(set(item["recovery_parameters"]), recovery_fields)
        self.assertEqual(report["parameter_grid"]["configuration_count"], 8)

    def test_guardrail_is_derived_and_metrics_include_held_out_sections(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            report = self._report(directory)
        expected = next(candidate for candidate in derive_candidates(_calibration_fixture()) if candidate.name == BASELINE_GUARDRAIL)
        self.assertEqual(report["baseline_profiles"][BASELINE_GUARDRAIL]["profile"]["config"], {
            key: getattr(expected.config, key) for key in expected.config.__dataclass_fields__
        })
        self.assertTrue(report["baseline_profiles"][BASELINE_GUARDRAIL]["profile"]["derived_from_calibration"])
        self.assertIn("section_aggregates", report["aggregates"][BASELINE_DEFAULT]["held_out"])
        self.assertIn("simulator_section_aggregates", report["aggregates"][BASELINE_DEFAULT]["held_out"])
        self.assertIn("recovery_success_rate", report["aggregates"][BASELINE_DEFAULT]["held_out"])
        self.assertIn("projected_progress_completion_rate", report["aggregates"][BASELINE_DEFAULT]["held_out"])
        self.assertEqual(report["selection"]["ranking_source"], "train")
        self.assertEqual(report["selection"]["acceptance_source"], "held_out")

    def test_unseen_disturbance_families_are_held_out_only_and_reported(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            report = self._report(directory)

        held_out_names = {item.name for item in HELD_OUT_DISTURBANCES}
        self.assertEqual(
            report["disturbance_novelty"]["held_out_only_disturbance_families"],
            sorted(held_out_names),
        )
        self.assertTrue(report["disturbance_novelty"]["unseen_family"])
        self.assertEqual(report["disturbance_novelty"]["source_overlap_check"], "passed")
        self.assertEqual(
            {item["name"] for item in report["held_out_only_disturbances"]},
            held_out_names,
        )

        for case in report["maps"]["A01-Race"]["cases_by_profile"][BASELINE_DEFAULT]:
            name = case["disturbance"]["name"]
            if name in held_out_names:
                self.assertEqual(case["split"], "held_out")
                self.assertEqual(case["disturbance"]["scope"], "held_out_only")
            self.assertIn("realized_initial_lateral_error_m", case["disturbed_pose"])
            self.assertIn("realized_initial_heading_error_rad", case["disturbed_pose"])

        for profile_name, cases in report["maps"]["A07-Race"]["cases_by_profile"].items():
            self.assertTrue(
                all(case["split"] == "held_out" for case in cases if case["disturbance"]["name"] in held_out_names),
                msg=profile_name,
            )

    def test_aggregate_exposes_recovery_denominators_and_ceiling_status(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            report = self._report(directory)

        aggregate = report["aggregates"][BASELINE_DEFAULT]["held_out"]
        self.assertEqual(
            aggregate["recovery_not_activated_count"] + aggregate["recovery_activated_count"],
            aggregate["case_count"],
        )
        self.assertEqual(
            aggregate["recovery_success_count"] + aggregate["recovery_failure_count"],
            aggregate["recovery_success_observation_count"],
        )
        self.assertIn("conditional_recovery_success_rate", aggregate)
        self.assertIn("all_case_recovery_pass_rate", aggregate)
        self.assertIn("effective_recovery_sample_size", aggregate)
        self.assertIn("stable_rejoin_rate", aggregate)
        self.assertIn("mean_recovery_latency_s", aggregate)
        self.assertIn("total_redivergence_count", aggregate)
        self.assertEqual(report["evaluation"]["evaluation_status"], "uninformative_ceiling")
        self.assertTrue(report["evaluation"]["ceiling_reason"])

    def test_report_has_explicit_offline_no_live_training_or_checkpoint_flags(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            report = self._report(directory)
        self.assertTrue(report["offline_only"])
        self.assertFalse(report["trackmania_launched"])
        self.assertFalse(report["training_started"])
        self.assertFalse(report["live_control_used"])
        self.assertFalse(report["checkpoints_loaded"])
        self.assertFalse(report["safety_flags"]["trackmania_launched"])
        self.assertFalse(report["safety_flags"]["training_started"])
        self.assertFalse(report["safety_flags"]["checkpoints_loaded"])
        self.assertIn("accepted", report["selection"])
        self.assertIn("rejected_if_none_pass", report["selection"])


if __name__ == "__main__":
    unittest.main()
