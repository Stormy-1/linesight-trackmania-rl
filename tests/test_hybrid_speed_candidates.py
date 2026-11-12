import unittest

from scripts.evaluate_hybrid_speed_candidates import (
    CURVATURE_GAIN_BOUNDS,
    MAX_SPEED_BOUNDS,
    MIN_SPEED_BOUNDS,
    build_report,
    derive_candidates,
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
                        "actual_speed_mps": {"p25": 64.0, "p50": 74.0, "p75": 84.0},
                        "nominal_target_speed_mps": {"p25": 12.0, "p50": 16.0, "p75": 60.0},
                        "absolute_curvature": {"p50": 0.35},
                    },
                ]
            }
        ],
    }


class HybridSpeedCandidateTests(unittest.TestCase):
    def test_candidates_are_bounded_deterministic_and_separate_from_defaults(self):
        defaults_before = ConservativeControllerConfig()
        first = derive_candidates(_calibration_fixture())
        second = derive_candidates(_calibration_fixture())

        self.assertEqual(first, second)
        self.assertEqual([item.name for item in first], ["captured_p25_guardrail", "captured_p50_balanced", "captured_p75_exploratory"])
        self.assertEqual(ConservativeControllerConfig(), defaults_before)
        self.assertEqual(len({item.config.max_speed_mps for item in first}), 3)
        for candidate in first:
            self.assertGreaterEqual(candidate.config.max_speed_mps, MAX_SPEED_BOUNDS[0])
            self.assertLessEqual(candidate.config.max_speed_mps, MAX_SPEED_BOUNDS[1])
            self.assertGreaterEqual(candidate.config.min_speed_mps, MIN_SPEED_BOUNDS[0])
            self.assertLessEqual(candidate.config.min_speed_mps, MIN_SPEED_BOUNDS[1])
            self.assertGreaterEqual(candidate.config.curvature_gain, CURVATURE_GAIN_BOUNDS[0])
            self.assertLessEqual(candidate.config.curvature_gain, CURVATURE_GAIN_BOUNDS[1])

    def test_report_compares_completion_tracking_recovery_and_speed_targets(self):
        report = build_report(_calibration_fixture())

        self.assertFalse(report["controller_defaults_modified"])
        self.assertTrue(report["candidate_profiles_are_descriptive_only"])
        self.assertEqual(len(report["profiles"]), 4)
        self.assertEqual(report["profiles"][0]["role"], "reference")
        for profile in report["profiles"]:
            summary = profile["summary"]
            self.assertIn("completion", summary)
            self.assertIn("progress", summary)
            self.assertIn("lateral_error", summary)
            self.assertIn("heading_error", summary)
            self.assertIn("recovery", summary)
            self.assertIn("speed_target_behavior", summary)
            self.assertEqual(len(profile["scenarios"]), 2)
            self.assertTrue(all("finished" in item["completion"] for item in profile["scenarios"]))
            self.assertTrue(all("mean_target_speed_mps" in item["speed_target_behavior"] for item in profile["scenarios"]))


if __name__ == "__main__":
    unittest.main()
