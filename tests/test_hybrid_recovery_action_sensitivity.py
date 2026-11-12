import unittest
from unittest.mock import patch

from scripts import sweep_hybrid_recovery_action_sensitivity as module


class HybridRecoveryActionSensitivityTests(unittest.TestCase):
    def test_rejects_duplicate_or_invalid_steps(self):
        with self.assertRaises(ValueError):
            module.build_sensitivity_report({}, steps=(25, 25))
        with self.assertRaises(ValueError):
            module.build_sensitivity_report({}, steps=(0, 70))

    def test_records_are_diagnostic_and_selection_is_not_promotion(self):
        fake_report = {
            "candidates": [{"configuration_id": "candidate_a"}],
            "selection": {"eligible_candidates": ["candidate_a"], "selected_candidate": "candidate_a"},
            "aggregates": {"candidate_a": {"held_out": {key: value for key, value in {
                "case_count": 1,
                "recovery_activated_count": 1,
                "recovery_success_rate": 1.0,
                "recovery_failure_count": 0,
                "progress_continuity_rate": 1.0,
                "stable_rejoin_rate": 1.0,
                "mean_recovery_latency_s": 0.5,
                "total_redivergence_count": 0,
                "projected_progress_completion_rate": 1.0,
            }.items()}}},
            "disturbance_aggregates": [],
        }
        with patch.object(module, "build_report", return_value=fake_report):
            report = module.build_sensitivity_report({}, steps=(25,))
        self.assertFalse(report["promotion_allowed"])
        self.assertTrue(report["experiment"]["selection_rule_unchanged"])
        self.assertEqual(len(report["heading_lateral_records"]), 1)
        self.assertEqual(len(report["brake_records"]), 1)
        self.assertTrue(report["heading_lateral_records"][0]["selection_selected"])


if __name__ == "__main__":
    unittest.main()
