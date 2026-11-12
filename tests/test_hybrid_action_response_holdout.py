import unittest

from scripts.evaluate_hybrid_action_response_holdout import compare_action_response_splits


class HybridActionResponseHoldoutTests(unittest.TestCase):
    def test_split_comparison_requires_counts_and_reports_sign_agreement(self):
        fit = {
            "0": {"transition_count": 20, "mean_speed_change_rate_mps2": 8.0},
            "6": {"transition_count": 20, "mean_speed_change_rate_mps2": -10.0},
            "1": {"transition_count": 2, "mean_speed_change_rate_mps2": 4.0},
        }
        heldout = {
            "0": {"transition_count": 10, "mean_speed_change_rate_mps2": 5.0},
            "6": {"transition_count": 10, "mean_speed_change_rate_mps2": -8.0},
            "1": {"transition_count": 20, "mean_speed_change_rate_mps2": -1.0},
        }
        result = compare_action_response_splits(fit, heldout, minimum_transitions=10)

        self.assertEqual(result["compared_actions"], ["0", "6"])
        self.assertEqual(result["sign_agreement_count"], 2)
        self.assertTrue(result["signs_replicate"])
        self.assertFalse(result["causal_fitting_allowed"])

    def test_nonfinite_or_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_action_response_splits({}, {}, minimum_transitions=0)
        with self.assertRaises(ValueError):
            compare_action_response_splits(
                {"0": {"transition_count": 10, "mean_speed_change_rate_mps2": float("nan")}},
                {"0": {"transition_count": 10, "mean_speed_change_rate_mps2": 1.0}},
            )


if __name__ == "__main__":
    unittest.main()
