import unittest

from scripts.calibrate_hybrid_predictive_features import (
    FeatureObservation,
    _aggregate_case_summaries,
    _summary_for_threshold,
)


class HybridPredictiveFeatureCalibrationTests(unittest.TestCase):
    def test_warning_lead_time_is_measured_before_divergence(self):
        observations = [
            FeatureObservation(i, i * 0.1, float(i), 0.0, 0.0, i >= 4, _features(i < 2))
            for i in range(6)
        ]
        summary = _summary_for_threshold(
            observations,
            feature="speed_margin_to_min_target_mps",
            threshold=0.0,
            horizon_s=0.5,
        )
        self.assertTrue(summary["warning_before_divergence"])
        self.assertTrue(summary["warning_within_horizon"])
        self.assertAlmostEqual(summary["lead_time_s"], 0.4)

    def test_no_warning_is_not_reported_as_predictive(self):
        observations = [FeatureObservation(i, i * 0.1, float(i), 0.0, 0.0, i >= 2, _features(False)) for i in range(4)]
        summary = _summary_for_threshold(
            observations,
            feature="speed_margin_to_min_target_mps",
            threshold=0.0,
            horizon_s=1.0,
        )
        self.assertFalse(summary["warning_before_divergence"])
        self.assertFalse(summary["warning_within_horizon"])
        self.assertIsNone(summary["lead_time_s"])

    def test_aggregate_counts_observed_divergences(self):
        summaries = [
            {"divergence_observed": True, "warning_before_divergence": True, "warning_within_horizon": True, "lead_time_s": 0.5, "warning_precision_before_divergence": 1.0},
            {"divergence_observed": True, "warning_before_divergence": False, "warning_within_horizon": False, "lead_time_s": None, "warning_precision_before_divergence": 0.0},
        ]
        aggregate = _aggregate_case_summaries(summaries)
        self.assertEqual(aggregate["divergence_observed_count"], 2)
        self.assertEqual(aggregate["warning_before_divergence_count"], 1)
        self.assertEqual(aggregate["warning_within_horizon_count"], 1)


def _features(warning: bool):
    class Features:
        speed_margin_to_min_target_mps = -1.0 if warning else 1.0
        lateral_acceleration_margin_mps2 = 1.0
        max_abs_tangent_change_rad = 0.1
        one_step_forward_alignment_margin_rad = 1.0

    return Features()


if __name__ == "__main__":
    unittest.main()
