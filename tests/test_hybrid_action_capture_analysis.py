import unittest

from scripts.analyze_hybrid_action_transition_capture import build_report


class HybridActionCaptureAnalysisTests(unittest.TestCase):
    def test_measured_capture_is_aligned_and_not_promoted(self):
        report = build_report()
        summary = report["capture_summary"]

        self.assertTrue(report["real_capture_verified"])
        self.assertFalse(report["real_physics_validated"])
        self.assertFalse(report["promotion_allowed"])
        self.assertEqual(summary["run_count"], 3)
        self.assertEqual(summary["finish_count"], 0)
        self.assertTrue(summary["all_action_counts_aligned"])
        self.assertTrue(summary["all_transition_metrics_finite"])
        self.assertEqual([run["sample_count"] for run in report["runs"]], [1168, 535, 906])
        self.assertTrue(report["limitations"])


if __name__ == "__main__":
    unittest.main()
