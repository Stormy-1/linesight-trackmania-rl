import unittest

from scripts.evaluate_hybrid_mpc import build_report


class HybridMPCEvaluationTests(unittest.TestCase):
    def test_report_is_offline_deterministic_and_candidate_is_finite(self):
        report = build_report()
        candidate = report["profiles"]["short_horizon_mpc_candidate"]

        self.assertFalse(report["trackmania_launched"])
        self.assertFalse(report["training_started"])
        self.assertFalse(report["real_physics_validated"])
        self.assertFalse(report["controller_defaults_modified"])
        self.assertEqual(report["candidate_status"], "diagnostic_only")
        self.assertTrue(all(item["finished"] for item in candidate))
        self.assertTrue(report["determinism"]["candidate_repeat_matches"])
        self.assertTrue(all(item["mpc"]["all_costs_finite"] for item in candidate))
        self.assertTrue(all(item["mpc"]["candidate_count"] > 0 for item in candidate))
        self.assertTrue(report["comparison"]["mpc_better_than_nominal_tracking"])
        self.assertLess(report["comparison"]["mpc_minus_nominal_mean_abs_lateral_error_m"], 0.0)
        self.assertTrue(report["limitations"])


if __name__ == "__main__":
    unittest.main()
