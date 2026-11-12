import unittest

from scripts.validate_hybrid_controller import build_report


class HybridControllerValidationTests(unittest.TestCase):
    def test_report_runs_both_offline_scenarios_and_states_limitations(self):
        report = build_report()
        self.assertFalse(report["trackmania_launched"])
        self.assertFalse(report["training_started"])
        self.assertFalse(report["real_physics_validated"])
        self.assertEqual(len(report["scenarios"]), 2)
        self.assertTrue(all(item["finished"] for item in report["scenarios"]))
        self.assertGreater(report["scenarios"][1]["recovery_samples"], 0)
        self.assertTrue(report["limitations"])


if __name__ == "__main__":
    unittest.main()
