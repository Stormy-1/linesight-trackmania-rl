import unittest

from scripts.evaluate_hybrid_local_planning import build_report


class HybridLocalPlanningEvaluationTests(unittest.TestCase):
    def test_report_is_offline_deterministic_and_candidate_completes(self):
        report = build_report()

        self.assertFalse(report["trackmania_launched"])
        self.assertFalse(report["training_started"])
        self.assertFalse(report["real_physics_validated"])
        self.assertFalse(report["controller_defaults_modified"])
        self.assertEqual(report["candidate_status"], "diagnostic_only")
        self.assertEqual(set(report["profiles"]), {"nominal_reference", "planned_candidate"})
        self.assertTrue(all(item["finished"] for item in report["profiles"]["planned_candidate"]))
        self.assertTrue(report["determinism"]["candidate_repeat_matches"])
        self.assertEqual(report["planning_constraints"]["normal_constraint_violations"], 0)
        self.assertEqual(report["planning_constraints"]["fallback_constraint_violations"], 0)
        self.assertTrue(report["limitations"])


if __name__ == "__main__":
    unittest.main()
