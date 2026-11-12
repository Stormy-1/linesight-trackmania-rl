import unittest
from pathlib import Path

from scripts.analyze_hybrid_predictive_features import analyze_centerline


class HybridPredictiveFeaturesTests(unittest.TestCase):
    def test_saved_centerline_analysis_is_json_safe_and_observable(self):
        path = Path("maps/A01-Race_0.5m_cl2.npy")
        report = analyze_centerline(path, max_steps=100)
        self.assertEqual(report["map"], path.name)
        self.assertGreater(report["summary"]["sample_count"], 0)
        self.assertIn("speed_target_excess", report["summary"]["first_observable_risk_steps"])
        self.assertFalse(any("action" in record for record in report["records"]))

    def test_rejects_invalid_simulator_limits(self):
        with self.assertRaises(ValueError):
            analyze_centerline(Path("maps/A01-Race_0.5m_cl2.npy"), max_steps=0)
        with self.assertRaises(ValueError):
            analyze_centerline(Path("maps/A01-Race_0.5m_cl2.npy"), search_radius=-1)


if __name__ == "__main__":
    unittest.main()
