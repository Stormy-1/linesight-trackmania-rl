import json
import tempfile
import unittest
from pathlib import Path

from scripts.calibrate_hybrid_speed_profile import build_report


class HybridSpeedCalibrationTests(unittest.TestCase):
    def test_build_report_is_descriptive_and_does_not_modify_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            states = [
                {
                    "progress_m": 0.0,
                    "speed_mps": 40.0,
                    "target_speed_mps": 30.0,
                    "local_curvature": 0.01,
                    "lateral_error_m": 1.0,
                    "heading_error_rad": 0.1,
                    "elevation_error_m": 0.2,
                    "recovery": False,
                },
                {
                    "progress_m": 10.0,
                    "speed_mps": 20.0,
                    "target_speed_mps": 30.0,
                    "local_curvature": 0.05,
                    "lateral_error_m": 2.0,
                    "heading_error_rad": 0.2,
                    "elevation_error_m": 0.3,
                    "recovery": True,
                },
            ]
            path.write_text(json.dumps({"duration_s": 1.0, "progress_gain_m": 10.0, "end_progress_m": 10.0, "states": states}), encoding="utf-8")
            report = build_report([path], bins=2)
            self.assertFalse(report["controller_modified"])
            self.assertEqual(len(report["traces"][0]["sections"]), 2)
            self.assertAlmostEqual(report["traces"][0]["mean_actual_minus_nominal_mps"], 0.0)
            self.assertEqual(report["traces"][0]["sections"][1]["recovery_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
