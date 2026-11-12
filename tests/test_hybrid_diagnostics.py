import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.diagnose_hybrid_state import build_report


class HybridDiagnosticsTests(unittest.TestCase):
    def test_report_is_offline_and_marks_missing_trajectory_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            maps = root / "maps"
            maps.mkdir()
            np.save(maps / "demo.npy", np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 5.0]]))
            evaluation = root / "evaluation.csv"
            with evaluation.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["track_name", "dist_pct", "finished"])
                writer.writeheader()
                writer.writerow({"track_name": "demo", "dist_pct": "0.5", "finished": "False"})

            report = build_report(maps, [evaluation])

            self.assertFalse(report["trackmania_launched"])
            self.assertFalse(report["training_started"])
            self.assertEqual(report["schema_version"], 2)
            self.assertTrue(report["centerlines"][0]["finite_geometry"])
            self.assertFalse(report["evaluations"][0]["trajectory_data_available"])
            self.assertEqual(
                report["state_trajectory_validation"]["status"],
                "blocked_by_missing_recorded_pose_telemetry",
            )

    def test_report_can_be_serialized_as_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            maps = root / "maps"
            maps.mkdir()
            np.save(maps / "demo.npy", np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
            report = build_report(maps, [])
            encoded = json.dumps(report)
            self.assertIn('"finite_geometry": true', encoded)


if __name__ == "__main__":
    unittest.main()
