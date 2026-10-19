import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trackmania_rl.hybrid_rollout import analyze_rollout, derive_headings_from_positions, load_rollout_csv, parse_rollout_rows
from trackmania_rl.hybrid_state import build_centerline_geometry


def _sample_rows():
    return [
        {
            "timestamp_s": "0.0",
            "position_x_m": "0.0",
            "position_y_m": "0.0",
            "position_z_m": "0.0",
            "heading_x": "1.0",
            "heading_y": "0.0",
            "heading_z": "0.0",
            "speed_mps": "10.0",
        },
        {
            "timestamp_s": "1.0",
            "position_x_m": "10.0",
            "position_y_m": "0.0",
            "position_z_m": "0.0",
            "heading_x": "1.0",
            "heading_y": "0.0",
            "heading_z": "0.0",
            "speed_mps": "10.0",
        },
        {
            "timestamp_s": "2.0",
            "position_x_m": "20.0",
            "position_y_m": "0.0",
            "position_z_m": "9.0",
            "heading_x": "1.0",
            "heading_y": "0.0",
            "heading_z": "0.0",
            "speed_mps": "10.0",
        },
    ]


class HybridRolloutTests(unittest.TestCase):
    def test_parse_and_analyze_rollout(self):
        geometry = build_centerline_geometry(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 10.0]]))
        report = analyze_rollout(parse_rollout_rows(_sample_rows()), geometry)
        self.assertTrue(report["trajectory_data_available"])
        self.assertEqual(report["samples"], 3)
        self.assertAlmostEqual(report["duration_s"], 2.0)
        self.assertGreater(report["progress_gain_m"], 0.0)
        self.assertEqual(report["nonmonotonic_progress_steps"], 0)
        self.assertEqual(len(report["states"]), 3)
        self.assertIn("target_speed_mps", report["states"][1])
        self.assertIn("horizontal_distance_to_centerline_m", report["states"][1])
        self.assertIn("elevation_error_m", report["states"][1])
        self.assertEqual(len(report["section_metrics"]), 10)
        self.assertIn("finish_diagnostic", report)
        finished_report = analyze_rollout(
            parse_rollout_rows(_sample_rows()),
            geometry,
            finish_progress_m=19.0,
        )
        self.assertTrue(finished_report["strict_3d_finish"])

    def test_csv_loader_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_sample_rows()[0].keys())
                writer.writeheader()
                writer.writerows(_sample_rows())
            self.assertEqual(len(load_rollout_csv(path)), 3)

        invalid = list(_sample_rows())
        invalid[1] = dict(invalid[1], timestamp_s="0.0")
        with self.assertRaises(ValueError):
            parse_rollout_rows(invalid)

    def test_rejects_empty_and_invalid_search_radius(self):
        geometry = build_centerline_geometry(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
        with self.assertRaises(ValueError):
            analyze_rollout([], geometry)
        with self.assertRaises(ValueError):
            analyze_rollout(parse_rollout_rows(_sample_rows()), geometry, search_radius=-1)
        with self.assertRaises(ValueError):
            analyze_rollout(parse_rollout_rows(_sample_rows()), geometry, finish_progress_m=0.0)
        with self.assertRaises(ValueError):
            analyze_rollout(parse_rollout_rows(_sample_rows()), geometry, finish_distance_limit_m=0.0)

    def test_can_repair_headings_from_position_differences(self):
        samples = parse_rollout_rows(_sample_rows())
        repaired = derive_headings_from_positions(samples)
        self.assertGreater(repaired[0].heading[0], 0.0)
        self.assertGreater(repaired[1].heading[2], 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(repaired[0].heading)), 1.0)


if __name__ == "__main__":
    unittest.main()
