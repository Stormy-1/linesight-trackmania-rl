import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.diagnose_hybrid_finish_gate import (
    FINISH_GATE_MATRIX,
    PROFILE_NAME,
    build_report,
    synthetic_straight_centerline,
)
from trackmania_rl.hybrid_simulator import ClosedLoopConfig


class HybridFinishGateDiagnosticTests(unittest.TestCase):
    def test_temporary_npy_centerlines_are_reported_and_defaults_remain_unchanged(self):
        defaults_before = ClosedLoopConfig()
        with tempfile.TemporaryDirectory() as directory:
            maps_dir = Path(directory)
            np.save(
                maps_dir / "A01-Race_0.5m_cl2.npy",
                np.asarray([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=np.float64),
            )
            np.save(
                maps_dir / "A07-Race_0.5m_cl2.npy",
                np.asarray([[0.0, 0.0, 0.0], [55.0, 0.0, 0.0]], dtype=np.float64),
            )
            report = build_report(maps_dir, max_steps=400, search_radius=8)

        self.assertEqual(ClosedLoopConfig(), defaults_before)
        self.assertFalse(report["controller_defaults_modified"])
        self.assertEqual(report["default_closed_loop_config"]["finish_distance_limit_m"], 4.0)
        self.assertEqual(report["default_closed_loop_config"]["finish_elevation_tolerance_m"], 8.0)
        self.assertEqual({item["map"] for item in report["maps"]}, {"A01", "A07", "synthetic_straight"})
        self.assertEqual(len(report["results"]), 3 * len(FINISH_GATE_MATRIX))
        self.assertEqual({row["profile"] for row in report["results"]}, {PROFILE_NAME})

    def test_projected_progress_can_complete_without_strict_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            maps_dir = Path(directory)
            np.save(
                maps_dir / "fixture.npy",
                synthetic_straight_centerline(length_m=30.0, spacing_m=2.0),
            )
            report = build_report(maps_dir, max_steps=200, search_radius=8)

        synthetic_rows = [row for row in report["results"] if row["map"] == "synthetic_straight"]
        default_row = next(row for row in synthetic_rows if row["gate"]["is_default"])
        self.assertTrue(default_row["projected_progress_completion"])
        self.assertFalse(default_row["strict_finish"])
        self.assertTrue(default_row["elevation_tolerant_finish"])
        self.assertEqual(default_row["termination_reason"], "elevation_tolerant_finish_only")
        self.assertGreater(default_row["final_3d_error_m"], default_row["gate"]["finish_distance_limit_m"])
        self.assertLessEqual(
            default_row["final_horizontal_error_m"], default_row["gate"]["finish_distance_limit_m"]
        )
        self.assertGreater(default_row["final_abs_elevation_error_m"], 0.0)

    def test_report_is_json_safe_and_contains_pass_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            maps_dir = Path(directory)
            np.save(maps_dir / "fixture.npy", np.asarray([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]]))
            report = build_report(maps_dir, max_steps=300, search_radius=4)

        encoded = json.dumps(report, allow_nan=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["scope"], "offline_hybrid_finish_gate_calibration_diagnostic")
        self.assertTrue(decoded["safety"]["offline_only"])
        self.assertFalse(decoded["safety"]["trackmania_launched"])
        self.assertFalse(decoded["safety"]["training_started"])
        self.assertFalse(decoded["safety"]["checkpoints_loaded"])
        self.assertFalse(decoded["safety"]["live_control_used"])
        self.assertTrue(decoded["finish_gate_pass_rates"])
        self.assertTrue(all("strict_finish_rate" in item for item in decoded["finish_gate_pass_rates"]))


if __name__ == "__main__":
    unittest.main()
