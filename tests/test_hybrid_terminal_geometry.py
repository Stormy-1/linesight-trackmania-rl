import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.diagnose_hybrid_terminal_geometry import build_report
from trackmania_rl.hybrid_simulator import ClosedLoopConfig


def _write_repaired_csv(path: Path, positions: list[tuple[float, float, float]]) -> None:
    fields = (
        "timestamp_s",
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "heading_x",
        "heading_y",
        "heading_z",
        "speed_mps",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (x, y, z) in enumerate(positions):
            writer.writerow(
                {
                    "timestamp_s": index * 0.1,
                    "position_x_m": x,
                    "position_y_m": y,
                    "position_z_m": z,
                    "heading_x": 1.0,
                    "heading_y": 0.0,
                    "heading_z": 0.0,
                    "speed_mps": 10.0,
                }
            )


class HybridTerminalGeometryDiagnosticTests(unittest.TestCase):
    def _fixture_report(self, *, endpoint_y: float = 3.0) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            telemetry_dir = root / "telemetry"
            maps_dir = root / "maps"
            telemetry_dir.mkdir()
            maps_dir.mkdir()
            centerline = np.asarray(
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 3.0, 0.0]], dtype=np.float64
            )
            for label in ("A01", "A07"):
                np.save(maps_dir / f"{label}-Race_0.5m_cl2.npy", centerline)
                _write_repaired_csv(
                    telemetry_dir / f"{label}-Race.repaired.csv",
                    [(0.0, 0.0, 0.0), (20.0, endpoint_y, 0.0)],
                )
            return build_report(telemetry_dir, maps_dir, max_steps=200, search_radius=8)

    def test_json_safe_output_and_safety_flags(self):
        report = self._fixture_report()
        decoded = json.loads(json.dumps(report, allow_nan=False))
        self.assertEqual(decoded["scope"], "offline_hybrid_terminal_geometry_diagnostic")
        safety = decoded["safety"]
        for flag in (
            "offline_only",
            "no_trackmania",
            "no_training",
            "no_checkpoints",
            "no_live_control",
            "no_external_services",
            "unchanged_defaults",
        ):
            self.assertTrue(safety[flag])
        self.assertEqual(len(decoded["maps"]), 2)

    def test_recorded_endpoint_decomposition_and_gate_outcome(self):
        report = self._fixture_report(endpoint_y=6.0)
        item = report["maps"][0]
        self.assertAlmostEqual(item["telemetry_start_projected_progress_m"], 0.0)
        self.assertGreater(item["telemetry_end_projected_progress_m"], 19.0)
        self.assertAlmostEqual(item["final_recorded_pose_errors"]["elevation_m"], 3.0, places=6)
        self.assertTrue(item["default_finish_gate"]["projected_progress_completion"])
        self.assertTrue(item["default_finish_gate"]["strict_finish"])
        self.assertTrue(item["default_finish_gate"]["elevation_tolerant_finish"])

        decomposition = item["simulator"]["terminal_3d_error_decomposition"]
        self.assertAlmostEqual(
            decomposition["reported_3d_error_m"], decomposition["reconstructed_3d_error_m"], places=12
        )
        self.assertGreaterEqual(decomposition["elevation_component_m"], 0.0)

    def test_default_gate_preservation_and_zero_vertical_motion_for_flat_model(self):
        report = self._fixture_report(endpoint_y=0.0)
        self.assertEqual(report["default_closed_loop_config"], {
            "dt_s": 0.1,
            "max_speed_mps": 80.0,
            "max_acceleration_mps2": 10.0,
            "max_braking_mps2": 18.0,
            "linear_drag_mps2_per_mps": 0.05,
            "max_yaw_rate_rad_s": 1.8,
            "max_steps": 3000,
            "search_radius": None,
            "map_match_horizontal_only": True,
            "finish_progress_tolerance_m": 1.0,
            "finish_distance_limit_m": 4.0,
            "finish_elevation_tolerance_m": 8.0,
            "section_bins": 10,
        })
        self.assertEqual(ClosedLoopConfig().finish_distance_limit_m, 4.0)
        self.assertEqual(ClosedLoopConfig().finish_elevation_tolerance_m, 8.0)
        for item in report["maps"]:
            simulator = item["simulator"]
            self.assertTrue(simulator["zero_vertical_motion"])
            self.assertAlmostEqual(simulator["vertical_displacement_m"], 0.0, places=12)
            self.assertTrue(simulator["strict_finish"])


if __name__ == "__main__":
    unittest.main()
