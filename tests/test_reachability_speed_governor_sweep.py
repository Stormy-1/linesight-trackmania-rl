import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.sweep_reachability_speed_governor import (
    MAX_GRID_CONFIGURATIONS,
    build_report,
)


class ReachabilitySpeedGovernorSweepTests(unittest.TestCase):
    def _maps_dir(self, directory: str) -> Path:
        maps_dir = Path(directory)
        points = np.asarray(
            [[0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [80.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        for filename in (
            "A01-Race_0.5m_cl2.npy",
            "A07-Race_0.5m_cl2.npy",
            "ESL-Hockolicious_0.5m_cl2.npy",
        ):
            np.save(maps_dir / filename, points)
        return maps_dir

    def test_bounded_grid_reports_each_centerline_and_classifications(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            report = build_report(
                self._maps_dir(directory),
                control_horizons_s=(0.25, 0.75),
                max_lateral_accelerations_mps2=(6.0, 12.0),
                max_steps=100,
            )

        self.assertEqual(report["scope"], "offline_reachability_speed_governor_parameter_sweep")
        self.assertTrue(report["offline_only"])
        self.assertFalse(report["controller_defaults_modified"])
        self.assertEqual(report["parameter_grid"]["configuration_count"], 4)
        self.assertEqual(len(report["configurations"]), 4)
        for configuration in report["configurations"]:
            self.assertIn(configuration["viability"], {"viable", "partial", "non_viable"})
            self.assertIn(configuration["pareto"]["classification"], {"pareto_optimal", "dominated"})
            self.assertEqual(len(configuration["centerlines"]), 3)
            self.assertIn("elapsed_s", configuration["aggregate"])
            self.assertIn("nonmonotonic_progress_steps", configuration["aggregate"]["progress"])
            for centerline in configuration["centerlines"]:
                for field in (
                    "completion",
                    "progress",
                    "elapsed_s",
                    "horizontal_error",
                    "lateral_error",
                    "heading_error",
                    "elevation_error",
                    "recovery",
                    "non_monotonic_progress_steps",
                    "viability",
                ):
                    self.assertIn(field, centerline)

    def test_grid_size_is_bounded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            maps_dir = self._maps_dir(directory)
            with self.assertRaises(ValueError):
                build_report(
                    maps_dir,
                    control_horizons_s=tuple(float(index) for index in range(1, 6)),
                    max_lateral_accelerations_mps2=tuple(float(index) for index in range(1, 5)),
                )
        self.assertEqual(MAX_GRID_CONFIGURATIONS, 16)


if __name__ == "__main__":
    unittest.main()
