import unittest

import numpy as np

from scripts.calibrate_hybrid_heading_projection import (
    evaluate_samples,
)
from trackmania_rl.hybrid_rollout import RolloutSample
from trackmania_rl.hybrid_state import build_centerline_geometry


class HybridHeadingProjectionCalibrationTests(unittest.TestCase):
    def test_finite_difference_removes_stationary_fallback_artifact(self):
        geometry = build_centerline_geometry(
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                ]
            )
        )
        samples = [
            RolloutSample(0.0, np.asarray([0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, -1.0]), 0.0),
            RolloutSample(0.1, np.asarray([1.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), 10.0),
            RolloutSample(0.2, np.asarray([2.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), 10.0),
        ]

        recorded = evaluate_samples(
            samples,
            geometry,
            source_name="recorded",
            projection_name="current_3d",
        )
        finite_difference = evaluate_samples(
            samples,
            geometry,
            source_name="finite_difference",
            projection_name="current_3d",
        )

        self.assertEqual(recorded["stationary_sample_count"], 1)
        self.assertAlmostEqual(recorded["first_sample_abs_heading_error_rad"], np.pi / 2.0)
        self.assertLess(finite_difference["first_sample_abs_heading_error_rad"], 1e-9)
        self.assertLess(recorded["first_moving_abs_heading_error_rad"], 1e-9)

    def test_xz_projection_uses_horizontal_metric(self):
        geometry = build_centerline_geometry(
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [10.0, 10.0, 0.0],
                    [20.0, 20.0, 0.0],
                ]
            )
        )
        samples = [
            RolloutSample(0.0, np.asarray([5.0, 20.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), 10.0),
            RolloutSample(0.1, np.asarray([6.0, 20.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), 10.0),
        ]
        report = evaluate_samples(
            samples,
            geometry,
            source_name="recorded",
            projection_name="xz_horizontal",
        )

        self.assertAlmostEqual(report["metric_distance_m"].get("mean"), 0.0)
        self.assertAlmostEqual(report["horizontal_xz_distance_m"].get("mean"), 0.0)


if __name__ == "__main__":
    unittest.main()
