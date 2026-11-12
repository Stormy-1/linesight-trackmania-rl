import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.compare_reachability_speed_governor import build_report
from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineController,
    ConservativeCenterlineSpeedGovernor,
    ConservativeCenterlineSpeedGovernorConfig,
    ConservativeControllerConfig,
    governed_target_speed_mps,
)
from trackmania_rl.hybrid_state import build_centerline_geometry, make_track_relative_state


class ReachabilitySpeedGovernorTests(unittest.TestCase):
    def test_reachable_acceleration_caps_target_without_changing_defaults(self):
        geometry = build_centerline_geometry(
            np.asarray([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]], dtype=np.float64)
        )
        state = make_track_relative_state(
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            18.0,
            geometry,
        )
        default_command = ConservativeCenterlineController().command(state, geometry)
        governed_command = ConservativeCenterlineSpeedGovernor(
            ConservativeCenterlineSpeedGovernorConfig(
                control_horizon_s=0.5,
                reachable_acceleration_mps2=4.0,
                max_lateral_acceleration_mps2=100.0,
            )
        ).command(state, geometry)

        self.assertEqual(default_command.target_speed_mps, 70.0)
        self.assertAlmostEqual(governed_command.target_speed_mps, 20.0)
        self.assertEqual(ConservativeCenterlineController().config, ConservativeControllerConfig())

    def test_curvature_lateral_acceleration_cap_is_an_upper_bound(self):
        governed = governed_target_speed_mps(
            70.0,
            18.0,
            0.25,
            control_horizon_s=10.0,
            reachable_acceleration_mps2=10.0,
            max_lateral_acceleration_mps2=4.0,
        )
        self.assertAlmostEqual(governed, 4.0)

        straight = governed_target_speed_mps(
            70.0,
            18.0,
            0.0,
            control_horizon_s=1.0,
            reachable_acceleration_mps2=10.0,
            max_lateral_acceleration_mps2=4.0,
        )
        self.assertAlmostEqual(straight, 28.0)

    def test_steering_and_recovery_logic_are_inherited(self):
        geometry = build_centerline_geometry(
            np.asarray([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]], dtype=np.float64)
        )
        state = make_track_relative_state(
            np.asarray([0.0, 0.0, 12.0]),
            np.asarray([np.cos(0.3), 0.0, np.sin(0.3)]),
            16.0,
            geometry,
        )
        default_command = ConservativeCenterlineController().command(state, geometry)
        governed_command = ConservativeCenterlineSpeedGovernor().command(state, geometry)

        self.assertEqual(governed_command.steering, default_command.steering)
        self.assertEqual(governed_command.recovery, default_command.recovery)

    def test_invalid_governor_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            ConservativeCenterlineSpeedGovernorConfig(control_horizon_s=-1.0)
        with self.assertRaises(ValueError):
            ConservativeCenterlineSpeedGovernorConfig(reachable_acceleration_mps2=-1.0)
        with self.assertRaises(ValueError):
            ConservativeCenterlineSpeedGovernorConfig(max_lateral_acceleration_mps2=0.0)
        with self.assertRaises(ValueError):
            governed_target_speed_mps(
                70.0,
                18.0,
                0.2,
                control_horizon_s=1.0,
                reachable_acceleration_mps2=10.0,
                max_lateral_acceleration_mps2=0.0,
            )

    def test_comparison_report_covers_saved_centerline_metrics(self):
        points = np.asarray([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]], dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            maps_dir = Path(directory)
            for filename in (
                "A01-Race_0.5m_cl2.npy",
                "A07-Race_0.5m_cl2.npy",
                "ESL-Hockolicious_0.5m_cl2.npy",
            ):
                np.save(maps_dir / filename, points)
            report = build_report(maps_dir, max_steps=100)

        self.assertEqual(report["summary"]["centerline_count"], 3)
        self.assertFalse(report["controller_defaults_modified"])
        self.assertFalse(report["trackmania_launched"])
        for item in report["centerlines"]:
            governed = item["governed"]
            for field in (
                "speed_target_behavior",
                "completion",
                "horizontal_error",
                "lateral_error",
                "heading_error",
                "elevation_error",
                "recovery",
                "progress",
            ):
                self.assertIn(field, governed)
            self.assertIn("nonmonotonic_progress_steps", governed["progress"])


if __name__ == "__main__":
    unittest.main()
