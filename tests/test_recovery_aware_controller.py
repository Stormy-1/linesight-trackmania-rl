import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.compare_recovery_aware_controller import build_report
from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineController,
    ConservativeControllerConfig,
    RecoveryAwareCenterlineController,
    RecoveryAwareCenterlineControllerConfig,
)
from trackmania_rl.hybrid_state import build_centerline_geometry, make_track_relative_state


class RecoveryAwareControllerTests(unittest.TestCase):
    def setUp(self):
        self.geometry = build_centerline_geometry(
            np.asarray([[0.0, 0.0, 0.0], [120.0, 0.0, 0.0]], dtype=np.float64)
        )

    def test_nominal_states_match_conservative_defaults(self):
        state = make_track_relative_state(
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            18.0,
            self.geometry,
        )
        default = ConservativeCenterlineController().command(state, self.geometry)
        recovery_aware = RecoveryAwareCenterlineController().command(state, self.geometry)

        self.assertEqual(recovery_aware, default)
        self.assertEqual(ConservativeCenterlineController().config, ConservativeControllerConfig())

    def test_recovery_uses_stronger_bounded_correction_and_braking(self):
        state = make_track_relative_state(
            np.asarray([0.0, 0.0, 12.0]),
            np.asarray([np.cos(0.3), 0.0, np.sin(0.3)]),
            30.0,
            self.geometry,
        )
        config = RecoveryAwareCenterlineControllerConfig()
        default = ConservativeCenterlineController().command(state, self.geometry)
        candidate = RecoveryAwareCenterlineController(config).command(state, self.geometry)

        self.assertTrue(candidate.recovery)
        self.assertLessEqual(abs(candidate.steering), 1.0)
        self.assertGreaterEqual(candidate.brake, config.recovery_brake_bias)
        self.assertEqual(candidate.throttle, 0.0)
        # Both commands may legitimately saturate at the same bound; the
        # recovery path still applies the stronger pre-clamp gains and its
        # conservative brake bias is observable here.
        self.assertGreater(candidate.brake, default.brake)
        self.assertLess(config.recovery_lookahead_m, config.lookahead_m)

    def test_invalid_recovery_config_is_rejected(self):
        with self.assertRaises(ValueError):
            RecoveryAwareCenterlineControllerConfig(recovery_lookahead_m=0.0)
        with self.assertRaises(ValueError):
            RecoveryAwareCenterlineControllerConfig(recovery_brake_bias=1.1)

    def test_comparison_report_covers_synthetic_and_named_saved_centerlines(self):
        points = np.asarray([[0.0, 0.0, 0.0], [80.0, 0.0, 0.0]], dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            maps_dir = Path(directory)
            for filename in (
                "A01-Race_0.5m_cl2.npy",
                "A07-Race_0.5m_cl2.npy",
                "ESL-Hockolicious_0.5m_cl2.npy",
            ):
                np.save(maps_dir / filename, points)
            report = build_report(maps_dir, max_steps=100)

        self.assertEqual(report["summary"]["synthetic_scenario_count"], 2)
        self.assertEqual(report["summary"]["centerline_count"], 3)
        self.assertFalse(report["controller_defaults_modified"])
        self.assertTrue(report["offline_only"])
        for item in report["synthetic_scenarios"] + report["centerlines"]:
            candidate = item["recovery_aware"]
            for field in (
                "completion",
                "progress",
                "lateral_error",
                "horizontal_error",
                "heading_error",
                "elevation_error",
                "recovery",
            ):
                self.assertIn(field, candidate)
            self.assertIn("nonmonotonic_progress_steps", candidate["progress"])
            self.assertIn("non_monotonic_progress_steps", candidate["progress"])


if __name__ == "__main__":
    unittest.main()
