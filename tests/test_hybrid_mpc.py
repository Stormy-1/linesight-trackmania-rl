import unittest

import numpy as np

from trackmania_rl.hybrid_mpc import (
    ShortHorizonMPCConfig,
    ShortHorizonMPCController,
)
from trackmania_rl.hybrid_simulator import synthetic_centerline_geometry
from trackmania_rl.hybrid_state import make_track_relative_state


class HybridMPCTests(unittest.TestCase):
    def setUp(self):
        self.geometry = synthetic_centerline_geometry(
            length_m=120.0,
            spacing_m=2.0,
            amplitude_m=6.0,
            wavelength_m=70.0,
        )
        self.state = make_track_relative_state(
            self.geometry.points[0],
            self.geometry.segment_tangents[0],
            18.0,
            self.geometry,
        )

    def test_mpc_decision_is_deterministic_and_bounded(self):
        controller = ShortHorizonMPCController(
            ShortHorizonMPCConfig(horizon_steps=5, steering_candidates=(-0.5, 0.0, 0.5))
        )
        first = controller.decision(self.state, self.geometry)
        second = controller.decision(self.state, self.geometry)

        self.assertEqual(first, second)
        self.assertEqual(first.candidate_count, 15)
        self.assertTrue(np.isfinite(first.cost))
        self.assertTrue(-1.0 <= first.command.steering <= 1.0)
        self.assertTrue(0.0 <= first.command.throttle <= 1.0)
        self.assertTrue(0.0 <= first.command.brake <= 1.0)
        self.assertGreaterEqual(first.command.target_speed_mps, 0.0)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            ShortHorizonMPCConfig(horizon_steps=0)
        with self.assertRaises(ValueError):
            ShortHorizonMPCConfig(dt_s=0.0)
        with self.assertRaises(ValueError):
            ShortHorizonMPCConfig(steering_candidates=(2.0,))
        with self.assertRaises(ValueError):
            ShortHorizonMPCConfig(longitudinal_candidates=((1.0, 1.0),))
        with self.assertRaises(ValueError):
            ShortHorizonMPCConfig(lateral_error_weight=0.0, heading_error_weight=0.0, speed_error_weight=0.0)

    def test_prediction_trace_is_finite_and_bounded(self):
        controller = ShortHorizonMPCController(ShortHorizonMPCConfig(horizon_steps=4))
        decision = controller.decision(self.state, self.geometry)

        self.assertEqual(len(decision.predicted_states), 4)
        for state in decision.predicted_states:
            self.assertTrue(np.isfinite(state.progress_m))
            self.assertTrue(np.isfinite(state.lateral_error_m))
            self.assertTrue(np.isfinite(state.heading_error_rad))
            self.assertGreaterEqual(state.speed_mps, 0.0)
            self.assertLessEqual(state.speed_mps, controller.config.max_speed_mps)


if __name__ == "__main__":
    unittest.main()
