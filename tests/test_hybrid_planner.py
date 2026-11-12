import math
import unittest

import numpy as np

from trackmania_rl.hybrid_planner import predictive_risk_features
from trackmania_rl.hybrid_simulator import synthetic_centerline_geometry
from trackmania_rl.hybrid_state import make_track_relative_state


class HybridPlannerTests(unittest.TestCase):
    def setUp(self):
        self.geometry = synthetic_centerline_geometry(length_m=120.0, spacing_m=2.0, amplitude_m=5.0)
        self.position = self.geometry.points[0].copy()
        self.tangent = self.geometry.segment_tangents[0].copy()

    def test_features_are_finite_and_match_lookahead_shape(self):
        state = make_track_relative_state(self.position, self.tangent, 18.0, self.geometry)
        features = predictive_risk_features(state, self.geometry)
        self.assertEqual(features.lookahead_distances_m.shape, (4,))
        self.assertEqual(features.lookahead_curvatures.shape, (4,))
        self.assertTrue(np.isfinite(features.lookahead_target_speeds_mps).all())
        self.assertGreaterEqual(features.estimated_braking_distance_m, 0.0)

    def test_heading_alignment_margin_becomes_negative_when_reverse_aligned(self):
        reverse_heading = np.asarray(
            [math.cos(math.radians(-120.0)), 0.0, math.sin(math.radians(-120.0))], dtype=np.float64
        )
        state = make_track_relative_state(self.position, reverse_heading, 18.0, self.geometry)
        features = predictive_risk_features(state, self.geometry)
        self.assertLess(features.one_step_forward_alignment_margin_rad, 0.0)

    def test_rejects_invalid_dynamics(self):
        state = make_track_relative_state(self.position, self.tangent, 18.0, self.geometry)
        with self.assertRaises(ValueError):
            predictive_risk_features(state, self.geometry, max_lateral_acceleration_mps2=0.0)
        with self.assertRaises(ValueError):
            predictive_risk_features(state, self.geometry, lookahead_distances_m=(-1.0,))


if __name__ == "__main__":
    unittest.main()
