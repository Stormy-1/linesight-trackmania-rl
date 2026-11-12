import unittest

import numpy as np

from trackmania_rl.hybrid_state import (
    STRUCTURED_STATE_SCHEMA_VERSION,
    build_centerline_geometry,
    curvature_target_speed_mps,
    heading_error_rad,
    make_track_relative_state,
    progress_rate_mps,
    project_to_centerline,
    recovery_required,
    sample_centerline_lookahead,
)
from trackmania_rl.hybrid_controller import ConservativeCenterlineController


class HybridStateTests(unittest.TestCase):
    def test_schema_and_lookahead_are_deterministic(self):
        self.assertEqual(STRUCTURED_STATE_SCHEMA_VERSION, 2)
        geometry = build_centerline_geometry(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 10.0]]))
        lookahead = sample_centerline_lookahead(geometry, 5.0, [0.0, 5.0, 100.0])
        np.testing.assert_allclose(lookahead.points[0], [5.0, 0.0, 0.0])
        np.testing.assert_allclose(lookahead.points[-1], [20.0, 0.0, 10.0])
        self.assertEqual(lookahead.curvatures.shape, (3,))

    def test_projection_returns_progress_and_signed_lateral_error(self):
        centerline = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        geometry = build_centerline_geometry(centerline)

        state = make_track_relative_state(
            np.array([5.0, 0.0, 2.0]),
            np.array([1.0, 0.0, 0.0]),
            30.0,
            geometry,
        )

        self.assertEqual(state.segment_index, 0)
        self.assertAlmostEqual(state.progress_m, 5.0)
        self.assertAlmostEqual(state.lateral_error_m, -2.0)
        self.assertAlmostEqual(state.horizontal_distance_to_centerline_m, 2.0)
        self.assertAlmostEqual(state.elevation_error_m, 0.0)
        self.assertAlmostEqual(state.heading_error_rad, 0.0)
        self.assertAlmostEqual(state.local_curvature, 0.0)

    def test_temporal_search_can_restrict_map_matching(self):
        centerline = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        geometry = build_centerline_geometry(centerline)

        segment, fraction, projection, _, _ = project_to_centerline(
            np.array([19.0, 0.0, 0.0]), geometry, previous_segment=1, search_radius=0
        )
        self.assertEqual(segment, 1)
        self.assertAlmostEqual(fraction, 0.9)
        np.testing.assert_allclose(projection, [19.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            project_to_centerline(np.array([5.0, 0.0, 0.0]), geometry, previous_segment=0, search_radius=-1)

    def test_horizontal_projection_ignores_elevation_for_matching_but_state_keeps_3d_distance(self):
        centerline = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0], [20.0, 20.0, 0.0]])
        geometry = build_centerline_geometry(centerline)
        state = make_track_relative_state(
            np.array([5.0, 20.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            10.0,
            geometry,
            horizontal_only=True,
        )
        self.assertAlmostEqual(state.progress_m, 0.5 * np.sqrt(200.0))
        self.assertAlmostEqual(state.horizontal_distance_to_centerline_m, 0.0)
        self.assertAlmostEqual(state.elevation_error_m, 15.0)
        self.assertAlmostEqual(state.distance_to_centerline_m, 15.0)

    def test_horizontal_tracking_and_elevation_are_reported_separately(self):
        geometry = build_centerline_geometry(np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
        state = make_track_relative_state(
            np.array([5.0, 3.0, 2.0]),
            np.array([1.0, 0.0, 0.0]),
            10.0,
            geometry,
        )
        self.assertAlmostEqual(state.horizontal_distance_to_centerline_m, 2.0)
        self.assertAlmostEqual(state.elevation_error_m, 3.0)
        self.assertAlmostEqual(state.distance_to_centerline_m, np.sqrt(13.0))

    def test_heading_error_is_signed_and_wrapped(self):
        self.assertAlmostEqual(heading_error_rad(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])), np.pi / 2)
        self.assertAlmostEqual(heading_error_rad(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])), 0.0)

    def test_invalid_centerline_and_pose_are_rejected(self):
        with self.assertRaises(ValueError):
            build_centerline_geometry(np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            build_centerline_geometry(np.zeros((2, 3)))
        geometry = build_centerline_geometry(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        with self.assertRaises(ValueError):
            make_track_relative_state(np.array([0.0, 0.0, 0.0]), np.zeros(3), 1.0, geometry)

    def test_progress_speed_recovery_and_target_speed_boundaries(self):
        self.assertAlmostEqual(progress_rate_mps(10.0, 4.0, 2.0), 3.0)
        self.assertEqual(progress_rate_mps(4.0, 10.0, 2.0), 0.0)
        self.assertAlmostEqual(curvature_target_speed_mps(0.0, max_speed_mps=50.0, min_speed_mps=10.0), 50.0)
        self.assertGreater(curvature_target_speed_mps(0.2), 0.0)
        geometry = build_centerline_geometry(np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
        state = make_track_relative_state(np.array([5.0, 0.0, 9.0]), np.array([1.0, 0.0, 0.0]), 20.0, geometry)
        self.assertTrue(recovery_required(state))

    def test_conservative_controller_emits_bounded_action(self):
        geometry = build_centerline_geometry(np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [40.0, 0.0, 10.0]]))
        state = make_track_relative_state(np.array([5.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 15.0, geometry)
        command = ConservativeCenterlineController().command(state, geometry)
        self.assertGreaterEqual(command.steering, -1.0)
        self.assertLessEqual(command.steering, 1.0)
        self.assertGreaterEqual(command.throttle, 0.0)
        self.assertLessEqual(command.throttle, 1.0)
        self.assertGreaterEqual(command.brake, 0.0)
        self.assertLessEqual(command.brake, 1.0)
        self.assertGreater(command.target_speed_mps, 0.0)

    def test_controller_uses_vehicle_heading_error_in_correct_direction(self):
        geometry = build_centerline_geometry(np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]]))
        state = make_track_relative_state(np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 15.0, geometry)
        command = ConservativeCenterlineController().command(state, geometry)
        self.assertLess(command.steering, 0.0)


if __name__ == "__main__":
    unittest.main()
