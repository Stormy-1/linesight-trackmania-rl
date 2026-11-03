import unittest

import numpy as np

from trackmania_rl.hybrid_controller import (
    PlannedCenterlineController,
    PlannedCenterlineControllerConfig,
)
from trackmania_rl.hybrid_planner import LocalTrajectoryPlannerConfig, plan_local_trajectory
from trackmania_rl.hybrid_simulator import synthetic_centerline_geometry
from trackmania_rl.hybrid_state import build_centerline_geometry, make_track_relative_state


class HybridPlanningContractTests(unittest.TestCase):
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
            24.0,
            self.geometry,
        )

    def test_plan_is_deterministic_bounded_and_constraint_safe(self):
        config = LocalTrajectoryPlannerConfig(
            horizon_distances_m=(0.0, 5.0, 10.0, 20.0, 35.0, 50.0),
            max_speed_mps=70.0,
            min_speed_mps=12.0,
            max_acceleration_mps2=10.0,
            max_braking_mps2=18.0,
            fallback_speed_scale=0.75,
        )
        first = plan_local_trajectory(self.state, self.geometry, config)
        second = plan_local_trajectory(self.state, self.geometry, config)

        self.assertEqual(first.distances_m.shape, (6,))
        self.assertEqual(first.points.shape, (6, 3))
        self.assertEqual(first.tangents.shape, (6, 3))
        self.assertEqual(first.curvatures.shape, (6,))
        self.assertEqual(first.target_speeds_mps.shape, (6,))
        self.assertTrue(np.isfinite(first.points).all())
        self.assertTrue(np.isfinite(first.target_speeds_mps).all())
        self.assertTrue(np.all(first.target_speeds_mps >= 0.0))
        self.assertTrue(np.all(first.target_speeds_mps <= config.max_speed_mps))
        self.assertTrue(np.all(first.fallback_target_speeds_mps <= first.target_speeds_mps + 1e-9))
        np.testing.assert_allclose(first.target_speeds_mps, second.target_speeds_mps)
        np.testing.assert_allclose(first.points, second.points)

        for previous, current, distance in zip(
            first.target_speeds_mps[:-1],
            first.target_speeds_mps[1:],
            np.diff(first.distances_m),
        ):
            self.assertLessEqual(
                current**2,
                previous**2 + 2.0 * config.max_acceleration_mps2 * distance + 1e-8,
            )
        for earlier, later, distance in zip(
            first.target_speeds_mps[:-1],
            first.target_speeds_mps[1:],
            np.diff(first.distances_m),
        ):
            self.assertLessEqual(
                earlier**2,
                later**2 + 2.0 * config.max_braking_mps2 * distance + 1e-8,
            )

    def test_plan_clips_at_track_end_and_fallback_stays_centered(self):
        config = LocalTrajectoryPlannerConfig(horizon_distances_m=(0.0, 30.0, 100.0))
        state = make_track_relative_state(
            self.geometry.points[-2],
            self.geometry.segment_tangents[-1],
            10.0,
            self.geometry,
        )
        plan = plan_local_trajectory(state, self.geometry, config)

        self.assertTrue(np.all(plan.distances_m >= 0.0))
        np.testing.assert_allclose(plan.points[-1], self.geometry.points[-1])
        np.testing.assert_allclose(plan.desired_lateral_offsets_m, 0.0)
        np.testing.assert_allclose(plan.fallback_lateral_offsets_m, 0.0)

    def test_invalid_planner_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            LocalTrajectoryPlannerConfig(horizon_distances_m=(5.0, 2.0))
        with self.assertRaises(ValueError):
            LocalTrajectoryPlannerConfig(max_braking_mps2=0.0)
        with self.assertRaises(ValueError):
            LocalTrajectoryPlannerConfig(fallback_speed_scale=1.1)


class PlannedControllerTests(unittest.TestCase):
    def test_plan_consuming_controller_emits_bounded_command(self):
        geometry = build_centerline_geometry(
            np.asarray([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [60.0, 0.0, 15.0]], dtype=np.float64)
        )
        state = make_track_relative_state(
            np.asarray([5.0, 0.0, 1.0]),
            np.asarray([1.0, 0.0, 0.0]),
            18.0,
            geometry,
        )
        controller = PlannedCenterlineController(
            PlannedCenterlineControllerConfig(
                planner_config=LocalTrajectoryPlannerConfig(horizon_distances_m=(0.0, 8.0, 16.0, 30.0))
            )
        )
        command = controller.command(state, geometry)

        self.assertTrue(-1.0 <= command.steering <= 1.0)
        self.assertTrue(0.0 <= command.throttle <= 1.0)
        self.assertTrue(0.0 <= command.brake <= 1.0)
        self.assertGreaterEqual(command.target_speed_mps, 0.0)


if __name__ == "__main__":
    unittest.main()
