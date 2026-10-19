import unittest

import numpy as np

from trackmania_rl.hybrid_simulator import (
    ClosedLoopConfig,
    simulate_closed_loop,
    synthetic_centerline_geometry,
)
from trackmania_rl.hybrid_state import build_centerline_geometry


class HybridSimulatorTests(unittest.TestCase):
    def test_synthetic_closed_loop_is_deterministic_and_finishes(self):
        geometry = synthetic_centerline_geometry(
            length_m=120.0, spacing_m=2.0, amplitude_m=6.0, wavelength_m=70.0
        )
        config = ClosedLoopConfig(dt_s=0.1, max_steps=1600)
        initial_heading = geometry.segment_tangents[0]
        first = simulate_closed_loop(
            geometry,
            initial_position=geometry.points[0],
            initial_heading=initial_heading,
            initial_speed_mps=18.0,
            config=config,
        )
        second = simulate_closed_loop(
            geometry,
            initial_position=geometry.points[0],
            initial_heading=initial_heading,
            initial_speed_mps=18.0,
            config=config,
        )

        self.assertTrue(first.finished)
        self.assertEqual(first.termination_reason, "finished")
        self.assertGreater(first.progress_gain_m, 100.0)
        self.assertGreater(len(first.steps), 1)
        self.assertEqual(len(first.steps), len(second.steps))
        for first_step, second_step in zip(first.steps, second.steps):
            self.assertEqual(first_step.step_index, second_step.step_index)
            self.assertAlmostEqual(first_step.timestamp_s, second_step.timestamp_s)
            self.assertAlmostEqual(first_step.state.progress_m, second_step.state.progress_m)
            self.assertAlmostEqual(first_step.state.lateral_error_m, second_step.state.lateral_error_m)
            self.assertAlmostEqual(first_step.command.steering, second_step.command.steering)
        np.testing.assert_allclose(first.final_position, second.final_position)
        np.testing.assert_allclose(first.final_heading, second.final_heading)
        self.assertAlmostEqual(first.final_speed_mps, second.final_speed_mps)
        self.assertEqual(first.recovery_samples, 0)
        self.assertEqual(len(first.section_metrics), 10)
        self.assertTrue(first.section_metrics[0]["samples"] > 0)

        target_speeds = [step.command.target_speed_mps for step in first.steps]
        self.assertLess(min(target_speeds), max(target_speeds))
        self.assertTrue(all(-1.0 <= step.command.steering <= 1.0 for step in first.steps))
        self.assertTrue(all(0.0 <= step.command.throttle <= 1.0 for step in first.steps))
        self.assertTrue(all(0.0 <= step.command.brake <= 1.0 for step in first.steps))

    def test_offline_kinematic_update_recovers_lateral_and_heading_error(self):
        geometry = build_centerline_geometry(
            np.asarray([[0.0, 0.0, 0.0], [80.0, 0.0, 0.0]], dtype=np.float64)
        )
        result = simulate_closed_loop(
            geometry,
            initial_position=np.asarray([0.0, 0.0, 12.0]),
            initial_heading=np.asarray([np.cos(0.3), 0.0, np.sin(0.3)]),
            initial_speed_mps=16.0,
            config=ClosedLoopConfig(dt_s=0.1, max_steps=1200),
        )

        self.assertTrue(result.finished)
        self.assertGreater(result.recovery_samples, 0)
        self.assertTrue(result.steps[0].command.recovery)
        self.assertGreater(abs(result.steps[0].state.lateral_error_m), 8.0)
        self.assertGreater(abs(result.steps[0].state.heading_error_rad), 0.25)
        self.assertLess(abs(result.final_state.lateral_error_m), 4.0)
        self.assertLess(abs(result.final_state.heading_error_rad), 0.5)
        self.assertGreater(result.progress_gain_m, 70.0)
        self.assertIsNotNone(result.first_divergence_event)
        self.assertIsNotNone(result.divergence_recovery["first_recovery_activation"])
        self.assertIsNotNone(result.first_recovery_event)

    def test_finish_diagnostics_separate_strict_3d_from_elevation_tolerant_gate(self):
        geometry = build_centerline_geometry(
            np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float64)
        )
        result = simulate_closed_loop(
            geometry,
            initial_position=np.asarray([20.0, 6.0, 0.0]),
            initial_heading=np.asarray([1.0, 0.0, 0.0]),
            initial_speed_mps=0.0,
            config=ClosedLoopConfig(max_steps=2),
        )
        self.assertFalse(result.strict_3d_finished)
        self.assertTrue(result.elevation_tolerant_finished)
        self.assertEqual(result.termination_reason, "elevation_tolerant_finish_only")
        self.assertTrue(result.progress_complete)
        self.assertGreater(result.max_abs_elevation_error_m, 4.0)

    def test_max_step_termination_and_invalid_simulator_config(self):
        geometry = build_centerline_geometry(
            np.asarray([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]], dtype=np.float64)
        )
        result = simulate_closed_loop(
            geometry,
            initial_position=np.asarray([0.0, 0.0, 0.0]),
            initial_heading=np.asarray([1.0, 0.0, 0.0]),
            initial_speed_mps=1.0,
            config=ClosedLoopConfig(max_steps=1),
        )
        self.assertFalse(result.finished)
        self.assertEqual(result.termination_reason, "max_steps")
        self.assertEqual(len(result.steps), 1)

        with self.assertRaises(ValueError):
            ClosedLoopConfig(dt_s=0.0)
        with self.assertRaises(ValueError):
            ClosedLoopConfig(search_radius=-1)
        with self.assertRaises(ValueError):
            simulate_closed_loop(
                geometry,
                initial_position=np.zeros(3),
                initial_heading=np.asarray([1.0, 0.0, 0.0]),
                initial_speed_mps=81.0,
            )


if __name__ == "__main__":
    unittest.main()
