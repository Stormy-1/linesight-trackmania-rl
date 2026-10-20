import json
import math
import unittest

import numpy as np

from trackmania_rl.hybrid_replay import (
    DisturbanceSpec,
    disturb_sample,
    replay_disturbed_pose,
    select_source_samples,
)
from trackmania_rl.hybrid_rollout import RolloutSample
from trackmania_rl.hybrid_simulator import ClosedLoopConfig
from trackmania_rl.hybrid_state import build_centerline_geometry, make_track_relative_state


class HybridReplayTests(unittest.TestCase):
    def setUp(self):
        self.geometry = build_centerline_geometry(
            np.asarray([[0.0, 0.0, 0.0], [80.0, 0.0, 0.0]], dtype=np.float64)
        )
        self.sample = RolloutSample(
            timestamp_s=2.5,
            position=np.asarray([10.0, 0.0, 0.0]),
            heading=np.asarray([1.0, 0.0, 0.0]),
            speed_mps=16.0,
        )

    def test_disturbance_uses_track_lateral_sign_and_heading_offset(self):
        disturbed = disturb_sample(
            self.sample,
            self.geometry,
            DisturbanceSpec("left", lateral_offset_m=3.0, heading_offset_rad=0.25, speed_delta_mps=-20.0),
            source_index=4,
            section_bins=8,
        )
        self.assertAlmostEqual(disturbed.position[2], -3.0)
        self.assertAlmostEqual(disturbed.speed_mps, 0.0)
        self.assertAlmostEqual(math.atan2(disturbed.heading[2], disturbed.heading[0]), 0.25)
        state = make_track_relative_state(
            disturbed.position, disturbed.heading, disturbed.speed_mps, self.geometry
        )
        self.assertAlmostEqual(state.lateral_error_m, 3.0)
        self.assertEqual(disturbed.source_index, 4)
        self.assertEqual(disturbed.section_bin, 1)

    def test_combined_speed_and_heading_disturbance_is_exact_and_deterministic(self):
        disturbance = DisturbanceSpec(
            "hard_combined",
            lateral_offset_m=10.0,
            heading_offset_rad=math.radians(35.0),
            speed_delta_mps=12.0,
        )
        disturbed = disturb_sample(self.sample, self.geometry, disturbance, source_index=0)
        self.assertAlmostEqual(disturbed.speed_mps, 28.0)
        self.assertAlmostEqual(math.atan2(disturbed.heading[2], disturbed.heading[0]), math.radians(35.0))

        first = replay_disturbed_pose(
            self.geometry,
            disturbed,
            config=ClosedLoopConfig(dt_s=0.1, max_steps=400),
        )
        second = replay_disturbed_pose(
            self.geometry,
            disturbed,
            config=ClosedLoopConfig(dt_s=0.1, max_steps=400),
        )
        self.assertEqual(first.metrics, second.metrics)
        json.dumps(first.metrics)

    def test_stable_rejoin_latency_and_redivergence_are_deterministic(self):
        from trackmania_rl.hybrid_replay import _recovery_stability_metrics

        def record(step, timestamp, recovery):
            return {
                "step_index": step,
                "timestamp_s": timestamp,
                "progress_m": float(step),
                "lateral_error_m": 0.0,
                "heading_error_rad": 0.0,
                "horizontal_distance_to_centerline_m": 0.0,
                "elevation_error_m": 0.0,
                "recovery": recovery,
            }

        records = [
            record(0, 0.0, True),
            record(1, 0.1, False),
            record(2, 0.2, False),
            record(3, 0.3, False),
            record(4, 0.4, True),
            record(5, 0.5, True),
            record(6, 0.6, False),
            record(7, 0.7, False),
            record(8, 0.8, False),
            record(9, 0.9, True),
        ]
        metrics = _recovery_stability_metrics(records)
        self.assertEqual(metrics["stable_rejoin_required_samples"], 3)
        self.assertTrue(metrics["stable_rejoin"])
        self.assertEqual(metrics["stable_rejoin_event"]["step_index"], 3)
        self.assertAlmostEqual(metrics["recovery_latency_s"], 0.3)
        self.assertEqual(metrics["recovery_latency_samples"], 3)
        self.assertEqual(metrics["redivergence_count"], 2)
        self.assertEqual(metrics["first_redivergence_event"]["step_index"], 4)

    def test_recovery_flapping_does_not_count_as_stable_rejoin(self):
        from trackmania_rl.hybrid_replay import _recovery_stability_metrics

        records = [
            {"step_index": 0, "timestamp_s": 0.0, "progress_m": 0.0, "recovery": True},
            {"step_index": 1, "timestamp_s": 0.1, "progress_m": 1.0, "recovery": False},
            {"step_index": 2, "timestamp_s": 0.2, "progress_m": 2.0, "recovery": False},
            {"step_index": 3, "timestamp_s": 0.3, "progress_m": 3.0, "recovery": True},
            {"step_index": 4, "timestamp_s": 0.4, "progress_m": 4.0, "recovery": False},
            {"step_index": 5, "timestamp_s": 0.5, "progress_m": 5.0, "recovery": False},
        ]
        metrics = _recovery_stability_metrics(records)
        self.assertFalse(metrics["stable_rejoin"])
        self.assertIsNone(metrics["stable_rejoin_event"])
        self.assertIsNone(metrics["recovery_latency_s"])
        self.assertIsNone(metrics["recovery_latency_samples"])
        self.assertEqual(metrics["redivergence_count"], 0)

    def test_stability_metrics_are_not_applicable_without_activation_or_rejoin(self):
        from trackmania_rl.hybrid_replay import _recovery_stability_metrics

        nominal = [
            {"step_index": i, "timestamp_s": i * 0.1, "progress_m": float(i), "recovery": False}
            for i in range(4)
        ]
        nominal_metrics = _recovery_stability_metrics(nominal)
        self.assertFalse(nominal_metrics["stable_rejoin"])
        self.assertEqual(nominal_metrics["redivergence_count"], 0)
        self.assertIsNone(nominal_metrics["recovery_latency_s"])

        unfinished = [
            {"step_index": i, "timestamp_s": i * 0.1, "progress_m": float(i), "recovery": True}
            for i in range(4)
        ]
        unfinished_metrics = _recovery_stability_metrics(unfinished)
        self.assertFalse(unfinished_metrics["stable_rejoin"])
        self.assertEqual(unfinished_metrics["redivergence_count"], 0)
        self.assertIsNone(unfinished_metrics["recovery_latency_s"])

    def test_source_selection_is_nearest_and_stable(self):
        samples = [
            self.sample,
            RolloutSample(1.0, np.asarray([31.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), 10.0),
            RolloutSample(3.0, np.asarray([29.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), 10.0),
        ]
        selected = select_source_samples(samples, self.geometry, 30.0, count=2)
        self.assertEqual([index for index, _ in selected], [1, 2])

    def test_replay_is_deterministic_and_reports_recovery_metrics(self):
        pose = disturb_sample(
            self.sample,
            self.geometry,
            DisturbanceSpec("off_center", lateral_offset_m=12.0, heading_offset_rad=0.3),
            source_index=0,
        )
        config = ClosedLoopConfig(dt_s=0.1, max_steps=1200)
        first = replay_disturbed_pose(self.geometry, pose, config=config)
        second = replay_disturbed_pose(self.geometry, pose, config=config)
        self.assertTrue(first.metrics["recovery_activation"])
        self.assertTrue(first.metrics["recovery_completion"])
        self.assertTrue(first.metrics["recovery_success"])
        self.assertEqual(first.metrics["backward_progress_event_count"], 0)
        self.assertTrue(first.metrics["progress_continuity"])
        self.assertEqual(first.metrics["section_metrics"], second.metrics["section_metrics"])
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.metrics["stable_rejoin_required_samples"], 3)
        self.assertIn("stable_rejoin_event", first.metrics)
        self.assertIn("first_redivergence_event", first.metrics)
        self.assertIn("recovery_latency_samples", first.metrics)
        self.assertIn("recovery_latency_s", first.metrics)
        self.assertIn("stable_rejoin", first.metrics["recovery"])
        self.assertIn("redivergence_count", first.metrics["recovery"])
        json.dumps(first.metrics)
        np.testing.assert_allclose(first.result.final_position, second.result.final_position)

    def test_backward_progress_is_accounted_for_and_nominal_recovery_is_not_applicable(self):
        nominal = disturb_sample(self.sample, self.geometry, DisturbanceSpec("nominal"), source_index=0)
        outcome = replay_disturbed_pose(
            self.geometry,
            nominal,
            config=ClosedLoopConfig(dt_s=0.1, max_steps=1),
        )
        self.assertIsNone(outcome.metrics["recovery_success"])
        self.assertFalse(outcome.metrics["stable_rejoin"])
        self.assertEqual(outcome.metrics["redivergence_count"], 0)
        self.assertIsNone(outcome.metrics["recovery_latency_s"])
        self.assertIsNone(outcome.metrics["recovery_latency_samples"])
        self.assertEqual(outcome.metrics["backward_progress_event_count"], 0)

        # A deliberately reversed heading makes the first simulator update
        # travel backwards along this straight centerline.
        backward_sample = RolloutSample(
            0.0, np.asarray([30.0, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]), 8.0
        )
        backward_pose = disturb_sample(
            backward_sample,
            self.geometry,
            DisturbanceSpec("reverse"),
            source_index=1,
        )
        backward = replay_disturbed_pose(
            self.geometry,
            backward_pose,
            config=ClosedLoopConfig(dt_s=0.1, max_steps=1),
        )
        self.assertGreaterEqual(backward.metrics["backward_progress_event_count"], 1)
        self.assertGreater(backward.metrics["backward_progress_distance_m"], 0.0)
        self.assertFalse(backward.metrics["progress_continuity"])

    def test_projection_jitter_below_tolerance_is_not_a_backward_event(self):
        from trackmania_rl.hybrid_replay import _progress_accounting

        self.assertEqual(_progress_accounting([10.0, 9.999999]), (0, 0.0))

    def test_invalid_disturbances_are_rejected(self):
        with self.assertRaises(ValueError):
            DisturbanceSpec("bad", lateral_offset_m=float("nan"))
        with self.assertRaises(ValueError):
            DisturbanceSpec("bad", heading_offset_rad=float("inf"))
        with self.assertRaises(ValueError):
            DisturbanceSpec("bad", speed_delta_mps=-float("inf"))
        with self.assertRaises(ValueError):
            disturb_sample(self.sample, self.geometry, DisturbanceSpec("bad"), source_index=0, section_bins=0)


if __name__ == "__main__":
    unittest.main()
