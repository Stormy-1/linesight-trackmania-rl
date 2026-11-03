import unittest

import numpy as np

from trackmania_rl.hybrid_action_dynamics import action_transition_metrics
from trackmania_rl.hybrid_action_telemetry import ActionTelemetrySample
from trackmania_rl.hybrid_rollout import RolloutSample


def _sample(index: int, action_idx: int, *, speed: float) -> ActionTelemetrySample:
    return ActionTelemetrySample(
        telemetry=RolloutSample(
            timestamp_s=index * 0.1,
            position=np.asarray([index * speed * 0.1, 0.0, 0.0], dtype=np.float64),
            heading=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            speed_mps=speed,
        ),
        action_idx=action_idx,
        action_accelerate=action_idx == 0,
        action_brake=action_idx == 6,
        action_left=False,
        action_right=False,
        action_was_greedy=True,
    )


class HybridActionDynamicsTests(unittest.TestCase):
    def test_metrics_align_each_transition_with_preceding_action(self):
        samples = [_sample(0, 0, speed=10.0), _sample(1, 0, speed=11.0), _sample(2, 6, speed=9.0)]
        report = action_transition_metrics(samples)

        self.assertEqual(report["transition_count"], 2)
        self.assertEqual(report["action_counts"]["0"], 2)
        self.assertEqual(report["action_counts"]["6"], 1)
        self.assertEqual(report["by_action"]["0"]["transition_count"], 2)
        self.assertAlmostEqual(report["by_action"]["0"]["mean_speed_change_rate_mps2"], -5.0)
        self.assertTrue(report["finite_metrics"])

    def test_metrics_are_deterministic_and_reject_short_input(self):
        samples = [_sample(0, 0, speed=10.0), _sample(1, 6, speed=9.0)]
        self.assertEqual(action_transition_metrics(samples), action_transition_metrics(samples))
        with self.assertRaises(ValueError):
            action_transition_metrics(samples[:1])


if __name__ == "__main__":
    unittest.main()
