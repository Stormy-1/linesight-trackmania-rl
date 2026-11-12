import unittest

from scripts.diagnose_hybrid_recovery_action_trace import _trace_summary


class _State:
    def __init__(self, progress_m, speed_mps=10.0, lateral_error_m=0.0, heading_error_rad=0.0):
        self.progress_m = progress_m
        self.speed_mps = speed_mps
        self.lateral_error_m = lateral_error_m
        self.heading_error_rad = heading_error_rad
        self.horizontal_distance_to_centerline_m = abs(lateral_error_m)


class _Command:
    def __init__(self, steering, throttle, brake, target_speed_mps, recovery):
        self.steering = steering
        self.throttle = throttle
        self.brake = brake
        self.target_speed_mps = target_speed_mps
        self.recovery = recovery


class _Step:
    def __init__(self, index, progress, command, timestamp=0.1):
        self.step_index = index
        self.timestamp_s = index * timestamp
        self.state = _State(progress)
        self.command = command


class _Result:
    def __init__(self, steps):
        self.steps = steps


class HybridRecoveryActionTraceTests(unittest.TestCase):
    def test_trace_summary_finds_backward_event_and_saturation(self):
        result = _Result(
            [
                _Step(0, 0.0, _Command(0.0, 1.0, 0.0, 20.0, False)),
                _Step(1, 10.0, _Command(1.0, 0.0, 1.0, 12.0, True)),
                _Step(2, 9.0, _Command(0.5, 0.0, 0.8, 12.0, True)),
                _Step(3, 15.0, _Command(0.0, 0.0, 0.0, 20.0, False)),
            ]
        )
        summary = _trace_summary(result, 1)
        self.assertEqual(summary["recovery_sample_count"], 2)
        self.assertEqual(summary["recovery_full_steering_count"], 1)
        self.assertEqual(summary["recovery_full_brake_count"], 1)
        self.assertEqual(summary["first_backward_event"]["step_index"], 2)
        self.assertAlmostEqual(summary["first_backward_event"]["distance_m"], 1.0)


if __name__ == "__main__":
    unittest.main()
