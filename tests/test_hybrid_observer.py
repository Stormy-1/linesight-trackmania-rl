import unittest

from trackmania_rl.hybrid_observer import PredictiveObserverConfig, PredictiveShadowObserver, observer_signals
from trackmania_rl.hybrid_planner import PredictiveRiskFeatures


def _features(speed_margin: float = 1.0, tangent_change: float = 0.1):
    return PredictiveRiskFeatures(
        lookahead_distances_m=[5.0],
        lookahead_curvatures=[0.01],
        lookahead_target_speeds_mps=[20.0],
        max_abs_lookahead_curvature=0.01,
        min_lookahead_target_speed_mps=20.0,
        current_speed_mps=20.0 - speed_margin,
        speed_margin_to_min_target_mps=speed_margin,
        max_abs_tangent_change_rad=tangent_change,
        lateral_acceleration_demand_mps2=2.0,
        lateral_acceleration_margin_mps2=10.0,
        estimated_braking_distance_m=10.0,
        one_step_forward_alignment_margin_rad=1.0,
    )


class HybridObserverTests(unittest.TestCase):
    def test_signals_are_directionally_correct(self):
        config = PredictiveObserverConfig("speed", speed_margin_threshold_mps=0.0, tangent_change_threshold_rad=0.2)
        self.assertEqual(observer_signals(_features(speed_margin=-1.0, tangent_change=0.1), config), ("speed_margin",))
        self.assertEqual(observer_signals(_features(speed_margin=1.0, tangent_change=0.3), config), ("tangent_change",))

    def test_activation_and_release_hysteresis_form_one_event(self):
        observer = PredictiveShadowObserver(
            PredictiveObserverConfig("speed", speed_margin_threshold_mps=0.0, activation_samples=2, release_samples=2)
        )
        for index in range(2):
            observer.observe(index, index * 0.1, _features(speed_margin=-1.0))
        observer.observe(2, 0.2, _features(speed_margin=1.0))
        observer.observe(3, 0.3, _features(speed_margin=1.0))
        events = observer.finish()
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].start_index, events[0].end_index), (0, 1))

    def test_report_contains_no_command_fields(self):
        observer = PredictiveShadowObserver(PredictiveObserverConfig("speed", speed_margin_threshold_mps=0.0))
        observer.observe(0, 0.0, _features(speed_margin=-1.0))
        report = observer.report()
        self.assertNotIn("steering", report)
        self.assertEqual(report["sample_count"], 1)

    def test_rejects_empty_or_invalid_configuration(self):
        with self.assertRaises(ValueError):
            PredictiveObserverConfig("empty")
        with self.assertRaises(ValueError):
            PredictiveObserverConfig("bad", speed_margin_threshold_mps=0.0, activation_samples=0)


if __name__ == "__main__":
    unittest.main()
