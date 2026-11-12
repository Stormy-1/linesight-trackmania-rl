import math
import unittest
from types import SimpleNamespace

from scripts.evaluate_hybrid_longitudinal_recovery_intervention import (
    INTERVENTIONS,
    _delta_summary,
    _LongitudinalInterventionController,
)
from trackmania_rl.hybrid_controller import ControllerCommand


class HybridLongitudinalRecoveryInterventionTests(unittest.TestCase):
    def test_heading_trigger_replaces_only_longitudinal_action(self):
        command = ControllerCommand(steering=0.8, throttle=0.0, brake=0.25, target_speed_mps=50.0, recovery=True)
        base = SimpleNamespace(command=lambda state, geometry: command)
        controller = _LongitudinalInterventionController(base, "heading_90_full_brake")
        state = SimpleNamespace(heading_error_rad=math.radians(100.0), speed_mps=20.0)
        result = controller.command(state, None)
        self.assertEqual(result.steering, 0.8)
        self.assertEqual(result.throttle, 0.0)
        self.assertEqual(result.brake, 1.0)
        self.assertEqual(result.target_speed_mps, 50.0)

    def test_along_tangent_trigger_requires_recovery(self):
        command = ControllerCommand(steering=-0.4, throttle=0.3, brake=0.0, target_speed_mps=40.0, recovery=False)
        base = SimpleNamespace(command=lambda state, geometry: command)
        controller = _LongitudinalInterventionController(base, "along_tangent_0_full_brake")
        state = SimpleNamespace(heading_error_rad=math.pi, speed_mps=20.0)
        result = controller.command(state, None)
        self.assertEqual(result.brake, 0.0)
        self.assertEqual(result.throttle, 0.3)

    def test_intervention_registry_is_diagnostic_family(self):
        self.assertIn("baseline", INTERVENTIONS)
        self.assertEqual(INTERVENTIONS["heading_105_full_brake"]["kind"], "heading")

    def test_delta_summary_counts_tradeoffs(self):
        summary = _delta_summary(
            [
                {"continuity_delta": 1, "stable_rejoin_delta": 0, "backward_distance_delta_m": -0.5, "recovery_latency_delta_s": 0.1},
                {"continuity_delta": -1, "stable_rejoin_delta": -1, "backward_distance_delta_m": 0.2, "recovery_latency_delta_s": None},
            ]
        )
        self.assertEqual(summary["continuity_improved_count"], 1)
        self.assertEqual(summary["continuity_worsened_count"], 1)
        self.assertEqual(summary["stable_rejoin_worsened_count"], 1)
        self.assertAlmostEqual(summary["mean_backward_distance_delta_m"], -0.15)


if __name__ == "__main__":
    unittest.main()
