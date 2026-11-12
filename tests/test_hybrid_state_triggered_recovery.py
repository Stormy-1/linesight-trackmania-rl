import math
import unittest
from types import SimpleNamespace

from scripts.evaluate_hybrid_state_triggered_recovery import _StateTriggeredRecoveryController
from trackmania_rl.hybrid_controller import ControllerCommand


class HybridStateTriggeredRecoveryTests(unittest.TestCase):
    def test_triggers_before_base_recovery_flag(self):
        command = ControllerCommand(0.2, 0.4, 0.0, 40.0, False)
        base = SimpleNamespace(command=lambda state, geometry: command)
        controller = _StateTriggeredRecoveryController(base, 6.0, math.radians(35.0))
        state = SimpleNamespace(lateral_error_m=7.0, heading_error_rad=math.radians(40.0))
        result = controller.command(state, None)
        self.assertEqual(result.steering, 1.0)
        self.assertEqual(result.throttle, 0.0)
        self.assertEqual(result.brake, 1.0)
        self.assertEqual(controller.triggered_before_recovery, 1)
        self.assertEqual(controller.trigger_events, 1)

    def test_risk_condition_is_latched_to_one_bounded_burst(self):
        command = ControllerCommand(0.2, 0.4, 0.0, 40.0, False)
        base = SimpleNamespace(command=lambda state, geometry: command)
        controller = _StateTriggeredRecoveryController(base, 6.0, math.radians(35.0), burst_steps=2)
        state = SimpleNamespace(lateral_error_m=7.0, heading_error_rad=math.radians(40.0))
        results = [controller.command(state, None) for _ in range(5)]
        self.assertEqual(sum(result.brake == 1.0 for result in results), 2)
        self.assertEqual(controller.trigger_events, 1)

    def test_does_not_trigger_when_one_state_dimension_is_below_limit(self):
        command = ControllerCommand(0.2, 0.4, 0.0, 40.0, False)
        base = SimpleNamespace(command=lambda state, geometry: command)
        controller = _StateTriggeredRecoveryController(base, 6.0, math.radians(35.0))
        state = SimpleNamespace(lateral_error_m=5.9, heading_error_rad=math.radians(80.0))
        self.assertIs(controller.command(state, None), command)
        self.assertEqual(controller.triggered_commands, 0)

    def test_rejects_non_positive_limits(self):
        with self.assertRaises(ValueError):
            _StateTriggeredRecoveryController(SimpleNamespace(), 0.0, math.radians(35.0))


if __name__ == "__main__":
    unittest.main()
