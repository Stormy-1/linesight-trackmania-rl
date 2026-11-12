import math
import unittest
from types import SimpleNamespace

from scripts.evaluate_hybrid_predictive_prerotation import _OraclePreRotationController
from trackmania_rl.hybrid_controller import ControllerCommand


class HybridPredictivePrerotationTests(unittest.TestCase):
    def test_steers_against_known_future_heading_offset_only_before_injection(self):
        command = ControllerCommand(-0.2, 0.4, 0.0, 40.0, False)
        controller = _OraclePreRotationController(
            SimpleNamespace(command=lambda state, geometry: command),
            injection_step=3,
            heading_offset_rad=math.radians(-120.0),
            lead_steps=2,
            steering_level=0.5,
            brake_level=1.0,
        )
        state = SimpleNamespace()
        results = [controller.command(state, None) for _ in range(5)]
        self.assertEqual(results[0], command)
        self.assertEqual(results[1].steering, 0.5)
        self.assertEqual(results[2].steering, 0.5)
        self.assertEqual(results[3], command)
        self.assertEqual(controller.pre_rotation_commands, 2)

    def test_zero_lead_time_does_not_override_commands(self):
        command = ControllerCommand(0.2, 0.4, 0.0, 40.0, False)
        controller = _OraclePreRotationController(
            SimpleNamespace(command=lambda state, geometry: command),
            injection_step=2,
            heading_offset_rad=math.radians(-120.0),
            lead_steps=0,
            steering_level=1.0,
            brake_level=1.0,
        )
        self.assertEqual(controller.command(SimpleNamespace(), None), command)
        self.assertEqual(controller.pre_rotation_commands, 0)

    def test_rejects_invalid_oracle_configuration(self):
        with self.assertRaises(ValueError):
            _OraclePreRotationController(
                SimpleNamespace(),
                injection_step=1,
                heading_offset_rad=math.radians(-120.0),
                lead_steps=1,
                steering_level=1.2,
                brake_level=0.0,
            )
        with self.assertRaises(ValueError):
            _OraclePreRotationController(
                SimpleNamespace(),
                injection_step=1,
                heading_offset_rad=0.0,
                lead_steps=1,
                steering_level=1.0,
                brake_level=0.0,
            )


if __name__ == "__main__":
    unittest.main()
