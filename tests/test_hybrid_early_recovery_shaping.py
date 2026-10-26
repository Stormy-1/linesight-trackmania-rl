import unittest
from types import SimpleNamespace

from scripts.evaluate_hybrid_early_recovery_shaping import _EarlyBrakeController
from trackmania_rl.hybrid_controller import ControllerCommand


class HybridEarlyRecoveryShapingTests(unittest.TestCase):
    def test_forces_only_the_first_recovery_commands(self):
        commands = iter(
            [
                ControllerCommand(0.2, 0.0, 0.25, 30.0, True),
                ControllerCommand(0.3, 0.0, 0.25, 30.0, True),
                ControllerCommand(0.4, 0.0, 0.25, 30.0, True),
                ControllerCommand(0.0, 0.4, 0.0, 40.0, False),
            ]
        )
        controller = _EarlyBrakeController(SimpleNamespace(command=lambda state, geometry: next(commands)), 2)
        results = [controller.command(None, None) for _ in range(4)]
        self.assertEqual([result.brake for result in results], [1.0, 1.0, 0.25, 0.0])
        self.assertEqual(controller.forced_commands, 2)

    def test_zero_steps_preserves_base_command(self):
        command = ControllerCommand(0.2, 0.0, 0.25, 30.0, True)
        controller = _EarlyBrakeController(SimpleNamespace(command=lambda state, geometry: command), 0)
        self.assertIs(controller.command(None, None), command)

    def test_rejects_negative_steps(self):
        with self.assertRaises(ValueError):
            _EarlyBrakeController(SimpleNamespace(), -1)


if __name__ == "__main__":
    unittest.main()
