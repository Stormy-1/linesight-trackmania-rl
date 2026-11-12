import unittest
from types import SimpleNamespace

from scripts.evaluate_hybrid_coordinated_recovery import _CoordinatedRecoveryController
from trackmania_rl.hybrid_controller import ControllerCommand


class HybridCoordinatedRecoveryTests(unittest.TestCase):
    def test_forces_steering_direction_and_braking_for_bounded_window(self):
        commands = iter(
            [
                ControllerCommand(-0.2, 0.0, 0.25, 30.0, True),
                ControllerCommand(-0.4, 0.0, 0.25, 30.0, True),
                ControllerCommand(-0.6, 0.0, 0.25, 30.0, True),
            ]
        )
        controller = _CoordinatedRecoveryController(SimpleNamespace(command=lambda state, geometry: next(commands)), 2)
        results = [controller.command(None, None) for _ in range(3)]
        self.assertEqual([result.steering for result in results], [-1.0, -1.0, -0.6])
        self.assertEqual([result.brake for result in results], [1.0, 1.0, 0.25])
        self.assertEqual(controller.coordinated_commands, 2)

    def test_non_recovery_command_is_preserved(self):
        command = ControllerCommand(0.2, 0.4, 0.0, 40.0, False)
        controller = _CoordinatedRecoveryController(SimpleNamespace(command=lambda state, geometry: command), 3)
        self.assertIs(controller.command(None, None), command)
        self.assertEqual(controller.coordinated_commands, 0)

    def test_rejects_negative_duration(self):
        with self.assertRaises(ValueError):
            _CoordinatedRecoveryController(SimpleNamespace(), -1)


if __name__ == "__main__":
    unittest.main()
