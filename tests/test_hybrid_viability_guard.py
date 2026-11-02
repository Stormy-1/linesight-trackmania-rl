import math
import unittest
from types import SimpleNamespace

from scripts.evaluate_hybrid_viability_guard import (
    _ViabilityGuardController,
    predicted_max_authority_displacement_m,
)
from trackmania_rl.hybrid_controller import ControllerCommand
from trackmania_rl.hybrid_simulator import ClosedLoopConfig


class HybridViabilityGuardTests(unittest.TestCase):
    def test_forward_aligned_state_has_positive_predicted_displacement(self):
        state = SimpleNamespace(speed_mps=20.0, heading_error_rad=0.0)
        self.assertGreater(predicted_max_authority_displacement_m(state, ClosedLoopConfig()), 0.0)

    def test_reverse_aligned_state_has_negative_predicted_displacement(self):
        state = SimpleNamespace(speed_mps=20.0, heading_error_rad=math.radians(120.0))
        self.assertLess(predicted_max_authority_displacement_m(state, ClosedLoopConfig()), 0.0)

    def test_guard_uses_tangent_opposing_steering_only_when_needed(self):
        command = ControllerCommand(-0.2, 0.4, 0.0, 40.0, False)
        controller = _ViabilityGuardController(
            SimpleNamespace(command=lambda state, geometry: command), ClosedLoopConfig()
        )
        safe = SimpleNamespace(speed_mps=20.0, heading_error_rad=0.0)
        dangerous = SimpleNamespace(speed_mps=20.0, heading_error_rad=math.radians(-120.0))
        self.assertEqual(controller.command(safe, None), command)
        result = controller.command(dangerous, None)
        self.assertEqual(result.steering, 1.0)
        self.assertEqual(result.brake, 1.0)
        self.assertEqual(controller.guard_commands, 1)


if __name__ == "__main__":
    unittest.main()
