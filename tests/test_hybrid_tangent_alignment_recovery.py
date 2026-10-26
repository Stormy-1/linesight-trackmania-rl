import math
import unittest
from types import SimpleNamespace

from scripts.evaluate_hybrid_tangent_alignment_recovery import _TangentAlignmentController
from trackmania_rl.hybrid_controller import ControllerCommand


class HybridTangentAlignmentRecoveryTests(unittest.TestCase):
    def test_uses_heading_error_sign_for_bounded_alignment_burst(self):
        command = ControllerCommand(-0.3, 0.0, 0.25, 30.0, True)
        controller = _TangentAlignmentController(SimpleNamespace(command=lambda state, geometry: command), math.radians(90), 2)
        state = SimpleNamespace(heading_error_rad=math.radians(-120.0))
        result = controller.command(state, None)
        self.assertEqual(result.steering, 1.0)
        self.assertEqual(result.brake, 1.0)
        self.assertEqual(controller.alignment_commands, 1)

    def test_does_not_align_below_heading_threshold(self):
        command = ControllerCommand(0.3, 0.0, 0.25, 30.0, True)
        controller = _TangentAlignmentController(SimpleNamespace(command=lambda state, geometry: command), math.radians(105), 2)
        state = SimpleNamespace(heading_error_rad=math.radians(90.0))
        self.assertIs(controller.command(state, None), command)
        self.assertEqual(controller.alignment_commands, 0)

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            _TangentAlignmentController(SimpleNamespace(), 0.0, 1)
        with self.assertRaises(ValueError):
            _TangentAlignmentController(SimpleNamespace(), math.radians(90), -1)


if __name__ == "__main__":
    unittest.main()
