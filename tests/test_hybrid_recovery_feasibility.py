import math
import unittest

from scripts.analyze_hybrid_recovery_feasibility import maximum_authority_one_step


class HybridRecoveryFeasibilityTests(unittest.TestCase):
    def test_reverse_alignment_can_be_unavoidable_for_one_bounded_step(self):
        result = maximum_authority_one_step(speed_mps=25.0, heading_error_rad=math.radians(120.0))
        self.assertTrue(result["reverse_unavoidable_proxy"])
        self.assertLess(result["estimated_along_tangent_displacement_m"], 0.0)

    def test_forward_alignment_is_not_marked_unavoidable(self):
        result = maximum_authority_one_step(speed_mps=12.0, heading_error_rad=0.0)
        self.assertFalse(result["reverse_unavoidable_proxy"])
        self.assertGreaterEqual(result["estimated_along_tangent_displacement_m"], 0.0)

    def test_rejects_negative_speed(self):
        with self.assertRaises(ValueError):
            maximum_authority_one_step(speed_mps=-1.0, heading_error_rad=0.0)


if __name__ == "__main__":
    unittest.main()
