import math
import unittest

from scripts.analyze_hybrid_recovery_authority_gap import (
    _required_lead_steps,
    _required_yaw_rate_for_forward_projection,
)


class HybridRecoveryAuthorityGapTests(unittest.TestCase):
    def test_required_yaw_rate_is_zero_inside_forward_half_plane(self):
        self.assertEqual(_required_yaw_rate_for_forward_projection(math.radians(90.0), dt_s=0.1), 0.0)

    def test_reverse_heading_requires_extra_yaw_authority(self):
        rate = _required_yaw_rate_for_forward_projection(math.radians(120.0), dt_s=0.1)
        self.assertAlmostEqual(rate, (math.radians(120.0) - math.pi / 2.0) / 0.1)
        self.assertGreater(rate, 1.8)

    def test_lead_steps_use_maximum_yaw_rate(self):
        self.assertEqual(_required_lead_steps(math.radians(120.0), dt_s=0.1, max_yaw_rate_rad_s=1.8), 3)


if __name__ == "__main__":
    unittest.main()
