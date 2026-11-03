import tempfile
import unittest
from pathlib import Path

from scripts.check_hybrid_real_validation_readiness import build_readiness_report


class HybridRealValidationReadinessTests(unittest.TestCase):
    def test_workspace_preflight_is_offline_and_requires_authorization(self):
        report = build_readiness_report(Path("."))
        self.assertTrue(report["offline_preflight_complete"])
        self.assertTrue(report["real_validation_authorization_required"])
        self.assertFalse(report["trackmania_launched"])
        self.assertFalse(report["promotion_allowed"])

    def test_missing_root_fails_preflight_without_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            report = build_readiness_report(Path(directory))
        self.assertFalse(report["offline_preflight_complete"])
        self.assertEqual(report["reports_checked"], 0)


if __name__ == "__main__":
    unittest.main()
