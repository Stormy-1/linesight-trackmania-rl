import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.validate_hybrid_centerlines import build_report


class HybridCenterlineValidationTests(unittest.TestCase):
    def test_report_runs_saved_style_centerlines_without_game_access(self):
        with tempfile.TemporaryDirectory() as directory:
            maps_dir = Path(directory)
            np.save(maps_dir / "a.npy", np.asarray([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]]))
            report = build_report(maps_dir, max_steps=1000)
            self.assertFalse(report["trackmania_launched"])
            self.assertFalse(report["real_physics_validated"])
            self.assertEqual(len(report["centerlines"]), 1)
            self.assertTrue(report["centerlines"][0]["finished"])
            self.assertGreater(report["centerlines"][0]["progress_fraction"], 0.9)


if __name__ == "__main__":
    unittest.main()
