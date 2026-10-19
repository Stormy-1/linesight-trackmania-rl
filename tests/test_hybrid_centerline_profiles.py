import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.compare_hybrid_centerline_profiles import SELECTED_PROFILE, build_report
from trackmania_rl.hybrid_controller import ConservativeControllerConfig


def _calibration_fixture():
    return {
        "scope": "offline_recorded_speed_and_section_calibration",
        "traces": [
            {
                "sections": [
                    {
                        "actual_speed_mps": {"p25": 60.0, "p50": 70.0, "p75": 80.0},
                        "nominal_target_speed_mps": {"p25": 12.0, "p50": 14.0, "p75": 55.0},
                        "absolute_curvature": {"p50": 0.25},
                    }
                ]
            }
        ],
    }


class HybridCenterlineProfileComparisonTests(unittest.TestCase):
    def test_report_compares_selected_profile_on_each_saved_style_centerline(self):
        with tempfile.TemporaryDirectory() as directory:
            maps_dir = Path(directory)
            np.save(maps_dir / "b.npy", np.asarray([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0]]))
            np.save(maps_dir / "a.npy", np.asarray([[0.0, 0.0, 0.0], [45.0, 0.0, 0.0]]))
            report = build_report(maps_dir, calibration_report=_calibration_fixture(), max_steps=1000)

        self.assertEqual(report["selected_profile"], SELECTED_PROFILE)
        self.assertEqual(report["summary"]["centerline_count"], 2)
        self.assertFalse(report["controller_defaults_modified"])
        self.assertFalse(report["trackmania_launched"])
        self.assertFalse(report["training_started"])
        self.assertFalse(report["checkpoints_loaded"])
        self.assertEqual(report["profiles"]["reference"]["config"], {
            key: getattr(ConservativeControllerConfig(), key)
            for key in ConservativeControllerConfig().__dataclass_fields__
        })
        for item in report["centerlines"]:
            self.assertIn("completion", item["reference"])
            self.assertIn("finish_diagnostic", item["candidate"])
            self.assertIn("horizontal_error", item["candidate"])
            self.assertIn("elevation_error", item["candidate"])
            self.assertIn("lateral_error", item["candidate"])
            self.assertIn("heading_error", item["candidate"])
            self.assertIn("recovery", item["candidate"])
            self.assertIn("progress_fraction", item["comparison_deltas_candidate_minus_reference"]["progress"])
            self.assertIn("strict_3d_finished_delta", item["comparison_deltas_candidate_minus_reference"]["completion"])

    def test_empty_map_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            build_report(Path(directory), calibration_report=_calibration_fixture())


if __name__ == "__main__":
    unittest.main()
