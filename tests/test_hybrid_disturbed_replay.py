import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.compare_hybrid_disturbed_replay import (
    PROFILE_NAMES,
    build_report,
    select_source_anchors,
)
from scripts.evaluate_hybrid_speed_candidates import derive_candidates


def _calibration_fixture():
    return {
        "scope": "offline_recorded_speed_and_section_calibration",
        "traces": [
            {
                "sections": [
                    {
                        "actual_speed_mps": {"p25": 20.0, "p50": 25.0, "p75": 30.0},
                        "nominal_target_speed_mps": {"p25": 12.0, "p50": 13.0, "p75": 15.0},
                        "absolute_curvature": {"p50": 0.15},
                    },
                    {
                        "actual_speed_mps": {"p25": 24.0, "p50": 28.0, "p75": 34.0},
                        "nominal_target_speed_mps": {"p25": 12.0, "p50": 14.0, "p75": 16.0},
                        "absolute_curvature": {"p50": 0.20},
                    },
                ]
            }
        ],
    }


def _write_telemetry(path: Path, length: float = 120.0) -> None:
    fields = [
        "timestamp_s",
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "heading_x",
        "heading_y",
        "heading_z",
        "speed_mps",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, x in enumerate(np.arange(0.0, length + 0.1, 10.0)):
            writer.writerow(
                {
                    "timestamp_s": index * 0.5,
                    "position_x_m": x,
                    "position_y_m": 0.0,
                    "position_z_m": 0.0,
                    "heading_x": 1.0,
                    "heading_y": 0.0,
                    "heading_z": 0.0,
                    "speed_mps": 12.0,
                }
            )


class HybridDisturbedReplayReportTests(unittest.TestCase):
    def test_report_emits_three_profiles_metrics_and_source_sections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            telemetry_dir = root / "telemetry"
            maps_dir = root / "maps"
            telemetry_dir.mkdir()
            maps_dir.mkdir()
            for map_name in ("A01-Race", "A07-Race"):
                _write_telemetry(telemetry_dir / f"{map_name}.repaired.csv")
                np.save(maps_dir / f"{map_name}_0.5m_cl2.npy", np.column_stack((np.arange(0.0, 121.0, 1.0), np.zeros(121), np.zeros(121))))

            report = build_report(
                telemetry_dir,
                maps_dir,
                calibration_report=_calibration_fixture(),
                max_steps=8,
                search_radius=8,
                anchor_progress_m={"A01-Race": (20.0, 80.0), "A07-Race": (20.0, 80.0)},
            )

            self.assertEqual(set(report["profiles"]), set(PROFILE_NAMES))
            self.assertEqual(set(report["maps"]), {"A01-Race", "A07-Race"})
            for map_report in report["maps"].values():
                self.assertTrue(map_report["source_anchors"])
                self.assertEqual(set(map_report["profile_aggregates"]), set(PROFILE_NAMES))
                self.assertTrue(map_report["profile_aggregates"][PROFILE_NAMES[0]]["section_aggregates"])
                for case in map_report["cases"]:
                    metrics = case["metrics"]
                    for key in (
                        "recovery_activated",
                        "recovery_success",
                        "mean_abs_lateral_error_m",
                        "mean_horizontal_error_m",
                        "mean_abs_heading_error_rad",
                        "progress_continuity",
                        "backward_progress_event_count",
                        "backward_progress_distance_m",
                        "completion",
                        "section_metrics",
                    ):
                        self.assertIn(key, metrics)
            self.assertEqual(set(report["summary"]["metric_winners"]), {
                "completion_rate",
                "progress_complete_rate",
                "mean_final_progress_fraction",
                "mean_abs_lateral_error_m",
                "mean_horizontal_error_m",
                "mean_abs_heading_error_rad",
                "backward_progress_event_rate",
            })
            json.dumps(report)

    def test_guardrail_config_is_derived_from_calibration_and_flags_are_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            telemetry_dir = root / "telemetry"
            maps_dir = root / "maps"
            telemetry_dir.mkdir()
            maps_dir.mkdir()
            for map_name in ("A01-Race", "A07-Race"):
                _write_telemetry(telemetry_dir / f"{map_name}.repaired.csv")
                np.save(maps_dir / f"{map_name}_0.5m_cl2.npy", np.column_stack((np.arange(0.0, 121.0, 1.0), np.zeros(121), np.zeros(121))))
            calibration = _calibration_fixture()
            report = build_report(
                telemetry_dir,
                maps_dir,
                calibration_report=calibration,
                max_steps=2,
                anchor_progress_m={"A01-Race": (20.0,), "A07-Race": (20.0,)},
            )
            expected = next(item for item in derive_candidates(calibration) if item.name == "captured_p25_guardrail")
            self.assertEqual(report["profiles"]["captured_p25_guardrail"]["config"]["max_speed_mps"], expected.config.max_speed_mps)
            self.assertEqual(report["profiles"]["captured_p25_guardrail"]["config"]["curvature_gain"], expected.config.curvature_gain)
            self.assertTrue(report["profiles"]["captured_p25_guardrail"]["derived_from_calibration"])
            self.assertTrue(report["offline_only"])
            self.assertFalse(report["trackmania_launched"])
            self.assertFalse(report["training_started"])
            self.assertFalse(report["azure_started"])
            self.assertFalse(report["checkpoints_loaded"])
            self.assertFalse(report["safety_flags"]["live_control_used"])
            self.assertFalse(report["safety_flags"]["controller_defaults_modified"])

    def test_anchor_selection_is_nearest_and_configurable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            telemetry_dir = root / "telemetry"
            maps_dir = root / "maps"
            telemetry_dir.mkdir()
            maps_dir.mkdir()
            _write_telemetry(telemetry_dir / "A01-Race.repaired.csv")
            np.save(maps_dir / "A01-Race_0.5m_cl2.npy", np.column_stack((np.arange(0.0, 121.0, 1.0), np.zeros(121), np.zeros(121))))
            from trackmania_rl.hybrid_rollout import load_rollout_csv
            from trackmania_rl.hybrid_state import build_centerline_geometry

            samples = load_rollout_csv(telemetry_dir / "A01-Race.repaired.csv")
            geometry = build_centerline_geometry(np.load(maps_dir / "A01-Race_0.5m_cl2.npy"))
            anchors = select_source_anchors("A01-Race", samples, geometry, target_progress_m=(21.0, 79.0))
            self.assertEqual([item["source_index"] for item in anchors], [2, 8])
            self.assertEqual([item["target_progress_m"] for item in anchors], [21.0, 79.0])


if __name__ == "__main__":
    unittest.main()
