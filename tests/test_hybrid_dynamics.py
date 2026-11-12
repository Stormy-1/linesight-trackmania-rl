import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.calibrate_hybrid_recorded_dynamics import build_report, calibrate_transition_split
from trackmania_rl.hybrid_dynamics import transition_metrics
from trackmania_rl.hybrid_rollout import RolloutSample, parse_rollout_rows


def _samples(count=5):
    return [
        RolloutSample(
            timestamp_s=index * 0.1,
            position=np.asarray([index * 2.0, 0.0, 0.0], dtype=np.float64),
            heading=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            speed_mps=20.0,
        )
        for index in range(count)
    ]


class HybridDynamicsTests(unittest.TestCase):
    def test_transition_metrics_are_finite_and_deterministic(self):
        first = transition_metrics(_samples())
        second = transition_metrics(_samples())

        self.assertEqual(first, second)
        self.assertEqual(first["transition_count"], 4)
        self.assertAlmostEqual(first["timestamp"]["p50_s"], 0.1)
        self.assertAlmostEqual(first["displacement_rate"]["moving_p50_mps"], 20.0)
        self.assertAlmostEqual(first["position_prediction_residual"]["moving_p50_m"], 0.0)
        self.assertTrue(np.isfinite(first["observed_abs_yaw_rate"]["p99_rad_s"]))

    def test_parser_rejects_non_monotonic_or_nonfinite_rows(self):
        row = {
            "timestamp_s": "0.0",
            "position_x_m": "0.0",
            "position_y_m": "0.0",
            "position_z_m": "0.0",
            "heading_x": "1.0",
            "heading_y": "0.0",
            "heading_z": "0.0",
            "speed_mps": "1.0",
        }
        with self.assertRaises(ValueError):
            parse_rollout_rows([row, {**row, "timestamp_s": "0.0"}])
        with self.assertRaises(ValueError):
            parse_rollout_rows([{**row, "speed_mps": "nan"}])

    def test_split_calibration_rejects_overlapping_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"samples": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                calibrate_transition_split(
                    {"fit": [path], "heldout": [path]},
                )

    def test_saved_report_is_disjoint_finite_and_diagnostic_only(self):
        report = build_report()

        self.assertFalse(report["trackmania_launched"])
        self.assertFalse(report["training_started"])
        self.assertFalse(report["controller_defaults_modified"])
        self.assertEqual(report["promotion_allowed"], False)
        self.assertGreater(report["splits"]["fit"]["transition_count"], 0)
        self.assertGreater(report["splits"]["heldout"]["transition_count"], 0)
        self.assertTrue(set(report["source_paths"]["fit"]).isdisjoint(report["source_paths"]["heldout"]))
        self.assertTrue(report["limitations"])


if __name__ == "__main__":
    unittest.main()
