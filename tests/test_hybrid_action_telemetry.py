import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trackmania_rl.hybrid_action_telemetry import (
    load_action_rollout_csv,
    parse_action_rollout_rows,
)
from trackmania_rl.hybrid_telemetry import HybridTelemetryWriter


def _row(timestamp="0.0", **overrides):
    row = {
        "timestamp_s": timestamp,
        "position_x_m": "0.0",
        "position_y_m": "0.0",
        "position_z_m": "0.0",
        "heading_x": "1.0",
        "heading_y": "0.0",
        "heading_z": "0.0",
        "speed_mps": "10.0",
        "action_idx": "0",
        "action_accelerate": "True",
        "action_brake": "False",
        "action_left": "False",
        "action_right": "False",
        "action_was_greedy": "True",
    }
    row.update(overrides)
    return row


class HybridActionTelemetryTests(unittest.TestCase):
    def test_parser_reads_action_fields_and_preserves_alignment(self):
        parsed = parse_action_rollout_rows([_row(), _row("0.1", action_idx="8", action_brake="True", action_accelerate="False")])

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].action_idx, 0)
        self.assertEqual(parsed[1].action_idx, 8)
        self.assertEqual(parsed[1].action_inputs, (False, True, False, False))
        self.assertAlmostEqual(parsed[1].telemetry.timestamp_s, 0.1)

    def test_parser_rejects_invalid_actions_and_timestamp_order(self):
        with self.assertRaises(ValueError):
            parse_action_rollout_rows([_row(), _row("0.1", action_left="True", action_right="True")])
        with self.assertRaises(ValueError):
            parse_action_rollout_rows([_row(), _row("0.0")])
        with self.assertRaises(ValueError):
            parse_action_rollout_rows([_row(action_was_greedy="maybe")])

    def test_load_parser_accepts_writer_output_and_rejects_legacy_file(self):
        with tempfile.TemporaryDirectory() as directory:
            action_path = Path(directory) / "action.csv"
            with HybridTelemetryWriter(action_path, include_action_columns=True) as writer:
                writer.write_sample(
                    0.0,
                    np.zeros(3),
                    np.asarray([1.0, 0.0, 0.0]),
                    10.0,
                    action_idx=0,
                    action_inputs={"accelerate": True, "brake": False, "left": False, "right": False},
                    action_was_greedy=True,
                )
            self.assertEqual(load_action_rollout_csv(action_path)[0].action_idx, 0)

            legacy_path = Path(directory) / "legacy.csv"
            with legacy_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(
                    [
                        ["timestamp_s", "position_x_m", "position_y_m", "position_z_m", "heading_x", "heading_y", "heading_z", "speed_mps"],
                        ["0.0", "0.0", "0.0", "0.0", "1.0", "0.0", "0.0", "10.0"],
                    ]
                )
            with self.assertRaises(ValueError):
                load_action_rollout_csv(legacy_path)

    def test_parser_preserves_optional_randomization_provenance_and_rejects_partial_rows(self):
        rows = [_row(action_is_randomized="True"), _row("0.1", action_is_randomized="False")]
        parsed = parse_action_rollout_rows(rows)
        self.assertEqual([sample.action_is_randomized for sample in parsed], [True, False])
        with self.assertRaises(ValueError):
            parse_action_rollout_rows([_row(action_is_randomized="True"), _row("0.1")])


if __name__ == "__main__":
    unittest.main()
