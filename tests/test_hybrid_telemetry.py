import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trackmania_rl.hybrid_telemetry import (
    ACTION_TELEMETRY_COLUMNS,
    RANDOMIZED_ACTION_TELEMETRY_COLUMNS,
    TELEMETRY_COLUMNS,
    HybridTelemetryWriter,
)


class HybridTelemetryTests(unittest.TestCase):
    def test_writer_emits_rollout_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.csv"
            with HybridTelemetryWriter(path) as writer:
                writer.write_sample(0.0, np.array([1.0, 2.0, 3.0]), np.array([2.0, 0.0, 0.0]), 12.5)
                writer.write_sample(0.1, np.array([2.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0]), 13.0)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(rows[0].keys()), TELEMETRY_COLUMNS)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["heading_x"], "1.0")

    def test_writer_rejects_invalid_heading_and_duplicate_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.csv"
            writer = HybridTelemetryWriter(path)
            with self.assertRaises(ValueError):
                writer.write_sample(0.0, np.zeros(3), np.zeros(3), 1.0)
            writer.close()
            with self.assertRaises(FileExistsError):
                HybridTelemetryWriter(path)

    def test_extended_writer_emits_action_columns_without_changing_legacy_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action_rollout.csv"
            with HybridTelemetryWriter(path, include_action_columns=True) as writer:
                writer.write_sample(
                    0.0,
                    np.array([1.0, 2.0, 3.0]),
                    np.array([2.0, 0.0, 0.0]),
                    12.5,
                    action_idx=np.int64(1),
                    action_inputs={"accelerate": True, "brake": False, "left": True, "right": False},
                    action_was_greedy=np.bool_(True),
                )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(rows[0].keys()), ACTION_TELEMETRY_COLUMNS)
            self.assertEqual(rows[0]["action_idx"], "1")
            self.assertEqual(rows[0]["action_left"], "True")

    def test_extended_writer_requires_complete_valid_action_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action_rollout.csv"
            with HybridTelemetryWriter(path, include_action_columns=True) as writer:
                with self.assertRaises(ValueError):
                    writer.write_sample(
                        0.0,
                        np.ones(3),
                        np.asarray([1.0, 0.0, 0.0]),
                        1.0,
                        action_idx=0,
                    )
                with self.assertRaises(ValueError):
                    writer.write_sample(
                        0.0,
                        np.ones(3),
                        np.asarray([1.0, 0.0, 0.0]),
                        1.0,
                        action_idx=0,
                        action_inputs={"accelerate": False, "brake": False, "left": True, "right": True},
                        action_was_greedy=False,
                    )

    def test_randomized_writer_adds_validated_intervention_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "randomized.csv"
            with HybridTelemetryWriter(
                path,
                include_action_columns=True,
                include_randomization_columns=True,
            ) as writer:
                writer.write_sample(
                    0.0,
                    np.ones(3),
                    np.asarray([1.0, 0.0, 0.0]),
                    1.0,
                    action_idx=np.int64(0),
                    action_inputs={"accelerate": True, "brake": False, "left": False, "right": False},
                    action_was_greedy=np.bool_(False),
                    action_is_randomized=np.bool_(True),
                )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(rows[0].keys()), RANDOMIZED_ACTION_TELEMETRY_COLUMNS)
            self.assertEqual(rows[0]["action_is_randomized"], "True")
            with self.assertRaises(ValueError):
                HybridTelemetryWriter(
                    Path(directory) / "invalid.csv",
                    include_randomization_columns=True,
                )


if __name__ == "__main__":
    unittest.main()
