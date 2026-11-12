import tempfile
import unittest
from pathlib import Path

import joblib

from trackmania_rl.utilities import create_stage_baseline, load_stage_baseline_start_frames, restore_stage_baseline


class _Stateful:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"value": self.value}


class StageBaselineTests(unittest.TestCase):
    def test_baseline_is_created_once_and_can_restore_all_runtime_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "stage_baseline"
            destination = root / "run"
            destination.mkdir()
            stats = {"cumul_number_frames_played": 123, "cumul_number_batches_done": 0}

            create_stage_baseline(
                baseline,
                _Stateful("online-start"),
                _Stateful("target-start"),
                _Stateful("optimizer-start"),
                _Stateful("scaler-start"),
                stats,
            )
            (baseline / "weights1.torch").write_bytes(b"changed-active-copy")

            create_stage_baseline(
                baseline,
                _Stateful("should-not-replace"),
                _Stateful("should-not-replace"),
                _Stateful("should-not-replace"),
                _Stateful("should-not-replace"),
                {"cumul_number_frames_played": 999},
            )
            self.assertEqual((baseline / "weights1.torch").read_bytes(), b"changed-active-copy")

            restore_stage_baseline(baseline, destination)
            self.assertEqual(joblib.load(destination / "accumulated_stats.joblib"), stats)
            self.assertEqual((destination / "weights2.torch").read_bytes(), (baseline / "weights2.torch").read_bytes())
            self.assertTrue((destination / "optimizer1.torch").is_file())
            self.assertTrue((destination / "scaler.torch").is_file())
            self.assertEqual(load_stage_baseline_start_frames(baseline), 123)

    def test_stage_baseline_start_frames_rejects_invalid_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory)
            joblib.dump({"cumul_number_frames_played": -1}, baseline / "accumulated_stats.joblib")
            with self.assertRaisesRegex(RuntimeError, "cannot be negative"):
                load_stage_baseline_start_frames(baseline)


if __name__ == "__main__":
    unittest.main()
