import collections
import pickle
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch

from trackmania_rl.safe_load import safe_joblib_load
from trackmania_rl.utilities import load_model_checkpoint

_MARKER_NAME = "pwned.txt"


class _WritesMarkerOnLoad:
    """Pickles as a call to ``Path.touch``: the shape of a real pickle exploit."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (self.marker,))


class SafeJoblibLoadTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_training_stats_shapes(self):
        stats = {
            "cumul_number_frames_played": 123,
            "rolling_mean_ms": collections.defaultdict(int, {"A01": 5}),
            "curve": np.arange(6, dtype=np.float32).reshape(2, 3),
            "best": np.float64(51.25),
            "names": ["A01", "A05"],
            "pair": (1.5, None, True),
        }
        path = self.tmp / "accumulated_stats.joblib"
        joblib.dump(stats, path)

        loaded = safe_joblib_load(path)

        self.assertEqual(loaded["cumul_number_frames_played"], 123)
        self.assertEqual(loaded["rolling_mean_ms"]["A01"], 5)
        self.assertEqual(loaded["rolling_mean_ms"]["missing"], 0)
        np.testing.assert_array_equal(loaded["curve"], stats["curve"])
        self.assertEqual(loaded["best"], 51.25)
        self.assertEqual(loaded["names"], ["A01", "A05"])
        self.assertEqual(loaded["pair"], (1.5, None, True))

    def test_loads_compressed_list_of_arrays(self):
        q_values = [np.ones((4, 12), dtype=np.float32) * i for i in range(3)]
        path = self.tmp / "q_values.joblib"
        joblib.dump(q_values, path, compress=3)

        loaded = safe_joblib_load(path)

        self.assertEqual(len(loaded), 3)
        np.testing.assert_array_equal(loaded[2], q_values[2])

    def test_rejects_pickle_that_calls_a_function(self):
        marker = self.tmp / _MARKER_NAME
        path = self.tmp / "evil.joblib"
        joblib.dump({"stats": _WritesMarkerOnLoad(marker)}, path)

        with self.assertRaises(pickle.UnpicklingError):
            safe_joblib_load(path)
        self.assertFalse(marker.exists())

    def test_rejects_exploit_hidden_in_object_array(self):
        marker = self.tmp / _MARKER_NAME
        hidden = np.empty(1, dtype=object)
        hidden[0] = _WritesMarkerOnLoad(marker)
        path = self.tmp / "evil_array.joblib"
        joblib.dump(hidden, path)

        with self.assertRaises(pickle.UnpicklingError):
            safe_joblib_load(path)
        self.assertFalse(marker.exists())

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            safe_joblib_load(self.tmp / "nope.joblib")


class LoadModelCheckpointSafetyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_a_plain_state_dict(self):
        model = torch.nn.Linear(3, 2)
        path = self.tmp / "weights1.torch"
        torch.save(model.state_dict(), path)
        fresh = torch.nn.Linear(3, 2)

        load_model_checkpoint(fresh, path, map_location="cpu")

        torch.testing.assert_close(fresh.weight, model.weight)

    def test_rejects_checkpoint_that_calls_a_function(self):
        marker = self.tmp / _MARKER_NAME
        path = self.tmp / "weights1.torch"
        torch.save({"weight": _WritesMarkerOnLoad(marker)}, path)

        with self.assertRaises(pickle.UnpicklingError):
            load_model_checkpoint(torch.nn.Linear(3, 2), path, map_location="cpu")
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
