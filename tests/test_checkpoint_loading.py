import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("PyTorch is not installed") from error

from trackmania_rl.utilities import load_model_checkpoint


class TestLoadModelCheckpoint(unittest.TestCase):
    def test_accepts_matching_state_dict(self):
        with self.subTest("matching state dict"):
            source = torch.nn.Linear(2, 1)
            checkpoint_path = self._checkpoint_path()
            torch.save(source.state_dict(), checkpoint_path)

            destination = torch.nn.Linear(2, 1)
            load_model_checkpoint(destination, checkpoint_path, map_location="cpu")

            self.assertTrue(
                all(torch.equal(source_state, destination.state_dict()[name]) for name, source_state in source.state_dict().items())
            )

    def test_reports_missing_and_unexpected_keys(self):
        checkpoint_path = self._checkpoint_path()
        torch.save({"weight": torch.ones(1, 2), "extra": torch.ones(1)}, checkpoint_path)

        with self.assertRaisesRegex(RuntimeError, r"missing_keys=\['bias'\].*unexpected_keys=\['extra'\]"):
            load_model_checkpoint(torch.nn.Linear(2, 1), checkpoint_path, map_location="cpu")

    def test_allows_only_explicit_compatibility_keys(self):
        checkpoint_path = self._checkpoint_path()
        torch.save({"weight": torch.ones(1, 2), "extra": torch.ones(1)}, checkpoint_path)

        load_model_checkpoint(
            torch.nn.Linear(2, 1),
            checkpoint_path,
            map_location="cpu",
            allowed_missing_keys=("bias",),
            allowed_unexpected_keys=("extra",),
        )

    def test_requires_checkpoint(self):
        with self.assertRaisesRegex(FileNotFoundError, "Model checkpoint not found"):
            load_model_checkpoint(torch.nn.Linear(2, 1), self._checkpoint_path(), map_location="cpu")

    def _checkpoint_path(self):
        import tempfile

        if not hasattr(self, "_tempdir"):
            self._tempdir = tempfile.TemporaryDirectory()
            self.addCleanup(self._tempdir.cleanup)
        return Path(self._tempdir.name) / "weights.torch"
