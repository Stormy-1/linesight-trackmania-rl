import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from trackmania_rl.reproducibility import MissingCheckpointError, record_run_metadata, safe_config_snapshot, validate_checkpoint_files


class ReproducibilityTests(unittest.TestCase):
    def test_missing_required_checkpoint_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(MissingCheckpointError, "weights1.torch.*weights2.torch"):
                validate_checkpoint_files(Path(directory))

    def test_metadata_contains_hashes_and_redacts_sensitive_values(self):
        with tempfile.TemporaryDirectory() as directory:
            base_dir = Path(directory)
            save_dir = base_dir / "save" / "gen_v1"
            save_dir.mkdir(parents=True)
            for filename, contents in (("weights1.torch", b"online"), ("weights2.torch", b"target")):
                (save_dir / filename).write_bytes(contents)

            config = SimpleNamespace(run_name="gen_v1", batch_size=4, api_key="do-not-record", target_path=Path("secret"))
            snapshot = safe_config_snapshot(config)
            self.assertNotIn("do-not-record", json.dumps(snapshot))
            self.assertEqual(snapshot["api_key"], "<redacted>")
            self.assertEqual(snapshot["target_path"], "<local path redacted>")

            metadata_path = record_run_metadata(
                base_dir,
                save_dir,
                config,
                require_checkpoint=True,
                checkpoint_origin="save/source_checkpoint",
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["run_name"], "gen_v1")
            self.assertEqual(metadata["checkpoint"]["origin"], "save/source_checkpoint")
            self.assertEqual(len(metadata["checkpoint"]["files"]), 2)
            self.assertTrue(all(file_metadata["sha256"] for file_metadata in metadata["checkpoint"]["files"]))


if __name__ == "__main__":
    unittest.main()
