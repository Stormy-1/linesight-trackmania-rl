import hashlib
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import joblib

from scripts.bootstrap_generalization import bootstrap_generalization


class BootstrapGeneralizationTests(unittest.TestCase):
    def test_bootstraps_weights_and_sanitized_stats_without_optimizer_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base_dir = Path(directory)
            source_dir = base_dir / "save" / "hocko_v1_foundational_checkpoint"
            source_dir.mkdir(parents=True)
            (source_dir / "weights1.torch").write_bytes(b"online")
            (source_dir / "weights2.torch").write_bytes(b"target")
            source_stats = defaultdict(
                int,
                {
                    "alltime_min_ms": {"hock": 54450},
                    "rolling_mean_ms": {"hock": 55434.38},
                    "cumul_number_frames_played": 3180286,
                    "cumul_number_batches_done": 12,
                    "cumul_number_memories_generated": 34,
                    "cumul_number_single_memories_used": 56,
                    "cumul_number_single_memories_should_have_been_used": 78,
                    "cumul_number_single_memories_used_next_target_network_update": 90,
                    "cumul_number_target_network_updates": 3,
                    "cumul_training_hours": 1.5,
                },
            )
            joblib.dump(source_stats, source_dir / "accumulated_stats.joblib")

            destination_dir = bootstrap_generalization(base_dir)

            self.assertEqual(
                sorted(path.name for path in destination_dir.iterdir()),
                ["accumulated_stats.joblib", "bootstrap_manifest.json", "weights1.torch", "weights2.torch"],
            )
            self.assertEqual((destination_dir / "weights1.torch").read_bytes(), b"online")
            self.assertFalse((destination_dir / "optimizer1.torch").exists())
            self.assertFalse((destination_dir / "scaler.torch").exists())

            stats = joblib.load(destination_dir / "accumulated_stats.joblib")
            self.assertEqual(stats["cumul_number_frames_played"], source_stats["cumul_number_frames_played"])
            self.assertEqual(stats["alltime_min_ms"], source_stats["alltime_min_ms"])
            self.assertEqual(stats["rolling_mean_ms"], source_stats["rolling_mean_ms"])
            for key in source_stats:
                if key.startswith("cumul_") and key != "cumul_number_frames_played":
                    self.assertEqual(stats[key], 0, key)
            self.assertEqual(stats["cumul_training_hours"], 0.0)

            manifest = json.loads((destination_dir / "bootstrap_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_checkpoint"], "save/hocko_v1_foundational_checkpoint")
            self.assertEqual(manifest["destination"], "save/gen_v3_r1")
            weights1_metadata = next(item for item in manifest["source_files"] if item["name"] == "weights1.torch")
            self.assertEqual(weights1_metadata["sha256"], hashlib.sha256(b"online").hexdigest())

    def test_existing_destination_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base_dir = Path(directory)
            source_dir = base_dir / "save" / "hocko_v1_foundational_checkpoint"
            source_dir.mkdir(parents=True)
            for filename, contents in (("weights1.torch", b"one"), ("weights2.torch", b"two")):
                (source_dir / filename).write_bytes(contents)
            joblib.dump(
                {"alltime_min_ms": {}, "rolling_mean_ms": {}, "cumul_number_frames_played": 1},
                source_dir / "accumulated_stats.joblib",
            )
            destination_dir = base_dir / "save" / "gen_v3_r1"
            destination_dir.mkdir(parents=True)
            marker = destination_dir / "keep-me"
            marker.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                bootstrap_generalization(base_dir)
            self.assertTrue(marker.exists())

            bootstrap_generalization(base_dir, overwrite=True)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
