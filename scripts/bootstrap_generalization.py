"""Create a clean generalization checkpoint from a foundational checkpoint.

The bootstrap intentionally copies no optimizer, scaler, replay buffer, or
historical run artifacts.  It is safe to run repeatedly: an existing
destination is rejected unless ``--overwrite`` (or its explicit alias
``--allow-existing``/``--force``) is supplied.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib


SOURCE_RELATIVE_PATH = Path("save") / "hocko_v1_foundational_checkpoint"
DESTINATION_RELATIVE_PATH = Path("save") / "gen_v3_r1"
WEIGHT_FILENAMES = ("weights1.torch", "weights2.torch")
STATS_FILENAME = "accumulated_stats.joblib"
MANIFEST_FILENAME = "bootstrap_manifest.json"
REQUIRED_SOURCE_FILENAMES = (*WEIGHT_FILENAMES, STATS_FILENAME)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file without loading it in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zero_like(value: Any) -> Any:
    """Use a numeric zero compatible with a source counter's value type."""

    if isinstance(value, float):
        return 0.0
    if isinstance(value, bool):
        return False
    return 0


def sanitize_stats(source_stats: Mapping[str, Any]) -> defaultdict[str, Any]:
    """Make stats suitable for a new run while retaining historical results.

    ``cumul_number_frames_played`` drives the learning schedules and therefore
    remains inherited.  The historical best and rolling metrics are useful for
    comparing the new run and are retained.  Every other cumulative value is a
    fresh counter; this also covers future ``cumul_*`` fields added to the
    training loop instead of silently carrying them into a new replay run.
    """

    required_metrics = ("alltime_min_ms", "rolling_mean_ms")
    missing = [name for name in (*required_metrics, "cumul_number_frames_played") if name not in source_stats]
    if missing:
        raise ValueError(f"Foundational stats are missing required fields: {', '.join(missing)}")

    sanitized: defaultdict[str, Any] = defaultdict(int)
    for key, value in source_stats.items():
        if key in required_metrics or key == "cumul_number_frames_played":
            sanitized[key] = copy.deepcopy(value)
        elif key.startswith("cumul_"):
            sanitized[key] = _zero_like(value)
        else:
            # Preserve non-counter metadata if a newer stats file contains it.
            sanitized[key] = copy.deepcopy(value)

    # Keep the fields currently consumed by learner_process.py present even if
    # an older foundational stats file did not yet contain one of them.
    defaults = {
        "cumul_number_batches_done": 0,
        "cumul_number_memories_generated": 0,
        "cumul_number_single_memories_used": 0,
        "cumul_number_single_memories_should_have_been_used": 0,
        "cumul_number_target_network_updates": 0,
        "cumul_number_single_memories_used_next_target_network_update": 0,
        "cumul_training_hours": 0.0,
    }
    for key, value in defaults.items():
        sanitized[key] = value
    return sanitized


def _path_for_manifest(path: Path, base_dir: Path) -> str:
    """Prefer stable workspace-relative paths in the manifest."""

    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return str(path)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def bootstrap_generalization(
    base_dir: Path | None = None,
    *,
    source_dir: Path | None = None,
    destination_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Create ``gen_v3_r1`` and return its path.

    The destination is assembled in a temporary sibling directory and moved
    into place only after all files and the manifest have been written.  The
    source checkpoint and any other run directory are read-only inputs here.
    """

    base_dir = (Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]).resolve()
    source_dir = _resolve_path(base_dir, source_dir or SOURCE_RELATIVE_PATH).resolve()
    destination_dir = _resolve_path(base_dir, destination_dir or DESTINATION_RELATIVE_PATH).resolve()

    if source_dir == destination_dir:
        raise ValueError("Source and destination checkpoints must be different directories")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Foundational checkpoint directory not found: {source_dir}")

    source_paths = {filename: source_dir / filename for filename in REQUIRED_SOURCE_FILENAMES}
    missing = [filename for filename, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Foundational checkpoint is missing required files: {', '.join(missing)}")

    if destination_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing destination: {destination_dir}. "
            "Re-run with --overwrite only if replacing it is intentional."
        )

    # Load and validate stats before creating or replacing anything.
    source_stats = joblib.load(source_paths[STATS_FILENAME])
    if not isinstance(source_stats, Mapping):
        raise ValueError(f"Foundational stats must be a mapping, got {type(source_stats).__name__}")
    sanitized_stats = sanitize_stats(source_stats)

    source_metadata = []
    for filename in REQUIRED_SOURCE_FILENAMES:
        path = source_paths[filename]
        source_metadata.append(
            {
                "name": filename,
                "path": _path_for_manifest(path, base_dir),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{destination_dir.name}.bootstrap-", dir=str(destination_dir.parent)))
    committed = False
    try:
        for filename in WEIGHT_FILENAMES:
            shutil.copy2(source_paths[filename], staging_dir / filename)
        joblib.dump(sanitized_stats, staging_dir / STATS_FILENAME)

        manifest = {
            "schema_version": 1,
            "bootstrapped_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": _path_for_manifest(source_dir, base_dir),
            "source_files": source_metadata,
            "destination": _path_for_manifest(destination_dir, base_dir),
            "copied_files": [*WEIGHT_FILENAMES, STATS_FILENAME],
        }
        (staging_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if destination_dir.exists():
            _remove_path(destination_dir)
        os.replace(staging_dir, destination_dir)
        committed = True
    finally:
        if not committed and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    return destination_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        "--allow-existing",
        "--force",
        dest="overwrite",
        action="store_true",
        help="explicitly replace an existing save/gen_v3_r1 destination",
    )
    parser.add_argument("--base-dir", type=Path, help="repository root (defaults to the repository containing this script)")
    parser.add_argument("--source", type=Path, default=SOURCE_RELATIVE_PATH, help="foundational checkpoint path")
    parser.add_argument("--destination", type=Path, default=DESTINATION_RELATIVE_PATH, help="new checkpoint path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        destination = bootstrap_generalization(
            args.base_dir,
            source_dir=args.source,
            destination_dir=args.destination,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError, OSError) as error:
        print(f"Bootstrap aborted: {error}", file=sys.stderr)
        return 1
    print(f"Generalization checkpoint created at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
