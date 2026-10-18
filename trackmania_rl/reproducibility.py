"""Small, local helpers for recording reproducible training-run metadata.

The metadata file contains configuration values and hashes, never configuration
source text or checkpoint contents.  Checkpoints are only inspected; this
module never copies or modifies them.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


CHECKPOINT_FILENAMES = ("weights1.torch", "weights2.torch")
METADATA_FILENAME = "reproducibility.json"
_SENSITIVE_NAME = re.compile(r"(?:api[_-]?key|auth|credential|password|private[_-]?key|secret|token)", re.I)
_PATH_NAME = re.compile(r"(?:^|[_-])(path|directory|dir|folder|location)(?:$|[_-])", re.I)
_SOURCE_SUFFIXES = {".as", ".json", ".py", ".toml", ".yaml", ".yml"}


class MissingCheckpointError(FileNotFoundError):
    """Raised when a run that requires an initial checkpoint cannot resume."""


def _safe_config_value(value: Any, name: str = "") -> Any:
    """Convert a config value to JSON without exposing secrets or local paths."""

    if _SENSITIVE_NAME.search(name):
        return "<redacted>"
    if _PATH_NAME.search(name):
        return "<local path redacted>"
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return "<local path redacted>"
    if isinstance(value, dict):
        return {str(key): _safe_config_value(item, str(key)) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_safe_config_value(item, name) for item in value]

    # Numpy scalar values are common in config.py, but importing numpy here
    # would make this reusable helper less lightweight.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _safe_config_value(item(), name)
        except (TypeError, ValueError):
            pass

    if inspect.ismodule(value) or inspect.isclass(value) or inspect.isfunction(value):
        return None
    # Avoid invoking arbitrary repr implementations, which could expose a
    # token or another value that was not covered by the name-based filter.
    return f"<{type(value).__name__}>"


def safe_config_snapshot(config_module: ModuleType) -> dict[str, Any]:
    """Return serializable, non-secret public values from a config module."""

    snapshot: dict[str, Any] = {}
    for name, value in sorted(vars(config_module).items()):
        if name.startswith("_") or inspect.ismodule(value) or inspect.isclass(value) or inspect.isfunction(value):
            continue
        safe_value = _safe_config_value(value, name)
        if safe_value is not None:
            snapshot[name] = safe_value
    return snapshot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path, base_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base_dir).as_posix() if path.is_relative_to(base_dir) else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def validate_checkpoint_files(save_dir: Path, required_names: tuple[str, ...] = CHECKPOINT_FILENAMES) -> list[Path]:
    """Require the files needed to initialize a resumed learner.

    This deliberately checks presence only.  Loading remains the learner's
    responsibility, and no checkpoint is copied or altered here.
    """

    save_dir = Path(save_dir)
    checkpoint_paths = [save_dir / filename for filename in required_names]
    missing = [path.name for path in checkpoint_paths if not path.is_file()]
    if missing:
        expected = ", ".join(str(path) for path in checkpoint_paths)
        raise MissingCheckpointError(
            "Required initial checkpoint files are missing for this run: "
            f"{', '.join(missing)}. Expected files: {expected}. "
            "Training was stopped before workers started; no random-weight fallback was used."
        )
    return checkpoint_paths


def _source_metadata(base_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=base_dir, check=False, capture_output=True, text=True
        )
        if commit.returncode == 0:
            metadata["git_commit"] = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=base_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode == 0:
            metadata["git_dirty"] = bool(status.stdout.strip())
    except OSError:
        pass
    source_paths = [base_dir / filename for filename in ("pyproject.toml", "setup.py")]
    source_roots = (base_dir / "config_files", base_dir / "scripts", base_dir / "trackmania_rl")
    for root in source_roots:
        if root.is_dir():
            source_paths.extend(path for path in root.rglob("*") if path.is_file() and path.suffix in _SOURCE_SUFFIXES)
    metadata["files"] = [
        _file_metadata(path, base_dir)
        for path in sorted(set(source_paths))
        if path.is_file() and path.name != "config_copy.py"
    ]
    return metadata


def record_run_metadata(
    base_dir: Path,
    save_dir: Path,
    config_module: ModuleType,
    *,
    require_checkpoint: bool = False,
    checkpoint_origin: str | None = None,
) -> Path:
    """Record active config, source state, and checkpoint fingerprints."""

    base_dir = Path(base_dir)
    save_dir = Path(save_dir)
    checkpoint_paths = validate_checkpoint_files(save_dir) if require_checkpoint else [
        save_dir / filename for filename in CHECKPOINT_FILENAMES if (save_dir / filename).is_file()
    ]
    save_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": str(getattr(config_module, "run_name", save_dir.name)),
        "config": safe_config_snapshot(config_module),
        "source": _source_metadata(base_dir),
        "checkpoint": {
            "required": require_checkpoint,
            "origin": checkpoint_origin,
            "files": [_file_metadata(path, base_dir) for path in checkpoint_paths],
        },
    }
    metadata_path = save_dir / METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path
