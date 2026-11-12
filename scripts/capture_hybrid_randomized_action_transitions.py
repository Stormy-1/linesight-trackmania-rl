"""Prepare and run the explicitly authorized randomized A01 capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Lock
from pathlib import Path
from typing import Any

import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from config_files import config_copy
from trackmania_rl import utilities
from trackmania_rl.agents import iqn
from trackmania_rl.hybrid_randomized_capture import (
    SAFE_RANDOMIZED_ACTION_INDICES,
    SeededRandomizedActionPolicy,
)
from trackmania_rl.map_loader import load_next_map_zone_centers
from trackmania_rl.tmi_interaction import game_instance_manager

CHECKPOINT = BASE_DIR / "save" / "hocko_v1_foundational_checkpoint" / "weights1.torch"
MAP_PATH = r'"Official Maps\A01-Race.Challenge.Gbx"'
CENTERLINE_FILE = "A01-Race_0.5m_cl2.npy"
DEFAULT_SEEDS = (2026110301, 2026110302, 2026110303, 2026110304, 2026110305, 2026110306)
CAPTURE_ENV = "LINESIGHT_HYBRID_RANDOMIZED_ACTION_CAPTURE_DIR"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inferer() -> iqn.Inferer:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_network, _ = iqn.make_untrained_iqn_network(config_copy.use_jit, is_inference=True)
    inference_network.eval()
    utilities.load_model_checkpoint(
        inference_network,
        CHECKPOINT,
        map_location=device,
        allowed_missing_keys=utilities.NON_PERSISTENT_NORMALIZATION_KEYS,
        allowed_unexpected_keys=utilities.NON_PERSISTENT_NORMALIZATION_KEYS,
    )
    inferer = iqn.Inferer(inference_network, config_copy.iqn_k, config_copy.tau_epsilon_boltzmann)
    inferer.is_explo = False
    inferer.epsilon = 0.0
    inferer.epsilon_boltzmann = 0.0
    return inferer


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_capture(output_dir: Path, seeds: tuple[int, ...], *, port: int) -> dict[str, Any]:
    if len(seeds) != 6 or len(set(seeds)) != len(seeds):
        raise ValueError("exactly six distinct seeds are required: four fit and two held-out")
    resolved_output_dir = output_dir.resolve()
    try:
        resolved_output_dir.relative_to(BASE_DIR.resolve())
    except ValueError as exc:
        raise ValueError("capture output must remain inside the workspace") from exc
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse capture directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()

    zone_centers = load_next_map_zone_centers(CENTERLINE_FILE, BASE_DIR)
    inferer = _load_inferer()
    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=Lock(),
        running_speed=config_copy.running_speed,
        run_steps_per_action=config_copy.tm_engine_step_per_action,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        tmi_port=port,
    )
    manifest: dict[str, Any] = {
        "capture_name": output_dir.name,
        "status": "started",
        "map": "A01",
        "checkpoint": str(CHECKPOINT.relative_to(BASE_DIR)),
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "training_enabled": False,
        "checkpoint_writes_enabled": False,
        "intervention_probability": 0.25,
        "safe_randomized_action_indices": list(SAFE_RANDOMIZED_ACTION_INDICES),
        "fit_seeds": list(seeds[:4]),
        "heldout_seeds": list(seeds[4:]),
        "runs": [],
    }
    _write_manifest(output_dir, manifest)

    previous_capture_dir = os.environ.get(CAPTURE_ENV)
    previous_action_dir = os.environ.pop("LINESIGHT_HYBRID_ACTION_CAPTURE_DIR", None)
    previous_telemetry_dir = os.environ.pop("LINESIGHT_HYBRID_TELEMETRY_DIR", None)
    os.environ.pop(CAPTURE_ENV, None)
    try:
        # Synchronize the visible game without writing a measured capture.
        tmi.rollout(
            exploration_policy=inferer.get_exploration_action,
            map_path=MAP_PATH,
            zone_centers=zone_centers,
            update_network=lambda: None,
        )
        os.environ[CAPTURE_ENV] = str(output_dir)
        for run_index, seed in enumerate(seeds, start=1):
            split = "fit" if run_index <= 4 else "heldout"
            policy = SeededRandomizedActionPolicy(inferer.get_exploration_action, seed=seed)
            rollout_results, stats = tmi.rollout(
                exploration_policy=policy,
                map_path=MAP_PATH,
                zone_centers=zone_centers,
                update_network=lambda: None,
            )
            telemetry_path = stats.get("hybrid_telemetry_path")
            if not telemetry_path:
                raise RuntimeError("measured rollout did not report randomized telemetry path")
            run_record = {
                "run_index": run_index,
                "split": split,
                "seed": seed,
                "telemetry_path": str(Path(telemetry_path).as_posix()),
                "finished": bool(stats.get("race_finished", False)),
                "furthest_zone_idx": int(rollout_results.get("furthest_zone_idx", 0)),
            }
            manifest["runs"].append(run_record)
            _write_manifest(output_dir, manifest)
        manifest["status"] = "complete"
        _write_manifest(output_dir, manifest)
        return manifest
    finally:
        if manifest["status"] != "complete":
            manifest["status"] = "partial"
            _write_manifest(output_dir, manifest)
        if previous_capture_dir is None:
            os.environ.pop(CAPTURE_ENV, None)
        else:
            os.environ[CAPTURE_ENV] = previous_capture_dir
        if previous_action_dir is not None:
            os.environ["LINESIGHT_HYBRID_ACTION_CAPTURE_DIR"] = previous_action_dir
        if previous_telemetry_dir is not None:
            os.environ["LINESIGHT_HYBRID_TELEMETRY_DIR"] = previous_telemetry_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--seeds", type=int, nargs=6, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--authorize-named-capture",
        action="store_true",
        help="explicitly authorize the visible randomized A01 capture",
    )
    args = parser.parse_args()
    if not args.authorize_named_capture:
        parser.error("visible capture requires --authorize-named-capture")
    manifest = run_capture(args.output_dir, tuple(args.seeds), port=args.port)
    print(json.dumps({"output_dir": str(args.output_dir), "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()
