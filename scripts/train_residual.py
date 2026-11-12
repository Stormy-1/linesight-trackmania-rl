"""Train the residual TD3 policy live on the rl_train maps (Step 4, HANDOVER §77).

One game instance; the maps are driven in blocks (``--episodes-per-map`` in a row, so the
map loads once per block). Every episode: the classical stack (v7a settings + virtual lidar)
drives, the actor + Gaussian noise adds the residual, then the trainer learns from the run
(``trackmania_rl/residual_training.py``). Only ``rl_train`` maps are allowed: dev maps are for
choices, the test set is scored once at the end.

Needs TrackMania + TMInterface on the visible desktop. Resumable: rerun with ``--resume``.

    python scripts/train_residual.py --name td3_v1 --episodes 60
    python scripts/train_residual.py --name td3_v1 --episodes 120 --resume

Evaluate a checkpoint like any other run (deterministic actor, no noise):
    python scripts/run_hybrid_controller_live.py --map A01-Race ... --residual eval_results/residual_rl/td3_v1/latest.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from multiprocessing import Lock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_map_split import MANIFEST, load_split
from run_hybrid_controller_live import campaign_meta, make_controller, make_range_sensor, summarize_run
from run_test_set import max_race_s

from config_files import config_copy
from trackmania_rl.hybrid_live import HybridLivePolicy
from trackmania_rl.map_loader import load_next_map_zone_centers
from trackmania_rl.residual_action import ResidualConfig
from trackmania_rl.residual_td3 import TD3Agent, TD3Config
from trackmania_rl.residual_training import ResidualTrainer, TrainingConfig, illegal_training_maps
from trackmania_rl.tmi_interaction import game_instance_manager

LIVE_DIR = ROOT / "eval_results" / "hybrid_live"
OUT_ROOT = ROOT / "eval_results" / "residual_rl"
# The base stack = v7a settings (HANDOVER §70) + the virtual lidar (§72). Global, never per map.
SPEED_MODEL = ROOT / "eval_results" / "speed_model" / "speed_model_v1.pt"
GRIP_MODEL = ROOT / "eval_results" / "grip_model" / "grip_model_v1.json"
FOOTPRINTS = ROOT / "eval_results" / "range_sensor" / "footprints_v1.npz"
TOP_SPEED_MPS = 70.0
GRIP_RISK = 0.2
STUCK_CUTOFF_S = 6.0


@dataclass
class MapContext:
    stem: str
    map_path: str
    zone_centers: np.ndarray
    finish_zone_idx: int
    max_race_s: float
    policy: HybridLivePolicy


def build_context(stem: str, trainer: ResidualTrainer, manifest: dict) -> MapContext:
    meta = campaign_meta(stem)
    centerline = np.load(ROOT / "maps" / meta["cl_file"])
    zone_centers = load_next_map_zone_centers(meta["cl_file"], ROOT)
    controller = make_controller(
        "purepursuit", None, centerline, (), SPEED_MODEL, 1.0, TOP_SPEED_MPS, GRIP_MODEL, GRIP_RISK
    )
    policy = HybridLivePolicy(
        centerline,
        controller,
        range_sensor=make_range_sensor(FOOTPRINTS, stem),
        residual_policy=trainer.policy,
        residual_config=ResidualConfig(),
    )
    return MapContext(
        stem=stem,
        map_path=meta["map_path"],
        zone_centers=zone_centers,
        finish_zone_idx=len(zone_centers) - config_copy.n_zone_centers_extrapolate_after_end_of_map,
        max_race_s=float(max_race_s(manifest[stem]["author_time_s"])),
        policy=policy,
    )


def run_rollout(tmi, ctx: MapContext):
    return tmi.rollout(
        exploration_policy=ctx.policy, map_path=ctx.map_path, zone_centers=ctx.zone_centers, update_network=lambda: None
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="output folder under eval_results/residual_rl/")
    parser.add_argument("--episodes", type=int, required=True, help="total episodes (including earlier ones on --resume)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maps", nargs="*", default=None, help="subset of the rl_train maps (default: all)")
    parser.add_argument("--episodes-per-map", type=int, default=3)
    parser.add_argument("--init", type=Path, default=OUT_ROOT / "critic_burnin_v2.pt", help="starting checkpoint")
    parser.add_argument("--seed-data", default="*_rl_train_v7zero", help="zero-residual run folders preloaded into the buffer")
    parser.add_argument("--noise", type=float, default=TrainingConfig().exploration_noise)
    parser.add_argument("--actor-warmup", type=int, default=TrainingConfig().actor_warmup_transitions)
    parser.add_argument("--noise-hold", type=int, default=TrainingConfig().noise_hold_steps,
                        help="hold each exploration-noise draw for this many 50 ms steps (10 = 0.5 s)")
    parser.add_argument("--anchor-alpha", default=str(TD3Config().anchor_alpha),
                        help="TD3+BC anchor alpha ('none' = plain TD3); larger = weaker pull to the base command")
    parser.add_argument("--buffer-from", type=Path, nargs="*", default=[],
                        help="episodes folders of earlier training runs to preload (they count as exploration)")
    parser.add_argument("--speed", type=float, default=10.0, help="game speed multiplier")
    parser.add_argument("--port", type=int, default=config_copy.base_tmi_port)
    parser.add_argument("--keep-game", action="store_true")
    args = parser.parse_args()
    rl_train = load_split()["rl_train"]
    args.maps = args.maps or list(rl_train)
    illegal = illegal_training_maps(args.maps, rl_train)
    if illegal:
        parser.error(f"training is only allowed on rl_train maps; not: {illegal}")
    if args.episodes_per_map < 1 or args.episodes < 1:
        parser.error("--episodes and --episodes-per-map must be >= 1")
    return args


def make_trainer(args: argparse.Namespace, out_dir: Path) -> ResidualTrainer:
    if args.resume:
        if not (out_dir / "latest.pt").is_file():
            raise SystemExit(f"--resume: no {out_dir / 'latest.pt'}")
        agent = TD3Agent.load(out_dir / "latest.pt")
    else:
        if out_dir.exists():
            raise SystemExit(f"{out_dir} exists (use --resume or a new --name)")
        if not args.init.is_file():
            raise SystemExit(f"--init {args.init} not found (run scripts/burn_in_residual_critic.py)")
        anchor = None if args.anchor_alpha == "none" else float(args.anchor_alpha)
        # training hyperparameters (target noise back on), with the chosen anchor
        agent = TD3Agent.load(args.init, config=TD3Config(anchor_alpha=anchor))
    config = TrainingConfig(
        exploration_noise=args.noise, actor_warmup_transitions=args.actor_warmup, noise_hold_steps=args.noise_hold
    )
    trainer = ResidualTrainer(agent, config, out_dir)
    seed_files = sorted(LIVE_DIR.glob(f"{args.seed_data}/run01_rl.npz"))
    seeded = trainer.add_seed_data(seed_files)
    earlier_files = sorted(p for folder in args.buffer_from for p in folder.glob("ep*.npz"))
    earlier = trainer.add_seed_data(earlier_files, explored=True)
    resumed = trainer.resume() if args.resume else 0
    print(f"[train] buffer {trainer.buffer.size} ({seeded} seed + {earlier} earlier-run transitions), "
          f"explored {trainer.explored_transitions}, resumed {resumed} episodes; anchor {trainer.agent.config.anchor_alpha}")
    if not args.resume:
        trainer.write_config({"init": str(args.init), "maps": args.maps, "seed_runs": [p.parent.name for p in seed_files],
                              "episodes_per_map": args.episodes_per_map, "game_speed": args.speed,
                              "buffer_from": [str(f) for f in args.buffer_from]})
    return trainer


def main() -> None:
    args = parse_args()
    out_dir = OUT_ROOT / args.name
    trainer = make_trainer(args, out_dir)
    manifest = json.loads(MANIFEST.read_text())
    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=Lock(),
        running_speed=args.speed,
        run_steps_per_action=config_copy.tm_engine_step_per_action,
        max_overall_duration_ms=300 * 1000,
        max_minirace_duration_ms=int(STUCK_CUTOFF_S * 1000),
        tmi_port=args.port,
    )
    contexts: dict[str, MapContext] = {}
    loaded_map = None
    try:
        while trainer.episodes < args.episodes:
            stem = args.maps[(trainer.episodes // args.episodes_per_map) % len(args.maps)]
            ctx = contexts.get(stem) or contexts.setdefault(stem, build_context(stem, trainer, manifest))
            tmi.max_overall_duration_ms = int(ctx.max_race_s * 1000)
            if stem != loaded_map:  # the first rollout after a map switch loads the map; it isn't trained on
                print(f"[sync] loading {stem}", flush=True)
                ctx.policy.reset()
                run_rollout(tmi, ctx)
                loaded_map = stem
            ctx.policy.reset(trainer.episode_path(stem, ".csv"))
            rollout_results, end_race_stats = run_rollout(tmi, ctx)
            ctx.policy.close()
            run = summarize_run(rollout_results, end_race_stats, ctx.finish_zone_idx, tmi.last_rollout_crashed, ctx.max_race_s)
            run.update(ctx.policy.summary())
            if run["termination"] == "TMI_Disrupt":
                print("[train] game unreachable (is the desktop visible?); stopping. Rerun with --resume.")
                break
            stats = trainer.record_episode(stem, ctx.policy.step_records, run["termination"], run)
            print(
                f"ep {stats['episode']:>4} {stem:<15} {run['termination']:<16} {run['completion_pct']:6.2f}% "
                f"t={run['race_time_s']} ret={stats.get('episode_return')} |res|={stats.get('mean_abs_residual')} "
                f"actor={'on' if stats.get('actor_trained') else 'off'} critic={stats.get('critic_loss')} ({stats.get('train_s')} s)",
                flush=True,
            )
    finally:
        for ctx in contexts.values():
            ctx.policy.close()
        trainer.agent.save(out_dir / "latest.pt")
        print(f"[OK] {trainer.episodes} episodes; checkpoint {out_dir / 'latest.pt'}; log {out_dir / 'train_log.jsonl'}")
        if not args.keep_game:
            os.system("taskkill /F /IM TmForever.exe 2>nul")


if __name__ == "__main__":
    main()
