"""Drive TrackMania live with a hybrid (non-RL) controller.

Needs TrackMania + TMInterface on a visible desktop, TMLoader GUI closed.

Example:
    python scripts/run_hybrid_controller_live.py --track A01 --runs 3
    python scripts/run_hybrid_controller_live.py --track A01 --controller mpc --max-speed 40 --speed 10
    python scripts/run_hybrid_controller_live.py --map B07-Race --runs 1   # any campaign map, official line

Each run writes a per-step CSV (state, command, chosen key combination, impact
proxy) plus one JSON summary under eval_results/hybrid_live/<name>/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from multiprocessing import Lock
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from evaluate_checkpoints import TRACKS

from config_files import config_copy
from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineController,
    ConservativeControllerConfig,
    PurePursuitController,
    RecoveryAwareCenterlineController,
    RecoveryAwareCenterlineControllerConfig,
)
from trackmania_rl.hybrid_live import HybridLivePolicy, HybridLivePolicyConfig
from trackmania_rl.hybrid_mpc import ShortHorizonMPCConfig, ShortHorizonMPCController
from trackmania_rl.hybrid_state import build_centerline_geometry
from trackmania_rl.hybrid_track_profile import TrackProfileConfig, build_track_speed_profile
from trackmania_rl.speed_model import learned_base_speed, load_speed_model
from trackmania_rl.grip_model import load_grip_model
from trackmania_rl.hybrid_block_map import find_campaign_challenge, load_block_map
from trackmania_rl.range_sensor import FootprintModel, RangeSensor
from trackmania_rl.residual_action import ResidualConfig, ZeroResidualPolicy
from trackmania_rl.residual_reward import build_transitions
from trackmania_rl.residual_td3 import ActorResidualPolicy, TD3Agent
from trackmania_rl.map_loader import load_next_map_zone_centers
from trackmania_rl.tmi_interaction import game_instance_manager


def parse_speed_cap(text: str) -> tuple[float, float, float]:
    try:
        start_m, end_m, speed_mps = (float(part) for part in text.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected START:END:MPS, got {text!r}") from error
    if not (0.0 <= start_m < end_m and speed_mps > 0.0):
        raise argparse.ArgumentTypeError(f"need 0 <= START < END and MPS > 0, got {text!r}")
    return start_m, end_m, speed_mps


def make_controller(
    name: str,
    max_speed_mps: float | None,
    centerline: np.ndarray,
    speed_caps: tuple[tuple[float, float, float], ...] = (),
    speed_model: Path | None = None,
    speed_scale: float = 1.0,
    top_speed_mps: float | None = None,
    grip_model: Path | None = None,
    grip_risk: float = 0.2,
):
    if name == "purepursuit":
        profile_config = TrackProfileConfig(speed_caps=speed_caps)
        for cap in (top_speed_mps, max_speed_mps):  # top speed: global; max speed: legacy per-run override
            if cap is not None:
                profile_config = dataclasses.replace(profile_config, max_speed_mps=cap)
        geometry = build_centerline_geometry(centerline)
        base = None
        if speed_model is not None:
            base = learned_base_speed(geometry, load_speed_model(speed_model), speed_scale)
        grip_limit = None
        if grip_model is not None:
            fitted = load_grip_model(grip_model)
            grip_limit = lambda curvature: fitted.speed_limit_mps(curvature, grip_risk)  # noqa: E731
        profile = build_track_speed_profile(geometry, profile_config, base, grip_limit)
        return PurePursuitController(profile)
    if name == "mpc":
        config = ShortHorizonMPCConfig()
        if max_speed_mps is not None:
            planner = dataclasses.replace(config.planner_config, max_speed_mps=max_speed_mps)
            config = dataclasses.replace(config, planner_config=planner)
        return ShortHorizonMPCController(config)
    config_cls, controller_cls = {
        "conservative": (ConservativeControllerConfig, ConservativeCenterlineController),
        "recovery": (RecoveryAwareCenterlineControllerConfig, RecoveryAwareCenterlineController),
    }[name]
    config = config_cls() if max_speed_mps is None else config_cls(max_speed_mps=max_speed_mps)
    return controller_cls(config)


def make_range_sensor(footprints_path: Path, map_stem: str) -> RangeSensor:
    """Virtual lidar over the map's block layout, rasters prebuilt so no live step stalls."""

    challenge = find_campaign_challenge(map_stem)
    if challenge is None:
        raise SystemExit(f"--range-sensor needs the campaign map file for {map_stem!r} (not found)")
    sensor = RangeSensor(load_block_map(challenge), FootprintModel.load(footprints_path))
    sensor.prebuild()
    return sensor


def make_residual_policy(spec: str):
    """'none' -> base stack only; 'zero' -> zero residual (logs transitions); else a TD3 checkpoint, no exploration noise."""

    if spec == "none":
        return None
    if spec == "zero":
        return ZeroResidualPolicy()
    return ActorResidualPolicy(TD3Agent.load(Path(spec)))


# A run that ended within this margin of the race limit timed out; earlier ends were stuck cutoffs.
RACE_LIMIT_MARGIN_S = 10.0


def summarize_run(
    rollout_results: dict, end_race_stats: dict, finish_zone_idx: int, crashed: bool, max_race_s: float
) -> dict:
    furthest = int(rollout_results.get("furthest_zone_idx", 0))
    finished = bool(end_race_stats.get("race_finished", False))
    if finished:
        cause = "Finished"
    elif crashed:
        cause = "TMI_Disrupt"
    elif float(end_race_stats.get("race_time_for_ratio", 0.0)) >= (max_race_s - RACE_LIMIT_MARGIN_S) * 1000:
        cause = f"Timeout_{max_race_s:.0f}s"
    else:
        cause = "Stuck_Cutoff_2s"
    return {
        "finished": finished,
        "termination": cause,
        "race_time_s": float(end_race_stats.get("race_time", 0.0)) / 1000.0 if finished else None,
        "completion_pct": 100.0 if finished else round(min(100.0, furthest / finish_zone_idx * 100.0), 2),
        "actions": len(rollout_results.get("actions", [])),
    }


def campaign_meta(stem: str) -> dict:
    """Track metadata for any Nations campaign map installed under ``Official Maps``."""

    return {
        "name": stem,
        "map_path": rf'"Official Maps\{stem}.Challenge.Gbx"',
        "cl_file": f"{stem}_official_0.5m_cl.npy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", default="A01", choices=sorted(TRACKS))
    parser.add_argument("--map", default=None, help="any campaign map stem, e.g. B07-Race (overrides --track)")
    parser.add_argument(
        "--max-race-s",
        type=float,
        default=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms / 1000,
        help="give up a race after this long (endurance maps need more than the default)",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--controller", default="purepursuit", choices=["purepursuit", "conservative", "recovery", "mpc"])
    parser.add_argument("--max-speed", type=float, default=None, help="override controller max speed (m/s)")
    parser.add_argument(
        "--stuck-cutoff-s",
        type=float,
        default=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms / 1000,
        help="end a run after this long without new progress (training default 2 s; reversing needs more)",
    )
    parser.add_argument("--no-unstick", action="store_true", help="disable the reverse-and-retry maneuver")
    parser.add_argument("--speed", type=float, default=config_copy.running_speed, help="game simulation speed multiplier")
    parser.add_argument("--port", type=int, default=config_copy.base_tmi_port)
    parser.add_argument("--name", default=None, help="output folder name (must not exist)")
    parser.add_argument("--keep-game", action="store_true", help="do not kill TmForever at the end")
    parser.add_argument(
        "--centerline",
        default=None,
        help="centerline file under maps/ to drive instead of the track default (e.g. A05-Race_official_0.5m_cl.npy)",
    )
    parser.add_argument("--speed-model", type=Path, default=None, help="learned speed model (.pt) for the target profile")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="global safety scale on the learned speed")
    parser.add_argument("--grip-model", type=Path, default=None, help="fitted grip model (.json) capping corner speeds")
    parser.add_argument("--grip-risk", type=float, default=0.2, help="global run-wide risk the grip cap allows (0-1)")
    parser.add_argument(
        "--range-sensor", type=Path, default=None, help="learned obstacle shapes (.npz): log virtual-lidar rays every step"
    )
    parser.add_argument("--top-speed", type=float, default=None, help="global top-speed limit of the profile (m/s)")
    parser.add_argument(
        "--residual",
        default="none",
        help="Step 4 residual policy on top of pure pursuit: 'none', 'zero' (must reproduce the base run exactly) "
        "or a TD3 checkpoint .pt (deterministic actor); any residual also records RL transitions (runXX_rl.npz)",
    )
    parser.add_argument(
        "--speed-cap",
        action="append",
        default=[],
        type=parse_speed_cap,
        metavar="START:END:MPS",
        help="pace note: cap the target speed between two progress marks (repeatable)",
    )
    args = parser.parse_args()
    if not 0.0 < args.grip_risk < 1.0:
        parser.error("--grip-risk must be in (0, 1)")
    if args.range_sensor is not None and not args.range_sensor.is_file():
        parser.error(f"{args.range_sensor} not found (run scripts/build_range_sensor.py)")
    if args.grip_model is not None and not args.grip_model.is_file():
        parser.error(f"{args.grip_model} not found (run scripts/fit_grip_model.py)")

    meta = campaign_meta(args.map) if args.map else dict(TRACKS[args.track])
    if not (BASE_DIR / "maps" / meta["cl_file"]).is_file() and args.centerline is None:
        parser.error(f"maps/{meta['cl_file']} not found (run scripts/build_official_centerlines.py --install)")
    if args.centerline is not None:
        if not (BASE_DIR / "maps" / args.centerline).is_file():
            parser.error(f"maps/{args.centerline} not found")
        meta["cl_file"] = args.centerline
    name = args.name or f"{args.map or args.track}_{args.controller}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = BASE_DIR / "eval_results" / "hybrid_live" / name
    out_dir.mkdir(parents=True, exist_ok=False)

    centerline = np.load(BASE_DIR / "maps" / meta["cl_file"])
    zone_centers = load_next_map_zone_centers(meta["cl_file"], BASE_DIR)
    finish_zone_idx = len(zone_centers) - config_copy.n_zone_centers_extrapolate_after_end_of_map
    controller = make_controller(
        args.controller, args.max_speed, centerline, tuple(args.speed_cap), args.speed_model, args.speed_scale, args.top_speed,
        args.grip_model, args.grip_risk,
    )
    if args.residual != "none" and args.controller != "purepursuit":
        parser.error("--residual needs --controller purepursuit")
    if args.residual not in ("none", "zero") and not Path(args.residual).is_file():
        parser.error(f"--residual: {args.residual} is not 'none', 'zero' or an existing checkpoint")
    residual_config = ResidualConfig()
    policy = HybridLivePolicy(
        centerline,
        controller,
        config=HybridLivePolicyConfig(unstick_enabled=not args.no_unstick),
        range_sensor=make_range_sensor(args.range_sensor, args.map or meta["name"]) if args.range_sensor else None,
        residual_policy=make_residual_policy(args.residual),
        residual_config=residual_config,
    )

    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=Lock(),
        running_speed=args.speed,
        run_steps_per_action=config_copy.tm_engine_step_per_action,
        max_overall_duration_ms=int(args.max_race_s * 1000),
        max_minirace_duration_ms=int(args.stuck_cutoff_s * 1000),
        tmi_port=args.port,
    )

    def rollout():
        return tmi.rollout(exploration_policy=policy, map_path=meta["map_path"], zone_centers=zone_centers, update_network=lambda: None)

    summary = {
        "track": args.map or args.track,
        "controller": args.controller,
        "speed_model": str(args.speed_model) if args.speed_model else None,
        "speed_scale": args.speed_scale,
        "top_speed_mps": args.top_speed,
        "gravity_mps2": TrackProfileConfig().gravity_mps2,
        "range_sensor": str(args.range_sensor) if args.range_sensor else None,
        "residual": args.residual,
        "residual_config": dataclasses.asdict(residual_config) if args.residual != "none" else None,
        "grip_model": str(args.grip_model) if args.grip_model else None,
        "grip_risk": args.grip_risk if args.grip_model else None,
        "centerline": meta["cl_file"],
        "max_speed_override_mps": args.max_speed,
        "speed_caps": [list(cap) for cap in args.speed_cap],
        "game_speed": args.speed,
        "stuck_cutoff_s": args.stuck_cutoff_s,
        "unstick_enabled": not args.no_unstick,
        "jumps": [dataclasses.asdict(j) for j in getattr(getattr(controller, "profile", None), "jumps", ())],
        "runs": [],
    }
    try:
        print(f"[sync] loading {meta['name']} and capturing start state")
        policy.reset()
        rollout()
        for run_idx in range(1, args.runs + 1):
            policy.reset(out_dir / f"run{run_idx:02d}.csv")
            rollout_results, end_race_stats = rollout()
            policy.close()
            run = summarize_run(rollout_results, end_race_stats, finish_zone_idx, tmi.last_rollout_crashed, args.max_race_s)
            run.update(policy.summary(), run=run_idx)
            if args.residual != "none":
                transitions = build_transitions(policy.step_records, run["termination"])
                if transitions is not None:
                    np.savez_compressed(out_dir / f"run{run_idx:02d}_rl.npz", **transitions.as_dict())
                    run["rl_transitions"] = int(len(transitions.rewards))
                    run["rl_return"] = round(float(transitions.rewards.sum()), 3)
            summary["runs"].append(run)
            print(
                f"run {run_idx}/{args.runs}: {run['termination']:<15} {run['completion_pct']:6.2f}% "
                f"time={run['race_time_s']} impacts={run['impact_suspects']} recovery_steps={run['recovery_steps']} reversals={run['reversals']}"
            )
    finally:
        policy.close()
        runs = summary["runs"]
        summary["finishes"] = f"{sum(r['finished'] for r in runs)}/{len(runs)}"
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[OK] wrote {out_dir}")
        if not args.keep_game:
            os.system("taskkill /F /IM TmForever.exe 2>nul")


if __name__ == "__main__":
    main()
