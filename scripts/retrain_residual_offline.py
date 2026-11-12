"""Offline TD3 on already-collected residual data, plus diagnostics (Step 4, HANDOVER §79).

Continues training a checkpoint (critic + actor) on a fixed buffer: the zero-residual seed runs
plus a live training run's episodes. No game. Used to try a learner change (e.g. the anchor off)
before spending live hours on it. Reports, per map, what the deterministic actor now does:

* mean / mean-absolute residual and the share of saturated outputs (|a| > 0.9: a sign that the
  actor exploits critic extrapolation outside the data rather than a real effect);
* the critic's gain Q(s, π(s)) − Q(s, 0) (its own opinion, not a live result);
* whether the corrections make physical sense: correlation of Δsteering with the lateral and
  heading error, and the mean Δtarget speed within 50 m before a jump vs elsewhere.

    python scripts/retrain_residual_offline.py --from eval_results/residual_rl/td3_v1/latest.pt \
        --episodes-dir eval_results/residual_rl/td3_v1/episodes --anchor none --out eval_results/residual_rl/td3_v1_offline_noanchor
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trackmania_rl.residual_observation import OBSERVATION_NAMES
from trackmania_rl.residual_td3 import ReplayBuffer, TD3Agent

LIVE_DIR = ROOT / "eval_results" / "hybrid_live"
EPISODE_MAP = re.compile(r"ep\d{5}_(.+)\.npz$")
SEED_MAP = re.compile(r"(.+)_rl_train_")
NEAR_JUMP = 50.0 / 200.0  # distance_to_next_jump feature is distance / 200 m
SATURATED = 0.9


def load_by_map(seed_glob: str, episodes_dir: Path) -> dict[str, list[dict[str, np.ndarray]]]:
    groups: dict[str, list] = {}
    sources = [(p, SEED_MAP.match(p.parent.name)) for p in sorted(LIVE_DIR.glob(f"{seed_glob}/run01_rl.npz"))]
    sources += [(p, EPISODE_MAP.search(p.name)) for p in sorted(episodes_dir.glob("ep*.npz"))]
    for path, match in sources:
        if match is None:
            continue
        with np.load(path) as data:
            groups.setdefault(match.group(1), []).append({k: data[k] for k in data.files})
    return groups


@torch.no_grad()
def diagnose(agent: TD3Agent, runs: list[dict[str, np.ndarray]]) -> dict:
    obs = np.concatenate([r["observations"][r["residual_active"]] for r in runs])
    o = torch.as_tensor(obs)
    a = agent.actor(o)
    gain = torch.min(*agent.critic(o, a)) - torch.min(*agent.critic(o, torch.zeros_like(a)))
    a = a.numpy()
    col = {name: obs[:, i] for i, name in enumerate(OBSERVATION_NAMES)}
    near_jump = col["distance_to_next_jump"] < NEAR_JUMP
    return {
        "states": len(obs),
        "mean_action": a.mean(0).round(3).tolist(),
        "mean_abs_action": np.abs(a).mean(0).round(3).tolist(),
        "saturated_share": (np.abs(a) > SATURATED).mean(0).round(3).tolist(),
        "q_gain_mean": round(float(gain.mean()), 4),
        "steer_vs_lateral_r": round(float(np.corrcoef(a[:, 0], col["lateral_error"])[0, 1]), 3),
        "steer_vs_sin_heading_r": round(float(np.corrcoef(a[:, 0], col["sin_heading_error"])[0, 1]), 3),
        "dtarget_near_jump": round(float(a[near_jump, 1].mean()), 3) if near_jump.any() else None,
        "dtarget_elsewhere": round(float(a[~near_jump, 1].mean()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", type=Path, required=True, help="checkpoint to continue from")
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--seed-data", default="*_rl_train_v7zero")
    parser.add_argument("--anchor", default="none", help="'none' (plain TD3) or the TD3+BC alpha, e.g. 2.5")
    parser.add_argument("--updates", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = TD3Agent.load(args.start)
    anchor = None if args.anchor == "none" else float(args.anchor)
    agent = TD3Agent.load(args.start, config=dataclasses.replace(base.config, anchor_alpha=anchor))
    groups = load_by_map(args.seed_data, args.episodes_dir)
    buffer = ReplayBuffer(capacity=sum(len(r["rewards"]) for runs in groups.values() for r in runs))
    for runs in groups.values():
        for run in runs:
            buffer.add(run)
    print(f"buffer {buffer.size} transitions, maps {sorted(groups)}; anchor {anchor}; from {args.start}", flush=True)

    report = {"from": str(args.start), "anchor_alpha": anchor, "updates": args.updates, "buffer": buffer.size,
              "before": {m: diagnose(base, runs) for m, runs in groups.items()}}
    rng = np.random.default_rng(args.seed)
    log_every = max(args.updates // 5, 1)
    for step in range(1, args.updates + 1):
        stats = agent.update(buffer.sample(agent.config.batch_size, rng))
        if step % log_every == 0:
            print(f"  update {step:>6}: {stats}", flush=True)
    report["after"] = {m: diagnose(agent, runs) for m, runs in groups.items()}
    for m in sorted(groups):
        print(f"{m:<15} before {report['before'][m]['mean_abs_action']} -> after {report['after'][m]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.out.with_suffix(".pt"))
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"[OK] wrote {args.out.with_suffix('.pt')} and .json")


if __name__ == "__main__":
    main()
