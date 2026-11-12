"""Critic burn-in for the residual TD3 agent (Step 4, HANDOVER §76).

Trains only the twin critics (the actor stays zero = the base stack) on zero-residual
runs, so the first actor gradients later come from a sensible Q. On this data the
critic should learn the base stack's value: Q(s, 0) ~ the discounted return G_t
(the base stack is deterministic, so G_t is exact). Checks reported per map:
Pearson r and mean absolute error between Q and G_t.

``--lomo`` also runs leave-one-map-out: train on 5 maps, score the held-out one, which
says whether the value function generalises to a map it hasn't seen.

Usage (repo root):
    python scripts/burn_in_residual_critic.py --runs-glob "*_rl_train_v7zero" --out eval_results/residual_rl/critic_burnin_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trackmania_rl.residual_td3 import ReplayBuffer, TD3Agent, TD3Config, discounted_returns

LIVE_DIR = ROOT / "eval_results" / "hybrid_live"


def load_runs(pattern: str) -> dict[str, dict[str, np.ndarray]]:
    runs = {}
    for npz in sorted(LIVE_DIR.glob(f"{pattern}/run01_rl.npz")):
        with np.load(npz) as data:
            runs[npz.parent.name] = {k: data[k] for k in data.files}
    if not runs:
        raise SystemExit(f"no run01_rl.npz under {LIVE_DIR}/{pattern}")
    return runs


@torch.no_grad()
def score(agent: TD3Agent, run: dict[str, np.ndarray]) -> dict[str, float]:
    """Q(s, a_logged) against the Monte-Carlo return of the logged episode."""

    q1, q2 = agent.critic(torch.as_tensor(run["observations"]), torch.as_tensor(run["actions"]))
    q = torch.min(q1, q2).numpy()
    g = discounted_returns(run["rewards"], run["terminals"], agent.config.gamma)
    return {
        "pearson_r": round(float(np.corrcoef(q, g)[0, 1]), 3),
        "mae": round(float(np.abs(q - g).mean()), 3),
        "bias": round(float((q - g).mean()), 3),
        "return_mean": round(float(g.mean()), 3),
        "return_std": round(float(g.std()), 3),
    }


def train_critic(
    runs: dict, names: list[str], updates: int, seed: int, config: TD3Config, log_every: int = 0, log=None
) -> TD3Agent:
    agent = TD3Agent(config, seed=seed)
    buffer = ReplayBuffer(capacity=sum(len(runs[n]["rewards"]) for n in names))
    for name in names:
        buffer.add(runs[name])
    rng = np.random.default_rng(seed)
    for step in range(1, updates + 1):
        stats = agent.update(buffer.sample(agent.config.batch_size, rng), train_actor=False)
        if log_every and step % log_every == 0:
            fit = {name: score(agent, runs[name])["pearson_r"] for name in names}
            print(f"  update {step:>6}: critic loss {stats['critic_loss']:.4f}  q mean {stats['q_mean']:.2f}  r per map {fit}", flush=True)
            if log is not None:
                log.append({"update": step, **stats, "pearson_r": fit})
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-glob", default="*_rl_train_v7zero")
    parser.add_argument("--updates", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lomo", action="store_true", help="also run leave-one-map-out")
    parser.add_argument(
        "--target-noise", type=float, default=0.0,
        help="TD3 target smoothing noise during burn-in (0: evaluate the deterministic base stack exactly)",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "eval_results" / "residual_rl" / "critic_burnin_v1")
    args = parser.parse_args()

    runs = load_runs(args.runs_glob)
    names = list(runs)
    print(f"{len(names)} runs, {sum(len(r['rewards']) for r in runs.values())} transitions: {names}")
    config = TD3Config(target_noise=args.target_noise)
    report: dict = {"runs": names, "updates": args.updates, "seed": args.seed, "config": config.__dict__}

    start = time.perf_counter()
    curve: list = []
    agent = train_critic(runs, names, args.updates, args.seed, config, log_every=max(args.updates // 6, 1), log=curve)
    report["train_fit"] = {name: score(agent, runs[name]) for name in names}
    report["curve"] = curve
    print(f"fit on all maps ({time.perf_counter() - start:.0f} s):")
    for name, s in report["train_fit"].items():
        print(f"  {name:<36} {s}")

    if args.lomo:
        report["lomo"] = {}
        for held_out in names:
            held_agent = train_critic(runs, [n for n in names if n != held_out], args.updates, args.seed, config)
            report["lomo"][held_out] = score(held_agent, runs[held_out])
            print(f"  LOMO {held_out:<31} {report['lomo'][held_out]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.out.with_suffix(".pt"))
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"[OK] wrote {args.out.with_suffix('.pt')} and .json")


if __name__ == "__main__":
    main()
