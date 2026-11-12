"""Training loop core for the residual TD3 policy (Step 4), independent of the game.

One episode = one live run on an ``rl_train`` map with the actor + exploration noise as the
residual. After each episode the trainer:

1. turns the run's step records into transitions (``residual_reward.build_transitions``),
   saves them (``episodes/epNNNNN_<map>.npz``) and adds them to the replay buffer;
2. does ``updates_per_transition`` TD3 updates per new transition (capped per episode);
   the actor only trains once ``actor_warmup_transitions`` explored steps are in the buffer,
   so the critic has seen varied actions first (burn-in data holds only a = 0);
3. appends one JSON line of statistics to ``train_log.jsonl`` and checkpoints the agent.

Everything is resumable from the output folder (episode files + ``latest.pt``).
The game side lives in ``scripts/train_residual.py``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from trackmania_rl.residual_reward import StepRecord, build_transitions
from trackmania_rl.residual_td3 import ActorResidualPolicy, ReplayBuffer, TD3Agent

EPISODE_FILE = re.compile(r"ep(\d{5})_.+\.npz$")


@dataclass(frozen=True)
class TrainingConfig:
    exploration_noise: float = 0.3  # std on the [-1, 1] output: ±0.075 steering, ±2.4 m/s target, 1 sigma
    noise_hold_steps: int = 1  # one noise draw per this many 50 ms steps (10 = 0.5 s)
    updates_per_transition: float = 1.0
    max_updates_per_episode: int = 5000
    actor_warmup_transitions: int = 5000
    checkpoint_every_episodes: int = 1
    buffer_capacity: int = 500_000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.exploration_noise < 0.0 or self.updates_per_transition < 0.0:
            raise ValueError("exploration_noise and updates_per_transition must be >= 0")
        if self.noise_hold_steps < 1:
            raise ValueError("noise_hold_steps must be >= 1")


class ResidualTrainer:
    def __init__(self, agent: TD3Agent, config: TrainingConfig, out_dir: Path):
        self.agent = agent
        self.config = config
        self.out_dir = Path(out_dir)
        self.episode_dir = self.out_dir / "episodes"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = ReplayBuffer(config.buffer_capacity, agent.obs_size)
        self.rng = np.random.default_rng(config.seed)
        # The live policy calls this on every step where the residual is applied.
        self.policy = ActorResidualPolicy(
            agent, noise_std=config.exploration_noise, seed=config.seed, noise_hold_steps=config.noise_hold_steps
        )
        self.episodes = 0
        self.explored_transitions = 0

    # --- data -------------------------------------------------------------------
    def add_seed_data(self, paths: list[Path], explored: bool = False) -> int:
        """Preload transitions. Zero-residual runs (``explored=False``) don't count toward the actor
        warm-up; episodes of an earlier training run (``explored=True``) do."""

        total = 0
        for path in paths:
            total += self.buffer.add_npz(path)
            if explored:
                with np.load(path) as data:
                    self.explored_transitions += int(data["residual_active"].sum())
        return total

    def resume(self) -> int:
        """Reload this run's own episodes into the buffer; returns how many were found."""

        files = sorted(p for p in self.episode_dir.glob("ep*.npz") if EPISODE_FILE.search(p.name))
        for path in files:
            self.buffer.add_npz(path)
            with np.load(path) as data:
                self.explored_transitions += int(data["residual_active"].sum())
        self.episodes = len(files)
        return self.episodes

    def episode_path(self, map_name: str, suffix: str) -> Path:
        return self.episode_dir / f"ep{self.episodes + 1:05d}_{map_name}{suffix}"

    # --- one episode ---------------------------------------------------------------
    def record_episode(self, map_name: str, records: list[StepRecord], termination: str, run_summary: dict) -> dict:
        start = time.perf_counter()
        stats = {"episode": self.episodes + 1, "map": map_name, "termination": termination, **_run_fields(run_summary)}
        transitions = build_transitions(records, termination)
        if transitions is None:
            stats["skipped"] = "no usable transitions"
            self._log(stats)
            return stats

        np.savez_compressed(self.episode_path(map_name, ".npz"), **transitions.as_dict())
        self.buffer.add(transitions)
        active = transitions.residual_active
        self.explored_transitions += int(active.sum())
        stats.update(
            transitions=len(transitions.rewards),
            episode_return=round(float(transitions.rewards.sum()), 3),
            mean_abs_residual=[round(float(v), 4) for v in np.abs(transitions.actions[active]).mean(axis=0)]
            if active.any()
            else [0.0, 0.0],
        )
        stats.update(self._train(len(transitions.rewards)))
        self.episodes += 1
        if self.episodes % self.config.checkpoint_every_episodes == 0:
            self.agent.save(self.out_dir / "latest.pt")
        stats["train_s"] = round(time.perf_counter() - start, 1)
        self._log(stats)
        return stats

    def _train(self, new_transitions: int) -> dict:
        cfg = self.config
        n_updates = min(cfg.max_updates_per_episode, round(cfg.updates_per_transition * new_transitions))
        train_actor = self.explored_transitions >= cfg.actor_warmup_transitions
        critic_losses, actor_losses = [], []
        for _ in range(n_updates):
            step = self.agent.update(self.buffer.sample(self.agent.config.batch_size, self.rng), train_actor=train_actor)
            critic_losses.append(step["critic_loss"])
            if "actor_loss" in step:
                actor_losses.append(step["actor_loss"])
        return {
            "updates": n_updates,
            "actor_trained": train_actor,
            "explored_transitions": self.explored_transitions,
            "buffer_size": self.buffer.size,
            "critic_loss": round(float(np.mean(critic_losses)), 4) if critic_losses else None,
            "actor_loss": round(float(np.mean(actor_losses)), 4) if actor_losses else None,
        }

    def _log(self, stats: dict) -> None:
        with open(self.out_dir / "train_log.jsonl", "a", encoding="utf-8") as log:
            log.write(json.dumps(stats) + "\n")

    def write_config(self, extra: dict) -> None:
        payload = {"training": asdict(self.config), "td3": asdict(self.agent.config), **extra}
        (self.out_dir / "train_config.json").write_text(json.dumps(payload, indent=2))


def illegal_training_maps(maps: list[str], rl_train: list[str]) -> list[str]:
    """Maps that may not be trained on: anything outside the fixed rl_train split (dev = choices, test = scored once)."""

    return [m for m in maps if m not in rl_train]


def _run_fields(run_summary: dict) -> dict:
    keys = ("completion_pct", "race_time_s", "steps", "impact_suspects", "recovery_steps", "reversals", "residual_active_steps")
    return {k: run_summary[k] for k in keys if k in run_summary}
