"""TD3 learner for the residual policy (Step 4).

TD3 (Fujimoto et al. 2018) is an off-policy actor-critic for continuous actions:
twin critics (take the smaller Q to curb overestimation), a deterministic actor
updated every ``policy_delay`` critic steps, and noise on the target action
(target policy smoothing). Three choices make it a *residual* learner:

* **Zero-initialised actor output layer.** ``tanh(0) = 0``, so before any update the
  policy is exactly the base stack (verified live: zero residual = v7a, HANDOVER §73).
* **Anchor to the base command** (TD3+BC form, Fujimoto & Gu 2021, with the base
  stack's zero residual as the anchor): ``actor loss = -λ·Q(s, π(s)) + |π(s)|²`` with
  ``λ = alpha / mean|Q|``. The policy moves away from the base command only where the
  critic says it pays, and the scale of Q doesn't change that balance.
  ``anchor_alpha=None`` turns it off (plain TD3; the reward's |a|² cost still prefers small
  corrections). td3_v1 showed λ = 2.5/|Q| is far too strong here (HANDOVER §78): Q sums ~100
  steps of reward while one 50 ms correction moves it a little, so the actor stayed at |a| ≈ 0.02.
* **Shield-aware targets.** Where a safety layer owns the next state (recovery, slip
  guard, reverse) the residual isn't applied, so the TD target uses ``a' = 0`` there;
  and the actor is trained only on states where its output is actually applied.

The critic can be trained alone first (``update(..., train_actor=False)``): a burn-in
on zero-residual runs, so the first actor gradients come from a sensible Q.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from trackmania_rl.residual_action import RESIDUAL_SIZE
from trackmania_rl.residual_observation import OBSERVATION_SIZE
from trackmania_rl.residual_reward import Transitions

CHECKPOINT_MODULES = ("actor", "critic", "actor_target", "critic_target", "actor_optimizer", "critic_optimizer")
TRANSITION_KEYS = (
    "observations", "actions", "rewards", "next_observations", "terminals", "residual_active", "next_residual_active",
)


@dataclass(frozen=True)
class TD3Config:
    hidden_units: int = 256
    gamma: float = 0.99  # 50 ms steps: a ~5 s horizon
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    policy_delay: int = 2
    target_noise: float = 0.2
    target_noise_clip: float = 0.5
    anchor_alpha: float | None = 2.5  # None = no anchor (plain TD3)
    max_grad_norm: float = 10.0


def mlp(inputs: int, outputs: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


class Actor(nn.Module):
    """obs -> residual action in [-1, 1]^2; outputs exactly 0 until trained."""

    def __init__(self, obs_size: int = OBSERVATION_SIZE, hidden: int = 256):
        super().__init__()
        self.net = mlp(obs_size, RESIDUAL_SIZE, hidden)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class TwinCritic(nn.Module):
    def __init__(self, obs_size: int = OBSERVATION_SIZE, hidden: int = 256):
        super().__init__()
        self.q1 = mlp(obs_size + RESIDUAL_SIZE, 1, hidden)
        self.q2 = mlp(obs_size + RESIDUAL_SIZE, 1, hidden)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReplayBuffer:
    """Fixed-size ring buffer of transitions (numpy, CPU)."""

    def __init__(self, capacity: int, obs_size: int = OBSERVATION_SIZE):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity, self.obs_size = capacity, obs_size
        self._data = {
            "observations": np.zeros((capacity, obs_size), np.float32),
            "actions": np.zeros((capacity, RESIDUAL_SIZE), np.float32),
            "rewards": np.zeros(capacity, np.float32),
            "next_observations": np.zeros((capacity, obs_size), np.float32),
            "terminals": np.zeros(capacity, bool),
            "residual_active": np.zeros(capacity, bool),
            "next_residual_active": np.zeros(capacity, bool),
        }
        self._next = 0
        self.size = 0

    def add(self, transitions: Transitions | dict[str, np.ndarray]) -> None:
        batch = transitions.as_dict() if isinstance(transitions, Transitions) else transitions
        missing = [k for k in TRANSITION_KEYS if k not in batch]
        if missing:
            raise KeyError(f"transitions lack {missing} (re-record with the current runner)")
        n = len(batch["rewards"])
        if batch["observations"].shape != (n, self.obs_size):
            raise ValueError(f"expected observations ({n}, {self.obs_size}), got {batch['observations'].shape}")
        idx = (self._next + np.arange(n)) % self.capacity
        for key in TRANSITION_KEYS:
            self._data[key][idx] = batch[key]
        self._next = int((self._next + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def add_npz(self, path: Path) -> int:
        with np.load(path) as data:
            batch = {k: data[k] for k in data.files}
        self.add(batch)
        return len(batch["rewards"])

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = rng.integers(0, self.size, size=batch_size)
        return {k: torch.as_tensor(v[idx]) for k, v in self._data.items()}


class TD3Agent:
    def __init__(self, config: TD3Config | None = None, obs_size: int = OBSERVATION_SIZE, seed: int = 0):
        self.config = config or TD3Config()
        self.obs_size = obs_size
        torch.manual_seed(seed)
        self.actor = Actor(obs_size, self.config.hidden_units)
        self.critic = TwinCritic(obs_size, self.config.hidden_units)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.critic_updates = 0

    # --- acting ------------------------------------------------------------
    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(observation, np.float32)).unsqueeze(0)
        return self.actor(obs).squeeze(0).numpy()

    @torch.no_grad()
    def target_next_actions(self, next_obs: torch.Tensor, next_active: torch.Tensor) -> torch.Tensor:
        """Smoothed target action a'; zero where a safety layer owns s' (the residual isn't applied there)."""

        cfg = self.config
        action = self.actor_target(next_obs)
        noise = (torch.randn_like(action) * cfg.target_noise).clamp(-cfg.target_noise_clip, cfg.target_noise_clip)
        action = (action + noise).clamp(-1.0, 1.0)
        return action * next_active.unsqueeze(-1).to(action.dtype)

    # --- learning -----------------------------------------------------------
    def update(self, batch: dict[str, torch.Tensor], train_actor: bool = True) -> dict[str, float]:
        cfg = self.config
        obs, actions, rewards = batch["observations"], batch["actions"], batch["rewards"]
        with torch.no_grad():
            next_actions = self.target_next_actions(batch["next_observations"], batch["next_residual_active"])
            q1_next, q2_next = self.critic_target(batch["next_observations"], next_actions)
            not_done = 1.0 - batch["terminals"].float()
            target = rewards + cfg.gamma * not_done * torch.min(q1_next, q2_next)
        q1, q2 = self.critic(obs, actions)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self._step(self.critic_optimizer, critic_loss, self.critic)
        self.critic_updates += 1
        stats = {"critic_loss": float(critic_loss), "q_mean": float(q1.mean())}

        if train_actor and self.critic_updates % cfg.policy_delay == 0:
            stats.update(self._update_actor(obs, batch["residual_active"]))
        self._soft_update(self.critic, self.critic_target)
        return stats

    def _update_actor(self, obs: torch.Tensor, active: torch.Tensor) -> dict[str, float]:
        if not bool(active.any()):
            return {}
        obs = obs[active]
        action = self.actor(obs)
        q = self.critic.q1(torch.cat([obs, action], dim=-1)).squeeze(-1)
        anchor = action.pow(2).sum(-1).mean()
        if self.config.anchor_alpha is None:
            actor_loss = -q.mean()
        else:
            lam = self.config.anchor_alpha / q.abs().mean().detach().clamp_min(1e-6)
            actor_loss = -lam * q.mean() + anchor
        self._step(self.actor_optimizer, actor_loss, self.actor)
        self._soft_update(self.actor, self.actor_target)
        return {"actor_loss": float(actor_loss), "residual_l2": float(anchor)}

    def _step(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor, module: nn.Module) -> None:
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite TD3 loss {float(loss)}")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(module.parameters(), self.config.max_grad_norm)
        optimizer.step()

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for p, tp in zip(online.parameters(), target.parameters()):
                tp.mul_(1.0 - tau).add_(p, alpha=tau)

    # --- checkpoints ---------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "obs_size": self.obs_size,
                "critic_updates": self.critic_updates,
                **{name: getattr(self, name).state_dict() for name in CHECKPOINT_MODULES},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, config: TD3Config | None = None) -> TD3Agent:
        """Restore a checkpoint; ``config`` overrides the saved hyperparameters (e.g. burn-in -> training noise)."""

        payload = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(config or TD3Config(**payload["config"]), payload["obs_size"])
        for name in CHECKPOINT_MODULES:
            if name in payload:  # optimiser states are absent from older checkpoints
                getattr(agent, name).load_state_dict(payload[name])
        agent.critic_updates = int(payload["critic_updates"])
        return agent


def discounted_returns(rewards: np.ndarray, terminals: np.ndarray, gamma: float) -> np.ndarray:
    """Monte-Carlo return G_t = r_t + γ·G_{t+1}, restarting after each terminal.

    For one episode recorded in order this is the value the critic should learn for the
    policy that produced it (the base stack is deterministic, so G_t = V(s_t) exactly).
    A trailing non-terminal step (a timeout) is treated as if the episode ended there.
    """

    returns = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        if terminals[t]:
            running = 0.0
        running = float(rewards[t]) + gamma * running
        returns[t] = running
    return returns


class ActorResidualPolicy:
    """``ResidualPolicy`` for ``HybridLivePolicy``: the actor, plus Gaussian exploration noise when collecting.

    ``HybridLivePolicy`` only calls ``act`` on steps where the residual is applied, so
    noise is only ever added where it changes what the car does.

    ``noise_hold_steps``: one noise draw is held for that many calls (temporally correlated
    exploration). White noise redrawn every 50 ms mostly averages out in the sigma-delta key
    adapter, so the critic saw corrections that barely changed anything (HANDOVER §79);
    a perturbation held for ~0.5 s actually moves the line or the speed into a jump.
    """

    def __init__(self, agent: TD3Agent, noise_std: float = 0.0, seed: int = 0, noise_hold_steps: int = 1):
        if noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        if noise_hold_steps < 1:
            raise ValueError("noise_hold_steps must be >= 1")
        self.agent = agent
        self.noise_std = noise_std
        self.noise_hold_steps = noise_hold_steps
        self._rng = np.random.default_rng(seed)
        self._noise = np.zeros(2)
        self._calls = 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        action = self.agent.act(observation)
        if self.noise_std > 0.0:
            if self._calls % self.noise_hold_steps == 0:
                self._noise = self._rng.normal(0.0, self.noise_std, size=action.shape)
            self._calls += 1
            action = action + self._noise
        return np.clip(action, -1.0, 1.0).astype(np.float32)
