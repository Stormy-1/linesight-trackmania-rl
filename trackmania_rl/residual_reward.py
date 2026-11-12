"""Reward and transitions for the residual RL policy (Step 4).

    r_t = progress_t                       metres gained along the line (clipped), x 0.1
          - impact_penalty  * impact_{t+1} speed drop > 4 m/s in 50 ms (wall or hard landing)
          - recovery_penalty* recovery_{t+1}  the base stack had to take over (off line)
          - reverse_penalty * reversing_{t+1} the unstick manoeuvre is backing up
          - residual_penalty* |a_t|^2      prefer the base command unless a change pays

and at the end of the episode: ``- stuck_penalty`` when the run was cut off for
making no progress. Contacts are a quality metric, not a failure, so an impact
costs about as much as 4 m of progress; getting stuck costs about 100 m.
A timeout is a truncation (the value of the last state is still bootstrapped);
a TMI disruption is an infrastructure failure, so the episode is discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FINISHED = "Finished"
STUCK = "Stuck_Cutoff_2s"
TMI_DISRUPT = "TMI_Disrupt"


@dataclass(frozen=True)
class RewardConfig:
    progress_reward_per_m: float = 0.1
    # A 50 ms step can't gain more than ~6 m (120 m/s); bigger jumps are map-matching hops.
    max_progress_per_step_m: float = 6.0
    max_regress_per_step_m: float = 6.0
    impact_penalty: float = 0.4
    recovery_penalty: float = 0.05
    reverse_penalty: float = 0.05
    residual_penalty: float = 0.05
    stuck_penalty: float = 10.0


@dataclass(frozen=True)
class StepRecord:
    """What the live policy saw and did at one decision step."""

    observation: np.ndarray
    action: np.ndarray  # policy output in [-1, 1]^2 (zero when the residual was gated off)
    residual_active: bool
    progress_m: float
    impact: bool
    recovery: bool
    reversing: bool


@dataclass(frozen=True)
class Transitions:
    observations: np.ndarray  # (T, D)
    actions: np.ndarray  # (T, 2)
    rewards: np.ndarray  # (T,)
    next_observations: np.ndarray  # (T, D)
    terminals: np.ndarray  # (T,) no bootstrap after this step
    residual_active: np.ndarray  # (T,) the actor's output was applied
    # (T,) the actor will act in s' (False: a safety layer owns s', so the TD target uses a' = 0)
    next_residual_active: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "terminals": self.terminals,
            "residual_active": self.residual_active,
            "next_residual_active": self.next_residual_active,
        }


def step_rewards(records: list[StepRecord], config: RewardConfig) -> np.ndarray:
    """Reward of the transition from step t to step t+1, for t = 0 .. T-2."""

    progress = np.array([r.progress_m for r in records])
    gain = np.clip(np.diff(progress), -config.max_regress_per_step_m, config.max_progress_per_step_m)
    after = records[1:]
    impacts = np.array([r.impact for r in after], dtype=np.float64)
    recovery = np.array([r.recovery for r in after], dtype=np.float64)
    reversing = np.array([r.reversing for r in after], dtype=np.float64)
    effort = np.array([float(np.sum(np.square(r.action))) for r in records[:-1]])
    return (
        config.progress_reward_per_m * gain
        - config.impact_penalty * impacts
        - config.recovery_penalty * recovery
        - config.reverse_penalty * reversing
        - config.residual_penalty * effort
    )


def build_transitions(records: list[StepRecord], termination: str, config: RewardConfig | None = None) -> Transitions | None:
    """Turn one episode's step records into (s, a, r, s', terminal) arrays.

    The last record has no successor: it becomes a terminal transition when the run
    finished or got stuck, and is dropped on a timeout (bootstrapping needs s').
    Returns ``None`` for a TMI disruption or an episode too short to learn from.
    """

    config = config or RewardConfig()
    if termination == TMI_DISRUPT or len(records) < 2:
        return None
    observations = np.stack([r.observation for r in records]).astype(np.float32)
    actions = np.stack([r.action for r in records]).astype(np.float32)
    rewards = step_rewards(records, config)
    terminals = np.zeros(len(records) - 1, dtype=bool)
    next_obs = observations[1:]
    flags = np.array([r.residual_active for r in records], dtype=bool)
    obs, acts, active, next_active = observations[:-1], actions[:-1], flags[:-1], flags[1:]
    if termination in (FINISHED, STUCK):
        # The final step ends the episode: no future value after it.
        final_reward = -config.stuck_penalty if termination == STUCK else 0.0
        obs = np.concatenate([obs, observations[-1:]])
        acts = np.concatenate([acts, actions[-1:]])
        rewards = np.append(rewards, final_reward - config.residual_penalty * float(np.sum(np.square(actions[-1]))))
        next_obs = np.concatenate([next_obs, observations[-1:]])
        terminals = np.append(terminals, True)
        active = np.append(active, flags[-1])
        next_active = np.append(next_active, False)  # terminal: never bootstrapped
    return Transitions(obs, acts, rewards.astype(np.float32), next_obs, terminals, active, next_active)
