"""TD3 learner for the residual policy: zero start, buffer, updates, checkpoints."""

import numpy as np
import pytest
import torch

from trackmania_rl.residual_observation import OBSERVATION_SIZE
from trackmania_rl.residual_reward import StepRecord, build_transitions
from trackmania_rl.residual_td3 import ActorResidualPolicy, ReplayBuffer, TD3Agent, TD3Config, discounted_returns

SMALL = TD3Config(hidden_units=64, batch_size=128)


def random_batch(rng, n, obs_size=OBSERVATION_SIZE):
    return {
        "observations": rng.normal(size=(n, obs_size)).astype(np.float32),
        "actions": rng.uniform(-1, 1, size=(n, 2)).astype(np.float32),
        "rewards": rng.normal(size=n).astype(np.float32),
        "next_observations": rng.normal(size=(n, obs_size)).astype(np.float32),
        "terminals": np.zeros(n, bool),
        "residual_active": np.ones(n, bool),
        "next_residual_active": np.ones(n, bool),
    }


def test_untrained_actor_is_exactly_the_base_stack():
    agent = TD3Agent(SMALL)
    obs = np.random.default_rng(0).normal(size=(50, OBSERVATION_SIZE)).astype(np.float32)
    assert all(np.all(agent.act(o) == 0.0) for o in obs)
    assert np.all(ActorResidualPolicy(agent).act(obs[0]) == 0.0)


def test_exploration_noise_stays_in_range():
    policy = ActorResidualPolicy(TD3Agent(SMALL), noise_std=5.0, seed=1)
    actions = np.stack([policy.act(np.zeros(OBSERVATION_SIZE, np.float32)) for _ in range(100)])
    assert actions.min() >= -1.0 and actions.max() <= 1.0 and actions.std() > 0.3
    with pytest.raises(ValueError):
        ActorResidualPolicy(TD3Agent(SMALL), noise_std=-0.1)


def test_held_noise_changes_only_every_n_steps():
    policy = ActorResidualPolicy(TD3Agent(SMALL), noise_std=0.2, seed=3, noise_hold_steps=4)
    actions = np.stack([policy.act(np.zeros(OBSERVATION_SIZE, np.float32)) for _ in range(12)])
    for block in (actions[0:4], actions[4:8], actions[8:12]):
        assert np.all(block == block[0])
    assert not np.all(actions[0] == actions[4])
    with pytest.raises(ValueError):
        ActorResidualPolicy(TD3Agent(SMALL), noise_std=0.2, noise_hold_steps=0)


def test_buffer_wraps_and_samples():
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(capacity=10)
    buffer.add(random_batch(rng, 7))
    buffer.add(random_batch(rng, 7))
    assert buffer.size == 10
    batch = buffer.sample(32, rng)
    assert batch["observations"].shape == (32, OBSERVATION_SIZE) and batch["terminals"].dtype == torch.bool


def test_buffer_rejects_bad_transitions():
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(capacity=10)
    with pytest.raises(ValueError):
        buffer.sample(4, rng)
    with pytest.raises(ValueError):
        buffer.add(random_batch(rng, 3, obs_size=5))
    old = random_batch(rng, 3)
    del old["next_residual_active"]
    with pytest.raises(KeyError):
        buffer.add(old)


def test_buffer_takes_episode_transitions(tmp_path):
    records = [
        StepRecord(np.zeros(OBSERVATION_SIZE, np.float32), np.zeros(2, np.float32), i != 2, float(i), False, i == 2, False)
        for i in range(5)
    ]
    transitions = build_transitions(records, "Finished")
    # step 2 was a recovery step: the transition into it must not bootstrap with the actor
    np.testing.assert_array_equal(transitions.next_residual_active, [True, False, True, True, False])
    path = tmp_path / "run01_rl.npz"
    np.savez_compressed(path, **transitions.as_dict())
    buffer = ReplayBuffer(capacity=100)
    assert buffer.add_npz(path) == 5 and buffer.size == 5


def test_target_action_is_zero_where_a_safety_layer_owns_the_next_state():
    agent = TD3Agent(SMALL)
    with torch.no_grad():
        agent.actor_target.net[-1].bias.fill_(1.0)  # a confident non-zero target policy
    next_obs = torch.zeros(4, OBSERVATION_SIZE)
    actions = agent.target_next_actions(next_obs, torch.tensor([True, False, True, False]))
    assert torch.all(actions[1] == 0.0) and torch.all(actions[3] == 0.0)
    assert torch.all(actions[0] != 0.0)


def test_critic_learns_a_terminal_reward():
    rng = np.random.default_rng(0)
    batch = random_batch(rng, 512)
    batch["rewards"][:] = 1.0
    batch["terminals"][:] = True
    buffer = ReplayBuffer(capacity=512)
    buffer.add(batch)
    agent = TD3Agent(SMALL)
    first = agent.update(buffer.sample(128, rng), train_actor=False)["critic_loss"]
    for _ in range(300):
        stats = agent.update(buffer.sample(128, rng), train_actor=False)
    assert stats["critic_loss"] < 0.1 * first
    assert stats["q_mean"] == pytest.approx(1.0, abs=0.1)
    assert np.all(agent.act(batch["observations"][0]) == 0.0)  # burn-in leaves the actor untouched


def test_actor_moves_off_zero_only_where_it_pays():
    """One-step bandit: reward peaks at steering residual +0.6; the speed residual doesn't matter."""
    rng = np.random.default_rng(0)
    batch = random_batch(rng, 4096)
    batch["rewards"] = (1.0 - (batch["actions"][:, 0] - 0.6) ** 2).astype(np.float32)
    batch["terminals"][:] = True
    buffer = ReplayBuffer(capacity=4096)
    buffer.add(batch)
    agent = TD3Agent(SMALL, seed=0)
    for _ in range(300):
        agent.update(buffer.sample(256, rng), train_actor=False)
    for _ in range(1500):
        agent.update(buffer.sample(256, rng))
    actions = np.stack([agent.act(o) for o in batch["observations"][:200]])
    assert actions[:, 0].mean() > 0.3  # pulled toward the payoff, a little short of it (anchor to zero)
    assert abs(actions[:, 1].mean()) < 0.15  # no payoff -> the anchor keeps it near the base command


def test_without_anchor_the_actor_reaches_the_payoff():
    """Same bandit as above, anchor off: plain TD3 goes (nearly) all the way to the +0.6 optimum."""
    rng = np.random.default_rng(0)
    batch = random_batch(rng, 4096)
    batch["rewards"] = (1.0 - (batch["actions"][:, 0] - 0.6) ** 2).astype(np.float32)
    batch["terminals"][:] = True
    buffer = ReplayBuffer(capacity=4096)
    buffer.add(batch)
    agent = TD3Agent(TD3Config(hidden_units=64, batch_size=128, anchor_alpha=None), seed=0)
    for _ in range(300):
        agent.update(buffer.sample(256, rng), train_actor=False)
    for _ in range(1500):
        agent.update(buffer.sample(256, rng))
    actions = np.stack([agent.act(o) for o in batch["observations"][:200]])
    assert actions[:, 0].mean() > 0.5


def test_soft_update_moves_targets_by_tau():
    agent = TD3Agent(SMALL)
    before = [p.clone() for p in agent.critic_target.parameters()]
    with torch.no_grad():
        for p in agent.critic.parameters():
            p.add_(1.0)
    agent._soft_update(agent.critic, agent.critic_target)
    for b, t in zip(before, agent.critic_target.parameters()):
        torch.testing.assert_close(t, b + SMALL.tau)


def test_discounted_returns_restart_after_terminals():
    rewards = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    terminals = np.array([False, False, True, False, True])
    np.testing.assert_allclose(discounted_returns(rewards, terminals, 0.5), [1.75, 1.5, 1.0, 3.0, 2.0])


def test_checkpoint_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    agent = TD3Agent(SMALL)
    buffer = ReplayBuffer(capacity=256)
    buffer.add(random_batch(rng, 256))
    for _ in range(4):
        agent.update(buffer.sample(64, rng))
    agent.save(tmp_path / "agent.pt")
    loaded = TD3Agent.load(tmp_path / "agent.pt")
    obs = rng.normal(size=OBSERVATION_SIZE).astype(np.float32)
    np.testing.assert_array_equal(agent.act(obs), loaded.act(obs))
    assert loaded.critic_updates == 4 and loaded.config == SMALL
