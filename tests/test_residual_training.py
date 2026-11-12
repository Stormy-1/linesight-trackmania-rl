"""Residual TD3 training loop core, driven by fake episodes (no game)."""

import json
from pathlib import Path

import numpy as np
import pytest

from trackmania_rl.residual_observation import OBSERVATION_SIZE
from trackmania_rl.residual_reward import StepRecord, build_transitions
from trackmania_rl.residual_td3 import TD3Agent, TD3Config
from trackmania_rl.residual_training import ResidualTrainer, TrainingConfig, illegal_training_maps

ROOT = Path(__file__).resolve().parents[1]
SMALL = TD3Config(hidden_units=32, batch_size=16)


def fake_records(n=40, active=True, seed=0):
    rng = np.random.default_rng(seed)
    return [
        StepRecord(
            rng.normal(size=OBSERVATION_SIZE).astype(np.float32),
            rng.uniform(-1, 1, size=2).astype(np.float32) if active else np.zeros(2, np.float32),
            active,
            float(i),
            False,
            False,
            False,
        )
        for i in range(n)
    ]


def make_trainer(tmp_path, **overrides):
    config = TrainingConfig(**{"actor_warmup_transitions": 60, "buffer_capacity": 1000, **overrides})
    return ResidualTrainer(TD3Agent(SMALL), config, tmp_path / "run")


def log_lines(trainer):
    return [json.loads(line) for line in (trainer.out_dir / "train_log.jsonl").read_text().splitlines()]


def test_an_episode_is_saved_learned_from_logged_and_checkpointed(tmp_path):
    trainer = make_trainer(tmp_path)
    stats = trainer.record_episode("B12-Race", fake_records(), "Finished", {"completion_pct": 100.0, "steps": 40})
    assert trainer.episodes == 1 and trainer.buffer.size == 40
    assert (trainer.episode_dir / "ep00001_B12-Race.npz").is_file()
    assert (trainer.out_dir / "latest.pt").is_file()
    assert stats["updates"] == 40 and stats["completion_pct"] == 100.0
    assert log_lines(trainer)[0]["map"] == "B12-Race"
    assert trainer.agent.critic_updates == 40


def test_actor_waits_for_enough_explored_transitions(tmp_path):
    trainer = make_trainer(tmp_path)
    first = trainer.record_episode("A06-Obstacle", fake_records(seed=1), "Stuck_Cutoff_2s", {})
    assert not first["actor_trained"] and first["actor_loss"] is None
    assert np.all(trainer.agent.act(np.zeros(OBSERVATION_SIZE, np.float32)) == 0.0)  # still the base stack
    second = trainer.record_episode("A06-Obstacle", fake_records(seed=2), "Stuck_Cutoff_2s", {})
    assert second["actor_trained"] and second["actor_loss"] is not None


def test_seed_data_is_not_exploration(tmp_path):
    trainer = make_trainer(tmp_path)
    seed_file = tmp_path / "seed_rl.npz"
    np.savez_compressed(seed_file, **build_transitions(fake_records(active=True), "Finished").as_dict())
    assert trainer.add_seed_data([seed_file]) == 40
    assert trainer.buffer.size == 40 and trainer.explored_transitions == 0


def test_earlier_training_episodes_count_as_exploration(tmp_path):
    trainer = make_trainer(tmp_path)
    earlier = tmp_path / "ep00001_B12-Race.npz"
    np.savez_compressed(earlier, **build_transitions(fake_records(active=True), "Finished").as_dict())
    assert trainer.add_seed_data([earlier], explored=True) == 40
    assert trainer.explored_transitions == 40 and trainer.episodes == 0


def test_disrupted_episode_is_skipped(tmp_path):
    trainer = make_trainer(tmp_path)
    stats = trainer.record_episode("D05-Race", fake_records(), "TMI_Disrupt", {})
    assert stats["skipped"] and trainer.episodes == 0 and trainer.buffer.size == 0
    assert not list(trainer.episode_dir.glob("*.npz"))


def test_resume_restores_episodes_and_buffer(tmp_path):
    trainer = make_trainer(tmp_path)
    for seed in range(3):
        trainer.record_episode("B11-Race", fake_records(seed=seed), "Finished", {})
    resumed = ResidualTrainer(TD3Agent.load(trainer.out_dir / "latest.pt"), trainer.config, trainer.out_dir)
    assert resumed.resume() == 3
    assert resumed.buffer.size == 120 and resumed.explored_transitions == 120
    assert resumed.episode_path("B11-Race", ".csv").name == "ep00004_B11-Race.csv"


def test_training_config_rejects_negative_noise():
    with pytest.raises(ValueError):
        TrainingConfig(exploration_noise=-0.1)


def test_load_can_override_burn_in_hyperparameters(tmp_path):
    burn_in = TD3Agent(TD3Config(hidden_units=32, target_noise=0.0))
    burn_in.save(tmp_path / "burnin.pt")
    agent = TD3Agent.load(tmp_path / "burnin.pt", config=TD3Config(hidden_units=32))
    assert agent.config.target_noise == 0.2


def test_only_rl_train_maps_may_be_trained_on():
    split = json.loads((ROOT / "maps" / "map_split.json").read_text())
    assert illegal_training_maps(split["rl_train"], split["rl_train"]) == []
    assert illegal_training_maps(["B12-Race", split["test"][0], split["dev"][0]], split["rl_train"]) == [split["test"][0], split["dev"][0]]
