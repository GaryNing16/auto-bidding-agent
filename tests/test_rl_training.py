"""Tiny real-engine training, deployed actions and exact batch-boundary resume."""
from dataclasses import replace
import importlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
train_module = importlib.import_module("harmony_rl.train")
from harmony_rl.config import TrainConfig
from harmony_rl.markets import synthetic_day
from harmony_rl.env import BiddingEnv
from harmony_rl.policy import PPOAgent, load_payload, model_from_payload
from harmony_rl.models import sample_action
from harmony_bidding.agent_api import PublicConfig


@pytest.fixture
def tiny_config(tmp_path, monkeypatch):
    opponents = {
        "fixed": {"kind": "constant", "alpha": 20.},
        "pid": {"kind": "pid", "initial_alpha": 20., "kp": 2., "ki": .1, "kd": .1},
        "auctionnet_pid": {"kind": "custom",
            "class_path": "harmony_baselines.five_scenarios.auctionnet_pid:AuctionNetPIDAgent",
            "kwargs": {"initial_alpha": 20.}},
    }
    path = tmp_path / "opponents.json"
    path.write_text(json.dumps(opponents), encoding="utf-8")
    monkeypatch.setattr(train_module, "synthetic_day", lambda seed: synthetic_day(seed, players=3, requests_per_slot=2))
    # Resume numerical checks must not depend on unrelated parallel edits.
    monkeypatch.setattr(train_module, "execution_hashes", lambda: {"test_fixture": "frozen"})
    def no_real_data(day):
        raise AssertionError("smoke training attempted to open real/test market: " + day)
    monkeypatch.setattr(train_module, "cached_day", no_real_data)
    config = TrainConfig(smoke=True, episodes=2, batch_episodes=1, workers=1,
        max_seconds=90., hidden=8, lstm_hidden=8, lstm_epochs=1,
        validate_every=2, validation_roles=("0",), scenarios=("Fixed",),
        initial_alpha=20., opponents_path=str(path), opponent_randomization=0.,
        residual_spec={"kind": "pid", "initial_alpha": 20., "kp": 2., "ki": .1, "kd": .1})
    config.ppo = {**config.ppo, "epochs": 1, "minibatch_size": 72}
    return config, opponents


@pytest.mark.parametrize("variant", ["E1", "E3", "E5"])
def test_end_to_end_training_and_frozen_deployment(tiny_config, tmp_path, variant):
    config, opponents = tiny_config
    config.variant = variant
    output = tmp_path / variant
    result = train_module.train(config, output)
    assert result["complete"] and result["validated"]
    assert result["completed_episodes"] == 2
    assert result["environment_steps"] == 144
    assert result["lstm_versions"] == (2 if config.redistribution else 0)
    payload = load_payload(output / "best.pt")
    model, encoder = model_from_payload(payload)
    assert payload["provenance"]["test_read_during_training"] is False
    assert payload["history"][0]["used_lstm_version"] == 0
    if config.redistribution:
        assert payload["history"][1]["used_lstm_version"] == 1
        assert payload["history"][1]["return_conservation_max_error"] < 1e-8
    env = BiddingEnv(synthetic_day(733, players=3, requests_per_slot=2), "0", opponents,
        residual_spec=config.residual_spec if config.residual else None)
    observation, _ = env.reset()
    deployed = PPOAgent(output / "best.pt")
    deployed.reset(observation.own_config, PublicConfig(72))
    for _ in range(72):
        expected = sample_action(model, encoder.encode(observation),
                                base_alpha=env.base_alpha(), deterministic=True)["alpha"]
        actual = deployed.bidding(observation)
        assert actual == expected
        observation, _, _, _, info = env.step(actual)
        deployed.update(info["feedback"])
    assert env.metrics()["completed"]
    files = sorted((output / "trajectories").glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0], allow_pickle=False) as saved:
        assert saved["states"].shape[0] == 72
        assert saved["dones"].sum() == 1


def test_resumed_training_matches_uninterrupted_batch_boundaries(tiny_config, tmp_path):
    config, _ = tiny_config
    config = replace(config, variant="E3", episodes=4, batch_episodes=2, validate_every=1)
    train_module.train(config, tmp_path / "full")
    train_module.train(replace(config, episodes=2), tmp_path / "resumed")
    train_module.train(config, tmp_path / "resumed", resume=True)
    full = load_payload(tmp_path / "full" / "last.pt")
    resumed = load_payload(tmp_path / "resumed" / "last.pt")
    for kind in ("model_state", "lstm_state"):
        for name, tensor in full[kind].items():
            assert torch.equal(tensor, resumed[kind][name]), (kind, name)
    assert full["policy_version"] == resumed["policy_version"] == 2
    assert full["lstm_version"] == resumed["lstm_version"] == 2
    for left, right in zip(full["history"], resumed["history"]):
        for key in ("actor_loss", "critic_loss", "mean_raw_return", "used_lstm_version"):
            assert left[key] == right[key]
    assert torch.equal(full["torch_rng"], resumed["torch_rng"])


def test_train_validation_test_splits_cannot_overlap(tiny_config):
    config, _ = tiny_config
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, test_days=config.train_days).validate()


def test_serial_rollout_does_not_change_optimizer_random_stream(tiny_config):
    config, opponents = tiny_config
    day = synthetic_day(730, players=3, requests_per_slot=2)
    from harmony_rl.models import StateEncoder, ActorCritic
    encoder = StateEncoder([a.advertiser_id for a in day.advertisers], ["0"])
    model = ActorCritic(encoder.state_dim, hidden=8, initial_alpha=20.)
    policy = train_module.policy_payload(model, encoder, config)
    job = train_module._job(config, policy, opponents, 0, 0, "0", "Fixed")
    torch.manual_seed(52)
    before = torch.get_rng_state().clone()
    trajectory = train_module._map(None, [job])[0]
    assert trajectory.states.shape[0] == 72
    assert torch.equal(before, torch.get_rng_state())


def test_interrupt_during_optimizer_preserves_committed_checkpoint(tiny_config, tmp_path, monkeypatch):
    config, _ = tiny_config
    config = replace(config, variant="E3", episodes=2, validate_every=1)
    train_module.train(config, tmp_path / "full")
    real_update = train_module.update_ppo
    def interrupt_after_partial_mutation(model, optimizer, *args, **kwargs):
        real_update(model, optimizer, *args, **kwargs)
        with torch.no_grad():
            next(model.parameters()).add_(123.)
        raise KeyboardInterrupt()
    monkeypatch.setattr(train_module, "update_ppo", interrupt_after_partial_mutation)
    result = train_module.train(config, tmp_path / "interrupted")
    assert result["termination"] == "interrupted"
    assert result["completed_episodes"] == result["ppo_updates"] == 0
    stable = load_payload(tmp_path / "interrupted" / "last.pt")
    assert stable["episode_count"] == stable["policy_version"] == 0
    assert stable["history"] == []
    assert max(float(t.abs().max()) for t in stable["model_state"].values()) < 123.
    monkeypatch.setattr(train_module, "update_ppo", real_update)
    completed = train_module.train(config, tmp_path / "interrupted", resume=True)
    assert completed["complete"]
    full = load_payload(tmp_path / "full" / "last.pt")
    resumed = load_payload(tmp_path / "interrupted" / "last.pt")
    for kind in ("model_state", "lstm_state"):
        assert all(torch.equal(value, resumed[kind][key]) for key, value in full[kind].items())
    assert full["history"][-1]["mean_raw_return"] == resumed["history"][-1]["mean_raw_return"]
