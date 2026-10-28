"""Mathematical and data-boundary checks for on-policy PPO and return learning."""
import math
from dataclasses import replace
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from harmony_bidding.agent_api import OwnConfig, Observation
from harmony_rl.models import StateEncoder, ActorCritic, sample_action, decode_action
from harmony_rl.buffer import Trajectory, ReplayBuffer
from harmony_rl.ppo import PPOConfig, compute_gae, update_ppo
from harmony_rl.redistribution import ReturnPredictor, trajectory_features, redistribute, train_predictor, predict_rewards


@pytest.fixture(autouse=True)
def deterministic_small_models():
    torch.set_num_threads(1)
    torch.manual_seed(17)


def trajectory(model=None, version=0, reward=1.):
    model = ActorCritic(3, hidden=8, initial_alpha=12.) if model is None else model
    states = np.column_stack((np.arange(72)/72, np.ones(72), np.zeros(72))).astype(np.float32)
    samples = [sample_action(model, state) for state in states]
    return Trajectory(states, [s["alpha"] for s in samples], [s["latent"] for s in samples],
                      [s["log_prob"] for s in samples], [s["value"] for s in samples],
                      np.full(72, reward), np.full(72, .1), states.copy(),
                      np.r_[np.zeros(71, dtype=bool), True],
                      dict(policy_version=version, initial_budget=10., player_id="a", seed=7))


def test_encoder_roundtrip_and_own_role_conditioning():
    encoder = StateEncoder(["a", "b"], ["type0", "type1"])
    obs = Observation(OwnConfig("a", "type0", 2000., 10), 0, 72, 2000., ())
    encoded = encoder.encode(obs)
    assert np.array_equal(encoded, StateEncoder.from_dict(encoder.to_dict()).encode(obs))
    other = replace(obs, own_config=replace(obs.own_config, advertiser_id="b"))
    assert np.array_equal(encoded[:18], encoder.encode(other)[:18])
    assert not np.array_equal(encoded, encoder.encode(other))
    # No current market field exists in the encoded observation contract.
    assert not any("opponent" in name or "pvalue" in name or "request" in name for name in encoder.feature_names)


def test_action_has_no_artificial_cap_and_rejects_overflow():
    assert decode_action(math.log(1e12)) == pytest.approx(1e12)
    assert decode_action(math.log(7.), 11.) == pytest.approx(77.)
    assert decode_action(0., 0.) == 0.
    with pytest.raises(FloatingPointError):
        decode_action(1000.)
    with pytest.raises(FloatingPointError):
        decode_action(-1000.)


@pytest.mark.parametrize("residual,base", [(False, 1.), (True, 300.)])
def test_latent_probability_ratio_equals_transformed_alpha_ratio(residual, base):
    model = ActorCritic(3, hidden=8, initial_alpha=30., residual=residual)
    state = np.array([0., 1., .5], dtype=np.float32)
    sample = sample_action(model, state, base_alpha=base)
    assert sample["alpha"] > 5.
    with torch.no_grad():
        model.actor[-1].bias.add_(.1)
    distribution = model.distribution(torch.tensor(state).reshape(1, -1))
    new_log_prob = float(distribution.log_prob(torch.tensor(sample["latent"])).item())
    latent_ratio = math.exp(new_log_prob - sample["log_prob"])
    alpha_ratio = math.exp((new_log_prob - math.log(sample["alpha"])) - sample["alpha_log_prob"])
    assert latent_ratio == pytest.approx(alpha_ratio, rel=1e-12)


def test_gae_monte_carlo_at_lambda_one_and_terminal_reset():
    rewards = np.array([1., 2., 7., 3.])
    values = np.array([.5, 9., 2., 4.])
    advantages, returns = compute_gae(rewards, values, [False, True, False, True], gamma=1., gae_lambda=1.)
    assert returns == pytest.approx([3., 2., 10., 3.])
    assert advantages == pytest.approx(returns-values)
    _, one_step = compute_gae(rewards, values, [False, True, False, True], gamma=.9, gae_lambda=0.)
    assert one_step == pytest.approx([1.+.9*9., 2., 7.+.9*4., 3.])


def test_ppo_changes_parameters_and_rejects_replay_and_duplicate_rollout():
    model = ActorCritic(3, hidden=8, initial_alpha=12.)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    episodes = [trajectory(model), trajectory(model, reward=.5)]
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    metrics = update_ppo(model, optimizer, episodes, PPOConfig(epochs=2, minibatch_size=72), expected_policy_version=0)
    assert metrics["optimizer_steps"] > 0
    assert all(np.isfinite(metrics[key]) for key in ("actor_loss", "critic_loss", "approx_kl", "clip_fraction"))
    assert any(not torch.equal(initial[name], value) for name, value in model.state_dict().items())
    with pytest.raises(ValueError, match="already-consumed"):
        update_ppo(model, optimizer, episodes)
    replay = ReplayBuffer()
    replay.add(trajectory(model, version=1))
    with pytest.raises(ValueError, match="LSTM replay"):
        update_ppo(model, optimizer, replay.trajectories)


def test_ppo_rejects_mixed_stale_versions_and_bad_redistribution():
    model = ActorCritic(3, hidden=8)
    optimizer = torch.optim.Adam(model.parameters())
    episodes = [trajectory(model, version=0), trajectory(model, version=1)]
    with pytest.raises(ValueError, match="policy version"):
        update_ppo(model, optimizer, episodes)
    with pytest.raises(ValueError, match="stale"):
        update_ppo(model, optimizer, episodes[:1], expected_policy_version=2)
    with pytest.raises(ValueError, match="conservation"):
        update_ppo(model, optimizer, episodes[:1], reward_sequences=[np.zeros(72)])


def test_trajectory_requires_complete_day_and_lstm_replay_roundtrips():
    episode = trajectory()
    state = episode.state_dict()
    assert Trajectory.from_state_dict(state).episode_return == episode.episode_return
    state["dones"][5] = True
    with pytest.raises(ValueError, match="72nd"):
        Trajectory.from_state_dict(state)
    replay = ReplayBuffer(1)
    replay.add(episode)
    replay.add(trajectory(reward=2.))
    restored = ReplayBuffer()
    restored.load_state_dict(replay.state_dict())
    assert len(restored) == 1
    assert restored.trajectories[0].episode_return == 144.
    assert restored.trajectories[0].metadata["lstm_replay_only"]


def test_return_prediction_is_causal_for_every_prefix():
    episode = trajectory()
    model = ReturnPredictor(8, hidden=8)
    features = trajectory_features(episode)
    with torch.no_grad():
        original = model(torch.tensor(features).unsqueeze(0))
        features[35:] += 100.
        changed = model(torch.tensor(features).unsqueeze(0))
    assert torch.equal(original[:, :35], changed[:, :35])
    assert not torch.equal(original[:, 35:], changed[:, 35:])


@pytest.mark.parametrize("seed", [2, 18, 199])
def test_redistribution_conserves_individual_raw_return(seed):
    rng = np.random.default_rng(seed)
    rewards = rng.uniform(0., 20., 72)
    prediction = rng.normal(0., 1000., 72)
    result = redistribute(prediction, rewards)
    assert result.sum() == pytest.approx(rewards.sum(), rel=1e-12, abs=1e-9)
    assert result[0] == prediction[0]
    assert result[:-1].sum() == pytest.approx(prediction[-2])
    assert np.any(result < 0)


def test_return_predictor_learns_then_freezes_for_reward_computation(tmp_path):
    replay = ReplayBuffer()
    for _ in range(4):
        replay.add(trajectory(reward=.02))
    model = ReturnPredictor(8, hidden=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    first = train_predictor(model, optimizer, replay, epochs=1)
    later = train_predictor(model, optimizer, replay, epochs=15)
    assert later["lstm_loss"] < first["lstm_loss"]
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    rewards, predictions = predict_rewards(model, replay.trajectories[0])
    assert rewards.sum() == pytest.approx(1.44)
    assert predictions.shape == (72,)
    assert all(torch.equal(saved[name], value) for name, value in model.state_dict().items())
    path = tmp_path / "replay.pt"
    torch.save(dict(replay=replay.state_dict(), model=model.state_dict()), path)
    payload = torch.load(path, weights_only=True)
    restored = ReplayBuffer()
    restored.load_state_dict(payload["replay"])
    assert restored.trajectories[0].episode_return == pytest.approx(1.44)
