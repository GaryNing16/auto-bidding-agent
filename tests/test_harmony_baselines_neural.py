"""Real AuctionEngine smoke, serialization, causality, and numerical regression."""
from dataclasses import replace
import json
import math
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from harmony_bidding.agent_api import OwnConfig, PublicConfig, Observation, OwnFeedback
from harmony_bidding.schema import Advertiser, DayData, Opportunity
from harmony_bidding.runner import run_episode
from harmony_baselines.neural import (ALGORITHMS, FEATURE_NAMES, NeuralAgent, NeuralPolicy,
                                      encode_observation, train_candidates, load_training_data,
                                      SUPPORTED_ACTION_TRANSFORMS)
from harmony_baselines.upstream import make_model


def tiny_day():
    advertisers = (Advertiser("a", "0", 5.), Advertiser("b", "1", 5.))
    slots = tuple(tuple(Opportunity("neural_smoke", t, f"r{t}_{j}", j,
                           (.01 + .001 * (t % 3), .012), (True, j % 3 != 0))
                         if j % 3 != 0 else Opportunity("neural_smoke", t, f"r{t}_{j}", j,
                           (.01 + .001 * (t % 3), 0.), (True, False)) for j in range(3)) for t in range(72))
    return DayData("neural_smoke", advertisers, slots)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("auctionnet_neural_smoke")
    # Collect transitions from the same authority engine used for evaluation.
    arrays = {k: [] for k in ("states", "actions", "rewards", "next_states", "dones", "episode_id")}
    config = {"rules": {}, "pid": {"initial_alpha": 15.}, "write_trajectories": True}
    for episode, alpha in enumerate((8., 20.)):
        report = run_episode(tiny_day(), config, assignments={"a": "behavior"},
             team_specs={"behavior": {"kind": "constant", "alpha": alpha}})
        # Runner emits legal public snapshots regardless of actor family.
        own = OwnConfig("a", "0", 5., 0)
        history = []
        for t in range(72):
            slot = next(x for x in report["slots"] if x["time_slot"] == t and x["advertiser_id"] == "a")
            before = Observation(own, t, 72, 5. - sum(x.cost for x in history), tuple(history))
            feedback = OwnFeedback(t, alpha, slot["wins"], slot["cost"], slot["value"],
                slot["remaining_budget"], slot["cumulative_cost"], slot["cumulative_value"])
            history.append(feedback)
            after = Observation(own, t+1, 72, feedback.remaining_budget, tuple(history))
            for key, value in zip(arrays, (encode_observation(before), alpha, feedback.value,
                          encode_observation(after), t == 71, episode)):
                arrays[key].append(value)
    source = out / "core_trajectories.npz"
    np.savez_compressed(source, **{k: np.asarray(v) for k, v in arrays.items()})
    records = train_candidates(source, out / "models", steps=3, checkpoint_steps=(3,),
          seeds=(17,), configs={a: [{"learning_rate": 3e-4}] for a in ALGORITHMS}, batch_size=8, num_threads=1)
    assert len(records) == len(ALGORITHMS), (out / "models/failures.json").read_text()
    return out, records


def test_all_upstream_algorithms_train_reload_and_run_core(trained):
    out, records = trained
    results = []
    for record in records:
        checkpoint = record["checkpoint"]
        assert NeuralPolicy.load(checkpoint).parameter_count > 1000
        config = {"rules": {}, "pid": {"initial_alpha": 15.}}
        specification = {"kind": "custom", "class_path": "harmony_baselines.neural:NeuralAgent",
                         "kwargs": {"checkpoint": checkpoint}}
        first = run_episode(tiny_day(), config, assignments={"a": "candidate"},
                            team_specs={"candidate": specification})
        second = run_episode(tiny_day(), config, assignments={"a": "candidate"},
                            team_specs={"candidate": specification})
        assert len(first["slots"]) == 144
        assert first["summaries"] == second["summaries"]
        assert all(math.isfinite(x["alpha"]) and x["alpha"] >= 0 for x in first["slots"])
        results.append({"algorithm": record["algorithm"], "summaries": first["summaries"]})
    (out / "smoke_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def test_encoder_has_only_own_history_fields():
    own = OwnConfig("player", "0", 10., 42)
    observation = Observation(own, 0, 72, 10., ())
    encoded = encode_observation(observation)
    assert encoded.shape == (18,)
    assert all("pvalue" not in name and "opponent" not in name for name in FEATURE_NAMES)
    changed_identity = replace(observation, own_config=replace(own, advertiser_id="other", advertiser_type="5"))
    assert np.array_equal(encoded, encode_observation(changed_identity))
    feedback = OwnFeedback(0, 40., 1, 1., .1, 9., 1., .1)
    assert not np.array_equal(encoded, encode_observation(Observation(own, 1, 72, 9., (feedback,))))


@pytest.mark.parametrize("algorithm", ["BCQ", "TD3BC"])
def test_two_hot_upper_boundary_is_valid_probability(algorithm):
    model = make_model(algorithm, 18, 1, {"max_action": 3.})
    encoded = model.two_hot(torch.tensor([[0.], [math.expm1(5.)], [1e20]]))
    assert encoded.shape == (3, 101)
    assert torch.allclose(encoded.sum(1), torch.ones(3))
    assert encoded[-1, -1] == 1


def test_td3_actor_optimizes_scalar_q():
    model = make_model("TD3BC", 18, 1, {"max_action": 3.})
    assert model.critic.Q1(torch.zeros(4, 18), torch.zeros(4, 1)).shape == (4, 1)


def test_cql_targets_start_from_online_network():
    model = make_model("CQL", 18, 1, {})
    for first, second in zip(model.qf1.parameters(), model.target_qf1.parameters()):
        assert torch.equal(first, second)


def test_incomplete_episodes_are_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, states=np.zeros((1,18)), next_states=np.zeros((1,18)), actions=[20.],
             rewards=[1.], dones=[1], episode_id=[0])
    with pytest.raises(ValueError, match="72"):
        load_training_data(path)


def test_linear_cql_config_local_training_reload_and_core(trained, tmp_path, monkeypatch):
    """A linear candidate must not change the next legacy candidate's data."""
    import harmony_baselines.neural as neural
    source = trained[0] / "core_trajectories.npz"
    observed = []
    original_factory = neural.make_model

    def tracked_factory(algorithm, dim_obs, seed, config):
        model = original_factory(algorithm, dim_obs, seed, config)
        original_step = model.step
        def tracked_step(*batch):
            observed.append((config["action_transform"], batch[1].detach().clone().numpy()))
            return original_step(*batch)
        model.step = tracked_step
        return model

    monkeypatch.setattr(neural, "make_model", tracked_factory)
    configs = {"CQL": [{"learning_rate": 1e-4, "action_transform": "linear_scaled"},
                       {"learning_rate": 1e-4}]}
    records = train_candidates(source, tmp_path / "models", algorithms=("CQL",),
          steps=3, checkpoint_steps=(3,), seeds=(17,), configs=configs,
          batch_size=8, num_threads=1)
    assert len(records) == 2
    assert "linear_scaled" in SUPPORTED_ACTION_TRANSFORMS
    assert [r["action_transform"] for r in records] == ["linear_scaled", "log1p"]
    with np.load(source, allow_pickle=False) as original:
        raw = original["actions"].astype(np.float32).reshape(-1, 1)
    scale = float(np.median(raw[raw > 0]))
    for variant_index, transform in enumerate(("linear_scaled", "log1p")):
        rng = np.random.default_rng(17)
        for step in range(3):
            ids = rng.integers(len(raw), size=8)
            expected = raw[ids] / scale
            if transform == "log1p":
                expected = np.log1p(expected)
            actual_transform, actual = observed[variant_index * 3 + step]
            assert actual_transform == transform
            assert np.array_equal(actual, expected)
    policy = NeuralPolicy.load(records[0]["checkpoint"])
    assert policy.config["action_transform"] == policy.metadata["action_transform"] == "linear_scaled"
    assert policy.decode(1000.) == scale * 1000.  # No exponential and no upper cap.
    assert policy.decode(-2.) == 0.
    spec = {"kind": "custom", "class_path": "harmony_baselines.neural:NeuralAgent",
            "kwargs": {"checkpoint": records[0]["checkpoint"]}}
    first = run_episode(tiny_day(), {"rules": {}, "pid": {"initial_alpha": 15.}},
                        assignments={"a": "candidate"}, team_specs={"candidate": spec})
    second = run_episode(tiny_day(), {"rules": {}, "pid": {"initial_alpha": 15.}},
                         assignments={"a": "candidate"}, team_specs={"candidate": spec})
    assert first["summaries"] == second["summaries"]
    assert first["summaries"]["a"]["bidding_calls"] == 72
    assert all(math.isfinite(row["alpha"]) and row["alpha"] >= 0. for row in first["slots"])


def test_untagged_legacy_checkpoints_keep_exact_log_actions(trained, tmp_path):
    """Simulate existing checkpoints with no tag, for all six model families."""
    for record in trained[1]:
        tagged_path = record["checkpoint"]
        payload = torch.load(tagged_path, map_location="cpu", weights_only=True)
        payload.pop("action_transform")
        payload["config"].pop("action_transform")
        payload["metadata"].pop("action_transform")
        legacy_path = tmp_path / f"legacy_{record['algorithm']}.pt"
        torch.save(payload, legacy_path)
        policy = NeuralPolicy.load(legacy_path)
        assert policy.action_transform == "log1p"
        for coordinate in (-1., 0., .01, .5, 2.):
            assert policy.decode(coordinate) == policy.action_scale * math.expm1(max(0., coordinate))
        results = []
        for path in (tagged_path, str(legacy_path)):
            spec = {"kind": "custom", "class_path": "harmony_baselines.neural:NeuralAgent",
                    "kwargs": {"checkpoint": path}}
            results.append(run_episode(tiny_day(), {"rules": {}, "pid": {"initial_alpha": 15.}},
                assignments={"a": "candidate"}, team_specs={"candidate": spec}))
        assert results[0]["summaries"] == results[1]["summaries"]
        assert [r["alpha"] for r in results[0]["slots"]] == [r["alpha"] for r in results[1]["slots"]]


def test_conflicting_action_transform_checkpoint_is_rejected(trained, tmp_path):
    payload = torch.load(trained[1][0]["checkpoint"], map_location="cpu", weights_only=True)
    payload["metadata"]["action_transform"] = "linear_scaled"
    path = tmp_path / "conflicting.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="differs between config and metadata"):
        NeuralPolicy.load(path)
