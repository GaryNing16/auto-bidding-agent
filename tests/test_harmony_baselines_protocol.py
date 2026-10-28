"""Baseline fairness and repeated-episode state regression checks."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from harmony_bidding.agent_api import OwnConfig, PublicConfig, Observation
from harmony_bidding.schema import Advertiser
from harmony_baselines.classical import FixedAlphaAgent, spec
from harmony_baselines.data import MatrixDay, MatrixSlot
from harmony_baselines.evaluator import opponent_specs, run_episode
from harmony_baselines import experiment


def _day():
    slots = tuple(MatrixSlot(np.array([[.04, .03, 0.]]), np.array([[True, True, False]]),
               np.array([f"r{t}"]), np.array([0])) for t in range(72))
    return MatrixDay("protocol_smoke", (Advertiser("0", "0", 10.), Advertiser("1", "0", 10.),
                        Advertiser("2", "1", 10.)), slots, "test-synthetic", {})


def test_fixed_reset_does_not_multiply_again():
    agent = FixedAlphaAgent(alpha=40., multiplier=2., player_alphas={"0": 30.})
    for player, alpha in [("0", 60.), ("1", 80.), ("1", 80.), ("0", 60.)]:
        own = OwnConfig(player, "0", 10., 1)
        agent.reset(own, PublicConfig(72))
        assert agent.bidding(Observation(own, 0, 72, 10., ())) == alpha


def test_mixed_types_are_episode_stable_and_balanced():
    opponents = {"fixed": spec("FixedAlpha", alpha=20.), "pid": spec("LocalPID", initial_alpha=20.)}
    ids = [str(i) for i in range(48)]
    first = opponent_specs(ids, "0", "Mixed", opponents, 123, "train")
    second = opponent_specs(ids, "0", "Mixed", opponents, 123, "train")
    assert first == second
    assert "0" not in first
    assert sum(x == opponents["fixed"] for x in first.values()) == 23
    assert sum(x == opponents["pid"] for x in first.values()) == 24


def test_target_policy_cannot_change_initial_opponent_allocation():
    opponents = {"fixed": spec("FixedAlpha", alpha=20.), "pid": spec("LocalPID", initial_alpha=20.)}
    first, slots, transitions = run_episode(_day(), "0", spec("FixedAlpha", alpha=20.), "Mixed",
                                         opponents, seed=123, collect=True)
    second, _, _ = run_episode(_day(), "0", spec("FixedAlpha", alpha=40.), "Mixed", opponents, seed=123)
    assert first["assignment_fingerprint"] == second["assignment_fingerprint"]
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert len(slots) == len(transitions) == 72
    assert transitions[-1]["done"] is True
    assert not any(x["done"] for x in transitions[:-1])
    assert all(x["state"].shape == (18,) for x in transitions)


def test_unused_future_market_cannot_affect_first_decision():
    opponents = {"fixed": spec("FixedAlpha", alpha=20.), "pid": spec("LocalPID", initial_alpha=20.)}
    day = _day()
    altered = replace(day, slots=tuple(replace(s, values=np.array([[100., .001, 0.]])) for s in day.slots))
    target = spec("LocalPID", initial_alpha=30.)
    _, first, _ = run_episode(day, "0", target, "Mixed", opponents, seed=123)
    _, second, _ = run_episode(altered, "0", target, "Mixed", opponents, seed=123)
    assert first[0]["alpha"] == second[0]["alpha"] == 30.


def test_all_player_tasks_share_identical_market_role_seed_order():
    candidates = [{"algorithm": "A", "spec": spec("FixedAlpha", alpha=20.)},
                  {"algorithm": "B", "spec": spec("LocalPID", initial_alpha=20.)}]
    opponents = {"fixed": candidates[0]["spec"], "pid": candidates[1]["spec"]}
    tasks = experiment.tasks_for(candidates, opponents, ["period-24"], [str(i) for i in range(48)], [11, 12])
    assert len(tasks) == 2 * 3 * 48 * 2
    for algorithm in ("A", "B"):
        selected = [t for t in tasks if t["algorithm"] == algorithm]
        assert {t["target"] for t in selected} == {str(i) for i in range(48)}
        assert len({(t["scenario"], t["target"], t["seed"]) for t in selected}) == len(selected)
    first = [t for t in tasks if t["algorithm"] == "A"]
    second = [t for t in tasks if t["algorithm"] == "B"]
    for a, b in zip(first, second):
        for key in ("day_id", "target", "scenario", "opponents", "seed"):
            assert a[key] == b[key]


def test_prepare_does_not_read_test_market_and_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "OUT", tmp_path)
    manifest = {"periods": [{"day_id": "period-7", "split": "train"},
                            {"day_id": "period-21", "split": "validation"},
                            {"day_id": "period-24", "split": "local_holdout"}]}
    monkeypatch.setattr(experiment, "source_manifest", lambda: manifest)
    loaded = []
    monkeypatch.setattr(experiment, "build_cache", loaded.append)
    experiment.prepare()
    assert loaded == experiment.TRAIN_DAYS + experiment.VALID_DAYS
    assert not set(loaded) & set(experiment.TEST_DAYS)
    before = (tmp_path / "protocol.json").read_bytes()
    with pytest.raises(ValueError, match="fresh-only"):
        experiment.prepare()
    assert (tmp_path / "protocol.json").read_bytes() == before


def test_fresh_output_cli_does_not_change_existing_protocol_or_source(tmp_path, monkeypatch):
    import sys
    old = tmp_path / 'sealed_run'
    fresh = tmp_path / 'new_run'
    monkeypatch.setattr(experiment, 'OUT', old)
    experiment.dump(old/'protocol.json', {'status': 'completed', 'run_id': 'sealed_run'})
    old_bytes = (old/'protocol.json').read_bytes()
    source = Path(experiment.__file__)
    source_before = source.read_bytes()
    monkeypatch.setattr(experiment, 'source_manifest', lambda: {'periods': []})
    loaded = []
    monkeypatch.setattr(experiment, 'build_cache', loaded.append)
    monkeypatch.setattr(sys, 'argv', ['experiment', 'prepare', '--output', str(fresh)])
    experiment.main()
    assert experiment.read(fresh/'protocol.json')['run_id'] == 'new_run'
    assert loaded == experiment.TRAIN_DAYS + experiment.VALID_DAYS
    assert (old/'protocol.json').read_bytes() == old_bytes
    assert source.read_bytes() == source_before
    with pytest.raises(ValueError, match='fresh-only'):
        experiment.main()


def test_training_trajectory_serialization_is_completion_order_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "OUT", tmp_path)
    classical = [{"algorithm": "FixedAlpha", "spec": spec("FixedAlpha", alpha=20., multiplier=2.)},
                 {"algorithm": "LocalPID", "spec": spec("LocalPID", initial_alpha=20., multiplier=4.)}]
    experiment.dump(tmp_path / "classical_selected.json", classical)
    experiment.dump(tmp_path / "opponents_frozen.json", {"fixed": classical[0]["spec"], "pid": classical[1]["spec"]})
    experiment.dump(tmp_path / "calibration.json", {})
    def fake_execute(tasks, *args, **kwargs):
        assert len(tasks) == len(experiment.TRAIN_DAYS) * 48 * 3
        assert {t["target"] for t in tasks} == {str(i) for i in range(48)}
        assert {t["day_id"] for t in tasks} == set(experiment.TRAIN_DAYS)
        assert all(t["collect"] and not t["collect_all"] for t in tasks)
        for task in tasks:
            relative = float(task["algorithm"].rsplit("_", 1)[-1])
            base = 2. if "FixedAlpha" in task["algorithm"] else 4.
            actual = task["target_spec"]["kwargs"]["base_spec"]["kwargs"]["multiplier"]
            assert actual == pytest.approx(base * relative)
        transitions = [{"episode_id": f"{episode}:0", "player_id": "0", "time_slot": slot,
                       "state": np.full(18, episode + slot), "next_state": np.full(18, episode + slot + 1),
                       "action": 20., "reward": .2, "done": slot == 71}
                      for episode in [10, 2] for slot in range(71, -1, -1)]
        return None, None, transitions
    monkeypatch.setattr(experiment, "execute", fake_execute)
    experiment.trajectories(workers=1)
    with np.load(tmp_path / "train_transitions.npz", allow_pickle=False) as data:
        assert data["episode_id"].tolist() == ["2:0"] * 72 + ["10:0"] * 72
        assert data["states"][:, 0].tolist() == list(range(2,74)) + list(range(10,82))


def test_select_excludes_incomplete_training_seed_coverage(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(experiment, "OUT", tmp_path)
    monkeypatch.setattr(experiment, "TRAIN_SEEDS", [1, 2])
    experiment.dump(tmp_path / "opponents_frozen.json", {"fixed": spec("FixedAlpha"), "pid": spec("LocalPID")})
    experiment.dump(tmp_path / "classical_selected.json", [{"algorithm": "LocalPID", "spec": spec("LocalPID")}])
    experiment.dump(tmp_path / "protocol.json", {"status": "preselection"})
    (tmp_path / "online_lp_table.csv").write_text("timeStepIndex,cum_cost\n0,0\n")
    records = []
    for algorithm, seed in [("BC", 1), ("BC", 2), ("CQL", 1)]:
        checkpoint = tmp_path / f"{algorithm}_{seed}.pt"
        checkpoint.write_bytes(f"fixture {algorithm} {seed}".encode())
        records.append({"algorithm": algorithm, "seed": seed, "config_index": 0, "step": 3,
                        "checkpoint": str(checkpoint)})
    experiment.dump(tmp_path / "models/candidates.json", records)
    def fake_execute(tasks, *args, **kwargs):
        return pd.DataFrame([{"algorithm": t["algorithm"], "training_seed": t["training_seed"],
               "scenario": t["scenario"], "player_id": t["target"], "total_value": 1.} for t in tasks]), [], []
    monkeypatch.setattr(experiment, "execute", fake_execute)
    experiment.select(1)
    frozen = experiment.read(tmp_path / "selected_frozen.json")
    assert [s["algorithm"] for s in frozen["selected"]] == ["LocalPID", "BC", "BC"]
    assert frozen["excluded_candidates"] == [{"candidate": "CQL|0|3", "reason": "missing training seed"}]
    upstream = [key.replace('\\','/') for key in frozen['code_sha256']]
    assert 'strategy_train_env/bidding_train_env/baseline/bcq/bcq.py' in upstream
    assert 'simul_bidding_env/strategy/onlinelp_bidding_strategy.py' in upstream
    assert experiment.read(tmp_path / "protocol.json")["status"] == "frozen_before_test"


@pytest.mark.parametrize("changed", ["opponents_frozen.json", "online_lp_table.csv", "model.pt", "selected_frozen.json"])
def test_test_market_is_not_loaded_when_frozen_asset_changed(tmp_path, monkeypatch, changed):
    monkeypatch.setattr(experiment, "OUT", tmp_path)
    for name in ["opponents_frozen.json", "online_lp_table.csv", "model.pt"]:
        (tmp_path / name).write_bytes(b"frozen fixture")
    frozen = {"opponents_sha256": experiment.sha256_file(tmp_path / "opponents_frozen.json"),
              "lp_table_sha256": experiment.sha256_file(tmp_path / "online_lp_table.csv"),
              "checkpoints": {str(tmp_path / "model.pt"): experiment.sha256_file(tmp_path / "model.pt")},
              "selected": []}
    experiment.dump(tmp_path / "selected_frozen.json", frozen)
    experiment.dump(tmp_path / "protocol.json", {"status": "frozen_before_test",
                  "selection_sha256": experiment.sha256_file(tmp_path / "selected_frozen.json")})
    if changed == "selected_frozen.json":
        frozen["selected"] = [{"changed": True}]
        experiment.dump(tmp_path / changed, frozen)
    else:
        (tmp_path / changed).write_bytes(b"changed fixture")
    def prohibited(*args, **kwargs):
        pytest.fail("Test market must not be loaded before frozen assets validate")
    monkeypatch.setattr(experiment, "build_cache", prohibited)
    with pytest.raises(ValueError, match="changed"):
        experiment.test(1)


@pytest.mark.parametrize("status", ["preselection", "completed", "failed"])
def test_test_requires_frozen_pretest_status(tmp_path, monkeypatch, status):
    monkeypatch.setattr(experiment, "OUT", tmp_path)
    experiment.dump(tmp_path / "selected_frozen.json", {})
    experiment.dump(tmp_path / "protocol.json", {"status": status})
    def prohibited(*args, **kwargs):
        pytest.fail("Test markets cannot be loaded for a non-frozen stage")
    monkeypatch.setattr(experiment, "build_cache", prohibited)
    with pytest.raises(ValueError, match="pre-test freeze"):
        experiment.test(1)
