"""Synthetic provenance and mock-training tests; never start formal training."""
from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from harmony_bidding.schema import fingerprint
from harmony_baselines.data import sha256_file
from harmony_baselines.five_scenarios import common as c
from harmony_baselines.five_scenarios import data_training as d


def arrays(episode_ids, reward=.125):
    count = len(episode_ids)
    states = np.zeros((72 * count, 18), dtype=np.float32)
    next_states = states.copy()
    states[:, 0] = np.tile((np.arange(72) / 72).astype(np.float32), count)
    next_states[:, 0] = np.tile((np.arange(1, 73) / 72).astype(np.float32), count)
    return dict(states=states, next_states=next_states, actions=np.full(72 * count, 2., dtype=np.float64),
                rewards=np.full(72 * count, reward, dtype=np.float64),
                dones=np.tile(np.arange(72) == 71, count), episode_id=np.repeat(episode_ids, 72))


def save_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **values)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    old, out = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    out.mkdir()
    monkeypatch.setattr(c, "OLD", old)
    monkeypatch.setattr(c, "OUT", out)
    opponents = {"fixed": {"kind": "constant", "alpha": 2.}, "pid": {"kind": "pid", "initial_alpha": 2.},
                 "auctionnet_pid": {"kind": "pid", "initial_alpha": 3.}}
    behaviors = [{"algorithm": f"behavior_{i}", "spec": {"kind": "constant", "alpha": float(i + 1)}}
                 for i in range(6)]
    rows, ids = [], []
    for di, day in enumerate(c.TRAIN_DAYS):
        for pi, player in enumerate(c.ROLES):
            for si, scenario in enumerate(("Fixed", "PID", "Mixed")):
                index = len(rows)
                behavior = behaviors[(pi + si + di) % 6]
                rows.append(dict(task_index=index, algorithm=behavior["algorithm"], day_id=day,
                    scenario=scenario, player_id=player, seed=c.SEEDS[di], total_value=9., completed=True,
                    model_fingerprint=fingerprint(behavior["spec"]), assignment_fingerprint=f"assignment-{index}",
                    dataset_fingerprint=f"market-{day}"))
                ids.append(f"{index}:{player}")
    save_npz(old / "train_transitions.npz", arrays(ids))
    pd.DataFrame(rows).to_csv(old / "behavior_episodes.csv", index=False)
    c.dump(old / "behavior_episodes.status.json", {"status": "completed", "episodes": 288,
           "csv_sha256": sha256_file(old / "behavior_episodes.csv")})
    c.dump(old / "trajectory_provenance.json", {"days": list(c.TRAIN_DAYS), "episodes": 288,
           "transitions": 20736, "test_used": False, "sha256": sha256_file(old / "train_transitions.npz"),
           "behavior_candidates": behaviors})
    c.dump(old / "opponents_frozen.json", {k: opponents[k] for k in ("fixed", "pid")})
    return old, out, opponents, behaviors


def extension_executor(calls, corrupt=False):
    def execute(tasks, directory, phase, workers, fail_group):
        calls.append(tasks)
        assert phase == "trajectory_extension" and not fail_group
        assert len(tasks) == 192
        assert Counter(task["scenario"] for task in tasks) == {"AuctionNetPID": 96, "Mixed2": 96}
        identities = [{"task_key": fingerprint(task)} for task in tasks]
        c.dump(directory / "task_protocol.json", {"tasks": tasks, "identities": identities})
        rows = []
        for index, task in enumerate(tasks):
            assert task["collect"] is True and task["collect_all"] is False
            path = directory / "results" / f"task_{index:05d}.json"
            npz = path.with_suffix(".npz")
            save_npz(npz, arrays([f'new:{index}:{task["target"]}'], reward=.25))
            row = dict(task_index=index, algorithm=task["algorithm"], scenario=task["scenario"],
                day_id=task["day_id"], player_id=task["target"], seed=task["seed"], training_seed=0,
                total_value=18., model_fingerprint=fingerprint(task["target_spec"]),
                assignment_fingerprint=f"new-assignment-{index}", dataset_fingerprint=f'market-{task["day_id"]}')
            rows.append(row)
            c.dump(path, {"status": "completed", "identity": identities[index], "row": row,
                "transition_path": str(npz), "transition_sha256": "wrong" if corrupt and index == 0 else sha256_file(npz)})
        return pd.DataFrame(rows), [], []
    return execute


def test_reuses_exact_old_prefix_and_balances_192_new_episodes(workspace, monkeypatch):
    old, out, opponents, behaviors = workspace
    calls = []
    before = {path.name: sha256_file(path) for path in old.iterdir()}
    monkeypatch.setattr(d.execution, "execute", extension_executor(calls))
    receipt = d.generate_training(opponents, workers=2)
    assert receipt["episodes"] == 480 and receipt["transitions"] == 34560
    assert receipt["reused_episodes"] == 288 and receipt["new_episodes"] == 192
    assert receipt["scenario_counts"] == {s: 96 for s in c.SCENARIOS}
    with np.load(old / "train_transitions.npz") as source, np.load(out / "train_transitions.npz") as combined:
        assert set(combined.files) == set(d.ARRAY_KEYS)
        for key in source.files:
            np.testing.assert_array_equal(combined[key][:20736], source[key])
        assert combined["episode_id"][20736] == "288:0"
        assert combined["episode_id"][-1] == "479:47"
    sidecar = pd.read_csv(out / "trajectory_episodes.csv", dtype={"player_id": str})
    assert sidecar.iloc[1].scenario == "LocalPID" and sidecar.iloc[1].source_scenario == "PID"
    assert sidecar.iloc[288].source_episode_id == "new:0:0"
    for local, task in enumerate(calls[0]):
        di, within = divmod(local, 48 * 2)
        pi, scene = divmod(within, 2)
        assert task["target_spec"] == behaviors[(pi + scene + 3 + di) % 6]["spec"]
        assert task["seed"] == c.SEEDS[di]
    monkeypatch.setattr(d.execution, "execute", lambda *a, **kw: pytest.fail("completed data must be reused"))
    assert d.generate_training(opponents) == receipt
    assert before == {path.name: sha256_file(path) for path in old.iterdir()}


def test_changed_legacy_opponents_or_indices_cannot_be_reused(workspace, monkeypatch):
    old, _, opponents, _ = workspace
    monkeypatch.setattr(d.execution, "execute", lambda *a, **kw: pytest.fail("must fail before execution"))
    with pytest.raises(ValueError, match="unchanged"):
        d.generate_training({**opponents, "fixed": {"kind": "constant", "alpha": 4.}})
    frame = pd.read_csv(old / "behavior_episodes.csv")
    frame.loc[0, "day_id"] = "period-24"
    frame.to_csv(old / "behavior_episodes.csv", index=False)
    status = c.read(old / "behavior_episodes.status.json")
    status["csv_sha256"] = sha256_file(old / "behavior_episodes.csv")
    c.dump(old / "behavior_episodes.status.json", status)
    with pytest.raises(ValueError, match="source index"):
        d.generate_training(opponents)


def test_corrupt_extension_or_failed_rollout_never_produces_training_data(workspace, monkeypatch):
    _, out, opponents, _ = workspace
    monkeypatch.setattr(d.execution, "execute", extension_executor([], corrupt=True))
    with pytest.raises(ValueError, match="NPZ identity"):
        d.generate_training(opponents)
    assert not (out / "train_transitions.npz").exists()
    monkeypatch.setattr(d.execution, "execute", lambda *a, **kw: (pd.DataFrame(), [{"status": "failed"}], []))
    with pytest.raises(RuntimeError, match="must complete"):
        d.generate_training(opponents)


def training_input(out):
    save_npz(out / "train_transitions.npz", arrays([f"episode:{i}" for i in range(480)]))
    sidecar = out / "trajectory_episodes.csv"
    pd.DataFrame({"episode_index": range(480), "scenario": list(c.SCENARIOS) * 96}).to_csv(sidecar, index=False)
    c.dump(out / "trajectory_provenance.json", {"status": "completed", "test_used": False, "episodes": 480,
        "days": list(c.TRAIN_DAYS), "sha256": sha256_file(out / "train_transitions.npz"),
        "episode_metadata_path": str(sidecar.resolve()), "episode_metadata_sha256": sha256_file(sidecar),
        "scenario_counts": {s: 96 for s in c.SCENARIOS}})


def fake_trainer(calls, fail_once=None):
    def train(data_path, output_dir, *, algorithms, seeds, steps, checkpoint_steps, configs, batch_size, num_threads):
        algorithm, seed = algorithms[0], seeds[0]
        variant = configs[algorithm][0]
        calls.append((algorithm, seed, dict(variant), output_dir))
        assert steps == 1800 and checkpoint_steps == (600, 1800) and batch_size == 64 and num_threads == 2
        assert not output_dir.exists() or not any(output_dir.iterdir())
        model_dir = output_dir / algorithm / f"config_0_seed_{seed}"
        model_dir.mkdir(parents=True)
        if fail_once is not None and not fail_once[0]:
            fail_once[0] = True
            (model_dir / "partial.pt").write_bytes(b"interrupted-original")
            raise RuntimeError("interrupted run")
        metadata = {"source_sha256": sha256_file(data_path), "episodes": 480, "transitions": 34560,
                    "action_transforms": [variant["action_transform"]]}
        records = []
        for step in checkpoint_steps:
            checkpoint = model_dir / f"step_{step}.pt"
            checkpoint.write_text(json.dumps([algorithm, seed, variant, step]))
            records.append({"algorithm": algorithm, "seed": seed, "config": dict(variant), "metadata": metadata,
                "step": step, "config_index": 0, "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint), "parameter_count": 100, "training_seconds": step / 600})
        c.dump(model_dir / "training.json", {"status": "completed", "seconds": 3., "parameters": 100,
               "config": variant, "seed": seed, "history": [{"step": 1800, "losses": [.1]}]})
        c.dump(output_dir / "training_metadata.json", metadata)
        c.dump(output_dir / "failures.json", [])
        c.dump(output_dir / "candidates.json", records)
        return records
    return train


def test_24_from_scratch_runs_with_fair_checkpoint_budgets_and_complete_resume(workspace, monkeypatch):
    _, out, _, _ = workspace
    training_input(out)
    calls = []
    monkeypatch.setattr(d, "train_candidates", fake_trainer(calls))
    records = d.train_models()
    assert len(calls) == 24 and len(records) == 48
    assert {r["config_index"] for r in records} == {0, 1}
    assert all(r["origin_config_index"] == 0 and r["training_origin"] == "from_scratch" for r in records)
    for algorithm in c.NEURAL:
        subset = [r for r in records if r["algorithm"] == algorithm]
        assert len(subset) == 8
        assert {r["step"] for r in subset} == {600, 1800}
        assert {r["seed"] for r in subset} == set(c.SEEDS)
        assert {r["config"]["learning_rate"] for r in subset} == ({1e-4, 3e-4} if algorithm == "CQL" else {3e-4, 1e-3})
        assert {r["config"]["action_transform"] for r in subset} == ({"linear_scaled"} if algorithm == "CQL" else {"log1p"})
    assert d.train_models() == records and len(calls) == 24
    status = c.read(out / "models/training_status.json")
    assert status["status"] == "completed" and status["completed_runs"] == 24


def test_incomplete_run_restarts_only_it_and_preserves_partial_attempt(workspace, monkeypatch):
    _, out, _, _ = workspace
    training_input(out)
    monkeypatch.setattr(c, "NEURAL", ("DT",))
    calls, failed = [], [False]
    monkeypatch.setattr(d, "train_candidates", fake_trainer(calls, failed))
    with pytest.raises(RuntimeError, match="1 neural runs incomplete"):
        d.train_models()
    assert len(calls) == 4
    partial = calls[0][3] / "DT" / f"config_0_seed_{c.SEEDS[0]}" / "partial.pt"
    original = partial.read_bytes()
    records = d.train_models()
    assert len(calls) == 5 and calls[-1][3].name == "attempt_1"
    assert partial.read_bytes() == original
    assert len(records) == 8
    assert c.read(calls[0][3].with_name("attempt_0.status.json"))["status"] == "failed"


def test_corrupt_completed_checkpoint_is_preserved_and_retrained_separately(workspace, monkeypatch):
    _, out, _, _ = workspace
    training_input(out)
    monkeypatch.setattr(c, "NEURAL", ("BC",))
    calls = []
    monkeypatch.setattr(d, "train_candidates", fake_trainer(calls))
    records = d.train_models()
    corrupt = Path(records[0]["checkpoint"])
    corrupt.write_bytes(b"damaged-old-weight")
    fresh = d.train_models()
    assert len(calls) == 5
    assert corrupt.read_bytes() == b"damaged-old-weight"
    assert all(Path(r["checkpoint"]) != corrupt for r in fresh)
    assert any(r["attempt"] == "attempt_1" for r in fresh)
    # Input/config changes never silently reuse or overwrite this run identity.
    monkeypatch.setattr(d, "_training_code_hashes", lambda: {"changed": "code"})
    with pytest.raises(ValueError, match="Immutable record differs"):
        d.train_models()


def test_changed_training_sidecar_rejects_before_starting_any_run(workspace, monkeypatch):
    _, out, _, _ = workspace
    training_input(out)
    (out / "trajectory_episodes.csv").write_text("changed metadata", encoding="utf-8")
    monkeypatch.setattr(d, "train_candidates", lambda *a, **kw: pytest.fail("must reject before training"))
    with pytest.raises(ValueError, match="balanced five-scenario data"):
        d.train_models()


def test_changed_completion_receipt_is_not_trusted_over_original_logs(workspace, monkeypatch):
    _, out, _, _ = workspace
    training_input(out)
    monkeypatch.setattr(c, "NEURAL", ("BC",))
    calls = []
    monkeypatch.setattr(d, "train_candidates", fake_trainer(calls))
    d.train_models()
    status = calls[0][3].with_name("attempt_0.status.json")
    changed = c.read(status)
    changed["candidates"][0]["config_index"] = 99
    c.dump(status, changed)
    fresh = d.train_models()
    assert len(calls) == 5 and calls[-1][3].name == "attempt_1"
    assert {r["config_index"] for r in fresh} == {0, 1}
    assert c.read(status) == changed
