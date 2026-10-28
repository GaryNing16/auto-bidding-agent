"""Reproduction orchestration checks; never retrain the completed experiment."""
import json
from pathlib import Path
import numpy as np
import pytest

from scripts import train_cql_linear as script


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "train_transitions.npz"
    np.savez(source, states=np.zeros((144, 18)), next_states=np.zeros((144, 18)),
             actions=np.ones(144), rewards=np.ones(144),
             dones=np.tile([False]*71+[True], 2), episode_id=np.repeat([0, 1], 72))
    _json(tmp_path / "trajectory_provenance.json", {"days": list(script.TRAINING_DAYS),
        "test_used": False, "sha256": script.sha256_file(source), "transitions": 144, "episodes": 2})
    _json(tmp_path / "protocol.json", {"executed_split": {"train": list(script.TRAINING_DAYS),
        "validation": ["period-21"], "test": ["period-24"]}, "training_seeds": list(script.TRAINING_SEEDS)})
    (tmp_path / "behavior_episodes.csv").write_text("synthetic completed fixture", encoding="utf-8")
    _json(tmp_path / "behavior_episodes.status.json", {"status": "completed", "episodes": 2,
        "csv_sha256": script.sha256_file(tmp_path / "behavior_episodes.csv")})
    checkpoint = tmp_path / "models/original.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"protected original fixture")
    _json(tmp_path / "models/candidates.json", [{"checkpoint": str(checkpoint),
        "checkpoint_sha256": script.sha256_file(checkpoint)}])
    return tmp_path


def test_recipe_and_protected_assets(source, monkeypatch):
    calls = []
    def fake_train(data, destination, **kwargs):
        calls.append((data, destination, kwargs))
        destination.mkdir()
        records = [{"seed": seed, "step": step} for seed in script.TRAINING_SEEDS
                   for step in script.CHECKPOINT_STEPS]
        _json(destination / "candidates.json", records)
        return records
    monkeypatch.setattr(script, "train_candidates", fake_train)
    before = (source / "models/candidates.json").read_bytes()
    record = script.run(source)
    assert record["status"] == "completed" and record["original_assets_unchanged"]
    assert record["test_used"] is False
    assert (source / "models/candidates.json").read_bytes() == before
    assert calls == [(source / "train_transitions.npz", source / "models_cql_linear", {
        "algorithms": ("CQL",), "configs": {"CQL": [{"learning_rate": 1e-4,
        "action_transform": "linear_scaled"}]}, "seeds": (20261028, 20261029),
        "steps": 600, "checkpoint_steps": (200, 600), "num_threads": 2})]
    with pytest.raises(ValueError, match="existing"):
        script.run(source)


def test_changed_training_source_refused_before_output(source):
    with (source / "train_transitions.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="NPZ SHA256"):
        script.preflight(source)
    assert not (source / "models_cql_linear").exists()
    assert not (source / "models_cql_linear_run.json").exists()


def test_frozen_experiment_cannot_add_candidates(source):
    _json(source / "selected_frozen.json", {"selected": []})
    with pytest.raises(ValueError, match="after selection/test"):
        script.preflight(source)


def test_original_weight_mutation_invalidates_run(source, monkeypatch):
    def changing_train(data, destination, **kwargs):
        (source / "models/original.pt").write_bytes(b"unexpected fixture mutation")
        return [{"seed": seed, "step": step} for seed in script.TRAINING_SEEDS
                for step in script.CHECKPOINT_STEPS]
    monkeypatch.setattr(script, "train_candidates", changing_train)
    with pytest.raises(ValueError, match="original checkpoint/catalog changed"):
        script.run(source)
    record = json.loads((source / "models_cql_linear_run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed" and record["valid"] is False
