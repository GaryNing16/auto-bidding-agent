"""Frozen evaluation integrity, partial coverage and paired comparison tests."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from harmony_baselines.data import MatrixDay, MatrixSlot
from harmony_baselines.five_scenarios.evaluator import run_episode
from harmony_bidding.schema import Advertiser, fingerprint
from harmony_rl.evaluate import _validate_result
from harmony_rl.report import complete_algorithms, paired_gains, _verify_pair_conditions


def _episode():
    advertisers = (Advertiser("0", "0", 5.), Advertiser("1", "1", 7.))
    slots = tuple(MatrixSlot(np.array([[.4, .3], [.1, .8]]), np.ones((2, 2), dtype=bool),
                             np.array([f"{t}-a", f"{t}-b"]), np.array([0, 1])) for t in range(72))
    day = MatrixDay("synthetic-test", advertisers, slots, "synthetic-fingerprint", {})
    opponents = {"fixed": {"kind": "constant", "alpha": 1.}, "pid": {"kind": "pid"},
                 "auctionnet_pid": {"kind": "constant", "alpha": 1.}}
    task = dict(day_id=day.day_id, target="0", target_spec={"kind": "constant", "alpha": 10.},
                scenario="Fixed", opponents=opponents, seed=10, algorithm="PPO-Raw", training_seed=3)
    args = {k: v for k, v in task.items() if k != "day_id"}
    row, traces, _ = run_episode(day, **args)
    row["policy_id"] = "sealed-policy"
    identity = dict(policy_id="sealed-policy", dataset_fingerprint=day.fingerprint)
    return dict(status="completed", identity=identity, row=row, slots=traces), task, identity


def test_frozen_result_requires_complete_consistent_72_slots():
    result, task, identity = _episode()
    _validate_result(result, task, identity)
    broken = deepcopy(result)
    broken["slots"].pop()
    with pytest.raises(ValueError, match="72 slots"):
        _validate_result(broken, task, identity)
    broken = deepcopy(result)
    broken["slots"][2]["cost"] += 1.
    with pytest.raises(ValueError, match="cost mismatch"):
        _validate_result(broken, task, identity)
    broken = deepcopy(result)
    broken["row"]["training_seed"] += 1
    with pytest.raises(ValueError, match="task mismatch"):
        _validate_result(broken, task, identity)


def test_partial_algorithm_excluded_even_if_plausible_means_exist():
    tasks = [dict(algorithm="PPO-Raw", scenario="Fixed", day_id="d", target=p, seed=1, training_seed=2)
             for p in ("0", "1")]
    protocol = dict(tasks=tasks, candidates=[dict(algorithm="PPO-Raw")])
    frame = pd.DataFrame([dict(algorithm="PPO-Raw", scenario="Fixed", day_id="d", player_id="0", seed=1, training_seed=2)])
    valid, coverage = complete_algorithms(frame, protocol)
    assert not valid
    assert coverage.loc[coverage.algorithm == "PPO-Raw", "missing_episodes"].item() == 1
    complete = pd.concat([frame, frame.assign(player_id="1")], ignore_index=True)
    assert complete_algorithms(complete, protocol)[0] == ["PPO-Raw"]
    with pytest.raises(ValueError, match="Duplicate"):
        complete_algorithms(pd.concat([frame, frame]), protocol)
    with pytest.raises(ValueError, match="outside"):
        complete_algorithms(frame.assign(seed=999), protocol)


def test_pairing_averages_training_seed_and_never_epsilon_fills_zero():
    from harmony_baselines.five_scenarios.common import ALGORITHMS
    records = []
    for a in ALGORITHMS:
        for player, value in [("0", 0.), ("1", 4.)]:
            records.append(dict(algorithm=a, scenario="Fixed", day_id="d", player_id=player,
                                seed=10, training_seed=0, total_value=value))
    for train, value in [(1, 6.), (2, 10.)]:
        for player in ("0", "1"):
            records.append(dict(algorithm="PPO-Raw", scenario="Fixed", day_id="d", player_id=player,
                                seed=10, training_seed=train, total_value=value))
    pairs = paired_gains(pd.DataFrame(records))
    assert pairs.loc[pairs.player_id == "0", "relative_gain"].isna().all()
    assert (pairs.loc[pairs.player_id == "1", "relative_gain"] == 1.).all()
    assert (pairs.total_value == 8.).all()
    broken = pd.DataFrame(records)
    broken = broken[~((broken.algorithm == "PPO-Raw") & (broken.player_id == "0"))]
    with pytest.raises(ValueError, match="Unpaired"):
        paired_gains(broken)


def test_different_opponent_or_budget_cannot_enter_comparison():
    record = dict(day_id="d", player_id="0", scenario="Fixed", seed=10,
                  dataset_fingerprint="data", initial_budget=20., assignment_fingerprint="assignment",
                  rules_version="rules", opponent_fingerprint="opponents")
    frame = pd.DataFrame([record, record])
    _verify_pair_conditions(frame)
    for column, value in [("initial_budget", 30.), ("assignment_fingerprint", "other"),
                          ("dataset_fingerprint", "other"), ("rules_version", "other")]:
        broken = frame.copy()
        broken.loc[1, column] = value
        with pytest.raises(ValueError, match="Incomparable"):
            _verify_pair_conditions(broken)


def test_report_keeps_every_baseline_and_missing_ppo_visible(tmp_path, monkeypatch):
    from harmony_baselines.data import sha256_file
    from harmony_baselines.five_scenarios.common import ALGORITHMS, SCENARIOS, dump
    from harmony_rl import report
    payload, task, _ = _episode()
    baseline, ppo, slots, tasks = [], [], [], []
    for scenario in SCENARIOS:
        for algorithm in ALGORITHMS:
            baseline.append({**payload["row"], "algorithm": algorithm, "scenario": scenario, "training_seed": 0})
        ppo.append({**payload["row"], "scenario": scenario})
        slots.extend([{**s, "scenario": scenario} for s in payload["slots"]])
        tasks.append({**task, "scenario": scenario})
    directory = tmp_path / "fixture-evaluation"
    directory.mkdir()
    pd.DataFrame(ppo).to_csv(directory / "episode_results.csv", index=False)
    pd.DataFrame(slots).to_csv(directory / "slots.csv", index=False)
    dump(directory / "protocol.json", dict(candidates=[dict(algorithm="PPO-Raw", training_seed=3)], tasks=tasks,
        roles=["0"], scenarios=list(SCENARIOS), days=[task["day_id"]], seeds=[10]))
    dump(directory / "status.json", dict(completed_episodes=5, expected_episodes=5,
        result_sha256=sha256_file(directory / "episode_results.csv"),
        slots_sha256=sha256_file(directory / "slots.csv"), protocol_sha256=sha256_file(directory / "protocol.json")))
    monkeypatch.setattr(report, "audit_baseline", lambda: (pd.DataFrame(baseline), {}, {"episodes": 50}))
    monkeypatch.setattr(report, "_plots", lambda *a: [])
    monkeypatch.setattr(report, "_reference_slots", lambda x: x)
    result = report.generate_report(directory)
    table = pd.read_csv(directory / "report/main_comparison.csv")
    assert len(table) == 16
    assert set(ALGORITHMS) <= set(table.algorithm)
    assert result["complete_ppo_algorithms"] == ["PPO-Raw"]
    assert not result["full_evaluation_scope"]
    text = (directory / "report/research_report.md").read_text(encoding="utf-8")
    assert "未完成" in text and "只有1个训练seed" in text


def test_ablation_rejects_different_actual_interaction_budgets_and_unverified_sources():
    from harmony_rl.report import _ablation_status
    candidates = [dict(algorithm=a, training_seed=11) for a in ("PPO-Raw", "PPO-LSTM")]
    records = [dict(algorithm=a, training_seed=11, completed_episodes=96, environment_steps=96*72,
                    training_evidence_verified=True, verified_config={"train_days": ["train-day"], "batch_episodes": 8})
               for a in ("PPO-Raw", "PPO-LSTM")]
    assert _ablation_status("PPO-Raw", "PPO-LSTM", records, candidates) == "matched"
    broken = deepcopy(records)
    broken[1].update(completed_episodes=48, environment_steps=48*72)
    assert _ablation_status("PPO-Raw", "PPO-LSTM", broken, candidates) == "训练预算不匹配"
    broken = deepcopy(records)
    broken[1]["training_evidence_verified"] = False
    assert _ablation_status("PPO-Raw", "PPO-LSTM", broken, candidates) == "训练来源未核验"
    broken = deepcopy(records)
    broken[1]["verified_config"]["train_days"] = ["different-market"]
    assert _ablation_status("PPO-Raw", "PPO-LSTM", broken, candidates) == "非消融配置不匹配"


def test_training_budget_evidence_cross_checks_committed_model_not_only_status(tmp_path):
    import torch
    from harmony_baselines.data import sha256_file
    from harmony_baselines.five_scenarios.common import dump
    from harmony_rl.report import _training_records
    config = dict(variant="E1", smoke=False, batch_episodes=8, workers=1,
                  learning_rate=.0003, ppo={"gamma": 1., "gae_lambda": .95})
    payload = dict(config=config, provenance={"smoke": False}, episode_count=96,
                   policy_version=12, training_seed=11, history=[{"requests": 1000}])
    torch.save(payload, tmp_path / "last.pt")
    torch.save(payload, tmp_path / "best.pt")
    dump(tmp_path / "config.json", config)
    dump(tmp_path / "provenance.json", payload["provenance"])
    status = dict(algorithm="PPO-Raw", completed_episodes=96, environment_steps=96*72,
                  ppo_updates=12, training_seed=11, training_requests=1000, complete=True, validated=True)
    dump(tmp_path / "training_status.json", status)
    dump(tmp_path / "selection.json", dict(checkpoint_sha256=sha256_file(tmp_path / "best.pt"),
        test_used=False, episodes=96, update=12))
    candidates = [dict(algorithm="PPO-Raw", training_seed=11,
                       spec={"kwargs": {"checkpoint": str(tmp_path / "best.pt")}})]
    assert _training_records(tmp_path, candidates)[0]["training_evidence_verified"]
    dump(tmp_path / "training_status.json", {**status, "completed_episodes": 192, "environment_steps": 192*72})
    result = _training_records(tmp_path, candidates)[0]
    assert not result["training_evidence_verified"]
    assert any("counters" in reason for reason in result["training_evidence_errors"])


def test_prefix_validation_recomputes_errors_and_requires_exact_case_coverage(tmp_path):
    from harmony_baselines.data import sha256_file
    from harmony_baselines.five_scenarios.common import dump
    from harmony_rl.report import _prefix_evidence
    checkpoint = tmp_path / "fixture-checkpoint.pt"
    checkpoint.write_bytes(b"frozen test fixture")
    digest = sha256_file(checkpoint)
    training = [dict(variant="E3", training_seed=11, training_evidence_verified=True,
                    checkpoint_sha256=digest,
                    verified_config={"validation_days": ["v"], "validation_roles": ["0"], "scenarios": ["Fixed"]})]
    record = dict(market_id="v", player_id="0", scenario="Fixed", prefix_predictions=[2.]*72, realized_return=3.)
    payload = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, data_split="validation",
                   actor_updated=False, lstm_updated=False, complete=True, completed=1, expected=1,
                   records=[record], mse_by_prefix=[1.]*72, bias_by_prefix=[-1.]*72)
    dump(tmp_path / "prefix_validation.json", payload)
    summary, _, _ = _prefix_evidence(tmp_path, training, tmp_path)
    assert summary[0]["mse_1"] == 1. and summary[0]["bias_72"] == -1.
    dump(tmp_path / "prefix_validation.json", {**payload, "mse_by_prefix": [0.]*72})
    with pytest.raises(ValueError, match="summary"):
        _prefix_evidence(tmp_path, training, tmp_path)
    dump(tmp_path / "prefix_validation.json", {**payload, "records": [record, record], "completed": 2})
    with pytest.raises(ValueError, match="duplicated"):
        _prefix_evidence(tmp_path, training, tmp_path)


def test_learning_curve_uses_committed_checkpoint_not_interrupted_tail(tmp_path):
    import torch
    from harmony_baselines.five_scenarios.common import dump
    from harmony_rl.report import _committed_training_records
    history = [dict(episodes=8, update=1, mean_raw_return=3.)]
    torch.save(dict(episode_count=8, policy_version=1, history=history), tmp_path / "last.pt")
    path = tmp_path / "training_history.json"
    dump(path, history + [dict(episodes=16, update=2, mean_raw_return=99.)])
    rows, evidence = _committed_training_records(path)
    assert rows == history
    assert evidence["plotted_source"] == "committed_checkpoint_history"


def test_resumed_run_can_keep_earlier_selected_checkpoint_with_original_runtime_budget(tmp_path):
    import torch
    from harmony_baselines.data import sha256_file
    from harmony_baselines.five_scenarios.common import dump
    from harmony_rl.report import _training_records
    old_config = dict(variant="E1", smoke=False, episodes=96, max_seconds=1800., workers=4,
                      batch_episodes=8, learning_rate=.0003, ppo={"gamma": 1., "gae_lambda": .95})
    current = {**old_config, "episodes": 192, "max_seconds": 3600., "workers": 8}
    best = dict(config=old_config, provenance={"smoke": False}, episode_count=48,
                policy_version=6, training_seed=11, history=[{"requests": 500}])
    last = {**best, "config": current, "episode_count": 192, "policy_version": 24, "history": [{"requests": 2000}]}
    torch.save(last, tmp_path / "last.pt")
    torch.save(best, tmp_path / "best.pt")
    dump(tmp_path / "config.json", current)
    dump(tmp_path / "provenance.json", last["provenance"])
    dump(tmp_path / "training_status.json", dict(algorithm="PPO-Raw", completed_episodes=192, environment_steps=192*72,
        ppo_updates=24, training_seed=11, training_requests=2000, complete=True, validated=True))
    dump(tmp_path / "selection.json", dict(checkpoint_sha256=sha256_file(tmp_path / "best.pt"),
        test_used=False, episodes=48, update=6))
    candidates = [dict(algorithm="PPO-Raw", training_seed=11, spec={"kwargs": {"checkpoint": str(tmp_path / "best.pt")}})]
    result = _training_records(tmp_path, candidates)[0]
    assert result["training_evidence_verified"], result["training_evidence_errors"]
    assert result["completed_episodes"] == 192 and result["selected_at_episode"] == 48


def _qualified_checkpoint_fixture(directory, *, smoke=False):
    """Minimal trusted-local artifact exercising selection eligibility, not scores."""
    import torch
    from harmony_baselines.data import sha256_file
    from harmony_baselines.five_scenarios.common import dump
    directory.mkdir()
    path = directory / "best.pt"
    payload = dict(format="harmony-ppo-lstm-v1", config={"variant": "E1", "seed": 7, "smoke": smoke},
        provenance={"smoke": smoke, "test_read_during_training": False}, training_seed=7,
        policy_version=2, episode_count=8, best_validation=3.,
        validation_history=[dict(update=2, episodes=8, mean_value=3., episodes_evaluated=5)])
    torch.save(payload, path)
    dump(directory / "selection.json", dict(checkpoint_sha256=sha256_file(path), test_used=False,
        update=2, episodes=8, value=3.))
    dump(directory / "training_status.json", dict(algorithm="PPO-Raw", training_seed=7,
        complete=True, validated=True, requested_episodes=16, completed_episodes=16, checkpoint=str(path.resolve())))
    return path


def test_formal_candidates_require_real_selected_checkpoint_and_matching_seal(tmp_path):
    from harmony_baselines.five_scenarios.common import read, dump
    from harmony_rl.evaluate import checkpoint_candidates, qualify_checkpoint
    selected = _qualified_checkpoint_fixture(tmp_path / "real")
    assert checkpoint_candidates([selected])[0]["algorithm"] == "PPO-Raw"
    smoke = _qualified_checkpoint_fixture(tmp_path / "smoke", smoke=True)
    with pytest.raises(ValueError, match="smoke"):
        checkpoint_candidates([smoke])
    with pytest.raises(ValueError, match="best.pt"):
        qualify_checkpoint(selected.parent / "last.pt")
    selection_path = selected.parent / "selection.json"
    original = read(selection_path)
    for updates, message in [({"checkpoint_sha256": "changed"}, "SHA"),
                             ({"update": 99}, "update"), ({"episodes": 9}, "episodes"),
                             ({"value": 999.}, "score"), ({"test_used": True}, "separation")]:
        dump(selection_path, {**original, **updates})
        with pytest.raises(ValueError, match=message):
            qualify_checkpoint(selected)
    dump(selection_path, original)
    selection_path.rename(selected.parent / "selection.saved.json")
    with pytest.raises(ValueError, match="Missing"):
        qualify_checkpoint(selected)


def test_candidate_manifest_cannot_bypass_ppo_qualification_before_test_reads(tmp_path, monkeypatch):
    from harmony_rl import evaluate
    from harmony_baselines.five_scenarios.common import read, dump
    selected = _qualified_checkpoint_fixture(tmp_path / "real")
    candidate = evaluate.checkpoint_candidates([selected])[0]
    def no_market_read(*args, **kwargs):
        raise AssertionError("invalid candidate reached market access")
    monkeypatch.setattr(evaluate, "cached_day", no_market_read)
    monkeypatch.setattr(evaluate, "audit_baseline", no_market_read)
    wrong = {**candidate, "training_seed": 999}
    with pytest.raises(ValueError, match="label/seed"):
        evaluate.run_evaluation([wrong], tmp_path / "invalid", max_seconds=1)
    status_path = selected.parent / "training_status.json"
    status = read(status_path)
    dump(status_path, {**status, "complete": False})
    with pytest.raises(ValueError, match="not complete"):
        evaluate.run_evaluation([candidate], tmp_path / "incomplete", max_seconds=1)
    dump(status_path, {**status, "complete": False, "termination": "early_stopping", "requested_episodes": 32})
    _, evidence = evaluate.qualify_checkpoint(selected)
    assert evidence["training_termination"] == "early_stopping"
    assert not evidence["complete_interaction_budget"]
    assert evidence["selected_episodes"] == 8 < evidence["completed_training_episodes"] == 16
