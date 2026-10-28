"""Pipeline integrity/selection regressions using only synthetic local records."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

from harmony_baselines.classical import spec
from harmony_baselines.five_scenarios import common, pipeline as p
from harmony_baselines.five_scenarios.evaluator import opponent_specs
from harmony_bidding.schema import Advertiser, fingerprint


@pytest.fixture
def history(tmp_path, monkeypatch):
    root, old, out = tmp_path/"root", tmp_path/"root/old", tmp_path/"root/new"
    old.mkdir(parents=True)
    out.mkdir()
    monkeypatch.setattr(p, "ROOT", root)
    monkeypatch.setattr(p, "OLD", old)
    monkeypatch.setattr(p, "OUT", out)
    (root/"source.py").write_text("# frozen source\n", encoding="utf-8")
    (old/"checkpoint.pt").write_bytes(b"frozen weights")
    (old/"online_lp_table.csv").write_text("cost,alpha\n1,15\n", encoding="utf-8")
    (old/"episode_results.csv").write_text("synthetic,results\n1,2\n", encoding="utf-8")
    (old/"neural_validation.csv").write_text("synthetic,validation\n1,3\n", encoding="utf-8")
    common.dump(old/"opponents_frozen.json", {"fixed": spec("FixedAlpha", alpha=2.), "pid": spec("LocalPID")})
    common.dump(old/"protocol.json", {"status": "completed"})
    frozen = {"checkpoints": {str(old/"checkpoint.pt"): p.sha256_file(old/"checkpoint.pt")},
        "code_sha256": {"source.py": p.sha256_file(root/"source.py")},
        "opponents_sha256": p.sha256_file(old/"opponents_frozen.json"),
        "lp_table_sha256": p.sha256_file(old/"online_lp_table.csv")}
    common.dump(old/"selected_frozen.json", frozen)

    def reseal_manifest():
        paths = sorted(old.glob("*")) + [root/"source.py"]
        entries = [{"path": str(x.resolve()), "sha256": p.sha256_file(x),
                    "size": x.stat().st_size, "mtime_ns": x.stat().st_mtime_ns} for x in paths]
        common.dump(out/"historical_assets_manifest.json", {"files": entries})
        common.dump(out/"historical_audit.json", {"status": "passed", "valid": True,
            "mismatches": [], "manifest_sha256": p.sha256_file(out/"historical_assets_manifest.json")})

    reseal_manifest()
    return SimpleNamespace(root=root, old=old, out=out, frozen=frozen, reseal=reseal_manifest)


def test_full_historical_verification_uses_only_protected_fixture_files(history):
    assert p._verify_historical(full=True) == history.frozen
    result = common.read(history.out/"historical_integrity_preflight.json")
    assert result["status"] == "passed" and result["mismatches"] == []


@pytest.mark.parametrize("filename", ["online_lp_table.csv", "opponents_frozen.json", "episode_results.csv",
    "neural_validation.csv", "selected_frozen.json", "checkpoint.pt", "source.py"])
def test_historical_assets_cannot_change_before_reuse(history, filename):
    path = history.root/filename if filename == "source.py" else history.old/filename
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError):
        p._verify_historical()


@pytest.mark.parametrize("field, value", [("status", "failed"), ("valid", False), ("mismatches", [{"reason": "bad"}])])
def test_failed_audit_never_authorizes_history_reuse(history, field, value):
    audit = common.read(history.out/"historical_audit.json")
    audit[field] = value
    common.dump(history.out/"historical_audit.json", audit)
    with pytest.raises(ValueError, match="audit"):
        p._verify_historical()


def test_protection_manifest_cannot_be_rewritten_without_audit_digest(history):
    common.dump(history.out/"historical_assets_manifest.json", {"files": []})
    with pytest.raises(ValueError, match="manifest"):
        p._verify_historical()


@pytest.mark.parametrize("filename", ["online_lp_table.csv", "opponents_frozen.json"])
def test_historical_freeze_still_checks_assets_even_if_new_manifest_contains_changed_bytes(history, filename):
    (history.old/filename).write_bytes(b"different protected bytes")
    history.reseal()
    with pytest.raises(ValueError, match="freeze"):
        p._verify_historical()


def write_protocol(history):
    protocol = {"historical_selection_sha256": p.sha256_file(history.old/"selected_frozen.json"),
                "historical_results_sha256": p.sha256_file(history.old/"episode_results.csv"),
                "scenarios": list(p.SCENARIOS), "algorithms": list(p.ALGORITHMS)}
    common.dump(history.out/"protocol.json", protocol)
    return protocol


def test_experimental_cli_requires_predeclared_protocol_before_dispatch(history, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pipeline", "opponents"])
    monkeypatch.setattr(p, "calibrate_opponents", lambda *args: pytest.fail("No dispatch before prepare"))
    with pytest.raises(ValueError, match="prepare"):
        p.main()


@pytest.mark.parametrize("field", ["historical_selection_sha256", "historical_results_sha256", "scenarios", "algorithms"])
def test_protocol_rejects_changed_historical_identity_or_scenario_contract(history, field):
    protocol = write_protocol(history)
    assert p._require_protocol() == protocol
    protocol[field] = [] if isinstance(protocol[field], list) else "changed"
    common.dump(history.out/"protocol.json", protocol)
    with pytest.raises(ValueError, match="protocol"):
        p._require_protocol()


def candidate(name, seed=0):
    result = {"algorithm": name, "spec": spec("FixedAlpha", alpha=1.)}
    if seed:
        result["training_seed"] = seed
    return result


def candidate_data(candidates, values=None):
    tasks = p.tasks_for(candidates, {"fixed": spec("FixedAlpha", alpha=1.)},
                       ["validation-only"], ["0", "8"], p.SCENARIOS, [11])
    rows = []
    for i, t in enumerate(tasks):
        value = (values or {}).get((t["algorithm"], t["training_seed"]), 2.)
        rows.append(dict(task_index=i, algorithm=t["algorithm"], training_seed=t["training_seed"],
            day_id=t["day_id"], scenario=t["scenario"], player_id=t["target"], seed=t["seed"],
            total_value=value, budget_utilization=.5))
    return tasks, pd.DataFrame(rows)


def test_single_seed_candidate_cannot_win_against_complete_pair():
    a, b = p.SEEDS
    candidates = [candidate("BC|0|600", a), candidate("BC|1|600", a), candidate("BC|1|600", b)]
    tasks, frame = candidate_data(candidates, {("BC|0|600", a): 100., ("BC|1|600", a): 1., ("BC|1|600", b): 5.})
    scores = p._candidate_scores(frame, candidates, tasks)
    assert not scores[0]["complete"] and "mean_V" not in scores[0]
    assert scores[1]["complete"] and scores[1]["mean_V"] == 3.
    chosen = p._select(scores, candidates)
    assert len(chosen) == 2 and {r["training_seed"] for r in chosen} == set(p.SEEDS)
    assert {r["validation_choice"] for r in chosen} == {"BC|1|600"}


def test_missing_paired_seed_result_is_not_imputed_zero():
    candidates = [candidate("CQL|2|600", seed) for seed in p.SEEDS]
    tasks, frame = candidate_data(candidates)
    result = p._candidate_scores(frame.iloc[:-1], candidates, tasks)[0]
    assert not result["complete"] and "mean_V" not in result


def test_legal_zero_value_seed_is_complete_and_included_in_mean():
    candidates = [candidate("CQL|2|600", seed) for seed in p.SEEDS]
    tasks, frame = candidate_data(candidates, {("CQL|2|600", p.SEEDS[0]): 0., ("CQL|2|600", p.SEEDS[1]): 4.})
    result = p._candidate_scores(frame, candidates, tasks)[0]
    assert result["complete"] and result["mean_V"] == 2.


def test_two_seed_catalog_does_not_accept_rows_relabelled_to_one_seed():
    candidates = [candidate("IQL|1|1800", seed) for seed in p.SEEDS]
    tasks, frame = candidate_data(candidates)
    frame["training_seed"] = p.SEEDS[0]
    try:
        result = p._candidate_scores(frame, candidates, tasks)[0]
    except ValueError:
        return
    assert not result["complete"]


def test_duplicate_success_row_cannot_change_candidate_weight():
    candidates = [candidate("BC|1|600", seed) for seed in p.SEEDS]
    tasks, frame = candidate_data(candidates)
    duplicate = frame.iloc[[0]].copy()
    duplicate["total_value"] = 999.
    frame = pd.concat([frame, duplicate], ignore_index=True)
    try:
        result = p._candidate_scores(frame, candidates, tasks)[0]
    except ValueError:
        return
    assert not result["complete"]


def test_primary_V_selection_and_declared_lexicographic_tie_break():
    candidates = [candidate("FixedAlpha|b"), candidate("FixedAlpha|a")]
    scores = [{"candidate": c["algorithm"], "algorithm": "FixedAlpha", "complete": True,
               "mean_V": 3., "mean_R": 9. if i == 0 else 1.} for i, c in enumerate(candidates)]
    assert p._select(scores, candidates)[0]["validation_choice"] == "FixedAlpha|a"
    assert p._select(scores, candidates, criterion="mean_R")[0]["validation_choice"] == "FixedAlpha|b"


def test_auctionnet_pid_search_keeps_upstream_default_and_shared_target_parameters(monkeypatch):
    monkeypatch.setattr(p, "base_alpha", lambda: 100.)
    candidates = p.auctionnet_pid_candidates()
    default = dict(initial_alpha=15., lower_threshold=.7, upper_threshold=1.1,
                   increase_factor=1.2, decrease_factor=.7)
    assert any(c["spec"]["kwargs"] == default for c in candidates)
    assert all("player_alphas" not in c["spec"]["kwargs"] for c in candidates)


def test_full_protocol_counts_keep_both_training_and_evaluation_seeds():
    selected = [candidate(algorithm) for algorithm in ("FixedAlpha", "LocalPID", "OnlineLP", "AuctionNetPID")]
    selected += [candidate(algorithm, seed) for algorithm in p.NEURAL for seed in p.SEEDS]
    old = [c for c in selected if c["algorithm"] != "AuctionNetPID"]
    new = [c for c in selected if c["algorithm"] == "AuctionNetPID"]
    empty_opponents = {}
    full = p.tasks_for(selected, empty_opponents, ["test-placeholder"], p.ROLES, p.SCENARIOS, p.SEEDS)
    additions = p.tasks_for(old, empty_opponents, ["test-placeholder"], p.ROLES, p.SCENARIOS[3:], p.SEEDS)
    additions += p.tasks_for(new, empty_opponents, ["test-placeholder"], p.ROLES, p.SCENARIOS, p.SEEDS)
    assert len(full) == 7680
    assert len(additions) == 3360
    keys = {(t["algorithm"], t["training_seed"], t["scenario"], t["target"], t["seed"]) for t in full}
    assert len(keys) == len(full)


@pytest.fixture
def completed_A(tmp_path, monkeypatch):
    out = tmp_path/"new"
    directory = out/"experiment_A"
    directory.mkdir(parents=True)
    (out/"experiment_B").mkdir()
    monkeypatch.setattr(p, "OUT", out)
    monkeypatch.setattr(p, "ROLES", ("0", "1"))
    monkeypatch.setattr(p, "SCENARIOS", ("Fixed", "LocalPID"))
    monkeypatch.setattr(p, "SEEDS", (11, 12))
    monkeypatch.setattr(p, "COMPARISON_DAYS", ("synthetic-comparison",))
    selected = [candidate("FixedAlpha")]
    opponents = {"fixed": spec("FixedAlpha", alpha=1.), "pid": spec("LocalPID", initial_alpha=2.),
                 "auctionnet_pid": common.pid_spec()}
    common.dump(out/"opponents_frozen.json", opponents)
    frozen = {"selected": selected, "policy_ids": {"FixedAlpha:0": p.policy_id(selected[0]["spec"])},
              "comparison_days": list(p.COMPARISON_DAYS), "independent_days": ["synthetic-independent"]}
    monkeypatch.setattr(p, "verify_freeze", lambda phase: deepcopy(frozen))
    monkeypatch.setattr(p, "cached_day", lambda day: SimpleNamespace(fingerprint="synthetic-market",
        advertisers=tuple(Advertiser(player, "0", 100.) for player in p.ROLES)))
    rows, slots = [], []
    for index, t in enumerate(p.tasks_for(selected, opponents, p.COMPARISON_DAYS, p.ROLES, p.SCENARIOS, p.SEEDS)):
        assignment = opponent_specs(list(p.ROLES), t["target"], t["scenario"], opponents, t["seed"]+1000003, t["day_id"])
        metadata = dict(task_index=index, algorithm=t["algorithm"], training_seed=0, day_id=t["day_id"],
            scenario=t["scenario"], player_id=t["target"], player_type="0", seed=t["seed"],
            policy_id=p.policy_id(t["target_spec"]), model_fingerprint=fingerprint(t["target_spec"]),
            assignment_fingerprint=fingerprint(assignment), opponent_fingerprint=fingerprint(opponents),
            dataset_fingerprint="synthetic-market", rules_version="harmony-A-B-local-v2", budget_scale=1.,
            completed=True, initial_budget=100., phase="experiment_A", provenance="new_execution",
            source_task=f"synthetic:{index}", source_result_path="synthetic-only")
        rows.append(dict(metadata, total_value=72., total_cost=72., wins=72, budget_utilization=.72))
        slots.extend(dict(metadata, time_slot=i, slot=i, alpha=1., cost=1., value=1., wins=1,
            remaining_budget=99.-i, cumulative_cost=i+1., cumulative_value=i+1.) for i in range(72))
    frame, slotframe = pd.DataFrame(rows), pd.DataFrame(slots)

    def save(frame=frame, slots=slotframe, **status_overrides):
        frame.to_csv(directory/"episode_results.csv", index=False)
        slots.to_csv(directory/"slots.csv", index=False)
        status = dict(status="completed", valid=True, phase="experiment_A", expected_episodes=8,
            completed_episodes=len(frame), result_sha256=p.sha256_file(directory/"episode_results.csv"),
            slots_sha256=p.sha256_file(directory/"slots.csv"))
        status.update(status_overrides)
        common.dump(directory/"episode_results.status.json", status)
    save()
    return SimpleNamespace(out=out, directory=directory, selected=selected, frozen=frozen, frame=frame,
                           slots=slotframe, save=save)


def test_completed_stage_uses_frozen_identities_and_full_72_slot_records(completed_A):
    frame, slots, frozen = p._validated_stage("experiment_A")
    assert len(frame) == 8 and len(slots) == 8*72 and frozen == completed_A.frozen


@pytest.mark.parametrize("target", ["episode_results.csv", "slots.csv"])
def test_stage_csv_cannot_change_after_completion(completed_A, target):
    path = completed_A.directory/target
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="CSV changed"):
        p._validated_stage("experiment_A")


def test_incomplete_stage_cannot_be_reused(completed_A):
    completed_A.save(status="incomplete", valid=False)
    with pytest.raises(ValueError, match="incomplete"):
        p._validated_stage("experiment_A")


@pytest.mark.parametrize("column", ["policy_id", "model_fingerprint", "dataset_fingerprint", "assignment_fingerprint", "rules_version"])
def test_matching_csv_digest_does_not_authorize_wrong_strategy_or_market(completed_A, column):
    rows = completed_A.frame.copy()
    rows.loc[0, column] = "different"
    completed_A.save(frame=rows)
    with pytest.raises(ValueError, match="identity"):
        p._validated_stage("experiment_A")


def test_72_row_trace_with_duplicate_slot_is_rejected(completed_A):
    slots = completed_A.slots.copy()
    slots.loc[1, "time_slot"] = 0
    completed_A.save(slots=slots)
    with pytest.raises(ValueError, match="full cycle"):
        p._validated_stage("experiment_A")


def test_reuse_coverage_is_derived_from_frozen_protocol_not_mutable_status_count(completed_A):
    rows = completed_A.frame.iloc[:-1].copy()
    slots = completed_A.slots[completed_A.slots.task_index.isin(rows.task_index)].copy()
    completed_A.save(frame=rows, slots=slots, expected_episodes=7, completed_episodes=7)
    with pytest.raises(ValueError, match="coverage|protocol|expected"):
        p._validated_stage("experiment_A")


def test_all_identical_B_policies_reuse_A_without_empty_execute_or_csv_read(completed_A, monkeypatch):
    monkeypatch.setattr(p, "execute", lambda *args, **kwargs: pytest.fail("No new tasks may execute"))
    status = p.test_B(workers=1)
    assert status["valid"] is True and status["status"] == "completed"
    assert status["new_executed_episodes"] == 0 and status["reused_episodes"] == 8
    result = pd.read_csv(completed_A.out/"experiment_B/episode_results.csv")
    assert result.phase.eq("experiment_B").all()
    assert result.total_value.tolist() == [72.]*8
    assert not (completed_A.out/"experiment_B_new").exists()


def test_independent_market_is_not_opened_before_freeze_validation(completed_A, monkeypatch):
    def reject(phase):
        raise ValueError("freeze invalid")
    monkeypatch.setattr(p, "verify_freeze", reject)
    monkeypatch.setattr(p, "build_cache", lambda *args, **kwargs: pytest.fail("Heldout market opened too early"))
    with pytest.raises(ValueError, match="freeze"):
        p.test_B(independent=True)
