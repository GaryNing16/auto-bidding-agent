"""Five-scene durable execution/resume regressions, with bounded train-only smoke.

Synthetic worker scheduling tests create no process. The one real spawn test
uses period-7's existing cache and only four requests per slot, never test data.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from harmony_baselines.classical import spec
from harmony_baselines.five_scenarios import common, execution as ex
from harmony_baselines.five_scenarios.evaluator import opponent_specs
from harmony_bidding.schema import Advertiser, fingerprint


def task(algorithm="Good", *, seed=20261028, training_seed=0, **kwargs):
    return dict(day_id="period-7", target="0", target_spec=spec("FixedAlpha", alpha=1.),
                scenario="Fixed", opponents={"fixed": spec("FixedAlpha", alpha=1.),
                "pid": spec("LocalPID", initial_alpha=1.), "auctionnet_pid": common.pid_spec()},
                seed=seed, algorithm=algorithm, training_seed=training_seed, **kwargs)


def completed_payload(t, identity, result_path, phase, index):
    allocation = opponent_specs(["0", "1"], t["target"], t["scenario"], t["opponents"],
                                t["seed"] + 1000003, t["day_id"])
    metadata = dict(phase=phase, task_index=index, source_task=f"{phase}:{index}",
        provenance="new_execution", source_scenario=t["scenario"],
        source_result_path=str(Path(result_path).resolve()), policy_id=identity["policy_id"],
        task_key=identity["task_key"], algorithm=t["algorithm"], scenario=t["scenario"],
        day_id=t["day_id"], player_id=t["target"], player_type="0", seed=t["seed"],
        training_seed=t.get("training_seed", 0), budget_scale=1., initial_budget=100.,
        model_fingerprint=fingerprint(t["target_spec"]), opponent_fingerprint=fingerprint(t["opponents"]),
        assignment_fingerprint=fingerprint(allocation), dataset_fingerprint=identity["dataset_fingerprint"],
        rules_version="harmony-A-B-local-v2")
    row = dict(metadata, total_value=72., total_cost=72., wins=72, budget_utilization=.72,
        completed=True, completed_slots=72, elapsed_seconds=.01, bidding_p99_ms=.1,
        update_p99_ms=.1, paired_p99_ms=.2, rss_mb=1., peak_process_rss_mb=1.)
    slots = [dict(metadata, time_slot=i, slot=i, alpha=1., cost=1., value=1., wins=1,
        remaining_budget=99.-i, cumulative_cost=i+1., cumulative_value=i+1.,
        bidding_ms=.1, update_ms=.1) for i in range(72)]
    return dict(status="completed", identity=identity, row=row, slots=slots,
                transition_path=None, seconds=.01)


class ScriptedContext:
    """A deterministic Pipe/Process double for ACK and cancellation races."""
    def __init__(self, modes=None):
        self.modes = modes or {}
        self.processes = []
        self.submitted = []

    def Pipe(self):
        state = SimpleNamespace(message=None, alive=False, mode="complete", commit_at_termination=None)
        context = self

        class Connection:
            def send(self, request):
                if request is None:
                    state.alive = False
                    return
                t, identity, result_path, diagnostic, phase, index = request
                context.submitted.append(index)
                state.mode = context.modes.get(index, "complete")
                if state.mode in ("complete", "commit_no_ack", "commit_exit", "corrupt"):
                    payload = completed_payload(t, identity, result_path, phase, index)
                    if state.mode == "corrupt":
                        payload["slots"] = payload["slots"][:-1]
                    common.dump(result_path, payload)
                if state.mode in ("complete", "corrupt"):
                    state.message = {"status": "completed", "path": result_path}
                elif state.mode == "fail":
                    state.message = {"status": "failed", "exception_type": "ValueError", "reason": "synthetic failure"}
                elif state.mode == "commit_exit":
                    state.alive = False
                elif state.mode == "commit_on_terminate":
                    state.commit_at_termination = (result_path, completed_payload(t, identity, result_path, phase, index))

            def poll(self):
                return state.message is not None

            def recv(self):
                message, state.message = state.message, None
                return message

            def close(self):
                pass

        parent, child = Connection(), Connection()
        child.state = state
        return parent, child

    def Process(self, target, args):
        state = args[0].state

        class Process:
            exitcode = 0
            terminated = False

            def start(self):
                state.alive = True

            def is_alive(self):
                return state.alive

            def terminate(self):
                self.terminated = True
                state.alive = False
                if state.commit_at_termination is not None:
                    path, payload = state.commit_at_termination
                    common.dump(path, payload)
                    state.commit_at_termination = None

            def join(self, timeout=None):
                pass

        process = Process()
        self.processes.append(process)
        return process


@pytest.fixture
def scripted(monkeypatch):
    state = {"dataset": "synthetic-data", "code": "v1"}
    context = ScriptedContext()
    monkeypatch.setattr(ex, "cached_day", lambda _: SimpleNamespace(fingerprint=state["dataset"],
        advertisers=(Advertiser("0", "0", 100.), Advertiser("1", "0", 100.))))
    monkeypatch.setattr(ex, "runtime_code_hashes", lambda: {"test-code": state["code"]})
    monkeypatch.setattr(ex.mp, "get_context", lambda method: context)
    return context, state


def test_durable_commits_rebuild_csv_and_resume_without_reexecuting(scripted, tmp_path):
    context, _ = scripted
    tasks = [task(), task(seed=20261029)]
    frame, failures, skipped = ex.execute(tasks, tmp_path, "smoke", workers=2)
    assert len(frame) == 2 and failures == skipped == []
    before = [(p.name, p.read_bytes()) for p in sorted((tmp_path/"results").glob("*.json"))]
    (tmp_path/"episode_results.csv").write_text("tampered_csv\n9999\n", encoding="utf-8")
    frame2, failures2, skipped2 = ex.execute(tasks, tmp_path, "smoke", workers=1)
    assert context.submitted == [0, 1]
    assert failures2 == skipped2 == []
    assert frame2.total_value.tolist() == [72., 72.]
    assert pd.read_csv(tmp_path/"episode_results.csv").total_value.tolist() == [72., 72.]
    assert before == [(p.name, p.read_bytes()) for p in sorted((tmp_path/"results").glob("*.json"))]
    assert common.read(tmp_path/"episode_results.status.json")["resumed_completed"] == 2


@pytest.mark.parametrize("change", ["checkpoint", "table", "opponent_checkpoint", "opponent_spec", "target_spec", "dataset", "code"])
def test_resume_key_includes_all_runtime_assets(change, scripted, tmp_path):
    context, state = scripted
    checkpoint, table, opponent = (tmp_path/name for name in ("weights.pt", "table.csv", "opponent.pt"))
    for p in (checkpoint, table, opponent):
        p.write_bytes(b"initial")
    t = task()
    t["target_spec"]["kwargs"].update(checkpoint=str(checkpoint), table_path=str(table))
    t["opponents"]["fixed"]["kwargs"]["checkpoint"] = str(opponent)
    directory = tmp_path/"run"
    ex.execute([t], directory, "smoke", workers=1)
    if change in ("checkpoint", "table", "opponent_checkpoint"):
        {"checkpoint": checkpoint, "table": table, "opponent_checkpoint": opponent}[change].write_bytes(b"changed")
    elif change == "opponent_spec":
        t["opponents"]["fixed"]["kwargs"]["alpha"] = 2.
    elif change == "target_spec":
        t["target_spec"]["kwargs"]["alpha"] = 2.
    else:
        state[change] = "changed"
    with pytest.raises(ValueError, match="Immutable"):
        ex.execute([t], directory, "smoke", workers=1)
    assert context.submitted == [0]


def test_nested_behavior_policy_asset_hash(tmp_path):
    model = tmp_path/"nested.pt"
    model.write_bytes(b"first")
    policy = spec("Exploration", base_spec={"kind": "custom", "class_path": "unused:Agent",
        "kwargs": {"checkpoint": str(model)}}, noise=.15)
    first = common.policy_id(policy)
    model.write_bytes(b"second")
    assert common.policy_id(policy) != first


@pytest.mark.parametrize("mode", ["commit_no_ack", "commit_exit", "commit_on_terminate"])
def test_durable_commit_wins_over_missing_ack_or_worker_exit(mode, scripted, tmp_path):
    context, _ = scripted
    context.modes[0] = mode
    frame, failures, skipped = ex.execute([task()], tmp_path, "smoke", workers=1, timeout_seconds=.01)
    assert len(frame) == 1 and failures == skipped == []
    resumed, failures, skipped = ex.execute([task()], tmp_path, "smoke", workers=1, timeout_seconds=.01)
    assert len(resumed) == 1 and failures == skipped == []
    assert context.submitted == [0]


def test_timeout_commit_connection_is_not_reused_for_next_task(scripted, tmp_path):
    context, _ = scripted
    context.modes[0] = "commit_no_ack"
    tasks = [task(), task(seed=20261029)]
    frame, failures, skipped = ex.execute(tasks, tmp_path, "smoke", workers=1, timeout_seconds=.01)
    assert len(frame) == 2 and failures == skipped == []
    # A delayed acknowledgement on the old connection must never be read as
    # the next task's completion, even though the first result was committed.
    assert len(context.processes) == 2
    assert context.processes[0].terminated


@pytest.mark.parametrize("concurrent_mode, successes, skips", [("block", {2}, {1}), ("commit_no_ack", {1, 2}, set())])
def test_failure_poison_cancels_inflight_group_but_preserves_other_groups_and_commits(
        concurrent_mode, successes, skips, scripted, tmp_path):
    context, _ = scripted
    context.modes.update({0: "fail", 1: concurrent_mode})
    tasks = [task("Bad", training_seed=1), task("Bad", training_seed=2), task("Good")]
    frame, failures, skipped = ex.execute(tasks, tmp_path, "validation", workers=3)
    assert set(frame.task_index) == successes
    assert {r["task_index"] for r in failures} == {0}
    assert {r["task_index"] for r in skipped} == skips
    assert all("total_value" not in r and "total_cost" not in r for r in failures+skipped)
    assert successes | {0} | skips == {0, 1, 2}
    status = common.read(tmp_path/"episode_results.status.json")
    assert status["status"] == "completed_with_failures" and not status["valid"]
    assert all(not p.is_alive() for p in context.processes)


@pytest.mark.parametrize("corruption", ["task_key", "duplicate", "phase", "task_index"])
def test_resume_rejects_stale_or_duplicate_failure_records(corruption, scripted, tmp_path):
    context, _ = scripted
    context.modes[0] = "fail"
    tasks = [task("Bad"), task("Bad", seed=20261029)]
    ex.execute(tasks, tmp_path, "validation", workers=1)
    path = tmp_path/"failures.json"
    records = common.read(path)
    if corruption == "duplicate":
        records.append(deepcopy(records[0]))
    elif corruption == "task_index":
        records[0][corruption] = 1
    else:
        records[0][corruption] = "wrong"
    common.dump(path, records)
    with pytest.raises(ValueError):
        ex.execute(tasks, tmp_path, "validation", workers=1)
    assert context.submitted == [0]


@pytest.mark.parametrize("corruption", ["task_key", "duplicate", "phase"])
def test_resume_rejects_stale_or_duplicate_skipped_records(corruption, scripted, tmp_path):
    context, _ = scripted
    context.modes[0] = "fail"
    tasks = [task("Bad"), task("Bad", seed=20261029)]
    ex.execute(tasks, tmp_path, "validation", workers=1)
    path = tmp_path/"skipped.json"
    records = common.read(path)
    if corruption == "duplicate":
        records.append(deepcopy(records[0]))
    else:
        records[0][corruption] = "wrong"
    common.dump(path, records)
    with pytest.raises(ValueError):
        ex.execute(tasks, tmp_path, "validation", workers=1)


@pytest.mark.parametrize("field, value", [("alpha", float("nan")), ("alpha", -1.),
    ("cost", float("nan")), ("value", float("nan")), ("wins", -.5),
    ("remaining_budget", -1.), ("player_id", "47"), ("task_key", "wrong")])
def test_validate_rejects_illegal_or_misidentified_slots(field, value, scripted, tmp_path):
    t = task()
    ex.execute([t], tmp_path, "smoke", workers=1)
    payload = common.read(tmp_path/"results/task_00000.json")
    payload["slots"][20][field] = value
    with pytest.raises(ValueError):
        ex._validate(payload, t, payload["identity"], 0)


def test_validate_failure_does_not_leak_removed_worker(scripted, tmp_path):
    context, _ = scripted
    context.modes[0] = "corrupt"
    with pytest.raises(ValueError, match="72"):
        ex.execute([task()], tmp_path, "smoke", workers=1)
    assert len(context.processes) == 1
    assert not context.processes[0].is_alive()


@pytest.mark.parametrize("kwargs", [{"workers": 0}, {"workers": -1}, {"workers": True},
    {"workers": 1.5}, {"timeout_seconds": 0.}, {"timeout_seconds": -1.},
    {"timeout_seconds": float("nan")}, {"timeout_seconds": float("inf")}])
def test_invalid_execution_resources_rejected(kwargs, scripted, tmp_path):
    with pytest.raises(ValueError):
        ex.execute([task()], tmp_path, "smoke", **kwargs)


def test_atomic_dump_retries_transient_windows_file_lock(monkeypatch, tmp_path):
    path = tmp_path/"state.json"
    common.dump(path, {"old": True})
    replace = Path.replace
    attempts = []

    def intermittent(source, destination):
        if Path(destination) == path:
            attempts.append(1)
            if len(attempts) < 3:
                assert common.read(path) == {"old": True}
                raise PermissionError("simulated reader lock")
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", intermittent)
    monkeypatch.setattr(common.time, "sleep", lambda seconds: None)
    common.dump(path, {"new": True})
    assert len(attempts) == 3 and common.read(path) == {"new": True}


def test_real_spawn_train_smoke_success_failure_and_resume(monkeypatch, tmp_path):
    # Fixed hash isolates the process/resume test from concurrently edited new
    # orchestration files. The asset/code-change rejection is tested above.
    monkeypatch.setattr(ex, "runtime_code_hashes", lambda: {"smoke-test": "fixed"})
    good = task("Good", smoke=True, collect=True)
    bad = task("Bad", smoke=True)
    bad["target_spec"] = spec("FixedAlpha", alpha=-1.)
    later_bad = deepcopy(bad)
    later_bad["seed"] = 20261029
    tasks = [good, bad, later_bad]
    frame, failures, skipped = ex.execute(tasks, tmp_path, "smoke_execution", workers=2, timeout_seconds=90.)
    assert len(frame) == 1 and frame.iloc[0].algorithm == "Good"
    assert {r["task_index"] for r in failures} == {1}
    assert {r["task_index"] for r in skipped} == {2}
    assert failures[0]["exception_type"] == "ValueError"
    assert "total_value" not in failures[0]
    result = tmp_path/"results/task_00000.json"
    before = result.read_bytes()
    slots = pd.read_csv(tmp_path/"slots.csv")
    assert slots.time_slot.tolist() == list(range(72))
    assert slots.alpha.map(np.isfinite).all()
    payload = common.read(result)
    with np.load(payload["transition_path"], allow_pickle=False) as arrays:
        assert arrays["states"].shape == (72, 18)
        assert arrays["dones"].sum() == 1 and arrays["dones"][-1]
    frame2, failures2, skipped2 = ex.execute(tasks, tmp_path, "smoke_execution", workers=1, timeout_seconds=90.)
    assert result.read_bytes() == before
    assert failures2 == failures and skipped2 == skipped
    assert frame2.iloc[0].total_value == frame.iloc[0].total_value
    assert common.read(tmp_path/"episode_results.status.json")["resumed_completed"] == 1
    Path(payload["transition_path"]).write_bytes(b"corrupted transition sidecar")
    with pytest.raises(ValueError):
        ex.execute(tasks, tmp_path, "smoke_execution", workers=1, timeout_seconds=90.)
