"""Sequential, training-market-only 4/8-worker comparison; no policy tuning.

Run only when no other experiment pool is active. The selected worker count is
a recommendation, recorded separately from the immutable experiment protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform

import pandas as pd

from harmony_baselines.data import sha256_file
from harmony_bidding.schema import fingerprint
from . import common as c, execution

WORKERS = (4, 8)
SEED = 20261028
DAY = "period-7"
ROLES = tuple(str(i) for i in range(48))
IDENTITY_FIELDS = ("algorithm", "day_id", "scenario", "player_id", "seed", "training_seed",
                   "assignment_fingerprint", "dataset_fingerprint", "model_fingerprint",
                   "opponent_fingerprint", "policy_id", "task_key", "initial_budget", "rules_version")
BUSINESS_FIELDS = ("total_value", "total_cost", "wins")


def make_plan():
    selected_path = c.OLD / "selected_frozen.json"
    opponents_path = c.OLD / "opponents_frozen.json"
    protocol_path = c.OUT / "protocol.json"
    selected = [p for p in c.read(selected_path)["selected"] if p["algorithm"] == "LocalPID"]
    if len(selected) != 1:
        raise ValueError("Expected exactly one historical selected LocalPID policy")
    protocol = c.read(protocol_path)
    if DAY not in protocol["train_days"] or protocol["workers"] != 4:
        raise ValueError("Resource experiment requires original four-worker training-day protocol")
    opponents = c.read(opponents_path)
    tasks = execution.tasks_for(selected, opponents, [DAY], ROLES, ["Fixed"], [SEED],
                                collect=False, collect_all=False)
    plan = {"purpose": "training-market execution efficiency only", "tasks": tasks,
            "worker_counts_in_order": list(WORKERS), "episodes_per_group": 48, "total_episodes": 96,
            "full_episode_slots": 72, "train_days": [DAY], "validation_used": False, "test_used": False,
            "included_in_training_data": False, "included_in_performance_rankings": False,
            "historical_selection_sha256": sha256_file(selected_path),
            "historical_opponents_sha256": sha256_file(opponents_path),
            "experiment_protocol_sha256": sha256_file(protocol_path),
            "runtime_code_sha256": c.runtime_code_hashes(), "resource_script_sha256": sha256_file(__file__),
            "measurement": "runtime_segment.seconds; worker startup/import, execution, export and shutdown included",
            "excluded_from_segment": "parent-process preparation before execute starts its segment clock",
            "concurrency_rule": "sequential worker pools; operator must stop other experiment pools",
            "order_caveat": "4 workers always first; filesystem cache and external load may affect one trial",
            "repetition_count_per_worker_setting": 1}
    return tasks, plan


def _runtime_summary(directory, workers):
    segments = sorted(directory.glob("runtime_segment_*.json"))
    if len(segments) != 1:
        raise ValueError("A fair resource comparison requires one uninterrupted timing segment per group")
    segment = c.read(segments[0])
    status = c.read(directory / "episode_results.status.json")
    seconds = float(segment["seconds"])
    if (not math.isfinite(seconds) or seconds <= 0 or segment["resumed"] != 0
            or segment["new_completed"] != 48 or segment["failures"] != 0
            or status["status"] != "completed" or status["valid"] is not True
            or status["workers"] != workers or status["resumed_completed"] != 0
            or status["completed_episodes"] != 48 or status["expected_episodes"] != 48
            or status["failed_episodes"] != 0 or status["skipped_episodes"] != 0):
        raise ValueError("Only a complete, uninterrupted 48-episode resource group is comparable")
    return {"workers": workers, "episodes": 48, "seconds": seconds,
            "episodes_per_second": 48 / seconds, "startup_included": True,
            "timing_source": str(segments[0].resolve()), "timing_sha256": sha256_file(segments[0])}


def _read_completed_group(directory, tasks, workers, phase):
    """Validate durable results directly; do not create empty resume segments."""
    protocol = c.read(directory / "task_protocol.json")
    if (protocol["tasks"] != tasks or protocol["phase"] != phase
            or protocol["code_sha256"] != c.runtime_code_hashes()
            or protocol["fail_group"] is not False or len(protocol["identities"]) != 48):
        raise ValueError("Resource task/code identity differs from the declared benchmark")
    summary = _runtime_summary(directory, workers)
    rows, evidence = [], {}
    for index, task in enumerate(tasks):
        path = directory / "results" / f"task_{index:05d}.json"
        payload = c.read(path)
        execution._validate(payload, task, protocol["identities"][index], index)
        if payload["row"]["phase"] != phase or len(payload["slots"]) != 72:
            raise ValueError("Resource group phase/full-period mismatch")
        rows.append(payload["row"])
        evidence[str(path.resolve())] = sha256_file(path)
    for name in ("task_protocol.json", "episode_results.status.json", "episode_results.csv",
                 "slots.csv", "failures.json", "skipped.json", "runtime_segment_000.json"):
        path = directory / name
        evidence[str(path.resolve())] = sha256_file(path)
    if c.read(directory / "failures.json") or c.read(directory / "skipped.json"):
        raise ValueError("Resource group contains failed/skipped episodes")
    summary["artifact_sha256"] = evidence
    return pd.DataFrame(rows), summary


def _assert_equal_results(left, right):
    """Exact equality, without epsilon or rounding; timing fields are excluded."""
    frames = []
    for source in (left, right):
        frame = source.copy()
        frame["player_id"] = frame["player_id"].astype(str)
        if len(frame) != 48 or frame.player_id.duplicated().any() or set(frame.player_id) != set(ROLES):
            raise ValueError("Resource comparison must cover every one of the 48 roles exactly once")
        for field in BUSINESS_FIELDS:
            if not frame[field].map(lambda v: math.isfinite(v) and v >= 0).all():
                raise ValueError(f"Illegal resource comparison metric: {field}")
        frames.append(frame.set_index("player_id").reindex(ROLES))
    for field in IDENTITY_FIELDS + BUSINESS_FIELDS:
        if field == "player_id":
            continue
        if not (frames[0][field].to_numpy() == frames[1][field].to_numpy()).all():
            raise ValueError(f"Worker-count changes altered an episode's {field}")
    return {"matched_episodes": 48, "comparison": "exact numeric/string equality without tolerance",
            "business_fields": list(BUSINESS_FIELDS), "identity_fields": list(IDENTITY_FIELDS),
            "passed": True}


def benchmark():
    tasks, plan = make_plan()
    directory = c.OUT / "resource_benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    c.immutable(directory / "plan.json", plan)
    selection_path = c.OUT / "resource_selection.json"
    existing = c.read(selection_path) if selection_path.exists() else None
    if existing is not None and existing["plan_fingerprint"] != fingerprint(plan):
        raise ValueError("Completed resource selection has different inputs or code")
    frames, groups = [], []
    for workers in WORKERS:
        group_dir = directory / f"workers_{workers}"
        phase = f"resource_benchmark_workers_{workers}"
        if not group_dir.exists():
            print(json.dumps({"phase": phase, "status": "starting", "episodes": 48}), flush=True)
            _, failures, skipped = execution.execute(tasks, group_dir, phase, workers=workers, fail_group=False)
            if failures or skipped:
                raise RuntimeError("Failed resource group retained; no worker recommendation will be made")
        frame, summary = _read_completed_group(group_dir, tasks, workers, phase)
        frames.append(frame)
        groups.append(summary)
    equality = _assert_equal_results(*frames)
    winner = min(groups, key=lambda item: (item["seconds"], item["workers"]))
    selection = {"status": "completed", "plan_fingerprint": fingerprint(plan),
        "recommended_workers": winner["workers"], "initial_protocol_workers": 4,
        "selection_rule": "minimum recorded complete-stage seconds; exact tie chooses 4",
        "groups": groups, "paired_episode_equality": equality,
        "speedup_workers_8_over_4": groups[0]["seconds"] / groups[1]["seconds"],
        "source_days": [DAY], "test_used": False, "validation_used": False,
        "included_in_training_data": False, "included_in_performance_rankings": False,
        "measurement": plan["measurement"], "order_caveat": plan["order_caveat"],
        "host": {"platform": platform.platform(), "logical_cpu_count": os.cpu_count()},
        "application_status": "recommendation_only; pass --workers explicitly for subsequent stages",
        "immutable_protocol_changed": False}
    if existing is not None:
        if existing != selection:
            raise ValueError("Completed resource selection no longer matches its source evidence")
        return existing
    if winner["workers"] != 4:
        c.immutable(directory / "operational_adjustment.json", {
            "kind": "resource-only operational adjustment", "status": "recommended_not_applied",
            "initial_protocol_workers": 4, "recommended_workers": winner["workers"],
            "reason": "same 48 training-market episodes, identical outcomes, lower complete-stage wall time",
            "selection_path": str(selection_path.resolve()), "plan_fingerprint": fingerprint(plan),
            "experiment_protocol_sha256": plan["experiment_protocol_sha256"],
            "policy_parameters_changed": False, "test_used": False, "immutable_protocol_changed": False})
    c.immutable(selection_path, selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="Print inputs only; do not run episodes or write outputs")
    args = parser.parse_args()
    if args.plan_only:
        _, plan = make_plan()
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(benchmark(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
