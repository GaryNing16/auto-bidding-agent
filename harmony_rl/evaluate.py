"""Frozen PPO evaluation through the existing five-scenario evaluator.

The pool runs complete episodes, commits each result atomically, and stops at
the wall-clock budget. An interrupted batch is resumable, never a zero score.
Historical Baseline B is verified and reused, not retrained or replayed.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from harmony_baselines.data import sha256_file
from harmony_baselines.five_scenarios.common import (
    ROOT, OUT as BASELINE_ROOT, SCENARIOS, ROLES, SEEDS, ALGORITHMS,
    cached_day, read, dump, immutable, policy_id,
)
from harmony_baselines.five_scenarios.evaluator import run_episode, opponent_specs
from harmony_bidding.schema import fingerprint

VARIANTS = ("PPO-Raw", "PPO-Aug", "PPO-LSTM", "PPO-LSTM-Aug", "Residual-PPO", "Residual-PPO-LSTM")
KEYS = ["algorithm", "scenario", "day_id", "player_id", "seed", "training_seed"]


def audit_baseline(root=BASELINE_ROOT, *, phase="experiment_B", verify_slots=True):
    """Validate sealed historical results, assets, source and current markets."""
    from harmony_baselines.five_scenarios.report import Sources, load_completed_phase
    root = Path(root).resolve()
    sources = Sources()
    # load_completed_phase verifies both the CSV seal and complete Cartesian
    # coverage; the trace seal is deliberately not optional for real reuse.
    if not verify_slots:
        raise ValueError("Baseline reuse requires verification of the complete slot evidence")
    frame = load_completed_phase(root, phase, sources)
    selection = read(root / phase / "selected_frozen.json")
    protocol = read(root / "protocol.json")
    opponents = read(root / "opponents_frozen.json")
    checks = {str(root / "opponents_frozen.json"): selection["opponents_sha256"],
              str(root / "protocol.json"): selection["protocol_sha256"], **selection["assets"]}
    checks.update({str(ROOT / name): digest for name, digest in selection["execution_code_sha256"].items()})
    for path, digest in checks.items():
        if not Path(path).is_file() or sha256_file(path) != digest:
            raise ValueError("Frozen Baseline input changed: " + path)
    for candidate in selection["selected"]:
        key = f"{candidate['algorithm']}:{candidate.get('training_seed', 0)}"
        if policy_id(candidate["spec"]) != selection["policy_ids"][key]:
            raise ValueError("Frozen policy identity changed: " + key)
    if set(frame.algorithm) != set(ALGORITHMS) or protocol["scenarios"] != list(SCENARIOS):
        raise ValueError("Baseline algorithm/scenario roster differs")
    if set(frame.rules_version) != {"harmony-A-B-local-v2"} or not frame.budget_scale.eq(1.).all():
        raise ValueError("Historical auction rules or original budgets differ")
    # Per-role original budget and per-case opponent assignment must match
    # the very objects the new evaluation will use.
    for day_id, group in frame.groupby("day_id"):
        day = cached_day(day_id)
        budgets = {a.advertiser_id: a.budget for a in day.advertisers}
        if not group.dataset_fingerprint.eq(day.fingerprint).all():
            raise ValueError("Historical market fingerprint differs: " + day_id)
        if not np.allclose(group.initial_budget, group.player_id.map(budgets), atol=1e-9, rtol=0):
            raise ValueError("Historical advertiser budgets differ")
        ids = list(budgets)
        for (scenario, player, seed), cases in group.groupby(["scenario", "player_id", "seed"]):
            allocation = opponent_specs(ids, player, scenario, opponents, int(seed) + 1000003, day_id)
            if not cases.assignment_fingerprint.eq(fingerprint(allocation)).all():
                raise ValueError("Historical opponent assignment differs")
    if not frame.opponent_fingerprint.eq(fingerprint(opponents)).all():
        raise ValueError("Historical frozen opponents differ")
    audit = {"status": "passed", "phase": phase, "episodes": len(frame),
             "algorithms": list(ALGORITHMS), "scenarios": list(SCENARIOS),
             "days": sorted(frame.day_id.unique()), "evaluation_seeds": sorted(map(int, frame.seed.unique())),
             "rules_version": "harmony-A-B-local-v2", "source_sha256": sources.hashes,
             "verified_frozen_inputs": checks, "retraining": False,
             "limitation": "period-24 is previously exposed comparison data, not a new blind test"}
    return frame, opponents, audit


def qualify_checkpoint(checkpoint):
    """Require a real-market run's sealed validation-selected model.

    Reading a local checkpoint does not qualify it for the formal comparison.
    In particular, synthetic smoke runs and unselected last.pt snapshots must
    never silently acquire a validation-selected label through the CLI.
    """
    import torch
    from .config import VARIANTS as EXPERIMENTS
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.name != "best.pt":
        raise ValueError("Formal PPO evaluation requires the validation-selected best.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    provenance = payload.get("provenance", {})
    if payload.get("format") != "harmony-ppo-lstm-v1" or cfg.get("variant") not in EXPERIMENTS:
        raise ValueError("Unsupported PPO checkpoint format or experiment variant")
    if cfg.get("smoke") is not False or provenance.get("smoke") is not False:
        raise ValueError("Synthetic smoke checkpoints cannot enter formal PPO evaluation")
    if provenance.get("test_read_during_training") is not False:
        raise ValueError("Checkpoint does not attest training/test separation")
    selection_path = checkpoint.parent / "selection.json"
    status_path = checkpoint.parent / "training_status.json"
    if not selection_path.is_file() or not status_path.is_file():
        raise ValueError("Missing PPO validation selection or completed training evidence")
    selection, status = read(selection_path), read(status_path)
    checkpoint_sha = sha256_file(checkpoint)
    if selection.get("checkpoint_sha256") != checkpoint_sha or selection.get("test_used") is not False:
        raise ValueError("PPO validation selection SHA/test separation differs")
    if (selection.get("update") != payload.get("policy_version")
            or selection.get("episodes") != payload.get("episode_count")
            or not isinstance(payload.get("episode_count"), int) or payload["episode_count"] < 1):
        raise ValueError("PPO selected update/episodes differ from checkpoint")
    selected_value = selection.get("value")
    if (not isinstance(selected_value, (int, float)) or not np.isfinite(selected_value)
            or selected_value != payload.get("best_validation")):
        raise ValueError("PPO validation score differs from selected checkpoint")
    validation = payload.get("validation_history", [])
    matching = [row for row in validation if row.get("update") == selection["update"]
                and row.get("episodes") == selection["episodes"]]
    if (len(matching) != 1 or matching[0].get("mean_value") != selected_value
            or matching[0].get("episodes_evaluated", 0) < 1):
        raise ValueError("PPO checkpoint lacks matching completed validation evidence")
    algorithm = EXPERIMENTS[cfg["variant"]][0]
    training_seed = payload.get("training_seed")
    if (not isinstance(training_seed, int) or training_seed != cfg.get("seed")
            or status.get("training_seed") != training_seed or status.get("algorithm") != algorithm):
        raise ValueError("PPO algorithm/training seed differs across saved evidence")
    finished_budget = (status.get("complete") is True
                       and status.get("completed_episodes", 0) >= status.get("requested_episodes", 1))
    stopped_by_validation = (status.get("termination") == "early_stopping"
                             and status.get("completed_episodes", 0) > 0)
    if (not (finished_budget or stopped_by_validation) or status.get("validated") is not True
            or status.get("completed_episodes", 0) < selection["episodes"]
            or Path(status.get("checkpoint") or "").resolve() != checkpoint):
        raise ValueError("PPO training run is not complete and validated for this checkpoint")
    evidence = dict(algorithm=algorithm, training_seed=training_seed,
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        selection_sha256=sha256_file(selection_path), selected_update=selection["update"],
        selected_episodes=selection["episodes"], completed_training_episodes=status["completed_episodes"],
        training_termination=status.get("termination", "episode_limit"), complete_interaction_budget=finished_budget,
        real_market_training=True, validation_selected=True, test_used_for_selection=False)
    return payload, evidence


def checkpoint_candidates(checkpoints):
    """Build Agent specs only after validating actual training/selection seals."""
    result = []
    for checkpoint in checkpoints:
        _, evidence = qualify_checkpoint(checkpoint)
        result.append({"algorithm": evidence["algorithm"], "training_seed": evidence["training_seed"],
                       "spec": {"kind": "custom", "class_path": "harmony_rl.policy:PPOAgent",
                                "kwargs": {"checkpoint": evidence["checkpoint"]}}})
    return result


def _runtime_hashes():
    names = [ROOT / "harmony_bidding" / p for p in ("auction.py", "budget_policy.py", "schema.py", "agent_api.py", "agents.py")]
    names += [ROOT / "harmony_baselines/five_scenarios" / p for p in ("evaluator.py", "auctionnet_pid.py", "common.py")]
    names += [ROOT / "harmony_baselines" / p for p in ("data.py", "classical.py", "neural.py")]
    names += [Path(__file__).parent / p for p in ("evaluate.py", "policy.py", "models.py")]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in names if p.is_file()}


def _validate_result(payload, task, identity):
    if payload.get("status") != "completed" or payload.get("identity") != identity:
        raise ValueError("Result identity differs from the frozen task")
    row, slots = payload["row"], payload["slots"]
    for key, source in [("algorithm", "algorithm"), ("scenario", "scenario"), ("day_id", "day_id"),
                        ("player_id", "target"), ("seed", "seed"), ("training_seed", "training_seed")]:
        if str(row[key]) != str(task[source]):
            raise ValueError("Result task mismatch: " + key)
    if not row.get("completed") or row.get("completed_slots") != 72:
        raise ValueError("Incomplete episode")
    if row["dataset_fingerprint"] != identity["dataset_fingerprint"] or row["policy_id"] != identity["policy_id"]:
        raise ValueError("Result dataset/model differs")
    if row["model_fingerprint"] != fingerprint(task["target_spec"]):
        raise ValueError("Result model specification differs")
    if "initial_budget" in identity and row["initial_budget"] != identity["initial_budget"]:
        raise ValueError("Result original budget differs")
    if "assignment_fingerprint" in identity and row["assignment_fingerprint"] != identity["assignment_fingerprint"]:
        raise ValueError("Result opponent assignment differs")
    if row["rules_version"] != "harmony-A-B-local-v2" or row["opponent_fingerprint"] != fingerprint(task["opponents"]):
        raise ValueError("Result rules/opponents differ")
    if len(slots) != 72 or [s["time_slot"] for s in slots] != list(range(72)):
        raise ValueError("Result does not contain all 72 slots")
    if any(not np.isfinite([s[k] for k in ("alpha", "value", "cost", "remaining_budget")]).all() for s in slots):
        raise ValueError("Nonfinite slot data")
    if any(min(s[k] for k in ("alpha", "value", "cost", "remaining_budget")) < 0 for s in slots):
        raise ValueError("Negative slot data")
    if not np.isclose(sum(s["value"] for s in slots), row["total_value"], atol=1e-9, rtol=1e-12):
        raise ValueError("Slot/episode value mismatch")
    if not np.isclose(sum(s["cost"] for s in slots), row["total_cost"], atol=2e-6, rtol=0):
        raise ValueError("Slot/episode cost mismatch")
    if row["total_cost"] > row["initial_budget"] + 1e-6 or row["total_cost"] < 0:
        raise ValueError("Budget violation")
    cumulative_cost = cumulative_value = 0.
    for s in slots:
        cumulative_cost += s["cost"]
        cumulative_value += s["value"]
        if (not np.isclose(s["cumulative_cost"] + s["remaining_budget"], row["initial_budget"], atol=2e-6, rtol=0)
                or not np.isclose(cumulative_cost, s["cumulative_cost"], atol=2e-6, rtol=0)
                or not np.isclose(cumulative_value, s["cumulative_value"], atol=1e-9, rtol=1e-12)
                or any(str(s[k]) != str(row[k]) for k in KEYS)):
            raise ValueError("Slot ledger or case identity mismatch")
    if sum(s["wins"] for s in slots) != row["wins"]:
        raise ValueError("Slot/episode wins mismatch")


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)


def _worker(job):
    task, identity, path = job
    started = time.perf_counter()
    try:
        kwargs = dict(task)
        if policy_id(task["target_spec"]) != identity["policy_id"]:
            raise ValueError("Checkpoint changed after evaluation was frozen")
        day = cached_day(kwargs.pop("day_id"))
        row, slots, _ = run_episode(day, **kwargs)
        if policy_id(task["target_spec"]) != identity["policy_id"]:
            raise ValueError("Checkpoint changed during the episode")
        row.update(policy_id=identity["policy_id"], task_key=identity["task_key"], provenance="new_frozen_PPO_evaluation")
        for s in slots:
            s.update(policy_id=identity["policy_id"], task_key=identity["task_key"])
        payload = {"status": "completed", "identity": identity, "row": row, "slots": slots}
        _validate_result(payload, task, identity)
        dump(path, payload)
        return {"status": "completed", "seconds": time.perf_counter() - started}
    except Exception as error:
        return {"status": "failed", "exception_type": type(error).__name__, "reason": str(error),
                "traceback": traceback.format_exc(), "seconds": time.perf_counter() - started}


def run_evaluation(candidates, output, *, days=("period-24",), roles=ROLES, scenarios=SCENARIOS,
                   seeds=SEEDS, workers=8, max_seconds=3600., baseline_root=BASELINE_ROOT,
                   reuse_baselines=True, opponents=None):
    """Evaluate exact candidate x day x scenario x role x seed cells.

    ``candidates`` entries have algorithm, training_seed and standard ``spec``.
    ``max_seconds`` bounds this invocation; unchanged successful cells resume.
    A deadline may cancel an in-flight episode; it is rerun intact on resume.
    """
    output = Path(output).resolve()
    if workers < 1 or not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Positive workers and finite wall-clock budget required")
    if not candidates or len({(c["algorithm"], c.get("training_seed", 0)) for c in candidates}) != len(candidates):
        raise ValueError("Need unique algorithm/training-seed candidates")
    qualifications = []
    for candidate in candidates:
        spec = candidate["spec"]
        if spec.get("class_path") == "harmony_rl.policy:PPOAgent":
            _, evidence = qualify_checkpoint(spec.get("kwargs", {}).get("checkpoint", ""))
            if (candidate["algorithm"] != evidence["algorithm"]
                    or candidate.get("training_seed", 0) != evidence["training_seed"]):
                raise ValueError("PPO candidate label/seed differs from its qualified checkpoint")
            qualifications.append(evidence)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(exist_ok=True)
    started = time.perf_counter()
    if reuse_baselines:
        baseline, frozen_opponents, audit = audit_baseline(baseline_root)
        if opponents is not None and opponents != frozen_opponents:
            raise ValueError("Requested opponents differ from reused Baseline")
        opponents = frozen_opponents
        if not set(days) <= set(baseline.day_id) or not set(seeds) <= set(baseline.seed):
            raise ValueError("No sealed Baseline B result for requested test days/seeds")
        selected = baseline[baseline.day_id.isin(days) & baseline.player_id.isin(list(map(str, roles))) &
                            baseline.scenario.isin(scenarios) & baseline.seed.isin(seeds)]
        selected.to_csv(output / "baseline_reused.csv", index=False)
        dump(output / "baseline_reuse.json", audit)
    elif opponents is None:
        opponents = read(Path(baseline_root) / "opponents_frozen.json")
    tasks = [dict(day_id=d, target=str(p), target_spec=c["spec"], scenario=s, opponents=opponents,
                  seed=int(seed), algorithm=c["algorithm"], training_seed=int(c.get("training_seed", 0)))
             for c in candidates for d in days for s in scenarios for p in roles for seed in seeds]
    markets = {d: cached_day(d) for d in days}
    for d, market in markets.items():
        if not set(map(str, roles)) <= {a.advertiser_id for a in market.advertisers}:
            raise ValueError("Unknown evaluation role")
    code = _runtime_hashes()
    identities = []
    for task in tasks:
        day = markets[task["day_id"]]
        roster = {a.advertiser_id: a.budget for a in day.advertisers}
        assignment = opponent_specs(list(roster), task["target"], task["scenario"], task["opponents"],
                                    task["seed"] + 1000003, day.day_id)
        identity = dict(policy_id=policy_id(task["target_spec"]), dataset_fingerprint=markets[task["day_id"]].fingerprint,
                        code_fingerprint=fingerprint(code), rules_version="harmony-A-B-local-v2",
                        initial_budget=roster[task["target"]], assignment_fingerprint=fingerprint(assignment))
        identity["task_key"] = fingerprint({"task": task, **identity})
        identities.append(identity)
    protocol = dict(candidates=candidates, days=list(days), roles=list(map(str, roles)), scenarios=list(scenarios),
                    seeds=list(map(int, seeds)), tasks=tasks, identities=identities, execution_code_sha256=code,
                    baseline_reuse=reuse_baselines, budget_scale=1.,
                    checkpoint_qualifications=qualifications,
                    target_selection="frozen validation-selected checkpoint" if len(qualifications) == len(candidates)
                    else "explicit caller-supplied policy specifications; PPOAgent selections individually verified")
    immutable(output / "protocol.json", protocol)
    paths = [output / "results" / (identity["task_key"] + ".json") for identity in identities]
    completed = set()
    for i, path in enumerate(paths):
        if path.exists():
            _validate_result(read(path), tasks[i], identities[i])
            completed.add(i)
    resumed = len(completed)
    pending = [i for i in range(len(tasks)) if i not in completed]
    failures = []
    pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker) if pending else None
    running = {}
    last_print = time.perf_counter()
    deadline = False
    try:
        while pending or running:
            if time.perf_counter() - started >= max_seconds:
                deadline = True
                break
            while pending and len(running) < workers:
                i = pending.pop(0)
                running[i] = pool.apply_async(_worker, ((tasks[i], identities[i], str(paths[i])),))
            for i, job in list(running.items()):
                if not job.ready():
                    continue
                message = job.get()
                del running[i]
                if message["status"] == "completed":
                    _validate_result(read(paths[i]), tasks[i], identities[i])
                    completed.add(i)
                else:
                    failures.append({"task_index": i, "task": tasks[i], **message})
            if time.perf_counter() - last_print >= 15:
                print(json.dumps({"stage": "PPO frozen evaluation", "completed": len(completed),
                                  "expected": len(tasks), "failed": len(failures),
                                  "seconds": round(time.perf_counter() - started, 2)}), flush=True)
                last_print = time.perf_counter()
            if running:
                time.sleep(.05)
    finally:
        if pool is not None:
            pool.terminate() if deadline else pool.close()
            pool.join()
    # A worker can commit immediately before termination. The durable record,
    # not the acknowledgement, is authoritative.
    rows, slots = [], []
    for i, path in enumerate(paths):
        if path.exists():
            payload = read(path)
            _validate_result(payload, tasks[i], identities[i])
            completed.add(i)
            rows.append(payload["row"])
            slots.extend(payload["slots"])
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "episode_results.csv", index=False)
    pd.DataFrame(slots).to_csv(output / "slots.csv", index=False)
    segment = {"status": "completed" if len(completed) == len(tasks) else "incomplete",
               "valid": len(completed) == len(tasks), "completed_episodes": len(completed),
               "expected_episodes": len(tasks), "resumed_episodes": resumed, "new_completed": len(completed) - resumed,
               "missing_episodes": len(tasks) - len(completed), "failures": failures, "deadline_reached": deadline,
               "workers": workers, "max_seconds": max_seconds, "seconds": time.perf_counter() - started,
               "result_sha256": sha256_file(output / "episode_results.csv"),
               "slots_sha256": sha256_file(output / "slots.csv"), "protocol_sha256": sha256_file(output / "protocol.json")}
    dump(output / "status.json", segment)
    dump(output / f"runtime_{time.time_ns()}.json", segment)
    return frame, segment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", nargs="+", default=["period-24"])
    parser.add_argument("--roles", nargs="+", default=list(ROLES))
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--no-baseline-reuse", action="store_true")
    args = parser.parse_args(argv)
    candidates = read(args.candidate_manifest) if args.candidate_manifest else checkpoint_candidates(args.checkpoints or [])
    if isinstance(candidates, dict):
        candidates = candidates["candidates"]
    _, status = run_evaluation(candidates, args.output, days=args.days, roles=args.roles, scenarios=args.scenarios,
                              seeds=args.seeds, workers=args.workers, max_seconds=args.max_seconds,
                              reuse_baselines=not args.no_baseline_reuse)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return 0 if status["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
