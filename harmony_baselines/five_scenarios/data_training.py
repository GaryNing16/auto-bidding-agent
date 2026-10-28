"""Audited 288+192 trajectory reuse and restartable, from-scratch neural runs.

Only complete runs are reusable. A restart creates another attempt directory;
it never claims optimizer/RNG continuation from an intermediate checkpoint.
Scenario/source labels live in the episode CSV, never in actor state arrays.
"""
from __future__ import annotations

from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from harmony_baselines.data import sha256_file
from harmony_baselines.neural import load_training_data, provenance, train_candidates
from harmony_bidding.schema import fingerprint
from . import common as c, execution
from .evaluator import historical_scenario_name

ARRAY_KEYS = ("states", "actions", "rewards", "next_states", "dones", "episode_id")
CHECKPOINT_STEPS = (600, 1800)
TRAINING_STEPS = 1800


def _arrays(path, episodes):
    """Preserve saved dtypes; the old reader independently validates legality."""
    _, grouped = load_training_data(path)
    if len(grouped) != episodes:
        raise ValueError(f"Expected {episodes} complete episodes in {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_KEYS):
            raise ValueError("Training NPZ must contain only the six established arrays")
        arrays = {key: archive[key].copy() for key in ARRAY_KEYS}
    if any(arrays[key].shape != (72 * episodes,) for key in ("actions", "rewards", "dones", "episode_id")):
        raise ValueError("Expected one scalar action/reward/done/id per transition")
    clock = (np.arange(72) / 72).astype(np.float32)
    next_clock = (np.arange(1, 73) / 72).astype(np.float32)
    for indices in grouped:
        if (not np.array_equal(arrays["states"][indices, 0], clock)
                or not np.array_equal(arrays["next_states"][indices, 0], next_clock)
                or not np.array_equal(arrays["next_states"][indices[:-1]], arrays["states"][indices[1:]])):
            raise ValueError("Trajectory state order/clock/history continuity mismatch")
    return arrays


def _write_arrays_once(path, arrays):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                raise ValueError(f"Refusing to overwrite differing training arrays: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def _write_csv_once(path, frame):
    content = frame.to_csv(index=False, lineterminator="\n")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to overwrite differing trajectory metadata: {path}")
        return
    pending = path.with_name(path.name + ".pending")
    pending.write_text(content, encoding="utf-8")
    pending.replace(path)


def _legacy_inputs(opponents):
    paths = {name: c.OLD / name for name in ("train_transitions.npz", "trajectory_provenance.json",
             "behavior_episodes.csv", "behavior_episodes.status.json", "opponents_frozen.json")}
    digests = {name: sha256_file(path) for name, path in paths.items()}
    original = c.read(paths["opponents_frozen.json"])
    if any(opponents.get(key) != original[key] for key in ("fixed", "pid")):
        raise ValueError("Reusing historical trajectories requires unchanged Fixed/LocalPID opponents")
    source = c.read(paths["trajectory_provenance.json"])
    status = c.read(paths["behavior_episodes.status.json"])
    expected = len(c.TRAIN_DAYS) * len(c.ROLES) * len(c.OLD_SCENARIOS)
    if (source["sha256"] != digests["train_transitions.npz"] or source["episodes"] != expected
            or source["transitions"] != expected * 72 or source["test_used"] is not False
            or source["days"] != list(c.TRAIN_DAYS) or status["status"] != "completed"
            or status["episodes"] != expected or status["csv_sha256"] != digests["behavior_episodes.csv"]):
        raise ValueError("Historical trajectory provenance/completion mismatch")
    behaviors = source["behavior_candidates"]
    if len(behaviors) != 6:
        raise ValueError("Expected six frozen historical behavior specifications")
    frame = pd.read_csv(paths["behavior_episodes.csv"], dtype={"player_id": str})
    if len(frame) != expected or frame.task_index.duplicated().any() or not frame.completed.eq(True).all():
        raise ValueError("Historical behavior coverage is incomplete")
    frame = frame.set_index("task_index")
    if set(frame.index) != set(range(expected)):
        raise ValueError("Historical task indices differ from contiguous 288-task protocol")
    arrays = _arrays(paths["train_transitions.npz"], expected)
    episodes = []
    for di, day in enumerate(c.TRAIN_DAYS):
        for pi, player in enumerate(c.ROLES):
            for si, scenario in enumerate(c.OLD_SCENARIOS):
                index = (di * len(c.ROLES) + pi) * len(c.OLD_SCENARIOS) + si
                row = frame.loc[index]
                behavior = behaviors[(pi + si + di) % 6]
                if (row.day_id != day or row.player_id != player or int(row.seed) != c.SEEDS[di]
                        or historical_scenario_name(row.scenario) != scenario
                        or row.algorithm != behavior["algorithm"]
                        or row.model_fingerprint != fingerprint(behavior["spec"])):
                    raise ValueError(f"Historical source index/behavior mismatch: {index}")
                block = slice(index * 72, (index + 1) * 72)
                episode_id = f"{index}:{player}"
                if not np.all(arrays["episode_id"][block] == episode_id):
                    raise ValueError(f"Historical NPZ episode order differs at source index {index}")
                if not np.isclose(arrays["rewards"][block].sum(), row.total_value, atol=1e-8, rtol=0):
                    raise ValueError("Historical NPZ rewards differ from episode value")
                episodes.append(dict(episode_index=index, episode_id=episode_id, day_id=day,
                    scenario=scenario, source_scenario=row.scenario, player_id=player, seed=int(row.seed),
                    training_seed=0, algorithm=behavior["algorithm"], behavior_spec_json=json.dumps(behavior["spec"], sort_keys=True),
                    model_fingerprint=row.model_fingerprint, assignment_fingerprint=row.assignment_fingerprint,
                    dataset_fingerprint=row.dataset_fingerprint, source_run=str(c.OLD.resolve()),
                    source_task_index=index, source_episode_id=episode_id, provenance="reused_historical",
                    source_transition_path=str(paths["train_transitions.npz"].resolve()),
                    source_transition_sha256=digests["train_transitions.npz"], row_start=index * 72, rows=72))
    return arrays, episodes, behaviors, digests


def generate_training(opponents, workers=4):
    """Reuse the old 288 episodes and execute only the 192 new-scene episodes."""
    started = perf_counter()
    c.OUT.mkdir(parents=True, exist_ok=True)
    old, metadata, behaviors, source_hashes = _legacy_inputs(opponents)
    old_count = len(metadata)
    scenes = c.SCENARIOS[3:]
    tasks = []
    for di, day in enumerate(c.TRAIN_DAYS):
        for pi, player in enumerate(c.ROLES):
            for si, scene in enumerate(scenes, 3):
                behavior = behaviors[(pi + si + di) % 6]
                tasks.append(dict(day_id=day, target=player, target_spec=behavior["spec"], scenario=scene,
                    opponents=opponents, seed=c.SEEDS[di], algorithm=behavior["algorithm"],
                    training_seed=0, collect=True, collect_all=False))
    plan = {"source_hashes": source_hashes, "opponents": opponents, "tasks": tasks,
            "behavior_candidates": behaviors, "source_code_sha256": sha256_file(__file__),
            "labels_are_actor_inputs": False, "old_episode_ids_preserved": True}
    c.immutable(c.OUT / "trajectory_plan.json", plan)
    destination = c.OUT / "train_transitions.npz"
    sidecar = c.OUT / "trajectory_episodes.csv"
    receipt_path = c.OUT / "trajectory_provenance.json"
    if receipt_path.exists():
        receipt = c.read(receipt_path)
        if (receipt.get("status") != "completed" or receipt["plan_fingerprint"] != fingerprint(plan)
                or receipt["sha256"] != sha256_file(destination)
                or receipt["episode_metadata_sha256"] != sha256_file(sidecar)):
            raise ValueError("Completed five-scenario training data changed")
        for path, digest in receipt["extension_artifacts"].items():
            if sha256_file(path) != digest:
                raise ValueError(f"Completed extension artifact changed: {path}")
        _arrays(destination, old_count + len(tasks))
        return receipt
    extension = c.OUT / "trajectory_extension"
    frame, failures, skipped = execution.execute(tasks, extension, "trajectory_extension", workers=workers,
                                                fail_group=False)
    if failures or skipped or len(frame) != len(tasks) or set(frame.task_index) != set(range(len(tasks))):
        raise RuntimeError("All 192 new-scene episodes must complete before constructing training data")
    protocol = c.read(extension / "task_protocol.json")
    if protocol["tasks"] != tasks or len(protocol["identities"]) != len(tasks):
        raise ValueError("Extension task protocol differs from the fixed data-generation plan")
    parts = {key: [value] for key, value in old.items()}
    extension_artifacts = {}
    for index, task in enumerate(tasks):
        result_path = extension / "results" / f"task_{index:05d}.json"
        result = c.read(result_path)
        row = result["row"]
        transition_path = result_path.with_suffix(".npz").resolve()
        if (result["status"] != "completed" or result["identity"] != protocol["identities"][index]
                or Path(result["transition_path"]).resolve() != transition_path
                or result.get("transition_sha256") != sha256_file(transition_path)
                or int(row["task_index"]) != index):
            raise ValueError(f"Extension result/NPZ identity mismatch: {index}")
        for key, expected in (("scenario", task["scenario"]), ("day_id", task["day_id"]),
                              ("player_id", task["target"]), ("seed", task["seed"]),
                              ("algorithm", task["algorithm"]), ("training_seed", 0)):
            if str(row[key]) != str(expected):
                raise ValueError(f"Extension source index mismatch: {index}/{key}")
        arrays = _arrays(transition_path, 1)
        source_id = f'new:{index}:{task["target"]}'
        if (not np.all(arrays["episode_id"] == source_id)
                or not np.isclose(arrays["rewards"].sum(), row["total_value"], atol=1e-8, rtol=0)
                or row["model_fingerprint"] != fingerprint(task["target_spec"])):
            raise ValueError(f"Extension transition source/reward mismatch: {index}")
        global_index = old_count + index
        episode_id = f'{global_index}:{task["target"]}'
        arrays["episode_id"] = np.repeat(episode_id, 72)
        for key in ARRAY_KEYS:
            parts[key].append(arrays[key])
        digest = result["transition_sha256"]
        extension_artifacts[str(result_path.resolve())] = sha256_file(result_path)
        extension_artifacts[str(transition_path)] = digest
        metadata.append(dict(episode_index=global_index, episode_id=episode_id, day_id=task["day_id"],
            scenario=task["scenario"], source_scenario=task["scenario"], player_id=task["target"],
            seed=task["seed"], training_seed=0, algorithm=task["algorithm"],
            behavior_spec_json=json.dumps(task["target_spec"], sort_keys=True), model_fingerprint=row["model_fingerprint"],
            assignment_fingerprint=row["assignment_fingerprint"], dataset_fingerprint=row["dataset_fingerprint"],
            source_run=str(extension.resolve()), source_task_index=index, source_episode_id=source_id,
            provenance="new_execution", source_transition_path=str(transition_path),
            source_transition_sha256=digest, row_start=global_index * 72, rows=72))
    counts = Counter(row["scenario"] for row in metadata)
    if counts != {scene: len(c.TRAIN_DAYS) * len(c.ROLES) for scene in c.SCENARIOS}:
        raise ValueError("Five-scenario trajectory coverage is not balanced")
    merged = {key: np.concatenate(values) for key, values in parts.items()}
    _write_arrays_once(destination, merged)
    _arrays(destination, len(metadata))
    _write_csv_once(sidecar, pd.DataFrame(metadata))
    receipt = {"status": "completed", "source": "unchanged historical three-scene trajectories plus new unified-core rollouts",
        "days": list(c.TRAIN_DAYS), "episodes": len(metadata), "transitions": len(metadata) * 72,
        "reused_episodes": old_count, "new_episodes": len(tasks), "scenario_counts": dict(counts),
        "behavior_candidates": behaviors, "behavior_selection": "(role_index + global_scenario_index + day_index) % 6",
        "new_global_scenario_indices": [3, 4], "source_hashes": source_hashes,
        "plan_fingerprint": fingerprint(plan), "sha256": sha256_file(destination),
        "episode_metadata_path": str(sidecar.resolve()), "episode_metadata_sha256": sha256_file(sidecar),
        "extension_artifacts": extension_artifacts, "labels_are_actor_inputs": False, "test_used": False,
        "seconds_this_completion_call": perf_counter() - started}
    c.immutable(receipt_path, receipt)
    return receipt


def _training_code_hashes():
    paths = [Path(__file__), c.ROOT / "harmony_baselines/neural.py", c.ROOT / "harmony_baselines/upstream.py"]
    paths += [c.ROOT / item["path"] for item in provenance()["files"].values()]
    return {str(path.resolve()): sha256_file(path) for path in paths}


def _completed_attempt(status_path, identity):
    receipt = c.read(status_path)
    if receipt.get("status") != "completed" or receipt.get("identity") != identity:
        raise ValueError("Training attempt is incomplete or has different inputs/configuration")
    for path, digest in receipt["artifacts"].items():
        if sha256_file(path) != digest:
            raise ValueError(f"Training artifact changed: {path}")
    if {int(r["step"]) for r in receipt["candidates"]} != set(CHECKPOINT_STEPS) or len(receipt["candidates"]) != 2:
        raise ValueError("Incomplete checkpoint pair")
    attempt = status_path.with_name(status_path.name.removesuffix(".status.json"))
    reconstructed = _finish_attempt(attempt, identity, c.read(attempt / "candidates.json"),
                                    receipt["call_wall_seconds"])
    if reconstructed != receipt:
        raise ValueError("Training receipt differs from audited candidate/log artifacts")
    return receipt


def _finish_attempt(attempt, identity, records, elapsed):
    metadata_path = attempt / "training_metadata.json"
    log_path = attempt / identity["algorithm"] / f'config_0_seed_{identity["seed"]}' / "training.json"
    metadata, log = c.read(metadata_path), c.read(log_path)
    if (c.read(attempt / "failures.json") or metadata["source_sha256"] != identity["data_sha256"]
            or log["status"] != "completed" or not log["history"]
            or int(log["seed"]) != identity["seed"]
            or any(log["config"].get(k) != v for k, v in identity["variant"].items())
            or int(log["history"][-1]["step"]) != TRAINING_STEPS
            or len(records) != 2 or {int(r["step"]) for r in records} != set(CHECKPOINT_STEPS)):
        raise ValueError("Training run did not produce both complete checkpoints/logs")
    candidates, paths = [], [metadata_path, log_path, attempt / "candidates.json", attempt / "failures.json"]
    for record in records:
        path = Path(record["checkpoint"]).resolve()
        if (not path.is_relative_to(attempt.resolve()) or record["algorithm"] != identity["algorithm"]
                or record["config_index"] != 0 or record["parameter_count"] != log["parameters"]
                or int(record["seed"]) != identity["seed"] or record["metadata"]["source_sha256"] != identity["data_sha256"]
                or any(record["config"].get(key) != value for key, value in identity["variant"].items())
                or record["checkpoint_sha256"] != sha256_file(path) or record["parameter_count"] <= 0):
            raise ValueError("Trained candidate identity/hash/configuration mismatch")
        candidates.append({**record, "origin_config_index": record["config_index"],
            "config_index": identity["config_index"], "run_id": identity["run_id"],
            "attempt": attempt.name, "training_log": str(log_path.resolve()),
            "training_metadata_path": str(metadata_path.resolve()), "training_origin": "from_scratch"})
        paths.append(path)
    return {"status": "completed", "identity": identity, "attempt": str(attempt.resolve()),
            "candidates": candidates, "artifacts": {str(p.resolve()): sha256_file(p) for p in paths},
            "training_metadata_path": str(metadata_path.resolve()), "training_log": str(log_path.resolve()),
            "training_seconds": float(log["seconds"]), "call_wall_seconds": elapsed,
            "parameter_count": int(log["parameters"]), "resume_semantics": "completed-run reuse; otherwise restart from scratch"}


def train_models():
    """Train 24 independent runs; 600/1800 checkpoints, two seeds per setting.

    No test/validation results enter these settings. Interrupted/corrupt attempts
    are preserved. Incomplete runs are retried on the next call in a new folder.
    A failed call saves all healthy runs but raises before model selection.
    """
    data_path = c.OUT / "train_transitions.npz"
    data_receipt_path = c.OUT / "trajectory_provenance.json"
    data_receipt = c.read(data_receipt_path)
    sidecar = c.OUT / "trajectory_episodes.csv"
    expected_episodes = len(c.TRAIN_DAYS) * len(c.ROLES) * len(c.SCENARIOS)
    if (data_receipt.get("status") != "completed" or data_receipt.get("test_used") is not False
            or data_receipt["episodes"] != expected_episodes or data_receipt["sha256"] != sha256_file(data_path)
            or data_receipt["days"] != list(c.TRAIN_DAYS)
            or Path(data_receipt["episode_metadata_path"]).resolve() != sidecar.resolve()
            or data_receipt["episode_metadata_sha256"] != sha256_file(sidecar)
            or data_receipt["scenario_counts"] != {s: len(c.TRAIN_DAYS) * len(c.ROLES) for s in c.SCENARIOS}):
        raise ValueError("Training requires completed, balanced five-scenario data")
    _arrays(data_path, expected_episodes)
    root = c.OUT / "models"
    root.mkdir(parents=True, exist_ok=True)
    code = _training_code_hashes()
    environment = {"python": sys.version, "numpy": np.__version__, "torch": importlib.metadata.version("torch")}
    records, run_receipts, failures = [], [], []
    for algorithm in c.NEURAL:
        rates = (1e-4, 3e-4) if algorithm == "CQL" else (3e-4, 1e-3)
        for config_index, rate in enumerate(rates):
            variant = {"learning_rate": rate, "action_transform": "linear_scaled" if algorithm == "CQL" else "log1p"}
            for seed in c.SEEDS:
                run_id = f"{algorithm}/config_{config_index}_seed_{seed}"
                run_root = root / "runs" / run_id
                identity = {"run_id": run_id, "algorithm": algorithm, "config_index": config_index,
                    "seed": seed, "variant": variant, "steps": TRAINING_STEPS,
                    "checkpoint_steps": list(CHECKPOINT_STEPS), "batch_size": 64, "num_threads": 2,
                    "data_path": str(data_path.resolve()), "data_sha256": data_receipt["sha256"],
                    "data_provenance_sha256": sha256_file(data_receipt_path),
                    "episode_metadata_sha256": data_receipt["episode_metadata_sha256"],
                    "code_sha256": code, "environment": environment, "training_origin": "from_scratch"}
                c.immutable(run_root / "run_identity.json", identity)
                completed = None
                status_paths = sorted(run_root.glob("attempt_*.status.json"),
                                      key=lambda p: int(p.name.split(".")[0].split("_")[1]), reverse=True)
                for status_path in status_paths:
                    try:
                        completed = _completed_attempt(status_path, identity)
                        break
                    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
                        # Preserve the original completion/partial artifacts and
                        # write a separate audit receipt, never replace weights.
                        audit = status_path.with_name(status_path.name + ".rejected.json")
                        if not audit.exists():
                            c.dump(audit, {"status": "not_reusable", "source": str(status_path), "reason": str(error)})
                if completed is None:
                    numbers = [int(m.group(1)) for p in run_root.iterdir()
                               if (m := re.match(r"attempt_(\d+)(?:\.|$)", p.name))]
                    attempt_number = max(numbers, default=-1) + 1
                    attempt = run_root / f"attempt_{attempt_number}"
                    c.immutable(run_root / f"attempt_{attempt_number}.request.json", identity)
                    started = perf_counter()
                    try:
                        trained = train_candidates(data_path, attempt, algorithms=[algorithm], seeds=[seed],
                            steps=TRAINING_STEPS, checkpoint_steps=CHECKPOINT_STEPS,
                            configs={algorithm: [variant]}, batch_size=64, num_threads=2)
                        completed = _finish_attempt(attempt, identity, trained, perf_counter() - started)
                        c.immutable(run_root / f"attempt_{attempt_number}.status.json", completed)
                    except Exception as error:
                        failure = {"status": "failed", "identity": identity, "attempt": str(attempt.resolve()),
                            "exception_type": type(error).__name__, "reason": str(error),
                            "call_wall_seconds": perf_counter() - started,
                            "resume_semantics": "next call restarts this run in a new attempt directory"}
                        c.immutable(run_root / f"attempt_{attempt_number}.status.json", failure)
                        failures.append(failure)
                if completed is not None:
                    records.extend(completed["candidates"])
                    run_receipts.append(completed)
                c.dump(root / "candidates.json", records)
    c.dump(root / "failures.json", failures)
    c.dump(root / "training_runs.json", run_receipts)
    if run_receipts:
        shared = c.read(run_receipts[0]["training_metadata_path"])
        shared.update(action_transforms=["linear_scaled", "log1p"], action_transform_scope="per-run explicit config",
                      training_origin="from_scratch", completed_runs=len(run_receipts))
        c.dump(root / "training_metadata.json", shared)
    c.dump(root / "training_status.json", {"status": "completed" if not failures else "incomplete",
        "expected_runs": len(c.NEURAL) * 2 * len(c.SEEDS), "completed_runs": len(run_receipts),
        "checkpoints": len(records), "failed_runs": len(failures), "steps": TRAINING_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS), "data_sha256": data_receipt["sha256"],
        "resume_semantics": "run-level; no optimizer/scheduler/RNG continuation"})
    if failures:
        raise RuntimeError(f"{len(failures)} neural runs incomplete; completed runs retained for resume")
    return records
