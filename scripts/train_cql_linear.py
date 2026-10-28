"""Reproduce the separately recorded CQL linear-coordinate training candidate.

Run after the ordinary baseline models have been trained, before selection/test:
    python -X utf8 scripts/train_cql_linear.py --output outputs/my_fresh_run

--output is the experiment root containing the training NPZ and its provenance.
No validation/test market is read and no original candidate list is modified.
The supervisor is responsible for assigning a global candidate index/selection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harmony_baselines.data import sha256_file
from harmony_baselines.neural import (
    SUPPORTED_ACTION_TRANSFORMS, load_training_data, train_candidates,
)

DEFAULT_OUTPUT = ROOT / "outputs/baseline_comparison_20261028"
TRAINING_DAYS = ("period-7", "period-8")
TRAINING_SEEDS = (20261028, 20261029)
STEPS = 600
CHECKPOINT_STEPS = (200, 600)
NUM_THREADS = 2
CONFIG = {"learning_rate": 1e-4, "action_transform": "linear_scaled"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def preflight(output):
    """Verify the completed training-only source before creating any artifact."""
    output = Path(output).resolve()
    destination = output / "models_cql_linear"
    run_record = output / "models_cql_linear_run.json"
    if destination.exists() or run_record.exists():
        raise ValueError("refusing existing models_cql_linear or run record; use a fresh experiment --output")
    if (output / "selected_frozen.json").exists() or (output / "test_task_protocol.json").exists():
        raise ValueError("refusing supplemental training after selection/test has been frozen")
    if "linear_scaled" not in SUPPORTED_ACTION_TRANSFORMS:
        raise ValueError("this adapter does not support linear_scaled")
    provenance_path = output / "trajectory_provenance.json"
    protocol_path = output / "protocol.json"
    status_path = output / "behavior_episodes.status.json"
    provenance, protocol, status = map(_read, (provenance_path, protocol_path, status_path))
    train_days = provenance.get("days")
    split = protocol["executed_split"]
    if (provenance.get("test_used") is not False or train_days != list(TRAINING_DAYS)
            or train_days != split["train"]
            or set(train_days) & set(split["test"] + split["validation"])):
        raise ValueError("trajectory provenance must identify only the two prescribed training periods")
    if protocol.get("training_seeds") != list(TRAINING_SEEDS):
        raise ValueError("protocol training seeds differ from the reproduced CQL candidate")
    source = output / "train_transitions.npz"
    source_sha = sha256_file(source)
    if source_sha != provenance["sha256"]:
        raise ValueError("training NPZ SHA256 differs from trajectory provenance")
    if (status.get("status") != "completed" or status.get("episodes") != provenance["episodes"]
            or sha256_file(output / "behavior_episodes.csv") != status["csv_sha256"]):
        raise ValueError("behavior episodes are incomplete or their CSV digest changed")
    data, episodes = load_training_data(source)
    if len(data["states"]) != provenance["transitions"] or len(episodes) != provenance["episodes"]:
        raise ValueError("training NPZ counts differ from trajectory provenance")
    catalog = output / "models/candidates.json"
    original_records = _read(catalog)
    if not original_records:
        raise ValueError("ordinary candidate catalog must exist and be nonempty")
    protected = {str(catalog): sha256_file(catalog)}
    for candidate in original_records:
        path = Path(candidate["checkpoint"]).resolve()
        actual = sha256_file(path)
        if candidate.get("checkpoint_sha256", actual) != actual:
            raise ValueError(f"original checkpoint digest mismatch: {path}")
        protected[str(path)] = actual
    record = {
        "status": "running", "valid": False, "algorithm": "CQL", "output": str(destination),
        "source_sha256": source_sha, "training_days": train_days,
        "transitions": len(data["states"]), "episodes": len(episodes),
        "seeds": list(TRAINING_SEEDS), "steps": STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS), "num_threads": NUM_THREADS,
        "config": dict(CONFIG), "test_used": False, "original_assets_sha256": protected,
        "neural_sha256": sha256_file(ROOT / "harmony_baselines/neural.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "source_metadata_sha256": {str(path): sha256_file(path)
                                   for path in (provenance_path, protocol_path, status_path)},
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "selection": "not performed; supervisor assigns global config index and validates separately",
    }
    return output, source, destination, run_record, record


def run(output=DEFAULT_OUTPUT):
    output, source, destination, run_record, record = preflight(output)
    # Reserve the new attempt without overwriting any earlier run record.
    with run_record.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    started = time.perf_counter()
    try:
        candidates = train_candidates(
            source, destination, algorithms=("CQL",), configs={"CQL": [dict(CONFIG)]},
            seeds=TRAINING_SEEDS, steps=STEPS, checkpoint_steps=CHECKPOINT_STEPS,
            num_threads=NUM_THREADS,
        )
        if (len(candidates) != 4
                or {(r["seed"], r["step"]) for r in candidates}
                != {(seed, step) for seed in TRAINING_SEEDS for step in CHECKPOINT_STEPS}):
            raise ValueError("supplemental training did not produce all four expected checkpoints")
        if any(sha256_file(path) != digest for path, digest in record["original_assets_sha256"].items()):
            raise ValueError("an original checkpoint/catalog changed during supplemental training")
        if sha256_file(source) != record["source_sha256"]:
            raise ValueError("training source changed during supplemental training")
        record.update(status="completed", valid=True, seconds=time.perf_counter()-started,
                      candidates_path=str(destination / "candidates.json"), candidate_count=len(candidates),
                      original_assets_unchanged=True)
        print(json.dumps({k: record[k] for k in
                         ("status", "seconds", "candidate_count", "candidates_path")}), flush=True)
        return record
    except Exception as exc:
        record.update(status="failed", valid=False, error=repr(exc), seconds=time.perf_counter()-started)
        raise
    finally:
        _write(run_record, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="experiment root; requires training NPZ/provenance and original models")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))
    run(args.output)


if __name__ == "__main__":
    main()
