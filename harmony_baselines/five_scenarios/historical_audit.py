"""Seal and verify the historical experiment without changing any historical file."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/baseline_comparison_20261028"
DEFAULT_OUT = ROOT / "outputs/baseline_five_scenarios_20261028"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def snapshot(out=DEFAULT_OUT):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "historical_assets_manifest.json"
    audit_path = out / "historical_audit.json"
    if manifest_path.exists() or audit_path.exists():
        raise ValueError("Historical baseline already sealed; use --verify, never replace it")
    frozen = read(OLD / "selected_frozen.json")
    protocol = read(OLD / "protocol.json")
    status = read(OLD / "episode_results.status.json")
    catalog = read(OLD / "models/candidates.json")
    before = read(OLD / "models/candidates_before_stability.json")
    paths = {p.resolve(): "historical_output" for p in OLD.rglob("*") if p.is_file()}
    for relative in frozen["code_sha256"]:
        paths[(ROOT / relative).resolve()] = "frozen_source"
    for p in (ROOT / "harmony_baselines").glob("*"):
        if p.is_file():
            paths[p.resolve()] = "historical_baseline_source_or_docs"
    for p in ROOT.parent.glob("*baseline*.md"):
        paths[p.resolve()] = "historical_report"
    for p in ROOT.glob("*baseline*.md"):
        paths[p.resolve()] = "historical_report"
    assets, mismatches = [], []
    for p, kind in sorted(paths.items(), key=lambda item: str(item[0])):
        stat = p.stat()
        assets.append({"path": str(p), "kind": kind, "size": stat.st_size,
                       "mtime_ns": stat.st_mtime_ns, "sha256": sha256(p)})
    hashes = {row["path"]: row["sha256"] for row in assets}
    for relative, expected in frozen["code_sha256"].items():
        p = str((ROOT / relative).resolve())
        if hashes[p] != expected:
            mismatches.append({"kind": "frozen_source", "path": p})
    for candidate in catalog:
        p = str(Path(candidate["checkpoint"]).resolve())
        if hashes.get(p) != candidate["checkpoint_sha256"]:
            mismatches.append({"kind": "catalog_checkpoint", "path": p})
    for name, expected in frozen["checkpoints"].items():
        p = str(Path(name).resolve())
        if hashes.get(p) != expected:
            mismatches.append({"kind": "selected_checkpoint", "path": p})
    if before != catalog[:len(before)]:
        mismatches.append({"kind": "historical_catalog_prefix"})
    # Parse the original CSV directly; no evaluator/import of old report code.
    import pandas as pd
    frame = pd.read_csv(OLD / "episode_results.csv", dtype={"player_id": str, "player_type": str})
    keys = ["algorithm", "scenario", "day_id", "player_id", "seed", "training_seed"]
    if len(frame) != 4320 or frame.duplicated(keys).any() or not frame.completed.all():
        mismatches.append({"kind": "historical_coverage"})
    metadata_changes = []
    for row in assets:
        current = Path(row["path"]).stat()
        if (current.st_size, current.st_mtime_ns) != (row["size"], row["mtime_ns"]):
            metadata_changes.append(row["path"])
    if metadata_changes:
        mismatches.append({"kind": "changed_during_snapshot", "paths": metadata_changes})
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "old_root": str(OLD),
                "files": assets, "file_count": len(assets),
                "all_historical_output_files_sha256": True,
                "historical_output_file_count": sum(p.is_file() for p in OLD.rglob("*"))}
    write(manifest_path, manifest)
    audit = {"status": "passed" if not mismatches else "failed", "valid": not mismatches,
        "scope": "read-only historical audit; all old output bytes/source/checkpoints/reports sealed",
        "created_utc": manifest["created_utc"], "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path), "mismatches": mismatches,
        "old_root": str(OLD), "historical_protocol": protocol, "historical_status": status,
        "historical_episodes": len(frame), "historical_algorithms": sorted(frame.algorithm.unique()),
        "historical_scenarios": sorted(frame.scenario.unique()),
        "historical_selected_strategies": frozen["selected"],
        "frozen_code_count": len(frozen["code_sha256"]), "catalog_checkpoints": len(catalog),
        "selected_checkpoints": len(frozen["checkpoints"]),
        "catalog_original_48_preserved": before == catalog[:len(before)],
        "historical_coverage_by_algorithm": frame.groupby("algorithm").size().to_dict(),
        "reuse": {"A_test_rows": 4320, "train_episodes": 288, "train_transitions": 20736,
            "old_neural_weights_in_A": "unchanged two seeds per algorithm",
            "B_changed_policy_results": "must run all five scenarios; no mixed-model concatenation",
            "old_PID_scenario_import": "canonical LocalPID only in new files; retain source_scenario=PID",
            "old_diagnostics": "72-slot alpha/cost/value/balance; wins unavailable, never fabricate"},
        "counts": {"old_strategy_instances": 15, "A_old_new_scenario_episodes": 2880,
            "A_AuctionNetPID_episodes": 480, "A_new_episodes": 3360, "A_total_episodes": 7680,
            "B_strategy_instances": 16, "B_total_episodes": 7680, "independent_B_period25_episodes": 7680,
            "user_estimate_2208": "not compatible with 48 roles, 2 evaluation seeds and paired neural training seeds"},
        "known_documentation_differences": str(OLD / "detailed_analysis/issues.md"),
        "historical_results_reusable": not mismatches,
        "limits": ["period-24 has already been inspected; no longer a fresh blind test",
                   "P10 order must be explicit; A/B use their own LocalPID denominators",
                   "historical input manifest is retained unchanged; no old generator was rerun"]}
    write(audit_path, audit)
    return audit


def verify(out=DEFAULT_OUT):
    out = Path(out)
    manifest = read(out / "historical_assets_manifest.json")
    mismatches = []
    for row in manifest["files"]:
        p = Path(row["path"])
        if not p.is_file():
            mismatches.append({"path": str(p), "reason": "missing"})
            continue
        stat = p.stat()
        if stat.st_size != row["size"] or sha256(p) != row["sha256"]:
            mismatches.append({"path": str(p), "reason": "content changed"})
        elif stat.st_mtime_ns != row["mtime_ns"]:
            mismatches.append({"path": str(p), "reason": "mtime changed"})
    return {"status": "passed" if not mismatches else "failed", "checked_files": len(manifest["files"]),
            "checked_utc": datetime.now(timezone.utc).isoformat(), "mismatches": mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify(args.output) if args.verify else snapshot(args.output)
    if args.verify:
        write(args.output / "historical_integrity_verification.json", result)
    print(json.dumps({key: result[key] for key in ("status", "mismatches")}, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
