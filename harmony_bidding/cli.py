"""CPU-only unified environment commands; every attempt has an isolated run_id."""
import argparse
from hashlib import sha256
import json
import ntpath
from pathlib import Path, PureWindowsPath
import sys
from .data import generate_synthetic, load_days, write_dataset
from .runner import run_episode, write_episode, write_json, EpisodeFailure, RunRecord
from .schema import load_config, Rules


def _day_output_directory(run_directory, day_id):
    """A data identity is not a filesystem path, including on Windows."""
    if (not isinstance(day_id, str) or not day_id or day_id in {".", ".."}
            or any(character in day_id for character in '/\\:<>"|?*')
            or any(ord(character) < 32 for character in day_id)
            or day_id != day_id.rstrip(" .") or PureWindowsPath(day_id).drive
            or getattr(ntpath, "isreserved", lambda name: PureWindowsPath(name).is_reserved())(day_id)):
        raise ValueError("day_id must be a safe single directory name for reports")
    root = Path(run_directory).resolve()
    target = (root / day_id).resolve()
    if target == root or target.parent != root:
        raise ValueError("day report directory must remain strictly inside this run directory")
    return target


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Harmony local unified environment; official PID unavailable")
    parser.add_argument("command", choices=["generate-demo", "import-organizer", "run", "evaluate-single", "evaluate-multi",
                                           "export-trajectories", "benchmark"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="Output root; a command/run_id directory is created")
    parser.add_argument("--split", choices=["train", "validation", "local_holdout", "all"])
    args = parser.parse_args(argv)
    record = None
    config = None
    try:
        config = load_config(args.config)
        if args.split:
            config["split"] = None if args.split == "all" else args.split
        record = RunRecord(Path(args.output or config.get("output_dir", "outputs/local")) / args.command, args.command)
        record.state["requested_config_path"] = str(Path(args.config).resolve())
        record.state["requested_config_sha256"] = sha256(Path(args.config).read_bytes()).hexdigest()
        record.bind(config)
        output = record.directory
        if args.command == "import-organizer":
            from .organizer_data import import_organizer_csv
            manifest = import_organizer_csv(config)
            write_json(output / "import_report.json", {**record.identity(), **manifest})
            summary = {"data_dir": config["data_dir"], "days": len(manifest["days"]),
                       "import_metadata": manifest["import_metadata"]}
        elif args.command == "generate-demo":
            days = generate_synthetic(config)
            reused = False
            if (Path(config["data_dir"]) / "manifest.json").is_file():
                existing = load_days(config["data_dir"], Rules(**config.get("rules", {})), config.get("expected_n"))
                actual = {d.day_id: d.fingerprint for d in existing}
                if actual != {d.day_id: d.fingerprint for d in days}:
                    raise ValueError("existing dataset differs; choose a new data_dir (no files overwritten)")
                reused = True
            else:
                write_dataset(days, config["data_dir"])
            summary = {"data_dir": config["data_dir"], "days": len(days), "total_slots": len(days[0].slots),
                       "synthetic_only": True, "reused_identical_dataset": reused}
        else:
            rules = Rules(**config.get("rules", {}))
            manifest_path = Path(config["data_dir"]) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            config.pop("_dataset_import_metadata", None)
            if isinstance(manifest, dict) and "import_metadata" in manifest:
                config["_dataset_import_metadata"] = manifest["import_metadata"]
            record.bind(config)
            days = load_days(config["data_dir"], rules, config.get("expected_n"), config.get("split"))
            if args.command.startswith("evaluate-"):
                from .evaluation import evaluate
                result = evaluate(days, config, args.command.removeprefix("evaluate-"), run_record=record)
                summary = {"mode": result["mode"], "teams": result["teams"], "valid": result["valid"],
                           "local_only": True, "reference_strategy": "LocalPIDAgent"}
            elif args.command == "benchmark":
                from .benchmark import run_benchmark
                result = run_benchmark(next(days), config, run_record=record)
                summary = {**result["aggregate"], "memory": result["memory"], "tested_strategy": result["tested_strategy"]}
            else:
                settings = config.get("run", {})
                assignments, teams = settings.get("assignments", {}), settings.get("teams", {})
                summary = {"days": [], "dataset_fingerprints": {}, "local_only": True}
                def results():
                    count = 0
                    for day in days:
                        day_output = _day_output_directory(output, day.day_id)
                        mode = "configured_run" if assignments else "all_pid_reference"
                        result = run_episode(day, config, assignments, teams,
                                             episode_id=f"{args.command}-{day.day_id}", mode=mode, run_id=record.run_id)
                        write_episode(result, day_output)
                        summary["days"].append(result["summaries"])
                        summary["dataset_fingerprints"][day.day_id] = day.fingerprint
                        count += 1
                        yield result
                    if not count:
                        raise ValueError("no days selected")
                if args.command == "export-trajectories":
                    from .trajectory import export_trajectories, validate_trajectory_file
                    path = output / "trajectories.jsonl"
                    summary["transitions"] = export_trajectories(results(), path)
                    if validate_trajectory_file(path) != summary["transitions"]:
                        raise ValueError("trajectory readback count mismatch")
                else:
                    for _ in results():
                        continue
        summary.update(record.identity())
        record.complete(summary, summary.get("valid", True))
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        if record is None:
            # Invalid YAML/rules cannot supply a trusted output_dir. Explicit output
            # or this documented safe default still gives the failed attempt an ID.
            record = RunRecord(Path(args.output or "outputs/local") / args.command, args.command)
            record.state["requested_config_path"] = str(Path(args.config).resolve())
            if Path(args.config).is_file():
                record.state["requested_config_sha256"] = sha256(Path(args.config).read_bytes()).hexdigest()
        if isinstance(exc, EpisodeFailure):
            write_episode(exc.report, record.directory / "failed_episode")
        record.fail(exc)
        print(json.dumps(record.state, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
