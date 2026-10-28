"""Re-run the unified CPU workflow into a new evidence directory; no training."""
import argparse
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New verification directory (must not exist)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output or (root / "outputs" / "unified_validation" / uuid4().hex)).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Each verification import gets a fresh destination. The real importer must
    # retain its refusal to overwrite an existing dataset on repeated runs.
    import yaml
    sparse_config = yaml.safe_load((root / "configs/sparse_example.yaml").read_text(encoding="utf-8"))
    sparse_config["data_dir"] = str(output / "sparse_dataset")
    sparse_config_path = output / "sparse_example_runtime.yaml"
    sparse_config_path.write_text(yaml.safe_dump(sparse_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    cli = ["-m", "harmony_bidding.cli"]
    commands = [
        ("dependencies", ["-m", "pip", "check"]),
        ("cli_help", cli + ["--help"]),
        ("tiny_generate", cli + ["generate-demo", "--config", "configs/tiny.yaml"]),
        ("tiny_run", cli + ["run", "--config", "configs/tiny.yaml"]),
        ("demo_generate", cli + ["generate-demo", "--config", "configs/demo.yaml"]),
        ("demo_run", cli + ["run", "--config", "configs/demo.yaml"]),
        ("single", cli + ["evaluate-single", "--config", "configs/single.yaml"]),
        ("multi", cli + ["evaluate-multi", "--config", "configs/multi.yaml"]),
        ("trajectories", cli + ["export-trajectories", "--config", "configs/demo.yaml"]),
        ("benchmark", cli + ["benchmark", "--config", "configs/benchmark.yaml"]),
        ("sparse_import", cli + ["import-organizer", "--config", str(sparse_config_path)]),
        ("sparse_run", cli + ["run", "--config", str(sparse_config_path)]),
        ("sparse_trajectories", cli + ["export-trajectories", "--config", str(sparse_config_path)]),
        ("custom_run", cli + ["run", "--config", "configs/custom_example.yaml"]),
        ("custom_single", cli + ["evaluate-single", "--config", "configs/custom_example.yaml"]),
        ("native_run", cli + ["run", "--config", "configs/auctionnet_native_sample.yaml"]),
        ("demo_repeat", cli + ["run", "--config", "configs/demo.yaml"]),
        ("tests", ["-m", "pytest", "-q", "tests", f"--junitxml={output / 'pytest.xml'}",
                   "--basetemp", str(output / "test_tmp"), "-o", f"cache_dir={output / 'pytest_cache'}"]),
    ]
    results = []
    for name, arguments in commands:
        arguments = list(arguments)
        is_cli = arguments[:2] == cli and name != "cli_help"
        if is_cli:
            arguments += ["--output", str(output / "runs")]
        started = perf_counter()
        completed = subprocess.run([sys.executable, *arguments], cwd=root, capture_output=True,
                                   text=True, encoding="utf-8", timeout=600,
                                   env={**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1",
                                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        elapsed = perf_counter() - started
        (output / f"{name}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        results.append({"name": name, "command": [sys.executable, *arguments],
                        "exit_code": completed.returncode, "wall_seconds": elapsed,
                        "log": str(output / f"{name}.log")})
        if is_cli and completed.returncode == 0:
            results[-1]["summary"] = json.loads(completed.stdout)
        print(f"{name}: exit={completed.returncode}, {elapsed:.3f}s", flush=True)
        if completed.returncode:
            print(completed.stdout + completed.stderr, flush=True)
            break
    report = {"rules_version": "harmony-A-B-local-v2", "assumption_profile": "local-unified-v2",
              "executable": sys.executable, "cwd": str(root), "commands": results,
              "complete": len(results) == len(commands) and all(r["exit_code"] == 0 for r in results)}
    if report["complete"]:
        try:
            report["artifact_checks"] = check_artifacts(results)
        except Exception as exc:
            report["complete"] = False
            report["artifact_check_error"] = {"type": type(exc).__name__, "message": str(exc)}
    (output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"complete": report["complete"], "report": str(output / "validation.json")}, ensure_ascii=False), flush=True)
    return 0 if report["complete"] else 1


def check_artifacts(results):
    """Independently sum saved ledgers and score rows, including a hand result."""
    import csv
    by_name = {entry["name"]: entry["summary"] for entry in results if "summary" in entry}
    directories = {name: Path(summary["report_directory"]) for name, summary in by_name.items()}
    checked_episodes = 0
    for name, directory in directories.items():
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        if status["status"] != "completed" or status["run_id"] != by_name[name]["run_id"]:
            raise AssertionError("run identity or completion mismatch")
        for episode_file in directory.rglob("episode.json"):
            episode = json.loads(episode_file.read_text(encoding="utf-8"))
            with (episode_file.parent / "slots.csv").open(encoding="utf-8", newline="") as handle:
                slots = list(csv.DictReader(handle))
            for identity, summary in episode["summaries"].items():
                own = [row for row in slots if row["advertiser_id"] == identity]
                assert len(own) == summary["bidding_calls"] == summary["update_calls"]
                assert [int(row["time_slot"]) for row in own] == list(range(len(own)))
                assert sum(Decimal(row["cost"]) for row in own) == Decimal(str(summary["cost"]))
                assert Decimal(str(summary["remaining_budget"])) + Decimal(str(summary["cost"])) == Decimal(str(summary["budget"]))
                assert summary["cost"] <= summary["budget"]
                assert sum(int(row["wins"]) for row in own) == summary["wins"]
                assert math.isclose(sum(float(row["value"]) for row in own), summary["value"], rel_tol=1e-12)
            checked_episodes += 1
    sparse = json.loads(next(directories["sparse_run"].rglob("episode.json")).read_text(encoding="utf-8"))
    a = sparse["summaries"]["A"]
    assert (a["wins"], a["cost"], a["value"], a["remaining_budget"]) == (3, 13, 17, 87)
    assert all(s["bidding_calls"] == s["update_calls"] == 72 for s in sparse["summaries"].values())
    evaluations = {}
    for name in ("single", "multi", "custom_single"):
        evaluation = json.loads((directories[name] / "evaluation.json").read_text(encoding="utf-8"))
        assert evaluation["valid"] and evaluation["local_only"] and not evaluation["official_strategy_available"]
        for team, summary in evaluation["teams"].items():
            games = [game for game in evaluation["games"] if game["team"] == team]
            assert len(games) == summary["games_expected"]
            assert all(game["pid_reference_value"] > 0 for game in games)
            ratio_mean = sum(game["value"] / game["pid_reference_value"] for game in games) / len(games)
            assert math.isclose(summary["G"], ratio_mean, rel_tol=1e-12)
        evaluations[name] = evaluation["teams"]
    assert evaluations["single"]["A"]["G"] == 1.0
    benchmark = json.loads((directories["benchmark"] / "benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["aggregate"]["samples_per_function"] == 72 * 20
    assert benchmark["tested_strategy"]["class"].endswith(":ConstantAlphaAgent")
    benchmark_status = json.loads((directories["benchmark"] / "status.json").read_text(encoding="utf-8"))
    assert benchmark["config_fingerprint"] == benchmark_status["config_fingerprint"]
    first = json.loads(next(directories["demo_run"].rglob("episode.json")).read_text(encoding="utf-8"))
    repeated = json.loads(next(directories["demo_repeat"].rglob("episode.json")).read_text(encoding="utf-8"))
    def business(report):
        return {"summaries": report["summaries"], "checks": report["checks"],
                "slots": [{k: v for k, v in row.items() if k != "run_id"} for row in report["slots"]]}
    assert business(first) == business(repeated)
    return {"checked_episode_ledgers": checked_episodes, "sparse_hand_result": a,
            "evaluations": evaluations, "benchmark": benchmark["aggregate"],
            "business_reproduction": True, "all_cli_runs_have_matching_completed_id": True}


if __name__ == "__main__":
    raise SystemExit(main())
