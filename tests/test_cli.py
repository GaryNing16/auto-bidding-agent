"""Small actual CLI integration; performance measurements remain a separate command."""
import json
from pathlib import Path
import subprocess
import sys
import yaml
import pytest
from harmony_bidding.schema import load_config
from harmony_bidding.runner import business_result
from harmony_bidding.trajectory import read_trajectories


ROOT = Path(__file__).resolve().parents[1]


def invoke(command, config):
    process = subprocess.run([sys.executable, "-m", "harmony_bidding.cli", command, "--config", str(config)],
                             cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    return json.loads(process.stdout)


def test_full_72_cli_roundtrip_reference_single_multi_and_trajectories(tmp_path):
    config = load_config(ROOT / "configs/demo.yaml")
    config.update(data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "outputs"))
    config["synthetic"]["base_opportunities"] = 3
    config["evaluation"] = {
        "roles": ["advertiser_1"], "teams": {"pid": {"kind": "pid"}}, "isolation": "process"}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert invoke("generate-demo", path)["total_slots"] == 72
    assert invoke("generate-demo", path)["reused_identical_dataset"]
    first = invoke("run", path)
    directory = Path(first["report_directory"]) / "2026-10-01"
    result = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    assert len(result["slots"]) == 4 * 72
    assert all(s["bidding_calls"] == s["update_calls"] == 72 for s in result["summaries"].values())
    again = invoke("run", path)
    repeat = json.loads((Path(again["report_directory"]) / "2026-10-01/episode.json").read_text(encoding="utf-8"))
    assert first["run_id"] != again["run_id"]
    assert business_result(result) == business_result(repeat)
    for command in ("evaluate-single", "evaluate-multi"):
        evaluated = invoke(command, path)
        report = json.loads((Path(evaluated["report_directory"]) / "evaluation.json").read_text(encoding="utf-8"))
        assert report["teams"]["pid"]["G"] == 1
        assert report["valid"]
    summary = invoke("export-trajectories", path)
    assert summary["transitions"] == 288
    rows = list(read_trajectories(Path(summary["report_directory"]) / "trajectories.jsonl"))
    assert len(rows) == 288 and sum(r["done"] for r in rows) == 4
    assert not (Path(summary["report_directory"]) / "training_statistics.json").exists()


def test_cli_unicode_paths_and_error_reports(tmp_path):
    config = load_config(ROOT / "configs/tiny.yaml")
    config.update(data_dir=str(tmp_path / "中文数据"), output_dir=str(tmp_path / "中文结果"))
    path = tmp_path / "配置.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    invoke("generate-demo", path)
    invoke("run", path)
    config["data_dir"] = str(tmp_path / "不存在的数据")
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    failed = subprocess.run([sys.executable, "-m", "harmony_bidding.cli", "run", "--config", str(path)],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert failed.returncode == 1
    assert json.loads(failed.stderr)["status"] == "failed"
    assert "UnicodeEncodeError" not in failed.stderr


def test_configured_run_export_and_benchmark_share_target_strategy(tmp_path):
    config = load_config(ROOT / "configs/tiny.yaml")
    config.update(data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "outputs"))
    role = "advertiser_1"
    config["run"] = {"assignments": {role: "constant"}, "teams": {"constant": {"kind": "constant", "alpha": 40}}}
    config["benchmark"] = {"target_role": role, "agent": {"kind": "constant", "alpha": 40},
                           "warmup_episodes": 1, "episodes": 2, "min_samples": 1000}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    invoke("generate-demo", path)
    run = invoke("run", path)
    episodes = list(Path(run["report_directory"]).glob("*/episode.json"))
    assert episodes
    episode = json.loads(episodes[0].read_text())
    assert {r["alpha"] for r in episode["slots"] if r["advertiser_id"] == role} == {40}
    assert episode["metadata"]["mode"] == "configured_run"
    exported = invoke("export-trajectories", path)
    rows = list(read_trajectories(Path(exported["report_directory"]) / "trajectories.jsonl"))
    assert {r["alpha_t"] for r in rows if r["advertiser_id"] == role} == {40}
    bench = invoke("benchmark", path)
    report = json.loads((Path(bench["report_directory"]) / "benchmark.json").read_text())
    status = json.loads((Path(bench["report_directory"]) / "status.json").read_text())
    assert report["aggregate"]["samples_per_function"] == 2 * config["rules"]["total_slots"]
    assert report["tested_strategy"]["specification"] == config["benchmark"]["agent"]
    assert report["config_fingerprint"] == status["config_fingerprint"]
    assert report["run_id"] == status["run_id"]


def test_invalid_yaml_creates_new_failed_attempt_without_old_score(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("rules: [unterminated", encoding="utf-8")
    output = tmp_path / "outputs"
    failed = subprocess.run([sys.executable, "-m", "harmony_bidding.cli", "evaluate-single", "--config", str(path),
                             "--output", str(output)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert failed.returncode == 1
    error = json.loads(failed.stderr)
    assert error["status"] == "failed" and error["valid"] is False
    assert error["config_fingerprint"] is None
    assert Path(error["report_directory"]).name == error["run_id"]
    assert json.loads((Path(error["report_directory"]) / "status.json").read_text())["run_id"] == error["run_id"]


@pytest.mark.parametrize("day_id", ["..", "/outside", "sub/name", "C:\\escape", "C:relative", "sub\\name"])
def test_cli_rejects_day_identity_as_report_path(tmp_path, day_id):
    config = load_config(ROOT / "configs/tiny.yaml")
    config.update(data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "reports"))
    config["synthetic"]["days"] = [day_id]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    invoke("generate-demo", path)
    failed = subprocess.run([sys.executable, "-m", "harmony_bidding.cli", "run", "--config", str(path)],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert failed.returncode == 1
    error = json.loads(failed.stderr)
    assert "safe single directory name" in error["message"]
    assert error["status"] == "failed" and error["valid"] is False
    directory = Path(error["report_directory"])
    assert directory.parent == tmp_path / "reports/run"
    assert not list((tmp_path / "reports").rglob("episode.json"))
