"""Actual CLI roundtrip of the user-described six-field source format."""
import json
from pathlib import Path
import subprocess
import sys
import yaml

from harmony_bidding.data import load_days
from harmony_bidding.schema import Rules, load_config


ROOT = Path(__file__).resolve().parents[1]


def invoke(command, path, *arguments, expected_code=0):
    completed = subprocess.run(
        [sys.executable, "-m", "harmony_bidding.cli", command, "--config", str(path), *arguments],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert completed.returncode == expected_code, completed.stdout + completed.stderr
    return json.loads(completed.stdout if expected_code == 0 else completed.stderr)


def test_import_run_evaluate_export_all_use_six_field_source(tmp_path):
    config = load_config(ROOT / "configs/organizer_example.yaml")
    config["data_dir"] = str(tmp_path / "导入数据")
    config["output_dir"] = str(tmp_path / "输出")
    path = tmp_path / "赛题配置.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    imported = invoke("import-organizer", path)
    assert imported["days"] == 3
    manifest = json.loads((Path(config["data_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["import_metadata"]["time_step_base"] == 1
    assert manifest["import_metadata"]["request_id_scope"] == "day"
    assert manifest["import_metadata"]["source_kind"] == "schema_fixture"
    days = list(load_days(config["data_dir"], Rules(**config["rules"]), expected_n=3))
    assert len(days) == 3
    assert [o.opportunity_id for o in days[0].slots[0]] == ["r20", "r03"]
    assert [o.values for o in days[0].slots[0]] == [(10, 8, 11), (3, 2, 1)]

    run_summary = invoke("run", path)
    result = json.loads((Path(run_summary["report_directory"]) / "2026-10-20/episode.json").read_text(encoding="utf-8"))
    assert len(result["slots"]) == 216
    assert result["metadata"]["data_import"] == manifest["import_metadata"]
    assert result["checks"] == {"legal": True, "budget": True}
    assert all(s["bidding_calls"] == s["update_calls"] == 72 for s in result["summaries"].values())
    for command in ("evaluate-single", "evaluate-multi"):
        summary = invoke(command, path, "--split", "local_holdout")
        report = json.loads((Path(summary["report_directory"]) / "evaluation.json").read_text(encoding="utf-8"))
        assert report["valid"] is True
        assert all(g["day_id"] == "2026-10-22" for g in report["games"])
        assert len(report["games"]) == 9
        if command == "evaluate-single":
            assert report["teams"]["A"]["G"] == 1
        assert report["references"][0]["metadata"]["data_import"] == manifest["import_metadata"]
    exported = invoke("export-trajectories", path)
    assert exported["transitions"] == 216
    text = (Path(exported["report_directory"]) / "trajectories.jsonl").read_text(encoding="utf-8")
    for forbidden in ("pvalue", "request_id", "data_import", "source_files", "row_count"):
        assert forbidden not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert sum(row["done"] for row in rows) == 3
    assert all(len(row["observation_t"]["history"]) == row["observation_t"]["time_slot"] for row in rows)
    # A second import cannot destroy/replace the previously validated dataset.
    failed = invoke("import-organizer", path, expected_code=1)
    assert failed["status"] == "failed"
    assert json.loads((Path(config["data_dir"]) / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_import_requires_explicit_mapping_and_reports_error(tmp_path):
    config = load_config(ROOT / "configs/organizer_example.yaml")
    config["data_dir"] = str(tmp_path / "data")
    config["output_dir"] = str(tmp_path / "results")
    del config["organizer_import"]["time_step_base"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    error = invoke("import-organizer", path, expected_code=1)
    assert "time_step_base" in error["message"]
    assert not (Path(config["data_dir"]) / "manifest.json").exists()
    status = json.loads((Path(error["report_directory"]) / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"


def test_sparse_six_field_cli_and_large_alpha(tmp_path):
    source = tmp_path / "sparse.csv"
    source.write_text("index,time_step,request_id,player_id,player_type,pvalue\n"
                      "0,0,r,A,app,1\n1,0,r,B,shop,10\n2,0,next,A,app,0\n", encoding="utf-8")
    config = {"data_dir": str(tmp_path / "data"), "output_dir": str(tmp_path / "results"),
              "rules": {"total_slots": 72}, "expected_n": 2,
              "run": {"assignments": {"A": "high", "B": "one"},
                      "teams": {"high": {"kind": "constant", "alpha": 40}, "one": {"kind": "constant", "alpha": 1}}},
              "organizer_import": {"sources": [{"path": str(source), "day_id": "local-day"}],
                                   "time_step_base": 0, "request_id_scope": "day", "ordering": {"mode": "file_first_seen"},
                                   "players": {"A": {"player_type": "app", "budget": 20}, "B": {"player_type": "shop", "budget": 20}}}}
    path = tmp_path / "sparse.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    invoke("import-organizer", path)
    summary = invoke("run", path)
    result = json.loads((Path(summary["report_directory"]) / "local-day/episode.json").read_text(encoding="utf-8"))
    assert len(result["slots"]) == 144
    assert result["summaries"]["A"]["value"] == 1
    assert result["summaries"]["A"]["cost"] == 10
    assert result["slots"][0]["alpha"] == 40
    day, = load_days(config["data_dir"], Rules())
    assert day.slots[0][1].values == (0., 0.) and day.slots[0][1].participation_mask == (True, False)
