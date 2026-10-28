import pytest
from harmony_bidding.benchmark import timing_summary, percentile99


def test_paired_p99_is_not_separate_function_allowance():
    rows = [{"bidding_ms": 120, "update_ms": 100} for _ in range(1000)]
    summary = timing_summary(rows)
    assert summary["p99_paired_sum_ms"] == 220
    assert summary["sum_of_p99_ms"] == 220
    assert not summary["local_paired_under_200ms"]
    assert summary["sufficient_for_local_diagnostic"]
    assert not timing_summary(rows[:10])["sufficient_for_local_diagnostic"]
    assert percentile99(range(1, 101)) == 99
    with pytest.raises(ValueError):
        percentile99([])


def test_benchmark_runs_selected_agent_and_excludes_opponents_and_warmup(monkeypatch, tmp_path):
    import json
    from pathlib import Path
    import harmony_bidding.runner as runner
    from harmony_bidding.agents import ConstantAlphaAgent
    from harmony_bidding.benchmark import run_benchmark
    from harmony_bidding.schema import Advertiser, DayData
    calls = {"init": 0, "bid": 0, "update": 0}
    class Spy(ConstantAlphaAgent):
        def __init__(self, alpha):
            super().__init__(alpha)
            calls["init"] += 1
        def bidding(self, observation):
            calls["bid"] += 1
            return super().bidding(observation)
        def update(self, feedback):
            calls["update"] += 1
            return super().update(feedback)
    original = runner.make_agent
    monkeypatch.setattr(runner, "make_agent", lambda spec: Spy(spec["alpha"]) if spec["kind"] == "constant" else original(spec))
    day = DayData("2026-01-01", (Advertiser("A", "x", 1), Advertiser("B", "x", 1)), ((), (), ()))
    config = {"rules": {"total_slots": 3}, "benchmark": {
        "target_role": "A", "agent": {"kind": "constant", "alpha": 100},
        "warmup_episodes": 1, "episodes": 2, "min_samples": 1000}}
    report = run_benchmark(day, config, tmp_path)
    assert calls == {"init": 3, "bid": 9, "update": 9}
    assert report["aggregate"]["samples_per_function"] == 6
    assert report["initialization_and_reset"]["samples"] == 2
    assert report["tested_strategy"]["class"].endswith("Spy")
    assert set(report["by_advertiser"]) == {"A"}
    assert report["memory"]["scope"] == "process_total_rss"
    assert report["memory"]["single_agent_rss_bytes"] is None
    status = json.loads((Path(report["report_directory"]) / "status.json").read_text())
    assert status["config_fingerprint"] == report["config_fingerprint"]
    assert status["run_id"] == report["run_id"]


def test_benchmark_requires_actual_target_and_records_unavailable_official(tmp_path):
    import json
    from harmony_bidding.benchmark import run_benchmark
    from harmony_bidding.schema import Advertiser, DayData
    day = DayData("d", (Advertiser("A", "x", 1),), ((),))
    for options in ({}, {"target_role": "A", "agent": {"kind": "official"}, "episodes": 1, "warmup_episodes": 0}):
        with pytest.raises(Exception):
            run_benchmark(day, {"rules": {"total_slots": 1}, "benchmark": options}, tmp_path)
    directories = list(tmp_path.iterdir())
    assert len(directories) == 2
    assert all(json.loads((p / "status.json").read_text())["status"] == "failed" for p in directories)
    assert all(not (p / "benchmark.json").exists() for p in directories)


def test_completed_reader_requires_valid_benchmark_report(tmp_path):
    import json
    from pathlib import Path
    from harmony_bidding.benchmark import run_benchmark
    from harmony_bidding.runner import read_completed_run
    from harmony_bidding.schema import Advertiser, DayData
    day = DayData("d", (Advertiser("A", "x", 1),), ((),))
    config = {"rules": {"total_slots": 1}, "benchmark": {
        "target_role": "A", "agent": {"kind": "constant", "alpha": 1}, "warmup_episodes": 0, "episodes": 1}}
    report = run_benchmark(day, config, tmp_path)
    directory = Path(report["report_directory"])
    assert read_completed_run(directory)["valid"] is True
    report["valid"] = False
    (directory / "benchmark.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="report is invalid"):
        read_completed_run(directory)
    (directory / "benchmark.json").unlink()
    with pytest.raises(ValueError, match="missing required report"):
        read_completed_run(directory)
