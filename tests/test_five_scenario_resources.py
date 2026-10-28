"""Resource comparison checks only; no auction or worker pool is started."""
from copy import deepcopy

import pandas as pd
import pytest

from harmony_baselines.five_scenarios import common as c, resources as r


def rows():
    result = []
    for player in r.ROLES:
        row = {name: f"{name}:{player}" for name in r.IDENTITY_FIELDS}
        row.update(player_id=player, total_value=1. + int(player) / 100,
                   total_cost=float(int(player)), wins=int(player))
        result.append(row)
    return pd.DataFrame(result)


def test_exact_pairing_is_order_independent_and_excludes_timing():
    original = rows()
    other = original.iloc[::-1].copy()
    original["bidding_p99_ms"] = 10.
    other["bidding_p99_ms"] = 20.
    assert r._assert_equal_results(original, other)["matched_episodes"] == 48


@pytest.mark.parametrize("field", r.BUSINESS_FIELDS + ("assignment_fingerprint", "dataset_fingerprint", "model_fingerprint"))
def test_metric_and_identity_changes_reject(field):
    original = rows()
    changed = original.copy()
    changed.loc[0, field] = changed.loc[0, field] + (1e-14 if field != "wins" else 1) if field in r.BUSINESS_FIELDS else "changed"
    with pytest.raises(ValueError, match=field):
        r._assert_equal_results(original, changed)


def test_missing_role_and_nonfinite_metrics_reject():
    original = rows()
    with pytest.raises(ValueError, match="48 roles"):
        r._assert_equal_results(original, original.iloc[:-1])
    changed = original.copy()
    changed.loc[0, "total_value"] = float("nan")
    with pytest.raises(ValueError, match="Illegal"):
        r._assert_equal_results(original, changed)


def timing_files(directory):
    c.dump(directory / "runtime_segment_000.json", {"seconds": 200., "resumed": 0, "new_completed": 48, "failures": 0})
    c.dump(directory / "episode_results.status.json", {"status": "completed", "valid": True, "workers": 4,
        "resumed_completed": 0, "completed_episodes": 48, "expected_episodes": 48, "failed_episodes": 0, "skipped_episodes": 0})


def test_only_clean_full_stage_timing_is_comparable(tmp_path):
    timing_files(tmp_path)
    summary = r._runtime_summary(tmp_path, 4)
    assert summary["seconds"] == 200. and summary["episodes_per_second"] == .24
    assert summary["startup_included"]
    segment = c.read(tmp_path / "runtime_segment_000.json")
    c.dump(tmp_path / "runtime_segment_001.json", segment)
    with pytest.raises(ValueError, match="uninterrupted"):
        r._runtime_summary(tmp_path, 4)


@pytest.mark.parametrize("field,value", [("resumed", 12), ("new_completed", 36), ("failures", 1), ("seconds", -1.)])
def test_resumed_failed_partial_or_invalid_timing_rejects(tmp_path, field, value):
    timing_files(tmp_path)
    segment = c.read(tmp_path / "runtime_segment_000.json")
    segment[field] = value
    c.dump(tmp_path / "runtime_segment_000.json", segment)
    with pytest.raises(ValueError, match="uninterrupted"):
        r._runtime_summary(tmp_path, 4)


def test_recommendation_uses_only_speed_after_exact_match_and_preserves_protocol(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(c, "OUT", out)
    c.dump(out / "protocol.json", {"workers": 4})
    protocol = (out / "protocol.json").read_bytes()
    plan = {"measurement": "startup included", "order_caveat": "single ordered trial",
            "experiment_protocol_sha256": "protocol"}
    monkeypatch.setattr(r, "make_plan", lambda: ([{"task": i} for i in range(48)], plan))
    calls = []
    def execute(tasks, directory, phase, *, workers, fail_group):
        calls.append(workers)
        directory.mkdir(parents=True)
        return rows(), [], []
    monkeypatch.setattr(r.execution, "execute", execute)
    monkeypatch.setattr(r, "_read_completed_group", lambda directory, tasks, workers, phase:
        (rows(), {"workers": workers, "seconds": 100. if workers == 4 else 80.}))
    selection = r.benchmark()
    assert calls == [4, 8] and selection["recommended_workers"] == 8
    assert selection["speedup_workers_8_over_4"] == 1.25
    assert not selection["test_used"] and not selection["included_in_training_data"]
    assert c.read(out / "resource_benchmark/operational_adjustment.json")["status"] == "recommended_not_applied"
    assert (out / "protocol.json").read_bytes() == protocol
    assert r.benchmark() == selection and calls == [4, 8]
