"""Native conversion expectations use hand-written requests, never its mapper."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from harmony_bidding import auctionnet_data as native


def source(tmp_path, requests=None, *, period=7, players=3, budget="100.0", category=None):
    path = tmp_path / f"period-{period}.csv"
    requests = [(0, 91, "0.0100"), (0, 2, "0.0200"), (1, 8, "0.0300"), (47, 9, "0.0400")] if requests is None else requests
    rows = []
    for slot, pv, value in requests:
        for player in range(players):
            rows.append([f"{period}.0", f"{player}.0", f"{player % 2 if category is None else category}.0", budget,
                         "20.0", f"{slot}.0", budget, f"{pv}.0", value, "0.01", "1", "0", "1", "0", "1", "0", "0", "0"])
    write_rows(path, rows)
    return path


def write_rows(path, rows, header=native.HEADER):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mutate(path, callback):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    callback(rows)
    write_rows(path, rows[1:], rows[0])


def convert(path, root, **kwargs):
    return native.convert_period(path, root, expected_players=3, **kwargs)


def test_hand_checked_mapping_literals_and_native_metadata(tmp_path):
    path = source(tmp_path)
    root = tmp_path / "out"
    metadata = convert(path, root)
    directory = root / "period-7"
    rows = read_rows(directory / "six_fields.csv")
    order = read_rows(directory / "request_order.csv")
    assert list(rows[0]) == list(native.SIX_HEADER)
    # Two slot-0 request midpoints land at .375 and 1.125 => slots 0 and 1.
    # One slot-1 midpoint lands at 2.25, one slot-47 midpoint at 71.25.
    assert [int(row["time_step"]) for row in order] == [0, 1, 2, 71]
    assert [row["request_id"] for row in order] == ["period-7_slot_0_pv_91", "period-7_slot_0_pv_2", "period-7_slot_1_pv_8", "period-7_slot_47_pv_9"]
    assert [row["pvalue"] for row in rows[::3]] == ["0.0100", "0.0200", "0.0300", "0.0400"]
    assert [int(row["index"]) for row in rows] == list(range(12))
    assert metadata["players"] == {"0": {"player_type": "0", "budget": 100.0}, "1": {"player_type": "1", "budget": 100.0}, "2": {"player_type": "0", "budget": 100.0}}
    assert metadata["row_count"] == 12 and metadata["request_count"] == 4
    assert metadata["sample_count"] == 4 and metadata["sample_row_count"] == 12
    assert metadata["source48counts"][:2] == [2, 1]
    assert len(metadata["target72counts"]) == 72
    assert metadata["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["pvalue_diagnostics"]["above1_count"] == 0
    for record in metadata["files"].values():
        output = directory / record["path"]
        assert record["bytes"] == output.stat().st_size
        assert record["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert not (directory / "staging.csv.part").exists()
    assert not (directory / "requests.sqlite.part").exists()
    assert json.loads((directory / "checkpoint.json").read_text())["stage"] == "complete"


def test_default_48_roster_and_bounded_complete_sample(tmp_path):
    path = source(tmp_path, [(0, pv, "0.1") for pv in range(30)], players=48)
    metadata = native.convert_period(path, tmp_path / "out", sample_requests_per_slot=4)
    rows = read_rows(tmp_path / "out/period-7/sample.csv")
    assert metadata["row_count"] == 30 * 48
    assert metadata["sample_count"] == 8  # original slot0 spans mapped slots0 and1
    assert len(rows) == 8 * 48
    by_request = {}
    for row in rows:
        by_request.setdefault(row["request_id"], set()).add(row["player_id"])
    assert all(players == {str(index) for index in range(48)} for players in by_request.values())
    assert metadata["sample72counts"][:2] == [4, 4]


def test_zero_sample_limit(tmp_path):
    metadata = convert(source(tmp_path), tmp_path / "out", sample_requests_per_slot=0)
    assert metadata["sample_count"] == metadata["sample_row_count"] == 0
    assert read_rows(tmp_path / "out/period-7/sample.csv") == []


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf", "junk", "-1e-999", "1e-999"])
def test_invalid_pvalue_rejected_without_published_metadata(tmp_path, value):
    path = source(tmp_path, [(0, 0, value)])
    with pytest.raises(ValueError, match="pValue"):
        convert(path, tmp_path / "out")
    assert not (tmp_path / "out/period-7/metadata.json").exists()


def test_source_above_one_is_preserved_and_diagnosed(tmp_path):
    metadata = convert(source(tmp_path, [(0, 0, "1.200")]), tmp_path / "out")
    assert metadata["pvalue_diagnostics"]["min"] == metadata["pvalue_diagnostics"]["max"] == 1.2
    assert metadata["pvalue_diagnostics"]["above1_count"] == 3
    assert read_rows(tmp_path / "out/period-7/six_fields.csv")[0]["pvalue"] == "1.200"


@pytest.mark.parametrize("column,value,match", [(0, "8.0", "deliveryPeriodIndex"), (1, "3.0", "advertiserNumber"),
    (1, "1.5", "integral"), (2, "-1", "integral"), (3, "-1", "budget"), (3, "0.0000001", "precision"),
    (5, "48", "timeStepIndex"), (7, "1.5", "integral"), (8, "", "missing")])
def test_invalid_fields(tmp_path, column, value, match):
    path = source(tmp_path)
    mutate(path, lambda rows: rows[1].__setitem__(column, value))
    with pytest.raises(ValueError, match=match):
        convert(path, tmp_path / "out")


def test_missing_roster(tmp_path):
    path = source(tmp_path)
    mutate(path, lambda rows: rows.pop())
    with pytest.raises(ValueError, match="missing advertisers"):
        convert(path, tmp_path / "out")


def test_duplicate_pair(tmp_path):
    path = source(tmp_path)
    mutate(path, lambda rows: rows.insert(2, list(rows[1])))
    with pytest.raises(ValueError, match="duplicate request/advertiser"):
        convert(path, tmp_path / "out")


def test_duplicate_request_within_source_slot(tmp_path):
    with pytest.raises(ValueError, match="duplicate/noncontiguous"):
        convert(source(tmp_path, [(0, 1, "0.1"), (0, 2, "0.1"), (0, 1, "0.1")]), tmp_path / "out")


def test_release_resets_pv_index_each_source_slot(tmp_path):
    requests = [(0, 0, "0.1"), (0, 1, "0.2"), (1, 0, "0.3"), (1, 1, "0.4"), (2, 0, "0.5")]
    metadata = convert(source(tmp_path, requests), tmp_path / "out")
    order = read_rows(tmp_path / "out/period-7/request_order.csv")
    assert [row["request_id"] for row in order] == ["period-7_slot_0_pv_0", "period-7_slot_0_pv_1", "period-7_slot_1_pv_0", "period-7_slot_1_pv_1", "period-7_slot_2_pv_0"]
    assert [int(row["time_step"]) for row in order] == [0, 1, 1, 2, 3]
    assert [int(row["original_order"]) for row in order] == [0, 0, 1, 0, 0]
    assert len({row["request_id"] for row in order}) == 5
    assert metadata["request_count"] == 5
    assert "timeStepIndex" in metadata["source_request_scope"]


def test_slot_order_never_silently_sorted(tmp_path):
    with pytest.raises(ValueError, match="nondecreasing"):
        convert(source(tmp_path, [(1, 1, "0.1"), (0, 2, "0.1")]), tmp_path / "out")


@pytest.mark.parametrize("column,value", [(2, "99.0"), (3, "101.0")])
def test_same_player_type_and_budget_fixed_within_period(tmp_path, column, value):
    path = source(tmp_path)
    mutate(path, lambda rows: rows[4].__setitem__(column, value))
    with pytest.raises(ValueError, match="changes type or budget"):
        convert(path, tmp_path / "out")


def test_exact_header_and_nonempty_required(tmp_path):
    path = source(tmp_path)
    mutate(path, lambda rows: rows[0].__setitem__(8, "pvalue"))
    with pytest.raises(ValueError, match="header"):
        convert(path, tmp_path / "out")
    empty = source(tmp_path, [], period=8)
    with pytest.raises(ValueError, match="no complete"):
        convert(empty, tmp_path / "empty")


@pytest.mark.parametrize("column", native.REQUIRED_SOURCE_INDICES)
def test_every_mapped_source_field_still_required(tmp_path, column):
    path = source(tmp_path)
    mutate(path, lambda rows: rows[1].__setitem__(column, ""))
    with pytest.raises(ValueError, match=f"missing required field {native.HEADER[column]}"):
        convert(path, tmp_path / "out")
    assert not (tmp_path / "out/period-7/metadata.json").exists()


@pytest.mark.parametrize("period,slot,player", [(9, 46, 33), (13, 47, 1)])
def test_real_release_empty_historical_bid_keeps_complete_request(tmp_path, period, slot, player):
    # Real failure shapes: period9 slot46 pv0 advertiser33, and period13
    # slot47 pv0 advertiser1. Only the unused historical bid is absent.
    path = source(tmp_path, [(slot, 0, "0.0005963842885026619")], period=period, players=48)
    mutate(path, lambda rows: rows[1 + player].__setitem__(10, ""))
    metadata = native.convert_period(path, tmp_path / "out")
    directory = tmp_path / "out" / f"period-{period}"
    rows = read_rows(directory / "six_fields.csv")
    assert len(rows) == metadata["row_count"] == 48
    assert metadata["request_count"] == metadata["sample_count"] == 1
    assert len({row["player_id"] for row in rows}) == 48
    assert all(row["pvalue"] == "0.0005963842885026619" for row in rows)
    assert metadata["ignored_missing_field_counts"]["bid"] == 1
    assert sum(metadata["ignored_missing_field_counts"].values()) == 1
    checkpoint = json.loads((directory / "checkpoint.json").read_text())
    assert checkpoint["ignored_missing_field_counts"] == metadata["ignored_missing_field_counts"]


def test_all_unused_history_fields_can_be_empty_without_changing_six_fields(tmp_path):
    path = source(tmp_path)
    original = convert(path, tmp_path / "before")

    def clear_ignored(rows):
        for row in rows[1:]:
            for index in native.IGNORED_SOURCE_INDICES:
                row[index] = ""

    mutate(path, clear_ignored)
    changed = convert(path, tmp_path / "after")
    assert changed["row_count"] == original["row_count"] == 12
    assert changed["files"] == original["files"]
    assert changed["source"]["sha256"] != original["source"]["sha256"]
    assert changed["ignored_missing_field_counts"] == {native.HEADER[index]: 12 for index in native.IGNORED_SOURCE_INDICES}
    manifest = native.aggregate_manifest(tmp_path / "after")
    assert manifest["ignored_missing_field_counts"] == changed["ignored_missing_field_counts"]


@pytest.mark.parametrize("row_change", [lambda row: row.pop(), lambda row: row.append("")])
def test_history_blanks_do_not_relax_eighteen_column_width(tmp_path, row_change):
    path = source(tmp_path)
    mutate(path, lambda rows: row_change(rows[1]))
    with pytest.raises(ValueError, match="missing/excess fields"):
        convert(path, tmp_path / "out")


def test_legacy_completed_metadata_is_reused_without_rewriting(tmp_path):
    path, root = source(tmp_path), tmp_path / "out"
    convert(path, root)
    directory = root / "period-7"
    metadata_path, checkpoint_path = directory / "metadata.json", directory / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())
    for key in ("ignored_missing_field_counts", "ignored_missing_field_policy"):
        metadata.pop(key, None)
        checkpoint.pop(key, None)
    checkpoint["metadata"] = metadata
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    before = metadata_path.read_bytes(), checkpoint_path.read_bytes()
    assert convert(path, root) == metadata
    assert (metadata_path.read_bytes(), checkpoint_path.read_bytes()) == before
    manifest = native.aggregate_manifest(root)
    assert all(count == 0 for count in manifest["ignored_missing_field_counts"].values())
    assert "ignored_missing_field_counts" not in manifest["periods"][0]
    assert (metadata_path.read_bytes(), checkpoint_path.read_bytes()) == before


def test_resume_completed_verifies_hash_and_parameters(tmp_path):
    path = source(tmp_path)
    root = tmp_path / "out"
    first = convert(path, root)
    assert convert(path, root) == first
    with pytest.raises(FileExistsError, match="resume"):
        convert(path, root, resume=False)
    with pytest.raises(ValueError, match="parameters"):
        convert(path, root, sample_requests_per_slot=1)
    output = root / "period-7/sample.csv"
    output.write_bytes(output.read_bytes().replace(b"0.0100", b"0.0101", 1))
    with pytest.raises(ValueError, match="SHA256"):
        convert(path, root)


def test_resume_checks_source_hash_even_same_size_and_mtime(tmp_path):
    path = source(tmp_path)
    root = tmp_path / "out"
    convert(path, root)
    stat = path.stat()
    path.write_bytes(path.read_bytes().replace(b"0.0100", b"0.0101", 1))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="source SHA256"):
        convert(path, root)


def test_first_pass_interruption_restarts_owned_parts(tmp_path, monkeypatch):
    path = source(tmp_path)
    original = native._pvalue
    calls = 0

    def interrupted(token):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("simulated interruption")
        return original(token)

    monkeypatch.setattr(native, "_pvalue", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        convert(path, tmp_path / "out")
    monkeypatch.setattr(native, "_pvalue", original)
    assert convert(path, tmp_path / "out")["row_count"] == 12


def test_resume_valid_staging_skips_native_parse(tmp_path, monkeypatch):
    path, root = source(tmp_path), tmp_path / "out"
    original = native._emit

    def interrupted(*args):
        raise RuntimeError("interrupted before emit")

    monkeypatch.setattr(native, "_emit", interrupted)
    with pytest.raises(RuntimeError):
        convert(path, root)
    assert json.loads((root / "period-7/checkpoint.json").read_text())["stage"] == "staged"
    monkeypatch.setattr(native, "_emit", original)
    monkeypatch.setattr(native, "_stage", interrupted)
    assert convert(path, root)["row_count"] == 12


def test_resume_publication_can_finish_after_some_files_renamed(tmp_path, monkeypatch):
    path, root = source(tmp_path), tmp_path / "out"
    original = native._publish

    def interrupted(directory, checkpoint):
        (directory / "six_fields.csv.part").rename(directory / "six_fields.csv")
        raise RuntimeError("interrupted midway publication")

    monkeypatch.setattr(native, "_publish", interrupted)
    with pytest.raises(RuntimeError):
        convert(path, root)
    assert not (root / "period-7/metadata.json").exists()
    monkeypatch.setattr(native, "_publish", original)
    assert convert(path, root)["row_count"] == 12


def test_tampered_staging_not_resumed(tmp_path, monkeypatch):
    path, root = source(tmp_path), tmp_path / "out"
    original = native._emit

    def interrupted(*args):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(native, "_emit", interrupted)
    with pytest.raises(RuntimeError):
        convert(path, root)
    stage = root / "period-7/staging.csv.part"
    stage.write_bytes(stage.read_bytes().replace(b"0.0100", b"0.0101", 1))
    monkeypatch.setattr(native, "_emit", original)
    with pytest.raises(ValueError, match="staging SHA256"):
        convert(path, root)


def test_refuse_unrelated_outputs(tmp_path):
    path, root = source(tmp_path), tmp_path / "out"
    directory = root / "period-7"
    directory.mkdir(parents=True)
    user_file = directory / "sample.csv"
    user_file.write_text("user data")
    with pytest.raises(FileExistsError, match="unrecognized"):
        convert(path, root)
    assert user_file.read_text() == "user data"


def test_aggregate_splits_and_per_period_budgets(tmp_path):
    root = tmp_path / "out"
    for period, budget in [(7, "100"), (21, "200"), (24, "300")]:
        convert(source(tmp_path, period=period, budget=budget), root)
    manifest = native.aggregate_manifest(root, expected_periods=[7, 21, 24])
    assert [item["split"] for item in manifest["periods"]] == ["train", "validation", "local_holdout"]
    assert [item["players"]["0"]["budget"] for item in manifest["periods"]] == [100, 200, 300]
    assert manifest["row_count"] == 36 and manifest["request_count"] == 12
    with pytest.raises(ValueError, match="periods mismatch"):
        native.aggregate_manifest(root, expected_periods=[7, 21])


def test_aggregate_rejects_changed_type(tmp_path):
    root = tmp_path / "out"
    convert(source(tmp_path, period=7, category=0), root)
    convert(source(tmp_path, period=8, category=1), root)
    with pytest.raises(ValueError, match="roster/type"):
        native.aggregate_manifest(root)
    assert not (root / "manifest.json").exists()


def test_directory_selection_and_duplicate_sources(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source(source_dir, period=7)
    source(source_dir, period=8)
    result = native.convert_directory(source_dir, tmp_path / "out", periods=[8], expected_players=3)
    assert [record["source_period"] for record in result["periods"]] == [8]
    nested = source_dir / "nested"
    nested.mkdir()
    source(nested, period=8)
    with pytest.raises(ValueError, match="ambiguous"):
        native.convert_directory(source_dir, tmp_path / "out", periods=[8], expected_players=3)


def test_large_integer_request_id_is_exact(tmp_path):
    large = 9007199254740993
    convert(source(tmp_path, [(0, large, "0.1")]), tmp_path / "out")
    assert read_rows(tmp_path / "out/period-7/request_order.csv")[0]["request_id"] == f"period-7_slot_0_pv_{large}"


def test_sample_roundtrip_through_existing_six_field_importer(tmp_path):
    from harmony_bidding.data import load_days
    from harmony_bidding.organizer_data import import_organizer_csv
    from harmony_bidding.schema import Rules

    root = tmp_path / "out"
    metadata = convert(source(tmp_path), root)
    directory = root / "period-7"
    config = {"data_dir": str(tmp_path / "internal"), "expected_n": 3, "rules": {"total_slots": 72},
              "organizer_import": {
                  "sources": [{"path": str(directory / metadata["files"]["sample"]["path"]), "day_id": "period-7"}],
                  "time_step_base": 0, "request_id_scope": "day", "source_kind": metadata["source_kind"],
                  "ordering": {"mode": "explicit_csv", "path": str(directory / metadata["files"]["sample_request_order"]["path"])},
                  "players": metadata["players"], "split_by_day": {"period-7": "train"}}}
    manifest = import_organizer_csv(config)
    day, = load_days(config["data_dir"], Rules(), expected_n=3)
    assert len(day.slots) == 72
    assert day.slots[0][0].opportunity_id == "period-7_slot_0_pv_91"
    assert day.slots[0][0].values == (0.01, 0.01, 0.01)
    assert sum(map(len, day.slots)) == metadata["sample_count"] == 4
    assert manifest["import_metadata"]["source_kind"] == "auctionnet_native_generated_game_data"


def test_actual_script_two_workers_and_period_selection(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for period in (7, 8, 9):
        source(source_dir, period=period)
    script = Path(__file__).resolve().parents[1] / "scripts/convert_auctionnet_data.py"
    completed = subprocess.run([sys.executable, str(script), "--source-dir", str(source_dir),
        "--output-dir", str(tmp_path / "out"), "--periods", "7-8", "--workers", "2", "--expected-players", "3"],
        capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert [record["source_period"] for record in manifest["periods"]] == [7, 8]
    assert manifest["row_count"] == 24
    assert not (tmp_path / "out/period-9").exists()
