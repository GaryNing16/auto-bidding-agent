"""Independent native-conversion checks using tiny, handcrafted 18-field CSVs."""
import csv
import json
import os
from pathlib import Path

import pytest

from harmony_bidding import auctionnet_data as native


def request(slot, pv, *, period=7, budget="100.0", categories=("0.0", "1.0")):
    rows = []
    for player in range(2):
        row = ["0"] * len(native.HEADER)
        values = {"deliveryPeriodIndex": f"{period}.0", "advertiserNumber": f"{player}.0",
                  "advertiserCategoryIndex": categories[player], "budget": budget, "CPAConstraint": "999.0",
                  "timeStepIndex": f"{slot}.0", "remainingBudget": budget, "pvIndex": f"{pv}.0",
                  "pValue": "1.2500e-2" if player == 0 else "0.250000",
                  "pValueSigma": "999.0", "bid": "999.0", "xi": "0", "adSlot": "0",
                  "cost": "999.0", "isExposed": "1", "conversionAction": "1", "leastWinningCost": "999.0", "isEnd": "0"}
        for index, field in enumerate(native.HEADER):
            row[index] = values[field]
        rows.append(row)
    return rows


def source(root, rows, period=7):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"period-{period}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(native.HEADER)
        writer.writerows(rows)
    return path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_integer_midpoint_mapping_preserves_every_row_and_large_request_ids(tmp_path):
    # Independent arithmetic: slot 0 has midpoints .25/.75 -> 0/1;
    # slot 1 -> 1/2; slot 47 -> 70/71. Target slot 1 contains two
    # requests from adjacent original slots, in original stream order.
    groups = [(0, 9007199254740993), (0, 1), (1, 20), (1, 2), (47, 9), (47, 3)]
    rows = [row for slot, pv in groups for row in request(slot, pv)]
    input_path = source(tmp_path / "source", rows)
    metadata = native.convert_period(input_path, tmp_path / "output", expected_players=2, sample_requests_per_slot=1)
    directory = tmp_path / "output" / "period-7"
    full = read_csv(directory / "six_fields.csv")
    orders = read_csv(directory / "request_order.csv")
    sample = read_csv(directory / "sample.csv")
    assert metadata["request_count"] == 6 and metadata["row_count"] == 12
    assert len(full) == 12 and len(orders) == 6
    assert [int(row["time_step"]) for row in orders] == [0, 1, 1, 2, 70, 71]
    assert [int(row["original_order"]) for row in orders] == [0, 0, 1, 0, 0, 0]
    assert [row["request_id"] for row in orders] == [f"period-7_slot_{slot}_pv_{pv}" for slot, pv in groups]
    assert [row["pvalue"] for row in full] == ["1.2500e-2", "0.250000"] * 6
    assert [int(row["index"]) for row in full] == list(range(12))
    assert metadata["sample_count"] == 5 and len(sample) == metadata["sample_row_count"] == 10
    assert all(sum(row["request_id"] == request_id for row in sample) == 2 for request_id in {row["request_id"] for row in sample})
    assert sum(metadata["target72counts"]) == 6 and len(metadata["target72counts"]) == 72
    assert "CPAConstraint" in metadata["source_ignored_fields"]
    assert "conversionAction" in metadata["source_ignored_fields"]


@pytest.mark.parametrize("case", ["duplicate_player", "missing_middle", "missing_final", "repeated_request", "backward_slot"])
def test_malformed_grouping_never_silently_drops_rows_or_publishes_output(tmp_path, case):
    groups = {
        "duplicate_player": request(0, 10) + [request(0, 10)[0]],
        "missing_middle": [request(0, 10)[0]] + request(0, 11),
        "missing_final": request(0, 10) + [request(0, 11)[0]],
        "repeated_request": request(0, 10) + request(0, 11) + request(0, 10),
        "backward_slot": request(1, 10) + request(0, 11),
    }
    input_path = source(tmp_path / "source", groups[case])
    with pytest.raises(ValueError, match="duplicate|missing|nondecreasing"):
        native.convert_period(input_path, tmp_path / "output", expected_players=2)
    period = tmp_path / "output" / "period-7"
    assert not (period / "metadata.json").exists()
    assert not (period / "six_fields.csv").exists()
    assert input_path.exists()


def test_native_pv_index_resets_at_each_original_slot(tmp_path):
    # Tiny analogue of observed period-7 boundaries: source line 600770 starts
    # slot 1 at pv 0 after slot 0 ends at pv 12515; line 881714 starts slot 2
    # at pv 0. These rows are hand-authored, not copied full source requests.
    groups = [(0, 12515), (1, 0), (2, 0)]
    input_path = source(tmp_path / "source", [row for slot, pv in groups for row in request(slot, pv)])
    metadata = native.convert_period(input_path, tmp_path / "output", expected_players=2)
    orders = read_csv(tmp_path / "output" / "period-7" / "request_order.csv")
    assert metadata["request_count"] == 3 and metadata["row_count"] == 6
    assert [row["request_id"] for row in orders] == [
        "period-7_slot_0_pv_12515", "period-7_slot_1_pv_0", "period-7_slot_2_pv_0"]
    assert [int(row["time_step"]) for row in orders] == [0, 2, 3]
    assert "source_request_scope" in metadata


def test_same_pv_in_adjacent_source_slots_remains_distinct_in_shared_target_slot(tmp_path):
    # Request order is first-seen, never the numeric pvIndex order. These two
    # source-slot pv 0 requests both map to target slot 1 and must not collide.
    groups = [(0, 7), (0, 0), (1, 0), (1, 1)]
    input_path = source(tmp_path / "source", [row for slot, pv in groups for row in request(slot, pv)])
    metadata = native.convert_period(input_path, tmp_path / "output", expected_players=2)
    directory = tmp_path / "output" / "period-7"
    orders = read_csv(directory / "request_order.csv")
    same_target = [row for row in orders if row["time_step"] == "1"]
    assert metadata["request_count"] == 4 and len(read_csv(directory / "six_fields.csv")) == 8
    assert [row["request_id"] for row in same_target] == ["period-7_slot_0_pv_0", "period-7_slot_1_pv_0"]
    assert [int(row["original_order"]) for row in same_target] == [0, 1]


@pytest.mark.parametrize("field,new_value", [("budget", "101.0"), ("advertiserCategoryIndex", "9.0")])
def test_type_and_budget_must_be_fixed_inside_one_period(tmp_path, field, new_value):
    rows = request(0, 1) + request(1, 2)
    rows[-2][native.HEADER.index(field)] = new_value
    input_path = source(tmp_path / "source", rows)
    with pytest.raises(ValueError, match="changes type or budget"):
        native.convert_period(input_path, tmp_path / "output", expected_players=2)


def test_staged_checkpoint_corruption_cannot_publish_on_resume(tmp_path, monkeypatch):
    input_path = source(tmp_path / "source", request(0, 10))
    output = tmp_path / "output"
    with monkeypatch.context() as patch:
        def interrupted(*args, **kwargs):
            raise RuntimeError("fixture interruption after staging")
        patch.setattr(native, "_emit", interrupted)
        with pytest.raises(RuntimeError, match="interruption"):
            native.convert_period(input_path, output, expected_players=2)
    checkpoint = json.loads((output / "period-7" / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["stage"] == "staged"
    staged = output / "period-7" / "staging.csv.part"
    staged.write_bytes(staged.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="staging SHA256/size mismatch"):
        native.convert_period(input_path, output, expected_players=2)
    assert not (output / "period-7" / "metadata.json").exists()
    assert not (output / "period-7" / "six_fields.csv").exists()


def test_resume_detects_source_changes_even_with_preserved_size_and_mtime(tmp_path):
    input_path = source(tmp_path / "source", request(0, 10))
    output = tmp_path / "output"
    native.convert_period(input_path, output, expected_players=2)
    original_stat = input_path.stat()
    raw = input_path.read_bytes()
    altered = raw.replace(b"1.2500e-2", b"1.3500e-2")
    assert len(raw) == len(altered) and raw != altered
    input_path.write_bytes(altered)
    os.utime(input_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(ValueError, match="source SHA256 changed"):
        native.convert_period(input_path, output, expected_players=2)


def test_successful_resume_checks_outputs_and_rejects_parameter_change(tmp_path):
    input_path = source(tmp_path / "source", request(0, 10))
    output = tmp_path / "output"
    first = native.convert_period(input_path, output, expected_players=2, sample_requests_per_slot=1)
    assert native.convert_period(input_path, output, expected_players=2, sample_requests_per_slot=1) == first
    with pytest.raises(ValueError, match="parameters"):
        native.convert_period(input_path, output, expected_players=2, sample_requests_per_slot=2)
    target = output / "period-7" / "six_fields.csv"
    raw = target.read_bytes()
    target.write_bytes(b"X" + raw[1:])
    with pytest.raises(ValueError, match="output SHA256 mismatch"):
        native.convert_period(input_path, output, expected_players=2, sample_requests_per_slot=1)


def test_cross_period_budgets_are_preserved_individually_and_types_are_consistent(tmp_path):
    output = tmp_path / "output"
    first = source(tmp_path / "source", request(0, 10, period=7, budget="100.0"), period=7)
    second = source(tmp_path / "source", request(0, 10, period=8, budget="250.0"), period=8)
    native.convert_period(first, output, expected_players=2)
    native.convert_period(second, output, expected_players=2)
    manifest = native.aggregate_manifest(output, expected_periods=[7, 8])
    assert [record["players"]["0"]["budget"] for record in manifest["periods"]] == [100, 250]
    assert manifest["request_count"] == 2 and manifest["row_count"] == 4
    mismatch = source(tmp_path / "source", request(0, 10, period=9, categories=("9.0", "1.0")), period=9)
    native.convert_period(mismatch, output, expected_players=2)
    with pytest.raises(ValueError, match="roster/type differs"):
        native.aggregate_manifest(output, expected_periods=[7, 8, 9])
