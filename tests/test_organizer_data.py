"""Hand-written six-column adapter cases, independent of synthetic generation."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from harmony_bidding.data import load_days
from harmony_bidding.organizer_data import SOURCE_FIELDS, import_organizer_csv
from harmony_bidding.schema import Rules


def write_csv(path, rows, fields=SOURCE_FIELDS):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    return path


def config(tmp_path, rows=None):
    if rows is None:
        rows = [(99, 0, "z", "001", "app", 10), (0, 0, "z", "002", "shop", 8),
                (1, 0, "a", "002", "shop", 4), (0, 0, "a", "001", "app", 3)]
    source = write_csv(tmp_path / "source.csv", rows)
    return {
        "data_dir": str(tmp_path / "imported"), "rules": {"total_slots": 3}, "expected_n": 2,
        "organizer_import": {
            "sources": [{"path": str(source), "day_id": "2026-10-01"}],
            "time_step_base": 0, "request_id_scope": "day", "ordering": {"mode": "file_first_seen"},
            "players": {"001": {"player_type": "app", "budget": 100},
                        "002": {"player_type": "shop", "budget": 80}}, "source_kind": "schema_fixture",
        },
    }


def read_days(conf):
    return list(load_days(conf["data_dir"], Rules(**conf["rules"]), expected_n=conf.get("expected_n")))


def assert_import_fails(conf, pattern):
    with pytest.raises(ValueError, match=pattern):
        import_organizer_csv(conf)
    assert not (Path(conf["data_dir"]) / "manifest.json").exists()


def test_first_seen_preserves_ids_ignores_index_and_writes_provenance(tmp_path):
    conf = config(tmp_path)
    manifest = import_organizer_csv(conf)
    day, = read_days(conf)
    assert [a.advertiser_id for a in day.advertisers] == ["001", "002"]
    assert [a.budget for a in day.advertisers] == [100, 80]
    assert [o.opportunity_id for o in day.slots[0]] == ["z", "a"]
    assert [o.original_order for o in day.slots[0]] == [0, 1]
    assert [o.values for o in day.slots[0]] == [(10., 8.), (3., 4.)]
    assert day.slots[1:] == ((), ())
    metadata = manifest["import_metadata"]
    assert metadata["row_count"] == 4 and metadata["request_count"] == 2
    assert metadata["missing_pvalue_policy"] == "error"
    assert metadata["missing_record_policy"] == "absent_participant"
    assert metadata["invalid_existing_value_policy"] == "error"
    assert [o.participation_mask for o in day.slots[0]] == [(True, True), (True, True)]
    assert metadata["source_schema_version"] == "organizer-six-fields-v1"
    assert metadata["source_kind"] == "schema_fixture"
    assert metadata["ordering"]["is_local_assumption"] is True
    assert "Local assumption" in metadata["ordering_assumption"]
    assert metadata["rules_version"] == Rules().rules_version
    assert metadata["assumption_profile"] == Rules().assumption_profile
    source = Path(conf["organizer_import"]["sources"][0]["path"])
    assert metadata["source_files"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(metadata["config_fingerprint"]) == 64
    assert manifest == json.loads((Path(conf["data_dir"]) / "manifest.json").read_text(encoding="utf-8"))


def test_reversed_headers_extra_fields_and_leading_zero_identifiers(tmp_path):
    conf = config(tmp_path)
    source = Path(conf["organizer_import"]["sources"][0]["path"])
    write_csv(source, [("unused", 7, "shop", "002", "00004", "00", "000"),
                       ("unused", 8, "app", "001", "00004", "00", "000")],
              ("extra", "pvalue", "player_type", "player_id", "request_id", "time_step", "index"))
    import_organizer_csv(conf)
    day, = read_days(conf)
    assert day.slots[0][0].opportunity_id == "00004"
    assert day.slots[0][0].values == (8., 7.)


def test_one_based_time_conversion_is_explicit(tmp_path):
    conf = config(tmp_path, [(0, 1, "r", "001", "app", 1), (1, 1, "r", "002", "shop", 2),
                             (2, 3, "last", "001", "app", 3), (3, 3, "last", "002", "shop", 4)])
    conf["organizer_import"]["time_step_base"] = 1
    import_organizer_csv(conf)
    day, = read_days(conf)
    assert day.slots[0][0].opportunity_id == "r"
    assert day.slots[1] == ()
    assert day.slots[2][0].opportunity_id == "last"


def test_request_scope_time_step_encodes_without_collision(tmp_path):
    rows = [(0, 0, "001", "001", "app", 1), (1, 0, "001", "002", "shop", 2),
            (2, 1, "001", "001", "app", 3), (3, 1, "001", "002", "shop", 4),
            (4, 1, '[0,"001"]', "001", "app", 5), (5, 1, '[0,"001"]', "002", "shop", 6)]
    conf = config(tmp_path, rows)
    conf["organizer_import"]["request_id_scope"] = "time_step"
    import_organizer_csv(conf)
    day, = read_days(conf)
    ids = [o.opportunity_id for slot in day.slots for o in slot]
    assert len(set(ids)) == 3
    assert [json.loads(oid) for oid in ids] == [[0, "001"], [1, "001"], [1, '[0,"001"]']]


def test_day_scope_request_cannot_change_time_step(tmp_path):
    conf = config(tmp_path, [(0, 0, "r", "001", "app", 1), (1, 1, "r", "002", "shop", 2)])
    assert_import_fails(conf, "request time_step conflict")


def test_shards_share_identity_and_follow_explicit_file_sequence(tmp_path):
    conf = config(tmp_path, [(100, 0, "z", "001", "app", 1), (1, 0, "a", "002", "shop", 4)])
    later = write_csv(tmp_path / "later.csv", [(0, 0, "a", "001", "app", 3), (200, 0, "z", "002", "shop", 2)])
    conf["organizer_import"]["sources"].append({"path": str(later), "day_id": "2026-10-01"})
    manifest = import_organizer_csv(conf)
    day, = read_days(conf)
    assert [o.opportunity_id for o in day.slots[0]] == ["z", "a"]
    assert [o.values for o in day.slots[0]] == [(1., 2.), (3., 4.)]
    assert [source["request_count"] for source in manifest["import_metadata"]["source_files"]] == [2, 2]
    assert manifest["import_metadata"]["request_count"] == 2
    reversed_conf = copy.deepcopy(conf)
    reversed_conf["data_dir"] = str(tmp_path / "reversed")
    reversed_conf["organizer_import"]["sources"].reverse()
    import_organizer_csv(reversed_conf)
    reversed_day, = read_days(reversed_conf)
    assert [o.opportunity_id for o in reversed_day.slots[0]] == ["a", "z"]


@pytest.mark.parametrize("different_value", [False, True])
def test_duplicate_pair_across_shards_is_never_silently_deduplicated(tmp_path, different_value):
    conf = config(tmp_path)
    shard = write_csv(tmp_path / "duplicate.csv", [(0, 0, "z", "001", "app", 99 if different_value else 10)])
    conf["organizer_import"]["sources"].append({"path": str(shard), "day_id": "2026-10-01"})
    assert_import_fails(conf, "duplicate request/player pair")


def test_duplicate_pair_inside_source_is_rejected(tmp_path):
    conf = config(tmp_path, [(0, 0, "r", "001", "app", 0), (1, 0, "r", "001", "app", 0)])
    assert_import_fails(conf, "duplicate request/player pair")


@pytest.mark.parametrize("day_id", ["2026-10-01", "2026-10-02"])
def test_repeated_source_path_is_rejected_even_across_days(tmp_path, day_id):
    conf = config(tmp_path)
    source = conf["organizer_import"]["sources"][0]
    conf["organizer_import"]["sources"].append({"path": source["path"], "day_id": day_id})
    assert_import_fails(conf, "duplicate source path")


@pytest.mark.parametrize("bad", ["-1", "nan", "inf", "-inf", "-1e-1000", "1e-1000", "not-number"])
def test_pvalue_is_strictly_validated(tmp_path, bad):
    conf = config(tmp_path, [(0, 0, "r", "001", "app", bad), (1, 0, "r", "002", "shop", 1)])
    assert_import_fails(conf, "pvalue")


def test_absent_record_gets_zero_mask_false_but_true_zero_stays_present(tmp_path):
    conf = config(tmp_path, [(0, 0, "r", "001", "app", 0)])
    manifest = import_organizer_csv(conf)
    day, = read_days(conf)
    assert day.slots[0][0].values == (0., 0.)
    assert day.slots[0][0].participation_mask == (True, False)
    assert manifest["days"][0]["opportunity_counts_by_player"] == {"001": 1, "002": 0}
    with (Path(conf["data_dir"]) / "day_0000" / "values.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["advertiser_id"] == "001" and float(rows[0]["estimated_value"]) == 0


def test_existing_row_with_missing_pvalue_is_not_an_absence(tmp_path):
    conf = config(tmp_path, [(0, 0, "r", "001", "app", "")])
    assert_import_fails(conf, "missing required value")


def test_empty_day_keeps_explicit_roster_and_zero_budget(tmp_path):
    conf = config(tmp_path, [])
    conf["organizer_import"]["players"]["001"]["budget"] = 0
    manifest = import_organizer_csv(conf)
    day, = read_days(conf)
    assert len(day.advertisers) == 2 and day.advertisers[0].budget == 0
    assert day.slots == ((), (), ())
    assert manifest["import_metadata"]["row_count"] == manifest["import_metadata"]["request_count"] == 0


def test_multiple_dates_splits_budgets_and_same_request_id(tmp_path):
    conf = config(tmp_path)
    for day in (2, 3):
        path = write_csv(tmp_path / f"day{day}.csv", [(0, 0, "z", "001", "app", day),
                                                   (0, 0, "z", "002", "shop", day + 1)])
        conf["organizer_import"]["sources"].append({"path": str(path), "day_id": f"2026-10-0{day}"})
    conf["organizer_import"]["budgets_by_day"] = {"2026-10-02": {"001": "20.000000", "002": 30}}
    manifest = import_organizer_csv(conf)
    days = read_days(conf)
    assert [entry["split"] for entry in manifest["days"]] == ["train", "validation", "local_holdout"]
    assert [[a.budget for a in day.advertisers] for day in days] == [[100, 80], [20, 30], [100, 80]]
    assert [day.slots[0][0].opportunity_id for day in days] == ["z", "z", "z"]


def test_player_type_conflict_on_later_date_never_publishes_manifest(tmp_path):
    conf = config(tmp_path)
    later = write_csv(tmp_path / "later.csv", [(0, 0, "r", "001", "changed", 1), (1, 0, "r", "002", "shop", 2)])
    conf["organizer_import"]["sources"].append({"path": str(later), "day_id": "2026-10-02"})
    assert_import_fails(conf, "player_type conflict")
    assert (Path(conf["data_dir"]) / "day_0000" / "values.csv").exists()


@pytest.mark.parametrize("row,pattern", [
    ((0, 0, "r", "unknown", "app", 1), "unknown player_id"),
    ((0, 0, "r", "001", "different", 1), "player_type conflict"),
    ((-1, 0, "r", "001", "app", 1), "index"),
    (("1.5", 0, "r", "001", "app", 1), "index"),
    ((0, -1, "r", "001", "app", 1), "time_step"),
    ((0, 3, "r", "001", "app", 1), "time_step out of range"),
    ((0, "0.5", "r", "001", "app", 1), "time_step"),
    ((0, 0, "", "001", "app", 1), "missing required value"),
    ((0, 0, "r", " 001", "app", 1), "player_id"),
])
def test_invalid_rows_fail(tmp_path, row, pattern):
    conf = config(tmp_path, [row])
    assert_import_fails(conf, pattern)


@pytest.mark.parametrize("mutate,pattern", [
    (lambda o: o.pop("time_step_base"), "time_step_base"),
    (lambda o: o.update(time_step_base=True), "time_step_base"),
    (lambda o: o.pop("request_id_scope"), "request_id_scope"),
    (lambda o: o.pop("ordering"), "ordering"),
    (lambda o: o.update(ordering={"mode": "index"}), "ordering"),
    (lambda o: o.update(players={1: {"player_type": "app", "budget": 10}}), "player_id"),
    (lambda o: o["players"]["001"].update(player_type=3), "player_type"),
    (lambda o: o["players"]["001"].update(budget="1.00000000000000000001"), "precision"),
    (lambda o: o.update(budgets_by_day={"2026-10-01": {"001": 30}}), "cover exactly all"),
    (lambda o: o.update(budgets_by_day={"unknown": {"001": 30, "002": 40}}), "unknown dates"),
    (lambda o: o.update(split_by_day={}), "assign exactly every date"),
    (lambda o: o.update(split_by_day={"2026-10-01": "official_hidden"}), "assign exactly every date"),
])
def test_required_assumptions_and_roster_are_not_inferred(tmp_path, mutate, pattern):
    conf = config(tmp_path)
    mutate(conf["organizer_import"])
    assert_import_fails(conf, pattern)


def test_expected_n_must_match_configured_roster(tmp_path):
    conf = config(tmp_path)
    conf["expected_n"] = 3
    assert_import_fails(conf, "expected_n")


def test_explicit_order_overrides_file_order_and_is_hashed(tmp_path):
    conf = config(tmp_path)
    order_path = write_csv(tmp_path / "order.csv", [("2026-10-01", 0, "a", 2), ("2026-10-01", 0, "z", 8)],
                           ("day_id", "time_step", "request_id", "original_order"))
    conf["organizer_import"]["ordering"] = {"mode": "explicit_csv", "path": str(order_path)}
    manifest = import_organizer_csv(conf)
    day, = read_days(conf)
    assert [o.opportunity_id for o in day.slots[0]] == ["a", "z"]
    assert [o.original_order for o in day.slots[0]] == [2, 8]
    assert manifest["import_metadata"]["ordering"]["sha256"] == hashlib.sha256(order_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("rows,pattern", [
    ([("2026-10-01", 0, "z", 0)], "missing requests"),
    ([("2026-10-01", 0, "z", 0), ("2026-10-01", 0, "a", 1), ("2026-10-01", 0, "unknown", 2)], "unknown requests"),
    ([("2026-10-01", 0, "z", 0), ("2026-10-01", 0, "a", 0)], "duplicate original_order"),
    ([("2026-10-01", 0, "z", 0), ("2026-10-01", 0, "z", 1)], "duplicate explicit order request"),
    ([("unknown-day", 0, "z", 0)], "unknown day_id"),
    ([("2026-10-01", 1, "z", 0), ("2026-10-01", 0, "a", 1)], "time_step conflict"),
    ([("2026-10-01", 0, "z", -1)], "original_order"),
])
def test_explicit_order_requires_complete_unique_valid_mapping(tmp_path, rows, pattern):
    conf = config(tmp_path)
    order_path = write_csv(tmp_path / "order.csv", rows, ("day_id", "time_step", "request_id", "original_order"))
    conf["organizer_import"]["ordering"] = {"mode": "explicit_csv", "path": str(order_path)}
    assert_import_fails(conf, pattern)


def test_explicit_order_uses_same_one_based_time_and_time_scope(tmp_path):
    conf = config(tmp_path, [(0, 1, "r", "001", "app", 1), (1, 1, "r", "002", "shop", 2),
                             (2, 2, "r", "001", "app", 3), (3, 2, "r", "002", "shop", 4)])
    order_path = write_csv(tmp_path / "order.csv", [("2026-10-01", 2, "r", 4), ("2026-10-01", 1, "r", 8)],
                           ("day_id", "time_step", "request_id", "original_order"))
    conf["organizer_import"].update(time_step_base=1, request_id_scope="time_step", ordering={"mode": "explicit_csv", "path": str(order_path)})
    import_organizer_csv(conf)
    day, = read_days(conf)
    assert day.slots[0][0].original_order == 8
    assert day.slots[1][0].original_order == 4


@pytest.mark.parametrize("fields,rows,pattern", [
    (("index", "time_step", "request_id", "player_id", "pvalue"), [(0, 0, "r", "001", 1)], "missing required columns"),
    (SOURCE_FIELDS + ("index",), [(0, 0, "r", "001", "app", 1, 2)], "duplicate CSV header"),
    (SOURCE_FIELDS, [(0, 0, "r", "001", "app")], "missing or excess columns"),
    (SOURCE_FIELDS, [(0, 0, "r", "001", "app", 1, "extra")], "missing or excess columns"),
])
def test_bad_csv_schema_is_rejected(tmp_path, fields, rows, pattern):
    conf = config(tmp_path)
    write_csv(Path(conf["organizer_import"]["sources"][0]["path"]), rows, fields)
    assert_import_fails(conf, pattern)


def test_refuses_overwrite_and_repeat_is_reproducible_elsewhere(tmp_path):
    conf = config(tmp_path)
    first = import_organizer_csv(conf)
    before = (Path(conf["data_dir"]) / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        import_organizer_csv(conf)
    assert (Path(conf["data_dir"]) / "manifest.json").read_bytes() == before
    second_conf = copy.deepcopy(conf)
    second_conf["data_dir"] = str(tmp_path / "second")
    second = import_organizer_csv(second_conf)
    assert second == first
    assert read_days(conf) == read_days(second_conf)


def test_config_caller_object_is_not_mutated(tmp_path):
    conf = config(tmp_path)
    saved = copy.deepcopy(conf)
    import_organizer_csv(conf)
    assert conf == saved


def test_source_changed_during_import_does_not_publish_manifest(tmp_path, monkeypatch):
    from harmony_bidding import organizer_data

    conf = config(tmp_path)
    calls = 0
    real_sha256 = organizer_data._sha256

    def changed_digest(path):
        nonlocal calls
        calls += 1
        return real_sha256(path) if calls == 1 else "changed"

    monkeypatch.setattr(organizer_data, "_sha256", changed_digest)
    assert_import_fails(conf, "source file changed during import")
