"""Complete-matrix I/O cache integrity and presence semantics."""
import csv
import json

import numpy as np
import pytest

from harmony_baselines.data import build_cache, load_day, sha256_file


def source_fixture(tmp_path, broken_player=False):
    root = tmp_path / "source"
    period = root / "period-1"
    period.mkdir(parents=True)
    with (period / "six_fields.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "time_step", "request_id", "player_id", "player_type", "pvalue"])
        writer.writerows([[0, 0, "r0", 0, 0, 0], [1, 0, "r0", 0 if broken_player else 1, 1, .125],
                          [2, 71, "r1", 0, 0, .2], [3, 71, "r1", 1, 1, .3]])
    with (period / "request_order.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["day_id", "time_step", "request_id", "original_order"])
        writer.writerows([["period-1", 0, "r0", 7], ["period-1", 71, "r1", 2]])
    files = {name: {"path": name+".csv", "bytes": (period/(name+".csv")).stat().st_size,
                    "sha256": sha256_file(period/(name+".csv"))} for name in ("six_fields", "request_order")}
    entry = {"day_id": "period-1", "split": "train", "files": files, "request_count": 2, "row_count": 4,
             "players": {"0": {"player_type": "0", "budget": 100}, "1": {"player_type": "1", "budget": 200}},
             "target72counts": [1]+[0]*70+[1], "rules_version": "source-v1", "converter_version": "fixture",
             "mapping_assumption": "test fixture only"}
    (period/"metadata.json").write_text(json.dumps(entry), encoding="utf-8")
    (root/"manifest.json").write_text(json.dumps({"periods": [entry]}), encoding="utf-8")
    return root


def test_complete_cache_preserves_values_zero_presence_and_order(tmp_path):
    root = source_fixture(tmp_path)
    day = build_cache("period-1", tmp_path/"cache", root)
    assert len(day.slots) == 72
    assert len(day.slots[1].values) == 0
    assert day.slots[0].values.tolist() == [[0, .125]]
    assert day.slots[0].participation_mask.tolist() == [[True, True]]
    assert day.slots[0].original_order.tolist() == [7]
    assert not day.slots[0].values.flags.writeable
    scaled = load_day("period-1", tmp_path/"cache", root, budget_multiplier=.5)
    assert scaled.advertisers[0].budget == 50
    assert scaled.fingerprint != day.fingerprint


def test_cache_rejects_content_mutation_with_same_file_size(tmp_path):
    root = source_fixture(tmp_path)
    build_cache("period-1", tmp_path/"cache", root)
    values = np.load(tmp_path/"cache"/"period-1"/"values.npy", mmap_mode="r+")
    values[0, 0] = .75
    values.flush()
    with pytest.raises(ValueError, match="integrity"):
        load_day("period-1", tmp_path/"cache", root)


def test_cache_rejects_missing_duplicate_player_coverage(tmp_path):
    root = source_fixture(tmp_path, broken_player=True)
    with pytest.raises(ValueError, match="coverage"):
        build_cache("period-1", tmp_path/"cache", root)


def test_source_digest_must_match_before_cache_build(tmp_path):
    root = source_fixture(tmp_path)
    with (root/"period-1"/"six_fields.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="source integrity"):
        build_cache("period-1", tmp_path/"cache", root)
