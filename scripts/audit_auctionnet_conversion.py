"""Independently compare bounded converted samples with original native CSVs.

Every source byte is SHA256 hashed. Only source rows selected by sample.index
are CSV-parsed; other binary lines only advance the original row index. This
audits samples, not every converted value. Mapping uses declared source-slot
counts and independently reconstructed request ordinals from original indices.
No source, extracted data, converter checkpoint or metadata is modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from time import perf_counter


SOURCE_HEADER = ("deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex", "budget", "CPAConstraint",
                 "timeStepIndex", "remainingBudget", "pvIndex", "pValue", "pValueSigma", "bid", "xi", "adSlot",
                 "cost", "isExposed", "conversionAction", "leastWinningCost", "isEnd")
SAMPLE_HEADER = ("index", "time_step", "request_id", "player_id", "player_type", "pvalue")
ORDER_HEADER = ("day_id", "time_step", "request_id", "original_order")
MAX_SAMPLE_ROWS = 100_000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integral(token):
    _require(isinstance(token, str) and re.fullmatch(r"[0-9]+(?:\.0+)?", token) is not None,
             f"invalid exact integral source token: {token!r}")
    return int(token.partition(".")[0])


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_ino, value.st_ctime_ns


def _mapped(slot, rank, count):
    # Midpoint at (2*rank+1)/(2*count), rescaled by 72/48 = 3/2.
    return (3 * (2 * slot * count + 2 * rank + 1)) // (4 * count)


def _lower_rank(slot, count, target):
    lower, upper = 0, count
    while lower < upper:
        middle = (lower + upper) // 2
        if _mapped(slot, middle, count) < target:
            lower = middle + 1
        else:
            upper = middle
    return lower


def _original_order(slot, rank, target, counts):
    before = 0
    for earlier in range(slot):
        count = counts[earlier]
        before += _lower_rank(earlier, count, target + 1) - _lower_rank(earlier, count, target)
    return before + rank - _lower_rank(slot, counts[slot], target)


def _read_bounded_csv(path, header):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        _require(reader.fieldnames == list(header), f"unexpected audit CSV header: {path}")
        for index, row in enumerate(reader):
            _require(index < MAX_SAMPLE_ROWS, f"sample exceeds audit row cap {MAX_SAMPLE_ROWS}")
            _require(None not in row and all(value is not None for value in row.values()), f"malformed sample row: {path}")
            yield row


def audit_period(conversion_root, extraction_manifest_path, period):
    started = perf_counter()
    directory = Path(conversion_root).resolve() / f"period-{period}"
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_digest = _hash(metadata_path)
    _require(metadata["source_period"] == period and metadata["day_id"] == f"period-{period}", "period metadata identity mismatch")
    players = metadata["parameters"]["expected_players"]
    _require(type(players) is int and 1 <= players <= 10000, "invalid expected player count")
    counts = metadata["source48counts"]
    _require(len(counts) == 48 and all(type(n) is int and n >= 0 for n in counts), "invalid source48counts")
    _require(sum(counts) == metadata["request_count"], "source slot counts differ from request count")
    _require(metadata["row_count"] == players * metadata["request_count"], "source row/roster count mismatch")
    source = Path(metadata["source"]["path"]).resolve()
    extraction_path = Path(extraction_manifest_path).resolve()
    extraction_digest = _hash(extraction_path)
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    relative_source = source.relative_to(extraction_path.parent).as_posix()
    extraction_records = []
    for archive in extraction["archives"].values():
        for member in archive["members"]:
            if member["path"].casefold() == relative_source.casefold() and not member["directory"]:
                _require(archive["status"] in ("verified", "verified_and_deleted"), "source archive lacks a completed verification status")
                extraction_records.append(member)
    _require(bool(extraction_records), "source missing from verified extraction manifest")
    source_sha = metadata["source"]["sha256"]
    source_bytes = metadata["source"]["bytes"]
    _require(all(record["sha256"] == source_sha and record["size"] == source_bytes for record in extraction_records),
             "converter/extraction manifests disagree on original source SHA256 or size")
    samples, groups, orders = {}, {}, {}
    for key in ("sample", "sample_request_order"):
        record = metadata["files"][key]
        expected_name = "sample.csv" if key == "sample" else "sample_request_order.csv"
        _require(record["path"] == expected_name, "unexpected sample output path")
        path = directory / expected_name
        _require(path.stat().st_size == record["bytes"] and _hash(path) == record["sha256"], f"sample hash/size mismatch: {path}")
    previous_index = -1
    for row in _read_bounded_csv(directory / "sample.csv", SAMPLE_HEADER):
        index = _integral(row["index"])
        _require(index > previous_index, "sample indices must be strictly increasing and unique")
        _require(index < metadata["row_count"], "sample index exceeds source row count")
        previous_index = index
        samples[index] = row
        groups.setdefault(row["request_id"], []).append(index)
    _require(len(samples) == metadata["sample_row_count"] and len(groups) == metadata["sample_count"], "sample metadata counts mismatch")
    _require(bool(samples), "source audit requires a nonempty bounded sample")
    previous_ordinal = -1
    for request_id, indices in groups.items():
        ordinal = indices[0] // players
        _require(ordinal > previous_ordinal, "sample request order differs from original request order")
        previous_ordinal = ordinal
        _require(indices == list(range(ordinal * players, (ordinal + 1) * players)), f"incomplete/noncontiguous sampled request: {request_id}")
        _require({_integral(samples[index]["player_id"]) for index in indices} == set(range(players)),
                 f"duplicate/missing sampled advertiser: {request_id}")
    for row in _read_bounded_csv(directory / "sample_request_order.csv", ORDER_HEADER):
        _require(row["day_id"] == metadata["day_id"] and row["request_id"] not in orders, "invalid/duplicate sampled order row")
        orders[row["request_id"]] = row
    _require(list(orders) == list(groups), "sample request order file differs from sampled source order")
    prefix = [0]
    for count in counts:
        prefix.append(prefix[-1] + count)
    before = _snapshot(source)
    digest, bytes_read, matched, source_rows = sha256(), 0, 0, 0
    next_progress = 1024 * 1024 * 1024
    print(f"Audit period {period}: stream/hash {source_bytes:,} bytes; parse {len(samples):,} sampled rows", flush=True)
    with source.open("rb", buffering=4 * 1024 * 1024) as handle:
        header = handle.readline()
        digest.update(header)
        bytes_read += len(header)
        parsed_header = next(csv.reader([header.decode("utf-8-sig")], strict=True))
        _require(parsed_header == list(SOURCE_HEADER), "original source header mismatch")
        for index, binary_line in enumerate(handle):
            digest.update(binary_line)
            bytes_read += len(binary_line)
            source_rows += 1
            sample = samples.get(index)
            if sample is not None:
                original = next(csv.reader([binary_line.decode("utf-8")], strict=True))
                _require(len(original) == len(SOURCE_HEADER), f"source physical row {index} is not one complete CSV record")
                original_period, player, category = map(_integral, original[:3])
                slot, pv = _integral(original[5]), _integral(original[7])
                _require(original_period == period and 0 <= slot < 48, f"source period/slot mismatch at row {index}")
                expected_id = f"period-{period}_slot_{slot}_pv_{pv}"
                _require(sample["request_id"] == expected_id, f"request identity mismatch at row {index}")
                _require(sample["player_id"] == str(player) and sample["player_type"] == str(category), f"player/type mismatch at row {index}")
                _require(sample["pvalue"] == original[8], f"pValue literal mismatch at row {index}")
                ordinal = index // players
                rank = ordinal - prefix[slot]
                _require(0 <= rank < counts[slot], f"source global ordinal/slot count mismatch at row {index}")
                mapped = _mapped(slot, rank, counts[slot])
                _require(_integral(sample["time_step"]) == mapped, f"independent 48-to-72 mapping mismatch at row {index}")
                order = orders[expected_id]
                _require(_integral(order["time_step"]) == mapped and _integral(order["original_order"]) == _original_order(slot, rank, mapped, counts),
                         f"independent original_order mismatch at row {index}")
                matched += 1
            if bytes_read >= next_progress:
                print(f"  period {period}: hashed {bytes_read:,}/{source_bytes:,} bytes; matched {matched:,} sample rows", flush=True)
                next_progress += 1024 * 1024 * 1024
    _require(_snapshot(source) == before, "source changed during audit")
    _require(digest.hexdigest() == source_sha, "full original source SHA256 mismatch")
    _require(bytes_read == source_bytes and source_rows == metadata["row_count"], "source byte or physical row count mismatch")
    _require(matched == len(samples), "some sampled indices were not found in source")
    _require(_hash(metadata_path) == metadata_digest and _hash(extraction_path) == extraction_digest, "metadata/extraction manifest changed during audit")
    result = {"period": period, "valid": True, "source_path": str(source), "source_bytes_hashed": bytes_read,
              "source_rows_streamed": source_rows, "source_sha256": digest.hexdigest(), "extraction_sha256_matches": True,
              "metadata_source_sha256_matches": True, "sample_rows_checked": matched, "sample_requests_checked": len(groups),
              "players_per_request": players, "source_fields_equal": True, "pvalue_literals_equal": True,
              "composite_request_ids_equal": True, "request_order_equal": True, "complete_request_rosters": True,
              "independent_integer_time_mapping_equal": True, "sample_file_sha256": metadata["files"]["sample"]["sha256"],
              "metadata_sha256": metadata_digest, "extraction_manifest_sha256": extraction_digest,
              "elapsed_seconds": perf_counter() - started}
    print(f"Audit period {period}: PASS ({matched:,} sample rows; {source_rows:,} original rows hashed)", flush=True)
    return result


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-root", default="data/auctionnet_72")
    parser.add_argument("--extraction-manifest", default="data/auto-bidding/extraction_manifest.json")
    parser.add_argument("--periods", type=int, nargs="+", default=[7, 21, 24])
    parser.add_argument("--output", default="outputs/native_sample_source_audit.json")
    args = parser.parse_args(argv)
    report = {"audit_version": "native-sample-source-audit-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "valid": False, "periods": [], "scope": "All original bytes hashed; only bounded sample rows CSV-parsed and compared.",
              "mapping_inputs": "Original index//N request ordinal and converter source48counts; integer midpoint 48-to-72 mapping independently recomputed."}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _require(len(set(args.periods)) == len(args.periods), "duplicate audit period selection")
        for period in args.periods:
            report["periods"].append(audit_period(args.conversion_root, args.extraction_manifest, period))
        report["valid"] = True
        report["total_source_bytes_hashed"] = sum(row["source_bytes_hashed"] for row in report["periods"])
        report["total_sample_rows_checked"] = sum(row["sample_rows_checked"] for row in report["periods"])
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"valid": report["valid"], "output": str(path), "sample_rows_checked": report["total_sample_rows_checked"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
