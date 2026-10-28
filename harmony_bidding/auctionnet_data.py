"""Bounded-memory native AuctionNet conversion; no downloads or auction replay.

Input requests must be contiguous groups and source slots nondecreasing. This
is checked, not guessed. Original request order and pValue literals are kept.
48->72 timing is an explicit local uniform-within-source-slot assumption.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3

from .data import _budget, _number
from .schema import Rules

VERSION = "auctionnet-six-fields-v1"
TOKEN_CACHE_LIMIT = 4096
HEADER = ("deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex", "budget", "CPAConstraint",
          "timeStepIndex", "remainingBudget", "pvIndex", "pValue", "pValueSigma", "bid", "xi", "adSlot",
          "cost", "isExposed", "conversionAction", "leastWinningCost", "isEnd")
REQUIRED_SOURCE_INDICES = (0, 1, 2, 3, 5, 7, 8)
IGNORED_SOURCE_INDICES = tuple(index for index in range(len(HEADER)) if index not in REQUIRED_SOURCE_INDICES)
SIX_HEADER = ("index", "time_step", "request_id", "player_id", "player_type", "pvalue")
ORDER_HEADER = ("day_id", "time_step", "request_id", "original_order")
STAGE_HEADER = ("source_slot", "request_rank", "pv", "player", "type", "pvalue")
FILES = {"six_fields": "six_fields.csv", "request_order": "request_order.csv", "sample": "sample.csv",
         "sample_request_order": "sample_request_order.csv"}
MAPPING = "floor((source_slot + (request_rank + 0.5) / source_slot_request_count) * 72 / 48)"
ASSUMPTION = ("LOCAL timing assumption: requests are uniformly positioned inside each original 1/48-day slot; "
              "map request midpoints into 72 slots. No original timestamps or official 72-slot timing are claimed. "
              "Keep first-seen request order, never sort by pvIndex; preserve source pValue and budgets. "
              "Historical bids, payments, exposure/conversion outcomes and CPA constraints are not replayed.")


class _HashReader(io.RawIOBase):
    def __init__(self, path):
        self.handle = Path(path).open("rb", buffering=0)
        self.digest = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.handle.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count

    def close(self):
        self.handle.close()
        super().close()


class _HashWriter(io.RawIOBase):
    def __init__(self, path):
        self.handle = Path(path).open("wb", buffering=0)
        self.digest = hashlib.sha256()

    def writable(self):
        return True

    def write(self, data):
        count = self.handle.write(data)
        self.digest.update(memoryview(data)[:count])
        return count

    def close(self):
        self.handle.close()
        super().close()


def _reader(path):
    raw = _HashReader(path)
    return io.TextIOWrapper(io.BufferedReader(raw, 1024 * 1024), encoding="utf-8-sig", newline=""), raw


def _writer(path):
    raw = _HashWriter(path)
    return io.TextIOWrapper(io.BufferedWriter(raw, 1024 * 1024), encoding="utf-8", newline=""), raw


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".part")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def _integer(token, label):
    # The release represents identifiers as e.g. 7.0, but they remain exact
    # integers. Never round through binary floats (large pvIndex is safe).
    integer, dot, fraction = token.partition(".")
    if not integer or not integer.isascii() or not integer.isdigit() or (dot and (not fraction or fraction.strip("0"))):
        raise ValueError(f"{label}: expected nonnegative integral token, got {token!r}")
    return int(integer)


def _pvalue(token):
    try:
        value = float(token)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"pValue: invalid numeric value {token!r}") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"pValue: expected finite nonnegative value, got {token!r}")
    if value == 0:
        # Rare exact/underflow zeros take the Decimal path, preserving the
        # strict negative/positive-underflow checks without Decimal per row.
        _number(token, "pValue")
    return value


def _split(period):
    if 7 <= period <= 20:
        return "train"
    if 21 <= period <= 23:
        return "validation"
    if 24 <= period <= 27:
        return "local_holdout"
    raise ValueError("native release period must be in 7..27")


@contextmanager
def _lock(directory):
    # OS locks release on process death, so crash recovery needs no unsafe PID
    # guessing or deletion of a possibly active worker's lock.
    with (directory / ".lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        import os
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _remove_work(directory):
    # Only exact converter-owned temporary names, never directories or source.
    for name in ("staging.csv.part", "requests.sqlite.part", "requests.sqlite.part-journal"):
        path = directory / name
        if path.parent.resolve() != directory.resolve():
            raise ValueError("temporary cleanup escaped the period output directory")
        if path.exists():
            path.unlink()


def _verify_files(directory, metadata, verify_hashes=True):
    for key, name in FILES.items():
        record = metadata["files"][key]
        if record["path"] != name:
            raise ValueError("metadata contains unexpected output path")
        path = directory / name
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"completed output size/missing-file mismatch: {path}")
        if verify_hashes and _sha(path) != record["sha256"]:
            raise ValueError(f"completed output SHA256 mismatch: {path}")


def _stage(source, directory, period, expected_players, checkpoint):
    _remove_work(directory)
    connection = sqlite3.connect(directory / "requests.sqlite.part")
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute("CREATE TABLE seen (source_slot INTEGER, pv TEXT, PRIMARY KEY(source_slot, pv)) WITHOUT ROWID")
        input_text, input_hash = _reader(source)
        output_text, output_hash = _writer(directory / "staging.csv.part")
        counts, players = [0] * 48, {}
        row_count, request_count = 0, 0
        pvalue_min, pvalue_max, pvalue_above_one = math.inf, 0.0, 0
        ignored_missing_counts = {HEADER[index]: 0 for index in IGNORED_SOURCE_INDICES}
        full_mask = (1 << expected_players) - 1
        current_key, current_mask, current_rank = None, 0, 0
        previous_slot = -1
        integer_cache = {}
        budget_cache = {}

        def cached_integer(token, label):
            value = integer_cache.get(token)
            if value is None:
                value = _integer(token, label)
                if len(integer_cache) < TOKEN_CACHE_LIMIT:
                    integer_cache[token] = value
            return value

        with input_text, output_text:
            reader = csv.reader(input_text, strict=True)
            if next(reader, None) != list(HEADER):
                raise ValueError("native CSV header does not match the 18-field AuctionNet release")
            writer = csv.writer(output_text, lineterminator="\n")
            writer.writerow(STAGE_HEADER)
            last_pv_token, pv = None, None
            for row in reader:
                if len(row) != len(HEADER):
                    raise ValueError(f"native CSV line {reader.line_num}: missing/excess fields")
                # The release can leave historical bids/results blank. They
                # are never replayed, so their absence cannot discard an
                # otherwise complete estimated-value request. Exact width and
                # all seven fields used in conversion remain mandatory.
                if "" in row:
                    for index in REQUIRED_SOURCE_INDICES:
                        if row[index] == "":
                            raise ValueError(f"native CSV line {reader.line_num}: missing required field {HEADER[index]}")
                    for index in IGNORED_SOURCE_INDICES:
                        if row[index] == "":
                            ignored_missing_counts[HEADER[index]] += 1
                if cached_integer(row[0], "deliveryPeriodIndex") != period:
                    raise ValueError("deliveryPeriodIndex does not match source filename")
                player = cached_integer(row[1], "advertiserNumber")
                if not 0 <= player < expected_players:
                    raise ValueError("advertiserNumber outside configured contiguous 0..N-1 roster")
                category = cached_integer(row[2], "advertiserCategoryIndex")
                raw_budget = row[3]
                budget = budget_cache.get(raw_budget)
                if budget is None:
                    budget = _budget(raw_budget, Rules(), "native budget")
                    if len(budget_cache) < TOKEN_CACHE_LIMIT:
                        budget_cache[raw_budget] = budget
                known = players.get(str(player))
                if known is None:
                    players[str(player)] = {"player_type": str(category), "budget": budget}
                elif known["player_type"] != str(category) or known["budget"] != budget:
                    raise ValueError(f"native player {player} changes type or budget within period")
                slot = cached_integer(row[5], "timeStepIndex")
                if not 0 <= slot < 48:
                    raise ValueError("native timeStepIndex must be in 0..47")
                if slot < previous_slot:
                    raise ValueError("source slots are not nondecreasing; conversion requires contiguous chronological slots")
                previous_slot = slot
                if row[7] != last_pv_token:
                    pv = str(_integer(row[7], "pvIndex"))
                    last_pv_token = row[7]
                key = slot, pv
                if key != current_key:
                    if current_key is not None and current_mask != full_mask:
                        raise ValueError(f"missing advertisers in request {current_key}; each request must be a contiguous complete roster")
                    try:
                        connection.execute("INSERT INTO seen VALUES (?, ?)", (slot, pv))
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"duplicate/noncontiguous source-slot/pvIndex request: {slot}/{pv}") from error
                    current_key, current_mask = key, 0
                    current_rank = counts[slot]
                    counts[slot] += 1
                    request_count += 1
                bit = 1 << player
                if current_mask & bit:
                    raise ValueError(f"duplicate request/advertiser pair: {pv}/{player}")
                current_mask |= bit
                value = _pvalue(row[8])
                pvalue_min = min(pvalue_min, value)
                pvalue_max = max(pvalue_max, value)
                pvalue_above_one += value > 1
                writer.writerow((slot, current_rank, pv, player, category, row[8]))
                row_count += 1
            if current_key is not None and current_mask != full_mask:
                raise ValueError("missing advertisers in final request")
            if not request_count or len(players) != expected_players:
                raise ValueError("native period has no complete request/roster")
        connection.commit()
        checkpoint.update(stage="staged", source_sha256=input_hash.digest.hexdigest(),
                          stage_sha256=output_hash.digest.hexdigest(), stage_bytes=(directory / "staging.csv.part").stat().st_size,
                          source48counts=counts, row_count=row_count, request_count=request_count, players=players,
                          ignored_missing_field_counts=ignored_missing_counts,
                          pvalue_diagnostics={"min": pvalue_min, "max": pvalue_max, "above1_count": pvalue_above_one,
                                              "policy": "Finite nonnegative literals preserved; >1 is a source probability-semantics anomaly, not clipped."})
        stat = source.stat()
        if stat.st_size != checkpoint["source_stat"]["bytes"] or stat.st_mtime_ns != checkpoint["source_stat"]["mtime_ns"]:
            raise ValueError("source changed during initial conversion pass")
        _json(directory / "checkpoint.json", checkpoint)
    finally:
        connection.close()


def _emit(directory, checkpoint):
    counts = checkpoint["source48counts"]
    target_counts, sample_counts = [0] * 72, [0] * 72
    sample_limit = checkpoint["parameters"]["sample_requests_per_slot"]
    sample_rows, row_index = 0, 0
    previous_key, sampled, mapped, request_id = None, False, 0, ""
    output_hashes = {}
    with ExitStack() as stack:
        writers = {}
        for key, name in FILES.items():
            text, raw = _writer(directory / (name + ".part"))
            stack.enter_context(text)
            output_hashes[key] = raw
            writers[key] = csv.writer(text, lineterminator="\n")
            writers[key].writerow(ORDER_HEADER if "order" in key else SIX_HEADER)
        text, input_hash = _reader(directory / "staging.csv.part")
        stack.enter_context(text)
        reader = csv.reader(text, strict=True)
        if next(reader, None) != list(STAGE_HEADER):
            raise ValueError("invalid staged header")
        for row in reader:
            if len(row) != 6:
                raise ValueError("invalid staged row")
            slot, rank = int(row[0]), int(row[1])
            key = slot, rank, row[2]
            if key != previous_key:
                count = counts[slot]
                if not 0 <= rank < count:
                    raise ValueError("staged request rank out of range")
                mapped = ((2 * slot * count + 2 * rank + 1) * 72) // (2 * count * 48)
                request_id = f"{checkpoint['day_id']}_slot_{slot}_pv_{row[2]}"
                order = target_counts[mapped]
                target_counts[mapped] += 1
                sampled = sample_counts[mapped] < sample_limit
                order_row = (checkpoint["day_id"], mapped, request_id, order)
                writers["request_order"].writerow(order_row)
                if sampled:
                    sample_counts[mapped] += 1
                    writers["sample_request_order"].writerow(order_row)
                previous_key = key
            out = (row_index, mapped, request_id, row[3], row[4], row[5])
            writers["six_fields"].writerow(out)
            if sampled:
                writers["sample"].writerow(out)
                sample_rows += 1
            row_index += 1
    if input_hash.digest.hexdigest() != checkpoint["stage_sha256"]:
        raise ValueError("staging SHA256 mismatch; no outputs published")
    if row_index != checkpoint["row_count"] or sum(target_counts) != checkpoint["request_count"]:
        raise ValueError("staging counts differ from validated native source")
    metadata = {"converter_version": VERSION, "day_id": checkpoint["day_id"], "source_period": checkpoint["source_period"],
                "split": _split(checkpoint["source_period"]), "source_kind": "auctionnet_native_generated_game_data",
                "rules_version": "harmony-A-B-local-v1", "assumption_profile": "local-auctionnet-48-to-72-v1",
                "source": {"path": checkpoint["source_path"], "sha256": checkpoint["source_sha256"], **checkpoint["source_stat"]},
                "parameters": checkpoint["parameters"], "players": checkpoint["players"],
                "pvalue_diagnostics": checkpoint["pvalue_diagnostics"],
                # Older validated stages rejected emptiness in all 18 fields;
                # an absent counter therefore means a guaranteed zero count.
                "ignored_missing_field_counts": checkpoint.get("ignored_missing_field_counts", {HEADER[index]: 0 for index in IGNORED_SOURCE_INDICES}),
                "ignored_missing_field_policy": "Blank historical fields not used in conversion are counted and ignored without filling values or discarding requests. All seven mapped source fields remain required; legacy completed metadata without these counters passed the prior all-18-fields nonempty check, so their counts are zero.",
                "row_count": row_index, "request_count": checkpoint["request_count"], "sample_count": sum(sample_counts),
                "sample_row_count": sample_rows, "source48counts": counts, "target72counts": target_counts,
                "sample72counts": sample_counts, "mapping_formula": MAPPING, "mapping_assumption": ASSUMPTION,
                "input_layout_requirement": "Contiguous complete request groups, nondecreasing source slot, pvIndex unique within each source slot; pvIndex may reset across slots.",
                "source_request_scope": "deliveryPeriodIndex + timeStepIndex + pvIndex; output request_id preserves all three components, including source slot after 48-to-72 remapping.",
                "value_semantics": "Original AuctionNet pValue numeric literal; generated estimated conversion probability, not observed click or conversion outcome.",
                "index_semantics": "New zero-based full-output row number, not request arrival time; bounded sample retains these row numbers.",
                "source_ignored_fields": [HEADER[index] for index in IGNORED_SOURCE_INDICES],
                "files": {key: {"path": name, "sha256": output_hashes[key].digest.hexdigest(),
                                "bytes": (directory / (name + ".part")).stat().st_size} for key, name in FILES.items()}}
    checkpoint.update(stage="publishing", metadata=metadata)
    _json(directory / "checkpoint.json", checkpoint)
    return metadata


def _publish(directory, checkpoint):
    metadata = checkpoint["metadata"]
    for key, name in FILES.items():
        final, pending = directory / name, directory / (name + ".part")
        expected = metadata["files"][key]
        if final.exists():
            if final.stat().st_size != expected["bytes"] or _sha(final) != expected["sha256"]:
                raise ValueError(f"refusing to overwrite unexpected existing output: {final}")
        else:
            if not pending.is_file() or pending.stat().st_size != expected["bytes"] or _sha(pending) != expected["sha256"]:
                raise ValueError(f"publishing output SHA256/size mismatch: {pending}")
            pending.rename(final)
    _json(directory / "metadata.json", metadata)
    checkpoint["stage"] = "complete"
    _json(directory / "checkpoint.json", checkpoint)
    _remove_work(directory)
    return metadata


def convert_period(source_csv, output_root, *, expected_players=48, sample_requests_per_slot=4, resume=True):
    """Convert one period. Safe for independent per-period worker processes.

    First-pass interruption restarts this period. Once staging validates, a
    restart checks source and stage hashes then resumes emission/publication.
    Completed output is reused only after every output and source hash verify.
    This function never rewrites a global manifest; call aggregate_manifest in
    the parent after workers finish. No original data/archive is removed here.
    """
    source = Path(source_csv).resolve()
    match = re.fullmatch(r"period-([0-9]+)\.csv", source.name, re.IGNORECASE)
    if not match:
        raise ValueError("source filename must be period-<7..27>.csv")
    period = int(match[1])
    _split(period)
    if type(expected_players) is not int or expected_players < 1 or expected_players > 10000:
        raise ValueError("expected_players must be an integer in 1..10000")
    if type(sample_requests_per_slot) is not int or sample_requests_per_slot < 0:
        raise ValueError("sample_requests_per_slot must be a nonnegative integer")
    stat = source.stat()
    parameters = {"expected_players": expected_players, "sample_requests_per_slot": sample_requests_per_slot}
    root = Path(output_root).resolve()
    directory = root / f"period-{period}"
    if directory.exists() and any(directory.iterdir()) and not ((directory / "checkpoint.json").is_file() or (directory / "metadata.json").is_file()):
        raise FileExistsError(f"refusing to overwrite an unrecognized output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    with _lock(directory):
        checkpoint_path, metadata_path = directory / "checkpoint.json", directory / "metadata.json"
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if not resume:
                raise FileExistsError("period output already exists; resume was disabled")
            if (checkpoint.get("converter_version") != VERSION or checkpoint.get("source_path") != str(source)
                    or checkpoint.get("source_period") != period or checkpoint.get("parameters") != parameters):
                raise ValueError("existing checkpoint does not match source/converter/parameters")
            if checkpoint["source_stat"] != {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
                raise ValueError("source size/mtime changed since checkpoint")
            if checkpoint.get("source_sha256") and _sha(source) != checkpoint["source_sha256"]:
                raise ValueError("source SHA256 changed since checkpoint")
        else:
            if metadata_path.exists():
                raise ValueError("completed metadata has no matching owned checkpoint")
            checkpoint = {"converter_version": VERSION, "stage": "started", "source_path": str(source),
                          "source_period": period, "day_id": f"period-{period}", "parameters": parameters,
                          "source_stat": {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}}
            _json(checkpoint_path, checkpoint)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if checkpoint.get("metadata") != metadata:
                raise ValueError("metadata differs from its publication checkpoint")
            _verify_files(directory, metadata)
            _remove_work(directory)
            return metadata
        stage = checkpoint["stage"]
        if stage == "started":
            _stage(source, directory, period, expected_players, checkpoint)
        elif stage == "staged":
            staged = directory / "staging.csv.part"
            if not staged.is_file() or staged.stat().st_size != checkpoint["stage_bytes"] or _sha(staged) != checkpoint["stage_sha256"]:
                raise ValueError("staging SHA256/size mismatch; refusing unsafe recovery")
        elif stage != "publishing":
            raise ValueError(f"unsupported/incomplete checkpoint stage: {stage}")
        if checkpoint["stage"] == "staged":
            _emit(directory, checkpoint)
        return _publish(directory, checkpoint)


def aggregate_manifest(output_root, expected_periods=None, *, verify_hashes=True):
    """Parent-only summary; validate output hashes and cross-period types."""
    root = Path(output_root).resolve()
    records = []
    known_players = None
    for path in sorted(root.glob("period-*/metadata.json"), key=lambda path: int(path.parent.name.split("-")[1])):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("converter_version") != VERSION or path.parent.name != metadata.get("day_id"):
            raise ValueError("invalid native period metadata")
        _verify_files(path.parent, metadata, verify_hashes)
        types = {player: spec["player_type"] for player, spec in metadata["players"].items()}
        if known_players is None:
            known_players = types
        elif known_players != types:
            raise ValueError("advertiser roster/type differs across native periods")
        records.append(metadata)
    if not records:
        raise ValueError("no complete converted periods found")
    actual = {record["source_period"] for record in records}
    if expected_periods is not None and actual != set(expected_periods):
        raise ValueError(f"converted periods mismatch: expected {sorted(expected_periods)}, actual {sorted(actual)}")
    manifest = {"converter_version": VERSION, "source_kind": "auctionnet_native_generated_game_data",
                "rules_version": "harmony-A-B-local-v1", "assumption_profile": "local-auctionnet-48-to-72-v1",
                "mapping_assumption": ASSUMPTION, "periods": records,
                "row_count": sum(record["row_count"] for record in records),
                "request_count": sum(record["request_count"] for record in records),
                "sample_count": sum(record["sample_count"] for record in records),
                "sample_row_count": sum(record["sample_row_count"] for record in records),
                "ignored_missing_field_counts": {HEADER[index]: sum(record.get("ignored_missing_field_counts", {}).get(HEADER[index], 0) for record in records) for index in IGNORED_SOURCE_INDICES},
                "legacy_missing_counter_policy": "A completed period lacking ignored_missing_field_counts passed the original all-18-fields nonempty check; its counts are zero. Existing period metadata is not modified.",
                "integrity": {"output_hashes_checked": verify_hashes, "cross_period_player_types_checked": True},
                "budget_policy": "Original budget retained separately in each period's players metadata; budgets may differ across periods.",
                "split_policy": "periods 7..20 train; 21..23 validation; 24..27 local_holdout"}
    existing = root / "manifest.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8")).get("converter_version") != VERSION:
        raise FileExistsError("refusing to replace an unrelated manifest")
    _json(existing, manifest)
    return manifest


def convert_directory(source_dir, output_root, periods=None, *, expected_players=48, sample_requests_per_slot=4, resume=True):
    """Sequential convenience wrapper; discover period files recursively."""
    selected = None if periods is None else set(periods)
    sources = {}
    for source in Path(source_dir).rglob("period-*.csv"):
        match = re.fullmatch(r"period-([0-9]+)\.csv", source.name)
        if match and (selected is None or int(match[1]) in selected):
            period = int(match[1])
            if period in sources:
                raise ValueError(f"ambiguous duplicate source file for period {period}")
            sources[period] = source
    if not sources or (selected is not None and selected != set(sources)):
        raise ValueError("no sources or some requested periods are missing")
    for period in sorted(sources):
        convert_period(sources[period], output_root, expected_players=expected_players,
                       sample_requests_per_slot=sample_requests_per_slot, resume=resume)
    return aggregate_manifest(output_root)
