"""Local, evaluator-only CSV data and deterministic synthetic fixtures.

The interchange format is ours, not an organizer-provided schema.  No object
from this module is passed to an Agent.  Dates are partitioned on disk so the
loader holds at most one day's value matrix in memory.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Iterator

from .schema import Advertiser, DayData, Opportunity, Rules, nonnegative, validate_values_mask


SCHEMA_VERSION = "harmony-local-csv-v2"
LEGACY_SCHEMA_VERSION = "harmony-local-csv-v1"
SPLITS = ("train", "validation", "local_holdout")
FIELDS = {
    "advertisers.csv": ("day_id", "advertiser_id", "advertiser_type", "budget"),
    "opportunities.csv": ("day_id", "time_slot", "opportunity_id", "original_order"),
    "values.csv": ("day_id", "opportunity_id", "advertiser_id", "estimated_value"),
}


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label}: expected nonempty string without edge whitespace")
    return value


def _budget(value: object, rules: Rules, label: str) -> float:
    amount = _number(value, label) if isinstance(value, str) else nonnegative(value, label)
    exact = Decimal(str(value))
    _, digits, exponent = exact.as_tuple()
    # Inspect the exponent without Decimal arithmetic, whose default precision
    # could silently round very long CSV literals before precision validation.
    trailing_zeros = 0
    for digit in reversed(digits):
        if digit != 0:
            break
        trailing_zeros += 1
    if exact != 0 and exponent + trailing_zeros < -rules.money_decimals:
        raise ValueError(f"{label}: budget exceeds money_decimals={rules.money_decimals} precision")
    if Decimal(str(amount)) != exact:
        raise ValueError(f"{label}: budget cannot round-trip through the local numeric representation")
    return amount


def _register_types(advertisers, registry: dict[str, str]) -> None:
    """Validate player/type identity across one dataset, never process-globally."""
    for advertiser in advertisers:
        aid = _identity(advertiser.advertiser_id, "advertiser_id")
        kind = _identity(advertiser.advertiser_type, "advertiser_type")
        if aid in registry and registry[aid] != kind:
            raise ValueError(f"player_type conflict across dates for {aid}: {registry[aid]!r} != {kind!r}")
        registry[aid] = kind


def validate_day(day: DayData, rules: Rules, expected_n: int | None = None) -> None:
    """Validate identities, value coverage, chronological order and numbers."""
    _identity(day.day_id, "day_id")
    if expected_n is not None and (type(expected_n) is not int or expected_n < 1):
        raise ValueError("expected_n must be a positive integer")
    if not day.advertisers:
        raise ValueError("a day must contain at least one advertiser")
    if expected_n is not None and len(day.advertisers) != expected_n:
        raise ValueError(f"advertiser count: expected {expected_n}, got {len(day.advertisers)}")
    ids = set()
    for advertiser in day.advertisers:
        aid = _identity(advertiser.advertiser_id, "advertiser_id")
        _identity(advertiser.advertiser_type, "advertiser_type")
        if aid in ids:
            raise ValueError(f"duplicate advertiser: {aid}")
        ids.add(aid)
        _budget(advertiser.budget, rules, f"budget {aid}")
    if len(day.slots) != rules.total_slots:
        raise ValueError(f"slot count: expected {rules.total_slots}, got {len(day.slots)}")
    opportunity_ids = set()
    for slot_index, opportunities in enumerate(day.slots):
        previous_order = -1
        for opportunity in opportunities:
            if opportunity.day_id != day.day_id:
                raise ValueError("opportunity day_id does not match its day")
            if type(opportunity.time_slot) is not int or opportunity.time_slot != slot_index:
                raise ValueError("opportunity time_slot does not match its slot")
            oid = _identity(opportunity.opportunity_id, "opportunity_id")
            if oid in opportunity_ids:
                raise ValueError(f"duplicate opportunity_id: {oid}")
            opportunity_ids.add(oid)
            if type(opportunity.original_order) is not int or opportunity.original_order < 0:
                raise ValueError("original_order must be a nonnegative integer")
            if opportunity.original_order <= previous_order:
                raise ValueError("original_order must be unique and strictly increasing inside each slot")
            previous_order = opportunity.original_order
            validate_values_mask(opportunity, len(day.advertisers))


def _rng(seed: int, stream: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{stream}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def generate_synthetic(config: dict) -> list[DayData]:
    """Generate local example data, with independent traffic/value/budget RNGs.

    ``synthetic.days`` is a list of day IDs or a positive number of dates.
    Optional empty_slots/burst_slots use internal zero-based slot indices.
    All defaults are local example parameters, not official contest values.
    """
    rules = Rules(**config.get("rules", {}))
    options = config.get("synthetic", {})
    if not isinstance(options, dict):
        raise ValueError("synthetic must be a mapping")
    seed, n = options.get("seed", 20261018), options.get("n", 4)
    base = options.get("base_opportunities", 8)
    if type(seed) is not int or type(n) is not int or n < 1:
        raise ValueError("synthetic seed must be integer; n must be positive integer")
    if type(base) is not int or base < 0:
        raise ValueError("base_opportunities must be a nonnegative integer")
    days = options.get("days", ["2026-01-01", "2026-01-02", "2026-01-03"])
    if type(days) is int:
        if days < 1:
            raise ValueError("synthetic.days count must be positive")
        days = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(days)]
    if not isinstance(days, list) or not days:
        raise ValueError("synthetic.days must be a nonempty list or positive integer")
    for day_id in days:
        _identity(day_id, "synthetic day_id")
    if len(set(days)) != len(days):
        raise ValueError("duplicate synthetic day_id")
    low = nonnegative(options.get("budget_min", 40.0), "budget_min")
    high = nonnegative(options.get("budget_max", 160.0), "budget_max")
    if high < low:
        raise ValueError("budget_max must be >= budget_min")
    default_empty = [rules.total_slots // 3] if rules.total_slots > 2 else []
    default_bursts = [rules.total_slots * 2 // 3] if rules.total_slots > 2 else []
    empty = options.get("empty_slots", default_empty)
    bursts = options.get("burst_slots", default_bursts)
    for label, slots in (("empty_slots", empty), ("burst_slots", bursts)):
        if not isinstance(slots, list) or any(type(t) is not int or not 0 <= t < rules.total_slots for t in slots):
            raise ValueError(f"{label} must contain valid zero-based slot indices")
        if len(set(slots)) != len(slots):
            raise ValueError(f"{label} contains duplicates")
    if set(empty) & set(bursts):
        raise ValueError("empty_slots and burst_slots cannot overlap")
    generated = []
    for day_id in days:
        traffic_rng = _rng(seed, f"traffic:{day_id}")
        value_rng = _rng(seed, f"values:{day_id}")
        budget_rng = _rng(seed, f"budget:{day_id}")
        advertisers = tuple(
            Advertiser(f"advertiser_{i + 1}", ("app", "commerce", "service")[i % 3],
                       round(budget_rng.uniform(low, high), rules.money_decimals))
            for i in range(n)
        )
        scales = [0.65 + 0.7 * i / max(1, n - 1) for i in range(n)]
        slots = []
        for t in range(rules.total_slots):
            diurnal = 0.45 + 1.1 * (0.5 + 0.5 * math.sin(2 * math.pi * (t / rules.total_slots - 0.25)))
            count = int(round(base * diurnal * traffic_rng.uniform(0.65, 1.35)))
            if t in bursts:
                count *= 4
            if t in empty:
                count = 0
            opportunities = []
            for j in range(count):
                # Shared request attractiveness creates correlation; advertiser
                # affinity and independent noise make the estimates unequal.
                common = value_rng.lognormvariate(-0.2, 0.55)
                category = value_rng.randrange(3)
                values = tuple(round(common * scales[i] * (1.65 if i % 3 == category else 0.8)
                                     * value_rng.uniform(0.7, 1.3), 6) for i in range(n))
                opportunities.append(Opportunity(day_id, t, f"{day_id}:slot{t}:request{j}", j, values, (True,) * n))
            slots.append(tuple(opportunities))
        day = DayData(day_id, advertisers, tuple(slots))
        validate_day(day, rules, n)
        generated.append(day)
    return generated


def _split_names(count: int) -> list[str]:
    if count == 1:
        return ["train"]
    if count == 2:
        return ["train", "local_holdout"]
    train_count = max(1, int(count * 0.6))
    validation_count = max(1, int(count * 0.2))
    return ["train"] * train_count + ["validation"] * validation_count + ["local_holdout"] * (count - train_count - validation_count)


def write_dataset(days: list[DayData], path: str | Path) -> None:
    """Write three CSVs per complete date, plus a manifest with date splits.

    Existing files are never replaced.  Use a new output directory for a new
    generation, or explicitly remove your own generated dataset beforehand.
    """
    if not days:
        raise ValueError("cannot write an empty dataset")
    ordered = sorted(days, key=lambda day: day.day_id)
    if len({day.day_id for day in ordered}) != len(ordered):
        raise ValueError("duplicate day_id")
    total_slots = len(ordered[0].slots)
    validation_rules = Rules(total_slots=total_slots, money_decimals=9)
    n = len(ordered[0].advertisers)
    type_registry = {}
    for day in ordered:
        validate_day(day, validation_rules, n)
        _register_types(day.advertisers, type_registry)
    root = Path(path)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"dataset directory is nonempty; refusing to overwrite: {root}")
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (day, split) in enumerate(zip(ordered, _split_names(len(ordered)))):
        directory = f"day_{index:04d}"
        destination = root / directory
        destination.mkdir()
        with (destination / "advertisers.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS["advertisers.csv"])
            writer.writerows((day.day_id, a.advertiser_id, a.advertiser_type, a.budget) for a in day.advertisers)
        with (destination / "opportunities.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS["opportunities.csv"])
            writer.writerows((day.day_id, o.time_slot, o.opportunity_id, o.original_order) for slot in day.slots for o in slot)
        with (destination / "values.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS["values.csv"])
            writer.writerows((day.day_id, o.opportunity_id, a.advertiser_id, value)
                             for slot in day.slots for o in slot
                             for a, value, present in zip(day.advertisers, o.values, o.participation_mask) if present)
        # CSV numeric columns reload as floats. Canonicalize int-valued Python
        # fixtures before hashing so 100 and 100.0 round-trip identically.
        stored_day = DayData(day.day_id,
                            tuple(Advertiser(a.advertiser_id, a.advertiser_type, float(a.budget)) for a in day.advertisers),
                            tuple(tuple(Opportunity(o.day_id, o.time_slot, o.opportunity_id, o.original_order,
                                                    tuple(float(v) for v in o.values), o.participation_mask)
                                        for o in slot) for slot in day.slots))
        entries.append({"day_id": day.day_id, "directory": directory, "split": split,
                        "fingerprint": stored_day.fingerprint, "total_slots": total_slots, "advertiser_count": n})
    manifest = {"schema_version": SCHEMA_VERSION, "data_origin": "local_interchange_not_official",
                "missing_record_policy": "absent_participant; zero padding with explicit participation_mask",
                "split_policy": "whole_date_sorted_60_20_remainder; tiny counts documented", "days": entries}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS[path.name]):
            raise ValueError(f"{path}: expected exact CSV header {FIELDS[path.name]}")
        for row_number, row in enumerate(reader, 2):
            if None in row or any(value is None or value == "" for value in row.values()):
                raise ValueError(f"{path}:{row_number}: missing or excess CSV fields")
            yield row


def _integer(value: str, label: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError(f"{label}: expected nonnegative integer")
    return int(value)


def _number(value: str, label: str) -> float:
    try:
        exact = Decimal(value)
    except (ValueError, TypeError, InvalidOperation) as error:
        raise ValueError(f"{label}: invalid number {value!r}") from error
    if not exact.is_finite() or exact < 0:
        raise ValueError(f"{label}: expected finite nonnegative number, got {value!r}")
    parsed = nonnegative(float(exact), label)
    if exact > 0 and parsed == 0:
        raise ValueError(f"{label}: positive number underflows float representation")
    return parsed


def _read_partition(directory: Path, day_id: str, rules: Rules, expected_n: int | None, materialize: bool,
                    schema_version: str = SCHEMA_VERSION, type_registry: dict[str, str] | None = None) -> DayData | None:
    if schema_version not in (SCHEMA_VERSION, LEGACY_SCHEMA_VERSION):
        raise ValueError("unsupported local data schema_version")
    advertisers = []
    indexes = {}
    for row in _rows(directory / "advertisers.csv"):
        if row["day_id"] != day_id:
            raise ValueError("advertiser day_id does not match manifest partition")
        aid = _identity(row["advertiser_id"], "advertiser_id")
        if aid in indexes:
            raise ValueError(f"duplicate advertiser: {aid}")
        indexes[aid] = len(advertisers)
        budget = _budget(row["budget"], rules, "budget")
        advertisers.append(Advertiser(aid, _identity(row["advertiser_type"], "advertiser_type"), budget))
    if not advertisers or (expected_n is not None and len(advertisers) != expected_n):
        raise ValueError("advertiser count mismatch or empty advertisers")
    if type_registry is not None:
        _register_types(advertisers, type_registry)
    metadata = {}
    slot_orders = [set() for _ in range(rules.total_slots)]
    for row in _rows(directory / "opportunities.csv"):
        if row["day_id"] != day_id:
            raise ValueError("opportunity day_id does not match manifest partition")
        oid = _identity(row["opportunity_id"], "opportunity_id")
        slot = _integer(row["time_slot"], "time_slot")
        order = _integer(row["original_order"], "original_order")
        if slot >= rules.total_slots:
            raise ValueError("time_slot out of range")
        if oid in metadata:
            raise ValueError(f"duplicate opportunity_id: {oid}")
        if order in slot_orders[slot]:
            raise ValueError("duplicate original_order inside time_slot")
        slot_orders[slot].add(order)
        # Bitsets validate value coverage using O(requests), not O(requests*N),
        # during the full-dataset integrity scan.
        metadata[oid] = [slot, order, 0, [0.0] * len(advertisers) if materialize else None]
    for row in _rows(directory / "values.csv"):
        if row["day_id"] != day_id:
            raise ValueError("value day_id does not match manifest partition")
        oid, aid = row["opportunity_id"], row["advertiser_id"]
        if oid not in metadata or aid not in indexes:
            raise ValueError(f"unknown opportunity/advertiser association: {oid}/{aid}")
        i = indexes[aid]
        entry = metadata[oid]
        bit = 1 << i
        if entry[2] & bit:
            raise ValueError(f"duplicate value key: {oid}/{aid}")
        value = _number(row["estimated_value"], "estimated_value")
        entry[2] |= bit
        if materialize:
            entry[3][i] = value
    complete = (1 << len(advertisers)) - 1
    if schema_version == LEGACY_SCHEMA_VERSION and any(entry[2] != complete for entry in metadata.values()):
        raise ValueError("missing estimated_value for opportunity/advertiser pair in complete-matrix v1 source")
    if not materialize:
        return None
    slots = [[] for _ in range(rules.total_slots)]
    for oid, (slot, order, presence, values) in metadata.items():
        mask = tuple(bool(presence & (1 << i)) for i in range(len(advertisers)))
        slots[slot].append(Opportunity(day_id, slot, oid, order, tuple(values), mask))
    day = DayData(day_id, tuple(advertisers), tuple(tuple(sorted(slot, key=lambda o: o.original_order)) for slot in slots))
    validate_day(day, rules, expected_n)
    return day


def load_days(path: str | Path, rules: Rules, expected_n: int | None = None, split: str | None = None) -> Iterator[DayData]:
    """Validate every partition, then yield selected dates one at a time.

    Values are streamed; unselected dates never materialize value matrices.
    Scanning all partitions intentionally catches invalid associations outside
    the selected split. A caller must consume the generator to trigger I/O.
    """
    if expected_n is not None and (type(expected_n) is not int or expected_n < 1):
        raise ValueError("expected_n must be a positive integer")
    if split is not None and split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in (SCHEMA_VERSION, LEGACY_SCHEMA_VERSION):
        raise ValueError("unsupported local data schema_version")
    source_schema = manifest["schema_version"]
    entries = manifest.get("days")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest.days must be a nonempty list")
    partitions, seen_ids, seen_directories = [], set(), set()
    type_registry = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid manifest day entry")
        day_id = _identity(entry.get("day_id"), "manifest day_id")
        if day_id in seen_ids:
            raise ValueError("duplicate day_id in manifest")
        seen_ids.add(day_id)
        if entry.get("split") not in SPLITS:
            raise ValueError("invalid or missing date split")
        relative = _identity(entry.get("directory"), "manifest directory")
        directory = (root / relative).resolve()
        if directory == root or root not in directory.parents:
            raise ValueError("manifest directory must stay inside dataset root")
        if directory in seen_directories:
            raise ValueError("duplicate manifest directory")
        seen_directories.add(directory)
        if entry.get("total_slots") != rules.total_slots:
            raise ValueError("manifest total_slots differs from rules")
        advertised_n = entry.get("advertiser_count")
        if type(advertised_n) is not int or advertised_n < 1:
            raise ValueError("manifest advertiser_count must be positive integer")
        if expected_n is not None and advertised_n != expected_n:
            raise ValueError("manifest advertiser_count differs from expected_n")
        _read_partition(directory, day_id, rules, advertised_n, materialize=False,
                        schema_version=source_schema, type_registry=type_registry)
        partitions.append((entry, directory))
    chosen = [(entry, directory) for entry, directory in partitions if split is None or entry["split"] == split]
    if not chosen:
        raise ValueError(f"dataset has no dates in requested split: {split}")
    for entry, directory in chosen:
        day = _read_partition(directory, entry["day_id"], rules, entry["advertiser_count"], materialize=True,
                              schema_version=source_schema)
        expected_fingerprint = entry.get("fingerprint")
        source_fingerprint = day.legacy_fingerprint if source_schema == LEGACY_SCHEMA_VERSION else day.fingerprint
        if expected_fingerprint is not None and (not isinstance(expected_fingerprint, str)
                                                or not re.fullmatch(r"[0-9a-f]{64}", expected_fingerprint)):
            raise ValueError("manifest fingerprint must be a SHA256 hex digest when supplied")
        if expected_fingerprint is not None and source_fingerprint != expected_fingerprint:
            raise ValueError(f"day fingerprint mismatch: {day.day_id}; regenerate manifest after an intentional data change")
        yield DayData(day.day_id, day.advertisers, day.slots, source_schema, expected_fingerprint)


def _history_rows(path: Path, kind: str, column_mapping: dict) -> Iterator[dict[str, str]]:
    fields = FIELDS[f"{kind}.csv"]
    mapping = column_mapping.get(kind, {})
    if not isinstance(mapping, dict) or any(field not in fields for field in mapping):
        raise ValueError(f"invalid column_mapping for {kind}")
    source_columns = {field: _identity(mapping.get(field, field), "source column") for field in fields}
    if len(set(source_columns.values())) != len(source_columns):
        raise ValueError(f"multiple {kind} fields cannot map to the same source column")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if not headers or len(set(headers)) != len(headers) or not set(source_columns.values()).issubset(headers):
            raise ValueError(f"{path}: source header is missing mapped columns or contains duplicates")
        for line, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"{path}:{line}: source row column count mismatch")
            row = {field: raw[source] for field, source in source_columns.items()}
            if any(value == "" for value in row.values()):
                raise ValueError(f"{path}:{line}: missing mapped value")
            yield row


def import_history_csv(
    source_advertisers: str | Path,
    source_opportunities: str | Path,
    source_values: str | Path,
    destination: str | Path,
    rules: Rules,
    column_mapping: dict | None = None,
    split_by_day: dict[str, str] | None = None,
) -> dict:
    """Import three multi-date historical CSVs through explicit column maps.

    Mappings are ``{table: {local_field: source_column}}``; unlisted fields
    retain their local names. No date, identity, value or slot inference occurs.
    Source files are rescanned per date, trading I/O for one-day peak memory.
    A failure leaves partial CSVs for diagnosis but never publishes a manifest.
    """
    mapping = {} if column_mapping is None else column_mapping
    if not isinstance(mapping, dict) or any(key not in ("advertisers", "opportunities", "values") for key in mapping):
        raise ValueError("column_mapping must map known table names to column maps")
    sources = {"advertisers": Path(source_advertisers), "opportunities": Path(source_opportunities), "values": Path(source_values)}
    day_ids = set()
    type_registry = {}
    for row in _history_rows(sources["advertisers"], "advertisers", mapping):
        day_ids.add(_identity(row["day_id"], "historical day_id"))
        _register_types((Advertiser(row["advertiser_id"], row["advertiser_type"], 0),), type_registry)
    if not day_ids:
        raise ValueError("historical advertisers contains no dates")
    ordered_days = sorted(day_ids)
    if split_by_day is None:
        splits = dict(zip(ordered_days, _split_names(len(ordered_days))))
    else:
        if not isinstance(split_by_day, dict) or set(split_by_day) != day_ids or any(split not in SPLITS for split in split_by_day.values()):
            raise ValueError("split_by_day must assign exactly every source date to a valid split")
        splits = dict(split_by_day)
    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"dataset directory is nonempty; refusing to overwrite: {root}")
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    expected_n = None
    for index, day_id in enumerate(ordered_days):
        directory = f"day_{index:04d}"
        partition = root / directory
        partition.mkdir()
        for kind, source in sources.items():
            with (partition / f"{kind}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS[f"{kind}.csv"])
                writer.writeheader()
                for row in _history_rows(source, kind, mapping):
                    if row["day_id"] not in day_ids:
                        raise ValueError(f"historical {kind} references a day with no advertisers: {row['day_id']}")
                    if row["day_id"] == day_id:
                        writer.writerow(row)
        day = _read_partition(partition, day_id, rules, expected_n, materialize=True, type_registry=type_registry)
        expected_n = len(day.advertisers)
        entries.append({"day_id": day_id, "directory": directory, "split": splits[day_id],
                        "fingerprint": day.fingerprint, "total_slots": rules.total_slots,
                        "advertiser_count": expected_n})
    manifest = {"schema_version": SCHEMA_VERSION, "data_origin": "user_supplied_history_explicit_mapping_not_official_schema",
                "missing_record_policy": "absent_participant; zero padding with explicit participation_mask",
                "split_policy": "explicit_whole_date" if split_by_day is not None else "whole_date_sorted_60_20_remainder; tiny counts documented",
                "column_mapping": mapping, "days": entries}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
