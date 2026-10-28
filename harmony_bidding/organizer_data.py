"""Strict adapter for the six-column long table described by the user.

Dates, budgets, slot base, request identity scope and ordering are explicitly
configured local assumptions.  This evaluator-only adapter does not establish
the organizer's full data/API contract and never exposes its inputs to Agents.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterator

from .data import FIELDS, SCHEMA_VERSION, SPLITS, _budget, _identity, _number, _register_types, _split_names, validate_day
from .schema import Advertiser, DayData, Opportunity, Rules, fingerprint


SOURCE_FIELDS = ("index", "time_step", "request_id", "player_id", "player_type", "pvalue")
ORDER_FIELDS = ("day_id", "time_step", "request_id", "original_order")
SOURCE_SCHEMA_VERSION = "organizer-six-fields-v1"
_OPTION_FIELDS = {
    "sources", "time_step_base", "request_id_scope", "ordering", "players",
    "budgets_by_day", "split_by_day", "source_kind",
}


@dataclass(slots=True)
class _Request:
    slot: int
    request_id: str
    order: int
    values: list[float | None]


def _rows(path: Path, required: tuple[str, ...]) -> Iterator[tuple[int, dict[str, str]]]:
    """Keep identifier spelling; parse no column through numeric inference."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        headers = reader.fieldnames
        if not headers or any(not name for name in headers) or len(headers) != len(set(headers)):
            raise ValueError(f"{path}: missing, empty or duplicate CSV header")
        if not set(required).issubset(headers):
            raise ValueError(f"{path}: missing required columns: {sorted(set(required) - set(headers))}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{reader.line_num}: CSV row has missing or excess columns")
            if any(row[field] == "" for field in required):
                raise ValueError(f"{path}:{reader.line_num}: missing required value")
            yield reader.line_num, {field: row[field] for field in required}


def _integer(value: str, label: str) -> int:
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"{label}: expected nonnegative integer")
    return int(value)


def _slot(value: str, base: int, rules: Rules) -> int:
    slot = _integer(value, "time_step") - base
    if not 0 <= slot < rules.total_slots:
        raise ValueError(f"time_step out of range for explicit base={base} and total_slots={rules.total_slots}")
    return slot


def _request_key(slot: int, request_id: str, scope: str) -> tuple[int | None, str]:
    return (None if scope == "day" else slot, request_id)


def _opportunity_id(slot: int, request_id: str, scope: str) -> str:
    if scope == "day":
        return request_id
    # An injective encoding, including strings that themselves look like JSON,
    # avoids delimiter collisions without altering the source request ID.
    return json.dumps([slot, request_id], ensure_ascii=False, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _explicit_orders(path: Path, day_id: str, all_days: set[str], rules: Rules,
                     base: int, scope: str) -> dict[tuple[int | None, str], tuple[int, int]]:
    orders = {}
    used_orders = [set() for _ in range(rules.total_slots)]
    for line, row in _rows(path, ORDER_FIELDS):
        source_day = _identity(row["day_id"], "order day_id")
        if source_day not in all_days:
            raise ValueError(f"{path}:{line}: explicit order has unknown day_id: {source_day}")
        slot = _slot(row["time_step"], base, rules)
        request_id = _identity(row["request_id"], "order request_id")
        order = _integer(row["original_order"], "original_order")
        if source_day != day_id:
            continue
        key = _request_key(slot, request_id, scope)
        if key in orders:
            if orders[key][0] != slot:
                raise ValueError(f"{path}:{line}: explicit order request time_step conflict: {request_id}")
            raise ValueError(f"{path}:{line}: duplicate explicit order request: {request_id}")
        if order in used_orders[slot]:
            raise ValueError(f"{path}:{line}: duplicate original_order inside time_step")
        used_orders[slot].add(order)
        orders[key] = slot, order
    return orders


def _write_day(day: DayData, partition: Path) -> None:
    partition.mkdir()
    with (partition / "advertisers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS["advertisers.csv"])
        writer.writerows((day.day_id, a.advertiser_id, a.advertiser_type, a.budget) for a in day.advertisers)
    with (partition / "opportunities.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS["opportunities.csv"])
        writer.writerows((day.day_id, o.time_slot, o.opportunity_id, o.original_order) for slot in day.slots for o in slot)
    with (partition / "values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS["values.csv"])
        writer.writerows((day.day_id, o.opportunity_id, a.advertiser_id, value)
                         for slot in day.slots for o in slot
                         for a, value, present in zip(day.advertisers, o.values, o.participation_mask) if present)


def import_organizer_csv(config: dict) -> dict:
    """Import configured six-column files into the existing daily CSV format.

    Required ``organizer_import`` fields are ``sources`` (path/day_id pairs),
    ``time_step_base`` (0/1), ``request_id_scope`` (day/time_step), ``ordering``
    (file_first_seen or explicit_csv with path), and a complete ``players``
    mapping with string IDs/types and explicit budgets. Optional day budgets
    and date splits must be complete for each date they specify.

    Source files for a day are shards of the same request namespace; duplicate
    request/player rows always fail, including identical duplicates in shards.
    Only recorded request/player pairs participate. Missing pairs become
    internal zero padding with mask=False; a real zero-valued row has mask=True.
    Existing rows with missing/invalid pvalue remain errors, never absences.
    Index is validated as a row label only; it never chooses request order.
    Peak value storage is one date; explicit order files are rescanned per day.
    Existing data is never overwritten. A failure may leave partial daily CSVs
    for diagnosis, but no manifest is published until all dates validate.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    options = config.get("organizer_import")
    if not isinstance(options, dict):
        raise ValueError("organizer_import must be an explicit mapping")
    if set(options) - _OPTION_FIELDS:
        raise ValueError(f"unknown organizer_import fields: {sorted(set(options) - _OPTION_FIELDS)}")
    options = copy.deepcopy(options)
    rules = Rules(**config.get("rules", {}))
    destination = Path(_identity(config.get("data_dir"), "data_dir")).resolve()
    base = options.get("time_step_base")
    if type(base) is not int or base not in (0, 1):
        raise ValueError("time_step_base must be explicitly configured as integer 0 or 1")
    scope = options.get("request_id_scope")
    if scope not in ("day", "time_step"):
        raise ValueError("request_id_scope must be explicitly configured as day or time_step")
    source_kind = _identity(options.get("source_kind", "user_supplied"), "source_kind")
    ordering = options.get("ordering")
    if not isinstance(ordering, dict) or ordering.get("mode") not in ("file_first_seen", "explicit_csv"):
        raise ValueError("ordering must explicitly select file_first_seen or explicit_csv")
    mode = ordering["mode"]
    if set(ordering) != ({"mode"} if mode == "file_first_seen" else {"mode", "path"}):
        raise ValueError("ordering has missing or unexpected fields; index-based ordering is not supported")
    order_path = Path(_identity(ordering["path"], "ordering.path")).resolve() if mode == "explicit_csv" else None
    source_options = options.get("sources")
    if not isinstance(source_options, list) or not source_options:
        raise ValueError("sources must be a nonempty explicit path/day_id list")
    sources = []
    seen_paths = set()
    for source in source_options:
        if not isinstance(source, dict) or set(source) != {"path", "day_id"}:
            raise ValueError("each source must contain exactly path and day_id")
        path = Path(_identity(source["path"], "source.path")).resolve()
        day_id = _identity(source["day_id"], "source.day_id")
        if path in seen_paths:
            raise ValueError(f"duplicate source path, including cross-day reuse: {path}")
        seen_paths.add(path)
        if not path.is_file():
            raise ValueError(f"source path is not a file: {path}")
        sources.append({"path": str(path), "day_id": day_id, "sha256": _sha256(path),
                        "row_count": 0, "request_count": 0})
    day_ids = sorted({source["day_id"] for source in sources})
    all_days = set(day_ids)
    roster = options.get("players")
    if not isinstance(roster, dict) or not roster:
        raise ValueError("players must be an explicit nonempty complete roster")
    for player_id, spec in roster.items():
        _identity(player_id, "players player_id (quote numeric IDs in YAML)")
        if not isinstance(spec, dict) or set(spec) != {"player_type", "budget"}:
            raise ValueError(f"player {player_id} must have exactly player_type and budget")
        _identity(spec["player_type"], "players player_type (quote numeric types in YAML)")
        _budget(spec["budget"], rules, f"budget for {player_id}")
    player_ids = sorted(roster)
    player_indexes = {player_id: index for index, player_id in enumerate(player_ids)}
    expected_n = config.get("expected_n")
    if expected_n is not None and (type(expected_n) is not int or expected_n < 1 or expected_n != len(roster)):
        raise ValueError("expected_n must be positive and match the configured player roster")
    budgets_by_day = options.get("budgets_by_day", {})
    if not isinstance(budgets_by_day, dict) or set(budgets_by_day) - all_days:
        raise ValueError("budgets_by_day contains unknown dates or is not a mapping")
    for day_id, budgets in budgets_by_day.items():
        if not isinstance(budgets, dict) or set(budgets) != set(roster):
            raise ValueError(f"budgets_by_day[{day_id}] must cover exactly all player IDs")
        for player_id, budget in budgets.items():
            _budget(budget, rules, f"budget for {day_id}/{player_id}")
    splits = options.get("split_by_day")
    if splits is None:
        splits = dict(zip(day_ids, _split_names(len(day_ids))))
    elif not isinstance(splits, dict) or set(splits) != all_days or any(value not in SPLITS for value in splits.values()):
        raise ValueError("split_by_day must assign exactly every date to train/validation/local_holdout")
    order_digest = _sha256(order_path) if order_path is not None else None
    import_config_fingerprint = fingerprint({"rules": asdict(rules), "organizer_import": options, "expected_n": expected_n})
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"dataset directory is nonempty; refusing to overwrite: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    total_requests = 0
    type_registry = {}
    for day_index, day_id in enumerate(day_ids):
        requests: dict[tuple[int | None, str], _Request] = {}
        next_orders = [0] * rules.total_slots
        for source in sources:
            if source["day_id"] != day_id:
                continue
            source_requests = set()
            path = Path(source["path"])
            for line, row in _rows(path, SOURCE_FIELDS):
                source["row_count"] += 1
                _integer(row["index"], "index")
                slot = _slot(row["time_step"], base, rules)
                request_id = _identity(row["request_id"], "request_id")
                player_id = _identity(row["player_id"], "player_id")
                player_type = _identity(row["player_type"], "player_type")
                if player_id not in roster:
                    raise ValueError(f"{path}:{line}: unknown player_id outside configured roster: {player_id}")
                if player_type != roster[player_id]["player_type"]:
                    raise ValueError(f"{path}:{line}: player_type conflict for {player_id}; type must be fixed across all dates/files")
                value = _number(row["pvalue"], "pvalue")
                key = _request_key(slot, request_id, scope)
                source_requests.add(key)
                if key not in requests:
                    requests[key] = _Request(slot, request_id, next_orders[slot], [None] * len(roster))
                    next_orders[slot] += 1
                request = requests[key]
                if request.slot != slot:
                    raise ValueError(f"{path}:{line}: request time_step conflict under day scope: {request_id}")
                index = player_indexes[player_id]
                if request.values[index] is not None:
                    raise ValueError(f"{path}:{line}: duplicate request/player pair, including across shards: {day_id}/{request_id}/{player_id}")
                request.values[index] = value
            source["request_count"] = len(source_requests)
        if order_path is not None:
            explicit = _explicit_orders(order_path, day_id, all_days, rules, base, scope)
            unknown, missing = set(explicit) - set(requests), set(requests) - set(explicit)
            if unknown:
                raise ValueError(f"explicit order contains unknown requests for {day_id}: {sorted(unknown, key=str)}")
            if missing:
                raise ValueError(f"explicit order is missing requests for {day_id}: {sorted(missing, key=str)}")
            for key, request in requests.items():
                slot, order = explicit[key]
                if request.slot != slot:
                    raise ValueError(f"explicit order request time_step conflict: {day_id}/{request.request_id}")
                request.order = order
        advertisers = tuple(Advertiser(player_id, roster[player_id]["player_type"],
                                       _budget(budgets_by_day.get(day_id, {}).get(player_id, roster[player_id]["budget"]),
                                               rules, f"budget for {day_id}/{player_id}"))
                            for player_id in player_ids)
        _register_types(advertisers, type_registry)
        slots = [[] for _ in range(rules.total_slots)]
        for request in requests.values():
            slots[request.slot].append(Opportunity(day_id, request.slot,
                                                   _opportunity_id(request.slot, request.request_id, scope),
                                                   request.order,
                                                   tuple(0.0 if value is None else value for value in request.values),
                                                   tuple(value is not None for value in request.values)))
        day = DayData(day_id, advertisers, tuple(tuple(sorted(slot, key=lambda opportunity: opportunity.original_order)) for slot in slots))
        validate_day(day, rules, len(roster))
        directory = f"day_{day_index:04d}"
        _write_day(day, destination / directory)
        request_count = len(requests)
        total_requests += request_count
        entries.append({"day_id": day_id, "directory": directory, "split": splits[day_id],
                        "fingerprint": day.fingerprint, "total_slots": rules.total_slots,
                        "advertiser_count": len(roster), "request_count": request_count,
                        "opportunity_counts_by_player": {
                            player: sum(opportunity.participation_mask[i] for slot in day.slots for opportunity in slot)
                            for i, player in enumerate(player_ids)}})
        # Do not retain previous dates' matrices while parsing the next date.
        del day, slots, requests
    for source in sources:
        if _sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"source file changed during import: {source['path']}")
    if order_path is not None and _sha256(order_path) != order_digest:
        raise ValueError("explicit order file changed during import")
    assumption = ("Local assumption: request order is first occurrence in the explicitly listed source files, "
                  "then CSV row order within each time_step; index and request-ID sorting are ignored."
                  if mode == "file_first_seen" else
                  "Local assumption: the explicitly supplied order CSV defines request order; "
                  "its provenance and agreement with official order remain the caller's responsibility.")
    ordering_metadata = {"mode": mode, "is_local_assumption": True}
    if order_path is not None:
        ordering_metadata.update({"path": str(order_path), "sha256": order_digest})
    metadata = {"source_schema_version": SOURCE_SCHEMA_VERSION, "source_kind": source_kind,
                "rules_version": rules.rules_version, "assumption_profile": rules.assumption_profile,
                "source_files": sources, "row_count": sum(source["row_count"] for source in sources),
                "request_count": total_requests, "ordering": ordering_metadata,
                "ordering_assumption": assumption, "request_id_scope": scope, "time_step_base": base,
                "missing_pvalue_policy": "error", "invalid_existing_value_policy": "error",
                "missing_record_policy": "absent_participant",
                "participation_mask_schema": "explicit-row-presence-v1",
                "participation_semantics": "Only recorded request/player pairs participate; real zero rows remain present.",
                "index_semantics": "Nonnegative row label only; not request identity or ordering.",
                "day_assignment": "Explicit sources.day_id configuration; no inference from path or index.",
                "budget_source": "Explicit local players/budgets_by_day configuration; not inferred from the six source columns.",
                "config_fingerprint": import_config_fingerprint, "import_config": options}
    if "_dataset_derivation_metadata" in config:
        metadata["derivation"] = copy.deepcopy(config["_dataset_derivation_metadata"])
    manifest = {"schema_version": SCHEMA_VERSION, "data_origin": "user_described_six_field_table_with_explicit_local_assumptions",
                "source_kind": source_kind, "split_policy": "explicit_whole_date" if options.get("split_by_day") is not None else "whole_date_sorted_60_20_remainder; tiny counts documented",
                "rules_version": rules.rules_version, "assumption_profile": rules.assumption_profile,
                "import_metadata": metadata, "days": entries}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
