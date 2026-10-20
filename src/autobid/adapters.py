"""Convert AuctionNet value logs into the competition request contract.

Historical bids, outcomes, budgets, CPA and exposure fields are intentionally
unused. The converted request stream must be auctioned again under our rules.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path

from .data import FIELDS, _integer, _text, _value

UPSTREAM_COMMIT = "c20de631a40e24669d1d6a8e2acefd8f492f172d"
REQUIRED = ("deliveryPeriodIndex", "timeStepIndex", "pvIndex", "advertiserNumber",
            "advertiserCategoryIndex", "pValue")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_auctionnet(source: str | Path, destination: str | Path, *, period: str | int,
                      source_steps: int = 48, target_steps: int = 72, time_base: int = 0,
                      pvalue_scale: float = 1.0, max_requests: int | None = None) -> dict:
    """Convert exactly one explicitly selected delivery period.

    Numeric pvIndex defines source order within a slot. This assumption is saved
    in the manifest and must be checked against the actual source shard. Without
    timestamps, reslotting uses uniformly spaced request midpoints within each
    old slot. All participants of a request receive the same new time_step.

    Optional systematic subsampling retains complete requests, spread across the
    original period, without compressing their time slots. SQLite staging keeps
    large input CSV rows off Python's heap; disk space is still required.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different files")
    for name, value in (("source_steps", source_steps), ("target_steps", target_steps)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if time_base not in (0, 1) or isinstance(time_base, bool):
        raise ValueError("time_base must explicitly be 0 or 1")
    if not math.isfinite(pvalue_scale) or pvalue_scale <= 0:
        raise ValueError("pvalue_scale must be finite and positive")
    if max_requests is not None and (not isinstance(max_requests, int) or isinstance(max_requests, bool) or max_requests <= 0):
        raise ValueError("max_requests must be a positive integer or None")
    period_text = _text(str(period), "period", 0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_rows = selected_rows = output_rows = output_requests = 0
    player_types = {}
    with tempfile.TemporaryDirectory(prefix="autobid-adapter-") as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "stage.sqlite"))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE rows (pv INTEGER NOT NULL, step INTEGER NOT NULL, player TEXT NOT NULL, type TEXT NOT NULL, value REAL NOT NULL, rownum INTEGER NOT NULL, PRIMARY KEY(pv, player))")
            with source.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or not set(REQUIRED).issubset(reader.fieldnames):
                    raise ValueError(f"AuctionNet CSV requires unique columns including: {', '.join(REQUIRED)}")
                for line, row in enumerate(reader, 2):
                    total_rows += 1
                    if None in row or any(v is None for v in row.values()):
                        raise ValueError(f"line {line}: column count does not match header")
                    if row["deliveryPeriodIndex"] != period_text:
                        continue
                    step = _integer(row["timeStepIndex"], "timeStepIndex", line) - time_base
                    if not 0 <= step < source_steps:
                        raise ValueError(f"line {line}: timeStepIndex outside configured range; check time_base")
                    pv = _integer(row["pvIndex"], "pvIndex", line)
                    if pv > 2**63 - 1:
                        raise ValueError(f"line {line}: pvIndex exceeds supported signed 64-bit integer")
                    pid = _text(row["advertiserNumber"], "advertiserNumber", line)
                    ptype = _text(row["advertiserCategoryIndex"], "advertiserCategoryIndex", line)
                    value = _value(row["pValue"], "pValue", line) * pvalue_scale
                    if not math.isfinite(value):
                        raise ValueError(f"line {line}: scaled pValue is non-finite")
                    if pid in player_types and player_types[pid] != ptype:
                        raise ValueError(f"line {line}: advertiser {pid!r} changes category")
                    player_types[pid] = ptype
                    try:
                        connection.execute("INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?)", (pv, step, pid, ptype, value, line))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"line {line}: duplicate (pvIndex, advertiserNumber) in selected period") from exc
                    selected_rows += 1
            if not selected_rows:
                raise ValueError(f"selected delivery period {period_text!r} has no rows")
            inconsistent = connection.execute("SELECT pv FROM rows GROUP BY pv HAVING MIN(step) != MAX(step) LIMIT 1").fetchone()
            if inconsistent:
                raise ValueError(f"pvIndex {inconsistent[0]} occurs in multiple source time slots")
            connection.execute("CREATE TABLE requests AS SELECT pv, MIN(step) AS step FROM rows GROUP BY pv")
            counts = dict(connection.execute("SELECT step, COUNT(*) FROM requests GROUP BY step"))
            total_requests = sum(counts.values())
            kept_count = min(total_requests, max_requests or total_requests)
            # O(number of retained requests), rather than O(number of source rows).
            chosen = None if kept_count == total_requests else {((2 * i + 1) * total_requests) // (2 * kept_count) for i in range(kept_count)}
            connection.execute("CREATE TABLE mapped (pv INTEGER PRIMARY KEY, target_step INTEGER NOT NULL)")
            current_step, ordinal = None, 0
            for global_order, (pv, step) in enumerate(connection.execute("SELECT pv, step FROM requests ORDER BY step, pv")):
                if step != current_step:
                    current_step, ordinal = step, 0
                count = counts[step]
                target_step = ((2 * step * count + 2 * ordinal + 1) * target_steps) // (2 * count * source_steps)
                if chosen is None or global_order in chosen:
                    connection.execute("INSERT INTO mapped VALUES (?, ?)", (pv, target_step))
                    output_requests += 1
                ordinal += 1
            handle = tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=destination.parent, prefix=".convert-", delete=False)
            tmp_output = Path(handle.name)
            try:
                with handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    writer.writeheader()
                    query = "SELECT r.pv, m.target_step, r.player, r.type, r.value FROM rows r JOIN mapped m ON r.pv=m.pv ORDER BY r.step, r.pv, r.rownum"
                    for pv, step, pid, ptype, value in connection.execute(query):
                        writer.writerow({"index": output_rows, "time_step": step,
                                         "request_id": f"auctionnet:{period_text}:{pv}",
                                         "player_id": pid, "player_type": ptype, "pvalue": repr(value)})
                        output_rows += 1
                os.replace(tmp_output, destination)
            finally:
                tmp_output.unlink(missing_ok=True)
        finally:
            connection.close()
    manifest = {
        "schema_version": 1, "kind": "auctionnet_value_stream_adaptation",
        "source_file": source.name, "source_sha256": sha256_file(source),
        "output_file": destination.name, "output_sha256": sha256_file(destination),
        "upstream_repository": "https://github.com/alimama-tech/AuctionNet",
        "upstream_reference_commit": UPSTREAM_COMMIT,
        "source_delivery_period": period_text, "source_rows_scanned": total_rows,
        "selected_period_rows": selected_rows, "selected_period_requests": total_requests,
        "output_rows": output_rows, "output_requests": output_requests,
        "source_steps": source_steps, "target_steps": target_steps, "source_time_base": time_base,
        "pvalue_scale": pvalue_scale, "player_types": player_types,
        "request_order": "source timeStepIndex, then numeric pvIndex; validate against source shard",
        "time_mapping": "uniform request midpoints within each source slot; synthetic timing assumption, no actual timestamps",
        "sampling": "none" if kept_count == total_requests else "deterministic systematic complete-request subsample across original period",
        "max_requests": max_requests,
        "missing_pairs": "not participating; no zero padding",
        "value_semantics": "AuctionNet pValue is conversion probability; used only as a proxy contest value",
        "historical_fields": "bid, cost, remainingBudget, CPAConstraint and outcomes ignored; replay again",
        "budgets": "provide separately, calibrate to selected request volume and pvalue scale",
        "licensing": "Code reference Apache-2.0; verify separate dataset usage conditions before redistribution",
    }
    destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
