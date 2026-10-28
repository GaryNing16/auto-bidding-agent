"""Verified evaluator-only matrices from the retained AuctionNet conversion.

This is an I/O cache, not an auction engine or a second source conversion.
It preserves every request and float64 pValue in a selected complete period.
Only the evaluator may receive these objects; agents never receive a handle.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from harmony_bidding.schema import Advertiser, fingerprint


CACHE_VERSION = "harmony-baseline-complete-matrix-v1"
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "data" / "auctionnet_72"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "outputs" / "baseline_comparison_20261028" / "cache"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MatrixSlot:
    values: np.ndarray
    participation_mask: np.ndarray
    request_ids: np.ndarray
    original_order: np.ndarray


@dataclass(frozen=True)
class MatrixDay:
    day_id: str
    advertisers: tuple[Advertiser, ...]
    slots: tuple[MatrixSlot, ...]
    fingerprint: str
    source: dict


def source_manifest(source_root=SOURCE_ROOT):
    root = Path(source_root)
    result = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    periods = result["periods"]
    types = {}
    ids = None
    for period in periods:
        roster = period["players"]
        if ids is None:
            ids = set(roster)
        if set(roster) != ids:
            raise ValueError("inconsistent AuctionNet player roster across periods")
        for player, spec in roster.items():
            kind = spec["player_type"]
            if player in types and types[player] != kind:
                raise ValueError("player_type changed across periods")
            types[player] = kind
    return result


def _source_entry(day_id, root):
    matches = [entry for entry in source_manifest(root)["periods"] if entry["day_id"] == day_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous day: {day_id}")
    return matches[0]


def _source_signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def build_cache(day_id, cache_root=DEFAULT_CACHE, source_root=SOURCE_ROOT):
    """Validate source digests and native full coverage, then cache one period.

    No test period is automatically loaded: the caller must name each period.
    An existing completed cache is verified and reused; source files are read
    only, and its original manifest remains untouched.
    """
    root, cache_root = Path(source_root), Path(cache_root)
    entry = _source_entry(day_id, root)
    destination = cache_root / day_id
    manifest_path = destination / "cache.json"
    if manifest_path.exists():
        return load_day(day_id, cache_root, source_root)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(f"incomplete/nonempty cache directory: {destination}")
    start = time.perf_counter()
    source_dir = root / day_id
    source_paths = {name: source_dir / entry["files"][name]["path"] for name in ("six_fields", "request_order")}
    signatures = {name: _source_signature(path) for name, path in source_paths.items()}
    for name, path in source_paths.items():
        expected = entry["files"][name]
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"source integrity failure: {path}")
    roster = entry["players"]
    players = sorted(roster, key=int)
    if players != [str(i) for i in range(len(players))]:
        raise ValueError("native full-matrix cache expects contiguous numeric player IDs")
    n, requests = len(players), int(entry["request_count"])
    if entry["row_count"] != n * requests or sum(entry["target72counts"]) != requests:
        raise ValueError("native full-matrix metadata has inconsistent coverage")
    order = pd.read_csv(source_paths["request_order"], dtype={"day_id": str, "time_step": np.int64,
                         "request_id": str, "original_order": np.int64}, keep_default_na=False)
    if list(order.columns) != ["day_id", "time_step", "request_id", "original_order"]:
        raise ValueError("unexpected request-order columns")
    if len(order) != requests or not order["day_id"].eq(day_id).all():
        raise ValueError("request-order count/day mismatch")
    times = order["time_step"].to_numpy(dtype=np.int64)
    ids = order["request_id"].to_numpy(dtype=str)
    orders = order["original_order"].to_numpy(dtype=np.int64)
    if len(set(ids)) != requests or any(not x or x.strip() != x for x in ids):
        raise ValueError("request IDs must be nonempty and unique")
    if np.any(times < 0) or np.any(times >= 72) or np.any(np.diff(times) < 0) or np.any(orders < 0):
        raise ValueError("invalid request order/time slots")
    counts = np.bincount(times, minlength=72)
    if counts.tolist() != entry["target72counts"]:
        raise ValueError("request-order slot counts disagree with conversion metadata")
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        if np.any(np.diff(orders[begin:end]) <= 0):
            raise ValueError("original_order is not strictly increasing inside slot")
    values = np.lib.format.open_memmap(destination / "values.npy", mode="w+", dtype=np.float64, shape=(requests, n))
    mask = np.lib.format.open_memmap(destination / "participation_mask.npy", mode="w+", dtype=np.bool_, shape=(requests, n))
    mask[:] = True  # Only after verifying every row in every group below.
    expected_players = np.arange(n, dtype=np.int64)
    expected_types = np.array([int(roster[p]["player_type"]) for p in players], dtype=np.int64)
    cursor = 0
    columns = ["index", "time_step", "request_id", "player_id", "player_type", "pvalue"]
    dtype = {"index": np.int64, "time_step": np.int64, "request_id": str,
             "player_id": np.int64, "player_type": np.int64, "pvalue": np.float64}
    for chunk in pd.read_csv(source_paths["six_fields"], dtype=dtype, chunksize=n * 10000,
                             float_precision="round_trip", keep_default_na=False):
        if list(chunk.columns) != columns or len(chunk) % n:
            raise ValueError("invalid native six-field schema or incomplete request group")
        count = len(chunk) // n
        stop = cursor + count
        if stop > requests:
            raise ValueError("more requests than declared")
        if not np.array_equal(chunk["index"].to_numpy(), np.arange(cursor*n, stop*n)):
            raise ValueError("native conversion row index is not contiguous")
        pids = chunk["player_id"].to_numpy().reshape(count, n)
        ptypes = chunk["player_type"].to_numpy().reshape(count, n)
        if np.any(pids != expected_players) or np.any(ptypes != expected_types):
            raise ValueError("native full request player/type coverage mismatch")
        chunk_ids = chunk["request_id"].to_numpy().reshape(count, n)
        chunk_times = chunk["time_step"].to_numpy().reshape(count, n)
        if np.any(chunk_ids != ids[cursor:stop, None]) or np.any(chunk_times != times[cursor:stop, None]):
            raise ValueError("six-field request order/time differs from explicit order file")
        block = chunk["pvalue"].to_numpy().reshape(count, n)
        if not np.isfinite(block).all() or np.any(block < 0):
            raise ValueError("invalid source pValue; never reinterpret invalid values as absent")
        values[cursor:stop] = block
        cursor = stop
    if cursor != requests:
        raise ValueError("truncated native source")
    for name, path in source_paths.items():
        if _source_signature(path) != signatures[name]:
            raise ValueError(f"source changed during cache construction: {path}")
    values.flush()
    mask.flush()
    np.save(destination / "request_ids.npy", ids, allow_pickle=False)
    np.save(destination / "original_order.npy", orders, allow_pickle=False)
    np.save(destination / "boundaries.npy", boundaries, allow_pickle=False)
    filenames = ("values.npy", "participation_mask.npy", "request_ids.npy", "original_order.npy", "boundaries.npy")
    files = {name: {"sha256": sha256_file(destination / name), "bytes": (destination / name).stat().st_size} for name in filenames}
    # This audit uses a selected training period only when the caller requests it.
    diagnostics = {"pvalue_min": float(values.min()), "pvalue_max": float(values.max()),
                   "pvalue_mean": float(values.mean()), "pvalue_mean_by_player": values.mean(axis=0).tolist(),
                   "positive_fraction": float(np.count_nonzero(values) / values.size),
                   "quantiles_deterministic_stride97": np.quantile(values.reshape(-1)[::97], [0, .1, .5, .9, .99, 1]).tolist(),
                   "quantile_levels": [0, .1, .5, .9, .99, 1]}
    manifest = {"cache_version": CACHE_VERSION, "complete": True, "day_id": day_id,
                "split": entry["split"], "request_count": requests, "player_order": players,
                "players": roster, "target72counts": counts.tolist(), "files": files,
                "source": {"root": str(root.resolve()), "metadata_sha256": sha256_file(source_dir / "metadata.json"),
                           "files": entry["files"], "source_rules_version": entry["rules_version"],
                           "source_schema_version": entry["converter_version"],
                           "mapping_assumption": entry["mapping_assumption"],
                           "mask_semantics": "full-native-coverage-validated-all-true; real zero remains present"},
                "diagnostics": diagnostics, "build_seconds": time.perf_counter()-start}
    manifest["fingerprint"] = fingerprint({k: manifest[k] for k in ("cache_version", "day_id", "files", "player_order", "players", "source")})
    temporary = destination / "cache.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return load_day(day_id, cache_root, source_root)


def load_day(day_id, cache_root=DEFAULT_CACHE, source_root=SOURCE_ROOT, budget_multiplier=1.0):
    """Load one verified immutable matrix day; budget scaling is explicit."""
    destination = Path(cache_root) / day_id
    manifest = json.loads((destination / "cache.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("cache_version") != CACHE_VERSION or manifest["day_id"] != day_id:
        raise ValueError("incomplete/unsupported matrix cache")
    entry = _source_entry(day_id, source_root)
    if manifest["source"]["files"] != entry["files"] or manifest["players"] != entry["players"]:
        raise ValueError("cache provenance differs from retained source manifest")
    payload = {k: manifest[k] for k in ("cache_version", "day_id", "files", "player_order", "players", "source")}
    if fingerprint(payload) != manifest["fingerprint"]:
        raise ValueError("cache metadata fingerprint mismatch")
    for name, info in manifest["files"].items():
        path = destination / name
        if path.stat().st_size != info["bytes"] or sha256_file(path) != info["sha256"]:
            raise ValueError(f"cache content integrity failure: {path}")
    values = np.load(destination / "values.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(destination / "participation_mask.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(destination / "request_ids.npy", mmap_mode="r", allow_pickle=False)
    orders = np.load(destination / "original_order.npy", mmap_mode="r", allow_pickle=False)
    bounds = np.load(destination / "boundaries.npy", allow_pickle=False)
    n = len(manifest["player_order"])
    count = manifest["request_count"]
    if (values.shape != (count, n) or values.dtype != np.dtype("float64") or masks.shape != values.shape
            or masks.dtype != np.dtype("bool") or ids.shape != (count,) or ids.dtype.kind != "U"
            or orders.shape != (count,) or orders.dtype != np.dtype("int64") or bounds.shape != (73,)
            or bounds[0] != 0 or bounds[-1] != count or np.any(np.diff(bounds) < 0)
            or np.diff(bounds).tolist() != manifest["target72counts"]):
        raise ValueError("cache array shape/type/coverage mismatch")
    if isinstance(budget_multiplier, bool) or not np.isfinite(budget_multiplier) or budget_multiplier <= 0:
        raise ValueError("budget_multiplier must be finite positive real")
    advertisers = tuple(Advertiser(p, manifest["players"][p]["player_type"],
                        round(manifest["players"][p]["budget"] * budget_multiplier, 6)) for p in manifest["player_order"])
    slots = tuple(MatrixSlot(values[b:e], masks[b:e], ids[b:e], orders[b:e]) for b, e in zip(bounds[:-1], bounds[1:]))
    source = dict(manifest)
    source["budget_multiplier"] = budget_multiplier
    digest = fingerprint({"matrix": manifest["fingerprint"], "budget_multiplier": budget_multiplier})
    return MatrixDay(day_id, advertisers, slots, digest, source)


def write_data_inventory(output_path, source_root=SOURCE_ROOT):
    """Describe inventory without reading holdout pValues or running policies."""
    manifest = source_manifest(source_root)
    rows = []
    for entry in manifest["periods"]:
        budgets = [v["budget"] for v in entry["players"].values()]
        rows.append({"day_id": entry["day_id"], "split": entry["split"], "players": len(budgets),
                     "requests": entry["request_count"], "rows": entry["row_count"],
                     "budget_min": min(budgets), "budget_max": max(budgets), "budget_sum": sum(budgets),
                     "csv_bytes": entry["files"]["six_fields"]["bytes"]})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("periods", nargs="+", help="Explicit period IDs, e.g. period-7")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    args = parser.parse_args()
    for period in args.periods:
        day = build_cache(period, args.cache_root, args.source_root)
        print(json.dumps({"day_id": day.day_id, "requests": sum(len(s.values) for s in day.slots),
                          "fingerprint": day.fingerprint, "diagnostics": day.source["diagnostics"],
                          "cache_build_seconds": day.source["build_seconds"]}, ensure_ascii=False), flush=True)
