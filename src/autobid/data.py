"""Strict six-column request data, without filling absent advertiser rows.

One CSV represents one complete episode. Budgets and the episode's identity live
in configuration/manifest files, not in extra competition columns.
"""
from __future__ import annotations

import csv
import math
import random
import re
from pathlib import Path
from typing import Iterable

from .types import Participation, Request

FIELDS = ("index", "time_step", "request_id", "player_id", "player_type", "pvalue")


def _integer(value: str, field: str, line: int) -> int:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError(f"line {line}: {field} must be a nonnegative integer, got {value!r}")
    return int(value)


def _text(value: str, field: str, line: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"line {line}: {field} must be nonempty without surrounding whitespace")
    return value


def _value(value: str, field: str, line: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line}: {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"line {line}: {field} must be finite and nonnegative")
    return result


def load_requests(path: str | Path, steps: int = 72) -> list[Request]:
    """Load a day, ordered by time slot then minimum numeric row index.

Rows may be grouped by advertiser in the input. Request membership is determined
by request_id, never adjacency. Duplicate indices/pairs, cross-slot request IDs,
type changes, missing values and non-finite values are errors. Zero is a valid
pvalue; absent rows are never synthesized. This loader holds a day in memory;
use a complete-request subset of large public datasets during initial work.
"""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    groups: dict[str, tuple[int, int, list[tuple[int, Participation]]]] = {}
    seen_indices: set[int] = set()
    seen_pairs: set[tuple[str, str]] = set()
    player_types: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(FIELDS) or set(reader.fieldnames) != set(FIELDS):
            raise ValueError(f"CSV must have exactly these six columns: {', '.join(FIELDS)}")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"line {line}: column count does not match header")
            index = _integer(row["index"], "index", line)
            step = _integer(row["time_step"], "time_step", line)
            if step >= steps:
                raise ValueError(f"line {line}: time_step {step} outside [0, {steps - 1}]")
            rid = _text(row["request_id"], "request_id", line)
            pid = _text(row["player_id"], "player_id", line)
            ptype = _text(row["player_type"], "player_type", line)
            val = _value(row["pvalue"], "pvalue", line)
            if index in seen_indices:
                raise ValueError(f"line {line}: duplicate index {index}")
            if (rid, pid) in seen_pairs:
                raise ValueError(f"line {line}: duplicate (request_id, player_id) {(rid, pid)!r}")
            if pid in player_types and player_types[pid] != ptype:
                raise ValueError(f"line {line}: player {pid!r} changes player_type")
            seen_indices.add(index)
            seen_pairs.add((rid, pid))
            player_types[pid] = ptype
            if rid not in groups:
                groups[rid] = (step, index, [])
            old_step, first_index, participants = groups[rid]
            if old_step != step:
                raise ValueError(f"line {line}: request {rid!r} occurs in multiple time slots")
            participants.append((index, Participation(pid, ptype, val)))
            groups[rid] = (step, min(index, first_index), participants)
    result = []
    for rid, (step, _, participants) in sorted(groups.items(), key=lambda item: (item[1][0], item[1][1])):
        result.append(Request(step, rid, tuple(p for _, p in sorted(participants, key=lambda x: x[0]))))
    return result


def write_requests(requests: Iterable[Request], path: str | Path) -> None:
    """Write six canonical fields; new indices preserve supplied request order."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        index = 0
        for req in requests:
            for participant in req.participants:
                writer.writerow({"index": index, "time_step": req.time_step,
                                 "request_id": req.request_id, "player_id": participant.player_id,
                                 "player_type": participant.player_type, "pvalue": repr(participant.pvalue)})
                index += 1


def synthetic_requests(seed: int = 0, steps: int = 72, players: int = 3,
                       requests_per_step: int = 12) -> list[Request]:
    """Small reproducible sparse fixture with participant-specific values.

    This is deliberately synthetic, not an imitation of real market statistics.
    Player IDs are "0", "1", ...; types alternate "0" and "1".
    """
    for name, value in (("steps", steps), ("players", players), ("requests_per_step", requests_per_step)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    rng = random.Random(seed)
    ids = [str(i) for i in range(players)]
    result = []
    for step in range(steps):
        # Deterministic daily shape with seed-dependent request counts.
        scale = 0.55 + 0.9 * (1 + math.sin(2 * math.pi * (step / steps - 0.25))) / 2
        count = max(1, round(requests_per_step * scale + rng.uniform(-1, 1)))
        for j in range(count):
            included = [i for i in range(players) if rng.random() < 0.72]
            if not included:
                included = [rng.randrange(players)]
            participants = tuple(Participation(ids[i], str(i % 2),
                                               round(rng.lognormvariate(-0.3 + 0.12 * i, 0.55), 6))
                                 for i in included)
            result.append(Request(step, f"synthetic-{seed}-{step}-{j}", participants))
    return result


def summarize_requests(requests: Iterable[Request], steps: int = 72) -> dict:
    """Environment-side supply statistics; never pass these to an online agent."""
    requests = list(requests)
    by_player: dict[str, dict] = {}
    slot_counts = [0] * steps
    rows = zero_rows = 0
    for req in requests:
        if not 0 <= req.time_step < steps:
            raise ValueError("request time_step outside configured steps")
        slot_counts[req.time_step] += 1
        for p in req.participants:
            item = by_player.setdefault(p.player_id, {"player_type": p.player_type, "opportunities": 0,
                                                      "potential_value": 0.0, "opportunities_by_step": [0] * steps})
            item["opportunities"] += 1
            item["potential_value"] += p.pvalue
            item["opportunities_by_step"][req.time_step] += 1
            rows += 1
            zero_rows += p.pvalue == 0
    return {"requests": len(requests), "rows": rows, "players": by_player,
            "requests_by_step": slot_counts, "zero_value_rows": zero_rows,
            "missing_pairs": len(requests) * len(by_player) - rows,
            "missing_pairs_semantics": "not participating; never zero-filled"}
