"""Train-only time/type supply forecasts and whole-episode split manifests.

The labels are eligible opportunities and potential value, NOT wins or reward.
Only time_step and the known player_type are used at prediction time.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable

from .data import load_requests
from .types import AuctionConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_identity(path: str | Path) -> dict:
    """Identify content and, when supplied, the original source period."""
    source = Path(path).resolve()
    content_hash = _sha256(source)
    identity = {"path": str(source), "sha256": content_hash,
                "source_group": "sha256:" + content_hash}
    sidecar = source.with_suffix(".manifest.json")
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("output_sha256") not in (None, content_hash):
            raise ValueError(f"stale manifest checksum for {source.name}")
        period = metadata.get("source_delivery_period")
        if period is not None:
            # Group the source day even when its file/shard hash or conversion
            # settings differ. Generated independent days must name a dataset ID.
            dataset = metadata.get("source_dataset_id", metadata.get("upstream_repository"))
            if dataset is None:
                dataset = metadata.get("source_sha256")
            if dataset is not None:
                identity["source_group"] = json.dumps([str(dataset), str(period)], separators=(",", ":"))
    return identity


def _identities(paths: Iterable[str | Path]) -> list[dict]:
    result = [episode_identity(path) for path in paths]
    if not result:
        raise ValueError("at least one complete episode is required")
    for field in ("path", "sha256"):
        if len({entry[field] for entry in result}) != len(result):
            raise ValueError(f"duplicate episode {field}; do not count a day twice")
    return result


def _targets(path: str | Path, config: AuctionConfig) -> dict[str, list[list[float]]]:
    # Include all configured advertisers and all time slots, including zero supply.
    targets = {pid: [[0.0, 0.0] for _ in range(config.steps)] for pid in config.budgets}
    for request in load_requests(path, steps=config.steps):
        for participant in request.participants:
            if participant.player_id not in targets:
                raise ValueError(f"player {participant.player_id!r} has no configuration")
            if config.player_types[participant.player_id] != participant.player_type:
                raise ValueError(f"player_type mismatch for {participant.player_id!r}")
            target = targets[participant.player_id][request.time_step]
            target[0] += 1
            target[1] += participant.pvalue
    return targets


class SupplyForecast:
    """Same-slot/type training mean; safe JSON persistence, no online hidden data."""

    def __init__(self, model: dict):
        if model.get("schema_version") != 1 or model.get("kind") != "time_type_supply_mean":
            raise ValueError("unsupported supply forecast model")
        steps = model.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("invalid forecast steps")
        for series in [model["global_mean"], *model["type_means"].values()]:
            if len(series) != steps:
                raise ValueError("forecast shape mismatch")
            for pair in series:
                if len(pair) != 2 or any(not math.isfinite(x) or x < 0 for x in pair):
                    raise ValueError("invalid forecast value")
        if not model["training_episodes"]:
            raise ValueError("forecast must retain training provenance")
        self.model = model
        self.steps = steps

    @classmethod
    def load(cls, path: str | Path) -> "SupplyForecast":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict(self, time_step: int, player_type: str) -> dict:
        if isinstance(time_step, bool) or not isinstance(time_step, int) or not 0 <= time_step < self.steps:
            raise ValueError("time_step outside trained day")
        fallback = player_type not in self.model["type_means"]
        series = self.model["global_mean"] if fallback else self.model["type_means"][player_type]
        count, value = series[time_step]
        return {"opportunity_count": count, "potential_value": value, "used_fallback": fallback}


def fit_forecast(paths: Iterable[str | Path], config: AuctionConfig, output: str | Path) -> dict:
    """Fit on entire training days only; equal weight per day and advertiser."""
    episodes = _identities(paths)
    if len({entry["source_group"] for entry in episodes}) != len(episodes):
        raise ValueError("multiple variants of one source period cannot be independent training days")
    sums = {kind: [[0.0, 0.0] for _ in range(config.steps)] for kind in set(config.player_types.values())}
    counts = {kind: 0 for kind in sums}
    overall = [[0.0, 0.0] for _ in range(config.steps)]
    overall_count = 0
    for entry in episodes:
        targets = _targets(entry["path"], config)
        for pid, series in targets.items():
            kind = config.player_types[pid]
            counts[kind] += 1
            overall_count += 1
            for step, pair in enumerate(series):
                for index in range(2):
                    sums[kind][step][index] += pair[index]
                    overall[step][index] += pair[index]
    model = {"schema_version": 1, "kind": "time_type_supply_mean", "steps": config.steps,
             "features": ["time_step", "player_type"],
             "labels": ["eligible_request_count", "potential_pvalue_sum"],
             "type_means": {kind: [[x / counts[kind] for x in pair] for pair in series]
                            for kind, series in sums.items()},
             "global_mean": [[x / overall_count for x in pair] for pair in overall],
             "unknown_type": "training-only per-time global advertiser mean",
             "training_episodes": episodes}
    SupplyForecast(model)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return model


def evaluate_forecast(model_path: str | Path, paths: Iterable[str | Path], config: AuctionConfig) -> dict:
    """Evaluate held-out full days; refuse train/content/source-period overlap."""
    forecast = SupplyForecast.load(model_path)
    if forecast.steps != config.steps:
        raise ValueError("forecast and evaluation day lengths differ")
    episodes = _identities(paths)
    if len({entry["source_group"] for entry in episodes}) != len(episodes):
        raise ValueError("evaluation contains repeated source periods")
    training = forecast.model["training_episodes"]
    for field in ("path", "sha256", "source_group"):
        if {entry[field] for entry in training} & {entry[field] for entry in episodes}:
            raise ValueError(f"training/evaluation overlap by {field}")
    errors = [0.0, 0.0]
    samples = fallback_count = 0
    for entry in episodes:
        for pid, series in _targets(entry["path"], config).items():
            for step, target in enumerate(series):
                prediction = forecast.predict(step, config.player_types[pid])
                errors[0] += abs(prediction["opportunity_count"] - target[0])
                errors[1] += abs(prediction["potential_value"] - target[1])
                samples += 1
                fallback_count += int(prediction["used_fallback"])
    return {"kind": "heldout_supply_forecast", "opportunity_count_mae": errors[0] / samples,
            "potential_value_mae": errors[1] / samples, "player_time_samples": samples,
            "fallback_samples": fallback_count, "evaluation_episodes": episodes,
            "note": "Potential supply is not won value; these errors do not measure policy performance."}


def split_by_episode(paths: Iterable[str | Path], *, seed: int = 0,
                     validation_fraction: float = 0.2, test_fraction: float = 0.2) -> dict:
    """Split complete source-period groups, never rows; return a JSON manifest.

    Sidecars identify known source periods. Files without provenance use content
    identity; users must not disguise shards of one day as independent episodes.
    """
    for fraction in (validation_fraction, test_fraction):
        if not math.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError("split fractions must be finite in [0,1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("training must retain a positive fraction")
    entries = _identities(paths)
    groups = {}
    for entry in entries:
        groups.setdefault(entry["source_group"], []).append(entry)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    active = int(validation_fraction > 0) + int(test_fraction > 0)
    if len(keys) < active + 1:
        raise ValueError("not enough independent source periods for requested nonempty splits")
    nval = max(1, int(len(keys) * validation_fraction)) if validation_fraction else 0
    ntest = max(1, int(len(keys) * test_fraction)) if test_fraction else 0
    # Fractions are floor-rounded at group level; reserve at least one train day.
    while nval + ntest >= len(keys):
        if nval >= ntest and nval > 1:
            nval -= 1
        elif ntest > 1:
            ntest -= 1
        else:
            raise ValueError("split cannot reserve a training period")
    partitions = {"validation": keys[:nval], "test": keys[nval:nval + ntest],
                  "train": keys[nval + ntest:]}
    return {"schema_version": 1, "seed": seed,
            "fractions": {"validation": validation_fraction, "test": test_fraction},
            "split_unit": "complete source period; content identity without a sidecar",
            "splits": {name: [entry for key in selection for entry in groups[key]]
                       for name, selection in partitions.items()}}


def write_split_manifest(manifest: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
