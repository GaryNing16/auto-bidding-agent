"""Deterministic request-group augmentation and explicit budget perturbations.

Each matrix row is one complete request group. Rows (including their sparse
participation masks) are sampled together; pValues are never synthesized by
augmentation. Historical matrices remain immutable and are reused in memory.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import math

import numpy as np

from harmony_baselines.data import MatrixDay, MatrixSlot
from harmony_bidding.schema import Advertiser, fingerprint


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (float, int, np.number)):
        raise ValueError(f"invalid {name}")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"invalid {name}")
    return value


def matrix_fingerprint(day_id, advertisers, slots):
    digest = hashlib.sha256(fingerprint([day_id, [asdict(a) for a in advertisers]]).encode())
    for slot in slots:
        for array in (slot.values, slot.participation_mask, slot.original_order):
            array = np.ascontiguousarray(array)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        digest.update(fingerprint(np.asarray(slot.request_ids).tolist()).encode())
    return digest.hexdigest()


def synthetic_day(seed=0, players=48, requests_per_slot=4):
    """Explicitly synthetic sparse fixture, never a substitute for measured markets."""
    if players < 1 or requests_per_slot < 0:
        raise ValueError("invalid fixture shape")
    rng = np.random.default_rng(seed)
    advertisers = tuple(Advertiser(str(i), str(i // 8), round(.15 + i * .007, 6)) for i in range(players))
    slots = []
    for t in range(72):
        values = rng.lognormal(-6, .7, (requests_per_slot, players))
        mask = rng.random(values.shape) > .18
        values[~mask] = 0.
        ids = np.asarray([f"fixture-{t}-{j}" for j in range(requests_per_slot)], dtype=str)
        slots.append(MatrixSlot(values, mask, ids, np.arange(requests_per_slot, dtype=np.int64)))
    name = f"synthetic-{seed}-{players}-{requests_per_slot}"
    return MatrixDay(name, advertisers, tuple(slots), matrix_fingerprint(name, advertisers, slots),
                     {"kind": "synthetic_fixture", "seed": seed, "not_historical": True})


def randomize_budgets(advertisers, controlled_player, *, seed=0, mode="original",
                      low=.8, high=1.2, scale=1., traffic_multiplier=1.,
                      traffic_budget_mode="fixed"):
    """Return native heterogeneous budgets times recorded, reproducible factors.

    ``traffic_budget_mode`` distinguishes unchanged budgets from proportional
    scaling when traffic increases. No equal-budget fallback is permitted.
    """
    allowed = ("original", "controlled", "independent", "common")
    if mode not in allowed or traffic_budget_mode not in ("fixed", "proportional"):
        raise ValueError("unknown budget randomization or traffic budget mode")
    low, high = _positive(low, "low"), _positive(high, "high")
    scale = _positive(scale, "scale")
    traffic_multiplier = _positive(traffic_multiplier, "traffic_multiplier")
    if low > high:
        raise ValueError("low must not exceed high")
    ids = [a.advertiser_id for a in advertisers]
    if str(controlled_player) not in ids:
        raise ValueError("unknown controlled player")
    rng = np.random.default_rng(int(seed) + 49371)
    factors = np.ones(len(ids))
    if mode == "controlled":
        factors[ids.index(str(controlled_player))] = rng.uniform(low, high)
    elif mode == "independent":
        factors = rng.uniform(low, high, len(ids))
    elif mode == "common":
        factors[:] = rng.uniform(low, high)
    factors *= scale * (traffic_multiplier if traffic_budget_mode == "proportional" else 1.)
    result = tuple(replace(a, budget=round(a.budget * float(f), 6)) for a, f in zip(advertisers, factors))
    if any(not math.isfinite(a.budget) or a.budget < 0 for a in result):
        raise ValueError("nonfinite/negative budget")
    metadata = dict(mode=mode, low=low, high=high, scale=scale, traffic_multiplier=traffic_multiplier,
        traffic_budget_mode=traffic_budget_mode, seed=int(seed),
        original_budgets={a.advertiser_id: a.budget for a in advertisers},
        budgets={a.advertiser_id: a.budget for a in result},
        realized_multipliers={a.advertiser_id: (b.budget / a.budget if a.budget else None)
                              for a, b in zip(advertisers, result)})
    return result, metadata


def randomize_opponents(specs, seed=0, low=1., high=1., gain_low=1., gain_high=1.,
                        permute_roles=False):
    """Mild explicit perturbations of frozen calibrated parameters, training only.

    Defaults are exact frozen evaluation. Permuting specifications changes
    family assignment; per-player alpha tables stay associated with identity.
    """
    for value, name in ((low, "low"), (high, "high"), (gain_low, "gain_low"), (gain_high, "gain_high")):
        _positive(value, name)
    if low > high or gain_low > gain_high:
        raise ValueError("randomization bounds reversed")
    rng = np.random.default_rng(int(seed) + 81173)
    result, factors = {}, {}
    players = list(specs)
    assigned = list(rng.permutation(players)) if permute_roles else players
    for p, source in zip(players, assigned):
        spec = deepcopy(specs[source])
        alpha_factor = float(rng.uniform(low, high))
        gain_factor = float(rng.uniform(gain_low, gain_high))
        params = spec.get("kwargs", spec)
        for key in ("alpha", "initial_alpha"):
            if key in params:
                params[key] *= alpha_factor
        if "player_alphas" in params:
            params["player_alphas"] = {key: value * alpha_factor
                                        for key, value in params["player_alphas"].items()}
        for key in ("kp", "ki", "kd"):
            if key in params:
                params[key] *= gain_factor
        result[p] = spec
        factors[p] = dict(alpha_factor=alpha_factor, gain_factor=gain_factor, source_role=source)
    return result, dict(low=low, high=high, gain_low=gain_low, gain_high=gain_high,
                        permute_roles=bool(permute_roles), per_player=factors)


def _sample_indices(n, count, rng, method, block_size):
    if not n or not count:
        return np.empty(0, dtype=np.int64)
    if method == "within_slot":
        return rng.integers(n, size=count)
    if method == "request_blocks":
        starts = rng.integers(n, size=math.ceil(count / block_size))
        return ((starts[:, None] + np.arange(block_size)) % n).reshape(-1)[:count]
    raise ValueError("unknown group bootstrap method")


def bootstrap_market(day, seed, *, traffic_multiplier=1., slot_weights=None,
                     method="within_slot", block_size=32, temporal_block_size=6):
    """Resample complete groups within strata or contiguous time blocks.

    Time-block bootstrap preserves neighboring source slots and, at unchanged
    slot volume, their exact within-slot order. It changes temporal composition.
    All copied requests receive unique IDs, even repeated draws.
    """
    q = _positive(traffic_multiplier, "traffic_multiplier")
    if len(day.slots) != 72:
        raise ValueError("market requires exactly 72 slots")
    if type(block_size) is not int or block_size < 1 or type(temporal_block_size) is not int or temporal_block_size < 1:
        raise ValueError("block sizes must be positive integers")
    weights = np.ones(72) if slot_weights is None else np.asarray(slot_weights, dtype=float)
    if weights.shape != (72,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("slot_weights requires 72 finite nonnegative multipliers")
    rng = np.random.default_rng(seed)
    source_slots = np.arange(72)
    draw_method = method
    if method == "time_blocks":
        starts = rng.integers(72, size=math.ceil(72 / temporal_block_size))
        source_slots = ((starts[:, None] + np.arange(temporal_block_size)) % 72).reshape(-1)[:72]
        draw_method = "request_blocks"
    if draw_method not in ("within_slot", "request_blocks"):
        raise ValueError("unknown bootstrap method")
    config = dict(kind="bootstrap", parent_day_id=day.day_id, parent_fingerprint=day.fingerprint,
                  seed=int(seed), traffic_multiplier=q, slot_weights=weights.tolist(), method=method,
                  block_size=block_size, temporal_block_size=temporal_block_size,
                  source_slots=source_slots.tolist())
    tag = fingerprint(config)[:16]
    slots, lineage = [], []
    for t, source_t in enumerate(source_slots):
        source = day.slots[int(source_t)]
        n = len(source.request_ids)
        count = int(round(n * q * weights[t]))
        indices = (np.arange(n, dtype=np.int64) if method == "time_blocks" and count == n
                   else _sample_indices(n, count, rng, draw_method, block_size))
        # Entire row and mask remain paired; each output row is a distinct auction.
        ids = np.asarray([f"aug-{tag}-{t}-{k}" for k in range(len(indices))], dtype=str)
        slots.append(MatrixSlot(np.array(source.values[indices], copy=True),
            np.array(source.participation_mask[indices], copy=True), ids, np.arange(len(indices), dtype=np.int64)))
        lineage.append({"source_slot": int(source_t), "source_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                        "request_count": len(indices)})
    config["lineage"] = lineage
    name = f"{day.day_id}-bootstrap-{tag}"
    return MatrixDay(name, day.advertisers, tuple(slots), matrix_fingerprint(name, day.advertisers, slots), config)


STRESS_CASES = ("front_loaded", "back_loaded", "late_high_value", "strong_competition",
                "weak_competition", "traffic_tight", "traffic_scaled", "budget_tight", "budget_loose")


def stress_market(day, seed, case):
    if case not in STRESS_CASES:
        raise ValueError("unknown stress case")
    weights = np.ones(72)
    if case in ("front_loaded", "back_loaded"):
        weights[:36], weights[36:] = (1.6, .4) if case == "front_loaded" else (.4, 1.6)
    q = 1.5 if case in ("traffic_tight", "traffic_scaled") else 1.
    result = bootstrap_market(day, seed, traffic_multiplier=q, slot_weights=weights)
    source = {**result.source, "kind": "stress", "stress_case": case,
              "budget_scale": .7 if case == "budget_tight" else 1.3 if case == "budget_loose" else 1.,
              "traffic_budget_mode": "proportional" if case == "traffic_scaled" else "fixed",
              "opponent_scale": 1.2 if case == "strong_competition" else .8 if case == "weak_competition" else 1.}
    if case == "late_high_value":
        # Move complete groups between paired half-day slots, keeping all joint
        # values/masks intact. No independent player values are constructed.
        slots = list(result.slots)
        for early in range(36):
            late = early + 36
            a, b = slots[early], slots[late]
            values = np.concatenate([a.values, b.values])
            masks = np.concatenate([a.participation_mask, b.participation_mask])
            ids = np.concatenate([a.request_ids, b.request_ids])
            score = values.sum(axis=1) / np.maximum(masks.sum(axis=1), 1)
            ranked = np.argsort(score, kind="stable")
            for t, indices in ((early, ranked[:len(a.values)]), (late, ranked[len(a.values):])):
                slots[t] = MatrixSlot(values[indices], masks[indices], ids[indices], np.arange(len(indices), dtype=np.int64))
        result = replace(result, slots=tuple(slots))
    name = f"{day.day_id}-stress-{case}-{seed}"
    return replace(result, day_id=name, source=source,
                   fingerprint=matrix_fingerprint(name, result.advertisers, result.slots))


class MarketSampler:
    """Explicit training whitelist; no validation/test loader is consulted."""
    def __init__(self, days, seed=0, probabilities=(.6, .3, .1), allowed_day_ids=None,
                 bootstrap_config=None, stress_cases=STRESS_CASES):
        self.days = tuple(days)
        if not self.days:
            raise ValueError("at least one training market is required")
        self.allowed_day_ids = frozenset(allowed_day_ids if allowed_day_ids is not None
                                        else (d.day_id for d in self.days))
        if any(d.day_id not in self.allowed_day_ids for d in self.days):
            raise ValueError("market outside explicitly allowed training split")
        self.probabilities = np.asarray(probabilities, dtype=float)
        if (self.probabilities.shape != (3,) or not np.isfinite(self.probabilities).all()
                or (self.probabilities < 0).any() or not np.isclose(self.probabilities.sum(), 1.)):
            raise ValueError("historical/bootstrap/stress probabilities must sum to one")
        self.rng = np.random.default_rng(seed)
        self.bootstrap_config = deepcopy(bootstrap_config or {})
        self.stress_cases = tuple(stress_cases)
        if not self.stress_cases or any(c not in STRESS_CASES for c in self.stress_cases):
            raise ValueError("invalid stress cases")

    def state_dict(self):
        return deepcopy(self.rng.bit_generator.state)

    def load_state_dict(self, state):
        self.rng.bit_generator.state = deepcopy(state)

    def sample(self, seed=None):
        rng = self.rng if seed is None else np.random.default_rng(seed)
        day = self.days[int(rng.integers(len(self.days)))]
        kind = str(rng.choice(("historical", "bootstrap", "stress"), p=self.probabilities))
        generation_seed = int(rng.integers(0, 2**32 - 1))
        if kind == "bootstrap":
            result = bootstrap_market(day, generation_seed, **self.bootstrap_config)
        elif kind == "stress":
            result = stress_market(day, generation_seed, str(rng.choice(self.stress_cases)))
        else:
            result = day
        metadata = dict(kind=kind, parent_day_id=day.day_id, parent_fingerprint=day.fingerprint,
            generation_seed=generation_seed, generated_fingerprint=result.fingerprint,
            parameters=deepcopy(result.source) if kind != "historical" else {},
            request_count=sum(len(s.request_ids) for s in result.slots),
            traffic_multiplier=result.source.get("traffic_multiplier", 1.) if kind != "historical" else 1.,
            traffic_budget_mode=result.source.get("traffic_budget_mode", "fixed") if kind != "historical" else "fixed",
            budget_scale=result.source.get("budget_scale", 1.) if kind != "historical" else 1.,
            opponent_scale=result.source.get("opponent_scale", 1.) if kind != "historical" else 1.)
        return result, metadata


def distribution_summary(day):
    """Evaluator-only exact descriptive statistics for an augmentation audit."""
    width = len(day.advertisers)
    opportunity_counts = np.zeros(width, dtype=np.int64)
    cooccurrence = np.zeros((width, width), dtype=np.int64)
    participation = np.zeros(width + 1, dtype=np.int64)
    value_parts, counts = [], []
    for slot in day.slots:
        mask = slot.participation_mask
        counts.append(len(slot.request_ids))
        opportunity_counts += mask.sum(axis=0)
        numeric = mask.astype(np.int64)
        cooccurrence += numeric.T @ numeric
        participation += np.bincount(mask.sum(axis=1), minlength=width + 1)
        value_parts.append(slot.values[mask])
    values = np.concatenate(value_parts)
    total = sum(counts)
    return dict(day_id=day.day_id, fingerprint=day.fingerprint, request_count=total,
        per_slot_request_counts=counts,
        per_player_opportunities={a.advertiser_id: int(opportunity_counts[i]) for i, a in enumerate(day.advertisers)},
        pvalue_mean=float(values.mean()) if len(values) else None,
        pvalue_quantiles=dict(zip(("0", "0.1", "0.5", "0.9", "0.99", "1"),
                                 np.quantile(values, (0, .1, .5, .9, .99, 1)).tolist())) if len(values) else {},
        participants_per_request_histogram=participation.tolist(), competition_cooccurrence=cooccurrence.tolist(),
        total_budget=sum(a.budget for a in day.advertisers),
        budget_per_request=sum(a.budget for a in day.advertisers) / total if total else None)
