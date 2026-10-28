"""Sequential, single-slot second-price reference engine; no CPA/ROI penalty."""
from decimal import Decimal, ROUND_HALF_UP, localcontext
from hashlib import sha256
import json
import math
from numbers import Real
import random
from types import MappingProxyType
from .budget_policy import ExcludeUnaffordablePolicy
from .schema import Opportunity, Rules, nonnegative, validate_values_mask


class Money:
    def __init__(self, decimals=6):
        self.scale = 10 ** decimals

    def units(self, value):
        nonnegative(value, "money")
        with localcontext() as ctx:
            ctx.prec = 400
            return int((Decimal(str(value)) * self.scale).to_integral_value(rounding=ROUND_HALF_UP))

    def bid_units(self, alpha, value):
        alpha = validate_alpha(alpha)
        value = nonnegative(value, "estimated_value")
        # This local protocol uses finite binary64 business inputs/outputs,
        # while monetary accounting and rounding use exact integer units.
        if not math.isfinite(alpha * value):
            raise ValueError("bid product overflow: alpha * estimated_value must be finite")
        with localcontext() as ctx:
            ctx.prec = 700
            return int((Decimal(str(alpha)) * Decimal(str(value)) * self.scale)
                       .to_integral_value(rounding=ROUND_HALF_UP))

    def budget_units(self, value):
        units = self.units(value)
        with localcontext() as ctx:
            ctx.prec = 400
            if Decimal(str(value)) * self.scale != units:
                raise ValueError("budget must be exactly representable at configured money precision")
        return units

    def amount(self, units):
        return units / self.scale


def validate_alpha(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"illegal action: expected real scalar, got {value!r}")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("illegal action: scalar cannot be represented as a finite float") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"illegal action {value!r}; expected finite nonnegative real scalar")
    return value


def choose_request_tie(seed, day_id, time_slot, opportunity_id, tied_ids, round_index):
    """Local seeded uniform draw, independent of input order and global RNG.

    randrange uses rejection sampling, avoiding hash-modulo bias. A fresh
    request/reauction-specific generator prevents unrelated calls from changing
    the draw. The Python runtime and rule version are recorded by the runner.
    """
    candidates = tuple(sorted(tied_ids))
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("tie candidates must be nonempty and unique")
    payload = json.dumps([seed, day_id, time_slot, opportunity_id, round_index, candidates],
                         ensure_ascii=False, separators=(",", ":"))
    request_seed = int.from_bytes(sha256(payload.encode("utf-8")).digest(), "big")
    return candidates[random.Random(request_seed).randrange(len(candidates))]


class AuctionEngine:
    def __init__(self, advertisers, rules: Rules, budget_policy=None):
        self.rules = rules
        self.advertisers = tuple(advertisers)
        self.ids = tuple(a.advertiser_id for a in self.advertisers)
        if (not self.ids or any(not isinstance(a, str) or not a.strip() or a != a.strip() for a in self.ids)
                or len(set(self.ids)) != len(self.ids)):
            raise ValueError("advertisers must have nonempty unique identities")
        self.money = Money(rules.money_decimals)
        self.initial = {a.advertiser_id: self.money.budget_units(a.budget) for a in self.advertisers}
        self.remaining = dict(self.initial)
        self.cumulative_value = dict.fromkeys(self.ids, 0.0)
        self.policy = budget_policy or ExcludeUnaffordablePolicy(rules.zero_balance_can_bid)

    def run_matrix_slot(self, values, participation_mask, alphas, *, day_id, time_slot,
                        opportunity_ids, original_order, record_debug=False):
        """Settle a matrix through this same engine and monetary/budget policies.

        This evaluator-only input avoids creating millions of Python records.
        Bids are rounded in bulk only where a conservative binary64 error band
        excludes a half-unit boundary; boundary cases use ``Money.bid_units``.
        Sequential spending is batched up to the first affordability/depletion
        event, then the existing budget policy settles the event. Large integer
        amounts and custom budget policies use the ordinary reference method.
        Nothing is committed until the complete slot has been validated.
        """
        import numpy as np

        if set(alphas) != set(self.ids):
            raise ValueError("actions must match all advertiser identities")
        actions = {a: validate_alpha(alphas[a]) for a in self.ids}
        if (not isinstance(day_id, str) or not day_id or day_id != day_id.strip()
                or type(time_slot) is not int or not 0 <= time_slot < self.rules.total_slots):
            raise ValueError("invalid request day or time_slot")
        raw = np.asarray(values)
        mask = np.asarray(participation_mask)
        if raw.ndim != 2 or raw.shape[1] != len(self.ids) or raw.dtype.kind not in "fiu":
            raise ValueError("values must be a real numeric request-by-advertiser matrix")
        matrix = np.asarray(raw, dtype=np.float64)
        if mask.shape != matrix.shape or mask.dtype != np.bool_:
            raise ValueError("participation_mask must be an explicit matching bool matrix")
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("estimated values must be finite and nonnegative")
        all_present = bool(mask.all())
        if not all_present and (matrix[~mask] != 0).any():
            raise ValueError("absent participant must have a zero padded value")
        count, width = matrix.shape
        # np.memmap iteration invokes its Python __getitem__ once per row;
        # tolist on the base ndarray avoids that overhead and yields str IDs.
        request_ids = (np.asarray(opportunity_ids).tolist() if isinstance(opportunity_ids, np.ndarray)
                       else tuple(opportunity_ids))
        if (len(request_ids) != count or len(set(request_ids)) != count
                or any(not isinstance(x, str) or not x or x != x.strip() for x in request_ids)):
            raise ValueError("request identities must be nonempty and unique within a slot")
        order = np.asarray(original_order)
        if (order.ndim != 1 or len(order) != count or order.dtype.kind not in "iu"
                or (order < 0).any() or (order[1:] <= order[:-1]).any()):
            raise ValueError("opportunities must follow strictly increasing original_order")

        def reference():
            opportunities = (Opportunity(day_id, time_slot, request_ids[k], int(order[k]),
                               tuple(float(v) for v in matrix[k]),
                               tuple(bool(v) for v in mask[k])) for k in range(count))
            stats, debug = self.run_slot(opportunities, actions)
            return stats, debug if record_debug else []

        if type(self.policy) is not ExcludeUnaffordablePolicy:
            return reference()
        # A decimal product differs from this two-operation binary product by
        # fewer than eight ulps; use 16 eps conservatively, including subnormals.
        with np.errstate(over="ignore", invalid="ignore"):
            product = matrix * np.asarray([actions[a] for a in self.ids])[None, :]
            scaled = product * self.money.scale
        if not np.isfinite(product if all_present else product[mask]).all():
            raise ValueError("bid product overflow: alpha * estimated_value must be finite")
        safe_max = (1 << 50)
        checked_scaled = scaled if all_present else scaled[mask]
        if (not np.isfinite(checked_scaled).all() or (checked_scaled >= safe_max).any()
                or any(self.initial[a] >= safe_max for a in self.ids)):
            return reference()
        if not all_present:
            scaled = np.where(mask, scaled, 0.0)
        floor = np.floor(scaled)
        uncertainty = 16 * np.finfo(np.float64).eps * np.maximum(scaled, 1.0)
        boundary = mask & (np.abs(scaled - floor - 0.5) <= uncertainty)
        bids = np.floor(scaled + 0.5).astype(np.int64)
        for row, column in zip(*np.nonzero(boundary)):
            bids[row, column] = self.money.bid_units(actions[self.ids[column]], float(matrix[row, column]))
        if count and int(bids.max(initial=0)) > np.iinfo(np.int64).max // max(count, 1):
            return reference()
        remaining = np.asarray([self.remaining[a] for a in self.ids], dtype=np.int64)
        cumulative = np.asarray([self.cumulative_value[a] for a in self.ids], dtype=np.float64)
        costs = np.zeros(width, dtype=np.int64)
        wins = np.zeros(width, dtype=np.int64)
        earned = np.zeros(width, dtype=np.float64)
        debug = []
        allow_zero = self.policy.zero_balance_can_bid
        cursor, events = 0, 0
        active_key = top_key = None
        active_cache = top_winners_cache = top_prices_cache = None
        ranking_key = None
        ranked_cache = ranked_bids_cache = next_prices_cache = ranked_ties_cache = None

        def commit_prefix(indices, winners, prices, exclusions=None):
            valid = winners >= 0
            chosen = winners[valid]
            row_indices = indices[valid]
            payments = prices[valid]
            # add.at deliberately preserves request-order float additions, just
            # like run_slot; a parallel sum would change value rounding.
            np.add.at(remaining, chosen, -payments)
            np.add.at(costs, chosen, payments)
            np.add.at(wins, chosen, 1)
            with np.errstate(over="ignore"):
                np.add.at(earned, chosen, matrix[row_indices, chosen])
                np.add.at(cumulative, chosen, matrix[row_indices, chosen])
            if record_debug:
                for offset, (index, winner, price) in enumerate(zip(indices, winners, prices)):
                    debug.append({"opportunity_id": request_ids[index],
                                  "winner": self.ids[winner] if winner >= 0 else None,
                                  "payment_units": int(price),
                                  "excluded": exclusions[offset] if exclusions is not None else ()})

        def settle_one(index):
            bid_map = {a: int(bids[index, j]) for j, a in enumerate(self.ids) if mask[index, j]}
            remaining_map = {a: int(remaining[j]) for j, a in enumerate(self.ids)}
            settlement = self.policy.settle(MappingProxyType(bid_map), MappingProxyType(remaining_map),
                lambda tied, round_index: choose_request_tie(self.rules.tie_seed, day_id, time_slot,
                                                             request_ids[index], tied, round_index))
            if settlement.winner is not None:
                j = self.ids.index(settlement.winner)
                payment = settlement.payment_units
                if payment < 0 or payment > remaining[j]:
                    raise AssertionError("budget policy violated payment bounds")
                remaining[j] -= payment
                costs[j] += payment
                wins[j] += 1
                earned[j] += matrix[index, j]
                cumulative[j] += matrix[index, j]
            if record_debug:
                debug.append({"opportunity_id": request_ids[index], "winner": settlement.winner,
                              "payment_units": settlement.payment_units, "excluded": settlement.excluded})

        while cursor < count:
            if count - cursor < 64:
                for index in range(cursor, count):
                    settle_one(index)
                break
            eligible = np.ones(width, dtype=bool) if allow_zero else remaining > 0
            current_key = eligible.tobytes()
            if current_key != active_key:
                active_cache = np.where(eligible[None, :], bids, 0)
                active_key = current_key
            exclusions = None
            if events < max(width // 2, 1):
                # Bid ordering changes only when a zero balance removes an
                # advertiser. Smaller positive balances change affordability,
                # handled below, but not these candidate/second-price arrays.
                if current_key != top_key:
                    top_winners_cache = np.argmax(active_cache, axis=1)
                    rows = np.arange(count)
                    top = active_cache[rows, top_winners_cache]
                    if width > 1:
                        # Temporarily remove one highest bidder. A second equal
                        # highest remains, so second price and ties stay exact.
                        active_cache[rows, top_winners_cache] = 0
                        top_prices_cache = active_cache.max(axis=1)
                        active_cache[rows, top_winners_cache] = top
                    else:
                        top_prices_cache = np.zeros(count, dtype=np.int64)
                    top_winners_cache[top == 0] = -1
                    ties = (np.flatnonzero((top > 0) & (top == top_prices_cache))
                            if width > 1 else np.empty(0, dtype=int))
                    # A draw is a pure request-keyed function, so calculating it
                    # once while this eligibility set persists changes no RNG.
                    for index in ties:
                        tied = tuple(self.ids[j] for j in np.flatnonzero(active_cache[index] == top[index]))
                        winner = choose_request_tie(self.rules.tie_seed, day_id, time_slot,
                                                   request_ids[index], tied, 0)
                        top_winners_cache[index] = self.ids.index(winner)
                    top_key = current_key
                winners = top_winners_cache[cursor:]
                prices = top_prices_cache[cursor:]
            else:
                # In an exhausted market many top bidders are already unable
                # to pay. Predict their per-request exclusions at the current
                # balances in bulk, then still stop at the FIRST balance event.
                # An excluded candidate can only remain excluded as spending
                # proceeds. Tied rankings use the existing exact budget policy.
                if current_key != ranking_key:
                    ranked_cache = np.argsort(-active_cache, axis=1, kind="stable")
                    ranked_bids_cache = np.take_along_axis(active_cache, ranked_cache, axis=1)
                    next_prices_cache = np.zeros_like(ranked_bids_cache)
                    next_prices_cache[:, :-1] = ranked_bids_cache[:, 1:]
                    ranked_ties_cache = np.flatnonzero(((ranked_bids_cache[:, :-1] > 0)
                        & (ranked_bids_cache[:, :-1] == ranked_bids_cache[:, 1:])).any(axis=1))
                    ranking_key = current_key
                ranked = ranked_cache[cursor:]
                ranked_bids = ranked_bids_cache[cursor:]
                next_prices = next_prices_cache[cursor:]
                affordable = (ranked_bids > 0) & (remaining[ranked] >= next_prices)
                rank = np.argmax(affordable, axis=1)
                row = np.arange(count - cursor)
                winners = ranked[row, rank]
                prices = next_prices[row, rank]
                winners[~affordable.any(axis=1)] = -1
                if record_debug:
                    exclusions = [tuple(self.ids[j] for j in ranked[k, :rank[k]])
                                  if winners[k] >= 0 else () for k in row]
                ties = ranked_ties_cache[ranked_ties_cache >= cursor] - cursor
                remaining_map = MappingProxyType({a: int(remaining[j]) for j, a in enumerate(self.ids)})
                for index in ties:
                    bid_map = MappingProxyType({a: int(bids[cursor + index, j])
                                                for j, a in enumerate(self.ids) if mask[cursor + index, j]})
                    settlement = self.policy.settle(bid_map, remaining_map,
                        lambda tied, round_index: choose_request_tie(self.rules.tie_seed, day_id, time_slot,
                                                       request_ids[cursor + index], tied, round_index))
                    winners[index] = self.ids.index(settlement.winner) if settlement.winner is not None else -1
                    prices[index] = settlement.payment_units
                    if record_debug:
                        exclusions[index] = settlement.excluded
            event_index = len(winners)
            include_event = False
            for j in range(width):
                indices = np.flatnonzero(winners == j)
                if not len(indices):
                    continue
                accumulated = np.cumsum(prices[indices], dtype=np.int64)
                # Equality must end a batch too: a depleted advertiser stops
                # affecting every later request, including its second price.
                event = accumulated > remaining[j]
                if not allow_zero and remaining[j] > 0:
                    event |= accumulated == remaining[j]
                positions = np.flatnonzero(event)
                if len(positions):
                    position = int(positions[0])
                    candidate = int(indices[position])
                    if candidate < event_index:
                        event_index = candidate
                        include_event = bool(accumulated[position] == remaining[j])
            prefix = event_index + (1 if include_event else 0)
            if prefix:
                commit_prefix(np.arange(cursor, cursor + prefix), winners[:prefix], prices[:prefix], exclusions)
                cursor += prefix
            if event_index == len(winners):
                break
            if not include_event:
                settle_one(cursor)
                cursor += 1
            events += 1
        if ((remaining < 0).any()
                or any(int(remaining[j]) > self.initial[a] for j, a in enumerate(self.ids))):
            raise AssertionError("remaining budget invariant failed")
        if not np.isfinite(cumulative).all() or not np.isfinite(earned).all():
            raise ValueError("cumulative value overflow")
        opportunities_per_player = np.full(width, count) if all_present else mask.sum(axis=0)
        stats = {a: {"wins": int(wins[j]), "cost_units": int(costs[j]), "value": float(earned[j]),
                     "opportunity_count": int(opportunities_per_player[j])} for j, a in enumerate(self.ids)}
        self.remaining = {a: int(remaining[j]) for j, a in enumerate(self.ids)}
        self.cumulative_value = {a: float(cumulative[j]) for j, a in enumerate(self.ids)}
        return stats, debug

    def run_slot(self, opportunities, alphas):
        """Settle sequentially on working ledgers; commit only a valid full slot.

        Failed requests/actions leave authoritative balances and values intact.
        This transaction does not sandbox arbitrary policy/Agent side effects.
        """
        if set(alphas) != set(self.ids):
            raise ValueError("actions must match all advertiser identities")
        alphas = {a: validate_alpha(alphas[a]) for a in self.ids}
        stats = {a: {"wins": 0, "cost_units": 0, "value": 0.0, "opportunity_count": 0} for a in self.ids}
        remaining_work = dict(self.remaining)
        value_work = dict(self.cumulative_value)
        debug = []
        previous_order = -1
        request_ids = set()
        day_slot = None
        for opportunity in opportunities:
            if not isinstance(opportunity, Opportunity):
                raise ValueError("run_slot requires validated Opportunity objects")
            if (not isinstance(opportunity.opportunity_id, str) or not opportunity.opportunity_id
                    or opportunity.opportunity_id != opportunity.opportunity_id.strip()
                    or opportunity.opportunity_id in request_ids):
                raise ValueError("request identities must be nonempty and unique within a slot")
            request_ids.add(opportunity.opportunity_id)
            if (not isinstance(opportunity.day_id, str) or not opportunity.day_id
                    or opportunity.day_id != opportunity.day_id.strip()
                    or type(opportunity.time_slot) is not int
                    or not 0 <= opportunity.time_slot < self.rules.total_slots):
                raise ValueError("invalid request day or time_slot")
            identity = (opportunity.day_id, opportunity.time_slot)
            if day_slot is not None and identity != day_slot:
                raise ValueError("run_slot must contain one day/time_slot only")
            day_slot = identity
            if type(opportunity.original_order) is not int or opportunity.original_order < 0:
                raise ValueError("original_order must be nonnegative integer")
            if opportunity.original_order <= previous_order:
                raise ValueError("opportunities must follow strictly increasing original_order")
            previous_order = opportunity.original_order
            validate_values_mask(opportunity, len(self.ids))
            bids = {}
            for i, a in enumerate(self.ids):
                if opportunity.participation_mask[i]:
                    stats[a]["opportunity_count"] += 1
                    try:
                        bids[a] = self.money.bid_units(alphas[a], opportunity.values[i])
                    except ValueError as error:
                        raise ValueError(f"request {opportunity.opportunity_id}, advertiser {a}: {error}") from error
            settlement = self.policy.settle(MappingProxyType(bids), MappingProxyType(remaining_work),
                lambda tied, round_index: choose_request_tie(
                    self.rules.tie_seed, opportunity.day_id, opportunity.time_slot,
                    opportunity.opportunity_id, tied, round_index))
            if settlement.winner is not None:
                a = settlement.winner
                payment = settlement.payment_units
                if a not in bids or bids[a] <= 0:
                    raise AssertionError("budget policy selected an ineligible winner")
                if type(payment) is not int or payment < 0 or payment > remaining_work[a]:
                    raise AssertionError("budget policy violated payment bounds")
                remaining_work[a] -= payment
                value = float(opportunity.values[self.ids.index(a)])
                stats[a]["wins"] += 1
                stats[a]["cost_units"] += payment
                stats[a]["value"] += value
                value_work[a] += value
                if not math.isfinite(value_work[a]) or not math.isfinite(stats[a]["value"]):
                    raise ValueError("cumulative value overflow")
            elif settlement.payment_units != 0:
                raise AssertionError("no-winner settlement must have zero payment")
            debug.append({"opportunity_id": opportunity.opportunity_id,
                          "winner": settlement.winner, "payment_units": settlement.payment_units,
                          "excluded": settlement.excluded})
            if any(not 0 <= remaining_work[a] <= self.initial[a] for a in self.ids):
                raise AssertionError("remaining budget invariant failed")
        self.remaining = remaining_work
        self.cumulative_value = value_work
        return stats, debug
