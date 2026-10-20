"""Request-by-request local competition simulator.

This module deliberately replaces AuctionNet's native multi-slot/CPA auction.
Budget protection is a configurable local assumption, not the official referee:
``cap_bid`` clips bids to current balances; ``reject_unaffordable`` removes an
unaffordable winner and repeats the auction. No rounding is assumed.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import blake2b
import json
import math
from numbers import Real

from .types import AuctionConfig, Feedback, Observation, Request


class DaySimulator:
    """One finite day; a call to :meth:`step` executes exactly one time slice."""

    def __init__(self, requests: list[Request], config: AuctionConfig, *, audit: bool = False):
        self.config = deepcopy(config)
        self._validate_config()
        self.time_step = 0
        self.balances = {p: float(b) for p, b in self.config.budgets.items()}
        self.histories: dict[str, list[Feedback]] = {p: [] for p in self.balances}
        self.audit: list[dict] = []
        self._audit_enabled = audit
        self._slots: list[list[Request]] = [[] for _ in range(self.config.steps)]
        seen_requests: set[str] = set()
        previous_step = -1
        for request in requests:
            if not isinstance(request.time_step, int) or isinstance(request.time_step, bool):
                raise ValueError("request time_step must be an integer")
            if not 0 <= request.time_step < self.config.steps:
                raise ValueError("request time_step is outside the configured day")
            if request.time_step < previous_step:
                raise ValueError("requests must be ordered by time_step and source request order")
            previous_step = request.time_step
            if not isinstance(request.request_id, str) or not request.request_id:
                raise ValueError("request_id must be a nonempty string")
            if request.request_id in seen_requests:
                raise ValueError(f"duplicate request_id: {request.request_id}")
            seen_requests.add(request.request_id)
            seen_players: set[str] = set()
            for participant in request.participants:
                player = participant.player_id
                if player not in self.balances:
                    raise ValueError(f"unconfigured player: {player}")
                if player in seen_players:
                    raise ValueError(f"duplicate participant in {request.request_id}: {player}")
                seen_players.add(player)
                if participant.player_type != self.config.player_types[player]:
                    raise ValueError(f"inconsistent player_type: {player}")
                if not self._finite_nonnegative(participant.pvalue):
                    raise ValueError("pvalue must be finite and nonnegative")
            self._slots[request.time_step].append(request)

    @staticmethod
    def _finite_nonnegative(value: object) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value >= 0

    def _validate_config(self) -> None:
        config = self.config
        if not isinstance(config.steps, int) or isinstance(config.steps, bool) or config.steps <= 0:
            raise ValueError("steps must be a positive integer")
        if not self._finite_nonnegative(config.alpha_max) or config.alpha_max <= 0:
            raise ValueError("alpha_max must be positive and finite")
        if not config.budgets or set(config.budgets) != set(config.player_types):
            raise ValueError("budgets and player_types must have identical nonempty player sets")
        if any(not isinstance(p, str) or not p for p in config.budgets):
            raise ValueError("player IDs must be nonempty strings")
        if any(not isinstance(t, str) or not t for t in config.player_types.values()):
            raise ValueError("player types must be nonempty strings")
        if any(not self._finite_nonnegative(b) for b in config.budgets.values()):
            raise ValueError("budgets must be finite and nonnegative")
        if config.budget_policy not in {"cap_bid", "reject_unaffordable"}:
            raise ValueError("unknown budget_policy")
        if config.tie_break not in {"seeded_random", "stable"}:
            raise ValueError("tie_break must be seeded_random or stable")
        if not isinstance(config.seed, int) or isinstance(config.seed, bool):
            raise ValueError("seed must be an integer")

    @property
    def done(self) -> bool:
        return self.time_step >= self.config.steps

    def observation(self, player_id: str) -> Observation:
        """Return a new immutable, own-history-only value object.

        Future requests, request counts, current values and competitor state are
        intentionally absent. This API separation is not a process sandbox.
        """
        if player_id not in self.balances:
            raise KeyError(player_id)
        return Observation(
            time_step=self.time_step,
            total_steps=self.config.steps,
            player_id=player_id,
            player_type=self.config.player_types[player_id],
            initial_budget=float(self.config.budgets[player_id]),
            remaining_budget=self.balances[player_id],
            alpha_max=float(self.config.alpha_max),
            history=tuple(self.histories[player_id]),
        )

    def _priority(self, request: Request, player: str) -> tuple:
        if self.config.tie_break == "stable":
            return (player,)
        payload = json.dumps(
            [self.config.seed, request.time_step, request.request_id, player],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return (blake2b(payload, digest_size=16).digest(), player)

    def step(self, actions: dict[str, float]) -> dict[str, Feedback]:
        """Validate every player's coefficient, then settle this slice in order."""
        if self.done:
            raise RuntimeError("day is complete; construct/reset a new simulator")
        if set(actions) != set(self.balances):
            raise ValueError("one action is required for every configured player")
        if any(not self._finite_nonnegative(a) or a > self.config.alpha_max for a in actions.values()):
            raise ValueError("actions must be finite coefficients in [0, alpha_max]")
        actions = {p: float(a) for p, a in actions.items()}
        # Validate multiplication before changing any state, including earlier
        # requests in the same slice. Invalid actions never partially settle.
        for request in self._slots[self.time_step]:
            if any(not math.isfinite(actions[p.player_id] * p.pvalue) for p in request.participants):
                raise ValueError("coefficient times pvalue overflowed")
        totals = {p: {"wins": 0, "cost": 0.0, "value": 0.0} for p in self.balances}
        for request in self._slots[self.time_step]:
            raw_bids = {p.player_id: actions[p.player_id] * p.pvalue for p in request.participants}
            values = {p.player_id: p.pvalue for p in request.participants}
            bids = {
                p: min(bid, self.balances[p]) if self.config.budget_policy == "cap_bid" else bid
                for p, bid in raw_bids.items()
                if bid > 0 and self.balances[p] > 0
            }
            initial_effective = dict(bids)
            rejected: list[str] = []
            winner: str | None = None
            cost = 0.0
            while bids:
                ordered = sorted(bids, key=lambda p: (-bids[p], self._priority(request, p)))
                candidate = ordered[0]
                price = bids[ordered[1]] if len(ordered) > 1 else 0.0
                if price > self.balances[candidate]:
                    # Only reachable under reject_unaffordable. Removing this
                    # bidder changes both the winner and the next payment.
                    rejected.append(candidate)
                    del bids[candidate]
                    continue
                winner, cost = candidate, price
                self.balances[winner] = max(0.0, self.balances[winner] - cost)
                totals[winner]["wins"] += 1
                totals[winner]["cost"] += cost
                totals[winner]["value"] += values[winner]
                break
            if self._audit_enabled:
                self.audit.append({
                    "time_step": self.time_step,
                    "request_id": request.request_id,
                    "participants": list(values),
                    "pvalues": values,
                    "raw_bids": raw_bids,
                    "effective_bids": initial_effective,
                    "rejected_unaffordable": rejected,
                    "winner": winner,
                    "cost": cost,
                    "value": values[winner] if winner is not None else 0.0,
                })
        feedback = {
            p: Feedback(
                time_step=self.time_step, alpha=actions[p],
                wins=int(totals[p]["wins"]), cost=float(totals[p]["cost"]),
                value=float(totals[p]["value"]), remaining_budget=self.balances[p],
            )
            for p in self.balances
        }
        for player, result in feedback.items():
            self.histories[player].append(result)
        self.time_step += 1
        return feedback
