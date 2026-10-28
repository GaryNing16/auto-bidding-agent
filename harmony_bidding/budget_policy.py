"""Explicit LOCAL budget policy. This is not a claim about the official evaluator."""
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class Settlement:
    winner: str | None
    payment_units: int
    excluded: tuple[str, ...]


class BudgetPolicy(Protocol):
    def settle(self, bids: Mapping[str, int], remaining: Mapping[str, int],
               choose_tie: Callable[[tuple[str, ...], int], str]) -> Settlement: ...


@dataclass(frozen=True, slots=True)
class ExcludeUnaffordablePolicy:
    zero_balance_can_bid: bool = False

    def settle(self, bids, remaining, choose_tie):
        candidates = {a: b for a, b in bids.items()
                      if b > 0 and (remaining[a] > 0 or self.zero_balance_can_bid)}
        excluded = []
        round_index = 0
        while candidates:
            top_bid = max(candidates.values())
            tied = tuple(sorted(a for a, bid in candidates.items() if bid == top_bid))
            winner = tied[0] if len(tied) == 1 else choose_tie(tied, round_index)
            if winner not in tied:
                raise AssertionError("tie chooser must select one of the tied candidates")
            price = max((b for a, b in candidates.items() if a != winner), default=0)
            if remaining[winner] >= price:
                return Settlement(winner, price, tuple(excluded))
            excluded.append(winner)
            del candidates[winner]
            round_index += 1
        return Settlement(None, 0, tuple(excluded))
