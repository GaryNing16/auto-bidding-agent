"""Shared contracts. Observations contain only one advertiser's own history."""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Participation:
    player_id: str
    player_type: str
    pvalue: float


@dataclass(frozen=True)
class Request:
    time_step: int
    request_id: str
    participants: tuple[Participation, ...]


@dataclass(frozen=True)
class Feedback:
    time_step: int
    alpha: float
    wins: int
    cost: float
    value: float
    remaining_budget: float


@dataclass(frozen=True)
class Observation:
    time_step: int
    total_steps: int
    player_id: str
    player_type: str
    initial_budget: float
    remaining_budget: float
    alpha_max: float
    history: tuple[Feedback, ...]


@dataclass
class AuctionConfig:
    budgets: dict[str, float] = field(default_factory=dict)
    player_types: dict[str, str] = field(default_factory=dict)
    steps: int = 72
    alpha_max: float = 5.0
    budget_policy: str = "cap_bid"
    tie_break: str = "seeded_random"
    seed: int = 0

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps <= 0:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(self.alpha_max) or self.alpha_max <= 0:
            raise ValueError("alpha_max must be finite and positive")
        if not self.budgets or set(self.budgets) != set(self.player_types):
            raise ValueError("budgets and player_types must contain the same nonempty player set")
        if any(not isinstance(k, str) or not k for k in self.budgets):
            raise ValueError("player IDs must be nonempty strings")
        if any(not isinstance(v, str) or not v for v in self.player_types.values()):
            raise ValueError("player types must be nonempty strings")
        if any(not math.isfinite(v) or v < 0 for v in self.budgets.values()):
            raise ValueError("budgets must be finite and nonnegative")
        if self.budget_policy not in {"cap_bid", "reject_unaffordable"}:
            raise ValueError("unknown provisional budget_policy")
        if self.tie_break not in {"seeded_random", "stable"}:
            raise ValueError("tie_break must be seeded_random or stable")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
