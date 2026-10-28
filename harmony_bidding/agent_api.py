"""Local scalar-action API. Frozen snapshots are an API boundary, not a security sandbox.

The upstream abstract strategy class is reused without its CPA constructor or
request-level arguments. Every upstream abstract operation is overridden.
"""
from abc import abstractmethod
from dataclasses import dataclass
import math
from numbers import Real
from simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy


@dataclass(frozen=True, slots=True)
class OwnConfig:
    advertiser_id: str
    advertiser_type: str
    initial_budget: float
    random_seed: int


@dataclass(frozen=True, slots=True)
class PublicConfig:
    total_slots: int


@dataclass(frozen=True, slots=True)
class OwnFeedback:
    time_slot: int
    alpha: float
    wins: int
    cost: float
    value: float
    remaining_budget: float
    cumulative_cost: float
    cumulative_value: float


@dataclass(frozen=True, slots=True)
class Observation:
    own_config: OwnConfig
    time_slot: int
    total_slots: int
    remaining_budget: float
    history: tuple[OwnFeedback, ...]


def finite_number(value, label):
    """Strict scalar contract shared by snapshots and trajectory ingestion."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label}: expected a finite nonnegative real")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label}: nonfinite real") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label}: expected a finite nonnegative real")
    return number


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: invalid integer")


def _identifier(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label}: invalid identity")


def observation_from_dict(raw):
    """Deserialize only the legal local API; never accept unknown nested fields."""
    def exact(value, keys, label):
        if not isinstance(value, dict) or set(value) != set(keys):
            raise ValueError(f"{label}: unexpected or missing fields")
    exact(raw, Observation.__dataclass_fields__, "observation")
    own = raw["own_config"]
    exact(own, OwnConfig.__dataclass_fields__, "own_config")
    _identifier(own["advertiser_id"], "advertiser_id")
    _identifier(own["advertiser_type"], "advertiser_type")
    finite_number(own["initial_budget"], "initial_budget")
    _integer(own["random_seed"], "random_seed")
    _integer(raw["total_slots"], "total_slots", 1)
    _integer(raw["time_slot"], "time_slot")
    remaining = finite_number(raw["remaining_budget"], "remaining_budget")
    history = raw["history"]
    if not isinstance(history, (list, tuple)) or len(history) != raw["time_slot"]:
        raise ValueError("history must contain exactly the completed slots")
    if raw["time_slot"] > raw["total_slots"] or remaining > own["initial_budget"]:
        raise ValueError("invalid observation time or balance")
    feedbacks = []
    for slot, feedback in enumerate(history):
        exact(feedback, OwnFeedback.__dataclass_fields__, "own_feedback")
        _integer(feedback["time_slot"], "feedback time_slot")
        _integer(feedback["wins"], "wins")
        if feedback["time_slot"] != slot:
            raise ValueError("feedback history out of sequence")
        for name in set(OwnFeedback.__dataclass_fields__) - {"time_slot", "wins"}:
            finite_number(feedback[name], name)
        feedbacks.append(OwnFeedback(**feedback))
    return Observation(OwnConfig(**own), raw["time_slot"], raw["total_slots"],
                       remaining, tuple(feedbacks))


class Agent(BaseBiddingStrategy):
    """Override all methods. reset is mandatory for every day/episode.

    No environment, market data, other agent, request count or value estimate is
    supplied. time_slot==total_slots occurs only in terminal next_observation.
    """
    def __init__(self):
        # Deliberately omit upstream CPA-bound state.
        self.own_config = None
        self.public_config = None

    @abstractmethod
    def reset(self, own_config: OwnConfig, public_config: PublicConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def bidding(self, observation: Observation) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(self, own_feedback: OwnFeedback) -> None:
        raise NotImplementedError


class OfficialStrategyUnavailable(RuntimeError):
    """The organizer's strategy implementation and parameters were not supplied."""


class OfficialBiddingStrategy(Agent):
    """Reserved local adapter contract, NOT a claim about an official SDK signature.

    A future provider must implement reset/bidding/update and initialization.
    There is deliberately no fabricated PID or fallback implementation here.
    """
    available = False
