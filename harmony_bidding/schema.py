"""Validated evaluator-only data, never passed to an Agent."""
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path


def fingerprint(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def nonnegative(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label}: expected finite nonnegative number, got {value!r}")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: number is not representable as a finite float") from error
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{label}: expected finite nonnegative number, got {value!r}")
    return converted


@dataclass(frozen=True, slots=True)
class Advertiser:
    advertiser_id: str
    advertiser_type: str
    budget: float


@dataclass(frozen=True, slots=True)
class Opportunity:
    day_id: str
    time_slot: int
    opportunity_id: str
    original_order: int
    values: tuple[float, ...]
    participation_mask: tuple[bool, ...]


def validate_values_mask(opportunity: Opportunity, advertiser_count: int) -> None:
    """Require explicit presence; a genuine zero-valued row remains present.

    This validates the canonical, immutable internal representation. CSV
    adapters construct it from actual row presence, never from value != 0.
    """
    if not isinstance(opportunity.values, tuple) or len(opportunity.values) != advertiser_count:
        raise ValueError("value vector does not match advertisers")
    mask = opportunity.participation_mask
    if not isinstance(mask, tuple) or len(mask) != advertiser_count or any(type(bit) is not bool for bit in mask):
        raise ValueError("participation_mask must be an explicit bool tuple matching advertisers")
    for index, (value, present) in enumerate(zip(opportunity.values, mask)):
        number = nonnegative(value, f"estimated_value {opportunity.opportunity_id}/{index}")
        if not present and number != 0:
            raise ValueError("absent participant must have a zero padded value")


@dataclass(frozen=True, slots=True)
class DayData:
    day_id: str
    advertisers: tuple[Advertiser, ...]
    slots: tuple[tuple[Opportunity, ...], ...]
    # Source provenance is evaluator-only and is not part of business equality
    # or the current semantic fingerprint. None means no source digest supplied.
    source_schema_version: str | None = field(default=None, compare=False)
    source_fingerprint: str | None = field(default=None, compare=False)

    def _payload(self):
        return {"day_id": self.day_id, "advertisers": [asdict(a) for a in self.advertisers],
                "slots": [[asdict(o) for o in slot] for slot in self.slots]}

    @property
    def fingerprint(self):
        return fingerprint(self._payload())

    @property
    def legacy_fingerprint(self):
        """v1 source digest, used only after enforcing complete v1 coverage."""
        payload = self._payload()
        for slot in payload["slots"]:
            for opportunity in slot:
                del opportunity["participation_mask"]
        return fingerprint(payload)


@dataclass(frozen=True, slots=True)
class Rules:
    total_slots: int = 72
    money_decimals: int = 6
    zero_balance_can_bid: bool = False
    tie_seed: int = 20261018
    budget_policy: str = "exclude_unaffordable_reauction"
    rules_version: str = "harmony-A-B-local-v2"
    assumption_profile: str = "local-unified-v2"

    def __post_init__(self):
        if type(self.total_slots) is not int or self.total_slots < 1:
            raise ValueError("total_slots must be positive integer")
        if type(self.money_decimals) is not int or not 0 <= self.money_decimals <= 9:
            raise ValueError("money_decimals must be integer in [0,9]")
        if type(self.zero_balance_can_bid) is not bool or type(self.tie_seed) is not int:
            raise ValueError("invalid zero balance switch or tie seed")
        if self.budget_policy != "exclude_unaffordable_reauction":
            raise ValueError("unsupported BudgetPolicy")
        if not self.rules_version or not self.assumption_profile:
            raise ValueError("rules_version/assumption_profile required")


def load_config(path: str | Path) -> dict:
    import yaml
    with Path(path).open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    if "alpha_max" in config.get("rules", {}):
        raise ValueError("rules.alpha_max is obsolete: the local v2 protocol has no global alpha cap; "
                         "remove it, and configure any voluntary action bound inside the strategy")
    Rules(**config.get("rules", {}))
    return config
