"""Strict legal actor transitions; no fixed features, statistics or training bridge."""
import json
import math
import os
from pathlib import Path
import tempfile
from .agent_api import observation_from_dict, finite_number
from .auction import Money

VERSION = "harmony-trajectory-v2"
FIELDS = {"observation_t", "alpha_t", "reward_t", "cost_t", "next_observation", "done",
          "episode_id", "advertiser_id", "day_id", "run_id", "rules_version", "assumption_profile",
          "trajectory_version", "money_decimals"}


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _validate_history(observation, money):
    budget = money.budget_units(observation.own_config.initial_budget)
    costs, values = 0, 0.0
    for feedback in observation.history:
        costs += money.budget_units(feedback.cost)
        values += feedback.value
        if (costs > budget or money.budget_units(feedback.cumulative_cost) != costs
                or money.budget_units(feedback.remaining_budget) != budget - costs):
            raise ValueError("feedback budget ledger is inconsistent")
        if not math.isfinite(values) or not math.isclose(feedback.cumulative_value, values, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("feedback cumulative value is inconsistent")
        if feedback.wins == 0 and (feedback.cost != 0 or feedback.value != 0):
            raise ValueError("nonzero result without a win")
        if feedback.wins > 0 and feedback.value <= 0:
            raise ValueError("winning feedback must have positive value")
        if feedback.alpha == 0 and (feedback.wins or feedback.cost or feedback.value):
            raise ValueError("zero alpha cannot win")
    if money.budget_units(observation.remaining_budget) != budget - costs:
        raise ValueError("observation balance differs from completed feedback")


class TrajectoryValidator:
    """Separate complete episode/advertiser streams may be interleaved."""
    def __init__(self):
        self.streams = {}
        self.count = 0

    def accept(self, row):
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("trajectory has unexpected or missing fields")
        if row["trajectory_version"] != VERSION:
            raise ValueError("unsupported trajectory version; regenerate observations")
        if type(row["money_decimals"]) is not int or not 0 <= row["money_decimals"] <= 9:
            raise ValueError("invalid trajectory money precision")
        for name in ("episode_id", "advertiser_id", "day_id", "run_id", "rules_version", "assumption_profile"):
            if not isinstance(row[name], str) or not row[name].strip() or row[name] != row[name].strip():
                raise ValueError(f"invalid trajectory {name}")
        for name in ("alpha_t", "reward_t", "cost_t"):
            finite_number(row[name], name)
        if type(row["done"]) is not bool:
            raise ValueError("done must be boolean")
        observation = observation_from_dict(row["observation_t"])
        following = observation_from_dict(row["next_observation"])
        if (observation.own_config != following.own_config
                or observation.own_config.advertiser_id != row["advertiser_id"]):
            raise ValueError("trajectory advertiser identity/config mismatch")
        if (following.time_slot != observation.time_slot + 1
                or following.total_slots != observation.total_slots
                or observation.time_slot >= observation.total_slots):
            raise ValueError("nonconsecutive trajectory")
        if row["done"] != (following.time_slot == following.total_slots):
            raise ValueError("done does not match day termination")
        if following.history[:-1] != observation.history:
            raise ValueError("next observation rewrites previous feedback")
        current = following.history[-1]
        if (current.alpha, current.value, current.cost) != (row["alpha_t"], row["reward_t"], row["cost_t"]):
            raise ValueError("action/reward/cost differs from actual own feedback")
        money = Money(row["money_decimals"])
        _validate_history(observation, money)
        _validate_history(following, money)
        key = tuple(row[k] for k in ("run_id", "episode_id", "day_id", "advertiser_id"))
        profile = tuple(row[k] for k in ("rules_version", "assumption_profile", "money_decimals"))
        if key in self.streams:
            previous, previous_profile, finished = self.streams[key]
            if finished or profile != previous_profile or observation != previous:
                raise ValueError("duplicate, unordered, changed profile or discontinuous trajectory")
        elif observation.time_slot != 0:
            raise ValueError("trajectory stream must begin at slot zero")
        self.streams[key] = (following, profile, row["done"])
        self.count += 1

    def finish(self):
        if not self.count:
            raise ValueError("no trajectory transitions")
        if any(not finished for _, _, finished in self.streams.values()):
            raise ValueError("incomplete trajectory: missing terminal slot")
        return self.count


def export_trajectories(results, path):
    """Atomically export only complete legal episodes, checking the whole stream."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validator = TrajectoryValidator()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            for result in results:
                if result.get("checks") != {"legal": True, "budget": True} or result.get("exception_count"):
                    raise ValueError("cannot export a failed episode")
                for row in result["trajectories"]:
                    validator.accept(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            count = validator.finish()
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return count
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_trajectories(path):
    """Validate rows and complete streams at EOF. Consume fully to verify EOF."""
    validator = TrajectoryValidator()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line, object_pairs_hook=_json_object, parse_constant=_reject_constant)
            validator.accept(row)
            yield row
    validator.finish()


def validate_trajectory_file(path):
    """Explicit full-file validation without retaining all rows in RAM."""
    return sum(1 for _ in read_trajectories(path))
