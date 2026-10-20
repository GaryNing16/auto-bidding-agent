"""Small reproducible local opponents; these are not the undisclosed official PID."""

from __future__ import annotations

import math
from typing import Protocol

from .types import Feedback, Observation


class Policy(Protocol):
    def reset(self, observation: Observation) -> None: ...
    def bidding(self, observation: Observation) -> float: ...
    def update(self, feedback: Feedback) -> None: ...


def _validate_number(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


class FixedPolicy:
    """A fixed coefficient, clipped only to the role's legal action bound."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = _validate_number("alpha", alpha)

    def reset(self, observation: Observation) -> None:
        pass

    def bidding(self, observation: Observation) -> float:
        return min(self.alpha, observation.alpha_max) if observation.remaining_budget > 0 else 0.0

    def update(self, feedback: Feedback) -> None:
        pass


class PacingPolicy:
    """Self-made budget-progress PID-style baseline, not a competition replica.

    At slice t, the elapsed budget target is t / total_steps. Its deviation
    from cumulative spending adjusts the starting coefficient. The controller
    reads only the role's own state. Repeated bidding on the same Observation
    is idempotent because all controller terms are derived from that history.
    """

    def __init__(self, alpha: float = 1.0, kp: float = 0.8, ki: float = 0.1, kd: float = 0.0):
        self.alpha = _validate_number("alpha", alpha)
        self.kp = _validate_number("kp", kp)
        self.ki = _validate_number("ki", ki)
        self.kd = _validate_number("kd", kd)

    def reset(self, observation: Observation) -> None:
        pass

    def bidding(self, observation: Observation) -> float:
        if observation.remaining_budget <= 0 or observation.initial_budget <= 0:
            return 0.0
        spent_fraction = 1.0 - observation.remaining_budget / observation.initial_budget
        error = observation.time_step / observation.total_steps - spent_fraction
        historical_errors = [
            (f.time_step + 1) / observation.total_steps
            - (1.0 - f.remaining_budget / observation.initial_budget)
            for f in observation.history
        ]
        integral = sum(historical_errors)
        previous_error = historical_errors[-2] if len(historical_errors) >= 2 else 0.0
        alpha = self.alpha + self.kp * error + self.ki * integral + self.kd * (error - previous_error)
        return min(observation.alpha_max, max(0.0, alpha))

    def update(self, feedback: Feedback) -> None:
        # Observation.history is the authoritative feedback record; keeping a
        # second mutable copy here would make replay/reset harder to reproduce.
        pass
