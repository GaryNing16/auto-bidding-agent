"""Local bidding/update adapter; replace the outer signature when SDK is issued.

This module never receives request data or another advertiser's configuration.
Each new day must construct or reset an Agent. This is not an official SDK.
"""

from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path

from .types import Feedback, Observation


class Agent:
    def __init__(
        self, *, player_id: str, player_type: str, budget: float,
        total_steps: int = 72, alpha_max: float = 5.0, policy=None,
        model_path: str | Path | None = None,
    ):
        if not isinstance(player_id, str) or not player_id or not isinstance(player_type, str):
            raise ValueError("player_id and player_type must be strings; player_id cannot be empty")
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("budget must be finite and nonnegative")
        if isinstance(total_steps, bool) or not isinstance(total_steps, Integral) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if not math.isfinite(alpha_max) or alpha_max <= 0:
            raise ValueError("alpha_max must be finite and positive")
        if policy is not None and model_path is not None:
            raise ValueError("Choose policy or model_path, not both")
        if model_path is not None:
            from .rl import PPOPolicy
            policy = PPOPolicy(model_path)
        if policy is None:
            from .policies import FixedPolicy
            policy = FixedPolicy(alpha=1.0)
        self.player_id = player_id
        self.player_type = player_type
        self.initial_budget = float(budget)
        self.total_steps = int(total_steps)
        self.alpha_max = float(alpha_max)
        self.policy = policy
        self.reset()

    def reset(self):
        self.remaining_budget = self.initial_budget
        self.history = []
        self._pending_alpha = None
        self.policy.reset(self.observation())

    def observation(self) -> Observation:
        return Observation(
            time_step=len(self.history), total_steps=self.total_steps,
            player_id=self.player_id, player_type=self.player_type,
            initial_budget=self.initial_budget, remaining_budget=self.remaining_budget,
            alpha_max=self.alpha_max, history=tuple(self.history),
        )

    def bidding(self) -> float:
        if len(self.history) >= self.total_steps:
            raise RuntimeError("Day finished; reset the Agent for a new day")
        if self._pending_alpha is None:
            alpha = 0.0 if self.remaining_budget <= 0 else float(self.policy.bidding(self.observation()))
            if not math.isfinite(alpha) or not 0 <= alpha <= self.alpha_max:
                raise ValueError("Policy must return one finite alpha within configured bounds")
            self._pending_alpha = alpha
        return self._pending_alpha

    def update(self, *, wins: int, cost: float, value: float):
        if self._pending_alpha is None:
            raise RuntimeError("Call bidding() once before each update()")
        if isinstance(wins, bool) or not isinstance(wins, Integral) or wins < 0:
            raise ValueError("wins must be a nonnegative integer")
        if not math.isfinite(cost) or not math.isfinite(value) or cost < 0 or value < 0:
            raise ValueError("cost and value must be finite and nonnegative")
        tolerance = 1e-9 * max(1.0, self.initial_budget)
        if cost > self.remaining_budget + tolerance:
            raise ValueError("Feedback cost exceeds the remaining budget")
        if wins == 0 and (cost > 0 or value > 0):
            raise ValueError("Zero wins cannot have positive cost or won value")
        self.remaining_budget = max(0.0, self.remaining_budget - float(cost))
        feedback = Feedback(
            time_step=len(self.history), alpha=self._pending_alpha, wins=int(wins),
            cost=float(cost), value=float(value), remaining_budget=self.remaining_budget,
        )
        self.history.append(feedback)
        self._pending_alpha = None
        self.policy.update(feedback)
