"""A custom Agent example using only the public local strategy contract."""
import math
from harmony_bidding.agent_api import Agent, Observation, OwnConfig, OwnFeedback, PublicConfig


class ExampleConstantAgent(Agent):
    """Fixed scalar action; every episode starts with fresh feedback state."""

    def __init__(self, alpha: float = 1.2):
        super().__init__()
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be a finite nonnegative number")
        self.alpha = float(alpha)
        self.last_feedback = None

    def reset(self, own_config: OwnConfig, public_config: PublicConfig) -> None:
        self.own_config = own_config
        self.public_config = public_config
        self.last_feedback = None

    def bidding(self, observation: Observation) -> float:
        return self.alpha

    def update(self, own_feedback: OwnFeedback) -> None:
        self.last_feedback = own_feedback
