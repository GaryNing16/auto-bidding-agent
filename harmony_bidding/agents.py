"""Small local baselines. LocalPIDAgent is not the unpublished official PID."""
from copy import deepcopy
import importlib
import math
import random
from .agent_api import Agent, OfficialBiddingStrategy, OfficialStrategyUnavailable


class ConstantAlphaAgent(Agent):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.configured_alpha = alpha

    def reset(self, own_config, public_config):
        self.own_config, self.public_config = own_config, public_config
        self.random = random.Random(own_config.random_seed)
        self.feedback_count = 0

    def bidding(self, observation):
        # No clipping: an invalid configured policy must fail strict validation.
        return self.configured_alpha

    def update(self, own_feedback):
        self.feedback_count += 1


class ZeroAgent(ConstantAlphaAgent):
    def __init__(self):
        super().__init__(0.0)


class LocalPIDAgent(Agent):
    def __init__(self, kp=1.5, ki=0.1, kd=0.15, initial_alpha=1.0, target_curve=None):
        super().__init__()
        for label, value in (("kp", kp), ("ki", ki), ("kd", kd), ("initial_alpha", initial_alpha)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid PID {label}")
        self.kp, self.ki, self.kd = kp, ki, kd
        self.initial_alpha = initial_alpha
        self.configured_curve = tuple(target_curve) if target_curve is not None else None

    def reset(self, own_config, public_config):
        self.own_config, self.public_config = own_config, public_config
        self.random = random.Random(own_config.random_seed)
        self.integral = 0.0
        self.previous_error = 0.0
        self.alpha = self.initial_alpha
        self.feedback_count = 0
        self.curve = (self.configured_curve if self.configured_curve is not None else
                      tuple((i + 1) / public_config.total_slots for i in range(public_config.total_slots)))
        if (len(self.curve) != public_config.total_slots or self.curve[-1] != 1
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in self.curve)
                or any(a > b for a, b in zip(self.curve, self.curve[1:]))):
            raise ValueError("PID target_curve must be monotone cumulative fractions ending at 1")

    def bidding(self, observation):
        return self.alpha

    def update(self, own_feedback):
        if own_feedback.time_slot != self.feedback_count:
            raise ValueError("PID feedback out of sequence")
        budget = self.own_config.initial_budget
        spent_fraction = own_feedback.cumulative_cost / budget if budget else 0.0
        error = self.curve[own_feedback.time_slot] - spent_fraction
        integral = self.integral + error / self.public_config.total_slots
        derivative = error - self.previous_error
        candidate = self.initial_alpha + self.kp * error + self.ki * integral + self.kd * derivative
        if not all(math.isfinite(v) for v in (integral, error, candidate)):
            raise ValueError("PID update produced nonfinite state")
        self.alpha = max(0.0, candidate)
        self.integral = integral
        self.previous_error = error
        self.feedback_count += 1


def make_agent(spec):
    """A sealed constructor specification is copied for every new episode.

    Custom class: {kind: custom, class_path: package.module:Class, kwargs: {...}}.
    Only trusted local Python is supported; process mode is lifecycle isolation.
    """
    if not isinstance(spec, dict):
        raise ValueError("agent specification must be a mapping")
    spec = deepcopy(spec)
    kind = spec.pop("kind", "pid")
    if kind == "official":
        raise OfficialStrategyUnavailable("OfficialBiddingStrategy is unavailable; no official PID was supplied")
    if kind == "custom":
        path = spec.pop("class_path")
        kwargs = spec.pop("kwargs", {})
        if spec:
            raise ValueError(f"unknown custom fields: {spec}")
        module, name = path.split(":")
        agent = getattr(importlib.import_module(module), name)(**kwargs)
    else:
        classes = {"zero": ZeroAgent, "constant": ConstantAlphaAgent, "pid": LocalPIDAgent}
        if kind not in classes:
            raise ValueError(f"unknown agent kind: {kind}")
        agent = classes[kind](**spec)
    if not isinstance(agent, Agent):
        raise TypeError("strategy must implement local Agent abstract class")
    if isinstance(agent, OfficialBiddingStrategy) and not agent.available:
        raise OfficialStrategyUnavailable("OfficialBiddingStrategy implementation is unavailable")
    return agent
