"""Complete 72-slot trajectories; historical replay is for return prediction only."""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np


@dataclass
class Trajectory:
    states: np.ndarray
    actions: np.ndarray
    latents: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    spends: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("states", "next_states", "actions", "latents", "log_probs", "values", "rewards", "spends"):
            array = np.asarray(getattr(self, name), dtype=np.float64 if name in ("actions", "rewards", "spends") else np.float32)
            setattr(self, name, array)
            if not np.isfinite(array).all():
                raise ValueError("nonfinite trajectory field: " + name)
        self.dones = np.asarray(self.dones, dtype=bool)
        if self.states.ndim != 2 or self.states.shape[0] != 72 or self.next_states.shape != self.states.shape:
            raise ValueError("trajectories must contain exactly 72 state transitions")
        for name in ("actions", "latents", "log_probs", "values", "rewards", "spends", "dones"):
            if getattr(self, name).shape != (72,):
                raise ValueError("trajectory field must be shape (72,): " + name)
        if self.dones[:-1].any() or not self.dones[-1]:
            raise ValueError("only the 72nd transition may terminate")
        if any(np.any(getattr(self, name) < 0) for name in ("actions", "rewards", "spends")):
            raise ValueError("raw actions, value rewards and spends must be nonnegative")
        self.metadata = deepcopy(self.metadata)

    @property
    def episode_return(self):
        return float(self.rewards.sum(dtype=np.float64))

    def state_dict(self):
        # Plain lists permit torch.load(weights_only=True) without NumPy globals.
        return {name: deepcopy(self.metadata) if name == "metadata" else getattr(self, name).tolist()
                for name in self.__dataclass_fields__}

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)


class ReplayBuffer:
    """Bounded complete-episode FIFO, explicitly excluded from actor rollouts."""
    def __init__(self, capacity=512):
        if int(capacity) < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.trajectories = []

    def add(self, trajectory):
        if not isinstance(trajectory, Trajectory):
            raise TypeError("ReplayBuffer accepts complete Trajectory objects only")
        item = Trajectory.from_state_dict(trajectory.state_dict())
        item.metadata["lstm_replay_only"] = True
        self.trajectories.append(item)
        self.trajectories = self.trajectories[-self.capacity:]

    def __len__(self):
        return len(self.trajectories)

    def state_dict(self):
        return dict(capacity=self.capacity, trajectories=[t.state_dict() for t in self.trajectories])

    def load_state_dict(self, state):
        if int(state["capacity"]) < 1 or len(state["trajectories"]) > int(state["capacity"]):
            raise ValueError("invalid replay checkpoint")
        self.capacity = int(state["capacity"])
        self.trajectories = []
        for trajectory in state["trajectories"]:
            self.add(Trajectory.from_state_dict(trajectory))
