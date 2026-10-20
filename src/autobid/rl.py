"""Optional history-only Gymnasium environment and Stable-Baselines3 PPO.

The simulator owns the request stream. Policies only receive ``Observation``.
No current opportunities, rival state, auction audit, or future statistics appear
in features or ``info``. Install the project's ``rl`` extra to train a model.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from typing import Callable, Sequence

from .types import AuctionConfig, Feedback, Observation, Request

try:
    import gymnasium as gym
    import numpy as np
except ImportError:
    gym = None
    np = None


FEATURE_VERSION = "own-history-v1"
HISTORY_LENGTH = 4
FEATURE_NAMES = [
    "time_fraction", "remaining_budget_fraction", "initial_budget_squashed",
    "cumulative_cost_fraction", "cumulative_value_squashed", "cumulative_wins_squashed",
] + [f"lag{lag}_{name}" for lag in range(1, HISTORY_LENGTH + 1)
     for name in ("present", "alpha_fraction", "cost_fraction", "value_squashed", "wins_squashed")]


def _require_rl():
    if gym is None or np is None:
        raise ImportError("PPO requires optional dependencies: pip install -e '.[rl]'")


def _squash(value: float, scale: float = 1.0) -> float:
    """Bound nonnegative values without fitted statistics or future data."""
    return 0.0 if value == 0 else 1.0 / (1.0 + scale / value)


def encode_observation(observation: Observation):
    """Encode only a player's own observations, with fixed, saved semantics.

    All features are in [0, 1]. Missing history has an explicit presence mask.
    Player ID and category are deliberately not treated as ordinal numbers.
    There is no learned normalization that can accidentally fit validation data.
    """
    _require_rl()
    obs = observation
    budget_scale = obs.initial_budget + 1.0
    total_cost = sum(f.cost for f in obs.history)
    total_value = sum(f.value for f in obs.history)
    total_wins = sum(f.wins for f in obs.history)
    values = [
        obs.time_step / obs.total_steps,
        obs.remaining_budget / obs.initial_budget if obs.initial_budget > 0 else 0.0,
        _squash(obs.initial_budget), total_cost / budget_scale,
        _squash(total_value, budget_scale), _squash(total_wins),
    ]
    history = list(reversed(obs.history[-HISTORY_LENGTH:]))
    for lag in range(HISTORY_LENGTH):
        if lag >= len(history):
            values.extend([0.0] * 5)
        else:
            f = history[lag]
            values.extend([
                1.0, f.alpha / obs.alpha_max, f.cost / budget_scale,
                _squash(f.value, budget_scale), _squash(f.wins),
            ])
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Own-history observation contains a non-finite feature")
    return np.clip(result, 0.0, 1.0)


class AutobidEnv(gym.Env if gym is not None else object):
    """One step is one entire time slot; one episode is one complete day.

    Optional ``episode_requests`` and ``player_ids`` sample *training* days and
    roles at reset. Pass only training splits. Opponents are freshly constructed
    and updated from their own feedback within every episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, requests: Sequence[Request], config: AuctionConfig, player_id: str,
        opponent_factory: Callable | None = None, *,
        episode_requests: Sequence[Sequence[Request]] | None = None,
        player_ids: Sequence[str] | None = None,
    ):
        _require_rl()
        from .policies import PacingPolicy
        self.config = config
        self._datasets = [tuple(requests)] if episode_requests is None else [tuple(r) for r in episode_requests]
        self._player_ids = [player_id] if player_ids is None else list(player_ids)
        if not self._datasets or not self._player_ids:
            raise ValueError("At least one training day and player are required")
        if any(p not in config.budgets for p in self._player_ids):
            raise ValueError("Training player IDs must exist in the configuration")
        self.player_id = player_id
        self._opponent_factory = opponent_factory or PacingPolicy
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(len(FEATURE_NAMES),), dtype=np.float32)
        self.action_space = gym.spaces.Box(0.0, config.alpha_max, shape=(1,), dtype=np.float32)
        self._simulator = None
        self._opponents = {}
        self._initialized = False

    def reset(self, *, seed=None, options=None):
        from .engine import DaySimulator
        if options:
            raise ValueError("Reset options are unsupported; configure training days/roles explicitly")
        if seed is None and not self._initialized:
            seed = self.config.seed
        super().reset(seed=seed)
        self._initialized = True
        dataset = self._datasets[int(self.np_random.integers(len(self._datasets)))]
        self.player_id = self._player_ids[int(self.np_random.integers(len(self._player_ids)))]
        simulator_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._simulator = DaySimulator(dataset, replace(self.config, seed=simulator_seed))
        self._opponents = {
            p: self._opponent_factory() for p in self.config.budgets if p != self.player_id
        }
        for p, opponent in self._opponents.items():
            opponent.reset(self._simulator.observation(p))
        return encode_observation(self._simulator.observation(self.player_id)), {}

    def step(self, action):
        if self._simulator is None:
            raise RuntimeError("Call reset() before step()")
        if self._simulator.done:
            raise RuntimeError("Episode finished; call reset() before another step()")
        scalar = np.asarray(action, dtype=np.float64)
        if scalar.shape not in ((), (1,)):
            raise ValueError("Each time slot requires exactly one alpha")
        alpha = float(scalar.reshape(-1)[0])
        if not math.isfinite(alpha) or not 0 <= alpha <= self.config.alpha_max:
            raise ValueError("Alpha must be finite and within the configured bounds")
        # Collect every action before processing any request in this slot.
        actions = {self.player_id: alpha}
        actions.update({p: opponent.bidding(self._simulator.observation(p))
                        for p, opponent in self._opponents.items()})
        feedback = self._simulator.step(actions)
        for p, opponent in self._opponents.items():
            opponent.update(feedback[p])
        own = feedback[self.player_id]
        obs = encode_observation(self._simulator.observation(self.player_id))
        # Natural finite-horizon completion is termination, not truncation.
        return obs, float(own.value), self._simulator.done, False, {"own_feedback": asdict(own)}


def _requests_digest(requests):
    digest = hashlib.sha256()
    for request in requests:
        digest.update(json.dumps(asdict(request), sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def train_ppo(
    requests: Sequence[Request], config: AuctionConfig, output: str | Path, *,
    player_id: str, total_timesteps: int = 4096, seed: int = 0,
    episode_requests: Sequence[Sequence[Request]] | None = None,
    player_ids: Sequence[str] | None = None,
) -> Path:
    """Train and save a CPU PPO plus the feature/configuration contract.

    The undiscounted value objective uses gamma=1. This is an engineering
    baseline, not a claim of superiority to the organizer's private PID.
    """
    _require_rl()
    from stable_baselines3 import PPO
    import torch
    if isinstance(total_timesteps, bool) or not isinstance(total_timesteps, int) or total_timesteps < 2:
        raise ValueError("total_timesteps must be an integer >= 2")
    torch.set_num_threads(1)
    env = AutobidEnv(requests, config, player_id, episode_requests=episode_requests, player_ids=player_ids)
    n_steps = min(144, max(2, total_timesteps))
    batch_size = next((b for b in range(min(48, n_steps), 1, -1) if n_steps % b == 0), n_steps)
    model = PPO(
        "MlpPolicy", env, device="cpu", seed=seed, gamma=1.0,
        n_steps=n_steps, batch_size=batch_size, n_epochs=4,
        learning_rate=3e-4, policy_kwargs={"net_arch": [64, 64]}, verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    output = Path(output)
    if output.suffix != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    datasets = [requests] if episode_requests is None else episode_requests
    metadata = {
        "feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
        "history_length": HISTORY_LENGTH, "alpha_max": config.alpha_max,
        "total_steps": config.steps, "gamma": 1.0, "reward": "sum_won_pvalue",
        "seed": seed, "requested_timesteps": total_timesteps,
        "actual_timesteps": model.num_timesteps, "training_config": asdict(config),
        "training_player_ids": [player_id] if player_ids is None else list(player_ids),
        "training_request_sha256": [_requests_digest(day) for day in datasets],
        "opponent": "local_PacingPolicy_not_official_PID",
        "versions": {name: version(name) for name in ("numpy", "torch", "gymnasium", "stable-baselines3")},
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    env.close()
    return output


class PPOPolicy:
    """Inference-only policy. Load only model files you trust (SB3 uses pickle)."""

    def __init__(self, model_path: str | Path):
        _require_rl()
        from stable_baselines3 import PPO
        model_path = Path(model_path)
        if model_path.suffix != ".zip":
            model_path = model_path.with_suffix(".zip")
        self.metadata = json.loads(model_path.with_suffix(".metadata.json").read_text())
        if (self.metadata.get("feature_version") != FEATURE_VERSION or
                self.metadata.get("feature_names") != FEATURE_NAMES):
            raise ValueError("Model feature contract does not match this code version")
        self.model = PPO.load(model_path, device="cpu")

    def reset(self, observation: Observation):
        if (observation.alpha_max != self.metadata["alpha_max"] or
                observation.total_steps != self.metadata["total_steps"]):
            raise ValueError("Model alpha bound or time horizon differs from the evaluation configuration")

    def bidding(self, observation: Observation) -> float:
        self.reset(observation)
        if observation.remaining_budget <= 0:
            return 0.0
        action, _ = self.model.predict(encode_observation(observation), deterministic=True)
        alpha = float(np.asarray(action).reshape(-1)[0])
        if not math.isfinite(alpha):
            raise ValueError("Model produced a non-finite alpha")
        return min(observation.alpha_max, max(0.0, alpha))

    def update(self, feedback: Feedback):
        # History belongs to the caller; inference performs no gradient updates.
        return None
