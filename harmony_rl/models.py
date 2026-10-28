"""Causal own-history encoding and uncapped log-space PPO actions.

Likelihoods are recorded in the sampled Normal coordinate. For positive base
alpha, the log-normal Jacobian is identical in old/new likelihoods and cancels.
For a zero residual baseline the action is zero; the latent policy is still a
well-defined stochastic policy followed by a deterministic environment action.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

from harmony_baselines.neural import FEATURE_NAMES as HISTORY_FEATURE_NAMES
from harmony_baselines.neural import encode_observation

FEATURE_VERSION = "ppo-own-history-identity-v1"


class StateEncoder:
    """Fixed analytic normalization: never updated between sampling and PPO.

Identities/types are static own configuration, not market or opponent features.
Unknown identities/types map to a separate unknown indicator, never an ordinal.
"""
    def __init__(self, player_ids=(), player_types=(), mean=None, std=None):
        self.player_ids = tuple(dict.fromkeys(map(str, player_ids)))
        self.player_types = tuple(dict.fromkeys(map(str, player_types)))
        self.feature_names = (list(HISTORY_FEATURE_NAMES)
                              + [f"player:{p}" for p in self.player_ids] + ["player:unknown"]
                              + [f"type:{p}" for p in self.player_types] + ["type:unknown"])
        self.state_dim = len(self.feature_names)
        default_std = np.ones(self.state_dim, dtype=np.float32)
        # Log magnitudes can span several orders; these are predefined scales.
        default_std[[2, 4, 7, 9, 10, 13, 14, 17]] = 10.
        self.mean = np.asarray(np.zeros(self.state_dim) if mean is None else mean, dtype=np.float32)
        self.std = np.asarray(default_std if std is None else std, dtype=np.float32)
        if self.mean.shape != (self.state_dim,) or self.std.shape != (self.state_dim,):
            raise ValueError("normalization dimensions differ from feature contract")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("normalization must be finite with strictly positive scale")

    def encode(self, observation):
        raw = np.zeros(self.state_dim, dtype=np.float32)
        raw[:18] = encode_observation(observation)
        player = str(observation.own_config.advertiser_id)
        kind = str(observation.own_config.advertiser_type)
        pindex = self.player_ids.index(player) if player in self.player_ids else len(self.player_ids)
        tindex = self.player_types.index(kind) if kind in self.player_types else len(self.player_types)
        raw[18 + pindex] = 1.
        raw[19 + len(self.player_ids) + tindex] = 1.
        return (raw - self.mean) / self.std

    def to_dict(self):
        return dict(feature_version=FEATURE_VERSION, player_ids=list(self.player_ids),
                    player_types=list(self.player_types), mean=self.mean.tolist(), std=self.std.tolist(),
                    normalization="fixed-analytic-or-explicit-frozen", feature_names=self.feature_names)

    @classmethod
    def from_dict(cls, state):
        if state.get("feature_version") != FEATURE_VERSION:
            raise ValueError("unsupported PPO observation contract")
        result = cls(state["player_ids"], state["player_types"], state["mean"], state["std"])
        if state.get("feature_names", result.feature_names) != result.feature_names:
            raise ValueError("feature ordering differs from saved encoder")
        return result


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=64, initial_alpha=1., residual=False, initial_std=.35):
        super().__init__()
        if obs_dim < 1 or hidden < 1 or not math.isfinite(initial_alpha) or initial_alpha <= 0:
            raise ValueError("positive model dimensions and initial alpha are required")
        if not math.isfinite(initial_std) or initial_std <= 0:
            raise ValueError("initial_std must be finite and positive")
        self.config = dict(obs_dim=int(obs_dim), hidden=int(hidden), initial_alpha=float(initial_alpha),
                           residual=bool(residual), initial_std=float(initial_std))
        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        for network in (self.actor, self.critic):
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.))
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=.01)
        nn.init.constant_(self.actor[-1].bias, 0. if residual else math.log(initial_alpha))
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.)
        self.log_std = nn.Parameter(torch.tensor(math.log(initial_std), dtype=torch.float32))

    def distribution(self, states):
        mean = self.actor(states).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise FloatingPointError("nonfinite Normal policy parameters")
        return torch.distributions.Normal(mean, std)

    def value(self, states):
        return self.critic(states).squeeze(-1)

    def evaluate(self, states, latents):
        distribution = self.distribution(states)
        return distribution.log_prob(latents), distribution.entropy(), self.value(states)


def decode_action(latent, base_alpha=1.):
    """Reject only floating-point failures; impose no strategy or contest cap."""
    z, base = float(latent), float(base_alpha)
    if not math.isfinite(z) or not math.isfinite(base) or base < 0:
        raise ValueError("nonfinite latent or invalid residual baseline")
    if base == 0:
        return 0.
    try:
        alpha = math.exp(z + math.log(base))
    except OverflowError as exc:
        raise FloatingPointError("log-space action exceeds float64 representation") from exc
    if not math.isfinite(alpha) or alpha <= 0:
        raise FloatingPointError("log-space action is not representable as positive float64")
    return alpha


@torch.no_grad()
def sample_action(model, state, base_alpha=1., deterministic=False):
    device = next(model.parameters()).device
    tensor = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device).reshape(1, -1)
    distribution = model.distribution(tensor)
    latent = distribution.mean if deterministic else distribution.sample()
    z = float(latent.item())
    alpha = decode_action(z, base_alpha)
    log_prob = float(distribution.log_prob(latent).item())
    value = float(model.value(tensor).item())
    if not all(math.isfinite(v) for v in (log_prob, value)):
        raise FloatingPointError("nonfinite action likelihood or critic prediction")
    return dict(alpha=alpha, latent=z, log_prob=log_prob, value=value,
                alpha_log_prob=log_prob - math.log(alpha) if alpha > 0 else None)
