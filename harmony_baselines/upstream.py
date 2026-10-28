"""Load the preserved AuctionNet implementation; never load upstream weights.

The upstream files remain unchanged. Compact layers and narrowly scoped numerical
repairs are local adaptations, explicitly recorded in UPSTREAM_AUDIT.md.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import MethodType

UPSTREAM_COMMIT = "c20de631a40e24669d1d6a8e2acefd8f492f172d"
ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "BC": ("bc.behavior_clone", "BC"),
    "IQL": ("iql.iql", "IQL"),
    "BCQ": ("bcq.bcq", "BCQ"),
    "TD3BC": ("td3_bc.td3_bc", "TD3_BC"),
    "CQL": ("cql.cql", "CQL"),
    "DT": ("dt.dt", "DecisionTransformer"),
}


def provenance():
    files = {}
    for name, (module, _) in MODULES.items():
        path = ROOT / "strategy_train_env/bidding_train_env/baseline" / (module.replace(".", "/") + ".py")
        files[name] = {"path": path.relative_to(ROOT).as_posix(),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"repository": "https://github.com/alimama-tech/AuctionNet",
            "commit": UPSTREAM_COMMIT, "license": "Apache-2.0", "files": files}


def _compact(module, width, original_widths):
    import torch.nn as nn
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            n_in = width if child.in_features in original_widths else child.in_features
            n_out = width if child.out_features in original_widths else child.out_features
            if (n_in, n_out) != (child.in_features, child.out_features):
                setattr(module, name, nn.Linear(n_in, n_out, bias=child.bias is not None))
        else:
            _compact(child, width, original_widths)


def _two_hot(self, values):
    """Repair upstream vmax out-of-range and n versus n-1 bin spacing."""
    import torch
    x = self.symlog(values).clamp(self.vmin, self.vmax).squeeze(1)
    location = (x - self.vmin) * (self.num_bins - 1) / (self.vmax - self.vmin)
    low = location.floor().long().clamp(0, self.num_bins - 1)
    high = (low + 1).clamp(max=self.num_bins - 1)
    fraction = location - low
    result = torch.zeros(x.shape[0], self.num_bins, device=x.device)
    result.scatter_add_(1, low[:, None], (1 - fraction)[:, None])
    result.scatter_add_(1, high[:, None], fraction[:, None])
    return result


def _td3_scalar_q1(self, state, action):
    """The policy objective needs decoded Q, not the mean of 101 logits."""
    import torch
    import torch.nn.functional as F
    sa = torch.cat([state.to(self.device), action.to(self.device)], 1)
    logits = self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))
    return self.two_hot_inv(logits)


def make_model(algorithm, dim_obs, seed, config):
    import torch
    from copy import deepcopy
    torch.manual_seed(int(seed))
    short, class_name = MODULES[algorithm]
    module = importlib.import_module("strategy_train_env.bidding_train_env.baseline." + short)
    cls = getattr(module, class_name)
    lr = float(config.get("learning_rate", 3e-4))
    gamma = float(config.get("gamma", 1.0))
    if algorithm == "BC":
        model = cls(dim_obs, actor_lr=lr, network_random_seed=seed, actor_train_iter=1)
    elif algorithm == "IQL":
        model = cls(dim_obs, gamma=gamma, V_lr=lr, critic_lr=lr, actor_lr=lr,
                    network_random_seed=seed, expectile=float(config.get("expectile", .7)),
                    temperature=float(config.get("temperature", 3.)))
    elif algorithm == "BCQ":
        model = cls(dim_obs, max_action=float(config["max_action"]), discount=gamma,
                    phi=float(config.get("phi", .05)))
        width = int(config.get("hidden_width", 128))
        _compact(model, width, {1400, 1300, 1750})
        model.actor_target = deepcopy(model.actor)
        model.critic_target = deepcopy(model.critic)
        model.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=lr)
        model.critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=lr)
        model.vae_optimizer = torch.optim.Adam(model.vae.parameters(), lr=lr)
        model.two_hot = MethodType(_two_hot, model)
    elif algorithm == "TD3BC":
        model = cls(dim_obs, discount=gamma, alpha=float(config.get("bc_weight", 2.5)))
        _compact(model, int(config.get("hidden_width", 128)), {756})
        model.actor_target = deepcopy(model.actor)
        model.critic_target = deepcopy(model.critic)
        model.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=lr)
        model.critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=lr)
        model.two_hot = MethodType(_two_hot, model)
        model.critic.Q1 = MethodType(_td3_scalar_q1, model.critic)
    elif algorithm == "CQL":
        model = cls(dim_obs, gamma=gamma, critic_lr=lr, actor_lr=lr,
                    network_random_seed=seed, expectile=float(config.get("expectile", .7)))
        model.target_qf1.load_state_dict(model.qf1.state_dict())
        model.target_qf2.load_state_dict(model.qf2.state_dict())
        model.min_q_weight = float(config.get("conservative_weight", 1.))
    else:
        model = cls(dim_obs, 1, torch.zeros(dim_obs), torch.ones(dim_obs),
                    K=int(config.get("context_length", 10)), max_ep_len=73,
                    scale=1., target_return=float(config.get("target_return", 1.)))
        # Original 10,000-step warmup is inappropriate for explicitly smaller
        # runs; preserve its scheduler with a configured warmup duration.
        model.warmup_steps = int(config.get("warmup_steps", 50))
        model.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=model.weight_decay)
        model.scheduler = torch.optim.lr_scheduler.LambdaLR(
            model.optimizer, lambda n: min((n + 1) / model.warmup_steps, 1.))
    model.cpu()
    return model
