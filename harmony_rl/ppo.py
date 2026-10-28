"""On-policy PPO with complete-episode TD/GAE and fixed reward scaling."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
import torch


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 1.
    gae_lambda: float = .95
    clip_epsilon: float = .2
    epochs: int = 4
    minibatch_size: int = 256
    value_coef: float = .5
    entropy_coef: float = .005
    max_grad_norm: float = .5
    target_kl: float = .03
    reward_scale: float = 1.

    def __post_init__(self):
        numeric = [self.gamma, self.gae_lambda, self.clip_epsilon, self.value_coef,
                   self.entropy_coef, self.max_grad_norm, self.target_kl, self.reward_scale]
        if not np.isfinite(numeric).all() or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid finite PPO discount settings")
        if (self.epochs < 1 or self.minibatch_size < 1 or self.clip_epsilon <= 0
                or self.value_coef < 0 or self.entropy_coef < 0 or self.max_grad_norm <= 0
                or self.target_kl <= 0 or self.reward_scale <= 0):
            raise ValueError("positive PPO optimization settings required")


def compute_gae(rewards, values, dones, gamma=1., gae_lambda=.95, bootstrap_value=0.):
    """Terminal masks reset both bootstrap and eligibility across episodes."""
    rewards, values = np.asarray(rewards, dtype=np.float64), np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=bool)
    if rewards.ndim != 1 or rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("GAE inputs must be equal one-dimensional arrays")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all() or not np.isfinite(bootstrap_value):
        raise ValueError("GAE inputs must be finite")
    advantage = np.zeros_like(rewards)
    tail = 0.
    for t in range(len(rewards)-1, -1, -1):
        next_value = values[t+1] if t+1 < len(values) else bootstrap_value
        continuation = 0. if dones[t] else 1.
        delta = rewards[t] + gamma * next_value * continuation - values[t]
        tail = delta + gamma * gae_lambda * continuation * tail
        advantage[t] = tail
    return advantage.astype(np.float32), (advantage + values).astype(np.float32)


def update_ppo(model, optimizer, trajectories, config=None, reward_sequences=None,
               expected_policy_version=None):
    """Use one fresh rollout batch once; replay trajectories are rejected.

Old policy values are in scaled reward units. Optional redistributed rewards
remain in raw units and must conserve every individual complete return.
"""
    started = perf_counter()
    config = PPOConfig() if config is None else PPOConfig(**config) if isinstance(config, dict) else config
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError("PPO requires fresh complete trajectories")
    versions = {t.metadata.get("policy_version") for t in trajectories}
    if None in versions or len(versions) != 1:
        raise ValueError("PPO rollout must have exactly one explicit policy version")
    if expected_policy_version is not None and versions != {expected_policy_version}:
        raise ValueError("stale PPO trajectory policy version")
    if any(t.metadata.get("lstm_replay_only") or t.metadata.get("_ppo_consumed") for t in trajectories):
        raise ValueError("LSTM replay or already-consumed trajectories cannot update PPO")
    rewards = [t.rewards for t in trajectories] if reward_sequences is None else list(reward_sequences)
    if len(rewards) != len(trajectories):
        raise ValueError("reward sequence count differs from rollout count")
    advantages, returns = [], []
    for trajectory, reward in zip(trajectories, rewards):
        reward = np.asarray(reward, dtype=np.float64)
        if reward.shape != (72,) or not np.isfinite(reward).all():
            raise ValueError("PPO needs 72 finite rewards per episode")
        if not np.isclose(reward.sum(), trajectory.episode_return, atol=1e-7, rtol=1e-10):
            raise ValueError("redistribution failed individual return conservation")
        advantage, target = compute_gae(reward / config.reward_scale, trajectory.values,
                                        trajectory.dones, config.gamma, config.gae_lambda)
        advantages.append(advantage)
        returns.append(target)
    device = next(model.parameters()).device
    tensor = lambda value: torch.as_tensor(np.concatenate(value), dtype=torch.float32, device=device)
    states = tensor([t.states for t in trajectories])
    latents = tensor([t.latents for t in trajectories])
    old_log_probs = tensor([t.log_probs for t in trajectories])
    advantage = tensor(advantages)
    target = tensor(returns)
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    metrics = []
    stopped = False
    for epoch in range(config.epochs):
        order = torch.randperm(len(states), device=device)
        for index in order.split(config.minibatch_size):
            log_probs, entropy, predictions = model.evaluate(states[index], latents[index])
            log_ratio = log_probs - old_log_probs[index]
            ratio = log_ratio.exp()
            if not torch.isfinite(ratio).all():
                raise FloatingPointError("PPO likelihood ratio overflow")
            surrogate = ratio * advantage[index]
            clipped = ratio.clamp(1-config.clip_epsilon, 1+config.clip_epsilon) * advantage[index]
            actor_loss = -torch.minimum(surrogate, clipped).mean()
            critic_loss = .5 * (predictions - target[index]).square().mean()
            loss = actor_loss + config.value_coef * critic_loss - config.entropy_coef * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite PPO objective")
            with torch.no_grad():
                kl = ((ratio - 1) - log_ratio).mean()
                clip_fraction = ((ratio - 1).abs() > config.clip_epsilon).float().mean()
            if float(kl) > 1.5 * config.target_kl:
                stopped = True
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm,
                                                       error_if_nonfinite=True)
            optimizer.step()
            metrics.append([float(actor_loss.detach()), float(critic_loss.detach()),
                            float(entropy.mean().detach()), float(kl), float(clip_fraction), float(grad_norm)])
        if stopped:
            break
    for trajectory in trajectories:
        trajectory.metadata["_ppo_consumed"] = True
    values = np.mean(metrics, axis=0) if metrics else np.zeros(6)
    result = dict(zip(("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm"), map(float, values)))
    result.update(optimizer_steps=len(metrics), kl_early_stopped=stopped, rollout_steps=len(states),
                  rollout_episodes=len(trajectories), ppo_seconds=perf_counter()-started,
                  mean_raw_return=float(np.mean([t.episode_return for t in trajectories])))
    return result
