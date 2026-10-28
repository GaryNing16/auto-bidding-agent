"""Causal prefix return prediction and exactly endpoint-corrected rewards.

Prediction differences describe statistical attribution, not action causality
or a true marginal opportunity cost. The actor never consumes this predictor.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
import torch
from torch import nn


class ReturnPredictor(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.config = dict(input_dim=int(input_dim), hidden=int(hidden))
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, sequences):
        output, _ = self.lstm(sequences)
        return self.head(output).squeeze(-1)


def trajectory_features(trajectory, reward_scale=1.):
    if not np.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("positive finite reward_scale required")
    budget = float(trajectory.metadata.get("initial_budget", 1.))
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("invalid trajectory initial budget")
    denominator = budget if budget > 0 else 1.
    remaining = (budget - np.cumsum(trajectory.spends)) / denominator
    extra = np.column_stack((trajectory.latents, trajectory.rewards / reward_scale,
                             trajectory.spends / denominator, remaining, (np.arange(72)+1)/72))
    features = np.concatenate((trajectory.states, extra.astype(np.float32)), axis=1)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite LSTM prefix features")
    return features.astype(np.float32)


def redistribute(predictions, rewards):
    """F(h_t)-F(h_(t-1)), F(h_-1)=0, with final residual G-F(h_71)."""
    predictions, rewards = np.asarray(predictions, dtype=np.float64), np.asarray(rewards, dtype=np.float64)
    if predictions.shape != (72,) or rewards.shape != (72,):
        raise ValueError("reward redistribution requires one complete 72-step trajectory")
    if not np.isfinite(predictions).all() or not np.isfinite(rewards).all():
        raise ValueError("return predictions and rewards must be finite")
    result = np.diff(np.r_[0., predictions])
    # Set the last reward from the preceding sum, minimizing roundoff drift.
    result[-1] = rewards.sum(dtype=np.float64) - result[:-1].sum(dtype=np.float64)
    if not np.isclose(result.sum(), rewards.sum(), atol=1e-7, rtol=1e-10):
        raise FloatingPointError("return conservation exceeded numerical tolerance")
    return result


@torch.no_grad()
def predict_rewards(model, trajectory, reward_scale=1.):
    device = next(model.parameters()).device
    features = torch.as_tensor(trajectory_features(trajectory, reward_scale), device=device).unsqueeze(0)
    predictions = model(features)[0].cpu().numpy().astype(np.float64) * reward_scale
    return redistribute(predictions, trajectory.rewards), predictions


def train_predictor(model, optimizer, replay, epochs=2, batch_size=16, reward_scale=1., rng=None):
    """Only complete historical trajectories train this separate supervised model."""
    start = perf_counter()
    trajectories = replay.trajectories if hasattr(replay, "trajectories") else list(replay)
    if not trajectories or epochs < 1 or batch_size < 1:
        raise ValueError("LSTM training requires data and positive training counts")
    rng = np.random.default_rng(0) if rng is None else rng
    device = next(model.parameters()).device
    features = torch.as_tensor(np.stack([trajectory_features(t, reward_scale) for t in trajectories]), device=device)
    returns = torch.as_tensor([t.episode_return / reward_scale for t in trajectories], dtype=torch.float32, device=device)
    losses = []
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(trajectories))
        for start_index in range(0, len(order), batch_size):
            index = torch.as_tensor(order[start_index:start_index+batch_size], device=device)
            predictions = model(features[index])
            # Uniform prefix weights deliberately expose early prediction error.
            loss = (predictions - returns[index, None]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite LSTM return prediction loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        error = (model(features) - returns[:, None]).cpu().numpy()
    return dict(lstm_loss=float(np.mean(losses)), lstm_optimizer_steps=len(losses),
                lstm_seconds=perf_counter()-start, lstm_replay_episodes=len(trajectories),
                prefix_mse={str(t+1): float(np.mean(error[:,t]**2)) for t in (0, 11, 35, 59, 71)},
                prefix_bias={str(t+1): float(np.mean(error[:,t])) for t in (0, 11, 35, 59, 71)})
