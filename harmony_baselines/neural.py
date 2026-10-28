"""Causal AuctionNet adapters and auditable offline training entry points."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from harmony_bidding.agent_api import Agent
from .upstream import make_model, provenance

FEATURE_NAMES = [
    "time_fraction", "remaining_fraction", "log_initial_budget", "pacing_error",
    "last_log_alpha", "last_cost_fraction", "last_log_value", "last_log_wins",
    "cumulative_log_value", "cumulative_log_wins",
    "last3_log_alpha", "last3_cost_fraction", "last3_log_value", "last3_log_wins",
    "last6_log_alpha", "last6_cost_fraction", "last6_log_value", "last6_log_wins",
]
FEATURE_VERSION = "own-history-18-v1"
ALGORITHMS = ("BC", "IQL", "BCQ", "TD3BC", "CQL", "DT")
SUPPORTED_ACTION_TRANSFORMS = ("log1p", "linear_scaled")


def _action_transform(config, metadata=None):
    """Old checkpoints without a coordinate tag retain their exact log mapping."""
    metadata = metadata or {}
    configured = config.get("action_transform")
    recorded = metadata.get("action_transform")
    if configured is not None and recorded is not None and configured != recorded:
        raise ValueError("Checkpoint action_transform differs between config and metadata")
    transform = configured if configured is not None else recorded if recorded is not None else "log1p"
    if transform not in SUPPORTED_ACTION_TRANSFORMS:
        raise ValueError(f"Unsupported action_transform: {transform!r}")
    return transform


def encode_actions(actions, action_scale, action_transform="log1p"):
    """Training-only action coordinates; neither transform imposes an alpha cap."""
    transform = _action_transform({"action_transform": action_transform})
    scaled = np.asarray(actions) / action_scale
    return np.log1p(scaled) if transform == "log1p" else scaled


def encode_observation(observation):
    """Only the frozen public clock, own config and *past* own feedback."""
    budget = observation.own_config.initial_budget
    denominator = budget if budget > 0 else 1.
    history = observation.history
    t = observation.time_slot / observation.total_slots
    remaining = observation.remaining_budget / denominator
    last = history[-1] if history else None
    features = [t, remaining, math.log1p(budget), t - (1 - remaining),
                math.log1p(last.alpha) if last else 0., last.cost / denominator if last else 0.,
                math.log1p(last.value) if last else 0., math.log1p(last.wins) if last else 0.,
                math.log1p(last.cumulative_value) if last else 0.,
                math.log1p(sum(x.wins for x in history))]
    for count in (3, 6):
        window = history[-count:]
        n = len(window) or 1
        features.extend([sum(math.log1p(x.alpha) for x in window) / n,
                         sum(x.cost for x in window) / n / denominator,
                         math.log1p(sum(x.value for x in window) / n),
                         math.log1p(sum(x.wins for x in window) / n)])
    array = np.asarray(features, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite legal history encoding")
    return array


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_training_data(path):
    with np.load(path, allow_pickle=False) as source:
        data = {key: np.asarray(source[key]) for key in
                ("states", "actions", "rewards", "next_states", "dones", "episode_id")}
    n = len(data["states"])
    if n == 0 or data["states"].shape != (n, len(FEATURE_NAMES)) or data["next_states"].shape != data["states"].shape:
        raise ValueError("Expected nonempty causal 18-feature transitions")
    for key in ("actions", "rewards", "dones"):
        data[key] = data[key].reshape(n, 1).astype(np.float32)
    for key in ("states", "actions", "rewards", "next_states", "dones"):
        if not np.isfinite(data[key]).all():
            raise ValueError("Nonfinite training data: " + key)
    if np.any(data["actions"] < 0) or np.any(data["rewards"] < 0) or not np.isin(data["dones"], [0, 1]).all():
        raise ValueError("Invalid alpha, value reward, or terminal flag")
    episode_ids = data["episode_id"].reshape(n)
    boundaries = np.r_[0, np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1, n]
    episodes = [np.arange(a, b) for a, b in zip(boundaries[:-1], boundaries[1:])]
    if len(set(str(episode_ids[e[0]]) for e in episodes)) != len(episodes):
        raise ValueError("Episode IDs must identify contiguous, unique trajectories")
    for ids in episodes:
        if len(ids) != 72 or data["dones"][ids[-1], 0] != 1 or data["dones"][ids[:-1]].any():
            raise ValueError("Each legal training episode must cover all 72 slots")
    data["episode_id"] = episode_ids
    return data, episodes


class NeuralPolicy:
    def __init__(self, algorithm, seed, config, metadata):
        import torch
        self.algorithm, self.seed, self.config, self.metadata = algorithm, int(seed), dict(config), dict(metadata)
        self.action_transform = _action_transform(self.config, self.metadata)
        self.config["action_transform"] = self.action_transform
        self.metadata["action_transform"] = self.action_transform
        self.model = make_model(algorithm, len(FEATURE_NAMES), self.seed, self.config)
        self.state_mean = np.asarray(metadata["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(metadata["state_std"], dtype=np.float32)
        self.action_scale = float(metadata["action_scale"])
        self.reward_scale = float(metadata["reward_scale"])
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.model.eval()

    def save(self, path, step, elapsed_seconds):
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"algorithm": self.algorithm, "seed": self.seed, "config": self.config,
                   "metadata": self.metadata, "feature_version": FEATURE_VERSION,
                   "action_transform": self.action_transform,
                   "state_dict": self.model.state_dict(), "step": int(step),
                   "training_seconds": float(elapsed_seconds), "parameter_count": self.parameter_count}
        # CQL temperature is an upstream plain Tensor rather than a Parameter.
        if self.algorithm == "CQL":
            payload["cql_log_alpha"] = self.model.log_alpha.detach().cpu()
        torch.save(payload, path)
        return {key: payload[key] for key in payload if key not in ("state_dict", "cql_log_alpha")}

    @classmethod
    def load(cls, path):
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["feature_version"] != FEATURE_VERSION:
            raise ValueError("Unsupported feature contract")
        policy = cls(payload["algorithm"], payload["seed"], payload["config"], payload["metadata"])
        if payload.get("action_transform", policy.action_transform) != policy.action_transform:
            raise ValueError("Checkpoint top-level action_transform differs from its configuration")
        policy.model.load_state_dict(payload["state_dict"], strict=True)
        if policy.algorithm == "CQL":
            policy.model.log_alpha.data.copy_(payload["cql_log_alpha"])
        policy.model.eval()
        return policy

    def normalize(self, state):
        return (np.asarray(state, dtype=np.float32) - self.state_mean) / self.state_std

    def decode(self, action):
        coordinate = float(np.asarray(action).reshape(-1)[0])
        if not math.isfinite(coordinate):
            raise ValueError("Network produced nonfinite alpha coordinate")
        if self.action_transform == "linear_scaled":
            alpha = self.action_scale * max(0., coordinate)
        else:
            # Preserve the exact legacy expression, including floating rounding.
            alpha = self.action_scale * math.expm1(max(0., coordinate))
        if not math.isfinite(alpha):
            raise ValueError("Network produced overflowing alpha")
        return alpha

    def action(self, state, *, random_seed=0, sequence=None):
        import torch
        tensor = torch.from_numpy(self.normalize(state)).reshape(1, -1)
        with torch.inference_mode():
            if self.algorithm == "DT":
                states, actions, returns, timesteps = sequence
                result = self.model.get_action(torch.as_tensor(np.asarray(states), dtype=torch.float32),
                                              torch.as_tensor(np.asarray(actions), dtype=torch.float32),
                                              torch.zeros(len(states), 1),
                                              torch.as_tensor(returns, dtype=torch.float32),
                                              torch.as_tensor(timesteps, dtype=torch.long))
            elif self.algorithm == "BCQ":
                # Local per-call RNG: other target algorithms and evaluation order
                # cannot consume this agent's random sequence.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(random_seed) % (2**63 - 1))
                    result = self.model(tensor)
            elif self.algorithm == "IQL":
                result = self.model.actors.get_det_action(tensor)
            elif self.algorithm == "CQL":
                result = self.model(tensor, eval_flag=True)
            else:
                result = self.model(tensor)
        coordinate = float(result.detach().cpu().reshape(-1)[0])
        return self.decode(coordinate), max(0., coordinate)


@lru_cache(maxsize=24)
def _cached_policy(path, size, mtime):
    return NeuralPolicy.load(path)


class NeuralAgent(Agent):
    """Independent episode state, immutable shared network weights."""
    def __init__(self, checkpoint):
        super().__init__()
        path = Path(checkpoint).resolve()
        stat = path.stat()
        self.policy = _cached_policy(str(path), stat.st_size, stat.st_mtime_ns)

    def reset(self, own_config, public_config):
        self.own_config, self.public_config = own_config, public_config
        self.states, self.actions, self.returns, self.timesteps = [], [], [], []
        self.return_remaining = float(self.policy.config.get("target_return", 1.))
        self.feedback_count = 0

    def bidding(self, observation):
        state = encode_observation(observation)
        sequence = None
        if self.policy.algorithm == "DT":
            self.states.append(self.policy.normalize(state))
            self.actions.append([0.])
            self.returns.append(self.return_remaining)
            self.timesteps.append(observation.time_slot)
            sequence = (self.states, self.actions, self.returns, self.timesteps)
        alpha, coordinate = self.policy.action(
            state, random_seed=self.own_config.random_seed * 73 + observation.time_slot, sequence=sequence)
        if self.policy.algorithm == "DT":
            self.actions[-1] = [coordinate]
        return alpha

    def update(self, own_feedback):
        if own_feedback.time_slot != self.feedback_count:
            raise ValueError("Neural agent feedback out of sequence")
        self.return_remaining -= own_feedback.value / self.policy.reward_scale
        self.feedback_count += 1


def train_candidates(data_path, output_dir, algorithms=ALGORITHMS, seeds=(20261028,),
                     steps=600, checkpoint_steps=(200, 600), configs=None, batch_size=64,
                     num_threads=2):
    """Train only on the supplied training NPZ; return every candidate checkpoint.

    No validation/test market is accepted by this function. The experiment runner
    must compare returned checkpoints on its sealed validation protocol.
    """
    import torch
    torch.set_num_threads(int(num_threads))
    data, episodes = load_training_data(data_path)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Training output must be new or empty; refusing stale checkpoints/status")
    out.mkdir(parents=True, exist_ok=True)
    state_mean = data["states"].mean(axis=0)
    state_std = data["states"].std(axis=0)
    state_std[state_std < 1e-6] = 1.
    positive_actions = data["actions"][data["actions"] > 0]
    action_scale = float(np.median(positive_actions)) if positive_actions.size else 1.
    episode_returns = np.asarray([float(data["rewards"][e].sum()) for e in episodes])
    reward_scale = float(np.quantile(episode_returns, .75)) or 1.
    metadata = {"state_mean": state_mean.tolist(), "state_std": state_std.tolist(),
                "action_scale": action_scale, "reward_scale": reward_scale,
                "feature_names": FEATURE_NAMES, "source_sha256": _file_hash(data_path),
                "transitions": len(data["states"]), "episodes": len(episodes),
                "upstream": provenance(), "device": "cpu", "torch_version": str(torch.__version__)}
    normalized = {"states": (data["states"] - state_mean) / state_std,
                  "next_states": (data["next_states"] - state_mean) / state_std,
                  "rewards": data["rewards"] / reward_scale, "dones": data["dones"]}
    common_tensors = {key: torch.from_numpy(value.astype(np.float32)) for key, value in normalized.items()}
    returns_to_go = np.zeros(len(data["states"]), dtype=np.float32)
    for episode in episodes:
        returns_to_go[episode] = np.cumsum(normalized["rewards"][episode, 0][::-1])[::-1]
    candidate_records, failures = [], []
    for algorithm in algorithms:
        variants = (configs or {}).get(algorithm, [{"learning_rate": 3e-4}, {"learning_rate": 1e-3}])
        for config_index, variant in enumerate(variants):
            action_transform = _action_transform(variant)
            # Config-local arrays prevent a CQL linear-coordinate candidate from
            # changing later legacy candidates or DT's past-action tokens.
            action_coordinates = encode_actions(data["actions"], action_scale, action_transform)
            tensors = {**common_tensors, "actions": torch.from_numpy(action_coordinates.astype(np.float32))}
            max_action = max(1., float(action_coordinates.max()) * 1.25)
            for seed in seeds:
                config = {"max_action": max_action, "target_return": float(np.quantile(episode_returns / reward_scale, .9)),
                          "hidden_width": 128, "gamma": 1., **variant, "action_transform": action_transform}
                run = out / algorithm / f"config_{config_index}_seed_{seed}"
                run.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                losses = []
                try:
                    policy = NeuralPolicy(algorithm, seed, config, metadata)
                    policy.model.train()
                    rng = np.random.default_rng(int(seed))
                    for step in range(1, int(steps) + 1):
                        if algorithm == "DT":
                            k = int(config.get("context_length", 10))
                            bs = min(int(batch_size), 32)
                            seq_s = np.zeros((bs, k, len(FEATURE_NAMES)), dtype=np.float32)
                            seq_a = np.zeros((bs, k, 1), dtype=np.float32)
                            seq_r = np.zeros((bs, k, 1), dtype=np.float32)
                            seq_rt = np.zeros((bs, k + 1, 1), dtype=np.float32)
                            seq_t = np.zeros((bs, k), dtype=np.int64)
                            seq_m = np.zeros((bs, k), dtype=np.float32)
                            for b in range(bs):
                                episode = episodes[int(rng.integers(len(episodes)))]
                                end = int(rng.integers(1, len(episode) + 1))
                                ids = episode[max(0, end - k):end]
                                n = len(ids)
                                seq_s[b, -n:] = normalized["states"][ids]
                                seq_a[b, -n:] = action_coordinates[ids]
                                seq_r[b, -n:] = normalized["rewards"][ids]
                                seq_rt[b, k-n:k, 0] = returns_to_go[ids]
                                seq_t[b, -n:] = np.arange(end-n, end)
                                seq_m[b, -n:] = 1
                            loss = policy.model.step(torch.from_numpy(seq_s), torch.from_numpy(seq_a),
                                torch.from_numpy(seq_r), torch.zeros(bs, k), torch.from_numpy(seq_rt),
                                torch.from_numpy(seq_t), torch.from_numpy(seq_m))
                            policy.model.scheduler.step()
                        else:
                            ids = rng.integers(len(data["states"]), size=int(batch_size))
                            batch = [tensors[key][ids] for key in ("states", "actions", "rewards", "next_states", "dones")]
                            loss = policy.model.step(batch[0].numpy(), batch[1].numpy()) if algorithm == "BC" else policy.model.step(*batch)
                        values = loss if isinstance(loss, (tuple, list)) else [loss]
                        values = [float(np.asarray(v.detach().cpu() if isinstance(v, torch.Tensor) else v).mean())
                                  for v in values if v is not None]
                        if not all(math.isfinite(v) for v in values):
                            raise FloatingPointError(f"Nonfinite training loss at step {step}")
                        if step == 1 or step % 50 == 0:
                            losses.append({"step": step, "losses": values, "elapsed_seconds": time.perf_counter()-started})
                        if step in checkpoint_steps or step == steps:
                            path = run / f"step_{step}.pt"
                            record = policy.save(path, step, time.perf_counter()-started)
                            record.update({"checkpoint": str(path.resolve()), "config_index": config_index,
                                           "checkpoint_sha256": _file_hash(path)})
                            candidate_records.append(record)
                            _json(out / "candidates.json", candidate_records)
                    _json(run / "training.json", {"status": "completed", "history": losses,
                           "seconds": time.perf_counter()-started, "parameters": policy.parameter_count,
                           "config": config, "seed": int(seed)})
                except Exception as exc:
                    failure = {"algorithm": algorithm, "seed": int(seed), "config": config,
                               "error": repr(exc), "seconds": time.perf_counter()-started}
                    failures.append(failure)
                    _json(run / "training.json", {"status": "failed", **failure, "history": losses})
                _json(out / "failures.json", failures)
    _json(out / "training_metadata.json", {**metadata,
        "action_transforms": sorted({r["action_transform"] for r in candidate_records}),
        "action_transform_scope": "each candidate config/checkpoint; legacy default log1p"})
    return candidate_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS))
    args = parser.parse_args()
    records = train_candidates(args.data, args.output, algorithms=args.algorithms, seeds=(args.seed,),
                               steps=args.steps, checkpoint_steps=(max(1, args.steps//3), args.steps))
    print(json.dumps({"candidates": len(records), "output": args.output}))


if __name__ == "__main__":
    main()
