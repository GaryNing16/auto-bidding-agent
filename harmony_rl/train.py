"""Bounded, resumable on-policy training with complete-trajectory supervision."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time

import numpy as np
import psutil
import torch

from harmony_baselines.data import sha256_file
from harmony_baselines.five_scenarios.common import cached_day, dump, read, ROOT, OUT, runtime_code_hashes
from harmony_bidding.schema import fingerprint, Rules
from .config import TrainConfig
from .models import ActorCritic, StateEncoder, sample_action
from .buffer import Trajectory, ReplayBuffer
from .ppo import update_ppo
from .redistribution import ReturnPredictor, trajectory_features, redistribute, train_predictor
from .policy import model_from_payload, load_payload
from .env import BiddingEnv
from .markets import MarketSampler, synthetic_day


def worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def execution_hashes():
    result = runtime_code_hashes()
    for name in ("__init__", "config", "env", "markets", "models", "buffer", "ppo", "redistribution", "policy", "train"):
        path = Path(__file__).parent / (name + ".py")
        result[str(path.relative_to(ROOT))] = sha256_file(path)
    return result


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".pending")
    torch.save(payload, temp)
    temp.replace(path)


def policy_payload(model, encoder, config):
    return {"format": "harmony-ppo-lstm-v1", "model_config": model.config,
            "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "encoder": encoder.to_dict(), "config": config.to_dict(), "training_seed": config.seed}


def rollout_job(job):
    """Worker owns matrices; only immutable Observation is passed to the model."""
    worker_init()
    cfg = TrainConfig(**job["config"])
    seed = int(job["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if cfg.smoke:
        day = synthetic_day(seed=731 if job.get("validation") else 730)
        market_provenance = {"kind": "synthetic-smoke", "not_formal_data": True}
    elif job.get("validation"):
        day = cached_day(job["day_id"])
        market_provenance = {"kind": "historical-validation"}
    else:
        days = [cached_day(d) for d in cfg.train_days]
        sampler = MarketSampler(days, seed=seed, allowed_day_ids=cfg.train_days,
            probabilities=cfg.market_probabilities if cfg.augmentation else (1., 0., 0.))
        day, market_provenance = sampler.sample()
    model, encoder = model_from_payload(job["policy"])
    validation = bool(job.get("validation"))
    budget_options = {} if validation else dict(job.get("budget_config", {}))
    opponent_options = {} if validation else dict(job.get("opponent_randomization", {}))
    if not validation:
        budget_options.update(traffic_multiplier=market_provenance.get("traffic_multiplier", 1.),
            traffic_budget_mode=market_provenance.get("traffic_budget_mode", "fixed"),
            scale=market_provenance.get("budget_scale", 1.))
        competition_scale = market_provenance.get("opponent_scale", 1.)
        opponent_options["low"] = opponent_options.get("low", 1.) * competition_scale
        opponent_options["high"] = opponent_options.get("high", 1.) * competition_scale
    env = BiddingEnv(day, job["player_id"], job["opponents"], job["scenario"], seed,
        budget_config=budget_options, opponent_randomization=opponent_options,
        residual_spec=cfg.residual_spec if cfg.residual else None)
    obs, info = env.reset()
    fields = {name: [] for name in ("states", "actions", "latents", "log_probs", "values",
                                   "rewards", "spends", "next_states", "dones")}
    begin = time.perf_counter()
    for _ in range(72):
        state = encoder.encode(obs)
        with torch.no_grad():
            action = sample_action(model, state, base_alpha=env.base_alpha(), deterministic=validation)
        nxt, reward, done, truncated, feedback = env.step(action["alpha"])
        if truncated:
            raise RuntimeError("Complete-trajectory training cannot use truncated episodes")
        entries = dict(states=state, actions=action["alpha"], latents=action["latent"],
            log_probs=action["log_prob"], values=action["value"], rewards=reward,
            spends=feedback["feedback"].cost, next_states=encoder.encode(nxt), dones=done)
        for key, value in entries.items():
            fields[key].append(value)
        obs = nxt
    metadata = dict(env.metadata, **info, policy_version=job["policy_version"],
        episode_index=job["episode_index"], market_generation=market_provenance,
        validation=validation, elapsed_seconds=time.perf_counter()-begin,
        rss_mb=psutil.Process().memory_info().rss / 2**20)
    trajectory = Trajectory(**{key: np.asarray(value) for key, value in fields.items()}, metadata=metadata)
    return trajectory


def _map(pool, jobs):
    if pool:
        return list(pool.map(rollout_job, jobs))
    with torch.random.fork_rng(devices=[]):
        return [rollout_job(j) for j in jobs]


def _job(config, policy, opponents, index, version, player_id, scenario, **extra):
    return dict(config=config.to_dict(), policy=policy, opponents=opponents,
        episode_index=index, policy_version=version, player_id=player_id, scenario=scenario,
        seed=int(config.seed + index * 1009), **extra)


def _save_trajectory(trajectory, output):
    index = trajectory.metadata["episode_index"]
    path = output / "trajectories" / f"episode_{index:06d}.npz"
    path.parent.mkdir(exist_ok=True)
    fields = {key: getattr(trajectory, key) for key in ("states", "actions", "latents", "log_probs",
        "values", "rewards", "spends", "next_states", "dones")}
    np.savez_compressed(path, **fields, metadata=np.array(json.dumps(trajectory.metadata)))
    return str(path)


def train(config: TrainConfig, output, resume=False):
    config.validate()
    worker_init()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not resume:
        raise ValueError("Existing training run: use --resume or a new output directory")
    opponents = read(config.opponents_path)
    if config.residual and config.residual_spec is None:
        selected = read(OUT / "experiment_B" / "selected_frozen.json")["selected"]
        config.residual_spec = next(c["spec"] for c in selected if c["algorithm"] == "LocalPID")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    exemplar = synthetic_day(seed=730) if config.smoke else cached_day(config.train_days[0])
    player_ids = [a.advertiser_id for a in exemplar.advertisers]
    player_types = sorted(set(a.advertiser_type for a in exemplar.advertisers))
    encoder = StateEncoder(player_ids, player_types)
    model = ActorCritic(encoder.state_dim, hidden=config.hidden,
                       initial_alpha=config.initial_alpha, residual=config.residual)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    predictor = ReturnPredictor(encoder.state_dim + 5, hidden=config.lstm_hidden)
    lstm_optimizer = torch.optim.Adam(predictor.parameters(), lr=config.lstm_learning_rate)
    replay = ReplayBuffer(config.replay_capacity)
    episode_count = policy_version = lstm_version = stale = 0
    best = None
    history, validation_history = [], []
    provenance = {"rules": asdict(Rules()), "code_sha256": execution_hashes(),
        "opponents_sha256": sha256_file(config.opponents_path),
        "opponents_fingerprint": fingerprint(opponents),
        "train_datasets": {d: cached_day(d).fingerprint for d in config.train_days} if not config.smoke else {},
        "validation_datasets": {d: cached_day(d).fingerprint for d in config.validation_days} if not config.smoke else {},
        "test_read_during_training": False, "smoke": config.smoke,
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "cpu_count": psutil.cpu_count(), "cuda_available": torch.cuda.is_available(),
        "normalization": "fixed analytic scales and role/type one-hot; no test statistics"}
    previous_seconds = 0.
    if resume:
        saved = load_payload(output / "last.pt")
        allowed_changes = {"max_seconds", "workers", "episodes"}
        for key, value in config.to_dict().items():
            if key not in allowed_changes and json.dumps(value, sort_keys=True) != json.dumps(saved["config"][key], sort_keys=True):
                raise ValueError("Resume configuration mismatch: " + key)
        if saved["provenance"] != provenance:
            raise ValueError("Resume input/code provenance differs")
        model.load_state_dict(saved["model_state"])
        encoder = StateEncoder.from_dict(saved["encoder"])
        optimizer.load_state_dict(saved["optimizer_state"])
        predictor.load_state_dict(saved["lstm_state"])
        lstm_optimizer.load_state_dict(saved["lstm_optimizer_state"])
        replay.load_state_dict(saved["replay"])
        episode_count, policy_version, lstm_version = (saved[k] for k in ("episode_count", "policy_version", "lstm_version"))
        best, stale = saved["best_validation"], saved["stale_validations"]
        history, validation_history = saved["history"], saved["validation_history"]
        previous_seconds = saved["elapsed_seconds"]
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
    dump(output / "config.json", config.to_dict())
    dump(output / "provenance.json", provenance)
    start = time.perf_counter()
    pool = (ProcessPoolExecutor(max_workers=config.workers, mp_context=mp.get_context("spawn"),
            initializer=worker_init) if config.workers > 1 else None)
    termination = "episode_limit"
    def checkpoint():
        payload = policy_payload(model, encoder, config)
        payload.update(optimizer_state=optimizer.state_dict(), lstm_config=predictor.config,
            lstm_state=predictor.state_dict(), lstm_optimizer_state=lstm_optimizer.state_dict(),
            replay=replay.state_dict(), episode_count=episode_count, policy_version=policy_version,
            lstm_version=lstm_version, best_validation=best, stale_validations=stale,
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            provenance=provenance, history=history, validation_history=validation_history,
            elapsed_seconds=previous_seconds+time.perf_counter()-start)
        atomic_torch_save(payload, output / "last.pt")
        return payload
    checkpoint()
    interrupted = False
    try:
        while episode_count < config.episodes:
            if time.perf_counter()-start >= config.max_seconds:
                termination = "time_limit"
                break
            policy = policy_payload(model, encoder, config)
            stop = min(episode_count+config.batch_episodes, config.episodes)
            jobs = [_job(config, policy, opponents, i, policy_version,
                player_ids[i % len(player_ids)], config.scenarios[(i // len(player_ids) + i) % len(config.scenarios)],
                budget_config={"mode": config.budget_mode, "low": config.budget_range[0], "high": config.budget_range[1]},
                opponent_randomization={"low": 1-config.opponent_randomization,
                    "high": 1+config.opponent_randomization,
                    "gain_low": 1-config.opponent_randomization,
                    "gain_high": 1+config.opponent_randomization})
                for i in range(episode_count, stop)]
            before = time.perf_counter()
            batch = _map(pool, jobs)
            sampling_seconds = time.perf_counter()-before
            # Freeze a predictor version for this entire PPO update. It is trained only afterwards.
            redistributed, max_error = None, 0.
            if config.redistribution and lstm_version > 0:
                predictor.eval()
                redistributed = []
                with torch.no_grad():
                    for trajectory in batch:
                        features = trajectory_features(trajectory, reward_scale=config.ppo["reward_scale"])
                        prediction = predictor(torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)).squeeze(0).cpu().numpy()
                        rewards = redistribute(prediction * config.ppo["reward_scale"], trajectory.rewards)
                        max_error = max(max_error, float(abs(rewards.sum()-trajectory.rewards.sum())))
                        redistributed.append(rewards)
            before = time.perf_counter()
            metrics = update_ppo(model, optimizer, batch, config.ppo,
                reward_sequences=redistributed, expected_policy_version=policy_version)
            ppo_seconds = time.perf_counter()-before
            used_lstm_version = lstm_version
            for trajectory in batch:
                replay.add(trajectory)
                _save_trajectory(trajectory, output)
            lstm_metrics, lstm_seconds = {}, 0.
            if config.redistribution:
                before = time.perf_counter()
                lstm_metrics = train_predictor(predictor, lstm_optimizer, replay,
                    epochs=config.lstm_epochs, batch_size=16, reward_scale=config.ppo["reward_scale"],
                    rng=np.random.default_rng(config.seed+policy_version))
                lstm_seconds = time.perf_counter()-before
                lstm_version += 1
                atomic_torch_save({"config": predictor.config, "state": predictor.state_dict(),
                    "version": lstm_version, "policy_version": policy_version},
                    output / "lstm_versions" / f"version_{lstm_version:04d}.pt")
            episode_count = stop
            policy_version += 1
            record = dict(update=policy_version, episodes=episode_count,
                environment_steps=episode_count*72,
                requests=sum(t.metadata["request_count"] for t in batch),
                sampling_seconds=sampling_seconds, lstm_seconds=lstm_seconds,
                used_lstm_version=used_lstm_version, return_conservation_max_error=max_error,
                rss_mb=psutil.Process().memory_info().rss/2**20,
                elapsed_seconds=previous_seconds+time.perf_counter()-start, **metrics)
            record.update(lstm_metrics)
            history.append(record)
            dump(output / "training_history.json", history)
            print(json.dumps(dict(algorithm=config.algorithm, seed=config.seed, **record)), flush=True)
            validate = policy_version % config.validate_every == 0 or episode_count >= config.episodes
            if validate and time.perf_counter()-start < config.max_seconds:
                policy = policy_payload(model, encoder, config)
                validation_jobs = [_job(config, policy, opponents, 100000+j, policy_version, role, scenario,
                    validation=True, day_id=day) for j, (day, role, scenario) in enumerate(
                    (d, p, s) for d in config.validation_days for p in config.validation_roles for s in config.scenarios)]
                before = time.perf_counter()
                validation = _map(pool, validation_jobs)
                score = float(np.mean([t.rewards.sum() for t in validation]))
                validation_history.append(dict(update=policy_version, episodes=episode_count,
                    mean_value=score, episodes_evaluated=len(validation), seconds=time.perf_counter()-before))
                dump(output / "validation_history.json", validation_history)
                improved = best is None or score > best
                if improved:
                    best, stale = score, 0
                else:
                    stale += 1
                payload = checkpoint()
                if improved:
                    atomic_torch_save(payload, output / "best.pt")
                    dump(output / "selection.json", {"criterion": "validation mean original V over complete declared grid",
                        "value": score, "update": policy_version, "episodes": episode_count,
                        "checkpoint_sha256": sha256_file(output / "best.pt"), "test_used": False})
                if stale >= config.early_stopping_patience:
                    termination = "early_stopping"
                    break
            else:
                checkpoint()
    except KeyboardInterrupt:
        termination = "interrupted"
        interrupted = True
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    if interrupted:
        # Discard any partial optimizer/rollout batch; the durable boundary is authoritative.
        committed = load_payload(output / "last.pt")
        episode_count, policy_version, lstm_version = (committed[k] for k in ("episode_count", "policy_version", "lstm_version"))
        history, best = committed["history"], committed["best_validation"]
    else:
        checkpoint()
    result = dict(algorithm=config.algorithm, training_seed=config.seed, termination=termination,
        completed_episodes=episode_count, requested_episodes=config.episodes, environment_steps=episode_count*72,
        complete=episode_count >= config.episodes, validated=(output / "best.pt").exists(),
        best_validation_value=best, ppo_updates=policy_version, lstm_versions=lstm_version,
        training_requests=sum(row["requests"] for row in history),
        elapsed_seconds=previous_seconds+time.perf_counter()-start,
        checkpoint=str(output / "best.pt") if (output / "best.pt").exists() else None,
        time_limit_scope="checked between complete rollout batches and before validation; active complete episodes finish")
    dump(output / "training_status.json", result)
    return result
