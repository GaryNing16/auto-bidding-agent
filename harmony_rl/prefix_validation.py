"""Held-out validation of frozen causal return prediction, never policy updates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from harmony_baselines.five_scenarios.common import read, dump
from harmony_baselines.data import sha256_file
from .config import TrainConfig
from .policy import load_payload
from .train import rollout_job, worker_init
from .redistribution import ReturnPredictor, predict_rewards


def evaluate_prefixes(checkpoint, output, workers=2, max_seconds=300):
    begin = time.perf_counter()
    payload = load_payload(checkpoint)
    config = TrainConfig(**payload["config"])
    if config.smoke or not config.redistribution:
        raise ValueError("Held-out return evaluation requires a real LSTM checkpoint")
    model = ReturnPredictor(**payload["lstm_config"])
    model.load_state_dict(payload["lstm_state"])
    model.eval()
    worker_init()
    opponents = read(config.opponents_path)
    jobs = []
    for day in config.validation_days:
        for role in config.validation_roles:
            for scenario in config.scenarios:
                index = 100000 + len(jobs)
                jobs.append(dict(config=config.to_dict(), policy={k: payload[k] for k in
                    ("format", "model_config", "model_state", "encoder", "config")},
                    opponents=opponents, episode_index=index, policy_version=payload["policy_version"],
                    player_id=role, scenario=scenario, seed=config.seed+index*1009,
                    validation=True, day_id=day))
    records = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=worker_init) as pool:
        for offset in range(0, len(jobs), workers):
            if time.perf_counter()-begin >= max_seconds:
                break
            for trajectory in pool.map(rollout_job, jobs[offset:offset+workers]):
                redistributed, predictions = predict_rewards(model, trajectory, config.ppo["reward_scale"])
                records.append(dict(player_id=trajectory.metadata["player_id"],
                    scenario=trajectory.metadata["scenario"], market_id=trajectory.metadata["market_id"],
                    initial_budget=trajectory.metadata["initial_budget"], realized_return=trajectory.episode_return,
                    prefix_predictions=predictions.tolist(),
                    conservation_error=float(redistributed.sum()-trajectory.episode_return)))
    errors = np.array([np.array(r["prefix_predictions"])-r["realized_return"] for r in records])
    result = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256_file(checkpoint),
        data_split="validation", days=list(config.validation_days), actor_updated=False, lstm_updated=False,
        completed=len(records), expected=len(jobs), complete=len(records)==len(jobs), records=records,
        mse_by_prefix=np.mean(errors**2, axis=0).tolist() if len(errors) else [],
        bias_by_prefix=np.mean(errors, axis=0).tolist() if len(errors) else [],
        elapsed_seconds=time.perf_counter()-begin,
        interpretation="Validation-selected actor; predictor is frozen, no validation return used for fitting")
    dump(Path(output) / "prefix_validation.json", result)
    if len(errors):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(np.arange(1, 73), result["mse_by_prefix"])
        axes[0].set(xlabel="Prefix length", ylabel="Held-out return MSE")
        axes[1].plot(np.arange(1, 73), result["bias_by_prefix"])
        axes[1].set(xlabel="Prefix length", ylabel="Held-out mean prediction error")
        fig.tight_layout()
        fig.savefig(Path(output) / "prefix_validation.png", dpi=150)
        plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=300)
    args = parser.parse_args()
    result = evaluate_prefixes(args.checkpoint, args.output, args.workers, args.max_seconds)
    print({"completed": result["completed"], "expected": result["expected"], "seconds": result["elapsed_seconds"]}, flush=True)


if __name__ == "__main__":
    main()
