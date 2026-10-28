"""Independently reload a frozen baseline and reproduce one completed test game.

Run from AuctionNet: python scripts/replay_baseline_episode.py --algorithm BC
This command never trains, chooses a model, changes parameters, or writes scores.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from harmony_baselines.report import audit_complete_results, DEFAULT_OUT
    from harmony_baselines.data import load_day
    from harmony_baselines.evaluator import run_episode
    import numpy as np
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(DEFAULT_OUT))
    parser.add_argument('--algorithm', default='BC')
    parser.add_argument('--scenario', choices=['Fixed', 'PID', 'Mixed'], default='Mixed')
    parser.add_argument('--player', default='0')
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--training-seed', type=int)
    parser.add_argument('--day', default='period-24')
    args = parser.parse_args()
    out = Path(args.root)
    frame, protocol, frozen = audit_complete_results(out)
    candidates = [c for c in frozen['selected'] if c['algorithm'] == args.algorithm
        and (args.training_seed is None or c.get('training_seed', 0) == args.training_seed)]
    if not candidates:
        raise ValueError('no frozen candidate for requested algorithm and training seed')
    candidate = candidates[0]
    train_seed = candidate.get('training_seed', 0)
    expected = frame[(frame.algorithm == args.algorithm) & (frame.scenario == args.scenario)
        & (frame.player_id == args.player) & (frame.seed == args.seed)
        & (frame.training_seed == train_seed) & (frame.day_id == args.day)]
    if len(expected) != 1:
        raise ValueError('expected exactly one completed episode')
    reference = expected.iloc[0]
    opponents = json.loads((out/'opponents_frozen.json').read_text(encoding='utf-8'))
    actual, _, _ = run_episode(load_day(args.day), args.player, candidate['spec'], args.scenario,
        opponents, args.seed, budget_scale=float(reference.budget_scale), algorithm=args.algorithm,
        training_seed=train_seed)
    fields = ['total_value', 'total_cost', 'budget_utilization', 'wins']
    errors = {field: abs(float(actual[field])-float(reference[field])) for field in fields}
    passed = all(np.isclose(actual[field], reference[field], rtol=1e-12, atol=1e-12) for field in fields)
    print(json.dumps({'algorithm':args.algorithm, 'scenario':args.scenario, 'player_id':args.player,
        'seed':args.seed, 'training_seed':train_seed, 'matched':bool(passed), 'absolute_errors':errors,
        'metrics':{field:actual[field] for field in fields}}, ensure_ascii=False))
    if not passed:
        raise AssertionError('independent frozen-checkpoint replay differs from published CSV')


if __name__ == '__main__':
    main()
