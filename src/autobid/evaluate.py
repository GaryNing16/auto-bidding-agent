"""Local comparisons; the local pacing controller is not the organizer's PID."""
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter_ns
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess

from .engine import DaySimulator
from .policies import FixedPolicy, PacingPolicy
from .adapters import sha256_file


def make_policy(spec, model=None):
    if spec == 'pacing':
        return PacingPolicy()
    if spec == 'ppo':
        from .rl import PPOPolicy
        if model is None:
            raise ValueError('ppo requires --model')
        return PPOPolicy(model)
    if spec == 'fixed' or spec.startswith('fixed:'):
        return FixedPolicy(float(spec.split(':')[1]) if ':' in spec else 1.0)
    raise ValueError(f'unknown policy: {spec}')


def percentile(values, q):
    if not values:
        return 0.0
    vals = sorted(values)
    return vals[max(0, math.ceil(len(vals) * q) - 1)]


def run_episode(requests, config, policies, *, audit=False):
    """Call all bidding methods before any feedback. Never pass the engine to policies."""
    sim = DaySimulator(requests, config, audit=audit)
    for player, policy in policies.items():
        policy.reset(sim.observation(player))
    rows, latencies = [], {p: [] for p in config.budgets}
    while not sim.done:
        actions, elapsed = {}, {}
        for player, policy in policies.items():
            start = perf_counter_ns()
            actions[player] = policy.bidding(sim.observation(player))
            elapsed[player] = (perf_counter_ns() - start) / 1e6
        feedback = sim.step(actions)
        for player, policy in policies.items():
            start = perf_counter_ns()
            policy.update(feedback[player])
            latency = elapsed[player] + (perf_counter_ns() - start) / 1e6
            latencies[player].append(latency)
            rows.append({'player_id': player, **asdict(feedback[player]), 'bidding_update_ms': latency})
    summaries = {}
    for player in config.budgets:
        own = [r for r in rows if r['player_id'] == player]
        cost = sum(r['cost'] for r in own)
        summaries[player] = {
            'value': sum(r['value'] for r in own), 'cost': cost,
            'wins': sum(r['wins'] for r in own), 'budget': config.budgets[player],
            'budget_utilization': cost/config.budgets[player] if config.budgets[player] else 0.0,
            'remaining_budget': own[-1]['remaining_budget'],
            'bidding_update_p99_ms': percentile(latencies[player], .99),
        }
    return {'players': summaries, 'steps': rows, 'audit': sim.audit}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_manifest(data_paths, config, *, command, extra=None):
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = None, True
    source = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(source.rglob('*.py')):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    dependencies = {}
    for package in ['numpy', 'gymnasium', 'stable-baselines3', 'torch']:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        'command': command, 'config': asdict(config), 'python': platform.python_version(),
        'code_revision': revision, 'working_tree_dirty': dirty, 'source_sha256': digest.hexdigest(),
        'dependencies': dependencies,
        'data': [{'path': str(p), 'sha256': sha256_file(p)} for p in data_paths],
        'extra': extra or {}, 'comparison_baseline': 'local_pacing_NOT_official_PID',
    }


def evaluate(requests_by_path, config, output, *, strategies=('fixed:1', 'pacing'), seeds=(7,), roles=None, model=None, audit=False):
    if not requests_by_path or not strategies or not seeds:
        raise ValueError('data, strategies and seeds must be nonempty')
    if model is not None:
        model = Path(model)
        if model.suffix != '.zip':
            model = model.with_suffix('.zip')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    roles = list(config.budgets) if roles is None else roles
    if not roles or any(p not in config.budgets for p in roles):
        raise ValueError('evaluation roles must be configured players')
    rows = []
    for day, (path, requests) in enumerate(requests_by_path.items()):
        for seed in seeds:
            cfg = replace(config, seed=seed)
            baseline = run_episode(requests, cfg, {p: PacingPolicy() for p in cfg.budgets})
            for role_index, player in enumerate(roles):
                denominator = baseline['players'][player]['value']
                for strategy_index, spec in enumerate(strategies):
                    policies = {p: PacingPolicy() for p in cfg.budgets}
                    policies[player] = make_policy(spec, model)
                    episode = run_episode(requests, cfg, policies, audit=audit)
                    result = episode['players'][player]
                    run_id = f'day{day}-seed{seed}-role{role_index}-policy{strategy_index}'
                    rows.append({'run_id': run_id, 'data': str(path), 'seed': seed, 'player_id': player,
                                 'strategy': spec, **result, 'local_pacing_value': denominator,
                                 'relative_to_local_pacing': result['value']/denominator if denominator > 0 else None})
                    save_csv(output / f'{run_id}.csv', episode['steps'])
                    if audit:
                        save_json(output / f'{run_id}-audit.json', episode['audit'])
    aggregates = {}
    for spec in strategies:
        group = [r for r in rows if r['strategy'] == spec]
        ratios = [r['relative_to_local_pacing'] for r in group if r['relative_to_local_pacing'] is not None]
        aggregates[spec] = {
            'episodes': len(group), 'mean_value': sum(r['value'] for r in group)/len(group),
            'mean_episode_ratio_to_local_pacing': sum(ratios)/len(ratios) if ratios else None,
            'undefined_zero_baseline_episodes': len(group)-len(ratios),
            'max_episode_bidding_update_p99_ms': max(r['bidding_update_p99_ms'] for r in group),
        }
    report = {'baseline': 'local_pacing_NOT_official_PID', 'zero_baseline_policy': 'null_and_excluded_with_count',
              'aggregates': aggregates, 'episodes': rows}
    save_json(output / 'summary.json', report)
    save_csv(output / 'summary.csv', rows)
    save_json(output / 'manifest.json', run_manifest(requests_by_path, config, command='evaluate',
              extra={'strategies': list(strategies), 'seeds': list(seeds), 'roles': roles, 'model': str(model) if model else None,
                     'model_sha256': sha256_file(model) if model else None}))
    return report


def arena(requests, config, output, lineup, *, model=None):
    players = list(config.budgets)
    if len(lineup) != len(players):
        raise ValueError('lineup must provide one strategy per configured player')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rotations = []
    for rotation in range(len(players)):
        assignments = {p: lineup[(i+rotation) % len(lineup)] for i, p in enumerate(players)}
        result = run_episode(requests, config, {p: make_policy(s, model) for p,s in assignments.items()})
        rotations.append({'rotation': rotation, 'assignments': assignments, 'players': result['players']})
        save_csv(output/f'rotation{rotation}.csv', result['steps'])
    save_json(output/'arena.json', {'rotations': rotations})
    return rotations
