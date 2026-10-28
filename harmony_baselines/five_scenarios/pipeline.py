"""Predeclared A/B five-scenario experiment, separate from immutable history."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import sys

import numpy as np
import pandas as pd
from harmony_baselines.classical import spec, fit_lp_table
from harmony_baselines.data import build_cache, sha256_file
from harmony_bidding.schema import fingerprint
from .common import (ROOT, OLD, OUT, SCENARIOS, OLD_SCENARIOS, TRAIN_DAYS, VALID_DAYS,
                     COMPARISON_DAYS, INDEPENDENT_DAYS, SEEDS, ROLES, VALID_ROLES,
                     NEURAL, ALGORITHMS, read, dump, immutable, assets, policy_id,
                     runtime_code_hashes, cached_day, pid_spec)
from .execution import execute, tasks_for


def control_grid():
    return [
        dict(lower_threshold=.7, upper_threshold=1.1, increase_factor=1.2, decrease_factor=.7),
        dict(lower_threshold=.7, upper_threshold=1.1, increase_factor=1.1, decrease_factor=.85),
        dict(lower_threshold=.7, upper_threshold=1.1, increase_factor=1.4, decrease_factor=.5),
        dict(lower_threshold=.5, upper_threshold=1.3, increase_factor=1.2, decrease_factor=.7),
        dict(lower_threshold=.9, upper_threshold=1.05, increase_factor=1.2, decrease_factor=.7),
    ]


def base_alpha():
    return float(read(OLD / 'calibration.json')['base_alpha'])


def alpha_grid():
    a = base_alpha()
    return [15., a/2, a, 2*a, 4*a]


def _verify_historical(full=False):
    audit = read(OUT / 'historical_audit.json')
    if audit.get('status') != 'passed' or audit.get('valid') is not True or audit.get('mismatches'):
        raise ValueError('Historical audit is not valid')
    manifest_path = OUT / 'historical_assets_manifest.json'
    if sha256_file(manifest_path) != audit['manifest_sha256']:
        raise ValueError('Historical protection manifest changed')
    protection = {str(Path(r['path']).resolve()): r['sha256'] for r in read(manifest_path)['files']}
    essentials = [OLD/'selected_frozen.json', OLD/'opponents_frozen.json', OLD/'online_lp_table.csv',
                  OLD/'episode_results.csv', OLD/'neural_validation.csv', OLD/'protocol.json']
    for path in essentials:
        if sha256_file(path) != protection.get(str(path.resolve())):
            raise ValueError('Protected historical input changed: ' + str(path))
    old = read(OLD / 'selected_frozen.json')
    if sha256_file(OLD/'opponents_frozen.json') != old['opponents_sha256'] or sha256_file(OLD/'online_lp_table.csv') != old['lp_table_sha256']:
        raise ValueError('Historical opponent/LP freeze differs')
    for path, expected in old['checkpoints'].items():
        if sha256_file(path) != expected:
            raise ValueError('Historical selected checkpoint changed')
    for path, expected in old['code_sha256'].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError('Historical execution source changed')
    if full:
        from .historical_audit import verify
        result = verify(OUT)
        if result['status'] != 'passed':
            raise ValueError('Historical asset verification failed: ' + json.dumps(result['mismatches']))
        dump(OUT/'historical_integrity_preflight.json', result)
    return old


def _require_protocol():
    path = OUT/'protocol.json'
    if not path.exists():
        raise ValueError('Run prepare before any experimental stage')
    protocol = read(path)
    if (protocol['historical_selection_sha256'] != sha256_file(OLD/'selected_frozen.json') or
            protocol['historical_results_sha256'] != sha256_file(OLD/'episode_results.csv') or
            protocol['scenarios'] != list(SCENARIOS) or protocol['algorithms'] != list(ALGORITHMS)):
        raise ValueError('Predeclared protocol no longer matches inputs')
    return protocol


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    _verify_historical(full=True)
    protocol = {
        'run_id': OUT.name, 'rules_version': 'harmony-A-B-local-v2',
        'upstream_commit': 'c20de631a40e24669d1d6a8e2acefd8f492f172d',
        'train_days': list(TRAIN_DAYS), 'validation_days': list(VALID_DAYS),
        'historically_exposed_comparison_days': list(COMPARISON_DAYS),
        'predeclared_independent_test_days': list(INDEPENDENT_DAYS),
        'independent_test_rule': 'Do not load period-25 pValues until B models and execution sources are frozen. Never retune after either test.',
        'scenarios': list(SCENARIOS), 'algorithms': list(ALGORITHMS),
        'roles': list(ROLES), 'validation_roles': list(VALID_ROLES),
        'evaluation_seeds': list(SEEDS), 'training_seeds': list(SEEDS),
        'full_episode_slots': 72, 'budget_scale': 1., 'workers': 4, 'episode_timeout_seconds': 180.,
        'A': {'old_9': 'unaltered params/checkpoints; reuse 4320 historical cases',
              'new_old9_cases': 2880, 'new_AuctionNetPID_cases': 480,
              'total_new_cases': 3360, 'total_cases': 7680,
              'attachment_estimate_2208': 'does not cover required 2 training seeds x 2 evaluation seeds',
              'AuctionNetPID_selection': 'old 3 scenarios, same 6 validation roles, mean relative value against old frozen LocalPID; preserve historical selection convention'},
        'B': {'selection_primary': 'mean validation total_value, equal scenarios and roles, average both training seeds',
              'selection_tiebreak': 'lexicographic immutable candidate ID; only complete candidates',
              'diagnostics': ['mean_R', 'budget_utilization', 'worst_scenario_V', 'role_P10_V', 'role_P10_R'],
              'total_comparison_cases': 7680, 'independent_cases': 7680,
              'reuse_rule': 'only identical policy spec and checkpoint/table hashes with same opponent assignment, market, rules and seeds'},
        'opponents': {'fixed_localpid': 'exact historical specs, never recalibrated',
                      'auctionnet_pid_train': 'period-7, all48 roles vs historical Fixed opponents, five initial alpha candidates',
                      'auctionnet_pid_validation': 'period-21, six roles, original3 scenarios, five threshold/multiplier configurations, primary mean V',
                      'mixed2': 'same seeded 23 Fixed members as Mixed; remaining24 AuctionNetPID; each agent fresh independent reset'},
        'classical_search': {'AuctionNetPID_initial_alphas': alpha_grid(), 'AuctionNetPID_controls': control_grid(),
                             'FixedAlpha_multipliers': [.5, 1., 1.5, 2., 3., 4., 6.],
                             'LocalPID_initial_multipliers': [.5, 1., 2., 4.],
                             'LocalPID_gains': [[.25, .025, .025], [.5, .05, .05], [1.5, .1, .15]],
                             'OnlineLP_multipliers': [.25, .5, .75, 1., 1.5, 2.],
                             'lp_fit': {'days': ['period-7'], 'roles': list(ROLES), 'scenarios': list(SCENARIOS),
                                        'constant_alpha_multipliers': [.5, 1., 2., 4.], 'episodes': 960}},
        'training_data': {'old_episodes': 288, 'new_episodes': 192, 'total_episodes': 480,
                          'transitions': 34560, 'episodes_per_scenario': 96,
                          'behavior': 'same six historical Fixed/LocalPID exploration specs; new scenes use global scene indices3/4; no new scenario feature'},
        'neural_training': {'initialization': 'from scratch', 'runs': 24, 'checkpoints': 48,
                            'steps': 1800, 'checkpoint_steps': [600, 1800], 'batch_size': 64,
                            'DT_sequence_batch': 32, 'DT_context': 10, 'hidden_models': 'unchanged historical adapters',
                            'learning_rates': [3e-4, 1e-3], 'CQL_learning_rates': [1e-4, 3e-4],
                            'CQL_action_transform': 'linear_scaled (same as historical selected CQL)',
                            'other_action_transform': 'log1p', 'num_threads': 2},
        'historical_selection_sha256': sha256_file(OLD / 'selected_frozen.json'),
        'historical_results_sha256': sha256_file(OLD / 'episode_results.csv'),
        'historical_audit_sha256': sha256_file(OUT / 'historical_audit.json'),
    }
    immutable(OUT / 'protocol.json', protocol)
    for day in TRAIN_DAYS + VALID_DAYS:
        cached_day(day)
    return protocol


def _frame(path):
    return pd.read_csv(path, dtype={'player_id': str, 'player_type': str, 'day_id': str})


def _candidate_scores(frame, candidates, tasks, reference=None):
    """Evaluate completed candidate groups; no missing/failed score filling."""
    keys = ['day_id', 'scenario', 'player_id', 'seed']
    if not frame.empty:
        if frame.task_index.duplicated().any():
            raise ValueError('Duplicate validation task result')
        for row in frame.itertuples():
            index = int(row.task_index)
            if index != row.task_index or not 0 <= index < len(tasks):
                raise ValueError('Validation task index outside protocol')
            task = tasks[index]
            for key, target_key in [('algorithm','algorithm'), ('training_seed','training_seed'),
                                    ('day_id','day_id'), ('scenario','scenario'), ('player_id','target'), ('seed','seed')]:
                if str(getattr(row,key)) != str(task.get(target_key,0)):
                    raise ValueError('Validation row/task identity mismatch: '+key)
            if not np.isfinite([row.total_value, row.budget_utilization]).all() or row.total_value < 0:
                raise ValueError('Invalid validation score')
    ref = None
    if reference is not None:
        reference = reference.copy()
        reference['player_id'] = reference.player_id.astype(str)
        reference['scenario'] = reference.scenario.replace({'PID': 'LocalPID'})
        ref = reference.groupby(keys, dropna=False).total_value.mean()
    scores = []
    for name in dict.fromkeys(c['algorithm'] for c in candidates):
        expected = {i for i, t in enumerate(tasks) if t['algorithm'] == name}
        part = frame[frame.algorithm == name].copy() if not frame.empty else pd.DataFrame()
        observed = set(part.task_index.astype(int)) if not part.empty else set()
        algorithm = name.split('|')[0]
        group_seeds = sorted(c.get('training_seed', 0) for c in candidates if c['algorithm'] == name)
        seed_complete = group_seeds == list(SEEDS) if algorithm in NEURAL else group_seeds == [0]
        record = {'candidate': name, 'algorithm': name.split('|')[0], 'expected_cases': len(expected),
                  'successful_cases': len(observed), 'complete': observed == expected and seed_complete,
                  'catalog_training_seeds': group_seeds, 'required_training_seeds': list(SEEDS) if algorithm in NEURAL else [0]}
        if observed != expected or not seed_complete:
            scores.append(record)
            continue
        part['player_id'] = part.player_id.astype(str)
        values = part.groupby(keys, dropna=False).agg(total_value=('total_value', 'mean'),
                                                     budget_utilization=('budget_utilization', 'mean'))
        record.update(mean_V=float(values.total_value.mean()), mean_budget_utilization=float(values.budget_utilization.mean()),
                      worst_scenario_V=float(values.groupby('scenario').total_value.mean().min()),
                      role_P10_V=float(values.groupby('player_id').total_value.mean().quantile(.1)))
        if ref is not None:
            denominator = ref.reindex(values.index)
            valid = denominator.notna() & (denominator > 0)
            ratios = values.total_value[valid] / denominator[valid]
            record['defined_reference_cases'] = int(valid.sum())
            record['undefined_reference_cases'] = int((~valid).sum())
            record['mean_R'] = float(ratios.mean()) if len(ratios) else None
            record['role_P10_R'] = float(ratios.groupby('player_id').mean().quantile(.1)) if len(ratios) else None
        scores.append(record)
    return scores


def _select(scores, candidates, criterion='mean_V'):
    selected = []
    for algorithm in dict.fromkeys(c['algorithm'].split('|')[0] for c in candidates):
        valid = [r for r in scores if r['algorithm'] == algorithm and r['complete']
                 and r.get(criterion) is not None]
        if not valid:
            raise ValueError(f'No complete validation candidate for {algorithm}')
        choice = sorted(valid, key=lambda r: (-r[criterion], r['candidate']))[0]
        for c in candidates:
            if c['algorithm'] == choice['candidate']:
                selected.append({**deepcopy(c), 'algorithm': algorithm, 'validation_choice': choice['candidate'],
                                 'validation_primary': criterion, 'validation_score': choice[criterion]})
    return selected


def _save_selection(directory, scores, candidates, selected):
    dump(directory / 'candidate_scores.json', scores)
    pd.DataFrame(scores).to_csv(directory / 'candidate_scores.csv', index=False)
    dump(directory / 'candidates.json', candidates)
    dump(directory / 'selected.json', selected)


def calibrate_opponents(workers=4):
    old = read(OLD / 'opponents_frozen.json')
    candidates = [{'algorithm': f'AuctionNetPID|initial_{i}', 'spec': pid_spec(initial_alpha=a)}
                  for i, a in enumerate(alpha_grid())]
    tasks = tasks_for(candidates, old, ['period-7'], ROLES, ['Fixed'])
    directory = OUT / 'opponent_train_calibration'
    frame, failures, skipped = execute(tasks, directory, 'opponent_train_calibration', workers)
    if failures or skipped:
        raise ValueError('Opponent initial-alpha calibration incomplete')
    alphas = {}
    for player in ROLES:
        part = frame[frame.player_id.astype(str) == player].sort_values(['total_value', 'algorithm'], ascending=[False, True])
        chosen = next(c for c in candidates if c['algorithm'] == part.iloc[0].algorithm)
        alphas[player] = chosen['spec']['kwargs']['initial_alpha']
    controls = [{'algorithm': f'AuctionNetPID|control_{i}',
                 'spec': pid_spec(initial_alpha=base_alpha(), player_alphas=alphas, **c)}
                for i, c in enumerate(control_grid())]
    tasks = tasks_for(controls, old, VALID_DAYS, VALID_ROLES, OLD_SCENARIOS)
    directory = OUT / 'opponent_validation'
    frame, _, _ = execute(tasks, directory, 'opponent_validation', workers)
    scores = _candidate_scores(frame, controls, tasks)
    selected = _select(scores, controls)
    _save_selection(directory, scores, controls, selected)
    frozen = {**deepcopy(old), 'auctionnet_pid': selected[0]['spec']}
    if frozen['fixed'] != old['fixed'] or frozen['pid'] != old['pid']:
        raise ValueError('Historical opponent parameters changed')
    immutable(OUT / 'opponents_frozen.json', frozen)
    immutable(OUT / 'opponents_provenance.json', {
        'source_old_sha256': sha256_file(OLD / 'opponents_frozen.json'),
        'fixed_and_localpid_exactly_unchanged': True, 'new_pid_initial_alpha_by_role': alphas,
        'new_pid_control_choice': selected[0], 'training_days': ['period-7'],
        'validation_days': list(VALID_DAYS), 'test_used': False,
        'opponents_sha256': sha256_file(OUT / 'opponents_frozen.json')})
    return frozen


def auctionnet_pid_candidates():
    return [{'algorithm': f'AuctionNetPID|a{i}_c{j}', 'spec': pid_spec(initial_alpha=a, **control)}
            for i, a in enumerate(alpha_grid()) for j, control in enumerate(control_grid())]


def tune_A(workers=4):
    candidates = auctionnet_pid_candidates()
    tasks = tasks_for(candidates, read(OUT / 'opponents_frozen.json'), VALID_DAYS, VALID_ROLES, OLD_SCENARIOS)
    directory = OUT / 'validation_A'
    frame, _, _ = execute(tasks, directory, 'validation_A', workers)
    historical_validation = _frame(OLD / 'neural_validation.csv')
    reference = historical_validation[historical_validation.algorithm == 'LocalPID']
    if len(reference) != 18:
        raise ValueError('Historical validation reference incomplete')
    scores = _candidate_scores(frame, candidates, tasks, reference)
    chosen = _select(scores, candidates, criterion='mean_R')
    _save_selection(directory, scores, candidates, chosen)
    selected = deepcopy(read(OLD / 'selected_frozen.json')['selected']) + chosen
    return freeze('experiment_A', selected)


def freeze(phase, selected):
    if len(selected) != 16 or set(s['algorithm'] for s in selected) != set(ALGORITHMS):
        raise ValueError('Freeze requires all 10 algorithms / 16 policy instances')
    for algorithm in NEURAL:
        if sorted(s['training_seed'] for s in selected if s['algorithm'] == algorithm) != list(SEEDS):
            raise ValueError('Neural training seed coverage missing')
    value = {'phase': phase, 'selected': selected,
             'policy_ids': {f"{s['algorithm']}:{s.get('training_seed',0)}": policy_id(s['spec']) for s in selected},
             'assets': {path: digest for s in selected for path, digest in assets(s['spec']).items()},
             'opponents_sha256': sha256_file(OUT / 'opponents_frozen.json'),
             'protocol_sha256': sha256_file(OUT / 'protocol.json'),
             'execution_code_sha256': runtime_code_hashes(),
             'pipeline_sha256': sha256_file(__file__), 'test_selection_used': False,
             'comparison_days': list(COMPARISON_DAYS), 'independent_days': list(INDEPENDENT_DAYS)}
    immutable(OUT / phase / 'selected_frozen.json', value)
    return value


def verify_freeze(phase):
    value = read(OUT / phase / 'selected_frozen.json')
    if value['execution_code_sha256'] != runtime_code_hashes() or value['pipeline_sha256'] != sha256_file(__file__):
        raise ValueError('Frozen execution/pipeline source changed')
    if value['opponents_sha256'] != sha256_file(OUT / 'opponents_frozen.json') or value['protocol_sha256'] != sha256_file(OUT / 'protocol.json'):
        raise ValueError('Frozen opponents/protocol changed')
    for path, digest in value['assets'].items():
        if sha256_file(path) != digest:
            raise ValueError('Frozen policy asset changed')
    return value


def _historical_rows(selected):
    """Verify reuse identity and preserve source scores without recomputation."""
    from .evaluator import opponent_specs
    _verify_historical()
    old = _frame(OLD / 'episode_results.csv')
    if len(old) != 4320 or not old.completed.all():
        raise ValueError('Historical rows incomplete')
    mapping = {(s['algorithm'], int(s.get('training_seed', 0))): s for s in selected}
    ids = {key: policy_id(s['spec']) for key, s in mapping.items()}
    historical_csv_hash = sha256_file(OLD / 'episode_results.csv')
    opponents = read(OUT / 'opponents_frozen.json')
    rows = []
    for source in old.to_dict('records'):
        row = dict(source)
        old_scenario = row['scenario']
        row['scenario'] = 'LocalPID' if old_scenario == 'PID' else old_scenario
        strategy = mapping[(row['algorithm'], int(row['training_seed']))]
        allocation = opponent_specs(list(ROLES), str(row['player_id']), row['scenario'], opponents,
                                    int(row['seed'])+1000003, row['day_id'])
        if (row['model_fingerprint'] != fingerprint(strategy['spec']) or
                row['assignment_fingerprint'] != fingerprint(allocation) or
                row['dataset_fingerprint'] != cached_day(row['day_id']).fingerprint):
            raise ValueError('Historical reuse identity differs')
        row.update(phase='experiment_A', source_scenario=old_scenario,
                   old_task_index=int(row['task_index']), source_task=f"historical:{int(row['task_index'])}",
                   provenance='historical_exact_policy_reuse', source_result_path=str(OLD / 'episode_results.csv'),
                   policy_id=ids[(row['algorithm'], int(row['training_seed']))], completed_slots=72,
                   legacy_opponent_fingerprint=row['opponent_fingerprint'])
        row['task_key'] = fingerprint({'historical_csv': historical_csv_hash,
                                       'task_index': row['task_index'], 'policy_id': row['policy_id']})
        rows.append(row)
    return pd.DataFrame(rows)


def _historical_slots(frame):
    rows = []
    for row in frame.to_dict('records'):
        path = OLD / 'test_diagnostics' / f"task_{int(row['old_task_index'])}.jsonl"
        feedback = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()
                    if json.loads(line).get('phase') == 'feedback']
        if len(feedback) != 72 or [f['slot'] for f in feedback] != list(range(72)):
            raise ValueError('Historical trace incomplete')
        cum_cost = cum_value = 0.
        for f in feedback:
            cum_cost += f['cost']; cum_value += f['value']
            item = {k: row[k] for k in ('algorithm', 'scenario', 'day_id', 'player_id', 'player_type', 'seed',
                    'training_seed', 'initial_budget', 'task_index', 'source_task', 'provenance', 'policy_id',
                    'model_fingerprint', 'opponent_fingerprint', 'assignment_fingerprint', 'dataset_fingerprint')}
            item.update(phase='experiment_A', time_slot=f['slot'], slot=f['slot'], alpha=f['alpha'],
                        cost=f['cost'], value=f['value'], remaining_budget=f['remaining_budget'],
                        cumulative_cost=cum_cost, cumulative_value=cum_value, wins=np.nan,
                        source_result_path=str(path))
            rows.append(item)
        if abs(cum_value-row['total_value']) > 1e-9 or abs(cum_cost-row['total_cost']) > 2e-6:
            raise ValueError('Historical trace does not match result')
    return pd.DataFrame(rows)


def complete_stage(phase, frame, slots, selected, days, new_count, reused_count):
    directory = OUT / phase
    expected = {(s['algorithm'], int(s.get('training_seed', 0)), d, scene, p, seed)
                for s in selected for d in days for scene in SCENARIOS for p in ROLES for seed in SEEDS}
    keys = ['algorithm', 'training_seed', 'day_id', 'scenario', 'player_id', 'seed']
    frame = frame.copy()
    frame['player_id'] = frame.player_id.astype(str)
    observed = set(frame[keys].itertuples(index=False, name=None))
    valid = observed == expected and len(frame) == len(expected)
    if len(observed) != len(frame) or not observed <= expected:
        raise ValueError('Duplicate or out-of-protocol final result')
    policies = {(s['algorithm'], int(s.get('training_seed', 0))): policy_id(s['spec']) for s in selected}
    for row in frame.itertuples():
        if row.policy_id != policies[(row.algorithm, int(row.training_seed))]:
            raise ValueError('Mixed policy versions in final result')
    _validate_final_identities(frame, slots, selected)
    frame.sort_values(keys).to_csv(directory / 'episode_results.csv', index=False)
    slots.to_csv(directory / 'slots.csv', index=False)
    counts = slots.groupby('task_index').size()
    if len(counts) != len(frame) or not counts.eq(72).all():
        raise ValueError('Final slots/result coverage mismatch')
    status = {'status': 'completed' if valid else 'incomplete', 'valid': valid, 'phase': phase,
              'completed_episodes': len(frame), 'expected_episodes': len(expected),
              'new_executed_episodes': new_count, 'reused_episodes': reused_count,
              'missing_episodes': len(expected-observed),
              'missing_cases': [list(k) for k in sorted(expected-observed)],
              'result_sha256': sha256_file(directory / 'episode_results.csv'),
              'slots_sha256': sha256_file(directory / 'slots.csv')}
    dump(directory / 'episode_results.status.json', status)
    return status


def _validate_final_identities(frame, slots, selected):
    from .evaluator import opponent_specs
    opponents = read(OUT/'opponents_frozen.json')
    specs = {(s['algorithm'], int(s.get('training_seed', 0))): s['spec'] for s in selected}
    policy_ids = {key: policy_id(value) for key, value in specs.items()}
    for row in frame.itertuples():
        key = (row.algorithm, int(row.training_seed))
        allocation = opponent_specs(list(ROLES), str(row.player_id), row.scenario, opponents,
                                    int(row.seed)+1000003, row.day_id)
        if (row.policy_id != policy_ids[key] or row.model_fingerprint != fingerprint(specs[key]) or
                row.assignment_fingerprint != fingerprint(allocation) or
                row.dataset_fingerprint != cached_day(row.day_id).fingerprint or
                row.rules_version != 'harmony-A-B-local-v2' or float(row.budget_scale) != 1. or not row.completed):
            raise ValueError('Final row protocol/strategy identity mismatch')
        if not np.isfinite([row.total_value, row.total_cost, row.wins]).all() or min(row.total_value, row.total_cost, row.wins) < 0:
            raise ValueError('Invalid final row metric')
    grouped = slots.groupby('task_index')
    for row in frame.itertuples():
        if row.task_index not in grouped.groups:
            raise ValueError('Final trace missing')
        part = grouped.get_group(row.task_index).sort_values('time_slot')
        if len(part) != 72 or part.time_slot.tolist() != list(range(72)):
            raise ValueError('Final trace not a full cycle')
        if not np.isfinite(part[['alpha', 'cost', 'value', 'remaining_budget']].to_numpy()).all():
            raise ValueError('Nonfinite final trace')
        if (part[['alpha', 'cost', 'value', 'remaining_budget']] < 0).any().any():
            raise ValueError('Negative final trace')
        for key in ('algorithm', 'scenario', 'day_id', 'player_id', 'seed', 'training_seed', 'policy_id',
                    'model_fingerprint', 'assignment_fingerprint', 'dataset_fingerprint'):
            if not part[key].astype(str).eq(str(getattr(row, key))).all():
                raise ValueError('Final trace identity mismatch: ' + key)
        if abs(part.value.sum()-row.total_value) > 1e-9 or abs(part.cost.sum()-row.total_cost) > 2e-6:
            raise ValueError('Final trace aggregate mismatch')


def _validated_stage(phase):
    frozen = verify_freeze(phase)
    directory = OUT/phase
    status = read(directory/'episode_results.status.json')
    if status.get('status') != 'completed' or status.get('valid') is not True:
        raise ValueError('Cannot reuse incomplete experiment')
    if (sha256_file(directory/'episode_results.csv') != status['result_sha256'] or
            sha256_file(directory/'slots.csv') != status['slots_sha256']):
        raise ValueError('Completed experiment CSV changed')
    frame = _frame(directory/'episode_results.csv')
    slots = _frame(directory/'slots.csv')
    keys = ['algorithm','training_seed','day_id','scenario','player_id','seed']
    days = frozen['independent_days'] if phase == 'independent_test' else frozen['comparison_days']
    expected = {(s['algorithm'], int(s.get('training_seed',0)), day, scene, player, seed)
                for s in frozen['selected'] for day in days for scene in SCENARIOS for player in ROLES for seed in SEEDS}
    observed = set(frame[keys].itertuples(index=False,name=None))
    if (observed != expected or len(frame) != len(expected) or len(frame) != status['expected_episodes']
            or frame.duplicated(keys).any()):
        raise ValueError('Completed experiment coverage differs')
    _validate_final_identities(frame, slots, frozen['selected'])
    return frame, slots, frozen


def test_A(workers=4):
    frozen = verify_freeze('experiment_A')
    selected = frozen['selected']
    opponents = read(OUT / 'opponents_frozen.json')
    old = [s for s in selected if s['algorithm'] != 'AuctionNetPID']
    new = [s for s in selected if s['algorithm'] == 'AuctionNetPID']
    tasks = tasks_for(old, opponents, COMPARISON_DAYS, ROLES, SCENARIOS[3:], SEEDS)
    tasks += tasks_for(new, opponents, COMPARISON_DAYS, ROLES, SCENARIOS, SEEDS)
    directory = OUT / 'experiment_A_new'
    frame, _, _ = execute(tasks, directory, 'experiment_A_new', workers)
    historical = _historical_rows(selected)
    old_slots = _historical_slots(historical)
    new_slots = _frame(directory / 'slots.csv')
    frame['task_index'] += 4320
    frame['phase'] = 'experiment_A'
    new_slots['task_index'] += 4320
    new_slots['phase'] = 'experiment_A'
    return complete_stage('experiment_A', pd.concat([historical, frame], ignore_index=True),
                          pd.concat([old_slots, new_slots], ignore_index=True), selected,
                          COMPARISON_DAYS, len(frame), len(historical))


def fit_new_lp(workers=4):
    candidates = [{'algorithm': f'LP_cost_alpha_{i}', 'spec': spec('FixedAlpha', alpha=base_alpha()*m)}
                  for i, m in enumerate([.5, 1., 2., 4.])]
    tasks = tasks_for(candidates, read(OUT / 'opponents_frozen.json'), ['period-7'], ROLES, SCENARIOS)
    directory = OUT / 'lp_training'
    frame, failures, skipped = execute(tasks, directory, 'lp_training', workers)
    if failures or skipped:
        raise ValueError('LP training response coverage incomplete')
    slots = _frame(directory / 'slots.csv')
    table = OUT / 'online_lp_table.csv'
    if table.exists():
        record = read(OUT / 'online_lp_provenance.json')
        if sha256_file(table) != record['table_sha256'] or sha256_file(directory / 'slots.csv') != record['slots_sha256']:
            raise ValueError('LP table/source changed')
    else:
        fit_lp_table(slots.to_dict('records'), table)
        dump(OUT / 'online_lp_provenance.json', {'fit_days': ['period-7'], 'scenarios': list(SCENARIOS),
             'episodes': len(frame), 'slots': len(slots), 'roles': list(ROLES),
             'alpha_grid': [base_alpha()*m for m in [.5, 1., 2., 4.]], 'test_used': False,
             'method': 'unchanged fit_lp_table: normalized future own costs, pooled equally over all5scenes, monotone alpha frontier',
             'table_sha256': sha256_file(table), 'slots_sha256': sha256_file(directory / 'slots.csv')})
    return str(table.resolve())


def classical_candidates():
    a = base_alpha()
    candidates = [{'algorithm': f'FixedAlpha|{i}', 'spec': spec('FixedAlpha', alpha=a, multiplier=m)}
                  for i, m in enumerate([.5, 1., 1.5, 2., 3., 4., 6.])]
    for i, mult in enumerate([.5, 1., 2., 4.]):
        for j, gains in enumerate([(.25, .025, .025), (.5, .05, .05), (1.5, .1, .15)]):
            candidates.append({'algorithm': f'LocalPID|a{i}_g{j}', 'spec': spec('LocalPID', initial_alpha=a,
                              player_alphas={}, kp=gains[0], ki=gains[1], kd=gains[2], multiplier=mult)})
    for i, mult in enumerate([.25, .5, .75, 1., 1.5, 2.]):
        candidates.append({'algorithm': f'OnlineLP|{i}', 'spec': spec('OnlineLP',
                           table_path=str((OUT / 'online_lp_table.csv').resolve()), multiplier=mult, fallback_alpha=4*a)})
    return candidates + auctionnet_pid_candidates()


def tune_B_classical(workers=4):
    candidates = classical_candidates()
    tasks = tasks_for(candidates, read(OUT / 'opponents_frozen.json'), VALID_DAYS, VALID_ROLES, SCENARIOS)
    directory = OUT / 'validation_B_classical'
    frame, _, _ = execute(tasks, directory, 'validation_B_classical', workers)
    first = _candidate_scores(frame, candidates, tasks)
    selected = _select(first, candidates)
    ref_name = next(s['validation_choice'] for s in selected if s['algorithm'] == 'LocalPID')
    scores = _candidate_scores(frame, candidates, tasks, frame[frame.algorithm == ref_name])
    _save_selection(directory, scores, candidates, selected)
    return selected


def tune_B_neural(workers=4):
    records = read(OUT / 'models' / 'candidates.json')
    candidates = [{'algorithm': f"{r['algorithm']}|{r['config_index']}|{r['step']}",
                   'training_seed': int(r['seed']),
                   'spec': {'kind': 'custom', 'class_path': 'harmony_baselines.neural:NeuralAgent',
                            'kwargs': {'checkpoint': r['checkpoint']}}} for r in records]
    tasks = tasks_for(candidates, read(OUT / 'opponents_frozen.json'), VALID_DAYS, VALID_ROLES, SCENARIOS)
    directory = OUT / 'validation_B_neural'
    frame, _, _ = execute(tasks, directory, 'validation_B_neural', workers)
    classic = read(OUT / 'validation_B_classical' / 'selected.json')
    ref_name = next(s['validation_choice'] for s in classic if s['algorithm'] == 'LocalPID')
    reference = _frame(OUT / 'validation_B_classical' / 'episode_results.csv')
    scores = _candidate_scores(frame, candidates, tasks, reference[reference.algorithm == ref_name])
    selected = _select(scores, candidates)
    _save_selection(directory, scores, candidates, selected)
    return freeze('experiment_B', classic + selected)


def test_B(workers=4, independent=False):
    frozen = verify_freeze('experiment_B')
    selected = frozen['selected']
    opponents = read(OUT / 'opponents_frozen.json')
    phase = 'independent_test' if independent else 'experiment_B'
    days = INDEPENDENT_DAYS if independent else COMPARISON_DAYS
    reused = pd.DataFrame()
    reused_slots = pd.DataFrame()
    to_run = selected
    if independent:
        # First access to held-out pValues occurs after strict frozen validation.
        for day in days:
            build_cache(day, cache_root=OUT / 'cache')
        immutable(OUT / phase / 'selected_frozen.json', frozen)
    else:
        a, aslots, a_frozen = _validated_stage('experiment_A')
        old_ids = a_frozen['policy_ids']
        identical = {f"{s['algorithm']}:{s.get('training_seed',0)}" for s in selected
                     if old_ids.get(f"{s['algorithm']}:{s.get('training_seed',0)}") == policy_id(s['spec'])}
        to_run = [s for s in selected if f"{s['algorithm']}:{s.get('training_seed',0)}" not in identical]
        mask = [f'{r.algorithm}:{int(r.training_seed)}' in identical for r in a.itertuples()]
        reused = a.loc[mask].copy()
        reused_slots = aslots[aslots.task_index.isin(reused.task_index)].copy()
        # Preserve source task/path, and relabel only the new experiment lineage.
        reused['provenance'] = 'experiment_A_exact_policy_reuse'
        reused_slots['provenance'] = 'experiment_A_exact_policy_reuse'
    tasks = tasks_for(to_run, opponents, days, ROLES, SCENARIOS, SEEDS)
    if not tasks:
        reused['phase'] = phase
        reused_slots['phase'] = phase
        return complete_stage(phase, reused, reused_slots, selected, days, 0, len(reused))
    directory = OUT / (phase + '_new')
    frame, _, _ = execute(tasks, directory, phase + '_new', workers)
    slots = _frame(directory / 'slots.csv')
    # Assign unique final IDs without changing the source_task references.
    if len(reused):
        mapping = {int(old): new for new, old in enumerate(reused.task_index)}
        reused['task_index'] = reused.task_index.map(mapping)
        reused_slots['task_index'] = reused_slots.task_index.map(mapping)
    frame['task_index'] += len(reused)
    slots['task_index'] += len(reused)
    merged = pd.concat([reused, frame], ignore_index=True) if len(reused) else frame
    merged_slots = pd.concat([reused_slots, slots], ignore_index=True) if len(reused) else slots
    merged['phase'] = phase
    merged_slots['phase'] = phase
    return complete_stage(phase, merged, merged_slots, selected, days, len(frame), len(reused))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'opponents', 'tune-A', 'test-A', 'trajectories', 'lp',
                                        'tune-B-classical', 'train', 'tune-B-neural', 'test-B', 'independent', 'all'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.stage != 'prepare' and args.stage != 'all':
        _require_protocol()
    from . import data_training
    actions = [('prepare', prepare), ('opponents', lambda: calibrate_opponents(args.workers)),
               ('tune-A', lambda: tune_A(args.workers)), ('test-A', lambda: test_A(args.workers)),
               ('trajectories', lambda: data_training.generate_training(read(OUT/'opponents_frozen.json'), args.workers)),
               ('lp', lambda: fit_new_lp(args.workers)), ('tune-B-classical', lambda: tune_B_classical(args.workers)),
               ('train', data_training.train_models), ('tune-B-neural', lambda: tune_B_neural(args.workers)),
               ('test-B', lambda: test_B(args.workers)), ('independent', lambda: test_B(args.workers, independent=True))]
    for name, action in actions:
        if args.stage not in ('all', name):
            continue
        started = time.perf_counter()
        print(json.dumps({'pipeline_stage': name, 'status': 'starting'}, ensure_ascii=False), flush=True)
        action()
        print(json.dumps({'pipeline_stage': name, 'status': 'finished', 'seconds': time.perf_counter()-started}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
