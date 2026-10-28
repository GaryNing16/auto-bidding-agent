"""Persistent spawn workers, immutable task keys, per-episode durable commits.

The worker only invokes the five-scenario adapter to the unique AuctionEngine.
Completed atomic per-task artifacts are the source of truth; CSVs are derived.
"""
from __future__ import annotations
import csv
from dataclasses import replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import math

import numpy as np
import pandas as pd
from harmony_bidding.schema import fingerprint
from harmony_baselines.data import sha256_file
from .common import OUT, assets, cached_day, dump, immutable, policy_id, read, runtime_code_hashes


def tasks_for(candidates, opponents, days, roles, scenarios, seeds=(20261028,), **extra):
    return [dict(day_id=d, target=str(p), target_spec=c['spec'], scenario=s,
                 opponents=opponents, seed=int(seed), algorithm=c['algorithm'],
                 training_seed=int(c.get('training_seed', 0)), **extra)
            for c in candidates for d in days for s in scenarios for p in roles for seed in seeds]


def _worker(connection):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    from .evaluator import run_episode
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            task, identity, result_path, diagnostic_path, phase, index = request
            started = time.perf_counter()
            try:
                arguments = dict(task)
                day = cached_day(arguments.pop('day_id'))
                if arguments.pop('smoke', False):
                    day = replace(day, slots=tuple(replace(s, values=s.values[:4],
                        participation_mask=s.participation_mask[:4], request_ids=s.request_ids[:4],
                        original_order=s.original_order[:4]) for s in day.slots))
                row, slots, transitions = run_episode(day, diagnostic_path=diagnostic_path, **arguments)
                common = {'phase': phase, 'task_index': index, 'source_task': f'{phase}:{index}',
                          'provenance': 'new_execution', 'source_scenario': task['scenario'],
                          'source_result_path': str(Path(result_path).resolve()),
                          'policy_id': identity['policy_id'], 'task_key': identity['task_key']}
                row.update(common)
                for slot in slots:
                    slot.update(common)
                    for key in ('algorithm', 'seed', 'training_seed', 'model_fingerprint',
                                'opponent_fingerprint', 'assignment_fingerprint', 'dataset_fingerprint'):
                        slot[key] = row[key]
                    slot['slot'] = slot['time_slot']
                    if 'alpha' in task['target_spec'].get('kwargs', {}):
                        slot['behavior_alpha'] = task['target_spec']['kwargs']['alpha']
                transition_path = None
                if transitions:
                    transition_path = str(Path(result_path).with_suffix('.npz'))
                    arrays = {
                        'states': np.stack([t['state'] for t in transitions]),
                        'actions': np.array([t['action'] for t in transitions]),
                        'rewards': np.array([t['reward'] for t in transitions]),
                        'next_states': np.stack([t['next_state'] for t in transitions]),
                        'dones': np.array([t['done'] for t in transitions]),
                        'episode_id': np.array([f'new:{index}:{t["player_id"]}' for t in transitions]),
                    }
                    with Path(transition_path + '.pending').open('wb') as handle:
                        np.savez_compressed(handle, **arrays)
                    Path(transition_path + '.pending').replace(transition_path)
                payload = {'status': 'completed', 'identity': identity, 'row': row, 'slots': slots,
                           'transition_path': transition_path,
                           'transition_sha256': sha256_file(transition_path) if transition_path else None,
                           'seconds': time.perf_counter()-started}
                dump(result_path, payload)
                connection.send({'status': 'completed', 'path': result_path})
            except BaseException as error:
                connection.send({'status': 'failed', 'exception_type': type(error).__name__,
                                 'reason': str(error), 'traceback': traceback.format_exc(),
                                 'seconds': time.perf_counter()-started})
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


def _validate(payload, task, identity, index):
    if payload['status'] != 'completed' or payload['identity'] != identity:
        raise ValueError(f'Completed identity mismatch: {index}')
    row, slots = payload['row'], payload['slots']
    for key, expected in [('algorithm', task['algorithm']), ('scenario', task['scenario']),
                          ('day_id', task['day_id']), ('player_id', task['target']),
                          ('seed', task['seed']), ('training_seed', task.get('training_seed', 0))]:
        if str(row[key]) != str(expected):
            raise ValueError(f'Completed row mismatch: {index}/{key}')
    if row['dataset_fingerprint'] != identity['dataset_fingerprint']:
        raise ValueError('Dataset differs on resume')
    if row['policy_id'] != identity['policy_id'] or row['task_key'] != identity['task_key'] or row['task_index'] != index:
        raise ValueError('Row policy/task identity differs')
    if row['model_fingerprint'] != fingerprint(task['target_spec']) or row['opponent_fingerprint'] != fingerprint(task['opponents']):
        raise ValueError('Row strategy fingerprints differ')
    from .evaluator import opponent_specs
    roster = [a.advertiser_id for a in cached_day(task['day_id']).advertisers]
    allocation = opponent_specs(roster, task['target'], task['scenario'], task['opponents'],
                                task['seed'] + 1000003, task['day_id'])
    if row['assignment_fingerprint'] != fingerprint(allocation):
        raise ValueError('Row opponent allocation differs')
    if len(slots) != 72 or [s['time_slot'] for s in slots] != list(range(72)):
        raise ValueError('Incomplete 72-slot result')
    if not row['completed'] or row['rules_version'] != 'harmony-A-B-local-v2':
        raise ValueError('Invalid completion/rules')
    metrics = [row['total_value'], row['total_cost'], row['wins'], row['initial_budget']]
    if not np.isfinite(metrics).all() or min(metrics) < 0 or row['total_cost'] > row['initial_budget'] + 1e-6:
        raise ValueError('Illegal result metrics')
    running_cost = running_value = 0.
    for slot in slots:
        for key in ('day_id', 'scenario', 'player_id', 'player_type', 'algorithm', 'seed', 'training_seed',
                    'model_fingerprint', 'opponent_fingerprint', 'assignment_fingerprint', 'dataset_fingerprint',
                    'task_index', 'task_key', 'policy_id', 'phase'):
            if str(slot[key]) != str(row[key]):
                raise ValueError('Slot identity mismatch: ' + key)
        values = [slot[k] for k in ('alpha', 'cost', 'value', 'wins', 'remaining_budget',
                                    'initial_budget', 'cumulative_cost', 'cumulative_value')]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('Nonfinite or negative slot metric')
        if isinstance(slot['wins'], bool) or int(slot['wins']) != slot['wins']:
            raise ValueError('Nonintegral slot wins')
        running_cost += slot['cost']
        running_value += slot['value']
        if (abs(running_cost-slot['cumulative_cost']) > 2e-6 or
                abs(running_value-slot['cumulative_value']) > 1e-9 or
                abs(row['initial_budget']-running_cost-slot['remaining_budget']) > 2e-6 or
                slot['initial_budget'] != row['initial_budget']):
            raise ValueError('Slot cumulative ledger mismatch')
    if abs(sum(s['value'] for s in slots)-row['total_value']) > 1e-9:
        raise ValueError('Value/slot audit mismatch')
    if abs(sum(s['cost'] for s in slots)-row['total_cost']) > 2e-6:
        raise ValueError('Cost/slot audit mismatch')
    if sum(s['wins'] for s in slots) != row['wins']:
        raise ValueError('Wins/slot audit mismatch')
    if task.get('collect'):
        path = payload.get('transition_path')
        if not path or not Path(path).is_file() or sha256_file(path) != payload.get('transition_sha256'):
            raise ValueError('Missing/changed transition sidecar')
        with np.load(path, allow_pickle=False) as data:
            if data['states'].shape != (72, 18) or data['next_states'].shape != (72, 18):
                raise ValueError('Invalid transition states')
            for key in ('states', 'next_states', 'actions', 'rewards', 'dones'):
                if len(data[key]) != 72 or not np.isfinite(data[key]).all():
                    raise ValueError('Invalid transition array')
            if (not np.array_equal(data['actions'], [s['alpha'] for s in slots]) or
                    not np.array_equal(data['rewards'], [s['value'] for s in slots]) or
                    not np.array_equal(data['dones'], [False]*71+[True])):
                raise ValueError('Transition/slot mismatch')


def _export(directory, tasks, identities, success):
    rows, slots = [], []
    for index in sorted(success):
        payload = read(directory / 'results' / f'task_{index:05d}.json')
        _validate(payload, tasks[index], identities[index], index)
        rows.append(payload['row'])
        slots.extend(payload['slots'])
    frame = pd.DataFrame(rows)
    for name, data in [('episode_results.csv', frame), ('slots.csv', pd.DataFrame(slots))]:
        pending = directory / (name + '.pending')
        data.to_csv(pending, index=False)
        pending.replace(directory / name)
    return frame


def execute(tasks, directory, phase, workers=4, timeout_seconds=180., fail_group=True):
    """Resume unchanged full episodes; failures are separate and poison a group.

    Candidate algorithm labels include config/step during validation, so a bad
    candidate never excludes another candidate of the same algorithm.
    """
    directory = Path(directory)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError('workers must be a positive integer')
    if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('timeout_seconds must be finite positive')
    (directory / 'results').mkdir(parents=True, exist_ok=True)
    (directory / 'diagnostics').mkdir(exist_ok=True)
    code = runtime_code_hashes()
    models = {}
    opponent_assets = {}
    for t in tasks:
        key = fingerprint(t['target_spec'])
        if key not in models:
            models[key] = policy_id(t['target_spec'])
        for spec in t['opponents'].values():
            opponent_assets.update(assets(spec))
    datasets = {d: cached_day(d).fingerprint for d in sorted({t['day_id'] for t in tasks})}
    identities = []
    for t in tasks:
        identity = {'policy_id': models[fingerprint(t['target_spec'])],
                    'dataset_fingerprint': datasets[t['day_id']], 'execution_code_fingerprint': fingerprint(code),
                    'opponent_assets': opponent_assets, 'rules_version': 'harmony-A-B-local-v2'}
        identity['task_key'] = fingerprint({'task': t, **identity})
        identities.append(identity)
    protocol = {'phase': phase, 'tasks': tasks, 'identities': identities, 'code_sha256': code,
                'timeout_seconds': timeout_seconds, 'fail_group': fail_group}
    immutable(directory / 'task_protocol.json', protocol)
    success = set()
    for index, (task, identity) in enumerate(zip(tasks, identities)):
        path = directory / 'results' / f'task_{index:05d}.json'
        if path.exists():
            _validate(read(path), task, identity, index)
            success.add(index)
    failures = read(directory / 'failures.json') if (directory / 'failures.json').exists() else []
    skipped = read(directory / 'skipped.json') if (directory / 'skipped.json').exists() else []
    failed_ids = {r['task_index'] for r in failures}
    skipped_ids = {r['task_index'] for r in skipped}
    if len(failed_ids) != len(failures) or len(skipped_ids) != len(skipped):
        raise ValueError('Duplicate failed/skipped task indexes')
    for records, allowed in ((failures, {'failed', 'resource_incomplete'}), (skipped, {'not_run'})):
        for record in records:
            index = record['task_index']
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(tasks):
                raise ValueError('Invalid failed/skipped task index')
            if (record.get('task_key') != identities[index]['task_key'] or
                    record.get('phase') != phase or record.get('status') not in allowed):
                raise ValueError('Failed/skipped task identity mismatch')
            if records is failures and record.get('task') != tasks[index]:
                raise ValueError('Failed task specification mismatch')
            if records is skipped and record.get('algorithm') != tasks[index]['algorithm']:
                raise ValueError('Skipped candidate identity mismatch')
    if success & (failed_ids | skipped_ids) or failed_ids & skipped_ids:
        raise ValueError('Overlapping task terminal states')
    expected = set(range(len(tasks)))
    if not (success | failed_ids | skipped_ids) <= expected:
        raise ValueError('Result outside immutable task protocol')
    poisoned = {tasks[i]['algorithm'] for i in failed_ids} if fail_group else set()
    pending = [i for i in range(len(tasks)) if i not in success | failed_ids | skipped_ids]
    ctx = mp.get_context('spawn')
    running, idle = {}, []
    proven = {(tasks[i]['algorithm'], tasks[i].get('training_seed', 0)) for i in success}
    started = time.perf_counter()
    resumed = len(success)
    segment_number = len(list(directory.glob('runtime_segment_*.json')))
    last_print = 0.

    def state(status='running'):
        dump(directory / 'failures.json', failures)
        dump(directory / 'skipped.json', skipped)
        value = {'status': status, 'valid': status == 'completed', 'phase': phase,
                 'completed_episodes': len(success), 'expected_episodes': len(tasks),
                 'failed_episodes': len(failures), 'skipped_episodes': len(skipped),
                 'resumed_completed': resumed, 'workers': workers,
                 'segment_seconds': time.perf_counter()-started,
                 'tasks_sha256': fingerprint(tasks)}
        dump(directory / 'episode_results.status.json', value)
        return value

    def record_skip(i, reason):
        skipped.append({'task_index': i, 'algorithm': tasks[i]['algorithm'], 'reason': reason,
                        'task_key': identities[i]['task_key'], 'phase': phase, 'status': 'not_run'})
        skipped_ids.add(i)

    def close_worker(process, conn):
        if process.is_alive():
            process.terminate()
        process.join(3)
        conn.close()

    state()
    try:
        while pending or running:
            for i in list(pending):
                if tasks[i]['algorithm'] in poisoned:
                    pending.remove(i)
                    record_skip(i, 'same candidate/algorithm has an incomplete required episode')
            for i, (process, conn, begin) in list(running.items()):
                if tasks[i]['algorithm'] in poisoned:
                    close_worker(process, conn)
                    del running[i]
                    committed = directory / 'results' / f'task_{i:05d}.json'
                    if committed.exists():
                        _validate(read(committed), tasks[i], identities[i], i)
                        success.add(i)
                    else:
                        record_skip(i, 'cancelled after same candidate/algorithm failure')
            while pending and len(running) < workers:
                active = {(tasks[i]['algorithm'], tasks[i].get('training_seed', 0)) for i in running}
                options = [i for i in pending if (tasks[i]['algorithm'], tasks[i].get('training_seed', 0)) in proven
                           or (tasks[i]['algorithm'], tasks[i].get('training_seed', 0)) not in active]
                if not options:
                    break
                i = options[0]
                pending.remove(i)
                if idle:
                    process, conn = idle.pop()
                else:
                    conn, child = ctx.Pipe()
                    process = ctx.Process(target=_worker, args=(child,))
                    process.start()
                    child.close()
                conn.send((tasks[i], identities[i], str(directory / 'results' / f'task_{i:05d}.json'),
                           str(directory / 'diagnostics' / f'task_{i:05d}.jsonl'), phase, i))
                running[i] = process, conn, time.perf_counter()
            changed = False
            for i, (process, conn, begin) in list(running.items()):
                message = None
                if conn.poll():
                    try:
                        message = conn.recv()
                    except EOFError:
                        message = {'status': 'failed', 'exception_type': 'WorkerExit', 'reason': 'EOF before completed result'}
                elif time.perf_counter()-begin > timeout_seconds:
                    message = {'status': 'resource_incomplete', 'exception_type': 'Timeout',
                               'reason': f'episode exceeded {timeout_seconds} seconds'}
                elif not process.is_alive():
                    message = {'status': 'failed', 'exception_type': 'WorkerExit', 'reason': str(process.exitcode)}
                if message is None:
                    continue
                changed = True
                committed = directory / 'results' / f'task_{i:05d}.json'
                if message['status'] != 'completed':
                    # Stop first so a commit cannot race the failure decision,
                    # and never reuse a connection carrying a delayed old ACK.
                    close_worker(process, conn)
                    if committed.exists():
                        _validate(read(committed), tasks[i], identities[i], i)
                        message = {'status': 'completed', 'path': str(committed)}
                if message['status'] == 'completed':
                    payload = read(message['path'])
                    _validate(payload, tasks[i], identities[i], i)
                    del running[i]
                    success.add(i)
                    proven.add((tasks[i]['algorithm'], tasks[i].get('training_seed', 0)))
                    if process.is_alive():
                        idle.append((process, conn))
                    else:
                        close_worker(process, conn)
                else:
                    close_worker(process, conn)
                    del running[i]
                    failure = {**message, 'task_index': i, 'task': tasks[i],
                               'algorithm': tasks[i]['algorithm'], 'candidate': tasks[i]['algorithm'],
                               'day_id': tasks[i]['day_id'], 'scenario': tasks[i]['scenario'],
                               'player_id': tasks[i]['target'], 'seed': tasks[i]['seed'],
                               'training_seed': tasks[i].get('training_seed', 0),
                               'phase': phase, 'task_key': identities[i]['task_key'], 'elapsed_seconds': time.perf_counter()-begin,
                               'diagnostic_path': str(directory / 'diagnostics' / f'task_{i:05d}.jsonl')}
                    failures.append(failure)
                    failed_ids.add(i)
                    if fail_group:
                        poisoned.add(tasks[i]['algorithm'])
                    state()
                    print(json.dumps({'stage': phase, 'failure': failure}, ensure_ascii=False), flush=True)
            now = time.perf_counter()
            if changed and (now-last_print > 15 or len(success | failed_ids | skipped_ids) == len(tasks)):
                value = state()
                print(json.dumps(value, ensure_ascii=False), flush=True)
                last_print = now
            if not changed:
                time.sleep(.1)
        if success | failed_ids | skipped_ids != expected:
            raise ValueError('Missing terminal task accounting')
        frame = _export(directory, tasks, identities, success)
        value = state('completed' if not failures and not skipped else 'completed_with_failures')
        return frame, failures, skipped
    finally:
        for process, conn, _ in running.values():
            close_worker(process, conn)
        for process, conn in idle:
            try:
                conn.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            process.join(3)
            if process.is_alive():
                process.terminate()
                process.join(3)
            conn.close()
        dump(directory / f'runtime_segment_{segment_number:03d}.json',
             {'seconds': time.perf_counter()-started, 'resumed': resumed,
              'new_completed': len(success)-resumed, 'failures': len(failures), 'phase': phase})
