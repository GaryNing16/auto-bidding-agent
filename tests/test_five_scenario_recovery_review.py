"""Independent recovery merge/evidence regressions, all synthetic and isolated."""
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from harmony_baselines.data import sha256_file
from harmony_baselines.five_scenarios import recovery as r
from test_five_scenario_recovery import sandbox, write, plan


def hashes(directory):
    return {str(p.relative_to(directory)): sha256_file(p) for p in directory.rglob('*') if p.is_file()}


@pytest.fixture
def finished(sandbox, monkeypatch):
    root, protocol, _, _ = sandbox
    sealed = plan(sandbox)
    directory = root / r.RECOVERY
    def payload(old, index, phase):
        task, identity = protocol['tasks'][old], protocol['identities'][old]
        row = {key: task[key] for key in ('algorithm', 'training_seed', 'day_id', 'scenario', 'seed')}
        row.update(player_id=task['target'], task_index=index, phase=phase, completed=True,
                   total_value=float(old + 1), total_cost=2., wins=7,
                   elapsed_seconds=1975.5 + old, bidding_p99_ms=1234. + old,
                   source_task=f'{phase}:{index}', provenance='new_execution',
                   source_result_path=str(root / phase / 'results' / f'task_{index:05d}.json'),
                   task_key=identity['task_key'], policy_id=identity['policy_id'])
        return {'status': 'completed', 'identity': identity, 'row': row,
                'slots': [dict(row, time_slot=i, slot=i, cost=2. / 72,
                               value=(old + 1.) / 72, alpha=1., wins=int(i == 0)) for i in range(72)]}
    original = [payload(i, i, r.ORIGINAL) for i in (0, 3)]
    recovered = [payload(old, i, r.RECOVERY) for i, old in enumerate((1, 2))]
    for phase, data in ((r.ORIGINAL, original), (r.RECOVERY, recovered)):
        folder = root / phase
        for item in data:
            write(folder / 'results' / f'task_{item["row"]["task_index"]:05d}.json', item)
        pd.DataFrame([x['row'] for x in data]).to_csv(folder / 'episode_results.csv', index=False)
        pd.DataFrame([s for x in data for s in x['slots']]).to_csv(folder / 'slots.csv', index=False)
    write(root / r.ORIGINAL / 'episode_results.status.json', {'status': 'completed_with_failures',
        'completed_episodes': 2, 'expected_episodes': 4, 'failed_episodes': 1, 'skipped_episodes': 1})
    write(directory / 'episode_results.status.json', {'status': 'completed', 'valid': True,
        'completed_episodes': 2, 'expected_episodes': 2, 'failed_episodes': 0, 'skipped_episodes': 0})
    write(directory / 'task_protocol.json', {'tasks': sealed['tasks'], 'identities': sealed['identities'],
                                           'code_sha256': sealed['runtime_code_sha256']})
    write(directory / 'failures.json', [])
    write(directory / 'skipped.json', [])
    write(directory / 'recovery_receipt.json', {'status': 'completed',
        'plan_sha256': sha256_file(directory / 'recovery_plan.json'),
        'result_sha256': sha256_file(directory / 'episode_results.csv'),
        'slots_sha256': sha256_file(directory / 'slots.csv')})
    final = root / r.FINAL
    for name in ('episode_results.csv', 'slots.csv'):
        (final / name).write_bytes((root / r.ORIGINAL / name).read_bytes())
    write(final / 'episode_results.status.json', {'status': 'incomplete', 'valid': False,
                                                'completed_episodes': 2, 'expected_episodes': 4})
    def validate(item, task, identity, index):
        if (item['identity'] != identity or item['row']['task_index'] != index
                or item['row']['seed'] != task['seed'] or len(item['slots']) != 72):
            raise ValueError('Synthetic durable identity failure')
    monkeypatch.setattr(r, '_validate', validate)
    monkeypatch.setattr(r, '_pipeline_processes', lambda: [])
    monkeypatch.setattr(r, '_validate_final_tables', lambda root, frame, slots, frozen: None)
    return root, protocol, sealed


def test_failed_diagnostic_is_required_before_planning(sandbox):
    root, _, failure, _ = sandbox
    Path(failure['diagnostic_path']).unlink()
    with pytest.raises(ValueError, match='diagnostic'):
        plan(sandbox)
    assert not (root / r.RECOVERY / 'recovery_plan.json').exists()


@pytest.mark.parametrize('key', ['tasks', 'identities', 'mapping'])
def test_each_single_length_mismatch_rejected_as_value_error(sandbox, key):
    root, *_ = sandbox
    sealed = plan(sandbox)
    sealed[key] = sealed[key][:-1]
    with pytest.raises(ValueError, match='length'):
        r._verify_plan(root, sealed)


def test_complete_union_preserves_original_business_timing_and_is_idempotent(finished):
    root, _, _ = finished
    before = hashes(root / r.ORIGINAL)
    prior_final = {name: (root / r.FINAL / name).read_bytes()
                   for name in ('episode_results.csv', 'slots.csv', 'episode_results.status.json')}
    status = r.finalize(root)
    assert status['completed_episodes'] == 4 and status['recovery_completed_episodes'] == 2
    assert status['historical_failed_episodes'] == status['historical_skipped_episodes'] == 1
    assert status['failed_episodes'] == status['skipped_episodes'] == 0
    assert hashes(root / r.ORIGINAL) == before
    final = pd.read_csv(root / r.FINAL / 'episode_results.csv').set_index('task_index')
    assert set(final.index) == {0, 1, 2, 3}
    for old in (0, 3):
        assert final.loc[old, 'elapsed_seconds'] == 1975.5 + old
        assert final.loc[old, 'bidding_p99_ms'] == 1234. + old
        assert final.loc[old, 'total_value'] == old + 1.
        assert final.loc[old, 'provenance'] == 'new_execution'
    for local, old in enumerate((1, 2)):
        assert final.loc[old, 'source_task'] == f'{r.RECOVERY}:{local}'
        assert final.loc[old, 'original_task_index'] == old
        assert final.loc[old, 'recovery_task_index'] == local
        assert final.loc[old, 'provenance'] == 'resource_recovery'
    snapshot = root / r.RECOVERY / 'evidence/original_final_snapshot'
    assert (snapshot / 'episode_results.sealed.csv').read_bytes() == prior_final['episode_results.csv']
    assert (snapshot / 'slots.sealed.csv').read_bytes() == prior_final['slots.csv']
    assert (snapshot / 'episode_status.sealed.json').read_bytes() == prior_final['episode_results.status.json']
    all_after = hashes(root)
    assert r.finalize(root) == status
    assert hashes(root) == all_after


def test_additional_missing_source_case_rejects_final_merge(finished):
    root, protocol, _ = finished
    before = hashes(root / r.FINAL)
    (root / r.ORIGINAL / 'results/task_00000.json').unlink()
    skipped = r.c.read(root / r.ORIGINAL / 'skipped.json')
    skipped.append({'task_index': 0, 'algorithm': 'TD3BC', 'status': 'not_run',
                    'task_key': protocol['identities'][0]['task_key'], 'phase': r.ORIGINAL})
    write(root / r.ORIGINAL / 'skipped.json', skipped)
    with pytest.raises(ValueError, match='missing set'):
        r.finalize(root)
    assert hashes(root / r.FINAL) == before


def test_missing_recovered_result_rejects_without_publishing(finished):
    root, _, _ = finished
    before = hashes(root / r.FINAL)
    (root / r.RECOVERY / 'results/task_00001.json').unlink()
    with pytest.raises(ValueError, match='durable result coverage'):
        r.finalize(root)
    assert hashes(root / r.FINAL) == before


def test_wrong_recovered_seed_rejects_without_publishing(finished):
    root, _, _ = finished
    before = hashes(root / r.FINAL)
    path = root / r.RECOVERY / 'results/task_00000.json'
    item = r.c.read(path)
    item['row']['seed'] += 1
    write(path, item)
    with pytest.raises(ValueError, match='durable identity'):
        r.finalize(root)
    assert hashes(root / r.FINAL) == before


@pytest.mark.parametrize('artifact', ['snapshot', 'manifest', 'receipt'])
def test_published_recovery_rechecks_all_sealed_evidence(finished, artifact):
    root, _, _ = finished
    r.finalize(root)
    if artifact == 'snapshot':
        path = root / r.RECOVERY / 'evidence/original_final_snapshot/episode_results.sealed.csv'
        path.write_text('corrupted evidence', encoding='utf8')
    else:
        path = root / r.FINAL / 'recovery_manifest.json' if artifact == 'manifest' else root / r.RECOVERY / 'recovery_receipt.json'
        value = r.c.read(path)
        value['changed_unexpectedly'] = True
        write(path, value)
    with pytest.raises(ValueError, match='Published recovery result changed'):
        r.finalize(root)


def test_pipeline_starting_during_merge_prevents_publication(finished, monkeypatch):
    root, _, _ = finished
    before = hashes(root / r.FINAL)
    calls = iter([[], [{'pid': 999}]])
    monkeypatch.setattr(r, '_pipeline_processes', lambda: next(calls))
    with pytest.raises(ValueError, match='started before publication'):
        r.finalize(root)
    assert hashes(root / r.FINAL) == before
