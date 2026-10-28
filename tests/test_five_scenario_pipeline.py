"""Selection intent and coverage calculations, independent of real test scores."""
import pandas as pd
import pytest

from harmony_baselines.five_scenarios import pipeline as p
from harmony_baselines.five_scenarios.common import SCENARIOS, OLD_SCENARIOS, SEEDS, ROLES
from harmony_baselines.five_scenarios.execution import tasks_for


def candidate(label, seed=0):
    return {'algorithm': label, 'training_seed': seed, 'spec': {'kind': 'constant', 'alpha': 1.}}


def synthetic_scores():
    candidates = [candidate('BC|raw_value', seed) for seed in SEEDS]
    candidates += [candidate('BC|relative', seed) for seed in SEEDS]
    tasks = tasks_for(candidates, {}, ['period-21'], ['0', '8'], ['Fixed'])
    rows = []
    for index, t in enumerate(tasks):
        # raw_value wins average V (5.5 vs 5), but relative wins mean R.
        value = {'BC|raw_value': {'0': 1., '8': 10.}, 'BC|relative': {'0': 5., '8': 5.}}[t['algorithm']][t['target']]
        rows.append({'task_index': index, 'algorithm': t['algorithm'], 'training_seed': t['training_seed'],
                     'day_id': t['day_id'], 'scenario': t['scenario'], 'player_id': t['target'],
                     'seed': t['seed'], 'total_value': value, 'budget_utilization': .8})
    reference = pd.DataFrame([{'day_id': 'period-21', 'scenario': 'Fixed', 'player_id': player,
                              'seed': SEEDS[0], 'total_value': value} for player, value in [('0', 1.), ('8', 10.)]])
    return candidates, tasks, pd.DataFrame(rows), reference


def test_B_prefers_V_and_keeps_both_training_seeds():
    candidates, tasks, frame, reference = synthetic_scores()
    scores = p._candidate_scores(frame, candidates, tasks, reference)
    selected = p._select(scores, candidates)
    assert {s['validation_choice'] for s in selected} == {'BC|raw_value'}
    assert sorted(s['training_seed'] for s in selected) == list(SEEDS)
    old_convention = p._select(scores, candidates, 'mean_R')
    assert {s['validation_choice'] for s in old_convention} == {'BC|relative'}


def test_missing_one_seed_role_excludes_entire_candidate_without_zero():
    candidates, tasks, frame, reference = synthetic_scores()
    scores = p._candidate_scores(frame.iloc[1:], candidates, tasks, reference)
    first = next(s for s in scores if s['candidate'] == 'BC|raw_value')
    assert not first['complete'] and 'mean_V' not in first
    selected = p._select(scores, candidates)
    assert {s['validation_choice'] for s in selected} == {'BC|relative'}


def test_zero_reference_is_undefined_without_epsilon():
    candidates, tasks, frame, reference = synthetic_scores()
    reference['total_value'] = 0.
    scores = p._candidate_scores(frame, candidates, tasks, reference)
    assert all(s['mean_R'] is None and s['undefined_reference_cases'] == 2 for s in scores)
    assert p._select(scores, candidates)[0]['validation_choice'] == 'BC|raw_value'
    with pytest.raises(ValueError, match='No complete'):
        p._select(scores, candidates, 'mean_R')


def test_protocol_counts_include_both_kinds_of_seed():
    old = [candidate(a) for a in ['FixedAlpha', 'LocalPID', 'OnlineLP']]
    old += [candidate(a, seed) for a in p.NEURAL for seed in SEEDS]
    added = [candidate('AuctionNetPID')]
    historical = tasks_for(old, {}, ['period-24'], ROLES, OLD_SCENARIOS, SEEDS)
    extension = tasks_for(old, {}, ['period-24'], ROLES, SCENARIOS[3:], SEEDS)
    added_full = tasks_for(added, {}, ['period-24'], ROLES, SCENARIOS, SEEDS)
    full = tasks_for(old + added, {}, ['period-24'], ROLES, SCENARIOS, SEEDS)
    assert (len(historical), len(extension), len(added_full), len(full)) == (4320, 2880, 480, 7680)
    assert len(extension) + len(added_full) == 3360


def test_search_grid_preserves_upstream_default_control():
    variants = p.auctionnet_pid_candidates()
    assert len(variants) == 25
    default = variants[0]['spec']['kwargs']
    assert default == {'initial_alpha': 15., 'lower_threshold': .7, 'upper_threshold': 1.1,
                       'increase_factor': 1.2, 'decrease_factor': .7}
    assert all('player_alphas' not in c['spec']['kwargs'] for c in variants)


def test_equal_scores_have_declared_lexicographic_tiebreak():
    candidates = [candidate('BC|z', s) for s in SEEDS] + [candidate('BC|a', s) for s in SEEDS]
    scores = [{'algorithm': 'BC', 'candidate': label, 'complete': True, 'mean_V': 10.}
              for label in ['BC|z', 'BC|a']]
    assert {s['validation_choice'] for s in p._select(scores, candidates)} == {'BC|a'}
