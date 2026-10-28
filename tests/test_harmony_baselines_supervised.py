"""Resume identity, paired-seed coverage and failure classification regressions."""
from pathlib import Path
import pandas as pd
import pytest
from harmony_baselines import supervised as s


def test_partial_role_coverage_is_not_a_complete_candidate(monkeypatch):
    monkeypatch.setattr(s.exp,'TRAIN_SEEDS',[1,2])
    monkeypatch.setattr(s.exp,'VALID_DAYS',['v'])
    monkeypatch.setattr(s.exp,'VALID_ROLES',['0','1'])
    rows=[]
    for seed in [1,2]:
        for scene in s.exp.SCENARIOS:
            for player in ['0','1']:
                rows.append({'algorithm':'CQL|0|200','training_seed':seed,'scenario':scene,'player_id':player})
    candidates=[{'algorithm':'CQL|0|200'}]
    valid,excluded=s._valid_groups(pd.DataFrame(rows[:-1]),candidates,[])
    assert valid==[] and excluded[0]['complete_cases']==11
    assert excluded[0]['expected_cases']==12
    valid,_=s._valid_groups(pd.DataFrame(rows),candidates,[])
    assert valid==['CQL|0|200']
    valid,excluded=s._valid_groups(pd.DataFrame(rows),candidates,[{'candidate':'CQL|0|200','status':'resource_incomplete'}])
    assert valid==[] and excluded[0]['resource_incomplete_count']==1


def test_failure_cases_do_not_create_scores_and_keep_test_identity():
    task={'algorithm':'CQL','training_seed':2,'day_id':'test-period','scenario':'Mixed','target':'7','seed':11}
    case=s._case(task,50,'test')
    assert case['task_index']==50 and case['player_id']=='7'
    assert case['phase']=='test' and case['test_used'] is True
    assert case['evaluation_seed']==11
    assert not {'total_value','wins','budget_utilization'} & set(case)


def test_diagnostic_completion_counts_feedback_not_an_unsettled_action(tmp_path):
    path=tmp_path/'trace.jsonl'
    path.write_text('{"phase":"action","slot":0}\n{"phase":"feedback","slot":0}\n'
                    '{"phase":"before_action","slot":1}\n{"phase":"action","slot":1}\n',encoding='utf-8')
    assert s._completed_slots(path)==1


def test_resume_rejects_changed_target_before_loading_market(monkeypatch):
    tasks=[{'algorithm':'BC|0|200','day_id':'v','scenario':'Fixed','target':'0','seed':1,'training_seed':2}]
    frame=pd.DataFrame([{'task_index':0,'algorithm':'BC|0|200','day_id':'v','scenario':'Fixed','player_id':'3'}])
    monkeypatch.setattr(s.exp,'cached_day',lambda _:pytest.fail('Must reject altered identity before market load'))
    with pytest.raises(ValueError,match='resume task mismatch'):
        s._validate_existing(frame,tasks)


def test_policy_timing_excludes_diagnostic_io_and_identical_opponent(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from harmony_baselines import evaluator
    now=[0.]
    class Agent:
        def bidding(self, observation):
            now[0]+=.002 if self.own_config.advertiser_id=='0' else .1
            return 1.
        def update(self, feedback):
            now[0]+=.003 if self.own_config.advertiser_id=='0' else .2
    monkeypatch.setattr(evaluator,'make_agent',lambda _:Agent())
    monkeypatch.setattr(s.time,'perf_counter',lambda:now[0])
    original_dumps=s.json.dumps
    def slow_diagnostic(value,**kwargs):
        now[0]+=10.
        return original_dumps(value,**kwargs)
    monkeypatch.setattr(s.json,'dumps',slow_diagnostic)
    spec={'kind':'fixed'}
    def worker(task):
        agents=[]
        for player in ['0','1']:
            agent=evaluator.make_agent(spec)
            agent.own_config=SimpleNamespace(advertiser_id=player)
            agents.append(agent)
        for slot in range(72):
            for agent in agents:
                agent.bidding(SimpleNamespace(own_config=agent.own_config,time_slot=slot,remaining_budget=1.))
                agent.update(SimpleNamespace(time_slot=slot,alpha=1.,value=0.,cost=0.,remaining_budget=1.))
        return {'bidding_p99_ms':999.,'update_p99_ms':999.,'paired_p99_ms':999.},[],[]
    monkeypatch.setattr(s.exp,'worker',worker)
    trace=tmp_path/'trace.jsonl'
    result=s._episode_payload({'target':'0','target_spec':spec},trace)
    assert result['status']=='completed'
    for field, expected in [('bidding_p99_ms',2.),('update_p99_ms',3.),('paired_p99_ms',5.)]:
        assert result['row'][field]==pytest.approx(expected)
    assert s._completed_slots(trace)==72
