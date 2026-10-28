"""Isolated recovery control-flow tests; no model, market or process pool runs."""
import json
from pathlib import Path
import pandas as pd
import pytest
from harmony_bidding.schema import fingerprint
from harmony_baselines.five_scenarios import recovery as r


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    root=tmp_path/'recovery';root.mkdir()
    frozen={'execution_code_sha256':{'mock_runtime.py':'sha'},'selected':[{'algorithm':'TD3BC','training_seed':29,'spec':{'kind':'zero'}}],'independent_days':['period-25']}
    monkeypatch.setattr(r,'_freeze',lambda root:frozen)
    tasks=[{'day_id':'period-25','target':str(i),'target_spec':{'kind':'zero'},'scenario':'Mixed2','opponents':{},'seed':28,'algorithm':'TD3BC','training_seed':29} for i in range(4)]
    identities=[{'task_key':fingerprint(t),'policy_id':'policy','dataset_fingerprint':'data'} for t in tasks]
    protocol={'tasks':tasks,'identities':identities,'code_sha256':frozen['execution_code_sha256'],'timeout_seconds':180.,'fail_group':True}
    write(root/'experiment_B/selected_frozen.json',frozen)
    write(root/r.FINAL/'selected_frozen.json',frozen)
    write(root/r.ORIGINAL/'task_protocol.json',protocol)
    diag=root/r.ORIGINAL/'diagnostics/task_00001.jsonl';diag.parent.mkdir(parents=True);diag.write_text('{"phase":"action","slot":4,"alpha":1}\n',encoding='utf8')
    failure={'status':'resource_incomplete','exception_type':'Timeout','task_index':1,'task':tasks[1],'task_key':identities[1]['task_key'],'phase':r.ORIGINAL,'algorithm':'TD3BC','elapsed_seconds':1966.,'diagnostic_path':str(diag)}
    skip={'status':'not_run','task_index':2,'task_key':identities[2]['task_key'],'phase':r.ORIGINAL,'algorithm':'TD3BC','reason':'same candidate failure'}
    write(root/r.ORIGINAL/'failures.json',[failure]);write(root/r.ORIGINAL/'skipped.json',[skip])
    write(root/r.ORIGINAL/'episode_results.status.json',{'status':'running','completed_episodes':2})
    write(root/'qa/independent_interruption_20261028/incident_diagnosis.json',{'user_report':'network','standby_seconds':1971.219687})
    write(root/'qa/independent_interruption_20261028/windows_power_events.json',[{'id':506},{'id':507}])
    write(root/'protocol.json',{'scenarios':['Mixed2'],'roles':['0','1','2','3'],'evaluation_seeds':[28]})
    (root/r.ORIGINAL/'results').mkdir()
    return root,protocol,failure,skip


def plan(sandbox):return r.create_plan(sandbox[0],expected_missing=2)


def test_plan_while_original_active_does_not_write_original(sandbox):
    root,protocol,_,_=sandbox
    before={str(p):p.read_bytes() for p in (root/r.ORIGINAL).rglob('*') if p.is_file()}
    value=plan(sandbox)
    assert value['tasks']==[protocol['tasks'][1],protocol['tasks'][2]]
    assert [(m['original_task_index'],m['recovery_task_index']) for m in value['mapping']]==[(1,0),(2,1)]
    assert value['identities']==[protocol['identities'][1],protocol['identities'][2]]
    assert all(p.read_bytes()==before[str(p)] for p in (root/r.ORIGINAL).rglob('*') if p.is_file())
    assert r.create_plan(root,expected_missing=2)==value


def test_original_mutable_status_is_not_plan_anchor(sandbox):
    root,*_=sandbox;value=plan(sandbox)
    write(root/r.ORIGINAL/'episode_results.status.json',{'status':'running','completed_episodes':3})
    (root/r.ORIGINAL/'episode_results.csv').write_text('changed derived export\n',encoding='utf8')
    r._verify_plan(root,value)


@pytest.mark.parametrize('field',['task_protocol','freeze','skip','failure','diagnostic','mapping','task_length'])
def test_plan_rejects_identity_or_evidence_changes(sandbox,field):
    root,protocol,failure,skip=sandbox;value=plan(sandbox)
    if field=='task_protocol':
        protocol['tasks'][1]['seed']=99;write(root/r.ORIGINAL/'task_protocol.json',protocol)
    elif field=='freeze':write(root/'experiment_B/selected_frozen.json',{'changed':True})
    elif field=='skip':skip['reason']='changed';write(root/r.ORIGINAL/'skipped.json',[skip])
    elif field=='failure':failure['elapsed_seconds']=1.;write(root/r.ORIGINAL/'failures.json',[failure])
    elif field=='diagnostic':Path(failure['diagnostic_path']).write_text('changed',encoding='utf8')
    elif field=='mapping':value['mapping'][1]['original_task_index']=1
    else:value['tasks']=value['tasks'][:1]
    with pytest.raises((ValueError,IndexError)):
        r._verify_plan(root,value)


def test_plan_rejects_nonresource_failure(sandbox):
    root,_,failure,_=sandbox;failure['exception_type']='OverflowError';failure['status']='failed';write(root/r.ORIGINAL/'failures.json',[failure])
    with pytest.raises(ValueError,match='resource timeouts'):plan(sandbox)


def test_run_only_sealed_tasks_two_workers_original_timeout(sandbox,monkeypatch):
    root,protocol,_,_=sandbox;value=plan(sandbox);captured={}
    def fake_execute(tasks,directory,phase,**kwargs):
        captured.update(tasks=tasks,directory=directory,phase=phase,**kwargs)
        pd.DataFrame({'task_index':[0,1]}).to_csv(directory/'episode_results.csv',index=False)
        pd.DataFrame({'slot':[0,1]}).to_csv(directory/'slots.csv',index=False)
        write(directory/'runtime_segment_000.json',{'seconds':1.,'phase':phase})
        return pd.DataFrame({'task_index':[0,1]}),[],[]
    monkeypatch.setattr(r,'execute',fake_execute)
    result=r.run_recovery(root)
    assert result['status']=='completed' and captured['tasks']==value['tasks']
    assert captured['workers']==2 and captured['timeout_seconds']==180. and captured['fail_group'] is True
    assert captured['directory']==root/r.RECOVERY and captured['phase']==r.RECOVERY
    assert json.loads((root/r.ORIGINAL/'episode_results.status.json').read_text())['status']=='running'
    with pytest.raises(ValueError,match='exactly2'):r.run_recovery(root,workers=3)


@pytest.mark.parametrize('live,status',[(True,'completed_with_failures'),(False,'running')])
def test_finalize_waits_for_pipeline_exit_and_terminal_stage(sandbox,monkeypatch,live,status):
    root,*_=sandbox;plan(sandbox)
    write(root/r.ORIGINAL/'episode_results.status.json',{'status':status})
    monkeypatch.setattr(r,'_pipeline_processes',lambda:[{'pid':1}] if live else [])
    with pytest.raises(ValueError,match='alive|terminal'):
        r.finalize(root)
