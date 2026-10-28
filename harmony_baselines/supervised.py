"""Audited resumable full-episode execution with per-process resource deadlines.

Failures are never scores. Validation rejects incomplete paired-seed candidates;
test never changes the frozen selection and quarantines incomplete algorithms.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import pandas as pd

from . import experiment as exp
from .data import sha256_file
from harmony_bidding.schema import fingerprint
from .evaluator import opponent_specs
from .upstream import provenance


def _episode_payload(task, diagnostics_path):
    """Fresh Agents/reset per episode; only immutable day/network caches persist."""
    from . import evaluator
    original_factory = evaluator.make_agent
    timings = {}
    handle = Path(diagnostics_path).open('w', encoding='utf-8', buffering=1)
    def write(value):
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        handle.flush()
    def factory(specification):
        agent = original_factory(specification)
        if specification == task['target_spec']:
            bidding, update = agent.bidding, agent.update
            def traced_bid(observation):
                if str(observation.own_config.advertiser_id) != str(task['target']):
                    return bidding(observation)
                write({'phase':'before_action','slot':observation.time_slot,
                       'remaining_budget':observation.remaining_budget})
                before = time.perf_counter()
                value = bidding(observation)
                duration = (time.perf_counter()-before)*1000
                timings[observation.time_slot] = [duration, None]
                write({'phase':'action','slot':observation.time_slot,'alpha':float(value)})
                return value
            def traced_update(feedback):
                if str(agent.own_config.advertiser_id) != str(task['target']):
                    return update(feedback)
                before = time.perf_counter()
                update(feedback)
                duration = (time.perf_counter()-before)*1000
                timings[feedback.time_slot][1] = duration
                write({'phase':'feedback','slot':feedback.time_slot,'alpha':feedback.alpha,
                       'value':feedback.value,'cost':feedback.cost,'remaining_budget':feedback.remaining_budget})
            agent.bidding, agent.update = traced_bid, traced_update
        return agent
    evaluator.make_agent = factory
    try:
        row, _, _ = exp.worker(task)
        timing = np.asarray([timings[t] for t in range(72)], dtype=float)
        if not np.isfinite(timing).all():
            raise ValueError('missing target policy timing')
        row.update(bidding_p99_ms=float(np.percentile(timing[:,0],99)),
                   update_p99_ms=float(np.percentile(timing[:,1],99)),
                   paired_p99_ms=float(np.percentile(timing.sum(axis=1),99)))
        return {'status':'completed','row':row}
    except BaseException as exc:
        return {'status':'failed','exception_type':type(exc).__name__,
                'reason':str(exc),'traceback':traceback.format_exc()}
    finally:
        evaluator.make_agent = original_factory
        handle.close()


def _child(task, diagnostics_path, connection):
    """Compatibility worker for an already running one-process-per-case resume."""
    exp.initialize_worker()
    try:
        connection.send(_episode_payload(task,diagnostics_path))
    finally:
        connection.close()


def _persistent_child(connection):
    exp.initialize_worker()
    try:
        while True:
            request=connection.recv()
            if request is None:break
            task,path=request
            connection.send(_episode_payload(task,path))
    except (EOFError,BrokenPipeError):
        pass
    finally:
        connection.close()


def _paths(phase):
    prefix = 'validation' if phase=='validation' else 'evaluation'
    csv_name = 'neural_validation.csv' if phase=='validation' else 'episode_results.csv'
    return exp.OUT/csv_name, exp.OUT/f'{prefix}_failures.json', exp.OUT/f'{prefix}_skipped.json'


def _case(task, index, phase):
    label=task['algorithm']
    parts=label.split('|')
    return {'task_index':index,'candidate':label,'algorithm':parts[0],
            'config_index':int(parts[1]) if len(parts)==3 else None,
            'step':int(parts[2]) if len(parts)==3 else None,
            'training_seed':int(task.get('training_seed',0)),'day_id':task['day_id'],
            'scenario':task['scenario'],'player_id':str(task['target']),
            'evaluation_seed':int(task['seed']),'phase':phase,
            'scope':'full_72_slot_period','test_used':phase=='test'}


def _completed_slots(path):
    if not Path(path).exists(): return 0
    count=0
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        try: count += json.loads(line).get('phase')=='feedback'
        except json.JSONDecodeError: pass
    return count


def _validate_existing(frame, tasks):
    if frame.empty: return
    if frame.task_index.duplicated().any(): raise ValueError('duplicate completed task index')
    for row in frame.itertuples():
        i=int(row.task_index)
        if i<0 or i>=len(tasks): raise ValueError('completed task outside frozen task protocol')
        task=tasks[i]
        for field, expected in [('algorithm',task['algorithm']),('day_id',task['day_id']),
                ('scenario',task['scenario']),('player_id',str(task['target']))]:
            if str(getattr(row,field)) != expected: raise ValueError(f'resume task mismatch: {i}/{field}')
        if int(row.seed)!=task['seed'] or int(row.training_seed)!=task.get('training_seed',0):
            raise ValueError('resume seed mismatch')
        if str(row.completed).lower()!='true': raise ValueError('incomplete row cannot be resumed as successful')
        if row.model_fingerprint!=fingerprint(task['target_spec']) or row.opponent_fingerprint!=fingerprint(task['opponents']):
            raise ValueError('resume policy definition mismatch')
        if row.rules_version!='harmony-A-B-local-v2': raise ValueError('resume rule mismatch')
        day=exp.cached_day(task['day_id'])
        if row.dataset_fingerprint!=day.fingerprint: raise ValueError('resume dataset mismatch')
        ids=[a.advertiser_id for a in day.advertisers]
        allocation=opponent_specs(ids,task['target'],task['scenario'],task['opponents'],task['seed']+1000003,task['day_id'])
        if row.assignment_fingerprint!=fingerprint(allocation): raise ValueError('resume opponent allocation mismatch')
        metrics=[row.total_value,row.total_cost,row.initial_budget,row.budget_utilization,row.wins]
        if not np.isfinite(metrics).all() or min(metrics)<0 or row.total_cost>row.initial_budget+1e-6:
            raise ValueError('invalid completed episode metrics')


def execute(tasks, phase, workers=4, timeout_seconds=180., persistent=True):
    """Resume only matching current-run tasks; reject/skip complete groups on failure."""
    output, failure_path, skipped_path = _paths(phase)
    exp.OUT.mkdir(parents=True, exist_ok=True)
    manifest_path=exp.OUT/f'{phase}_task_protocol.json'
    hashes={str(p):sha256_file(p) for p in {Path(t['target_spec']['kwargs']['checkpoint']) for t in tasks
             if 'checkpoint' in t['target_spec'].get('kwargs',{})}}
    task_hash=fingerprint(tasks)
    protocol={'tasks':tasks,'tasks_sha256':task_hash,'checkpoint_sha256':hashes,
              'phase':phase,'timeout_seconds':timeout_seconds}
    if manifest_path.exists():
        old=exp.read(manifest_path)
        # Stability tuning may append new candidates, but never rewrite existing tasks.
        if old['tasks'] != tasks[:len(old['tasks'])] or any(hashes.get(p)!=h for p,h in old['checkpoint_sha256'].items()):
            raise ValueError('resume task protocol/checkpoint changed')
        if len(old['tasks'])!=len(tasks):
            exp.dump(exp.OUT/f'{phase}_task_protocol_before_extension.json',old)
    exp.dump(manifest_path,protocol)
    frame=pd.read_csv(output,dtype={'player_id':str,'player_type':str,'day_id':str}) if output.exists() else pd.DataFrame()
    _validate_existing(frame,tasks)
    failures=exp.read(failure_path) if failure_path.exists() else []
    skipped=exp.read(skipped_path) if skipped_path.exists() else []
    successful=set(frame.task_index.astype(int)) if not frame.empty else set()
    failed_ids={int(r['task_index']) for r in failures}
    skipped_ids={int(r['task_index']) for r in skipped}
    if successful & (failed_ids|skipped_ids) or failed_ids & skipped_ids:
        raise ValueError('success/failure/skipped task sets overlap')
    if any(i>=len(tasks) or i<0 for i in failed_ids|skipped_ids): raise ValueError('invalid failure task index')
    group=lambda t:t['algorithm']
    candidate=lambda t:(t['algorithm'],int(t.get('training_seed',0)))
    poisoned={group(tasks[i]) for i in failed_ids}
    proven={candidate(tasks[i]) for i in successful}
    pending=[i for i in range(len(tasks)) if i not in successful|failed_ids|skipped_ids]
    context=mp.get_context('spawn')
    running={}
    idle=[]
    diagnostics=exp.OUT/f'{phase}_diagnostics'
    diagnostics.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    initial_success=len(successful)
    fields=list(frame.columns) if not frame.empty else None
    def save_status(status='running'):
        exp.dump(failure_path,failures);exp.dump(skipped_path,skipped)
        if phase=='test':
            exp.dump(exp.OUT/'test_failures.json',failures);exp.dump(exp.OUT/'test_skipped.json',skipped)
        exp.dump(output.with_suffix('.status.json'),{'status':status,'episodes':len(successful),
            'expected_episodes':len(tasks),'failure_episodes':len(failures),'skipped_episodes':len(skipped),
            'resumed_successful_episodes':initial_success,'seconds':time.perf_counter()-start,
            'tasks_sha256':task_hash,'csv_sha256':sha256_file(output) if output.exists() else None,
            'timeout_seconds':timeout_seconds,'workers':workers,'persistent_workers':persistent})
    def skip(i, reason, diagnostic=None):
        record={**_case(tasks[i],i,phase),'status':'not_run','reason':reason,
                'diagnostics_path':str(diagnostic) if diagnostic else None}
        skipped.append(record);skipped_ids.add(i)
    save_status()
    try:
        while pending or running:
            for i in list(pending):
                if group(tasks[i]) in poisoned:
                    pending.remove(i);skip(i,'candidate group failed; paired seeds excluded' if phase=='validation' else 'algorithm incomplete after a frozen-model failure')
            for i,(process,connection,started,path) in list(running.items()):
                if group(tasks[i]) in poisoned:
                    if process.is_alive(): process.terminate()
                    process.join(2);connection.close();del running[i]
                    skip(i,'cancelled in-flight after same candidate/algorithm failure',path)
            # At most one unproven episode for each model; the first is an actual
            # required complete validation/test cell, never a separate scoring proxy.
            while len(running)<workers and pending:
                active_candidates={candidate(tasks[i]) for i in running}
                options=[i for i in pending if candidate(tasks[i]) in proven or candidate(tasks[i]) not in active_candidates]
                if not options: break
                i=options[0];pending.remove(i)
                path=diagnostics/f'task_{i}.jsonl'
                if persistent:
                    if idle:
                        process,parent=idle.pop()
                    else:
                        parent,child=context.Pipe(duplex=True)
                        process=context.Process(target=_persistent_child,args=(child,))
                        process.start();child.close()
                    parent.send((tasks[i],str(path)))
                else:
                    parent,child=context.Pipe(duplex=False)
                    process=context.Process(target=_child,args=(tasks[i],str(path),child))
                    process.start();child.close()
                running[i]=(process,parent,time.perf_counter(),path)
            changed=False
            for i,(process,connection,started,path) in list(running.items()):
                elapsed=time.perf_counter()-started
                message=None
                if connection.poll():
                    try: message=connection.recv()
                    except EOFError: message={'status':'failed','exception_type':'WorkerExit','reason':'worker exited without a complete result'}
                elif elapsed>timeout_seconds:
                    if process.is_alive(): process.terminate()
                    message={'status':'resource_incomplete','exception_type':None,
                             'reason':f'full episode exceeded {timeout_seconds:g}s wall-clock resource deadline'}
                elif not process.is_alive():
                    message={'status':'failed','exception_type':'WorkerExit','reason':f'worker exited with code {process.exitcode}'}
                if message is None: continue
                if persistent and process.is_alive() and message['status']!='resource_incomplete':
                    idle.append((process,connection))
                else:
                    process.join(2);connection.close()
                del running[i];changed=True
                if message['status']=='completed':
                    row=message['row'];row['task_index']=i
                    _validate_existing(pd.DataFrame([row]),tasks)
                    if fields is None: fields=list(row)
                    if set(row)!=set(fields): raise ValueError('resumed result schema changed')
                    with output.open('a',newline='',encoding='utf-8') as handle:
                        writer=csv.DictWriter(handle,fieldnames=fields)
                        if not successful: writer.writeheader()
                        writer.writerow(row)
                    successful.add(i);proven.add(candidate(tasks[i]))
                else:
                    record={**_case(tasks[i],i,phase),**message,'elapsed_seconds':elapsed,
                        'timeout_seconds':timeout_seconds,'completed_slots':_completed_slots(path),
                        'diagnostics_path':str(path.resolve())}
                    failures.append(record);failed_ids.add(i);poisoned.add(group(tasks[i]))
                    print(json.dumps({'phase':phase,'failure':record},ensure_ascii=False),flush=True)
            if changed:
                save_status()
                print(json.dumps({'stage':phase,'successful':len(successful),'failed':len(failures),
                    'skipped':len(skipped),'expected':len(tasks),'seconds':round(time.perf_counter()-start,1)},ensure_ascii=False),flush=True)
            if not changed: time.sleep(.1)
        if successful|failed_ids|skipped_ids != set(range(len(tasks))): raise ValueError('incomplete task accounting')
        if output.exists():
            frame=pd.read_csv(output,dtype={'player_id':str,'player_type':str,'day_id':str}).sort_values('task_index')
            frame.to_csv(output,index=False)
        else: frame=pd.DataFrame()
        save_status('completed_with_failures' if failures else 'completed')
        return frame,failures,skipped
    finally:
        for process,connection,_,_ in running.values():
            if process.is_alive():process.terminate()
            process.join(2);connection.close()
        for process,connection in idle:
            try:connection.send(None)
            except (BrokenPipeError,EOFError,OSError):pass
            process.join(2)
            if process.is_alive():process.terminate();process.join(2)
            connection.close()


def _candidate_specs(records):
    return [{'algorithm':f"{r['algorithm']}|{r['config_index']}|{r['step']}",'training_seed':r['seed'],
        'spec':{'kind':'custom','class_path':'harmony_baselines.neural:NeuralAgent',
                'kwargs':{'checkpoint':r['checkpoint']}}} for r in records]


def _valid_groups(frame, candidates, failures):
    bad={x['candidate'] for x in failures}
    expected=len(exp.VALID_DAYS)*len(exp.SCENARIOS)*len(exp.VALID_ROLES)
    valid=[];excluded=[]
    for label in sorted({c['algorithm'] for c in candidates}):
        part=frame[frame.algorithm==label]
        seeds=set(part.training_seed.astype(int))
        counts=part.groupby('training_seed').size()
        complete=seeds==set(exp.TRAIN_SEEDS) and all(counts.get(s,0)==expected for s in exp.TRAIN_SEEDS)
        if complete and label not in bad: valid.append(label)
        else:
            matching=[x for x in failures if x['candidate']==label]
            excluded.append({'candidate':label,'reason':'incomplete paired-seed validation coverage',
                'failure_count':sum(x['status']=='failed' for x in matching),
                'resource_incomplete_count':sum(x['status']=='resource_incomplete' for x in matching),
                'expected_training_seeds':list(exp.TRAIN_SEEDS),'observed_training_seeds':sorted(seeds),
                'complete_cases':len(part),'expected_cases':expected*len(exp.TRAIN_SEEDS)})
    return valid,excluded


def select(workers=4, timeout_seconds=180.):
    out=exp.OUT
    if (out/'selected_frozen.json').exists():raise ValueError('selection already frozen')
    records=exp.read(out/'models/candidates.json')
    candidates=_candidate_specs(records)
    references=[c for c in exp.read(out/'classical_selected.json') if c['algorithm']=='LocalPID']
    opponents=exp.read(out/'opponents_frozen.json')
    all_tasks=exp.tasks_for(candidates+references,opponents,exp.VALID_DAYS,exp.VALID_ROLES)
    previous_protocol=out/'validation_task_protocol.json'
    if previous_protocol.exists():
        tasks=exp.read(previous_protocol)['tasks']
        known={fingerprint(t) for t in tasks}
        tasks += [t for t in all_tasks if fingerprint(t) not in known]
    else:tasks=all_tasks
    output,_,_=_paths('validation')
    if output.exists() and not (out/'neural_validation_interrupted.csv').exists():
        snapshot=out/'neural_validation_interrupted.csv'
        shutil.copy2(output,snapshot)
        exp.dump(out/'selection_interruption.json',{'reason':'CQL execution stalled; full case resource supervision required',
            'stopped_select_pids':[20692,2532,18308,22012,17244,24668],
            'command':'python -X utf8 -m harmony_baselines.experiment select --workers 4',
            'preserved_rows':len(pd.read_csv(snapshot)),'snapshot_sha256':sha256_file(snapshot),
            'original_tasks_sha256':fingerprint(tasks),'small_probe_is_not_formal_failure':True})
    frame,failures,_=execute(tasks,'validation',workers,timeout_seconds)
    valid,excluded=_valid_groups(frame,candidates,failures)
    # Formal overflow established a representation risk. Retain all healthy
    # original settings and add the upstream-like linear action representation
    # as an explicitly larger validation-only CQL search budget.
    if any(r['algorithm']=='CQL' for r in records):
        extension_path=out/'models/stability_tuning.json'
        if not extension_path.exists():
            extra_dir=out/'models_cql_linear'
            extra=exp.read(extra_dir/'candidates.json')
            metadata=exp.read(extra_dir/'training_metadata.json')
            if len(extra)!=4 or exp.read(extra_dir/'failures.json'):
                raise ValueError('expected four successfully trained additional CQL checkpoints')
            if metadata['source_sha256']!=sha256_file(out/'train_transitions.npz'):
                raise ValueError('CQL extension training source changed')
            if any(r['algorithm']!='CQL' or r['config'].get('action_transform')!='linear_scaled'
                   or r['config']['learning_rate']!=1e-4 or sha256_file(r['checkpoint'])!=r['checkpoint_sha256'] for r in extra):
                raise ValueError('unexpected CQL stability candidate provenance')
            next_index=max(r['config_index'] for r in records if r['algorithm']=='CQL')+1
            for record in extra:
                record['origin_config_index']=record['config_index']
                record['config_index']=next_index
            shutil.copy2(out/'models/candidates.json',out/'models/candidates_before_stability.json')
            exp.dump(out/'models/candidates.json',records+extra)
            exp.dump(extension_path,{'reason':'formal validation found log-action exponential feedback overflow; add linear action representation closer to upstream',
                'learning_rate':1e-4,'steps':600,'checkpoint_steps':[200,600],'seeds':exp.TRAIN_SEEDS,
                'action_transform':'linear_scaled',
                'extra_training_runs':len(exp.TRAIN_SEEDS),'extra_records':extra,'test_used':False,
                'note':'extra stability tuning budget, not equal search budget across algorithms'})
        else: extra=exp.read(extension_path)['extra_records']
        records=exp.read(out/'models/candidates.json')
        new_candidates=_candidate_specs(extra)
        # Preserve every original task index; extensions are append-only.
        known={fingerprint(t) for t in tasks}
        tasks += [t for t in exp.tasks_for(new_candidates,opponents,exp.VALID_DAYS,exp.VALID_ROLES)
                  if fingerprint(t) not in known]
        candidates = _candidate_specs(records)
        frame,failures,_=execute(tasks,'validation',workers,timeout_seconds)
        valid,excluded=_valid_groups(frame,candidates,failures)
    reference=frame[frame.algorithm=='LocalPID'].set_index(['scenario','player_id']).total_value
    if len(reference)!=len(exp.VALID_ROLES)*len(exp.SCENARIOS)*len(exp.VALID_DAYS):
        raise ValueError('validation LocalPID reference incomplete')
    frame['relative']=[r.total_value/reference.loc[(r.scenario,str(r.player_id))]
        if reference.loc[(r.scenario,str(r.player_id))]>0 else np.nan for r in frame.itertuples()]
    selected=exp.read(out/'classical_selected.json')
    for name in sorted({r['algorithm'] for r in records}):
        subset=frame[frame.algorithm.isin([x for x in valid if x.startswith(name+'|')])]
        if subset.empty:continue
        scores=subset.groupby('algorithm').relative.mean().dropna()
        if scores.empty:continue
        best=scores.idxmax()
        selected += [{**c,'algorithm':name,'validation_choice':best,'validation_relative':float(scores[best])}
                     for c in candidates if c['algorithm']==best]
    code=list(sorted((exp.ROOT/'harmony_baselines').glob('*.py')))+list(sorted((exp.ROOT/'harmony_bidding').glob('*.py')))
    code += [exp.ROOT/item['path'] for item in provenance()['files'].values()]
    code += [exp.ROOT/'strategy_train_env/bidding_train_env/baseline/dt/utils.py',
             exp.ROOT/'simul_bidding_env/strategy/onlinelp_bidding_strategy.py',
             exp.ROOT/'simul_bidding_env/strategy/base_bidding_strategy.py']
    exp.dump(out/'selected_frozen.json',{'selected':selected,'excluded_candidates':excluded,
        'validation_days':exp.VALID_DAYS,'test_used':False,'opponents_sha256':sha256_file(out/'opponents_frozen.json'),
        'lp_table_sha256':sha256_file(out/'online_lp_table.csv'),
        'code_sha256':{str(p.relative_to(exp.ROOT)):sha256_file(p) for p in code},
        'checkpoints':{c['spec']['kwargs']['checkpoint']:sha256_file(c['spec']['kwargs']['checkpoint'])
                      for c in selected if 'checkpoint' in c['spec']['kwargs']}})
    protocol=exp.read(out/'protocol.json');protocol['status']='frozen_before_test'
    protocol['selection_sha256']=sha256_file(out/'selected_frozen.json')
    protocol['validation_resource_timeout_seconds']=timeout_seconds
    exp.dump(out/'protocol.json',protocol)
    print(json.dumps({'selection_frozen':True,'selected_algorithms':sorted({c['algorithm'] for c in selected}),
                      'excluded_candidates':excluded},ensure_ascii=False),flush=True)


def test(workers=4, timeout_seconds=180.):
    out=exp.OUT
    frozen=exp.read(out/'selected_frozen.json');protocol=exp.read(out/'protocol.json')
    if protocol['status']!='frozen_before_test':raise ValueError('test requires a completed pre-test freeze')
    if protocol['selection_sha256']!=sha256_file(out/'selected_frozen.json'):raise ValueError('frozen selection changed')
    for path,digest in [(out/'opponents_frozen.json',frozen['opponents_sha256']),
                        (out/'online_lp_table.csv',frozen['lp_table_sha256'])]:
        if sha256_file(path)!=digest:raise ValueError('frozen asset changed')
    for path,digest in frozen['checkpoints'].items():
        if sha256_file(path)!=digest:raise ValueError('frozen checkpoint changed')
    for path,digest in frozen['code_sha256'].items():
        if sha256_file(exp.ROOT/path)!=digest:raise ValueError('frozen source changed: '+path)
    for day in exp.TEST_DAYS:exp.build_cache(day)
    tasks=exp.tasks_for(frozen['selected'],exp.read(out/'opponents_frozen.json'),
                        exp.TEST_DAYS,protocol['test_roles'],exp.SEEDS)
    frame,failures,skipped=execute(tasks,'test',workers,timeout_seconds)
    frame[frame.player_id.astype(str)==protocol['fixed_role']].to_csv(out/'fixed_player_results.csv',index=False)
    for scenario in exp.SCENARIOS:frame[frame.scenario==scenario].to_csv(out/f'players_{scenario}.csv',index=False)
    incomplete=sorted({x['algorithm'] for x in failures+skipped})
    protocol.update(status='completed_with_failures' if failures else 'completed',
        incomplete_algorithms=incomplete,test_expected_episodes=len(tasks),test_episodes=len(frame),
        test_failure_episodes=len(failures),test_skipped_episodes=len(skipped),
        test_resource_timeout_seconds=timeout_seconds)
    exp.dump(out/'protocol.json',protocol)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['select','test'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--timeout',type=float,default=180.)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output is not None:exp.OUT=args.output.resolve()
    globals()[args.stage](args.workers,args.timeout)


if __name__=='__main__':main()
