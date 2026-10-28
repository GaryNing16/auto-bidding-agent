"""Reproducible staged baseline experiment. Test requires a frozen selection."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
from harmony_bidding.schema import fingerprint
from .classical import spec, fit_lp_table
from .data import DEFAULT_CACHE, build_cache, load_day, source_manifest, sha256_file
from .evaluator import run_episode, SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'baseline_comparison_20261028'
TRAIN_DAYS = ['period-7', 'period-8']
VALID_DAYS = ['period-21']
TEST_DAYS = ['period-24']
SEEDS = [20261028, 20261029]
TRAIN_SEEDS = [20261028, 20261029]
VALID_ROLES = ['0','8','16','24','32','40']


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@lru_cache(maxsize=4)
def cached_day(day_id):
    return load_day(day_id)


def initialize_worker():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)


def worker(task):
    task = dict(task)
    day = cached_day(task.pop('day_id'))
    sample = task.pop('smoke', False)
    if sample:
        day = replace(day, slots=tuple(replace(s, values=s.values[:4], participation_mask=s.participation_mask[:4],
            request_ids=s.request_ids[:4], original_order=s.original_order[:4]) for s in day.slots))
    return run_episode(day, **task)


def execute(tasks, output_csv, workers=4, save_slots=False, save_transitions=False):
    """Persist each successful complete episode; no old experiment data imported."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        raise ValueError(f'refusing to overwrite stage output {output_csv}; use its completed artifact or a fresh run')
    rows, slots, transitions = [], [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        futures = {pool.submit(worker, t): (i,t) for i,t in enumerate(tasks)}
        for future in as_completed(futures):
            i, task = futures[future]
            row, episode_slots, episode_transitions = future.result()
            row['task_index'] = i
            rows.append(row)
            pd.DataFrame([row]).to_csv(output_csv, mode='a', index=False, header=len(rows)==1)
            if save_slots:
                for r in episode_slots:
                    r['behavior_alpha'] = task['target_spec'].get('kwargs',{}).get('alpha', np.nan)
                slots.extend(episode_slots)
            if save_transitions:
                for p in sorted({t['player_id'] for t in episode_transitions}, key=int):
                    sequence = [t for t in episode_transitions if t['player_id']==p]
                    for t in sequence:
                        t['episode_id'] = f'{i}:{p}'
                    transitions.extend(sequence)
            if len(rows) == 1 or len(rows) % 12 == 0 or len(rows) == len(tasks):
                print(json.dumps({'stage': output_csv.stem, 'completed':len(rows),'total':len(tasks),
                                  'seconds':round(time.perf_counter()-started,1)}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows).sort_values('task_index')
    frame.to_csv(output_csv,index=False)
    dump(output_csv.with_suffix('.status.json'), {'status':'completed', 'episodes':len(rows),
        'seconds':time.perf_counter()-started, 'tasks_sha256':fingerprint(tasks), 'csv_sha256':sha256_file(output_csv)})
    return frame, slots, transitions


def tasks_for(candidates, opponents, days, roles, seeds=(20261028,), **extra):
    return [dict(day_id=d,target=p,target_spec=c['spec'],scenario=s,opponents=opponents,seed=seed,
                 algorithm=c['algorithm'],training_seed=c.get('training_seed',0),**extra)
            for c in candidates for d in days for s in SCENARIOS for p in roles for seed in seeds]


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        raise ValueError('protocol already exists; prepare is fresh-only')
    manifest=source_manifest()
    protocol={'run_id':OUT.name,'status':'preselection','rules_version':'harmony-A-B-local-v2',
        'source_upstream_commit':'c20de631a40e24669d1d6a8e2acefd8f492f172d',
        'source_split':{s:[p['day_id'] for p in manifest['periods'] if p['split']==native]
                        for s,native in [('train','train'),('validation','validation'),('test','local_holdout')]},
        'executed_split':{'train':TRAIN_DAYS,'validation':VALID_DAYS,'test':TEST_DAYS},
        'scope':'complete 72-slot periods, full request matrices; 2 train / 1 validation / 1 test; remaining source periods not evaluated',
        'test_roles':[str(i) for i in range(48)],'fixed_role':'0','validation_roles':VALID_ROLES,
        'evaluation_seeds':SEEDS,'training_seeds':TRAIN_SEEDS,'budget_scales':[1.0],
        'mixed_assignment':'floor((N-1)/2) fixed, others local PID; seeded once per day/target, unchanged within cycle',
        'selection':'classical and neural mean validation role-relative value against same-scenario frozen LocalPID; no test selection',
        'aggregation':'mean over test days/seeds/training seeds within player, then per-player ratio to LocalPID, then equal role mean',
        'runtime':{'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count(),'cuda':False}}
    dump(OUT/'protocol.json',protocol)
    for day in TRAIN_DAYS+VALID_DAYS:
        build_cache(day)


def calibrate(workers=4):
    # Price scale from only training values and original budgets, independent of test.
    day=cached_day(TRAIN_DAYS[0])
    max_value=sum(np.max(s.values,axis=1).sum() for s in day.slots)
    base=sum(a.budget for a in day.advertisers)/max_value
    grid=[float(base*x) for x in (.125,.25,.5,1.,2.,4.)]
    initial={'fixed':spec('FixedAlpha',alpha=base),'pid':spec('LocalPID',initial_alpha=base)}
    # Fixed-player is an expressly auxiliary tuning mode; each type gets a representative.
    candidates=[{'algorithm':f'alpha_{i}','spec':spec('FixedAlpha',alpha=a)} for i,a in enumerate(grid)]
    tasks=[dict(day_id=TRAIN_DAYS[0],target=p,target_spec=c['spec'],scenario='Fixed',opponents=initial,
                seed=SEEDS[0],algorithm=c['algorithm']) for c in candidates for p in [str(i) for i in range(48)]]
    frame,slots,_=execute(tasks,OUT/'opponent_fixed_search.csv',workers,save_slots=True)
    mapping={p: grid[int(g.sort_values(['total_value','algorithm'],ascending=[False,True]).iloc[0].algorithm.split('_')[1])]
             for p,g in frame.groupby('player_id')}
    fixed=spec('FixedAlpha',alpha=base,player_alphas=mapping)
    pid_candidates=[{'algorithm':f'pid_{i}','spec':spec('LocalPID',initial_alpha=base,player_alphas=mapping,kp=kp,ki=ki,kd=kd)}
                    for i,(kp,ki,kd) in enumerate([(0.5,.05,.05),(1.5,.1,.15),(3.,.3,.2)])]
    provisional={'fixed':fixed,'pid':pid_candidates[1]['spec']}
    tasks=[dict(day_id=VALID_DAYS[0],target=p,target_spec=c['spec'],scenario='PID',opponents=provisional,
                seed=SEEDS[0],algorithm=c['algorithm']) for c in pid_candidates for p in VALID_ROLES]
    pid_df,_,_=execute(tasks,OUT/'opponent_pid_search.csv',workers)
    # Geometric normalization by within-role mean avoids high-value type domination.
    pid_df['norm']=pid_df.total_value/pid_df.groupby('player_id').total_value.transform('mean')
    best=pid_df.groupby('algorithm').norm.mean().idxmax()
    selected=next(c for c in pid_candidates if c['algorithm']==best)
    opponents={'fixed':fixed,'pid':selected['spec']}
    dump(OUT/'opponents_frozen.json',opponents)
    dump(OUT/'calibration.json',{'base_alpha':base,'alpha_grid':grid,'pid_candidates':pid_candidates,
        'selected_pid':best,'data_used':TRAIN_DAYS[:1]+VALID_DAYS,'test_used':False})
    pd.DataFrame(slots).to_csv(OUT/'lp_training_slots.csv',index=False)
    fit_lp_table(slots,OUT/'online_lp_table.csv')


def tune(workers=4):
    opponents=read(OUT/'opponents_frozen.json'); cal=read(OUT/'calibration.json')
    base=float(np.median(list(opponents['fixed']['kwargs']['player_alphas'].values())))
    candidates=[]
    for i,m in enumerate([.5,1.,2.,4.]):
        candidates.extend([
            {'algorithm':f'FixedAlpha_{i}','spec':spec('FixedAlpha',alpha=base,multiplier=m)},
            {'algorithm':f'LocalPID_{i}','spec':{**opponents['pid'],'kwargs':{**opponents['pid']['kwargs'],'initial_alpha':base,'player_alphas':{},'multiplier':m}}},
            {'algorithm':f'OnlineLP_{i}','spec':spec('OnlineLP',table_path=str((OUT/'online_lp_table.csv').resolve()),
                multiplier=m,fallback_alpha=max(cal['alpha_grid']))}])
    frame,_,_=execute(tasks_for(candidates,opponents,VALID_DAYS,VALID_ROLES),OUT/'classical_validation.csv',workers)
    reference=frame[frame.algorithm=='LocalPID_1'].set_index(['scenario','player_id']).total_value
    frame['relative']=[r.total_value/reference.loc[(r.scenario,str(r.player_id))]
        if reference.loc[(r.scenario,str(r.player_id))]>0 else np.nan for r in frame.itertuples()]
    selected=[]
    for name in ['FixedAlpha','LocalPID','OnlineLP']:
        names=[c['algorithm'] for c in candidates if c['algorithm'].startswith(name+'_')]
        best=frame[frame.algorithm.isin(names)].groupby('algorithm').relative.mean().idxmax()
        candidate=next(c for c in candidates if c['algorithm']==best)
        selected.append({'algorithm':name,'spec':candidate['spec'],'validation_choice':best})
    dump(OUT/'classical_selected.json',selected)
    dump(OUT/'classical_search.json',{'shared_initial_alpha':base,'multipliers':[.5,1.,2.,4.],
        'candidates':candidates,'selected':selected,'reference':'LocalPID_1',
        'boundary_selected':[c['algorithm'] for c in selected if c['validation_choice'].endswith('_3')]})


def trajectories(workers=4):
    opponents=read(OUT/'opponents_frozen.json'); cal=read(OUT/'calibration.json')
    classical=read(OUT/'classical_selected.json')
    # All roles across two full training days, balanced scenario/type coverage.
    candidates=[]
    for c in classical[:2]:
        for multiplier in [.7,1.,1.4]:
            selected_multiplier = c['spec']['kwargs'].get('multiplier', 1.)
            modified={**c['spec'],'kwargs':{**c['spec']['kwargs'],'multiplier':selected_multiplier * multiplier}}
            candidates.append({'algorithm':f"behavior_{c['algorithm']}_{multiplier}",
                'spec':spec('Exploration',base_spec=modified,noise=.15)})
    tasks=[]
    for di,d in enumerate(TRAIN_DAYS):
        for i,p in enumerate([str(x) for x in range(48)]):
            for si,s in enumerate(SCENARIOS):
                c=candidates[(i+si+di)%len(candidates)]
                tasks.append(dict(day_id=d,target=p,target_spec=c['spec'],scenario=s,opponents=opponents,
                    seed=SEEDS[di],algorithm=c['algorithm'],collect=True,collect_all=False))
    _,_,trans=execute(tasks,OUT/'behavior_episodes.csv',workers,save_transitions=True)
    trans.sort(key=lambda t:(int(t['episode_id'].split(':')[0]),int(t['player_id']),t['time_slot']))
    arrays={'states':np.stack([t['state'] for t in trans]),'actions':np.array([t['action'] for t in trans]),
        'rewards':np.array([t['reward'] for t in trans]),'next_states':np.stack([t['next_state'] for t in trans]),
        'dones':np.array([t['done'] for t in trans]),'episode_id':np.array([t['episode_id'] for t in trans])}
    np.savez_compressed(OUT/'train_transitions.npz',**arrays)
    dump(OUT/'trajectory_provenance.json',{'days':TRAIN_DAYS,'source':'fresh unified AuctionEngine rollouts',
        'transitions':len(trans),'episodes':len(set(arrays['episode_id'])),'includes_opponents_own_legal_trajectories':False,
        'behavior_candidates':candidates,'behavior_multiplier_semantics':'relative to selected classical policy multiplier',
        'state_fields':'harmony_baselines.neural.FEATURE_NAMES','reward':'raw won pvalue sum','test_used':False,
        'sha256':sha256_file(OUT/'train_transitions.npz')})


def train(steps=600):
    from .neural import train_candidates
    records=train_candidates(OUT/'train_transitions.npz',OUT/'models',seeds=TRAIN_SEEDS,steps=steps,
        checkpoint_steps=(steps//3,steps),num_threads=2)
    print(json.dumps({'trained_candidates':len(records)}),flush=True)


def select(workers=4):
    from .upstream import provenance
    opponents=read(OUT/'opponents_frozen.json')
    records=read(OUT/'models'/'candidates.json')
    candidates=[{'algorithm':f"{r['algorithm']}|{r['config_index']}|{r['step']}",'training_seed':r['seed'],
        'spec':{'kind':'custom','class_path':'harmony_baselines.neural:NeuralAgent','kwargs':{'checkpoint':r['checkpoint']}}} for r in records]
    references=[c for c in read(OUT/'classical_selected.json') if c['algorithm']=='LocalPID']
    df,_,_=execute(tasks_for(candidates+references,opponents,VALID_DAYS,VALID_ROLES),OUT/'neural_validation.csv',workers)
    reference=df[df.algorithm=='LocalPID'].set_index(['scenario','player_id']).total_value
    df['relative']=[r.total_value/reference.loc[(r.scenario,str(r.player_id))] if reference.loc[(r.scenario,str(r.player_id))]>0 else np.nan for r in df.itertuples()]
    selected=read(OUT/'classical_selected.json')
    excluded=[]
    for name in sorted({r['algorithm'] for r in records}):
        subset=df[df.algorithm.str.startswith(name+'|')]
        valid=[]
        for candidate_name,part in subset.groupby('algorithm'):
            if set(part.training_seed)==set(TRAIN_SEEDS): valid.append(candidate_name)
            else: excluded.append({'candidate':candidate_name,'reason':'missing training seed'})
        subset=subset[subset.algorithm.isin(valid)]
        if subset.empty: continue
        scores=subset.groupby('algorithm').relative.mean()
        best=scores.idxmax()
        for c in candidates:
            if c['algorithm']==best:
                selected.append({**c,'algorithm':name,'validation_choice':best,'validation_relative':float(scores[best])})
    dump(OUT/'selected_frozen.json',{'selected':selected,'excluded_candidates':excluded,'validation_days':VALID_DAYS,'test_used':False,
        'opponents_sha256':sha256_file(OUT/'opponents_frozen.json'),
        'lp_table_sha256':sha256_file(OUT/'online_lp_table.csv'),
        'code_sha256':{str(p.relative_to(ROOT)):sha256_file(p)
            for p in sorted((ROOT/'harmony_baselines').glob('*.py')) + sorted((ROOT/'harmony_bidding').glob('*.py'))
                + [ROOT / item['path'] for item in provenance()['files'].values()]
                + [ROOT/'strategy_train_env/bidding_train_env/baseline/dt/utils.py',
                   ROOT/'simul_bidding_env/strategy/onlinelp_bidding_strategy.py',
                   ROOT/'simul_bidding_env/strategy/base_bidding_strategy.py']},
        'checkpoints':{c['spec']['kwargs']['checkpoint']:sha256_file(c['spec']['kwargs']['checkpoint']) for c in selected if 'checkpoint' in c['spec']['kwargs']}})
    protocol=read(OUT/'protocol.json');protocol['status']='frozen_before_test';protocol['selection_sha256']=sha256_file(OUT/'selected_frozen.json')
    dump(OUT/'protocol.json',protocol)


def test(workers=4):
    frozen=read(OUT/'selected_frozen.json'); protocol=read(OUT/'protocol.json')
    if protocol['status'] != 'frozen_before_test':
        raise ValueError('test requires a completed pre-test freeze')
    if (protocol['selection_sha256']!=sha256_file(OUT/'selected_frozen.json')
        or frozen['opponents_sha256']!=sha256_file(OUT/'opponents_frozen.json')
        or frozen['lp_table_sha256']!=sha256_file(OUT/'online_lp_table.csv')):
        raise ValueError('frozen selection or opponents changed')
    for p,digest in frozen['checkpoints'].items():
        if sha256_file(p)!=digest: raise ValueError('checkpoint changed after selection')
    for p,digest in frozen.get('code_sha256',{}).items():
        if sha256_file(ROOT/p)!=digest: raise ValueError(f'code changed after selection: {p}')
    for day in TEST_DAYS: build_cache(day)
    tasks=tasks_for(frozen['selected'],read(OUT/'opponents_frozen.json'),TEST_DAYS,[str(i) for i in range(48)],SEEDS)
    frame,_,_=execute(tasks,OUT/'episode_results.csv',workers)
    frame[frame.player_id.astype(str)=='0'].to_csv(OUT/'fixed_player_results.csv',index=False)
    for s in SCENARIOS: frame[frame.scenario==s].to_csv(OUT/f'players_{s}.csv',index=False)
    protocol['status']='completed';protocol['test_episodes']=len(frame);dump(OUT/'protocol.json',protocol)


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','calibrate','tune','trajectories','train','select','test'])
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--output', type=Path, help='Fresh experiment output directory; never edits sealed source')
    a=parser.parse_args()
    if a.output is not None:
        OUT = a.output.resolve()
    if a.stage=='prepare':prepare()
    elif a.stage=='train':train(a.steps)
    else:globals()[a.stage](a.workers)


if __name__=='__main__':main()
