"""Audited resource-interruption recovery; never changes the frozen runner.

plan may run while the original stage is active. run writes a separate stage.
finalize requires the original pipeline to have exited and retains all failures.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import time

import pandas as pd
import psutil

from harmony_baselines.data import sha256_file
from harmony_bidding.schema import fingerprint
from . import common as c
from .execution import execute, _validate

ORIGINAL = 'independent_test_new'
FINAL = 'independent_test'
RECOVERY = 'independent_test_recovery_001'


def _once(path, value):
    c.immutable(Path(path), value)


def _copy_once(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) != digest:
            raise ValueError('Existing evidence differs: ' + str(destination))
    else:
        pending = destination.with_name(destination.name + '.pending')
        shutil.copyfile(source, pending)
        pending.replace(destination)
    return digest


def _freeze(root):
    frozen = c.read(root/'experiment_B/selected_frozen.json')
    for path, digest in frozen['execution_code_sha256'].items():
        if sha256_file(c.ROOT/path) != digest:
            raise ValueError('Frozen runtime source changed: ' + path)
    for path, digest in frozen['assets'].items():
        if sha256_file(path) != digest:
            raise ValueError('Frozen policy asset changed: ' + path)
    for path, digest in ((root/'protocol.json', frozen['protocol_sha256']),
                         (root/'opponents_frozen.json', frozen['opponents_sha256']),
                         (Path(__file__).with_name('pipeline.py'), frozen['pipeline_sha256'])):
        if sha256_file(path) != digest:
            raise ValueError('Frozen input changed: ' + str(path))
    if c.read(root/FINAL/'selected_frozen.json') != frozen:
        raise ValueError('Independent selection differs from frozen B')
    return frozen


def _terminal_records(root, protocol):
    directory = root/ORIGINAL
    failures, skipped = c.read(directory/'failures.json'), c.read(directory/'skipped.json')
    by_index = {}
    for label, records in (('failure', failures), ('skipped', skipped)):
        for record in records:
            i = record['task_index']
            if (isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(protocol['tasks'])
                    or i in by_index or record['phase'] != ORIGINAL
                    or record['task_key'] != protocol['identities'][i]['task_key']):
                raise ValueError('Invalid/duplicate terminal task identity')
            if label == 'failure':
                if record['task'] != protocol['tasks'][i] or record['status'] not in ('failed','resource_incomplete'):
                    raise ValueError('Failure task differs from immutable protocol')
            elif record['algorithm'] != protocol['tasks'][i]['algorithm'] or record['status'] != 'not_run':
                raise ValueError('Skipped task differs from immutable protocol')
            if (directory/'results'/f'task_{i:05d}.json').exists():
                raise ValueError('Terminal missing task also has a committed result')
            by_index[i] = {'kind':label, 'record':record}
    return failures, skipped, by_index


def _verify_plan(root, plan):
    frozen = _freeze(root)
    if sha256_file(__file__) != plan['orchestrator_sha256']:
        raise ValueError('Recovery orchestrator changed after plan')
    if (sha256_file(root/ORIGINAL/'task_protocol.json') != plan['original_task_protocol_sha256']
            or sha256_file(root/'experiment_B/selected_frozen.json') != plan['B_freeze_sha256']):
        raise ValueError('Immutable original protocol/freeze changed')
    original = c.read(root/ORIGINAL/'task_protocol.json')
    if original['code_sha256'] != frozen['execution_code_sha256']:
        raise ValueError('Original runtime identity differs from freeze')
    _, _, terminal = _terminal_records(root, original)
    if not len(plan['tasks']) == len(plan['identities']) == len(plan['mapping']):
        raise ValueError('Recovery task length differs')
    originals, locals_ = set(), set()
    for item in plan['mapping']:
        old, new = item['original_task_index'], item['recovery_task_index']
        if old in originals or new in locals_ or new != len(locals_):
            raise ValueError('Recovery mapping is not one-to-one/contiguous')
        originals.add(old); locals_.add(new)
        if (terminal.get(old) != item['terminal'] or plan['tasks'][new] != original['tasks'][old]
                or plan['identities'][new] != original['identities'][old]
                or item['task_key'] != original['identities'][old]['task_key']):
            raise ValueError('Recovery mapping/terminal evidence changed')
        if item.get('diagnostic_sha256'):
            p = Path(item['original_diagnostic_path'])
            if sha256_file(p) != item['diagnostic_sha256'] or p.stat().st_mtime_ns != item['diagnostic_mtime_ns']:
                raise ValueError('Original terminated diagnostic changed')
    for path, digest in plan['evidence_sha256'].items():
        if sha256_file(path) != digest:
            raise ValueError('Sealed recovery evidence changed: ' + path)
    return original, terminal


def create_plan(root=c.OUT, expected_missing=146):
    root = Path(root).resolve(); directory = root/RECOVERY; path = directory/'recovery_plan.json'
    if path.exists():
        plan = c.read(path); _verify_plan(root, plan); return plan
    frozen = _freeze(root); original = c.read(root/ORIGINAL/'task_protocol.json')
    if original['timeout_seconds'] != 180. or original['code_sha256'] != frozen['execution_code_sha256']:
        raise ValueError('Recovery requires original frozen runtime and180-second resource rule')
    failures, skipped, terminal = _terminal_records(root, original)
    if not failures or len(terminal) != expected_missing:
        raise ValueError('Unexpected terminal missing set; refusing a changing recovery scope')
    if any(r['status'] != 'resource_incomplete' or r['exception_type'] != 'Timeout' for r in failures):
        raise ValueError('This recovery is limited to resource timeouts and consequent skips')
    evidence = directory/'evidence'; evidence.mkdir(parents=True, exist_ok=True)
    anchors = {}
    for name, value in (('original_failures.sealed.json',failures),('original_skipped.sealed.json',skipped)):
        _once(evidence/name,value); anchors[str(evidence/name)] = sha256_file(evidence/name)
    incident = root/'qa/independent_interruption_20261028'
    if not (incident/'incident_diagnosis.json').exists():
        raise ValueError('Missing interruption diagnosis including user report and system evidence')
    for name in ('incident_diagnosis.json','windows_power_events.json'):
        anchors[str(evidence/name)] = _copy_once(incident/name,evidence/name)
    mapping, tasks, identities = [], [], []
    for local, old in enumerate(sorted(terminal)):
        item = {'original_task_index':old, 'recovery_task_index':local,
                'task_key':original['identities'][old]['task_key'], 'terminal':terminal[old],
                'recovery_diagnostic_path':str(directory/'diagnostics'/f'task_{local:05d}.jsonl')}
        source = root/ORIGINAL/'diagnostics'/f'task_{old:05d}.jsonl'
        if terminal[old]['kind'] == 'failure' and not source.is_file():
            raise ValueError('Missing original timeout diagnostic: ' + str(source))
        if source.exists():
            destination = evidence/'original_diagnostics'/source.name
            digest = _copy_once(source,destination); anchors[str(destination)] = digest
            item.update(original_diagnostic_path=str(source), original_diagnostic_snapshot=str(destination),
                        diagnostic_sha256=digest, diagnostic_mtime_ns=source.stat().st_mtime_ns)
        mapping.append(item); tasks.append(original['tasks'][old]); identities.append(original['identities'][old])
    plan = {'schema_version':1, 'recovery_stage':RECOVERY, 'original_stage':ORIGINAL, 'final_stage':FINAL,
            'original_task_protocol_sha256':sha256_file(root/ORIGINAL/'task_protocol.json'),
            'B_freeze_sha256':sha256_file(root/'experiment_B/selected_frozen.json'),
            'orchestrator_sha256':sha256_file(__file__), 'runtime_code_sha256':frozen['execution_code_sha256'],
            'workers':2, 'timeout_seconds':180., 'fail_group':original['fail_group'],
            'historical_failed_episodes':len(failures), 'historical_skipped_episodes':len(skipped),
            'expected_recovery_episodes':len(mapping), 'mapping':mapping, 'tasks':tasks, 'identities':identities,
            'evidence_sha256':anchors, 'user_report':'用户称“刚刚是断网了”，明确授权重跑超时任务。',
            'system_evidence':'Kernel-Power506/507 confirms Modern Standby for1971.219687 seconds; network outage remains user-reported.',
            'policy':'Keep all existing successful cases and their original timings, including standby-overlap successes. Recover only terminal missing tasks; no selection, cap, seed or timeout change.',
            'mutable_original_files_excluded_from_anchors':['episode_results.status.json','episode_results.csv','slots.csv']}
    _verify_plan(root,plan); _once(path,plan); return plan


def run_recovery(root=c.OUT, workers=2):
    root=Path(root).resolve(); directory=root/RECOVERY; plan=c.read(directory/'recovery_plan.json')
    _verify_plan(root,plan)
    if workers != 2 or isinstance(workers,bool):
        raise ValueError('This recovery plan permits exactly2workers')
    frame, failures, skipped = execute(plan['tasks'],directory,RECOVERY,workers=2,
                                       timeout_seconds=180.,fail_group=plan['fail_group'])
    _verify_plan(root,plan)
    receipt={'status':'completed' if not failures and not skipped and len(frame)==len(plan['tasks']) else 'incomplete',
             'recovery_stage':RECOVERY,'plan_sha256':sha256_file(directory/'recovery_plan.json'),
             'completed_episodes':len(frame),'expected_episodes':len(plan['tasks']),
             'failed_episodes':len(failures),'skipped_episodes':len(skipped),
             'result_sha256':sha256_file(directory/'episode_results.csv'),
             'slots_sha256':sha256_file(directory/'slots.csv'),
             'runtime_segments':[{'path':str(p),'sha256':sha256_file(p),**c.read(p)} for p in sorted(directory.glob('runtime_segment_*.json'))]}
    c.dump(directory/'recovery_receipt.json',receipt)
    return receipt


def _pipeline_processes():
    result=[]
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            cmd=process.info['cmdline'] or []
            if 'harmony_baselines.five_scenarios.pipeline' in cmd:
                result.append({'pid':process.pid,'command':cmd})
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    return result


def _snapshot_original(root):
    directory=root/RECOVERY/'evidence/original_final_snapshot'; receipt=directory/'snapshot_manifest.sealed.json'
    if receipt.exists():
        saved=c.read(receipt)
        for path,digest in saved['sealed_sha256'].items():
            if sha256_file(path)!=digest: raise ValueError('Original final snapshot changed')
        return saved
    original=c.read(root/FINAL/'episode_results.status.json')
    if original['status']!='incomplete' or original['valid'] is not False:
        raise ValueError('Expected original incomplete derived table before first merge')
    saved={'original_status':original,'sealed_sha256':{}}
    for name, destination in (('episode_results.csv','episode_results.sealed.csv'),
                              ('slots.csv','slots.sealed.csv'),
                              ('episode_results.status.json','episode_status.sealed.json')):
        target=directory/destination;saved['sealed_sha256'][str(target)]=_copy_once(root/FINAL/name,target)
    _once(receipt,saved);return saved


def _validate_final_tables(root, frame, slots, frozen):
    from . import pipeline
    # Actual runs use the canonical root; alternate roots support isolated tests.
    previous=pipeline.OUT
    try:
        pipeline.OUT=root
        pipeline._validate_final_identities(frame,slots,frozen['selected'])
    finally:
        pipeline.OUT=previous


def finalize(root=c.OUT):
    root=Path(root).resolve(); directory=root/RECOVERY; plan=c.read(directory/'recovery_plan.json')
    original,terminal=_verify_plan(root,plan); frozen=_freeze(root)
    if _pipeline_processes():
        raise ValueError('Original pipeline is still alive; final publication is forbidden')
    source_status=c.read(root/ORIGINAL/'episode_results.status.json')
    if source_status['status'] not in ('completed','completed_with_failures'):
        raise ValueError('Original stage must reach a terminal status')
    planned={r['original_task_index'] for r in plan['mapping']}
    if set(terminal)!=planned:
        raise ValueError('Original final missing set differs from fixed recovery scope')
    recovery_status=c.read(directory/'episode_results.status.json'); receipt=c.read(directory/'recovery_receipt.json')
    if (recovery_status['status']!='completed' or recovery_status['valid'] is not True
            or receipt['status']!='completed' or recovery_status['completed_episodes']!=len(planned)
            or receipt['plan_sha256']!=sha256_file(directory/'recovery_plan.json')
            or c.read(directory/'failures.json') or c.read(directory/'skipped.json')):
        raise ValueError('Recovery is not fully complete')
    rp=c.read(directory/'task_protocol.json')
    if rp['tasks']!=plan['tasks'] or rp['identities']!=plan['identities'] or rp['code_sha256']!=plan['runtime_code_sha256']:
        raise ValueError('Recovery task protocol differs')
    for name,field in (('episode_results.csv','result_sha256'),('slots.csv','slots_sha256')):
        if sha256_file(directory/name)!=receipt[field]:raise ValueError('Recovery CSV changed')
    manifest_path=root/FINAL/'recovery_manifest.json'
    if manifest_path.exists():
        saved=c.read(manifest_path);current=c.read(root/FINAL/'episode_results.status.json')
        if current.get('status')=='completed':
            if (saved['plan_sha256']!=sha256_file(directory/'recovery_plan.json')
                    or current.get('recovery_manifest_sha256')!=sha256_file(manifest_path)
                    or saved['receipt_sha256']!=sha256_file(directory/'recovery_receipt.json')
                    or any(sha256_file(path)!=digest for path,digest in saved['original_snapshot']['sealed_sha256'].items())
                    or any(sha256_file(root/FINAL/name)!=digest for name,digest in saved['final_export_sha256'].items())):
                raise ValueError('Published recovery result changed')
            return current
    original_ids={int(p.stem.split('_')[1]) for p in (root/ORIGINAL/'results').glob('task_*.json')}
    expected=set(range(len(original['tasks'])))
    if original_ids & planned or original_ids | planned != expected:
        raise ValueError('Original terminal accounting does not exactly partition expected tasks')
    if (source_status['completed_episodes']!=len(original_ids) or source_status['expected_episodes']!=len(expected)
            or source_status['failed_episodes']!=plan['historical_failed_episodes']
            or source_status['skipped_episodes']!=plan['historical_skipped_episodes']):
        raise ValueError('Original terminal status counts differ')
    recovery_ids={int(p.stem.split('_')[1]) for p in (directory/'results').glob('task_*.json')}
    if recovery_ids!=set(range(len(planned))):raise ValueError('Recovery durable result coverage differs')
    snapshots=_snapshot_original(root)
    rows,slots=[],[]
    for old in sorted(original_ids):
        payload=c.read(root/ORIGINAL/'results'/f'task_{old:05d}.json')
        _validate(payload,original['tasks'][old],original['identities'][old],old)
        row=deepcopy(payload['row']);row.update(phase=FINAL,original_task_index=old)
        rows.append(row)
        for value in payload['slots']:
            item=deepcopy(value);item.update(phase=FINAL,original_task_index=old);slots.append(item)
    for mapping in plan['mapping']:
        old,local=mapping['original_task_index'],mapping['recovery_task_index']
        payload=c.read(directory/'results'/f'task_{local:05d}.json')
        _validate(payload,plan['tasks'][local],plan['identities'][local],local)
        common={'phase':FINAL,'task_index':old,'original_task_index':old,'recovery_task_index':local,
                'recovery_stage':RECOVERY,'provenance':'resource_recovery'}
        row=deepcopy(payload['row']);row.update(common);rows.append(row)
        for value in payload['slots']:
            item=deepcopy(value);item.update(common);slots.append(item)
    frame=pd.DataFrame(rows);slot_frame=pd.DataFrame(slots)
    keys=['algorithm','training_seed','day_id','scenario','player_id','seed']
    protocol=c.read(root/'protocol.json')
    wanted={(s['algorithm'],int(s.get('training_seed',0)),d,scene,p,seed)
            for s in frozen['selected'] for d in frozen['independent_days'] for scene in protocol['scenarios']
            for p in protocol['roles'] for seed in protocol['evaluation_seeds']}
    observed=set(frame[keys].itertuples(index=False,name=None))
    if observed!=wanted or len(frame)!=len(wanted) or set(frame.task_index)!=expected or frame.task_index.duplicated().any():
        raise ValueError('Merged final Cartesian coverage differs')
    _validate_final_tables(root,frame,slot_frame,frozen)
    _verify_plan(root,plan)
    if _pipeline_processes():raise ValueError('A pipeline started before publication')
    export=directory/'finalized_exports';export.mkdir(exist_ok=True)
    for name,data in (('episode_results.csv',frame.sort_values(keys)),('slots.csv',slot_frame.sort_values(['task_index','time_slot']))):
        pending=export/(name+'.pending');data.to_csv(pending,index=False);pending.replace(export/name)
    hashes={name:sha256_file(export/name) for name in ('episode_results.csv','slots.csv')}
    manifest={'schema_version':1,'plan_path':str(directory/'recovery_plan.json'),
              'plan_sha256':sha256_file(directory/'recovery_plan.json'),'receipt_path':str(directory/'recovery_receipt.json'),
              'receipt_sha256':sha256_file(directory/'recovery_receipt.json'),
              'original_stage':ORIGINAL,'recovery_stage':RECOVERY,'original_completed_episodes':len(original_ids),
              'historical_failed_episodes':plan['historical_failed_episodes'],'historical_skipped_episodes':plan['historical_skipped_episodes'],
              'recovery_completed_episodes':len(planned),'unresolved_episodes':0,'mapping':plan['mapping'],
              'original_snapshot':snapshots,'original_failure_path':str(root/ORIGINAL/'failures.json'),
              'original_skipped_path':str(root/ORIGINAL/'skipped.json'),'final_export_sha256':hashes,
              'runtime_accounting':{'original_runtime_segments':[str(p) for p in sorted((root/ORIGINAL).glob('runtime_segment_*.json'))],
                                    'recovery_runtime_segments':[str(p) for p in sorted(directory.glob('runtime_segment_*.json'))],
                                    'policy':'Retain all measured seconds, including standby; snapshot/archive files are not extra execution segments.'},
              'successful_standby_overlap_cases_preserved':True}
    for name in hashes:
        pending=root/FINAL/(name+'.recovery.pending');shutil.copyfile(export/name,pending);pending.replace(root/FINAL/name)
    c.dump(manifest_path,manifest)
    result={'status':'completed','valid':True,'phase':FINAL,'completed_episodes':len(frame),'expected_episodes':len(wanted),
            'new_executed_episodes':len(frame),'reused_episodes':0,'missing_episodes':0,'missing_cases':[],
            'failed_episodes':0,'skipped_episodes':0,'unresolved_episodes':0,
            'historical_failed_episodes':plan['historical_failed_episodes'],'historical_skipped_episodes':plan['historical_skipped_episodes'],
            'original_completed_episodes':len(original_ids),'recovery_completed_episodes':len(planned),
            'recovery_manifest_path':str(manifest_path),'recovery_manifest_sha256':sha256_file(manifest_path),
            'result_sha256':hashes['episode_results.csv'],'slots_sha256':hashes['slots.csv']}
    c.dump(root/FINAL/'episode_results.status.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('plan','run','finalize'))
    parser.add_argument('--output',type=Path,default=c.OUT)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    result=create_plan(args.output) if args.stage=='plan' else run_recovery(args.output,args.workers) if args.stage=='run' else finalize(args.output)
    import json
    print(json.dumps({'stage':args.stage,'status':result.get('status','planned'),
                      'episodes':result.get('completed_episodes',result.get('expected_recovery_episodes'))},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
