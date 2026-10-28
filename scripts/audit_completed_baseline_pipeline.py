"""Validate the completed frozen experiment, including every 72-slot trace."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from harmony_baselines import experiment as exp, supervised
from harmony_baselines.data import sha256_file
from harmony_bidding.schema import fingerprint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=exp.OUT)
    out=parser.parse_args().output.resolve()
    protocol=exp.read(out/'protocol.json')
    status=exp.read(out/'episode_results.status.json')
    assert protocol['status']==status['status']
    assert status['status'] in ('completed','completed_with_failures'),'cannot audit a running experiment'
    frozen=exp.read(out/'selected_frozen.json')
    assert protocol['selection_sha256']==sha256_file(out/'selected_frozen.json')
    for relative,digest in frozen['code_sha256'].items():assert sha256_file(ROOT/relative)==digest,relative
    for path,digest in frozen['checkpoints'].items():assert sha256_file(path)==digest,path
    for name,key in [('opponents_frozen.json','opponents_sha256'),('online_lp_table.csv','lp_table_sha256')]:
        assert sha256_file(out/name)==frozen[key]
    tasks=exp.tasks_for(frozen['selected'],exp.read(out/'opponents_frozen.json'),
        protocol['executed_split']['test'],protocol['test_roles'],protocol['evaluation_seeds'])
    manifest=exp.read(out/'test_task_protocol.json')
    assert tasks==manifest['tasks']
    assert fingerprint(tasks)==manifest['tasks_sha256']==status['tasks_sha256']
    assert len(tasks)==status['expected_episodes']==protocol['test_expected_episodes']
    csv_path=out/'episode_results.csv'
    assert sha256_file(csv_path)==status['csv_sha256']
    frame=pd.read_csv(csv_path,dtype={'player_id':str,'player_type':str,'day_id':str})
    assert len(frame)==status['episodes']==protocol['test_episodes']
    supervised._validate_existing(frame,tasks)
    advertisers={day:{a.advertiser_id:a for a in exp.cached_day(day).advertisers}
                 for day in protocol['executed_split']['test']}
    for row in frame.itertuples():
        task=tasks[int(row.task_index)]
        advertiser=advertisers[row.day_id][row.player_id]
        assert str(row.player_type)==str(advertiser.advertiser_type)
        assert row.budget_scale==task.get('budget_scale',1.)
        assert abs(row.initial_budget-round(advertiser.budget*row.budget_scale,6))<=1e-9
    preserved=out/'episode_results_before_io_resume.csv'
    resumed_rows=0
    if preserved.exists():
        interruption=exp.read(out/'test_interruption_1.status.json')
        assert sha256_file(preserved)==interruption['csv_sha256']
        previous=pd.read_csv(preserved,dtype={'player_id':str,'player_type':str,'day_id':str}).set_index('task_index').sort_index()
        current=frame.set_index('task_index').loc[previous.index,previous.columns]
        pd.testing.assert_frame_equal(previous,current,check_exact=False,rtol=1e-12,atol=1e-10)
        resumed_rows=len(previous)
    failures=exp.read(out/'evaluation_failures.json')
    skipped=exp.read(out/'evaluation_skipped.json')
    assert len(failures)==status['failure_episodes']==protocol['test_failure_episodes']
    assert len(skipped)==status['skipped_episodes']==protocol['test_skipped_episodes']
    indices=list(frame.task_index.astype(int))
    for record in failures+skipped:
        index=int(record['task_index'])
        expected=supervised._case(tasks[index],index,'test')
        assert all(record[key]==value for key,value in expected.items())
        indices.append(index)
    assert len(indices)==len(set(indices))==len(tasks)
    assert set(indices)==set(range(len(tasks)))
    assert all(r['status'] in ('failed','resource_incomplete') for r in failures)
    assert all(r['status']=='not_run' for r in skipped)
    timing=frame[['bidding_p99_ms','update_p99_ms','paired_p99_ms','rss_mb','peak_process_rss_mb']].to_numpy()
    assert np.isfinite(timing).all() and (timing>=0).all()
    traces=0
    for row in frame.itertuples():
        path=out/'test_diagnostics'/f'task_{int(row.task_index)}.jsonl'
        trace=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        for phase in ['before_action','action','feedback']:
            assert [r['slot'] for r in trace if r['phase']==phase]==list(range(72)),str(path)
        actions=[r['alpha'] for r in trace if r['phase']=='action']
        assert np.isfinite(actions).all() and min(actions)>=0
        feedback=[r for r in trace if r['phase']=='feedback']
        assert np.isclose(sum(r['cost'] for r in feedback),row.total_cost,atol=1e-6,rtol=0)
        assert np.isclose(sum(r['value'] for r in feedback),row.total_value,atol=1e-8,rtol=1e-12)
        traces+=1
    counts={str(k):int(v) for k,v in frame.groupby('algorithm').size().items()}
    expected_counts={}
    for task in tasks:expected_counts[task['algorithm']]=expected_counts.get(task['algorithm'],0)+1
    complete=[name for name,count in expected_counts.items() if counts.get(name,0)==count]
    if not failures:
        assert counts==expected_counts and not skipped and status['status']=='completed'
    result={'status':'passed','experiment_status':status['status'],'expected_episodes':len(tasks),
        'successful_episodes':len(frame),'failure_episodes':len(failures),'skipped_episodes':len(skipped),
        'expected_counts_by_algorithm':expected_counts,'successful_counts_by_algorithm':counts,
        'complete_algorithms':complete,'full_72_slot_traces_verified':traces,
        'exact_task_index_coverage':True,'task_keys_match_frozen_protocol':True,
        'frozen_selection_source_models_opponents_lp_unchanged':True,
        'all_metric_and_action_values_finite':True,'trace_value_cost_sums_match_raw_csv':True,
        'resumed_successful_rows_semantically_unchanged':resumed_rows,
        'task_protocol_sha256':fingerprint(tasks),'episode_results_sha256':sha256_file(csv_path),
        'selection_sha256':sha256_file(out/'selected_frozen.json'),
        'timing_scope':'target original bidding/update only, diagnostic I/O excluded',
        'runtime_accounting':'see runtime_accounting.json for observed-segment timing limits'}
    exp.dump(out/'pipeline_completion_audit.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
