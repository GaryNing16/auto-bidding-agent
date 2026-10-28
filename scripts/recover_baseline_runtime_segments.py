"""Recover only observed, disjoint execution segments from immutable run logs."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harmony_baselines import experiment as exp
from harmony_baselines.data import sha256_file


def records(path):
    result=[]
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if value.get('stage')=='validation' and 'seconds' in value:result.append(value)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=exp.OUT)
    out=parser.parse_args().output.resolve()
    definitions=[('supervised_select.log',882,'validation_interrupted_recovery.status.json',576,714),
                 ('supervised_select_persistent.log',882,'validation_persistent_original_tasks.status.json',714,846)]
    recovered=[]
    for log_name,expected,output_name,start,end in definitions:
        log=out/log_name
        rows=[r for r in records(log) if r['expected']==expected]
        last=max(rows,key=lambda r:r['seconds'])
        assert last['successful']==end
        item={'status':'archived_execution_segment','result_artifact':False,
              'seconds':last['seconds'],'seconds_precision':'last logged elapsed time, rounded to 0.1 second; excludes unobserved tail',
              'source_log':log_name,'source_log_sha256':sha256_file(log),
              'resumed_successful_episodes':start,'completed_successful_episodes':end,
              'note':'disjoint prior segment; current neural_validation.status.json describes only the final 72-case extension'}
        target=out/output_name
        if target.exists():assert exp.read(target)==item
        else:exp.dump(target,item)
        recovered.append(output_name)
    exp.dump(out/'runtime_accounting.json',{
        'meaning':'sum of recorded evaluation execution segments, not end-to-end project elapsed time',
        'coverage_complete':False,
        'unavailable_segments':[{'stage':'initial experiment select','successful_episodes':576,
            'reason':'original stage timer was overwritten during supervised recovery; original stdout/timing backup not found',
            'seconds':None,'includes_unmeasured_stalled_CQL_time':True}],
        'recovered_segment_status_files':recovered,
        'test_interruption_segment':'test_interruption_1.status.json',
        'final_test_segment':'episode_results.status.json',
        'final_validation_extension_segment':'neural_validation.status.json',
        'no_double_counting':'each archived status adds only a prior segment; final statuses retain new segment timers',
        'excluded':'code edits, investigative waits and tests are not full experiment execution timers',
        'report_wording':'有记录的评测执行段合计；最初576局验证及早期卡住耗时未完整保存，不能解释为端到端总耗时。'})
    print(json.dumps({'recovered_status_files':recovered,'initial_select_seconds':None},ensure_ascii=False))


if __name__=='__main__':main()
