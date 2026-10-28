"""Measure complete training-episode throughput before choosing pool size.

These training-period measurements are resource diagnostics, never ranking rows.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import psutil

from . import experiment


def benchmark(worker_counts=(1,4,12)):
    out = experiment.OUT
    opponents = experiment.read(out/'opponents_frozen.json')
    selected = experiment.read(out/'classical_selected.json')
    pid = next(c for c in selected if c['algorithm']=='LocalPID')
    tasks = experiment.tasks_for([pid], opponents, experiment.TRAIN_DAYS[:1], ['0','8','16','24'],
                                 seeds=experiment.SEEDS[:1])
    results=[]
    for workers in worker_counts:
        start=time.perf_counter()
        frame,_,_=experiment.execute(tasks,out/f'resource_benchmark_workers_{workers}.csv',workers=workers)
        wall=time.perf_counter()-start
        results.append({'workers':workers,'episodes':len(frame),'wall_seconds':wall,
            'episodes_per_second':len(frame)/wall,
            'mean_episode_seconds':float(frame.elapsed_seconds.mean()),
            'max_process_rss_mb':float(frame.rss_mb.max()),'test_used':False})
        experiment.dump(out/'resource_benchmark.json',{'status':'running','measurements':results})
    eligible=[r for r in results if r['max_process_rss_mb']<4096]
    if not eligible: raise RuntimeError('No tested worker count meets 4GB per-process diagnostic bound')
    selected_workers=max(eligible,key=lambda r:r['episodes_per_second'])['workers']
    report={'status':'completed','selected_workers':selected_workers,'measurements':results,
        'periods':experiment.TRAIN_DAYS[:1],'roles':['0','8','16','24'],'scenarios':list(experiment.SCENARIOS),
        'policy':pid,'test_used':False,'purpose':'resource throughput only; never algorithm ranking',
        'method':'same 12 full training episodes at each pool size; wall includes spawn and verified cache loading',
        'limitations':'small throughput sample includes startup overhead; rss sampled at episode completion, not a certified process peak',
        'logical_cpus':os.cpu_count(),'physical_memory_gb':psutil.virtual_memory().total/2**30}
    experiment.dump(out/'resource_benchmark.json',report)
    print(f'Selected workers: {selected_workers}',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',nargs='+',type=int,default=[1,4,12])
    parser.add_argument('--output', type=Path, help='Experiment output directory')
    args=parser.parse_args()
    if args.output is not None:
        experiment.OUT = args.output.resolve()
    benchmark(args.workers)
