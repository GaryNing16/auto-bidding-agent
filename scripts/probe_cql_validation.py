"""Diagnostic ONLY: trace CQL on four requests/slot from validation, never test.

This is not a formal score and must never be merged into ranking CSVs.
"""
from pathlib import Path
import json
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harmony_baselines.neural import NeuralAgent
from harmony_baselines import experiment


def main():
    import torch
    torch.set_num_threads(1)
    out=experiment.OUT
    records=experiment.read(out/'models/candidates.json')
    record=next(r for r in records if r['algorithm']=='CQL' and r['config_index']==0
                and r['step']==200 and r['seed']==20261028)
    original=NeuralAgent.bidding
    def trace(self,observation):
        try:
            alpha=original(self,observation)
        except Exception as exc:
            print(json.dumps({'slot':observation.time_slot,'exception':repr(exc)}),flush=True)
            raise
        print(json.dumps({'slot':observation.time_slot,'alpha':alpha,
                         'remaining_budget':observation.remaining_budget}),flush=True)
        return alpha
    NeuralAgent.bidding=trace
    task=dict(day_id='period-21',target='0',scenario='Fixed',
        target_spec={'kind':'custom','class_path':'harmony_baselines.neural:NeuralAgent',
                     'kwargs':{'checkpoint':record['checkpoint']}},
        opponents=experiment.read(out/'opponents_frozen.json'),seed=20261028,smoke=True)
    print(json.dumps({'diagnostic_only':True,'test_used':False,'checkpoint':record['checkpoint']}),flush=True)
    try:
        experiment.worker(task)
    except Exception:
        traceback.print_exc()
        raise


if __name__=='__main__':
    main()
