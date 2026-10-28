"""Pre-test validation-only extension when a classical lower boundary wins."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from . import experiment


def refine_online_lp(workers=4, maximum_rounds=3):
    out=experiment.OUT
    if (out/'selected_frozen.json').exists() or (out/'episode_results.csv').exists():
        raise ValueError('Classical refinement must precede neural freeze and final test')
    search=experiment.read(out/'classical_search.json')
    if (out/'classical_refinement.json').exists():
        raise ValueError('Refinement already attempted; preserve its outputs')
    selected=experiment.read(out/'classical_selected.json')
    experiment.dump(out/'classical_selected_initial.json',selected)
    base_frame=pd.read_csv(out/'classical_validation.csv',dtype={'player_id':str})
    reference=base_frame[base_frame.algorithm=='LocalPID_1'].set_index(['scenario','player_id']).total_value
    def score(frame):
        frame=frame.copy()
        frame['relative']=[r.total_value/reference.loc[(r.scenario,str(r.player_id))]
            if reference.loc[(r.scenario,str(r.player_id))]>0 else np.nan for r in frame.itertuples()]
        return frame.groupby('algorithm').relative.mean()
    all_frames=[base_frame]
    all_candidates=list(search['candidates'])
    low=min(search['multipliers'])
    rounds=[]
    opponents=experiment.read(out/'opponents_frozen.json')
    lp=next(c for c in selected if c['algorithm']=='OnlineLP')
    for round_index in range(1,maximum_rounds+1):
        current=lp['spec']['kwargs']['multiplier']
        if current != low: break
        multipliers=[low/4,low/2]
        candidates=[{'algorithm':f'OnlineLP_low_{round_index}_{i}',
            'spec':{**lp['spec'],'kwargs':{**lp['spec']['kwargs'],'multiplier':m}}}
            for i,m in enumerate(multipliers)]
        frame,_,_=experiment.execute(experiment.tasks_for(candidates,opponents,
            experiment.VALID_DAYS,experiment.VALID_ROLES),
            out/f'classical_validation_lower_round_{round_index}.csv',workers=workers)
        all_frames.append(frame)
        all_candidates.extend(candidates)
        scores=score(pd.concat(all_frames,ignore_index=True))
        lp_scores=scores[scores.index.str.startswith('OnlineLP_')]
        best=lp_scores.idxmax()
        candidate=next(c for c in all_candidates if c['algorithm']==best)
        lp={'algorithm':'OnlineLP','spec':candidate['spec'],'validation_choice':best,
            'validation_relative':float(lp_scores[best])}
        rounds.append({'round':round_index,'multipliers':multipliers,'selected':best,
                       'scores':{k:float(v) for k,v in lp_scores.items()}})
        low=min(multipliers)
    selected=[lp if c['algorithm']=='OnlineLP' else c for c in selected]
    experiment.dump(out/'classical_selected.json',selected)
    search['candidates']=all_candidates
    search['selected']=selected
    search['algorithm_multipliers']={name:sorted({c['spec']['kwargs']['multiplier'] for c in all_candidates
        if c['algorithm'].startswith(name+'_')}) for name in ['FixedAlpha','LocalPID','OnlineLP']}
    search['boundary_selected']=[{'algorithm':c['algorithm'],
        'side':'lower' if c['spec']['kwargs']['multiplier']==min(search['algorithm_multipliers'][c['algorithm']]) else 'upper'}
        for c in selected if c['spec']['kwargs']['multiplier'] in
            [min(search['algorithm_multipliers'][c['algorithm']]),max(search['algorithm_multipliers'][c['algorithm']])]]
    search['refinement']='validation-only lower-bound extension for OnlineLP; common initial grid remains in multipliers'
    experiment.dump(out/'classical_search.json',search)
    result={'status':'completed','rounds':rounds,'selected':lp,'test_used':False,
            'remaining_boundary':search['boundary_selected']}
    experiment.dump(out/'classical_refinement.json',result)
    print(result,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output', type=Path, help='Experiment output directory')
    args=parser.parse_args()
    if args.output is not None:
        experiment.OUT = args.output.resolve()
    refine_online_lp(args.workers)
