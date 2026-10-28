"""Compare fresh-process and persistent-process complete training episodes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harmony_baselines import experiment as exp
from harmony_baselines import supervised


def main():
    original=exp.OUT
    opponents=exp.read(original/'opponents_frozen.json')
    pid=next(c for c in exp.read(original/'classical_selected.json') if c['algorithm']=='LocalPID')
    all_tasks=exp.tasks_for([pid],opponents,exp.TRAIN_DAYS[:1],['0','8','16'],seeds=exp.SEEDS[:1])
    tasks=[all_tasks[i] for i in [0,4,8]]
    frames=[]
    for label,persistent in [('fresh',False),('persistent',True)]:
        exp.OUT=original/'pool_verification'/label
        frame,failed,skipped=supervised.execute(tasks,'validation',workers=1,timeout_seconds=180,persistent=persistent)
        assert not failed and not skipped
        frames.append(frame)
    business=['algorithm','day_id','scenario','player_id','seed','training_seed','budget_scale','initial_budget',
              'total_value','total_cost','budget_utilization','wins','dataset_fingerprint','rules_version',
              'opponent_fingerprint','assignment_fingerprint','model_fingerprint']
    assert frames[0][business].equals(frames[1][business])
    exp.dump(original/'persistent_worker_verification.json',{'status':'passed','episodes_per_mode':3,
        'comparison':'exact equality of business outputs, IDs, seeds and policy/data fingerprints',
        'scenarios':['Fixed','PID','Mixed'],'periods':exp.TRAIN_DAYS[:1],'test_used':False,
        'fresh_agents_and_reset_every_episode':True,'only_readonly_market_and_weights_cached':True})
    print('Fresh and persistent full-episode business results are exactly equal',flush=True)


if __name__=='__main__':main()
