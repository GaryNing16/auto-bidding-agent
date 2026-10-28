"""Audit a reporting-only source amendment before any test data is requested."""
from copy import deepcopy
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harmony_baselines import experiment as exp
from harmony_baselines.data import DEFAULT_CACHE, sha256_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=exp.OUT)
    args=parser.parse_args()
    out=args.output.resolve()
    frozen_path=out/'selected_frozen.json'
    protocol_path=out/'protocol.json'
    before=exp.read(frozen_path)
    protocol=exp.read(protocol_path)
    assert protocol['status']=='frozen_before_test'
    assert protocol['selection_sha256']==sha256_file(frozen_path)
    absent=[out/'test_task_protocol.json',out/'episode_results.csv',out/'evaluation_failures.json']
    absent += [DEFAULT_CACHE/day for day in exp.TEST_DAYS]
    assert not any(p.exists() for p in absent),'test data/results already requested'
    changed={p:(h,sha256_file(ROOT/p)) for p,h in before['code_sha256'].items()
             if sha256_file(ROOT/p)!=h}
    assert {Path(p).as_posix() for p in changed}=={'harmony_baselines/report.py'},changed
    report_key=next(iter(changed))
    snapshot=out/'selected_frozen_before_report_amendment.json'
    audit_path=out/'pretest_report_amendment.json'
    assert not snapshot.exists() and not audit_path.exists()
    snapshot.write_bytes(frozen_path.read_bytes())
    old_sha=sha256_file(snapshot)
    after=deepcopy(before)
    after['code_sha256'][report_key]=changed[report_key][1]
    for key in before:
        if key!='code_sha256':assert after[key]==before[key]
    exp.dump(frozen_path,after)
    protocol['selection_sha256']=sha256_file(frozen_path)
    exp.dump(protocol_path,protocol)
    exp.dump(audit_path,{'reason':'final report adds actual CQL action coordinates and transparent extra validation-only search budget',
        'scope':'pre-test reporting-only source hash amendment; selection and model assets unchanged',
        'old_selection_sha256':old_sha,'new_selection_sha256':protocol['selection_sha256'],
        'changed_code_sha256':changed,'test_used':False,'absent_test_artifacts':[str(p) for p in absent],
        'selected_unchanged':after['selected']==before['selected'],
        'checkpoint_hashes_unchanged':after['checkpoints']==before['checkpoints'],
        'opponents_hash_unchanged':after['opponents_sha256']==before['opponents_sha256'],
        'lp_table_hash_unchanged':after['lp_table_sha256']==before['lp_table_sha256']})
    print('Pre-test reporting-only amendment sealed; selected policies unchanged.',flush=True)


if __name__=='__main__':main()
