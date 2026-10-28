import csv
from hashlib import sha256
import json

import pytest

from scripts.prepare_auctionnet_sample import prepare


def native_sample(root, period, split, values):
    folder = root / f'period-{period}'
    folder.mkdir(parents=True)
    sample = folder / 'sample.csv'
    sample.write_text('index,time_step,request_id,player_id,player_type,pvalue\n' +
                      ''.join(f'{i},0,p{period}:0,{i},0,{value}\n' for i, value in enumerate(values)), encoding='utf-8')
    order = folder / 'sample_request_order.csv'
    order.write_text(f'day_id,time_step,request_id,original_order\nperiod-{period},0,p{period}:0,0\n', encoding='utf-8')
    meta = {'day_id': f'period-{period}', 'source_period': period, 'split': split,
            'converter_version': 'auctionnet-six-fields-v1', 'rules_version': 'harmony-A-B-local-v1',
            'assumption_profile': 'local-auctionnet-48-to-72-v1', 'parameters': {'sample_requests_per_slot': 4},
            'players': {str(i): {'player_type': '0', 'budget': 100.0} for i in range(len(values))},
            'sample_count': 1, 'sample_row_count': len(values), 'mapping_assumption': 'test fixture only',
            'files': {key: {'path': p.name, 'sha256': sha256(p.read_bytes()).hexdigest()}
                      for key, p in [('sample', sample), ('sample_request_order', order)]}}
    (folder / 'metadata.json').write_text(json.dumps(meta), encoding='utf-8')
    return folder


def test_quickstart_uses_train_only_roles_and_preserves_source(tmp_path):
    root = tmp_path / 'data'
    native_sample(root, 7, 'train', [0.1, 0.2, 0.3, 0.9])
    native_sample(root, 24, 'local_holdout', [100, 0, 0, 0])
    config = tmp_path / 'sample.yaml'
    result = prepare(root, config, [7, 24])
    assert result['roles'] == ['3', '0', '1']
    assert result['rows'] == 8
    assert result == prepare(root, config, [7, 24])
    import yaml
    actual = yaml.safe_load(config.read_text(encoding='utf-8'))
    assert actual['expected_n'] == 4
    assert 'alpha_max' not in actual['rules']
    assert actual['rules']['rules_version'] == 'harmony-A-B-local-v2'
    assert actual['_dataset_derivation_metadata']['source_conversion_versions']['period-7']['rules_version'] == 'harmony-A-B-local-v1'
    assert actual['organizer_import']['players']['3']['budget'] == 100
    with (root / 'sample_request_order.csv').open() as handle:
        assert [r['day_id'] for r in csv.DictReader(handle)] == ['period-7', 'period-24']


def test_quickstart_checks_hash_and_does_not_overwrite(tmp_path):
    root = tmp_path / 'data'
    folder = native_sample(root, 7, 'train', [1, 2])
    config = tmp_path / 'sample.yaml'
    prepare(root, config, [7])
    config.write_text('user: changes\n', encoding='utf-8')
    with pytest.raises(FileExistsError):
        prepare(root, config, [7])
    (folder / 'sample.csv').write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        prepare(root, tmp_path / 'other.yaml', [7])


def test_quickstart_requires_explicit_train_and_stable_types(tmp_path):
    root = tmp_path / 'data'
    native_sample(root, 7, 'train', [1, 2])
    folder = native_sample(root, 24, 'local_holdout', [3, 4])
    with pytest.raises(ValueError, match='train'):
        prepare(root, tmp_path / 'config.yaml', [24])
    path = folder / 'metadata.json'
    meta = json.loads(path.read_text())
    meta['players']['0']['player_type'] = 'different'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='types'):
        prepare(root, tmp_path / 'config.yaml', [7, 24])
