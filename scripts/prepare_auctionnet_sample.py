"""Build a bounded, source-verified quickstart from completed native conversion.

Run from the repository root. This never substitutes samples for full conversion.
All request values, time slots and within-slot order come from the full conversion.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import yaml


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_identical_or_new(path: Path, content: str) -> None:
    encoded = content.encode('utf-8')
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f'refusing to replace different file: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(encoded)


def prepare(root: Path, config_path: Path, periods: list[int]) -> dict:
    if not periods or len(set(periods)) != len(periods):
        raise ValueError('periods must be nonempty and unique')
    metadata = []
    sources = []
    order_lines = ['day_id,time_step,request_id,original_order\n']
    player_types = None
    train_wins: dict[str, int] = {}
    for period in periods:
        directory = root / f'period-{period}'
        meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        expected_versions = {'converter_version': 'auctionnet-six-fields-v1',
                             'rules_version': 'harmony-A-B-local-v1',
                             'assumption_profile': 'local-auctionnet-48-to-72-v1'}
        if any(meta.get(key) != value for key, value in expected_versions.items()):
            raise ValueError('unsupported converter/rules/assumption version in metadata')
        sample_limit = meta.get('parameters', {}).get('sample_requests_per_slot')
        if type(sample_limit) is not int or sample_limit < 1:
            raise ValueError('metadata must declare a positive sample request limit')
        if meta['source_period'] != period or meta['day_id'] != f'period-{period}':
            raise ValueError('metadata period identity mismatch')
        types = {pid: spec['player_type'] for pid, spec in meta['players'].items()}
        if player_types is not None and types != player_types:
            raise ValueError('players/types differ across periods')
        player_types = types
        for name in ('sample', 'sample_request_order'):
            record = meta['files'][name]
            file_path = directory / record['path']
            if not file_path.resolve().is_relative_to(directory.resolve()):
                raise ValueError('sample file escapes period directory')
            if file_hash(file_path) != record['sha256']:
                raise ValueError(f'sample checksum mismatch: {file_path}')
        sample = directory / meta['files']['sample']['path']
        sources.append({'path': sample.as_posix(), 'day_id': meta['day_id']})
        with (directory / meta['files']['sample_request_order']['path']).open(encoding='utf-8', newline='') as handle:
            if handle.readline().strip() != 'day_id,time_step,request_id,original_order':
                raise ValueError('unexpected sample order header')
            order_lines.extend(line.rstrip('\r\n') + '\n' for line in handle)
        # Role choice is a reproducible local demo setting fitted on train only.
        # It does not use the validation/local_holdout outcomes or reach actors.
        if meta['split'] == 'train':
            previous = None
            winner = None
            best = -1.0
            with sample.open(encoding='utf-8', newline='') as handle:
                for row in csv.DictReader(handle):
                    if row['request_id'] != previous:
                        if winner is not None:
                            train_wins[winner] = train_wins.get(winner, 0) + 1
                        previous, winner, best = row['request_id'], None, -1.0
                    value = float(row['pvalue'])
                    if value > best:
                        winner, best = row['player_id'], value
                if winner is not None:
                    train_wins[winner] = train_wins.get(winner, 0) + 1
        metadata.append(meta)
    if not train_wins:
        raise ValueError('include a nonempty train period to choose demo roles')
    roles = sorted(player_types, key=lambda pid: (-train_wins.get(pid, 0), pid))[:3]
    order_path = root / 'sample_request_order.csv'
    config = {
        'rules': {'total_slots': 72, 'money_decimals': 6,
                  'zero_balance_can_bid': False, 'tie_seed': 20261018,
                  'rules_version': 'harmony-A-B-local-v2',
                  'assumption_profile': 'local-unified-v2'},
        'expected_n': len(player_types), 'agent_seed': 1234,
        'data_dir': (root / 'sample_internal').as_posix(),
        'output_dir': 'outputs/auctionnet_native_sample', 'split': 'local_holdout',
        'organizer_import': {
            'source_kind': 'auctionnet_native_simulation_bounded_sample',
            'sources': sources, 'time_step_base': 0, 'request_id_scope': 'day',
            'ordering': {'mode': 'explicit_csv', 'path': order_path.as_posix()},
            'players': metadata[0]['players'],
            'budgets_by_day': {m['day_id']: {pid: p['budget'] for pid, p in m['players'].items()}
                               for m in metadata},
            'split_by_day': {m['day_id']: m['split'] for m in metadata}},
        '_dataset_derivation_metadata': {
            'source_kind': 'auctionnet_native_simulation',
            'source_conversion_versions': {m['day_id']: {key: m[key] for key in
                ('converter_version', 'rules_version', 'assumption_profile')} for m in metadata},
            'value_semantics': 'unaltered source pValue (conversion probability), used as local value score',
            'sample_policy': 'first K requests per mapped slot, all advertisers; full-data mapping retained',
            'sample_requests_per_slot': {m['day_id']: m['parameters']['sample_requests_per_slot'] for m in metadata},
            'budget_policy': 'original source budgets retained; this small sample may be budget-loose',
            'role_policy': 'three highest raw-value winner counts in selected train sample only',
            'source_period_metadata_sha256': {m['day_id']: file_hash(root / m['day_id'] / 'metadata.json')
                                              for m in metadata},
            'mapping_assumptions': {m['day_id']: m['mapping_assumption'] for m in metadata}},
        'pid': {'kp': 1.5, 'ki': 0.1, 'kd': 0.15, 'initial_alpha': 1.0},
        'evaluation': {'roles': roles,
                       'teams': {name: {'kind': 'pid'} for name in ('A', 'B', 'C')[:len(roles)]},
                       'diagnostic_undefined': False, 'isolation': 'process'}}
    content = '# Generated native-data quickstart; local settings, not official parameters.\n'
    content += yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    write_identical_or_new(order_path, ''.join(order_lines))
    write_identical_or_new(config_path, content)
    return {'config': config_path.as_posix(), 'periods': periods, 'roles': roles,
            'requests': sum(m['sample_count'] for m in metadata),
            'rows': sum(m['sample_row_count'] for m in metadata)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/auctionnet_72'))
    parser.add_argument('--config', type=Path, default=Path('configs/auctionnet_native_sample.yaml'))
    parser.add_argument('--periods', type=int, nargs='+', default=[7, 21, 24])
    args = parser.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(prepare(args.root, args.config, args.periods), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
