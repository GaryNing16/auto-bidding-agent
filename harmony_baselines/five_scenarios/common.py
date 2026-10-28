"""Paths, durable records and identity for the five-scenario experiment."""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path
import time

from harmony_baselines.data import sha256_file
from harmony_bidding.schema import fingerprint

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'outputs' / 'baseline_comparison_20261028'
OUT = ROOT / 'outputs' / 'baseline_five_scenarios_20261028'
SCENARIOS = ('Fixed', 'LocalPID', 'Mixed', 'AuctionNetPID', 'Mixed2')
OLD_SCENARIOS = SCENARIOS[:3]
TRAIN_DAYS = ('period-7', 'period-8')
VALID_DAYS = ('period-21',)
COMPARISON_DAYS = ('period-24',)
INDEPENDENT_DAYS = ('period-25',)
SEEDS = (20261028, 20261029)
ROLES = tuple(str(i) for i in range(48))
VALID_ROLES = tuple(str(i) for i in range(0, 48, 8))
NEURAL = ('BC', 'IQL', 'BCQ', 'TD3BC', 'CQL', 'DT')
ALGORITHMS = ('FixedAlpha', 'LocalPID', 'OnlineLP', 'AuctionNetPID') + NEURAL


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    """Atomic replace with bounded retry for Windows readers; no truncation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05 * (attempt + 1))


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Immutable record differs: {path}')
    else:
        dump(path, value)


def assets(spec):
    result = {}
    kwargs = spec.get('kwargs', {})
    for field in ('checkpoint', 'table_path'):
        if field in kwargs:
            path = Path(kwargs[field]).resolve()
            result[str(path)] = sha256_file(path)
    if 'base_spec' in kwargs:
        result.update(assets(kwargs['base_spec']))
    return result


def policy_id(spec):
    return fingerprint({'spec': spec, 'assets': assets(spec)})


def runtime_code_hashes():
    old = read(OLD / 'selected_frozen.json')['code_sha256']
    result = {p: sha256_file(ROOT / p) for p in old}
    for name in ('__init__.py', 'common.py', 'execution.py', 'evaluator.py', 'auctionnet_pid.py'):
        path = Path(__file__).parent / name
        result[str(path.relative_to(ROOT))] = sha256_file(path)
    return result


def cache_root(day_id):
    old = OLD / 'cache'
    if (old / day_id / 'cache.json').exists():
        return old
    return OUT / 'cache'


@lru_cache(maxsize=6)
def cached_day(day_id):
    from harmony_baselines.data import load_day
    return load_day(day_id, cache_root=cache_root(day_id))


def pid_spec(**kwargs):
    return {'kind': 'custom', 'class_path': 'harmony_baselines.five_scenarios.auctionnet_pid:AuctionNetPIDAgent',
            'kwargs': kwargs}
