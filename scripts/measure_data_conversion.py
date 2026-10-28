"""Run the native converter and record wall time plus process-tree RSS sampling.

This is data preprocessing telemetry, not an Agent performance benchmark.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-prefix', default='outputs/native_conversion')
    parser.add_argument('converter_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.converter_args
    if extra[:1] == ['--']:
        extra = extra[1:]
    prefix = Path(args.log_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-X', 'utf8', 'scripts/convert_auctionnet_data.py', *extra]
    begin = time.perf_counter()
    status = {'command': command, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'python': platform.python_version(), 'logical_cpu_count': os.cpu_count(),
              'physical_memory_bytes': psutil.virtual_memory().total,
              'rss_scope': 'sum of converter parent and worker RSS; shared pages may be counted more than once',
              'rss_sampling_seconds': 1, 'peak_process_tree_rss_bytes': 0,
              'is_agent_benchmark': False, 'samples': 0, 'status': 'running'}
    def save():
        path = Path(str(prefix) + '_measurement.json')
        pending = path.with_suffix('.json.part')
        pending.write_text(json.dumps(status, indent=2), encoding='utf-8')
        pending.replace(path)
    with Path(str(prefix) + '.log').open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0) as child:
            monitored = psutil.Process(child.pid)
            while child.poll() is None:
                total = 0
                try:
                    processes = [monitored, *monitored.children(recursive=True)]
                except psutil.NoSuchProcess:
                    processes = []
                for process in processes:
                    try:
                        total += process.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                status['peak_process_tree_rss_bytes'] = max(status['peak_process_tree_rss_bytes'], total)
                status['samples'] += 1
                status['elapsed_seconds'] = time.perf_counter() - begin
                if status['samples'] % 10 == 1:
                    save()
                time.sleep(1)
            status.update(exit_code=child.returncode, status='success' if child.returncode == 0 else 'failed',
                          elapsed_seconds=time.perf_counter() - begin)
            save()
    print(json.dumps(status, indent=2))
    return status['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
