"""Bounded read-only sampling of a running experiment and its worker processes."""
import argparse
import json
from pathlib import Path
import time
import psutil


def monitor(pid, output, max_seconds=7200, interval=5):
    parent = psutil.Process(pid)
    born = parent.create_time()
    started = time.monotonic()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    known = {}
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        while time.monotonic()-started < max_seconds:
            try:
                if parent.create_time() != born or not parent.is_running():
                    break
                children = [parent] + parent.children(recursive=True)
                live = []
                for process in children:
                    try:
                        tracked = known.setdefault(process.pid, process)
                        cpu = tracked.cpu_times()
                        live.append(dict(pid=tracked.pid, rss_mb=tracked.memory_info().rss/2**20,
                            cpu_percent=tracked.cpu_percent(), cpu_seconds=cpu.user+cpu.system))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                row = dict(elapsed_seconds=time.monotonic()-started,
                    process_count=len(live), total_rss_mb=sum(p["rss_mb"] for p in live),
                    summed_cpu_percent=sum(p["cpu_percent"] for p in live),
                    system_cpu_percent=psutil.cpu_percent(),
                    available_memory_mb=psutil.virtual_memory().available/2**20, processes=live)
                handle.write(json.dumps(row)+"\n")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(max(0., min(interval, max_seconds-(time.monotonic()-started))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=float, default=7200)
    args = parser.parse_args()
    pid = args.pid
    if pid is None:
        matches = [p.pid for p in psutil.process_iter(["cmdline"])
            if "harmony_rl.experiments" in (p.info["cmdline"] or [])]
        # Windows venv launchers and the real interpreter can share argv.
        roots = [p for p in matches if psutil.Process(p).ppid() not in matches]
        if len(roots) != 1:
            raise ValueError("Specify --pid when there is not exactly one experiment process")
        pid = roots[0]
    monitor(pid, args.output, args.max_seconds)


if __name__ == "__main__":
    main()
