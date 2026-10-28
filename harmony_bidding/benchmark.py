"""Optional local benchmark. Agent call timers exclude auction and serialization."""
import math
import os
import platform
import statistics
import sys
import threading
from dataclasses import asdict
from importlib.metadata import version
from time import perf_counter
import psutil
from .runner import run_episode, write_json, code_fingerprint, UPSTREAM_COMMIT, RunRecord
from .schema import Rules, fingerprint


def percentile99(values):
    if not values:
        raise ValueError("P99 requires samples")
    return sorted(values)[math.ceil(0.99 * len(values)) - 1]


def timing_summary(rows, min_samples=1000):
    bid = [r["bidding_ms"] for r in rows]
    update = [r["update_ms"] for r in rows]
    paired = [r["bidding_ms"] + r["update_ms"] for r in rows]
    return {"samples_per_function": len(rows), "p99_bidding_ms": percentile99(bid),
            "p99_update_ms": percentile99(update), "p99_paired_sum_ms": percentile99(paired),
            "sum_of_p99_ms": percentile99(bid) + percentile99(update),
            "max_paired_sum_ms": max(paired), "mean_paired_sum_ms": statistics.mean(paired),
            "sufficient_for_local_diagnostic": len(rows) >= min_samples,
            "local_paired_under_200ms": percentile99(paired) <= 200,
            "local_sum_of_p99_under_200ms": percentile99(bid) + percentile99(update) <= 200}


def run_benchmark(day, config, output_dir=None, *, run_record=None):
    record = run_record or RunRecord(output_dir, "benchmark")
    try:
        record.bind(config)
        report = _run_benchmark(day, config, record)
        report.update(record.identity())
        if record.directory is not None:
            write_json(record.directory / "benchmark.json", report)
        record.complete({**record.identity(), "aggregate": report["aggregate"], "tested_strategy": report["tested_strategy"]})
        return report
    except Exception as exc:
        record.fail(exc)
        raise


def _run_benchmark(day, config, record):
    options = config.get("benchmark", {})
    warmup, episodes = options.get("warmup_episodes", 2), options.get("episodes", 20)
    minimum = options.get("min_samples", 1000)
    if type(warmup) is not int or warmup < 0 or type(episodes) is not int or episodes < 1:
        raise ValueError("invalid benchmark episode counts")
    if type(minimum) is not int or minimum < 1:
        raise ValueError("benchmark.min_samples must be positive integer")
    target = options.get("target_role")
    spec = options.get("agent")
    if target not in {a.advertiser_id for a in day.advertisers} or not isinstance(spec, dict):
        raise ValueError("benchmark requires explicit target_role and agent specification")
    sealed_assets = result_assets(config, spec)
    process = psutil.Process()
    measured_rss = [process.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.002):
            measured_rss.append(process.memory_info().rss)

    monitor = threading.Thread(target=sample, name="rss-sampler", daemon=True)
    monitor.start()
    rows, opponent_rows, initialization = [], [], []
    tested_strategy = None
    started = perf_counter()
    try:
        for repeat in range(warmup + episodes):
            if result_assets(config, spec) != sealed_assets:
                raise ValueError("sealed assets changed between benchmark episodes")
            result = run_episode(day, config, assignments={target: "benchmark_target"},
                                 team_specs={"benchmark_target": spec},
                                 episode_id=f"benchmark-{repeat}", mode="benchmark", run_id=record.run_id)
            tested_strategy = result["metadata"]["strategies"][target]
            if repeat >= warmup:
                rows.extend(r for r in result["timing"] if r["advertiser_id"] == target)
                opponent_rows.extend(r for r in result["timing"] if r["advertiser_id"] != target)
                initialization.append(result["initialization_ms"][target])
            del result
        if result_assets(config, spec) != sealed_assets:
            raise ValueError("sealed assets changed during benchmark")
    finally:
        stop.set()
        monitor.join()
    duration = perf_counter() - started
    memory_info = process.memory_info()
    rules = Rules(**config.get("rules", {}))
    report = {"rules_version": rules.rules_version, "assumption_profile": rules.assumption_profile,
        "local_assumptions": asdict(rules), "upstream_commit": UPSTREAM_COMMIT,
        "code_fingerprint": code_fingerprint(), "config_fingerprint": fingerprint(config),
        "dataset_fingerprint": day.fingerprint, "benchmark_only": True, "valid": True,
        "target_role": target, "tested_strategy": tested_strategy,
        "sealed_assets": sealed_assets,
        "official_strategy_available": False, "local_only": True,
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "logical_cpus": os.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                     "ram_bytes": psutil.virtual_memory().total},
        "runtime": {"executable": sys.executable, "python": sys.version,
                    "dependencies": {p: version(p) for p in ("PyYAML", "psutil", "pytest")},
                    "agent_execution_threads": 1, "rss_sampler_threads": 1,
                    "process_threads_at_end": process.num_threads()},
        "warmup_episodes": warmup, "measured_episodes": episodes, "wall_seconds": duration,
        "quantile_method": "nearest rank ceil(0.99*n), per function and paired same actor/slot",
        "timing_scope": "perf_counter_ns directly around bidding/update only; copies, auctions and IO excluded",
        "aggregate": timing_summary(rows, minimum),
        "by_advertiser": {target: timing_summary(rows, minimum)},
        "opponents_diagnostic": timing_summary(opponent_rows, minimum) if opponent_rows else None,
        "initialization_and_reset": {"samples": len(initialization), "p99_ms": percentile99(initialization),
                                     "max_ms": max(initialization), "included_in_call_times": False},
        "memory": {"scope": "process_total_rss", "sample_interval_ms": 2,
                   "sample_count": len(measured_rss), "sampled_peak_rss_bytes": max(measured_rss),
                   "os_lifetime_peak_wset_bytes": getattr(memory_info, "peak_wset", None),
                   "current_rss_bytes": memory_info.rss, "single_agent_rss_bytes": None,
                   "method": "psutil Process.memory_info; Windows peak_wset covers process lifetime",
                   "limitations": "Includes engine/data/logs/interpreter and warmup. Sampling may miss short peaks; single-agent memory not separable."},
        "interpretation": "Local CPU diagnostic only, not official P99/memory acceptance; official aggregation and load scope unknown."}
    if record.directory is not None:
        write_json(record.directory / "samples.json", rows)
    return report


def result_assets(config, spec):
    from .provenance import capture_assets
    return capture_assets(config, [spec])
