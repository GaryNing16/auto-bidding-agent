"""Native conversion only; no network, extraction, deletion of source or RL."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmony_bidding.auctionnet_data import aggregate_manifest, convert_period


def _periods(value):
    result = set()
    for part in value.split(","):
        if "-" in part:
            first, last = map(int, part.split("-"))
            if first > last:
                raise argparse.ArgumentTypeError("period range is reversed")
            result.update(range(first, last + 1))
        else:
            result.add(int(part))
    if not result or any(not 7 <= period <= 27 for period in result):
        raise argparse.ArgumentTypeError("periods must be in 7..27")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stream native AuctionNet periods to six fields with explicit local 48-to-72 mapping")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--periods", type=_periods, help="e.g. 7,8,13 or 7-27")
    parser.add_argument("--priority-periods", type=_periods, help="Convert these selected periods first, e.g. 7,21,24; output semantics unchanged")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-requests-per-slot", type=int, default=4)
    parser.add_argument("--expected-players", type=int, default=48)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in 1..8")
    sources = {}
    for path in Path(args.source_dir).rglob("period-*.csv"):
        match = re.fullmatch(r"period-([0-9]+)\.csv", path.name)
        if match and (args.periods is None or int(match[1]) in args.periods):
            period = int(match[1])
            if period in sources:
                raise ValueError(f"duplicate native source for period {period}")
            sources[period] = path
    if not sources or args.periods is not None and set(sources) != args.periods:
        raise ValueError("no sources or requested periods missing")
    priority = args.priority_periods or set()
    if not priority.issubset(sources):
        raise ValueError("priority periods must be among the selected sources")
    failures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {period: pool.submit(convert_period, path, args.output_dir,
                    expected_players=args.expected_players, sample_requests_per_slot=args.sample_requests_per_slot,
                    resume=not args.no_resume) for period, path in sorted(sources.items(), key=lambda item: (item[0] not in priority, item[0]))}
        by_future = {future: period for period, future in futures.items()}
        for future in as_completed(by_future):
            period = by_future[future]
            try:
                result = future.result()
            except Exception as exc:
                failures[period] = {"type": type(exc).__name__, "message": str(exc)}
                print(json.dumps({"period": period, "status": "failed", **failures[period]}), file=sys.stderr, flush=True)
                continue
            print(json.dumps({"period": period, "rows": result["row_count"], "requests": result["request_count"],
                              "sample_requests": result["sample_count"]}), flush=True)
    if failures:
        raise RuntimeError(f"conversion failed for periods {sorted(failures)}; successful periods retained, see preceding diagnostics")
    manifest = aggregate_manifest(args.output_dir)
    print(json.dumps({"manifest": str(Path(args.output_dir) / "manifest.json"), "periods": len(manifest["periods"]),
                      "row_count": manifest["row_count"], "request_count": manifest["request_count"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
