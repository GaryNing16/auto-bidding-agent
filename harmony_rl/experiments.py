"""One bounded E1-E6 run with explicit shared interaction and evaluation budgets."""
import argparse
import json
from pathlib import Path
import time

from harmony_baselines.five_scenarios.common import dump, read, immutable
from .config import TrainConfig, VARIANTS
from .train import train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[20261028])
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--training-max-seconds", type=float, default=1800)
    parser.add_argument("--evaluation-max-seconds", type=float, default=3600)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args(argv)
    if args.max_seconds <= 0:
        parser.error("finite positive --max-seconds required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(variants=args.variants, training_seeds=args.training_seeds,
        episodes_per_run=args.episodes, train_days=["period-7", "period-8"],
        validation_days=["period-21"], test_days=["period-24"],
        evaluation_seeds=[20261028, 20261029], evaluation_roles=list(map(str, range(48))),
        checkpoint_selection="validation mean raw V; same validation grid and cadence for each variant",
        status_scope="finite-budget pilot; no convergence or multi-training-seed stability claim",
        baseline_reuse="sealed five-scenario experiment B", test_retuning=False)
    immutable(output / "experiment_protocol.json", protocol)
    started = time.perf_counter()
    summaries, checkpoints = [], []
    for variant in args.variants:
        for seed in args.training_seeds:
            remaining = args.max_seconds - (time.perf_counter()-started)
            if remaining <= 0:
                break
            run = output / "training" / f"{variant}_seed{seed}"
            cfg = TrainConfig(variant=variant, seed=seed, episodes=args.episodes,
                workers=args.workers, max_seconds=min(args.training_max_seconds, remaining))
            existing = read(run / "training_status.json") if (run / "training_status.json").exists() else None
            if existing and existing["complete"] and existing["validated"]:
                # train(resume) verifies all input/source seals even for a completed run.
                result = train(cfg, run, resume=True)
            else:
                result = train(cfg, run, resume=(run / "last.pt").exists())
            summaries.append(result)
            dump(output / "training_summary.json", summaries)
            if result["complete"] and result["validated"]:
                checkpoints.append(result["checkpoint"])
            print(json.dumps({"run_finished": result}, ensure_ascii=False), flush=True)
    if not args.train_only and checkpoints:
        from .evaluate import checkpoint_candidates, run_evaluation
        candidates = checkpoint_candidates(checkpoints)
        dump(output / "candidates.json", candidates)
        remaining = args.max_seconds-(time.perf_counter()-started)
        if remaining > 0:
            _, status = run_evaluation(candidates, output / "evaluation", workers=args.workers,
                max_seconds=min(args.evaluation_max_seconds, remaining))
            print(json.dumps({"evaluation_finished": status}, ensure_ascii=False), flush=True)
            from .report import generate_report
            generate_report(output / "evaluation", training_root=output / "training")
    dump(output / "run_status.json", {"seconds": time.perf_counter()-started,
        "trained_candidates": len(checkpoints), "requested_candidates": len(args.variants)*len(args.training_seeds),
        "training_summary": summaries})


if __name__ == "__main__":
    main()
