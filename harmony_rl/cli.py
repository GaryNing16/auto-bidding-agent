"""Reproducible, bounded commands; no background training starts on import."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .config import TrainConfig, VARIANTS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("verify", help="run environment parity and PPO/LSTM mathematical tests")
    for name in ("train", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--config")
        p.add_argument("--variant", choices=VARIANTS)
        p.add_argument("--output", required=True)
        p.add_argument("--episodes", type=int)
        p.add_argument("--workers", type=int)
        p.add_argument("--max-seconds", type=float)
        p.add_argument("--seed", type=int)
        p.add_argument("--resume", action="store_true")
    sub.add_parser("evaluate", help="forward arguments to harmony_rl.evaluate")
    sub.add_parser("report", help="forward arguments to harmony_rl.report")
    sub.add_parser("diagnose", help="forward arguments to harmony_rl.diagnostics")
    args, remaining = parser.parse_known_args(argv)
    if args.command in ("evaluate", "report", "diagnose"):
        if args.command == "evaluate":
            from .evaluate import main as delegated
        elif args.command == "report":
            from .report import main as delegated
        else:
            from .diagnostics import main as delegated
        delegated(remaining)
        return
    if remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    if args.command == "verify":
        root = Path(__file__).resolve().parents[1]
        paths = sorted(str(p) for p in (root / "tests").glob("test_rl_*.py"))
        temporary = root / "outputs" / f"rl_verify_{time.time_ns()}"
        raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", *paths,
            "--basetemp", str(temporary), "-p", "no:cacheprovider"], cwd=root))
    from .train import train
    cfg = TrainConfig.from_file(args.config) if args.config else TrainConfig()
    if args.command == "smoke":
        cfg.smoke = True
        cfg.episodes = 4
        cfg.batch_episodes = 2
        cfg.workers = 1
        cfg.max_seconds = 180
        cfg.validate_every = 2
        cfg.validation_roles = ("0",)
        cfg.scenarios = ("Fixed", "Mixed2")
    for field in ("variant", "episodes", "workers", "max_seconds", "seed"):
        if getattr(args, field) is not None:
            setattr(cfg, field, getattr(args, field))
    print(json.dumps(train(cfg, args.output, resume=args.resume), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
