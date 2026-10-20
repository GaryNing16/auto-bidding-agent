"""One command surface for data, simulation, learning and repeatable evaluation."""
import argparse
from pathlib import Path
import json

from .data import load_requests, write_requests, synthetic_requests, summarize_requests
from .types import AuctionConfig
from .evaluate import evaluate, arena, save_json, run_manifest


def read_config(path):
    return AuctionConfig(**json.loads(Path(path).read_text()))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='autobid', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate', help='create sparse synthetic six-column data')
    gen.add_argument('--output', required=True)
    gen.add_argument('--seed', type=int, default=7)
    gen.add_argument('--steps', type=int, default=72)
    gen.add_argument('--players', type=int, default=3)
    gen.add_argument('--requests-per-step', type=int, default=12)
    validate = sub.add_parser('validate')
    validate.add_argument('--data', required=True)
    validate.add_argument('--steps', type=int, default=72)
    convert = sub.add_parser('convert-auctionnet')
    convert.add_argument('--source', required=True)
    convert.add_argument('--output', required=True)
    convert.add_argument('--period', required=True)
    convert.add_argument('--time-base', type=int, choices=[0, 1], required=True)
    convert.add_argument('--source-steps', type=int, default=48)
    convert.add_argument('--target-steps', type=int, default=72)
    convert.add_argument('--pvalue-scale', type=float, default=1.)
    convert.add_argument('--max-requests', type=int)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--data', nargs='+', required=True)
    ev.add_argument('--config', required=True)
    ev.add_argument('--output', required=True)
    ev.add_argument('--strategies', nargs='+', default=['fixed:0.5', 'fixed:1', 'pacing'])
    ev.add_argument('--seeds', nargs='+', type=int, default=[7])
    ev.add_argument('--roles', nargs='+')
    ev.add_argument('--model')
    ev.add_argument('--audit', action='store_true')
    ar = sub.add_parser('arena')
    ar.add_argument('--data', required=True)
    ar.add_argument('--config', required=True)
    ar.add_argument('--output', required=True)
    ar.add_argument('--lineup', nargs='+', default=['fixed:1', 'pacing', 'fixed:2'])
    ar.add_argument('--model')
    train = sub.add_parser('train')
    train.add_argument('--data', nargs='+', required=True)
    train.add_argument('--config', required=True)
    train.add_argument('--player', required=True)
    train.add_argument('--rotate-roles', action='store_true')
    train.add_argument('--timesteps', type=int, default=4096)
    train.add_argument('--seed', type=int, default=7)
    train.add_argument('--output', required=True)
    fit = sub.add_parser('fit-forecast')
    fit.add_argument('--data', nargs='+', required=True)
    fit.add_argument('--config', required=True)
    fit.add_argument('--output', required=True)
    fe = sub.add_parser('evaluate-forecast')
    fe.add_argument('--data', nargs='+', required=True)
    fe.add_argument('--config', required=True)
    fe.add_argument('--model', required=True)
    fe.add_argument('--output', required=True)
    split = sub.add_parser('split')
    split.add_argument('--data', nargs='+', required=True)
    split.add_argument('--output', required=True)
    split.add_argument('--seed', type=int, default=7)
    args = parser.parse_args(argv)
    if args.command == 'generate':
        requests = synthetic_requests(args.seed, args.steps, args.players, args.requests_per_step)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_requests(requests, args.output)
        save_json(Path(args.output).with_suffix('.manifest.json'), {'source':'synthetic', 'seed':args.seed,
                  'steps':args.steps, 'players':args.players, 'requests_per_step':args.requests_per_step,
                  'is_official_data':False})
        print(json.dumps(summarize_requests(requests, args.steps), ensure_ascii=False, indent=2))
    elif args.command == 'validate':
        print(json.dumps(summarize_requests(load_requests(args.data, args.steps), args.steps), ensure_ascii=False, indent=2))
    elif args.command == 'convert-auctionnet':
        from .adapters import convert_auctionnet
        result = convert_auctionnet(args.source, args.output, period=args.period, source_steps=args.source_steps,
                  target_steps=args.target_steps, time_base=args.time_base, pvalue_scale=args.pvalue_scale,
                  max_requests=args.max_requests)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == 'split':
        from .forecast import split_by_episode, write_split_manifest
        result = split_by_episode(args.data, seed=args.seed)
        write_split_manifest(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        config = read_config(args.config)
        if args.command == 'evaluate':
            requests = {path: load_requests(path, config.steps) for path in args.data}
            report = evaluate(requests, config, args.output, strategies=args.strategies,
                              seeds=args.seeds, roles=args.roles, model=args.model, audit=args.audit)
            print(json.dumps(report['aggregates'], ensure_ascii=False, indent=2))
        elif args.command == 'arena':
            requests = load_requests(args.data, config.steps)
            result = arena(requests, config, args.output, args.lineup, model=args.model)
            save_json(Path(args.output)/'manifest.json', run_manifest([args.data],config,command='arena',extra={'lineup':args.lineup}))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == 'train':
            from .rl import train_ppo
            episodes = [load_requests(path, config.steps) for path in args.data]
            path = train_ppo(episodes[0],config,args.output,player_id=args.player,total_timesteps=args.timesteps,
                            seed=args.seed,episode_requests=episodes,player_ids=list(config.budgets) if args.rotate_roles else None)
            save_json(Path(path).with_suffix('.run.json'),run_manifest(args.data,config,command='train',extra={
                      'player':args.player,'rotate_roles':args.rotate_roles,'timesteps':args.timesteps,'seed':args.seed}))
            print(path)
        elif args.command == 'fit-forecast':
            from .forecast import fit_forecast
            fit_forecast(args.data,config,args.output)
            print(args.output)
        elif args.command == 'evaluate-forecast':
            from .forecast import evaluate_forecast
            result = evaluate_forecast(args.model,args.data,config)
            save_json(args.output,result)
            print(json.dumps(result,ensure_ascii=False,indent=2))
