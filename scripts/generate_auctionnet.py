"""Run the pinned AuctionNet generator, then adapt its requests to our rules.

The optional participant thinning and 48-to-72 rebinning are synthetic assumptions.
Run from the repository after installing the package and NumPy.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

from autobid.adapters import convert_auctionnet

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "c20de631a40e24669d1d6a8e2acefd8f492f172d"
GENERATOR_PATH = ROOT / "third_party/auctionnet/simul_bidding_env/PvGenerator/NeurIPSPvGen.py"


def generate(output: Path, *, seed: int = 0, requests: int = 5000,
             players: int = 6, categories: int = 3,
             participation_rate: float = 0.7, pvalue_scale: float = 1000.0) -> dict:
    if requests < 1 or players < 1 or categories < 1 or players % categories:
        raise ValueError("requests/players/categories must be positive; players must divide evenly into categories")
    if not 0 < participation_rate <= 1:
        raise ValueError("participation_rate must be in (0, 1]")
    spec = importlib.util.spec_from_file_location("pinned_auctionnet_generator", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    generator = module.NeurIPSPvGen(
        episode=seed, num_tick=48, num_agent=players,
        num_agent_category=players // categories, num_category=categories,
        pv_num=requests,
    )
    # Independent RNG: upstream module has its own legacy global seed statements.
    rng = np.random.default_rng(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = output.with_suffix(".auctionnet.csv")
    fields = ["deliveryPeriodIndex", "timeStepIndex", "pvIndex", "advertiserNumber",
              "advertiserCategoryIndex", "pValue"]
    count = 0
    rows = 0
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for t, values in enumerate(generator.pv_values):
            for request_values in values:
                active = rng.random(players) < participation_rate
                if not active.any():
                    active[rng.integers(players)] = True
                for player in np.flatnonzero(active):
                    writer.writerow(dict(zip(fields, [seed, t, count, int(player),
                        int(player) // (players // categories), float(request_values[player])])))
                    rows += 1
                count += 1
    if count == 0:
        raise ValueError("upstream rounded all traffic counts to zero; increase --requests")
    manifest = convert_auctionnet(source, output, period=str(seed), source_steps=48,
        target_steps=72, time_base=0, pvalue_scale=pvalue_scale)
    manifest["source_dataset_id"] = f"auctionnet-synthetic:{seed}"
    manifest["source_kind"] = "synthetic"
    manifest["generator"] = {
        "repository": "https://github.com/alimama-tech/AuctionNet", "commit": COMMIT,
        "source_path": str(GENERATOR_PATH.relative_to(ROOT)), "seed": seed,
        "requested_opportunities": requests, "actual_opportunities": count,
        "players": players, "categories": categories, "rows": rows,
        "participation_rate": participation_rate,
        "participant_thinning": "independent Bernoulli mask; force one uniform participant if empty",
        "nature": "synthetic AuctionNet traffic; pValue is a proxy value; not official competition data",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/auctionnet-small.csv"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--requests", type=int, default=5000)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--categories", type=int, default=3)
    parser.add_argument("--participation-rate", type=float, default=0.7)
    parser.add_argument("--pvalue-scale", type=float, default=1000.0)
    args = parser.parse_args()
    manifest = generate(args.output, seed=args.seed, requests=args.requests,
        players=args.players, categories=args.categories,
        participation_rate=args.participation_rate, pvalue_scale=args.pvalue_scale)
    print(json.dumps({"output": str(args.output), "generator": manifest["generator"]}, indent=2))


if __name__ == "__main__":
    main()
