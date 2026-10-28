"""Real training-data differential evidence for the single core's matrix API.

Run: .venv-baselines/Scripts/python.exe -m scripts.verify_matrix_core
This is correctness evidence, never a performance-comparison experiment.
"""
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from harmony_baselines.data import load_day, sha256_file
from harmony_bidding.auction import AuctionEngine
from harmony_bidding.schema import Opportunity, Rules, fingerprint


def main():
    day = load_day("period-7")
    available = [i for i, slot in enumerate(day.slots) if len(slot.values) >= 5000]
    selected_slots = [available[0], available[len(available)//2], available[-1]]
    rng = np.random.default_rng(20261028)
    results = []
    for slot_number in selected_slots:
        slot = day.slots[slot_number]
        for count, scale, sparse in [(500, 1., False), (5000, .00001, False), (5000, .001, True)]:
            indices = np.sort(rng.choice(len(slot.values), size=count, replace=False))
            values = np.asarray(slot.values[indices]).copy()
            mask = np.asarray(slot.participation_mask[indices]).copy()
            if sparse:
                mask &= rng.random(mask.shape) > .3
                values[~mask] = 0.
            ids = np.asarray(slot.request_ids[indices]).tolist()
            order = np.asarray(slot.original_order[indices])
            advertisers = tuple(replace(a, budget=round(a.budget*scale, 6)) for a in day.advertisers)
            actions = {a.advertiser_id: float(alpha)
                       for a, alpha in zip(advertisers, rng.lognormal(6, 2, len(advertisers)))}
            rules = Rules(tie_seed=20261028)
            slow, fast = AuctionEngine(advertisers, rules), AuctionEngine(advertisers, rules)
            opportunities = tuple(Opportunity(day.day_id, slot_number, ids[k], int(order[k]),
                tuple(float(v) for v in values[k]), tuple(bool(x) for x in mask[k])) for k in range(count))
            t = perf_counter()
            expected = slow.run_slot(opportunities, actions)
            reference_seconds = perf_counter()-t
            t = perf_counter()
            actual = fast.run_matrix_slot(values, mask, actions, day_id=day.day_id, time_slot=slot_number,
                opportunity_ids=ids, original_order=order, record_debug=True)
            matrix_seconds = perf_counter()-t
            assert actual == expected, "Per-request/debug/statistics mismatch"
            assert fast.remaining == slow.remaining, "Integer ledger mismatch"
            assert fast.cumulative_value == slow.cumulative_value, "Value accumulation mismatch"
            result = {"day_id": day.day_id, "time_slot": slot_number, "requests": count,
                "advertisers": len(advertisers), "budget_scale": scale,
                "artificial_sparse_mask": sparse, "all_stats_debug_and_ledgers_exact": True,
                "reference_seconds": reference_seconds, "matrix_seconds": matrix_seconds,
                "actions": actions, "result_fingerprint": fingerprint(actual),
                "excluded_candidates": sum(len(r["excluded"]) for r in actual[1])}
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "actions"}), flush=True)
    root = Path(__file__).resolve().parents[1]
    destination = root / "outputs/baseline_comparison_20261028/matrix_core_real_data_verification.json"
    destination.write_text(json.dumps({"status": "completed", "valid": True,
        "scope": "correctness only; sampled training-period requests, not final scores",
        "seed": 20261028, "source_fingerprint": day.fingerprint,
        "core_sha256": sha256_file(root / "harmony_bidding/auction.py"), "cases": results},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
