"""Five opponent scenarios using the existing, unique AuctionEngine.

The first three scenarios deliberately preserve historical seeds, assignment
and settlement. ``PID`` is an import-only historical name, never a new task
name. Adding ``auctionnet_pid`` changes the full opponent-file fingerprint;
the separate legacy fingerprint permits an explicit provenance comparison.
Agents receive only own configuration and completed own feedback. Market
matrices, masks, opponent identities and scenario names stay in this wrapper.
"""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import psutil

from harmony_bidding.agent_api import Observation, OwnConfig, OwnFeedback, PublicConfig
from harmony_bidding.agents import make_agent
from harmony_bidding.auction import AuctionEngine, validate_alpha
from harmony_bidding.schema import Rules, fingerprint

SCENARIOS = ("Fixed", "LocalPID", "Mixed", "AuctionNetPID", "Mixed2")
LEGACY_SCENARIOS = ("Fixed", "LocalPID", "Mixed")
TIMING_SCOPE = "target bidding/update direct calls; excludes validation, settlement and serialization"


def historical_scenario_name(scenario):
    """Canonicalize a historical row during import, not a runnable task."""
    canonical = "LocalPID" if scenario == "PID" else scenario
    if canonical not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    return canonical


def opponent_specs(ids, target, scenario, frozen, assignment_seed, day_id):
    """Return episode-stable specifications; make_agent creates each instance.

    Mixed2 uses the same Fixed membership as Mixed for a matching day, target
    and seed, replacing only the LocalPID policy family. No target policy or
    scenario salt enters the historical assignment RNG.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown canonical scenario {scenario!r}; import historical PID as LocalPID")
    if target not in ids:
        raise ValueError(f"unknown target {target}")
    others = [player for player in ids if player != target]
    if scenario == "Fixed":
        fixed = set(others)
    elif scenario in ("Mixed", "Mixed2"):
        assignment = int(fingerprint([assignment_seed, day_id, target])[:16], 16)
        fixed = set(np.random.default_rng(assignment).permutation(others)[:len(others) // 2])
    else:
        fixed = set()
    dynamic = "auctionnet_pid" if scenario in ("AuctionNetPID", "Mixed2") else "pid"
    return {player: frozen["fixed"] if player in fixed else frozen[dynamic] for player in others}


def run_episode(day, target, target_spec, scenario, opponents, seed=20261028,
                budget_scale=1., collect=False, collect_all=False, algorithm="", training_seed=0,
                diagnostic_path=None):
    """Run one complete 72-slot target-replacement episode.

    The public signature and (row, slots, transitions) return match the old
    evaluator. Target slot diagnostics are always returned, including when
    ``collect`` is false or the budget is exhausted. ``collect_all`` retains
    its legacy meaning; only the target contributes inference timings.
    Runner-owned phase/task identifiers must be attached by the caller. An
    optional JSONL trace is flushed outside policy timing and retained if a
    policy or auction raises; only successfully completed calls return a row.
    """
    arguments = (day, target, target_spec, scenario, opponents, seed, budget_scale,
                 collect, collect_all, algorithm, training_seed)
    if diagnostic_path is None:
        return _run_episode(*arguments, diagnostic=lambda record: None)
    path = Path(diagnostic_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as handle:
        def write(record):
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        return _run_episode(*arguments, diagnostic=write)


def _run_episode(day, target, target_spec, scenario, opponents, seed, budget_scale,
                 collect, collect_all, algorithm, training_seed, *, diagnostic):
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown canonical scenario {scenario!r}; import historical PID as LocalPID")
    if len(day.slots) != 72:
        raise ValueError("a complete episode requires exactly 72 slots")
    started = perf_counter()
    advertisers = tuple(replace(a, budget=round(a.budget * budget_scale, 6)) for a in day.advertisers)
    ids = [a.advertiser_id for a in advertisers]
    if target not in ids:
        raise ValueError(f"unknown target {target}")
    rules = Rules(tie_seed=int(seed))
    engine = AuctionEngine(advertisers, rules)
    specs = opponent_specs(ids, target, scenario, opponents, int(seed) + 1000003, day.day_id)
    assignment_fingerprint = fingerprint(specs)
    specs[target] = target_spec
    opponent_fingerprint = fingerprint(opponents)
    legacy_opponent_fingerprint = (fingerprint({key: opponents[key] for key in ("fixed", "pid")})
                                   if all(key in opponents for key in ("fixed", "pid")) else None)
    model_fingerprint = fingerprint(target_spec)
    identities = dict(dataset_fingerprint=day.fingerprint, rules_version=rules.rules_version,
                      opponent_fingerprint=opponent_fingerprint,
                      legacy_opponent_fingerprint=legacy_opponent_fingerprint,
                      assignment_fingerprint=assignment_fingerprint, model_fingerprint=model_fingerprint)
    agents, owns = {}, {}
    histories = {player: [] for player in ids}
    for advertiser in advertisers:
        player = advertiser.advertiser_id
        # Preserve the old RNG inputs and per-role independence exactly.
        agent_seed = int(fingerprint([seed, day.day_id, player])[:16], 16)
        owns[player] = OwnConfig(player, advertiser.advertiser_type, advertiser.budget, agent_seed)
        agents[player] = make_agent(specs[player])
        agents[player].reset(owns[player], PublicConfig(72))
    times, slots, transitions = [], [], []
    selected = set(ids if collect_all else [target])

    def observation(player, slot):
        return Observation(owns[player], slot, 72, engine.money.amount(engine.remaining[player]),
                           tuple(histories[player]))

    if collect:
        from harmony_baselines.neural import encode_observation

    for slot in range(72):
        observations = {player: observation(player, slot) for player in ids}
        actions = {}
        for player in ids:
            if player == target:
                diagnostic({"phase": "before_action", "slot": slot,
                            "remaining_budget": observations[player].remaining_budget})
                before = perf_counter()
                raw = agents[player].bidding(observations[player])
                bidding_ms = (perf_counter() - before) * 1000
                actions[player] = validate_alpha(raw)
                diagnostic({"phase": "action", "slot": slot, "alpha": actions[player]})
            else:
                actions[player] = validate_alpha(agents[player].bidding(observations[player]))
        market = day.slots[slot]
        stats, _ = engine.run_matrix_slot(market.values, market.participation_mask, actions,
            day_id=day.day_id, time_slot=slot, opportunity_ids=market.request_ids,
            original_order=market.original_order)
        for player in ids:
            feedback = OwnFeedback(slot, actions[player], int(stats[player]["wins"]),
                engine.money.amount(stats[player]["cost_units"]), float(stats[player]["value"]),
                engine.money.amount(engine.remaining[player]),
                engine.money.amount(engine.initial[player] - engine.remaining[player]),
                engine.cumulative_value[player])
            if player == target:
                before = perf_counter()
                agents[player].update(feedback)
                update_ms = (perf_counter() - before) * 1000
                times.append((bidding_ms, update_ms))
                diagnostic({"phase": "feedback", "slot": slot, "alpha": actions[player],
                            "cost": feedback.cost, "value": feedback.value, "wins": feedback.wins,
                            "remaining_budget": feedback.remaining_budget})
            else:
                agents[player].update(feedback)
            histories[player].append(feedback)
            if player in selected and collect:
                transitions.append({"state": encode_observation(observations[player]),
                    "action": actions[player], "reward": feedback.value,
                    "next_state": encode_observation(observation(player, slot + 1)),
                    "done": slot == 71, "player_id": player, "time_slot": slot})
            if player in selected:
                slots.append(dict(algorithm=algorithm, scenario=scenario, day_id=day.day_id,
                    player_id=player, player_type=owns[player].advertiser_type, seed=int(seed),
                    training_seed=int(training_seed), budget_scale=budget_scale,
                    time_slot=slot, slot=slot, alpha=actions[player], cost=feedback.cost,
                    value=feedback.value, wins=feedback.wins, remaining_budget=feedback.remaining_budget,
                    cumulative_cost=feedback.cumulative_cost, cumulative_value=feedback.cumulative_value,
                    initial_budget=owns[player].initial_budget,
                    bidding_ms=bidding_ms if player == target else None,
                    update_ms=update_ms if player == target else None, **identities))
    timing = np.asarray(times, dtype=float)
    cost = engine.money.amount(engine.initial[target] - engine.remaining[target])
    memory = psutil.Process().memory_info()
    row = dict(algorithm=algorithm, scenario=scenario, day_id=day.day_id, player_id=target,
        player_type=owns[target].advertiser_type, seed=int(seed), training_seed=int(training_seed),
        budget_scale=budget_scale, initial_budget=owns[target].initial_budget,
        total_value=engine.cumulative_value[target], total_cost=cost,
        budget_utilization=cost / owns[target].initial_budget if owns[target].initial_budget else 0.,
        wins=sum(feedback.wins for feedback in histories[target]), elapsed_seconds=perf_counter() - started,
        bidding_p99_ms=float(np.percentile(timing[:, 0], 99)),
        update_p99_ms=float(np.percentile(timing[:, 1], 99)),
        paired_p99_ms=float(np.percentile(timing.sum(axis=1), 99)), rss_mb=memory.rss / 2**20,
        peak_process_rss_mb=getattr(memory, "peak_wset", memory.rss) / 2**20,
        completed=True, completed_slots=72, timing_scope=TIMING_SCOPE, **identities)
    return row, slots, transitions
