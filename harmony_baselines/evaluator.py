"""One evaluator for every baseline and stage; settlement ONLY in AuctionEngine.

This wrapper holds market matrices. Agents receive frozen own-only snapshots.
No policy receives this evaluator, a data handle, mask, or opponent information.
"""
from dataclasses import replace
from hashlib import sha256
from time import perf_counter
import json
import numpy as np
import psutil
from harmony_bidding.agent_api import OwnConfig, PublicConfig, Observation, OwnFeedback
from harmony_bidding.agents import make_agent
from harmony_bidding.auction import AuctionEngine, validate_alpha
from harmony_bidding.schema import Rules, fingerprint
from .classical import spec

SCENARIOS = ('Fixed', 'PID', 'Mixed')


def opponent_specs(ids, target, scenario, frozen, assignment_seed, day_id):
    others = [x for x in ids if x != target]
    if scenario == 'Fixed':
        fixed = set(others)
    elif scenario == 'PID':
        fixed = set()
    elif scenario == 'Mixed':
        seed = int(fingerprint([assignment_seed, day_id, target])[:16], 16)
        fixed = set(np.random.default_rng(seed).permutation(others)[:len(others)//2])
    else:
        raise ValueError(f'unknown scenario {scenario}')
    return {p: frozen['fixed'] if p in fixed else frozen['pid'] for p in others}


def run_episode(day, target, target_spec, scenario, opponents, seed=20261028,
                budget_scale=1., collect=False, collect_all=False, algorithm='', training_seed=0):
    start = perf_counter()
    advertisers = tuple(replace(a, budget=round(a.budget * budget_scale, 6)) for a in day.advertisers)
    ids = [a.advertiser_id for a in advertisers]
    if target not in ids:
        raise ValueError(f'unknown target {target}')
    rules = Rules(tie_seed=int(seed))
    engine = AuctionEngine(advertisers, rules)
    specs = opponent_specs(ids, target, scenario, opponents, int(seed) + 1000003, day.day_id)
    specs[target] = target_spec
    agents, histories, owns = {}, {p: [] for p in ids}, {}
    for a in advertisers:
        p = a.advertiser_id
        agent_seed = int(fingerprint([seed, day.day_id, p])[:16], 16)
        owns[p] = OwnConfig(p, a.advertiser_type, a.budget, agent_seed)
        agents[p] = make_agent(specs[p])
        agents[p].reset(owns[p], PublicConfig(72))
    times, slots, transitions = [], [], []
    selected = ids if collect_all else [target]
    def observation(p, t):
        return Observation(owns[p], t, 72, engine.money.amount(engine.remaining[p]), tuple(histories[p]))
    for t in range(72):
        observations = {p: observation(p, t) for p in ids}
        actions = {}
        before = perf_counter()
        for p in ids:
            if p == target:
                before = perf_counter()
                actions[p] = validate_alpha(agents[p].bidding(observations[p]))
                bid_ms = (perf_counter()-before)*1000
            else:
                actions[p] = validate_alpha(agents[p].bidding(observations[p]))
        market = day.slots[t]
        stats, _ = engine.run_matrix_slot(market.values, market.participation_mask, actions,
            day_id=day.day_id, time_slot=t, opportunity_ids=market.request_ids,
            original_order=market.original_order)
        for p in ids:
            feedback = OwnFeedback(t, actions[p], int(stats[p]['wins']), engine.money.amount(stats[p]['cost_units']),
                float(stats[p]['value']), engine.money.amount(engine.remaining[p]),
                engine.money.amount(engine.initial[p]-engine.remaining[p]), engine.cumulative_value[p])
            before = perf_counter()
            agents[p].update(feedback)
            if p == target:
                times.append((bid_ms, (perf_counter()-before)*1000))
            histories[p].append(feedback)
            if p in selected and collect:
                from .neural import encode_observation
                transitions.append({'state': encode_observation(observations[p]), 'action': actions[p],
                    'reward': feedback.value, 'next_state': encode_observation(observation(p, t+1)),
                    'done': t == 71, 'player_id': p, 'time_slot': t})
            if p in selected:
                slots.append({'day_id': day.day_id, 'scenario': scenario, 'player_id': p,
                    'player_type': owns[p].advertiser_type, 'time_slot': t, 'alpha': actions[p],
                    'cost': feedback.cost, 'value': feedback.value, 'wins': feedback.wins,
                    'initial_budget': owns[p].initial_budget})
    timing = np.array(times)
    value = engine.cumulative_value[target]
    cost = engine.money.amount(engine.initial[target]-engine.remaining[target])
    memory = psutil.Process().memory_info()
    row = dict(algorithm=algorithm, scenario=scenario, day_id=day.day_id, player_id=target,
        player_type=owns[target].advertiser_type, seed=int(seed), training_seed=int(training_seed),
        budget_scale=budget_scale, initial_budget=owns[target].initial_budget, total_value=value,
        total_cost=cost, budget_utilization=cost/owns[target].initial_budget if owns[target].initial_budget else 0.,
        wins=sum(f.wins for f in histories[target]), elapsed_seconds=perf_counter()-start,
        bidding_p99_ms=float(np.percentile(timing[:,0],99)), update_p99_ms=float(np.percentile(timing[:,1],99)),
        paired_p99_ms=float(np.percentile(timing.sum(axis=1),99)), rss_mb=memory.rss/2**20,
        peak_process_rss_mb=getattr(memory, 'peak_wset', memory.rss)/2**20,
        dataset_fingerprint=day.fingerprint, rules_version=rules.rules_version,
        opponent_fingerprint=fingerprint(opponents), assignment_fingerprint=fingerprint({p:s for p,s in specs.items() if p!=target}),
        model_fingerprint=fingerprint(target_spec), completed=True)
    return row, slots, transitions
