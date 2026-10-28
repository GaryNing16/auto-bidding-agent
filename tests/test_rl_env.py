"""Exact settlement/causality integration checks for the training adapter."""
from copy import deepcopy
from dataclasses import fields, replace

import numpy as np
import pytest

from harmony_baselines.classical import spec
from harmony_baselines.data import MatrixSlot
from harmony_baselines.five_scenarios import evaluator as five
from harmony_baselines.five_scenarios.common import pid_spec
from harmony_bidding.agent_api import Observation, OwnFeedback, observation_from_dict
from harmony_bidding.auction import AuctionEngine
from harmony_bidding.schema import Advertiser, Opportunity, Rules
from harmony_rl.env import BiddingEnv
from harmony_rl.markets import synthetic_day


def opponents():
    return {"fixed": spec("FixedAlpha", alpha=2.),
            "pid": spec("LocalPID", initial_alpha=2., kp=.5, ki=.05, kd=.05),
            "auctionnet_pid": pid_spec(initial_alpha=2.)}


@pytest.mark.parametrize("scenario", five.SCENARIOS)
@pytest.mark.parametrize("seed", [3, 4])
def test_step_adapter_exactly_matches_frozen_complete_evaluator(monkeypatch, scenario, seed):
    day = synthetic_day(12, players=5, requests_per_slot=6)
    expected_debug = []
    class RecordingEngine(AuctionEngine):
        def run_matrix_slot(self, *args, **kwargs):
            stats, debug = super().run_matrix_slot(*args, **kwargs, record_debug=True)
            expected_debug.append(debug)
            return stats, debug
    monkeypatch.setattr(five, "AuctionEngine", RecordingEngine)
    row, slots, _ = five.run_episode(day, "0", spec("FixedAlpha", alpha=9.), scenario, opponents(), seed=seed)
    env = BiddingEnv(day, "0", opponents(), scenario, seed, record_debug=True)
    obs, info = env.reset()
    assert set(info) == {"initial_budget"}
    for t in range(72):
        assert type(obs) is Observation and len(obs.history) == t
        assert {f.name for f in fields(obs)} == {"own_config", "time_slot", "total_slots", "remaining_budget", "history"}
        obs, reward, done, truncated, info = env.step(9.)
        assert set(info) == {"feedback"} and type(info["feedback"]) is OwnFeedback
        assert reward == slots[t]["value"]
        assert done == (t == 71) and not truncated
        for key in ("cost", "value", "wins", "remaining_budget", "cumulative_cost", "cumulative_value"):
            assert env.diagnostics[t][key] == slots[t][key]
    assert env.debug_log == expected_debug
    for key in ("total_value", "total_cost", "wins", "initial_budget", "budget_utilization"):
        assert env.metrics()[key] == row[key]
    assert all(len(h) == 72 for h in env.histories.values())
    assert all(a.feedback_count == 72 for a in env.agents.values())
    with pytest.raises(RuntimeError, match="active"):
        env.step(1.)


def test_sparse_extreme_ties_reauction_and_depletion_match_reference():
    day = synthetic_day(7, players=3, requests_per_slot=0)
    values = np.array([[10., 8., 11.], [3., 2., 0.], [1e100, 0., 0.],
                       [1., 1., 1.], [0., 2., 1.], [1., 1., 1.]])
    mask = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=bool)
    market = MatrixSlot(values, mask, np.array([str(i) for i in range(6)]), np.arange(6))
    day = replace(day, advertisers=(Advertiser("0", "0", 1.), Advertiser("1", "0", 3.), Advertiser("2", "0", 3.)),
                  slots=(market,) + day.slots[1:])
    config = {k: {"kind": "constant", "alpha": 1.} for k in ("fixed", "pid", "auctionnet_pid")}
    env = BiddingEnv(day, "0", config, "Fixed", seed=41, record_debug=True)
    env.reset()
    slow = AuctionEngine(day.advertisers, Rules(tie_seed=41))
    for t, market in enumerate(day.slots):
        alpha = 9. if t == 0 else 1.
        opportunities = [Opportunity(day.day_id, t, str(r), int(order), tuple(row), tuple(bool(x) for x in present))
                         for row, present, r, order in zip(market.values, market.participation_mask,
                                                         market.request_ids, market.original_order)]
        stats, expected = slow.run_slot(opportunities, {"0": alpha, "1": 1., "2": 1.})
        env.step(alpha)
        assert env.debug_log[t] == expected
        assert env.engine.remaining == slow.remaining
        assert env.engine.cumulative_value == slow.cumulative_value
    assert any(event["excluded"] for event in env.debug_log[0])
    assert env.debug_log[0][2]["payment_units"] == 0
    assert env.debug_log[0][2]["winner"] == "0"


@pytest.mark.parametrize("invalid", [True, "2", -1., float("nan"), float("inf"), 1e308])
def test_invalid_action_or_product_does_not_mutate_opponents_or_ledger(invalid):
    day = synthetic_day(0, 3, 2)
    slot = replace(day.slots[0], values=np.full((2, 3), 2.), participation_mask=np.ones((2, 3), dtype=bool))
    day = replace(day, slots=(slot,) + day.slots[1:])
    env = BiddingEnv(day, "0", opponents(), "AuctionNetPID", seed=5)
    env.reset()
    before = env.observation()
    agent_state = [deepcopy(a.__dict__) for a in env.agents.values()]
    with pytest.raises(ValueError):
        env.step(invalid)
    assert env.observation() == before
    assert not env.diagnostics and env.time_slot == 0
    assert [a.__dict__ for a in env.agents.values()] == agent_state


def test_future_changes_cannot_change_observation_or_initial_action():
    day = synthetic_day(2, 3, 2)
    changed = replace(day, slots=tuple(replace(s, values=s.values * 999.) for s in day.slots))
    a = BiddingEnv(day, "0", opponents(), residual_spec=opponents()["pid"])
    b = BiddingEnv(changed, "0", opponents(), residual_spec=opponents()["pid"])
    assert a.reset() == b.reset()
    assert a.base_alpha() == b.base_alpha()


def test_residual_receives_actual_spend_and_snapshot_continues_live_opponents():
    env = BiddingEnv(synthetic_day(4, 3, 3), "0", opponents(), "Mixed2", seed=7,
                     residual_spec=opponents()["auctionnet_pid"], record_debug=True)
    env.reset()
    for _ in range(9):
        env.step(env.base_alpha() * 1.3)
    assert env.residual_agent.feedback_count == 9
    assert env.residual_agent.remaining_budget == env.observation().remaining_budget
    twin = env.snapshot()
    changed = env.with_extra_budget(.1)
    assert changed.observation().own_config.initial_budget == pytest.approx(env.observation().own_config.initial_budget + .1)
    assert changed.observation().remaining_budget == pytest.approx(env.observation().remaining_budget + .1)
    assert changed.engine.cumulative_value == env.engine.cumulative_value
    for _ in range(9, 72):
        assert env.step(2.) == twin.step(2.)
        changed.step(2.)
    assert env.debug_log == twin.debug_log
    assert changed.time_slot == 72 and changed.metrics()["completed"]


def test_single_player_and_zero_initial_budget_keep_complete_trajectory():
    day = synthetic_day(1, 1, 2)
    env = BiddingEnv(day, "0", opponents())
    env.reset()
    for _ in range(72):
        env.step(17.)
    opportunities = sum(int(s.participation_mask.sum()) for s in day.slots)
    assert env.metrics()["total_cost"] == 0 and env.metrics()["wins"] == opportunities
    zero_day = replace(day, advertisers=(replace(day.advertisers[0], budget=0.),))
    env = BiddingEnv(zero_day, "0", opponents())
    env.reset()
    for _ in range(72):
        env.step(17.)
    assert env.metrics()["total_value"] == 0 and env.time_slot == 72
