"""Mechanism/oracle and real-core tests for the new AuctionNet PID adapter."""
from dataclasses import replace
import math
import sys

import numpy as np
import pytest

from harmony_baselines.five_scenarios.auctionnet_pid import AuctionNetPIDAgent
from harmony_bidding.agent_api import Observation, OwnConfig, OwnFeedback, PublicConfig
from harmony_bidding.agents import make_agent
from harmony_bidding.auction import AuctionEngine, validate_alpha
from harmony_bidding.schema import Advertiser, Opportunity, Rules
from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy


def own(budget=100.0, player="0"):
    return OwnConfig(player, "0", budget, 20261028)


def observe(agent, history=()):
    return Observation(agent.own_config, agent.feedback_count,
                       agent.public_config.total_slots, agent.remaining_budget, tuple(history))


def feedback(agent, remaining, cost=None):
    return OwnFeedback(agent.feedback_count, agent.alpha, 0,
                       agent.remaining_budget - remaining if cost is None else cost,
                       0.0, remaining, agent.own_config.initial_budget - remaining, 0.0)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("weighted", [False, True])
def test_48_slot_matches_pinned_upstream_on_valid_balances(seed, weighted):
    rng = np.random.default_rng(seed)
    ratios = rng.integers(1, 5, 48).astype(float) if weighted else np.ones(48)
    original = PidBiddingStrategy(budget=10000.0, exp_tempral_ratio=ratios)
    adapter = AuctionNetPIDAgent(expected_ratios=ratios)
    adapter.reset(own(10000.0), PublicConfig(48))
    for t in range(48):
        expected = float(original.bidding(t, np.ones(1), None, None, None, None, None, None)[0])
        assert adapter.bidding(observe(adapter)) == expected
        remaining = original.remaining_budget * (1.0 - float(rng.uniform(0.0, 0.06)))
        adapter.update(feedback(adapter, remaining))
        original.remaining_budget = remaining


def test_nonuniform_prior_preserves_asymmetric_branch_and_priority():
    # At t=1: weighted increase ratio=2*7.1/98<.7 while the plain decrease
    # ratio=2*71/98>1.1. The original if/elif must select the increase branch.
    agent = AuctionNetPIDAgent(expected_ratios=[1.0] + [0.1]*71)
    agent.reset(own(), PublicConfig(72))
    assert agent.bidding(observe(agent)) == 15.0
    agent.update(feedback(agent, 98.0))
    assert agent.bidding(observe(agent)) == 18.0


@pytest.mark.parametrize("cost, expected", [(0.0, 18.0), (1.0, 15.0), (2.0, 10.5)])
def test_72_slot_control_has_increase_hold_decrease(cost, expected):
    agent = AuctionNetPIDAgent()
    agent.reset(own(), PublicConfig(72))
    agent.bidding(observe(agent))
    agent.update(feedback(agent, 100.0-cost))
    assert agent.bidding(observe(agent)) == expected
    # Repeated inspection of the same decision cannot multiply alpha again.
    assert agent.bidding(observe(agent)) == expected


@pytest.mark.parametrize("lower, upper", [(1.0, 2.0), (0.7, 1.0)])
def test_threshold_comparisons_are_strict(lower, upper):
    agent = AuctionNetPIDAgent(lower_threshold=lower, upper_threshold=upper)
    agent.reset(own(72.0), PublicConfig(72))
    agent.bidding(observe(agent))
    agent.update(feedback(agent, 71.0))
    assert agent.bidding(observe(agent)) == 15.0


@pytest.mark.parametrize("cost, expected", [(0.0, 13.5), (2.0, 7.9)])
def test_configured_multipliers_are_applied(cost, expected):
    agent = AuctionNetPIDAgent(initial_alpha=10.0, increase_factor=1.35, decrease_factor=0.79)
    agent.reset(own(), PublicConfig(72))
    agent.bidding(observe(agent))
    agent.update(feedback(agent, 100.0-cost))
    assert agent.bidding(observe(agent)) == expected


@pytest.mark.parametrize("budget", [0.0, 100.0])
def test_zero_balance_continues_all_72_slots(budget):
    agent = AuctionNetPIDAgent()
    agent.reset(own(budget), PublicConfig(72))
    for t in range(72):
        alpha = agent.bidding(observe(agent))
        assert alpha == (15.0 if t == 0 and budget else 0.0)
        agent.update(feedback(agent, 0.0))
    assert agent.feedback_count == 72
    with pytest.raises(ValueError, match="time"):
        agent.bidding(observe(agent))


def test_reset_fully_clears_previous_episode_and_player_state():
    agent = AuctionNetPIDAgent(initial_alpha=12.0, player_alphas={"1": 21.0})
    agent.reset(own(), PublicConfig(72))
    for _ in range(3):
        agent.bidding(observe(agent))
        agent.update(feedback(agent, agent.remaining_budget))
    assert agent.alpha > 12.0
    agent.reset(own(), PublicConfig(72))
    assert agent.bidding(observe(agent)) == 12.0
    assert agent.feedback_count == 0 and agent.last_tick_cost == 0.0
    agent.reset(own(player="1"), PublicConfig(72))
    assert agent.bidding(observe(agent)) == 21.0
    agent.reset(own(player="2"), PublicConfig(72))
    assert agent.bidding(observe(agent)) == 12.0


def test_configuration_is_copied_and_opponents_have_independent_state():
    ratios, alphas = [1.0]*72, {"0": 30.0}
    agent = AuctionNetPIDAgent(expected_ratios=ratios, player_alphas=alphas)
    other = AuctionNetPIDAgent(expected_ratios=ratios, player_alphas=alphas)
    ratios[0], alphas["0"] = 9999.0, 1.0
    for a in (agent, other):
        a.reset(own(), PublicConfig(72))
        assert a.bidding(observe(a)) == 30.0
    agent.update(feedback(agent, 100.0))
    other.update(feedback(other, 98.0))
    assert agent.bidding(observe(agent)) == 36.0
    assert other.bidding(observe(other)) == 21.0


@pytest.mark.parametrize("kwargs", [
    {"initial_alpha": float("nan")}, {"initial_alpha": True}, {"initial_alpha": -1},
    {"initial_alpha": "15"}, {"lower_threshold": float("inf")},
    {"lower_threshold": 1.1}, {"increase_factor": 0.9}, {"decrease_factor": 0.0},
    {"decrease_factor": 1.1}, {"expected_ratios": [0.0]*72},
    {"expected_ratios": [float("nan")]*72}, {"player_alphas": {0: 15.0}},
    {"player_alphas": {"0": float("inf")}},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        AuctionNetPIDAgent(**kwargs)


@pytest.mark.parametrize("ratios", [[], [1.0]*48, [1e308]*72])
def test_curve_length_and_sum_are_validated_on_reset(ratios):
    agent = AuctionNetPIDAgent(expected_ratios=ratios)
    with pytest.raises(ValueError, match="expected_ratios"):
        agent.reset(own(), PublicConfig(72))


def test_finite_actions_have_no_global_cap_and_overflow_fails_atomically():
    agent = AuctionNetPIDAgent(initial_alpha=1e100)
    agent.reset(own(), PublicConfig(72))
    for _ in range(72):
        assert validate_alpha(agent.bidding(observe(agent))) >= 1e100
        agent.update(feedback(agent, 100.0))
    overflowing = AuctionNetPIDAgent(initial_alpha=sys.float_info.max)
    overflowing.reset(own(), PublicConfig(72))
    overflowing.bidding(observe(overflowing))
    overflowing.update(feedback(overflowing, 100.0))
    with pytest.raises(ValueError, match="overflow"):
        overflowing.bidding(observe(overflowing))
    assert overflowing.alpha == sys.float_info.max
    assert overflowing._last_bid_slot == 0


def test_intermediate_pacing_overflow_keeps_mathematical_formula():
    agent = AuctionNetPIDAgent()
    agent.reset(own(1e308), PublicConfig(72))
    agent.bidding(observe(agent))
    agent.update(feedback(agent, 9e307))
    # cost*71 overflows binary64, but (cost*71)/remaining is finite and >1.1.
    assert agent.bidding(observe(agent)) == 10.5


def test_bad_sequence_or_balance_rejected_without_mutation():
    agent = AuctionNetPIDAgent()
    with pytest.raises(ValueError, match="reset"):
        agent.bidding(Observation(own(), 0, 72, 100.0, ()))
    agent.reset(own(), PublicConfig(72))
    with pytest.raises(ValueError, match="sequence"):
        agent.update(feedback(agent, 99.0))
    agent.bidding(observe(agent))
    for bad in [replace(feedback(agent, 99.0), time_slot=1),
                replace(feedback(agent, 99.0), remaining_budget=float("nan")),
                feedback(agent, 101.0, cost=0.0), feedback(agent, 99.0, cost=4.0)]:
        with pytest.raises(ValueError):
            agent.update(bad)
        assert (agent.feedback_count, agent.remaining_budget, agent.last_tick_cost) == (0, 100.0, 0.0)
    with pytest.raises(ValueError, match="balance"):
        agent.bidding(replace(observe(agent), remaining_budget=99.0))


def test_policy_only_reads_own_clock_configuration_and_completed_feedback():
    class GuardedObservation:
        own_config = own()
        time_slot = 0
        total_slots = 72
        remaining_budget = 100.0

        def __getattr__(self, name):
            raise AssertionError(f"Unneeded/market information read: {name}")

    agent = AuctionNetPIDAgent()
    agent.reset(own(), PublicConfig(72))
    assert agent.bidding(GuardedObservation()) == 15.0
    agent.update(feedback(agent, 100.0))
    GuardedObservation.time_slot = 1
    assert agent.bidding(GuardedObservation()) == 18.0


def test_72_slot_real_auction_core_and_future_market_isolation():
    def rollout(future_value):
        ads = (Advertiser("0", "0", 30.0), Advertiser("1", "0", 100.0))
        engine = AuctionEngine(ads, Rules())
        agent = make_agent({"kind": "custom", "class_path":
                           "harmony_baselines.five_scenarios.auctionnet_pid:AuctionNetPIDAgent"})
        agent.reset(own(30.0), PublicConfig(72))
        history, actions = [], []
        for t in range(72):
            alpha = validate_alpha(agent.bidding(observe(agent, history)))
            actions.append(alpha)
            # Future market is evaluator-only. Change it from slot 36 onward;
            # the controller's earlier actions must be bit-identical.
            value = 0.2 if t < 36 else future_value
            req = Opportunity("pid_test", t, str(t), 0, (value, 0.05), (True, True))
            stats, _ = engine.run_slot((req,), {"0": alpha, "1": 1.0})
            fb = OwnFeedback(t, alpha, stats["0"]["wins"], engine.money.amount(stats["0"]["cost_units"]),
                             stats["0"]["value"], engine.money.amount(engine.remaining["0"]),
                             engine.money.amount(engine.initial["0"]-engine.remaining["0"]),
                             engine.cumulative_value["0"])
            agent.update(fb)
            history.append(fb)
        assert len(history) == agent.feedback_count == 72
        assert math.isclose(sum(f.cost for f in history), 30.0-agent.remaining_budget)
        return actions, engine.cumulative_value["0"]

    left, left_value = rollout(0.01)
    right, right_value = rollout(0.9)
    assert left[:37] == right[:37]  # slot36 action precedes its market feedback.
    assert left_value != right_value
