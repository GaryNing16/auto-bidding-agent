"""Hand-computable auction cases and information-boundary regression tests."""

from dataclasses import FrozenInstanceError, asdict
import math
import random
import unittest

from autobid.engine import DaySimulator
from autobid.policies import FixedPolicy, PacingPolicy
from autobid.types import AuctionConfig, Participation, Request


def request(time_step, request_id, **values):
    return Request(time_step, request_id, tuple(Participation(p, "type", v) for p, v in values.items()))


def config(budgets=None, **kwargs):
    budgets = {"A": 100.0, "B": 100.0, "C": 100.0} if budgets is None else budgets
    return AuctionConfig(budgets=budgets, player_types={p: "type" for p in budgets}, **kwargs)


class AuctionSemanticsTests(unittest.TestCase):
    def test_hand_calculated_three_player_second_price(self):
        sim = DaySimulator([request(0, "r1", A=10, B=8, C=11)], config(), audit=True)
        result = sim.step({"A": 1.2, "B": 1, "C": 1})
        self.assertEqual((result["A"].wins, result["A"].value, result["A"].cost), (1, 10, 11))
        self.assertEqual(result["A"].remaining_budget, 89)
        self.assertEqual(result["B"].cost, 0)
        self.assertEqual(sim.audit[0]["winner"], "A")

    def test_request_grouping_and_sparse_participation(self):
        sim = DaySimulator([
            request(0, "r1", A=10, B=8),
            request(0, "r2", B=1),
            request(0, "r3", A=2, C=3),
        ], config(), audit=True)
        result = sim.step({"A": 1, "B": 1, "C": 1})
        self.assertEqual((result["A"].value, result["A"].cost), (10, 8))
        self.assertEqual((result["B"].value, result["B"].cost), (1, 0))
        self.assertEqual((result["C"].value, result["C"].cost), (3, 2))
        self.assertEqual(sim.audit[1]["participants"], ["B"])
        self.assertNotIn("A", sim.audit[1]["raw_bids"])

    def test_zero_alpha_zero_pvalue_and_zero_budget_are_inactive(self):
        sim = DaySimulator([
            request(0, "r1", A=10, B=8, C=100),
            request(0, "r2", A=0, B=9),
        ], config({"A": 10, "B": 10, "C": 0}), audit=True)
        result = sim.step({"A": 1, "B": 0, "C": 5})
        self.assertEqual((result["A"].value, result["A"].cost), (10, 0))
        self.assertEqual(result["C"].wins, 0)
        self.assertIsNone(sim.audit[1]["winner"])

    def test_cap_bid_changes_winner_and_never_exceeds_budget(self):
        sim = DaySimulator([
            request(0, "r1", A=10, B=8),
            request(0, "r2", A=10, B=8),
        ], config({"A": 2, "B": 3}), audit=True)
        result = sim.step({"A": 1, "B": 1})
        self.assertEqual(sim.audit[0]["winner"], "B")
        self.assertEqual(sim.audit[0]["cost"], 2)
        self.assertEqual(sim.audit[1]["winner"], "A")
        self.assertEqual(sim.audit[1]["cost"], 1)
        self.assertEqual((result["A"].cost, result["B"].cost), (1, 2))

    def test_reject_unaffordable_repeats_auction(self):
        sim = DaySimulator([request(0, "r", A=10, B=8, C=6)],
                           config({"A": 1, "B": 2, "C": 10}, budget_policy="reject_unaffordable"), audit=True)
        result = sim.step({"A": 1, "B": 1, "C": 1})
        self.assertEqual(sim.audit[0]["rejected_unaffordable"], ["A", "B"])
        self.assertEqual((result["C"].wins, result["C"].cost, result["C"].value), (1, 0, 6))
        self.assertEqual(sim.balances["A"], 1)

    def test_full_budget_can_be_spent_then_player_is_inactive(self):
        sim = DaySimulator([request(0, "r1", A=3, B=2), request(0, "r2", A=100)],
                           config({"A": 2, "B": 2}, tie_break="stable"), audit=True)
        result = sim.step({"A": 1, "B": 1})
        self.assertEqual(result["A"].remaining_budget, 0)
        self.assertEqual(result["A"].wins, 1)
        self.assertIsNone(sim.audit[1]["winner"])

    def test_all_empty_slots_still_produce_72_steps(self):
        sim = DaySimulator([], config())
        for t in range(72):
            self.assertEqual(sim.observation("A").time_step, t)
            result = sim.step({"A": 1, "B": 1, "C": 1})
            self.assertEqual(result["A"].time_step, t)
            self.assertEqual(result["A"].value, 0)
        self.assertTrue(sim.done)
        self.assertEqual(len(sim.histories["A"]), 72)
        with self.assertRaises(RuntimeError):
            sim.step({"A": 1, "B": 1, "C": 1})

    def test_invalid_action_is_atomic(self):
        sim = DaySimulator([request(0, "r1", A=10, B=8)], config(), audit=True)
        for actions in (
            {"A": math.nan, "B": 1, "C": 1},
            {"A": math.inf, "B": 1, "C": 1},
            {"A": -1, "B": 1, "C": 1},
            {"A": 6, "B": 1, "C": 1},
            {"A": 1, "B": 1},
            {"A": 1, "B": 1, "C": 1, "D": 1},
        ):
            with self.assertRaises(ValueError):
                sim.step(actions)
            self.assertEqual(sim.time_step, 0)
            self.assertEqual(sim.balances["A"], 100)
            self.assertEqual(sim.histories["A"], [])
            self.assertEqual(sim.audit, [])

    def test_product_overflow_is_atomic(self):
        sim = DaySimulator([request(0, "small", A=1, B=1), request(0, "huge", A=1e308)], config())
        with self.assertRaises(ValueError):
            sim.step({"A": 5, "B": 1, "C": 1})
        self.assertEqual(sim.time_step, 0)
        self.assertEqual(sim.balances["A"], 100)

    def test_ties_are_seeded_reproducible_and_pay_tied_bid(self):
        requests = [request(0, f"r{i}", A=1, B=1) for i in range(12)]
        first = DaySimulator(requests, config(seed=321), audit=True)
        second = DaySimulator(requests, config(seed=321), audit=True)
        other = DaySimulator(requests, config(seed=123), audit=True)
        for sim in (first, second, other):
            sim.step({"A": 1, "B": 1, "C": 1})
            self.assertTrue(all(r["cost"] == 1 for r in sim.audit))
        self.assertEqual(first.audit, second.audit)
        self.assertNotEqual([r["winner"] for r in first.audit], [r["winner"] for r in other.audit])

    def test_unrelated_auction_does_not_shift_tie_randomness(self):
        requests = [request(0, "before", A=1, B=1), request(1, "target", A=1, B=1)]
        first = DaySimulator(requests, config(seed=41), audit=True)
        second = DaySimulator(requests, config(seed=41), audit=True)
        first.step({"A": 1, "B": 1, "C": 1})
        second.step({"A": 0, "B": 0, "C": 0})
        for sim in (first, second):
            sim.step({"A": 1, "B": 1, "C": 1})
        self.assertEqual(first.audit[1]["winner"], second.audit[1]["winner"])

    def test_randomized_second_price_matches_simple_oracle(self):
        rng = random.Random(14)
        for _ in range(40):
            values = {p: rng.random() * 10 for p in ("A", "B", "C")}
            actions = {p: rng.random() * 3 for p in values}
            bids = sorted((values[p] * actions[p], p) for p in values)
            expected_winner, expected_cost = bids[-1][1], bids[-2][0]
            sim = DaySimulator([request(0, "r", **values)], config())
            results = sim.step(actions)
            self.assertEqual(results[expected_winner].wins, 1)
            self.assertAlmostEqual(results[expected_winner].cost, expected_cost)
            self.assertEqual(sum(f.wins for f in results.values()), 1)

    def test_budget_conservation_across_a_day(self):
        rng = random.Random(7)
        requests = [request(t, f"{t}:{i}", A=rng.random()*10, B=rng.random()*10)
                    for t in range(72) for i in range(4)]
        for mode in ("cap_bid", "reject_unaffordable"):
            sim = DaySimulator(requests, config({"A": 50, "B": 70}, budget_policy=mode))
            while not sim.done:
                sim.step({"A": 1.2, "B": 0.9})
            for player, budget in sim.config.budgets.items():
                cost = sum(f.cost for f in sim.histories[player])
                self.assertLessEqual(cost, budget + 1e-10)
                self.assertAlmostEqual(cost + sim.balances[player], budget)

    def test_direct_request_validation(self):
        cases = [
            [request(0, "same", A=1), request(1, "same", A=2)],
            [request(1, "r1", A=1), request(0, "r2", A=1)],
            [request(72, "r1", A=1)],
            [request(0, "r1", A=math.nan)],
            [request(0, "r1", A=-1)],
            [request(0, "r1", D=1)],
            [Request(0, "r1", (Participation("A", "type", 1), Participation("A", "type", 2)))],
            [Request(0, "r1", (Participation("A", "wrong", 1),))],
        ]
        for requests in cases:
            with self.subTest(requests=requests), self.assertRaises(ValueError):
                DaySimulator(requests, config())


class InformationBoundaryTests(unittest.TestCase):
    def test_future_data_and_other_players_do_not_change_initial_observation(self):
        first = DaySimulator([request(0, "secret1", A=10, B=8)], config())
        second = DaySimulator([request(30, "secret2", A=1000, C=2)],
                              config({"A": 100, "B": 999, "C": 1}))
        self.assertEqual(first.observation("A"), second.observation("A"))
        obs = asdict(first.observation("A"))
        self.assertEqual(set(obs), {"time_step", "total_steps", "player_id", "player_type",
                                    "initial_budget", "remaining_budget", "alpha_max", "history"})
        self.assertEqual(obs["history"], ())

    def test_history_contains_only_completed_own_feedback_and_is_immutable(self):
        sim = DaySimulator([request(0, "r", A=10, B=8)], config())
        before = sim.observation("A")
        sim.step({"A": 1, "B": 1, "C": 1})
        after = sim.observation("A")
        self.assertEqual(before.history, ())
        self.assertEqual(len(after.history), 1)
        self.assertEqual(after.history[0].value, 10)
        self.assertEqual(after.history[0].time_step, 0)
        with self.assertRaises(FrozenInstanceError):
            after.remaining_budget = 0
        with self.assertRaises(FrozenInstanceError):
            after.history[0].value = 999

    def test_config_is_copied_at_construction(self):
        cfg = config()
        sim = DaySimulator([], cfg)
        cfg.budgets["A"] = 999
        self.assertEqual(sim.observation("A").initial_budget, 100)

    def test_audit_is_opt_in(self):
        sim = DaySimulator([request(0, "r", A=1)], config())
        sim.step({"A": 1, "B": 1, "C": 1})
        self.assertEqual(sim.audit, [])


class PolicyTests(unittest.TestCase):
    def test_pacing_increases_after_underspend(self):
        sim = DaySimulator([], config())
        policy = PacingPolicy()
        initial = policy.bidding(sim.observation("A"))
        for _ in range(10):
            feedback = sim.step({"A": 1, "B": 1, "C": 1})
            policy.update(feedback["A"])
        self.assertGreater(policy.bidding(sim.observation("A")), initial)
        self.assertEqual(policy.bidding(sim.observation("A")), policy.bidding(sim.observation("A")))

    def test_pacing_decreases_after_overspend(self):
        sim = DaySimulator([request(0, "r", A=100, B=90)], config())
        policy = PacingPolicy()
        initial = policy.bidding(sim.observation("A"))
        sim.step({"A": 1, "B": 1, "C": 1})
        self.assertLess(policy.bidding(sim.observation("A")), initial)

    def test_fixed_policy_bounds_and_zero_budget(self):
        sim = DaySimulator([], config({"A": 10, "B": 0}))
        policy = FixedPolicy(10)
        self.assertEqual(policy.bidding(sim.observation("A")), 5)
        self.assertEqual(policy.bidding(sim.observation("B")), 0)

    def test_negative_or_nonfinite_policy_parameters_rejected(self):
        for bad in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                FixedPolicy(bad)
            with self.assertRaises(ValueError):
                PacingPolicy(ki=bad)


if __name__ == "__main__":
    unittest.main()
