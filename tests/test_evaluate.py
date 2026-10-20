"""Independent evaluation checks using hand-calculated request streams."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

from autobid.data import write_requests
from autobid.evaluate import arena, evaluate, run_episode
from autobid.types import AuctionConfig, Participation, Request


def req(rid, **values):
    return Request(0, rid, tuple(Participation(player, "type", value) for player, value in values.items()))


def cfg(players=("A", "B"), steps=1):
    return AuctionConfig(budgets={p: 1000.0 for p in players},
                         player_types={p: "type" for p in players}, steps=steps)


class ScoreTests(unittest.TestCase):
    def test_score_is_mean_of_per_episode_ratios_not_ratio_of_totals(self):
        # At slot zero both local pacing policies bid alpha=1.
        # Day 1: pacing gets A value 1; fixed:2 gets A value 2.
        # Day 2: both get A value 100. Correct mean ratio is (2+1)/2.
        days = [[req("d1-r1", A=1, B=0.5), req("d1-r2", A=1, B=1.5)],
                [req("d2-r1", A=100, B=50)]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = {}
            for i, requests in enumerate(days):
                path = root / f"day-{i}.csv"
                write_requests(requests, path)
                data[path] = requests
            result = evaluate(data, cfg(), root / "out", strategies=["fixed:2"], roles=["A"])
        stats = result["aggregates"]["fixed:2"]
        self.assertEqual(stats["episodes"], 2)
        self.assertEqual(stats["mean_value"], 51)
        self.assertEqual(stats["mean_episode_ratio_to_local_pacing"], 1.5)
        self.assertNotAlmostEqual(stats["mean_episode_ratio_to_local_pacing"], 102 / 101)
        self.assertEqual([r["local_pacing_value"] for r in result["episodes"]], [1, 100])

    def test_zero_baseline_is_explicit_null_and_counted(self):
        requests = [req("r1", A=1, B=1.5)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "day.csv"
            write_requests(requests, source)
            result = evaluate({source: requests}, cfg(), root / "out", strategies=["fixed:2"], roles=["A"])
            self.assertTrue((root / "out" / "summary.json").is_file())
        row = result["episodes"][0]
        self.assertEqual(row["value"], 1)
        self.assertEqual(row["local_pacing_value"], 0)
        self.assertIsNone(row["relative_to_local_pacing"])
        self.assertIsNone(result["aggregates"]["fixed:2"]["mean_episode_ratio_to_local_pacing"])
        self.assertEqual(result["aggregates"]["fixed:2"]["undefined_zero_baseline_episodes"], 1)
        self.assertEqual(result["baseline"], "local_pacing_NOT_official_PID")

    def test_pacing_vs_itself_has_unit_ratio_across_roles_and_seeds(self):
        requests = [req("solo-a", A=1), req("solo-b", B=2), req("solo-c", C=3),
                    req("tie", A=5, B=5, C=5)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "day.csv"
            write_requests(requests, source)
            result = evaluate({source: requests}, cfg(("A", "B", "C")), root / "out",
                              strategies=["pacing"], seeds=[3, 8])
        self.assertEqual(len(result["episodes"]), 6)
        self.assertEqual({r["player_id"] for r in result["episodes"]}, {"A", "B", "C"})
        self.assertTrue(all(r["relative_to_local_pacing"] == 1 for r in result["episodes"]))

    def test_arena_rotates_every_strategy_through_every_role(self):
        config = cfg(("A", "B", "C"))
        with tempfile.TemporaryDirectory() as temporary:
            rotations = arena([req("r", A=10, B=8, C=6)], config, temporary,
                              ["fixed:0", "fixed:1", "fixed:2"])
        for player in config.budgets:
            self.assertEqual({r["assignments"][player] for r in rotations},
                             {"fixed:0", "fixed:1", "fixed:2"})
        self.assertEqual(len(rotations), 3)


class RunnerBoundaryTests(unittest.TestCase):
    def test_all_actions_precede_feedback_and_policies_receive_only_own_history(self):
        events, observations = [], {"A": [], "B": []}
        allowed_fields = {"time_step", "total_steps", "player_id", "player_type",
                          "initial_budget", "remaining_budget", "alpha_max", "history"}

        class Spy:
            def __init__(self, player):
                self.player = player

            def reset(self, observation):
                events.append(("reset", self.player, observation.time_step))
                assert observation.player_id == self.player

            def bidding(self, observation):
                events.append(("bid", self.player, observation.time_step))
                assert set(asdict(observation)) == allowed_fields
                assert observation.player_id == self.player
                assert len(observation.history) == observation.time_step
                assert all(f.time_step < observation.time_step for f in observation.history)
                observations[self.player].append(observation)
                return 1.0

            def update(self, feedback):
                events.append(("update", self.player, feedback.time_step))
                assert set(asdict(feedback)) == {"time_step", "alpha", "wins", "cost", "value", "remaining_budget"}

        requests = [req("first", A=10, B=8)]
        result = run_episode(requests, cfg(steps=2), {p: Spy(p) for p in ("A", "B")}, audit=True)
        for time_step in range(2):
            relevant = [e[0] for e in events if e[0] != "reset" and e[2] == time_step]
            self.assertEqual(relevant, ["bid", "bid", "update", "update"])
        self.assertEqual(observations["A"][1].history[0].value, 10)
        self.assertEqual(observations["B"][1].history[0].value, 0)
        self.assertEqual(result["players"]["A"]["cost"], 8)
        self.assertEqual(len(result["steps"]), 4)
        self.assertEqual(result["audit"][0]["winner"], "A")


if __name__ == "__main__":
    unittest.main()
