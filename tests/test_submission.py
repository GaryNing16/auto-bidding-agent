import unittest

from autobid.submission import Agent


class RecordingPolicy:
    def __init__(self):
        self.observations = []
        self.feedback = []

    def reset(self, observation):
        self.observations = []
        self.feedback = []

    def bidding(self, observation):
        self.observations.append(observation)
        return 1.5

    def update(self, feedback):
        self.feedback.append(feedback)


class SubmissionTests(unittest.TestCase):
    def make_agent(self, **kwargs):
        return Agent(player_id="A", player_type="apps", budget=10, **kwargs)

    def test_bidding_update_then_day_reset(self):
        policy = RecordingPolicy()
        agent = self.make_agent(policy=policy, total_steps=2)
        self.assertEqual(agent.bidding(), 1.5)
        self.assertEqual(agent.bidding(), 1.5)
        self.assertEqual(len(policy.observations), 1)
        agent.update(wins=1, cost=3, value=4)
        self.assertEqual(agent.observation().remaining_budget, 7)
        self.assertEqual(agent.observation().history[0].alpha, 1.5)
        agent.bidding()
        agent.update(wins=0, cost=0, value=0)
        with self.assertRaises(RuntimeError):
            agent.bidding()
        agent.reset()
        self.assertEqual(agent.observation().history, ())
        self.assertEqual(agent.remaining_budget, 10)

    def test_feedback_cannot_arrive_before_action(self):
        with self.assertRaises(RuntimeError):
            self.make_agent().update(wins=1, cost=1, value=1)

    def test_invalid_feedback_does_not_advance_state(self):
        agent = self.make_agent()
        agent.bidding()
        for feedback in (
            dict(wins=1, cost=11, value=1), dict(wins=1, cost=0, value=float("nan")),
            dict(wins=0, cost=1, value=0), dict(wins=1.5, cost=0, value=0),
        ):
            with self.assertRaises(ValueError):
                agent.update(**feedback)
            self.assertEqual(len(agent.history), 0)
            self.assertEqual(agent.remaining_budget, 10)

    def test_zero_budget_never_bids(self):
        agent = Agent(player_id="A", player_type="apps", budget=0)
        self.assertEqual(agent.bidding(), 0)
        agent.update(wins=0, cost=0, value=0)
        self.assertEqual(agent.bidding(), 0)

    def test_only_own_state_is_exposed(self):
        obs = self.make_agent().observation()
        self.assertEqual(set(obs.__dataclass_fields__), {
            "time_step", "total_steps", "player_id", "player_type", "initial_budget",
            "remaining_budget", "alpha_max", "history",
        })


if __name__ == "__main__":
    unittest.main()
