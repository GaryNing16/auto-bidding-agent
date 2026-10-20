from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from autobid.types import AuctionConfig, Feedback, Observation, Participation, Request


HAS_GYM = importlib.util.find_spec("gymnasium") is not None and importlib.util.find_spec("numpy") is not None
HAS_SB3 = HAS_GYM and importlib.util.find_spec("stable_baselines3") is not None


def fixture():
    config = AuctionConfig(steps=3, alpha_max=3.0, budgets={"A": 20.0, "B": 20.0},
                           player_types={"A": "apps", "B": "games"}, seed=7)
    requests = [Request(t, f"r{t}", (Participation("A", "apps", 2.0), Participation("B", "games", 1.0)))
                for t in range(3)]
    return requests, config


@unittest.skipUnless(HAS_GYM, "Install .[rl] to test Gymnasium")
class EnvironmentTests(unittest.TestCase):
    def test_gym_contract_and_natural_termination(self):
        import numpy as np
        from gymnasium.utils.env_checker import check_env
        from autobid.rl import AutobidEnv
        requests, config = fixture()
        env = AutobidEnv(requests, config, "A")
        check_env(env, skip_render_check=True)
        obs, info = env.reset(seed=5)
        self.assertEqual(info, {})
        self.assertTrue(env.observation_space.contains(obs))
        total_reward = 0
        for t in range(3):
            obs, reward, terminated, truncated, info = env.step(np.array([2.0], dtype=np.float32))
            total_reward += reward
            self.assertEqual(terminated, t == 2)
            self.assertFalse(truncated)
            self.assertEqual(set(info), {"own_feedback"})
            self.assertEqual(info["own_feedback"]["value"], reward)
            self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(total_reward, 6)
        with self.assertRaises(RuntimeError):
            env.step([1.0])

    def test_future_and_rival_changes_cannot_change_initial_features(self):
        import numpy as np
        from autobid.rl import AutobidEnv
        requests, config = fixture()
        altered = [Request(2, "secret", (Participation("B", "games", 999),))]
        first, _ = AutobidEnv(requests, config, "A").reset(seed=7)
        second, _ = AutobidEnv(altered, config, "A").reset(seed=7)
        np.testing.assert_array_equal(first, second)

    def test_history_mask_and_finite_features(self):
        import numpy as np
        from autobid.rl import encode_observation, FEATURE_NAMES
        obs = Observation(0, 72, "A", "apps", 0.0, 0.0, 5.0, ())
        features = encode_observation(obs)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[FEATURE_NAMES.index("lag1_present")], 0)
        obs = replace(obs, time_step=1, history=(Feedback(0, 0, 0, 0, 0, 0),))
        features = encode_observation(obs)
        self.assertEqual(features[FEATURE_NAMES.index("lag1_present")], 1)
        self.assertEqual(features[FEATURE_NAMES.index("lag2_present")], 0)


@unittest.skipUnless(HAS_SB3, "Install .[rl] to test PPO training and save/load")
class PPOTests(unittest.TestCase):
    def test_training_saved_model_loads_in_new_process(self):
        from autobid.rl import train_ppo, PPOPolicy
        from autobid.submission import Agent
        requests, config = fixture()
        with tempfile.TemporaryDirectory() as directory:
            model = train_ppo(requests, config, Path(directory) / "smoke", player_id="A", total_timesteps=12)
            self.assertTrue(model.is_file())
            metadata = json.loads(model.with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["gamma"], 1.0)
            policy = PPOPolicy(model)
            agent = Agent(player_id="A", player_type="apps", budget=20, total_steps=3, alpha_max=3, policy=policy)
            expected = agent.bidding()
            code = (
                "import json,sys; from autobid.submission import Agent; "
                "a=Agent(player_id='A',player_type='apps',budget=20,total_steps=3,alpha_max=3,model_path=sys.argv[1]); "
                "print(json.dumps(a.bidding()))"
            )
            result = subprocess.run([sys.executable, "-c", code, str(model)], check=True, capture_output=True, text=True)
            self.assertAlmostEqual(float(result.stdout.strip()), expected)
            with self.assertRaises(ValueError):
                Agent(player_id="A", player_type="apps", budget=20, total_steps=72, alpha_max=3, policy=policy)


if __name__ == "__main__":
    unittest.main()
