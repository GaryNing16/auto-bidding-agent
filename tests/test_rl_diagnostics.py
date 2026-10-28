from dataclasses import asdict
import numpy as np
import pytest
import torch

from harmony_bidding.agent_api import observation_from_dict
from harmony_rl.diagnostics import budget_observation, critic_shadow, counterfactual_budget, run_diagnostics, audit_markets
from harmony_rl.env import BiddingEnv
from harmony_rl.markets import synthetic_day
from harmony_rl.models import ActorCritic, StateEncoder
from harmony_rl.redistribution import ReturnPredictor


def setup():
    day = synthetic_day(3, 3, 2)
    opponents = {k: {"kind": "constant", "alpha": 2.} for k in ("fixed", "pid", "auctionnet_pid")}
    env = BiddingEnv(day, "0", opponents)
    env.reset()
    encoder = StateEncoder([a.advertiser_id for a in day.advertisers], ["0"])
    model = ActorCritic(encoder.state_dim, hidden=8, initial_alpha=2.)
    return day, opponents, env, model, encoder


def test_budget_perturbation_updates_all_derived_fields_without_changing_realized_history():
    _, _, env, _, encoder = setup()
    for _ in range(3):
        env.step(10.)
    obs = env.observation()
    changed = budget_observation(obs, .1)
    assert observation_from_dict(asdict(changed)) == changed
    assert changed.own_config.initial_budget == pytest.approx(obs.own_config.initial_budget + .1)
    for old, new in zip(obs.history, changed.history):
        assert old.cost == new.cost and old.cumulative_cost == new.cumulative_cost and old.alpha == new.alpha
        assert changed.own_config.initial_budget - new.remaining_budget == pytest.approx(new.cumulative_cost)
    assert changed.own_config.initial_budget - changed.remaining_budget == pytest.approx(obs.history[-1].cumulative_cost)
    encoded = encoder.encode(changed)
    assert encoded[1] == pytest.approx(changed.remaining_budget / changed.own_config.initial_budget)
    assert encoded[3] == pytest.approx(changed.time_slot / 72 - (1 - encoded[1]), abs=1e-7)
    assert encoded[5] == pytest.approx(changed.history[-1].cost / changed.own_config.initial_budget)
    np.testing.assert_allclose(encoded[11], sum(f.cost for f in changed.history[-3:]) / 3 / changed.own_config.initial_budget)


def test_known_budget_value_function_has_correct_shadow_scale_and_lstm_label():
    _, _, env, _, encoder = setup()
    class KnownValue(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.))
        def value(self, states):
            return states[:, 1] * torch.expm1(states[:, 2] * 10) + self.anchor
    result = critic_shadow(KnownValue(), encoder, env.observation(), .1, reward_scale=2.)
    assert result["marginal_estimate"] == pytest.approx(2., abs=1e-5)
    assert result["raw_value_interpretation"]
    result = critic_shadow(KnownValue(), encoder, env.observation(), .1, redistributed=True)
    assert not result["raw_value_interpretation"] and result["quantity"] == "redistributed_return_sensitivity"


def test_counterfactual_does_not_modify_actual_env_and_runs_both_live_continuations():
    _, _, env, model, encoder = setup()
    for _ in range(36):
        env.step(2.)
    before = env.observation()
    result = counterfactual_budget(env, model, encoder, .1)
    assert env.observation() == before and env.time_slot == 36
    assert result["baseline"]["steps"] == result["extra_budget"]["steps"] == 36
    assert result["dynamic_opponents"] and result["common_market"]


def test_diagnostic_cli_backend_writes_real_return_conserving_artifacts(tmp_path):
    day, opponents, env, model, encoder = setup()
    predictor = ReturnPredictor(encoder.state_dim + 5, hidden=8)
    checkpoint = tmp_path / "test.pt"
    payload = dict(format="harmony-ppo-lstm-v1", model_config=model.config, model_state=model.state_dict(),
        encoder=encoder.to_dict(), config={"variant": "E3", "ppo": {"reward_scale": 1.}},
        lstm_config=predictor.config, lstm_state=predictor.state_dict(), lstm_version=1)
    torch.save(payload, checkpoint)
    result = run_diagnostics(checkpoint, tmp_path / "diagnostics", day=day, opponents=opponents)
    assert result["metrics"]["completed_slots"] == len(result["rows"]) == 72
    assert abs(result["redistribution"]["return_conservation_error"]) < 1e-7
    assert all(row["estimated_spend_opportunity_cost"] is None for row in result["rows"])
    assert (tmp_path / "diagnostics" / "diagnostics.json").is_file()
    assert (tmp_path / "diagnostics" / "reward_redistribution.png").is_file()
    assert (tmp_path / "diagnostics" / "budget_behavior.svg").is_file()


def test_augmentation_audit_rejects_test_period_without_loading(tmp_path):
    with pytest.raises(ValueError, match="training"):
        audit_markets(tmp_path, "period-24")


def test_zero_budget_difference_is_rejected_before_simulation():
    _, _, env, model, encoder = setup()
    with pytest.raises(ValueError, match="positive"):
        critic_shadow(model, encoder, env.observation(), 0.)
    with pytest.raises(ValueError, match="positive"):
        counterfactual_budget(env, model, encoder, 0.)
    assert env.time_slot == 0
