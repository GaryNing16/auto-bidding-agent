"""Bounded own-budget diagnostics, live counterfactuals and market audits.

Neither these counterfactual returns nor evaluator-only distribution statistics
are policy inputs. A critic's finite difference is an estimate, not a causal
action attribution. Redistributed-return critics are explicitly distinguished.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from harmony_baselines.data import sha256_file
from harmony_baselines.five_scenarios.common import OUT, TRAIN_DAYS, cached_day, dump, read
from harmony_bidding.auction import Money, validate_alpha
from .buffer import Trajectory
from .env import BiddingEnv
from .markets import bootstrap_market, distribution_summary, stress_market
from .models import sample_action
from .policy import load_payload, model_from_payload
from .redistribution import ReturnPredictor, predict_rewards


def budget_observation(observation, delta):
    """Add funds to original/current balances; keep all realized past spends fixed.

    This defines a consistent funding intervention, not an isolated edit of a
    normalized remaining-budget feature. Encoding updates every cost fraction,
    remaining fraction, pacing error and initial-budget feature automatically.
    """
    money = Money(6)
    units = money.budget_units(validate_alpha(delta))
    if units <= 0:
        raise ValueError("finite difference requires a positive budget increment")
    add = lambda value: money.amount(money.units(value) + units)
    own = replace(observation.own_config, initial_budget=add(observation.own_config.initial_budget))
    history = tuple(replace(f, remaining_budget=add(f.remaining_budget)) for f in observation.history)
    return replace(observation, own_config=own, remaining_budget=add(observation.remaining_budget), history=history)


@torch.no_grad()
def critic_shadow(model, encoder, observation, delta, reward_scale=1., redistributed=False):
    if reward_scale <= 0 or not np.isfinite(reward_scale):
        raise ValueError("reward scale must be finite and positive")
    changed = budget_observation(observation, delta)
    actual_delta = changed.own_config.initial_budget - observation.own_config.initial_budget
    if not np.isfinite(actual_delta) or actual_delta <= 0:
        raise ValueError("budget increment is not representable at this budget magnitude")
    state = np.stack([encoder.encode(observation), encoder.encode(changed)])
    device = next(model.parameters()).device
    values = model.value(torch.as_tensor(state, dtype=torch.float32, device=device)).cpu().numpy() * reward_scale
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite critic budget diagnostic")
    return dict(delta_budget=actual_delta, critic_original=float(values[0]), critic_extra_budget=float(values[1]),
        marginal_estimate=float((values[1] - values[0]) / actual_delta),
        quantity="redistributed_return_sensitivity" if redistributed else "raw_future_value_sensitivity",
        raw_value_interpretation=not redistributed,
        intervention="add original/current funding; hold past spend/value/actions fixed")


def _continuation(env, model, encoder):
    before = env.engine.cumulative_value[env.controlled_player]
    steps = 0
    while env.time_slot < 72:
        action = sample_action(model, encoder.encode(env.observation()), env.base_alpha(), deterministic=True)
        env.step(action["alpha"])
        steps += 1
    return dict(future_value=env.engine.cumulative_value[env.controlled_player] - before,
                total_value=env.engine.cumulative_value[env.controlled_player], steps=steps,
                final_remaining_budget=env.observation().remaining_budget)


def counterfactual_budget(env, model, encoder, delta):
    """Exactly two bounded continuations; both retain live opponent/controller state."""
    delta = validate_alpha(delta)
    if delta <= 0:
        raise ValueError("counterfactual requires positive extra budget")
    original, extra = env.snapshot(), env.with_extra_budget(delta)
    baseline = _continuation(original, model, encoder)
    funded = _continuation(extra, model, encoder)
    return dict(time_slot=env.time_slot, delta_budget=delta, baseline=baseline, extra_budget=funded,
                empirical_marginal_value=(funded["future_value"] - baseline["future_value"]) / delta,
                common_market=True, request_keyed_tie_seed=env.seed, dynamic_opponents=True,
                interpretation="single simulated funding intervention; no statistical or causal generalization")


def audit_markets(output, day_id="period-7", seed=20261028):
    if day_id not in TRAIN_DAYS:
        raise ValueError("augmentation audit only accepts declared training periods")
    start = perf_counter()
    output = Path(output)
    day = cached_day(day_id)
    records = {"historical": distribution_summary(day)}
    cases = (("bootstrap", lambda: bootstrap_market(day, seed)),
             ("request_blocks", lambda: bootstrap_market(day, seed, method="request_blocks")),
             ("front_loaded", lambda: stress_market(day, seed, "front_loaded")),
             ("late_high_value", lambda: stress_market(day, seed, "late_high_value")),
             ("traffic_tight", lambda: stress_market(day, seed, "traffic_tight")),
             ("traffic_scaled", lambda: stress_market(day, seed, "traffic_scaled")))
    for label, build in cases:
        market = build()
        summary = distribution_summary(market)
        summary["generation"] = market.source
        # Matrix budgets remain the native source roster; the environment applies
        # traffic scaling. Record the corresponding effective experiment budget.
        multiplier = (market.source.get("traffic_multiplier", 1.)
                      if market.source.get("traffic_budget_mode") == "proportional" else 1.)
        summary["effective_budget_scale"] = multiplier * market.source.get("budget_scale", 1.)
        summary["effective_budget_per_request"] = (summary["budget_per_request"] * summary["effective_budget_scale"]
                                                    if summary["budget_per_request"] is not None else None)
        records[label] = summary
        del market
    result = dict(day_id=day_id, seed=seed, training_only=True, records=records,
        elapsed_seconds=perf_counter() - start,
        interpretation="Exact sampled-market descriptors; reproduction does not establish distributional equivalence")
    dump(output / "market_distribution_audit.json", result)
    _market_plot(records, output)
    return result


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(figure, output, name):
    output.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output / (name + ".png"), dpi=150)
    figure.savefig(output / (name + ".svg"))


def _market_plot(records, output):
    plt = _plotting()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for label in ("historical", "bootstrap", "front_loaded", "late_high_value"):
        row = records[label]
        axes[0].plot(np.arange(1, 73), row["per_slot_request_counts"], label=label)
        axes[1].plot(list(row["per_player_opportunities"].values()), label=label)
    axes[0].set(xlabel="Time slot", ylabel="Request groups")
    axes[1].set(xlabel="Player index", ylabel="Valid opportunities")
    axes[0].legend(fontsize=7)
    names = list(records)
    axes[2].bar(names, [records[n].get("effective_budget_per_request", records[n]["budget_per_request"]) for n in names])
    axes[2].set(ylabel="Total budget / request", title="Traffic and budget experiments")
    axes[2].tick_params(axis="x", labelrotation=70, labelsize=7)
    _save(fig, output, "market_distributions")
    plt.close(fig)


def run_diagnostics(checkpoint, output, day_id="period-7", player="0", scenario="Mixed2", seed=20261028,
                    counterfactual_slot=36, delta_fraction=.01, day=None, opponents=None):
    """One trajectory plus two half-day continuations; no model update or selection."""
    if not 0 <= counterfactual_slot < 72 or not 0 < delta_fraction <= 1.:
        raise ValueError("invalid bounded diagnostic configuration")
    torch.set_num_threads(1)
    output = Path(output)
    payload = load_payload(checkpoint)
    model, encoder = model_from_payload(payload)
    cfg = payload["config"]
    day = cached_day(day_id) if day is None else day
    opponents = read(cfg.get("opponents_path", OUT / "opponents_frozen.json")) if opponents is None else opponents
    redistributed = cfg["variant"] in ("E3", "E4", "E6")
    scale = cfg.get("ppo", {}).get("reward_scale", 1.)
    env = BiddingEnv(day, str(player), opponents, scenario, seed,
                     residual_spec=cfg.get("residual_spec") if payload["model_config"]["residual"] else None)
    obs, info = env.reset()
    delta = max(round(obs.own_config.initial_budget * delta_fraction, 6), .000001)
    arrays = {name: [] for name in ("states", "actions", "latents", "log_probs", "values", "rewards",
                                   "spends", "next_states", "dones")}
    rows, counterfactual = [], None
    for t in range(72):
        state = encoder.encode(obs)
        sensitivity = critic_shadow(model, encoder, obs, delta, scale, redistributed)
        if t == counterfactual_slot:
            counterfactual = counterfactual_budget(env, model, encoder, delta)
            counterfactual["critic"] = sensitivity
        action = sample_action(model, state, env.base_alpha(), deterministic=True)
        nxt, reward, done, _, feedback = env.step(action["alpha"])
        cost = feedback["feedback"].cost
        entries = dict(states=state, actions=action["alpha"], latents=action["latent"], log_probs=action["log_prob"],
                       values=action["value"], rewards=reward, spends=cost, next_states=encoder.encode(nxt), dones=done)
        for key, value in entries.items():
            arrays[key].append(value)
        rows.append({**env.diagnostics[-1], **sensitivity,
                     "estimated_spend_opportunity_cost": sensitivity["marginal_estimate"] * cost
                     if not redistributed else None})
        obs = nxt
    trajectory = Trajectory(**{k: np.asarray(v) for k, v in arrays.items()}, metadata={**env.metadata, **info})
    predictor_result = None
    if redistributed and payload.get("lstm_version", 0) > 0:
        predictor = ReturnPredictor(**payload["lstm_config"])
        predictor.load_state_dict(payload["lstm_state"])
        predictor.eval()
        rewards, predictions = predict_rewards(predictor, trajectory, scale)
        predictor_result = dict(redistributed_rewards=rewards.tolist(), prefix_predictions=predictions.tolist(),
            return_conservation_error=float(rewards.sum() - trajectory.rewards.sum()),
            lstm_version=payload["lstm_version"],
            prefix_errors=(predictions - trajectory.episode_return).tolist(),
            interpretation="statistical prefix-return prediction; differences are not causal action contributions")
    result = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256_file(checkpoint),
        config=cfg, metrics=env.metrics(), rows=rows, counterfactual=counterfactual,
        redistribution=predictor_result, raw_rewards=trajectory.rewards.tolist(),
        policy_frozen=True, diagnostics_only=True,
        caveats=["One role/market/seed cannot demonstrate stable opportunity-cost accuracy.",
                 "Budget perturbation changes both original and remaining funds while holding realized history fixed.",
                 "A redistributed-return critic is not an estimator of raw remaining value."])
    dump(output / "diagnostics.json", result)
    _diagnostic_plot(result, output)
    return result


def _diagnostic_plot(result, output):
    plt = _plotting()
    rows = result["rows"]
    t = np.arange(1, 73)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, name, label in zip(axes.flat,
        ("alpha", "cumulative_cost", "cumulative_value", "remaining_budget", "marginal_estimate", "value"),
        ("Alpha", "Cumulative spend", "Cumulative value", "Remaining budget", "Critic budget sensitivity", "Slot value")):
        ax.plot(t, [r[name] for r in rows])
        ax.set(xlabel="Time slot", ylabel=label)
    title = result["metrics"]["scenario"] + " / player " + result["metrics"]["player_id"]
    if not rows[0]["raw_value_interpretation"]:
        title += " (sensitivity is redistributed return, not raw value)"
    fig.suptitle(title, fontsize=10)
    _save(fig, output, "budget_behavior")
    plt.close(fig)
    if result["redistribution"]:
        predictor = result["redistribution"]
        raw = np.asarray(result["raw_rewards"])
        redistributed = np.asarray(predictor["redistributed_rewards"])
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        axes[0].plot(t, raw, label="Raw slot value")
        axes[0].plot(t, redistributed, label="Redistributed reward", alpha=.8)
        axes[1].plot(t, predictor["prefix_predictions"], label="Causal prefix prediction")
        axes[1].axhline(raw.sum(), color="black", linestyle="--", label="Realized total return")
        axes[2].plot(t, raw.cumsum(), label="Cumulative raw")
        axes[2].plot(t, redistributed.cumsum(), label="Cumulative redistributed")
        for ax in axes:
            ax.set_xlabel("Time slot")
            ax.legend(fontsize=8)
        _save(fig, output, "reward_redistribution")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--day", default="period-7")
    parser.add_argument("--player", default="0")
    parser.add_argument("--scenario", default="Mixed2")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--audit-markets", action="store_true")
    args = parser.parse_args(argv)
    if args.audit_markets:
        result = audit_markets(args.output, args.day, args.seed)
        print(json.dumps({"audit": str(Path(args.output) / "market_distribution_audit.json"),
                          "elapsed_seconds": result["elapsed_seconds"]}))
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --audit-markets is set")
        result = run_diagnostics(args.checkpoint, args.output, args.day, args.player, args.scenario, args.seed)
        print(json.dumps({"diagnostics": str(Path(args.output) / "diagnostics.json"),
                          "total_value": result["metrics"]["total_value"]}))


if __name__ == "__main__":
    main()
