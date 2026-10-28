"""Classical policies: parameters are fitted on training/validation only.

FixedAlpha is a local scalar baseline, NOT upstream Abid (which is quadratic).
OnlineLP reuses the upstream inference lookup on a newly fitted causal table;
its CPA variable is only a numerical fallback, with no CPA reward/constraint.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from harmony_bidding.agent_api import Agent
from harmony_bidding.agents import ConstantAlphaAgent, LocalPIDAgent


class FixedAlphaAgent(ConstantAlphaAgent):
    def __init__(self, alpha=40.0, player_alphas=None, multiplier=1.0):
        super().__init__(alpha)
        self.base_alpha = alpha
        self.player_alphas = dict(player_alphas or {})
        self.multiplier = multiplier

    def reset(self, own_config, public_config):
        super().reset(own_config, public_config)
        self.configured_alpha = self.multiplier * self.player_alphas.get(own_config.advertiser_id, self.base_alpha)


class ScaledPIDAgent(LocalPIDAgent):
    def __init__(self, initial_alpha=40., kp=1.5, ki=.1, kd=.15, player_alphas=None, multiplier=1.):
        self.player_alphas = dict(player_alphas or {})
        self.relative_gains = (kp, ki, kd)
        self.base_alpha = initial_alpha
        self.multiplier = multiplier
        super().__init__(kp, ki, kd, initial_alpha)

    def reset(self, own_config, public_config):
        self.initial_alpha = self.multiplier * self.player_alphas.get(own_config.advertiser_id, self.base_alpha)
        self.kp, self.ki, self.kd = [g * self.initial_alpha for g in self.relative_gains]
        super().reset(own_config, public_config)


class OnlineLPAgent(Agent):
    """Upstream first-over-budget lookup, using normalized empirical cost curves.

    Tables use new legal complete-episode rollouts, aggregated by training type,
    alpha and time. This replaces incompatible original leastWinningCost data.
    Calling the upstream vector interface with [1] extracts its scalar alpha;
    the dummy vector contains no live pvalue information.
    """
    def __init__(self, table_path, multiplier=1.0, fallback_alpha=400.0):
        super().__init__()
        from simul_bidding_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy
        self.lookup = OnlineLpBiddingStrategy.__new__(OnlineLpBiddingStrategy)
        self.lookup.model = pd.read_csv(table_path)
        self.lookup.cpa = float(fallback_alpha)
        self.multiplier = float(multiplier)

    def reset(self, own_config, public_config):
        self.own_config, self.public_config = own_config, public_config
        self.lookup.category = int(own_config.advertiser_type)

    def bidding(self, observation):
        self.lookup.remaining_budget = observation.remaining_budget / max(self.own_config.initial_budget, 1e-30)
        bid = self.lookup.bidding(observation.time_slot, np.ones(1), None, None, None, None, None, None)
        return float(bid[0]) * self.multiplier

    def update(self, own_feedback):
        pass


class ExplorationAgent(Agent):
    """Episode-stable policy with mild causal action perturbations for offline data."""
    def __init__(self, base_spec, noise=.15):
        super().__init__()
        self.base_spec, self.noise = base_spec, noise

    def reset(self, own_config, public_config):
        from harmony_bidding.agents import make_agent
        self.base = make_agent(self.base_spec)
        self.base.reset(own_config, public_config)
        self.rng = np.random.default_rng(own_config.random_seed)

    def bidding(self, observation):
        return float(self.base.bidding(observation) * np.exp(self.rng.normal(0, self.noise)))

    def update(self, own_feedback):
        self.base.update(own_feedback)


def spec(kind, **kwargs):
    classes = {'FixedAlpha': 'FixedAlphaAgent', 'LocalPID': 'ScaledPIDAgent', 'OnlineLP': 'OnlineLPAgent', 'Exploration': 'ExplorationAgent'}
    return {'kind': 'custom', 'class_path': 'harmony_baselines.classical:' + classes[kind], 'kwargs': kwargs}


def fit_lp_table(slot_rows, output_path):
    """Preserve cost-threshold lookup with causal rollout response observations.

    For each constant alpha/type estimate normalized remaining cost by slot.
    Pool only train periods/roles/scenarios. Monotonize cost to define a valid
    budget frontier. There is no simulated cost or historical CPA outcome.
    """
    df = pd.DataFrame(slot_rows)
    keys = ['day_id', 'scenario', 'player_id', 'behavior_alpha']
    df = df.sort_values(keys + ['time_slot'])
    df['normalized_cost'] = df.cost / df.initial_budget
    df['future_cost'] = df.iloc[::-1].groupby(keys, sort=False).normalized_cost.cumsum().iloc[::-1]
    table = df.groupby(['player_type', 'time_slot', 'behavior_alpha'], as_index=False).future_cost.mean()
    table = table.sort_values(['player_type', 'time_slot', 'behavior_alpha'])
    table['cum_cost'] = table.groupby(['player_type', 'time_slot']).future_cost.cummax()
    table = table.rename(columns={'player_type': 'advertiserCategoryIndex', 'time_slot': 'timeStepIndex', 'behavior_alpha': 'realCPA'})
    table.to_csv(output_path, index=False)
    return str(Path(output_path).resolve())
