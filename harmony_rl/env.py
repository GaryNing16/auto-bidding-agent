"""72-step training adapter of the sole competition AuctionEngine.

The policy boundary is Observation/OwnFeedback, exactly as in the frozen
five-scenario evaluator. Evaluator diagnostics and matrices never enter info.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, replace
from time import perf_counter

from harmony_bidding.agent_api import Observation, OwnConfig, OwnFeedback, PublicConfig
from harmony_bidding.agents import make_agent
from harmony_bidding.auction import AuctionEngine, validate_alpha
from harmony_bidding.schema import Rules, fingerprint
from harmony_baselines.five_scenarios.evaluator import opponent_specs
from .markets import randomize_budgets, randomize_opponents


class BiddingEnv:
    """Gymnasium-style interface returning immutable legal observations.

    Construction does not reset agents; call reset before use. ``base_alpha``
    is an optional residual controller; its output is only a causal baseline
    action, and its state always receives the actual controlled-player result.
    """
    def __init__(self, day, controlled_player, opponents, scenario="Fixed", seed=0,
                 budget_config=None, residual_spec=None, record_debug=False,
                 opponent_randomization=None, rules=None):
        self.day = day
        self.controlled_player = str(controlled_player)
        self.opponents = deepcopy(opponents)
        self.scenario, self.seed = scenario, int(seed)
        self.budget_config = deepcopy(budget_config or {})
        self.residual_spec = deepcopy(residual_spec)
        self.record_debug = bool(record_debug)
        self.opponent_randomization = deepcopy(opponent_randomization or {})
        self.rules = rules or Rules()
        self._ready = False

    def reset(self, *, market_config=None, controlled_player=None, opponent_config=None,
              budget_config=None, seed=None, options=None):
        if options:
            raise ValueError("use explicit reset arguments; unknown options are not accepted")
        if market_config is not None:
            self.day = market_config
        if controlled_player is not None:
            self.controlled_player = str(controlled_player)
        if opponent_config is not None:
            self.opponents = deepcopy(opponent_config)
        if budget_config is not None:
            self.budget_config = deepcopy(budget_config)
        if seed is not None:
            self.seed = int(seed)
        if len(self.day.slots) != 72 or self.rules.total_slots != 72:
            raise ValueError("a complete episode requires exactly 72 slots")
        self.ids = tuple(a.advertiser_id for a in self.day.advertisers)
        if self.controlled_player not in self.ids:
            raise ValueError("unknown controlled player")
        advertisers, budgets = randomize_budgets(self.day.advertisers, self.controlled_player,
                                                seed=self.seed, **self.budget_config)
        self.engine = AuctionEngine(advertisers, replace(self.rules, tie_seed=self.seed))
        specs = opponent_specs(self.ids, self.controlled_player, self.scenario,
                               self.opponents, self.seed + 1000003, self.day.day_id)
        specs, opponent_parameters = randomize_opponents(specs, self.seed,
                                                        **self.opponent_randomization)
        self.agents, self.owns = {}, {}
        self.histories = {p: [] for p in self.ids}
        for advertiser in advertisers:
            p = advertiser.advertiser_id
            agent_seed = int(fingerprint([self.seed, self.day.day_id, p])[:16], 16)
            self.owns[p] = OwnConfig(p, advertiser.advertiser_type, advertiser.budget, agent_seed)
            if p != self.controlled_player:
                self.agents[p] = make_agent(specs[p])
                self.agents[p].reset(self.owns[p], PublicConfig(72))
        self.residual_agent = None
        if self.residual_spec is not None:
            self.residual_agent = make_agent(self.residual_spec)
            self.residual_agent.reset(self.owns[self.controlled_player], PublicConfig(72))
        self.time_slot = 0
        self.diagnostics, self.debug_log = [], []
        self._base_action = None
        self._started = perf_counter()
        self.metadata = dict(market_id=self.day.day_id, dataset_fingerprint=self.day.fingerprint,
            rules=asdict(self.engine.rules), scenario=self.scenario, seed=self.seed,
            player_id=self.controlled_player, budget_config=budgets,
            opponent_fingerprint=fingerprint(self.opponents), assignment_fingerprint=fingerprint(specs),
            opponent_randomization=opponent_parameters,
            request_count=sum(len(s.request_ids) for s in self.day.slots),
            market_source=deepcopy(self.day.source))
        self._ready = True
        return self.observation(), {"initial_budget": self.owns[self.controlled_player].initial_budget}

    def observation(self):
        if not self._ready:
            raise RuntimeError("reset must be called before observation")
        return self._observation(self.controlled_player)

    def _observation(self, player):
        return Observation(self.owns[player], self.time_slot, 72,
            self.engine.money.amount(self.engine.remaining[player]), tuple(self.histories[player]))

    def base_alpha(self):
        if not self._ready or self.time_slot >= 72:
            raise RuntimeError("base_alpha requires an active episode")
        if self.residual_agent is None:
            return 1.0
        if self._base_action is None:
            self._base_action = validate_alpha(self.residual_agent.bidding(self.observation()))
        return self._base_action

    def step(self, alpha):
        if not self._ready or self.time_slot >= 72:
            raise RuntimeError("step requires an active episode")
        # Validate before even invoking a stateful opponent's bidding method.
        action = validate_alpha(alpha)
        prior_agents = deepcopy(self.agents)
        prior_residual = deepcopy(self.residual_agent)
        prior_base = self._base_action
        prior_remaining = dict(self.engine.remaining)
        prior_values = dict(self.engine.cumulative_value)
        try:
            if self.residual_agent is not None:
                self.base_alpha()
            actions = {self.controlled_player: action}
            for p in self.ids:
                if p != self.controlled_player:
                    actions[p] = validate_alpha(self.agents[p].bidding(self._observation(p)))
            slot = self.day.slots[self.time_slot]
            stats, debug = self.engine.run_matrix_slot(slot.values, slot.participation_mask, actions,
                day_id=self.day.day_id, time_slot=self.time_slot, opportunity_ids=slot.request_ids,
                original_order=slot.original_order, record_debug=self.record_debug)
            feedbacks = {}
            for p in self.ids:
                feedbacks[p] = OwnFeedback(self.time_slot, actions[p], int(stats[p]["wins"]),
                    self.engine.money.amount(stats[p]["cost_units"]), float(stats[p]["value"]),
                    self.engine.money.amount(self.engine.remaining[p]),
                    self.engine.money.amount(self.engine.initial[p] - self.engine.remaining[p]),
                    self.engine.cumulative_value[p])
                if p != self.controlled_player:
                    self.agents[p].update(feedbacks[p])
                elif self.residual_agent is not None:
                    self.residual_agent.update(feedbacks[p])
        except Exception:
            self.agents, self.residual_agent = prior_agents, prior_residual
            self._base_action = prior_base
            self.engine.remaining, self.engine.cumulative_value = prior_remaining, prior_values
            raise
        for p in self.ids:
            self.histories[p].append(feedbacks[p])
        feedback = feedbacks[self.controlled_player]
        self.diagnostics.append(asdict(feedback))
        if self.record_debug:
            self.debug_log.append(debug)
        self.time_slot += 1
        self._base_action = None
        return self.observation(), feedback.value, self.time_slot == 72, False, {"feedback": feedback}

    def snapshot(self):
        """Copy mutable simulator/controller state while sharing read-only market arrays."""
        if not self._ready:
            raise RuntimeError("reset before snapshot")
        result = copy(self)
        for name in ("engine", "agents", "owns", "histories", "residual_agent", "diagnostics",
                     "debug_log", "metadata"):
            setattr(result, name, deepcopy(getattr(self, name)))
        return result

    def with_extra_budget(self, delta):
        """Counterfactual clone with extra initial/current funds and fixed past spends.

        Update all redundant legal budget fields. Future opponents evolve from
        cloned live states; no fixed future alpha sequence is replayed.
        """
        delta = validate_alpha(delta)
        result = self.snapshot()
        p = self.controlled_player
        units = result.engine.money.budget_units(delta)
        delta = result.engine.money.amount(units)
        result.engine.initial[p] += units
        result.engine.remaining[p] += units
        result.owns[p] = replace(result.owns[p], initial_budget=result.engine.money.amount(result.engine.initial[p]))
        result.histories[p] = [replace(f, remaining_budget=f.remaining_budget + delta)
                               for f in result.histories[p]]
        if result.residual_agent is not None:
            result.residual_agent.own_config = result.owns[p]
            if hasattr(result.residual_agent, "remaining_budget"):
                result.residual_agent.remaining_budget = result.engine.money.amount(result.engine.remaining[p])
        result._base_action = None
        result.metadata = {**result.metadata, "counterfactual_extra_budget": delta}
        return result

    def metrics(self):
        p = self.controlled_player
        initial = self.owns[p].initial_budget
        cost = self.engine.money.amount(self.engine.initial[p] - self.engine.remaining[p])
        exhausted = next((f.time_slot for f in self.histories[p] if f.remaining_budget == 0), None)
        return dict(total_value=self.engine.cumulative_value[p], total_cost=cost, initial_budget=initial,
            remaining_budget=self.engine.money.amount(self.engine.remaining[p]),
            budget_utilization=cost / initial if initial else 0., budget_exhaustion_slot=exhausted,
            wins=sum(f.wins for f in self.histories[p]), completed=self.time_slot == 72,
            completed_slots=self.time_slot, elapsed_seconds=perf_counter() - self._started,
            **self.metadata)
