"""Causal scalar adapter of AuctionNet's threshold-based ``PidBiddingStrategy``.

The upstream controller is not a proportional/integral/derivative controller.
Its asymmetric increase/decrease formula and branch priority are preserved.
Only the evaluator sees market values; this module consumes own balances and
completed-slot feedback. See PID_ADAPTATION.md for the exact adaptation scope.
"""
from fractions import Fraction
import math

from harmony_bidding.agent_api import Agent, finite_number


def _pacing_ratio(cost, future_weight, previous_weight, remaining):
    """Use upstream arithmetic normally; avoid overflow/underflow in extremes.

    Fraction is only a comparison fallback for an intermediate nonfinite/zero
    float. It does not bound the auction action or alter the control formula.
    """
    if cost == 0.0:
        return 0.0
    product = cost * future_weight
    normalized = product / previous_weight
    result = normalized / remaining
    if (all(math.isfinite(x) and x > 0.0 for x in (product, normalized, result))):
        return result
    return (Fraction(cost) * Fraction(future_weight)
            / Fraction(previous_weight) / Fraction(remaining))


class AuctionNetPIDAgent(Agent):
    """AuctionNet budget-consumption thresholds, adapted to the local Agent API.

    ``expected_ratios`` is a fixed configuration prior, never live/future traffic.
    ``player_alphas`` is optional opponent calibration; shared target policies
    leave it unset. There is no action maximum or silent overflow clipping.
    """

    def __init__(self, initial_alpha=15.0, lower_threshold=0.7,
                 upper_threshold=1.1, increase_factor=1.2,
                 decrease_factor=0.7, expected_ratios=None, player_alphas=None):
        super().__init__()
        self.initial_alpha = finite_number(initial_alpha, "initial_alpha")
        self.lower_threshold = finite_number(lower_threshold, "lower_threshold")
        self.upper_threshold = finite_number(upper_threshold, "upper_threshold")
        self.increase_factor = finite_number(increase_factor, "increase_factor")
        self.decrease_factor = finite_number(decrease_factor, "decrease_factor")
        if self.lower_threshold >= self.upper_threshold:
            raise ValueError("lower_threshold must be less than upper_threshold")
        if self.increase_factor < 1.0:
            raise ValueError("increase_factor must be at least one")
        if not 0.0 < self.decrease_factor <= 1.0:
            raise ValueError("decrease_factor must be in (0, 1]")
        self.configured_ratios = (None if expected_ratios is None else
                                 tuple(finite_number(x, "expected_ratios")
                                       for x in expected_ratios))
        if self.configured_ratios is not None and any(x == 0 for x in self.configured_ratios):
            raise ValueError("expected_ratios must be strictly positive")
        self.player_alphas = {}
        for player, alpha in dict(player_alphas or {}).items():
            if not isinstance(player, str) or not player or player != player.strip():
                raise ValueError("player_alphas keys must be advertiser ID strings")
            self.player_alphas[player] = finite_number(alpha, "player_alphas")

    def reset(self, own_config, public_config):
        total = public_config.total_slots
        if type(total) is not int or total < 1:
            raise ValueError("total_slots must be a positive integer")
        budget = finite_number(own_config.initial_budget, "initial_budget")
        ratios = ((1.0,) * total if self.configured_ratios is None else self.configured_ratios)
        if len(ratios) != total:
            raise ValueError("expected_ratios length must equal total_slots")
        try:
            future_weights = tuple(math.fsum(ratios[t:]) for t in range(total))
        except OverflowError as exc:
            raise ValueError("expected_ratios sums must be finite") from exc
        if any(not math.isfinite(x) or x <= 0.0 for x in future_weights):
            raise ValueError("expected_ratios sums must be positive and finite")
        self.own_config, self.public_config = own_config, public_config
        self.expected_ratios, self._future_weights = ratios, future_weights
        self.alpha = self.player_alphas.get(own_config.advertiser_id, self.initial_alpha)
        self.remaining_budget = budget
        self.last_tick_cost = 0.0
        self.feedback_count = 0
        self._last_bid_slot = -1

    def bidding(self, observation):
        if self.own_config is None:
            raise ValueError("AuctionNetPID must be reset before bidding")
        t = observation.time_slot
        total = self.public_config.total_slots
        if (type(t) is not int or not 0 <= t < total or t != self.feedback_count
                or observation.total_slots != total):
            raise ValueError("AuctionNetPID observation time out of sequence")
        if observation.own_config != self.own_config:
            raise ValueError("AuctionNetPID observation belongs to another player")
        remaining = finite_number(observation.remaining_budget, "remaining_budget")
        if remaining != self.remaining_budget:
            raise ValueError("AuctionNetPID observation balance disagrees with own feedback")
        if self._last_bid_slot == t:
            return self.alpha
        candidate = self.alpha
        if remaining == 0.0:
            candidate = 0.0
        elif t > 0:
            # Keep the upstream asymmetry and if/elif priority exactly: only
            # the increase branch uses expected ratios; decrease uses T - t.
            increase_ratio = _pacing_ratio(self.last_tick_cost, self._future_weights[t],
                                           self.expected_ratios[t - 1], remaining)
            if increase_ratio < self.lower_threshold:
                candidate *= self.increase_factor
            elif _pacing_ratio(self.last_tick_cost, float(total - t), 1.0,
                               remaining) > self.upper_threshold:
                candidate *= self.decrease_factor
        candidate = finite_number(candidate, "AuctionNetPID alpha (overflow is not clipped)")
        self.alpha, self._last_bid_slot = candidate, t
        return candidate

    def update(self, own_feedback):
        if (self.own_config is None or type(own_feedback.time_slot) is not int
                or own_feedback.time_slot != self.feedback_count
                or self._last_bid_slot != self.feedback_count):
            raise ValueError("AuctionNetPID feedback out of sequence")
        remaining = finite_number(own_feedback.remaining_budget, "remaining_budget")
        reported_cost = finite_number(own_feedback.cost, "cost")
        if remaining > self.remaining_budget:
            raise ValueError("AuctionNetPID remaining budget cannot increase")
        # Upstream derives spend from consecutive own balances, not CPA, value,
        # or the current market. Preserve that arithmetic for its next decision.
        cost = self.remaining_budget - remaining
        tolerance = 8 * math.ulp(self.remaining_budget)
        if not math.isclose(cost, reported_cost, rel_tol=1e-12, abs_tol=tolerance):
            raise ValueError("AuctionNetPID feedback cost disagrees with own balances")
        self.last_tick_cost = cost
        self.remaining_budget = remaining
        self.feedback_count += 1
