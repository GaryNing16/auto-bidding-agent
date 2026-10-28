"""Hand-calculated examples, independent of engine-generated expected answers."""
from dataclasses import replace
import random
import pytest
from harmony_bidding.auction import AuctionEngine, Money, choose_request_tie, validate_alpha
from harmony_bidding.budget_policy import ExcludeUnaffordablePolicy
from harmony_bidding.schema import Advertiser, Opportunity, Rules


def engine(budgets=(100, 100, 100), **kwargs):
    return AuctionEngine(tuple(Advertiser(a, "test", b) for a, b in zip("ABC", budgets)), Rules(**kwargs))


def op(values, order=0, identifier="request", slot=0, mask=None):
    return Opportunity("day", slot, identifier, order, tuple(values),
                       (True,) * len(values) if mask is None else tuple(mask))


def test_standard_second_price_cost_above_value_allowed():
    e = engine()
    stats, debug = e.run_slot([op([10, 8, 11])], {"A": 1.2, "B": 1, "C": 1})
    assert stats["A"] == {"wins": 1, "cost_units": 11_000_000, "value": 10, "opportunity_count": 1}
    assert e.remaining == {"A": 89_000_000, "B": 100_000_000, "C": 100_000_000}
    assert stats["B"]["wins"] == stats["C"]["wins"] == 0


def test_zero_all_zero_single_positive_free():
    e = engine()
    stats, debug = e.run_slot([op([10, 8, 11])], dict.fromkeys("ABC", 0))
    assert debug[0]["winner"] is None
    stats, debug = e.run_slot([op([10, 8, 11])], {"A": 0, "B": 2, "C": 0})
    assert stats["B"] == {"wins": 1, "cost_units": 0, "value": 8, "opportunity_count": 1}


def test_tie_stable_across_rng_call_and_identity_order():
    e = engine(tie_seed=123)
    first = e.run_slot([op([10, 10, 10])], dict.fromkeys("ABC", 1))[1][0]["winner"]
    for _ in range(20):
        random.random()
    other = AuctionEngine(tuple(reversed(e.advertisers)), e.rules)
    second = other.run_slot([op([10, 10, 10])], {"C": 1, "A": 1, "B": 1})[1][0]["winner"]
    assert first == second
    assert choose_request_tie(123, "day", 0, "request", ("C", "A", "B"), 0) == first


def test_zero_balance_policy_is_explicit_and_replaceable():
    default = engine((0, 0, 0))
    assert default.run_slot([op([10, 0, 0])], dict.fromkeys("ABC", 1))[1][0]["winner"] is None
    allowed = engine((0, 0, 0), zero_balance_can_bid=True)
    assert allowed.run_slot([op([10, 0, 0])], dict.fromkeys("ABC", 1))[1][0]["winner"] == "A"


def test_raw_bid_above_remaining_but_price_affordable():
    e = engine((5, 100, 100))
    stats, _ = e.run_slot([op([100, 4, 3])], dict.fromkeys("ABC", 1))
    assert stats["A"]["cost_units"] == 4_000_000
    assert stats["A"]["value"] == 100


def test_unaffordable_winner_reauction():
    e = engine((5, 5, 5))
    stats, debug = e.run_slot([op([10, 8, 4])], dict.fromkeys("ABC", 1))
    assert debug[0]["excluded"] == ("A",)
    assert stats["B"] == {"wins": 1, "cost_units": 4_000_000, "value": 8, "opportunity_count": 1}
    e = engine((1, 1, 1))
    stats, debug = e.run_slot([op([10, 8, 4])], dict.fromkeys("ABC", 1))
    assert debug[0]["excluded"] == ("A", "B")
    assert stats["C"] == {"wins": 1, "cost_units": 0, "value": 4, "opportunity_count": 1}


def test_exclusion_only_affects_current_request():
    e = engine((5, 100, 100))
    stats, debug = e.run_slot([op([10, 8, 4]), op([10, 2, 1], order=1, identifier="next")], dict.fromkeys("ABC", 1))
    assert debug[0]["excluded"] == ("A",)
    assert debug[1]["winner"] == "A"
    assert stats["A"]["cost_units"] == 2_000_000


def test_request_order_and_mid_slot_exhaustion():
    first = op([10, 5, 0], identifier="z")
    second = op([100, 5, 0], order=1, identifier="a")
    e = engine((5, 100, 100))
    stats, debug = e.run_slot([first, second], dict.fromkeys("ABC", 1))
    assert stats["A"]["value"] == 10 and stats["A"]["wins"] == 1
    assert e.remaining["A"] == 0
    e = engine((5, 100, 100))
    reversed_order = [replace(second, original_order=0), replace(first, original_order=1)]
    assert e.run_slot(reversed_order, dict.fromkeys("ABC", 1))[0]["A"]["value"] == 100
    with pytest.raises(ValueError, match="original_order"):
        engine().run_slot([second, first], dict.fromkeys("ABC", 1))


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), -float("inf"), [1], {"a": 1}, None, True, False, "1", 1 + 0j, 10 ** 1000])
def test_invalid_actions_fail(bad):
    with pytest.raises(ValueError):
        validate_alpha(bad)
    e = engine()
    before = (dict(e.remaining), dict(e.cumulative_value))
    with pytest.raises(ValueError):
        e.run_slot([op([10, 8, 11])], {"A": 1.2, "B": 1, "C": bad})
    assert (e.remaining, e.cumulative_value) == before


@pytest.mark.parametrize("alpha", [0, 1, 5, 5.001, 40, 100, 1e100, 1e308])
def test_actions_have_no_global_cap(alpha):
    assert validate_alpha(alpha) == alpha
    e = engine()
    e.run_slot([op([1, 0, 0])], {"A": alpha, "B": 0, "C": 0})
    assert e.cumulative_value["A"] == (1 if alpha > 0 else 0)


def test_large_alpha_is_not_silently_clipped():
    stats, _ = engine().run_slot([op([1, 10, 2])], {"A": 40, "B": 1, "C": 1})
    assert stats["A"]["value"] == 1 and stats["A"]["cost_units"] == 10_000_000


def test_money_precision_and_invalid_budget():
    money = Money(2)
    assert money.bid_units(1, 1.005) == 101
    with pytest.raises(ValueError, match="precision"):
        engine((1.0000001, 1, 1))
    e = engine((0.3, 10, 10), money_decimals=2)
    stats, _ = e.run_slot([op([1, .1, 0], order=i, identifier=str(i)) for i in range(10)], dict.fromkeys("ABC", 1))
    assert stats["A"]["wins"] == 3 and e.remaining["A"] == 0


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), None])
def test_invalid_values_rejected(bad):
    with pytest.raises(ValueError):
        engine().run_slot([op([10, bad, 11])], dict.fromkeys("ABC", 1))


def test_direct_engine_rejects_duplicate_ids_and_mixed_slots():
    with pytest.raises(ValueError, match="unique identities"):
        AuctionEngine((Advertiser("A", "x", 100), Advertiser("A", "x", 100)), Rules())
    with pytest.raises(ValueError, match="unique identities"):
        AuctionEngine((), Rules())
    with pytest.raises(ValueError, match="identities"):
        engine().run_slot([op([10, 8, 11]), op([10, 8, 11], order=1)], dict.fromkeys("ABC", 1))
    with pytest.raises(ValueError, match="one day/time_slot"):
        engine().run_slot([op([10, 8, 11]), op([10, 8, 11], order=1, identifier="b", slot=1)], dict.fromkeys("ABC", 1))


def test_sparse_manual_payments_and_presence_counts():
    e = engine((20, 20, 20))
    requests = [op([10, 8, 11]), op([3, 2, 0], 1, "r2", mask=[True, True, False]),
                op([4, 0, 0], 2, "r3", mask=[True, False, False])]
    stats, debug = e.run_slot(requests, {"A": 1.2, "B": 1, "C": 1})
    assert [row["winner"] for row in debug] == ["A", "A", "A"]
    assert [row["payment_units"] for row in debug] == [11_000_000, 2_000_000, 0]
    assert stats["A"] == {"wins": 3, "cost_units": 13_000_000, "value": 17, "opportunity_count": 3}
    assert [stats[a]["opportunity_count"] for a in "ABC"] == [3, 2, 1]
    assert e.remaining["A"] == 7_000_000


def test_present_zero_and_absent_zero_have_different_counts():
    stats, debug = engine().run_slot([op([0, 0, 0], mask=[True, False, True])], dict.fromkeys("ABC", 100))
    assert debug[0]["winner"] is None
    assert [stats[a]["opportunity_count"] for a in "ABC"] == [1, 0, 1]
    assert all(row["value"] == row["wins"] == row["cost_units"] == 0 for row in stats.values())


@pytest.mark.parametrize("bad_request", [
    op([10, 8, 11], 1),  # duplicate request identity after one valid settlement
    op([10, 8, 11], 0, "next"),
    op([10, 8, 11], 1, "next", slot=1),
    op([10, -1, 11], 1, "next"),
    op([10, float("nan"), 11], 1, "next"),
    op([10, 8, 11], 1, "next", mask=[True, False, True]),
    op([10, 8, 11], 1, "next", mask=[True, True]),
    op([10, 8, 11], 1, "next", mask=[1, True, True]),
])
def test_late_invalid_request_does_not_commit_earlier_payment(bad_request):
    e = engine()
    before = (dict(e.remaining), dict(e.cumulative_value))
    with pytest.raises(ValueError):
        e.run_slot([op([10, 8, 11]), bad_request], {"A": 1.2, "B": 1, "C": 1})
    assert (e.remaining, e.cumulative_value) == before


def test_finite_large_action_with_small_value_and_product_overflow_atomicity():
    assert Money().bid_units(1e308, 1e-308) == 1_000_000
    e = engine()
    before = (dict(e.remaining), dict(e.cumulative_value))
    with pytest.raises(ValueError, match="request overflow, advertiser A: bid product overflow"):
        e.run_slot([op([1e-308, .5, 0]), op([2, 0, 0], 1, "overflow")],
                   {"A": 1e308, "B": 1, "C": 0})
    assert (e.remaining, e.cumulative_value) == before


def test_cumulative_value_overflow_is_atomic():
    e = engine()
    before = (dict(e.remaining), dict(e.cumulative_value))
    with pytest.raises(ValueError, match="cumulative value overflow"):
        e.run_slot([op([1e308, 0, 0]), op([1e308, 0, 0], 1, "second")],
                   {"A": 1e-308, "B": 0, "C": 0})
    assert (e.remaining, e.cumulative_value) == before


def test_ties_redraw_after_exclusion_with_new_round():
    calls = []

    def choose(tied, round_index):
        calls.append((tied, round_index))
        return tied[0] if round_index == 0 else tied[-1]

    settlement = ExcludeUnaffordablePolicy().settle(
        {"C": 10, "B": 10, "A": 10}, {"A": 1, "B": 10, "C": 10}, choose)
    assert settlement.winner == "C" and settlement.payment_units == 10 and settlement.excluded == ("A",)
    assert calls == [(("A", "B", "C"), 0), (("B", "C"), 1)]


def test_tie_sampler_distribution_diagnostic_on_fixed_seeds():
    counts = dict.fromkeys("ABC", 0)
    for seed in range(600):
        counts[choose_request_tie(seed, "day", 0, "same-request", ("C", "A", "B"), 0)] += 1
    # Broad deterministic diagnostic, not a demand for exactly equal outcomes.
    assert all(140 < count < 260 for count in counts.values()), counts


def test_multiple_unaffordable_candidates_recompute_second_price():
    advertisers = tuple(Advertiser(a, "x", b) for a, b in zip("ABCD", (10, 20, 100, 100)))
    e = AuctionEngine(advertisers, Rules())
    stats, debug = e.run_slot([op([100, 90, 80, 70])], dict.fromkeys("ABCD", 1))
    assert debug[0]["excluded"] == ("A", "B")
    assert debug[0]["winner"] == "C"
    assert stats["C"]["cost_units"] == 70_000_000 and stats["C"]["value"] == 80
