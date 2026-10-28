"""Semantic differential checks for the matrix entry of the one auction core."""
import math
import numpy as np
import pytest

from harmony_bidding.auction import AuctionEngine
from harmony_bidding.schema import Advertiser, Opportunity, Rules


def compare(values, mask, alphas, budgets, *, decimals=6, zero=False, seed=27):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    ids = tuple(str(i) for i in range(values.shape[1]))
    advertisers = tuple(Advertiser(i, "test", b) for i, b in zip(ids, budgets))
    rules = Rules(money_decimals=decimals, zero_balance_can_bid=zero, tie_seed=seed)
    slow, fast = AuctionEngine(advertisers, rules), AuctionEngine(advertisers, rules)
    actions = dict(zip(ids, alphas))
    request_ids = tuple(f"r-{i}" for i in range(len(values)))
    opportunities = tuple(Opportunity("test", 3, r, i, tuple(float(v) for v in row),
                                     tuple(bool(v) for v in m))
                          for i, (r, row, m) in enumerate(zip(request_ids, values, mask)))
    expected = slow.run_slot(opportunities, actions)
    actual = fast.run_matrix_slot(values, mask, actions, day_id="test", time_slot=3,
        opportunity_ids=request_ids, original_order=np.arange(len(values)), record_debug=True)
    assert actual == expected
    assert fast.remaining == slow.remaining
    assert fast.cumulative_value == slow.cumulative_value


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("zero", [False, True])
def test_random_sparse_matrix_matches_reference(seed, zero):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 12))
    values = rng.integers(0, 25, size=(180, n)).astype(float) / 100
    mask = rng.random(values.shape) > 0.3
    values[~mask] = 0
    # Deliberate ties, zeros, exhausted markets and multiple reauctions.
    compare(values, mask, rng.integers(0, 5, size=n).astype(float),
            rng.integers(0, 30, size=n).astype(float), decimals=2, zero=zero, seed=seed)


@pytest.mark.parametrize("n", [0, 1, 63, 64, 200])
def test_single_advertiser_and_empty_slot(n):
    compare(np.ones((n, 1)), np.ones((n, 1), dtype=bool), [3.], [1.])


def test_exact_depletion_changes_subsequent_second_prices():
    values = np.tile([3., 2., 1.], (180, 1))
    compare(values, np.ones_like(values, dtype=bool), [1., 1., 1.], [2., 500., 500.], decimals=0)


def test_rounding_boundaries_and_large_integer_fallback():
    for scale in [1., 1e-6, 1e5, 1e14]:
        values = np.tile([0.5000000000000001 * scale, 0.5 * scale,
                          0.49999999999999994 * scale], (100, 1))
        compare(values, np.ones_like(values, dtype=bool), [1., 1., 1.],
                [min(1e16, max(100., scale * 1000))] * 3)


def test_decimal_product_not_binary_product():
    rng = np.random.default_rng(6)
    values = rng.uniform(0, 1, (200, 6))
    values[0] = [0.0000005, 0.0000015, 0.0000025, 0.05, 0.15, 0.25]
    compare(values, np.ones_like(values, dtype=bool), [1., 1., 1., 0.00001, 0.00001, 0.00001], [1e5]*6)


@pytest.mark.parametrize("budget", [0., 0.01, 1., 100000.])
def test_full_48_player_market_continuous_values(budget):
    rng = np.random.default_rng(28)
    values = rng.lognormal(-5, 1, size=(350, 48))
    mask = rng.random(values.shape) > .1
    values[~mask] = 0
    compare(values, mask, rng.lognormal(3, 2, 48), [budget] * 48)


def test_decimal_half_boundaries_under_multiple_alpha_scales():
    rng = np.random.default_rng(29)
    alphas = np.array([1e-5, .1, .3, 1.25, 3.14159, 40., 500., 1e8])
    values = (rng.integers(0, 1000, size=(250, len(alphas))) + .5) / 1e6 / alphas
    # Include the representable neighbors on either side of each half unit.
    values[::3] = np.nextafter(values[::3], np.inf)
    values[1::3] = np.nextafter(values[1::3], 0.)
    compare(values, np.ones_like(values, dtype=bool), alphas, [1e5]*len(alphas))


@pytest.mark.parametrize("failure", ["nan", "absent_value", "mask_type", "overflow", "duplicate", "order"])
def test_matrix_failure_is_atomic(failure):
    engine = AuctionEngine((Advertiser("0", "t", 10.), Advertiser("1", "t", 10.)), Rules())
    values = np.ones((100, 2))
    mask = np.ones((100, 2), dtype=bool)
    ids = [str(i) for i in range(100)]
    order = np.arange(100)
    actions = {"0": 2., "1": 1.}
    if failure == "nan":
        values[-1, -1] = float("nan")
    elif failure == "absent_value":
        mask[-1, -1] = False
    elif failure == "mask_type":
        mask = mask.astype(int)
    elif failure == "overflow":
        values[-1, 0] = 1e308
    elif failure == "duplicate":
        ids[-1] = ids[0]
    elif failure == "order":
        order[-1] = 0
    before = (dict(engine.remaining), dict(engine.cumulative_value))
    with pytest.raises(ValueError):
        engine.run_matrix_slot(values, mask, actions, day_id="t", time_slot=0,
                               opportunity_ids=ids, original_order=order)
    assert (engine.remaining, engine.cumulative_value) == before


def test_no_debug_and_multi_slot_ledger_continuity():
    ads = tuple(Advertiser(str(i), "t", 1000.) for i in range(3))
    fast, slow = AuctionEngine(ads, Rules()), AuctionEngine(ads, Rules())
    rng = np.random.default_rng(17)
    actions = {str(i): float(i + 1) for i in range(3)}
    for slot in range(5):
        values = rng.uniform(0, 1, (150, 3))
        mask = np.ones_like(values, dtype=bool)
        ids = tuple(str(i) for i in range(150))
        expected, _ = slow.run_slot(tuple(Opportunity("t", slot, ids[i], i, tuple(row), (True,)*3)
                                         for i, row in enumerate(values)), actions)
        actual, debug = fast.run_matrix_slot(values, mask, actions, day_id="t", time_slot=slot,
                                            opportunity_ids=ids, original_order=np.arange(150))
        assert debug == []
        assert actual == expected
        assert fast.remaining == slow.remaining
        assert fast.cumulative_value == slow.cumulative_value
