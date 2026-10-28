from dataclasses import replace
import numpy as np
import pytest

from harmony_rl.markets import (MarketSampler, STRESS_CASES, bootstrap_market,
    distribution_summary, randomize_budgets, randomize_opponents, stress_market, synthetic_day)


def group_set(day):
    return {(tuple(row), tuple(mask)) for slot in day.slots for row, mask in zip(slot.values, slot.participation_mask)}


@pytest.mark.parametrize("method", ["within_slot", "request_blocks", "time_blocks"])
def test_complete_groups_masks_joint_values_and_unique_ids_are_preserved(method):
    day = synthetic_day(2, 5, 9)
    result = bootstrap_market(day, 6, traffic_multiplier=2., method=method)
    assert group_set(result) <= group_set(day)
    ids = [str(r) for s in result.slots for r in s.request_ids]
    assert len(ids) == 72 * 18 == len(set(ids))
    assert result.fingerprint == bootstrap_market(day, 6, traffic_multiplier=2., method=method).fingerprint
    assert result.fingerprint != bootstrap_market(day, 7, traffic_multiplier=2., method=method).fingerprint
    for s in result.slots:
        assert np.all(s.values[~s.participation_mask] == 0)
        assert np.all(np.diff(s.original_order) > 0)


@pytest.mark.parametrize("case", STRESS_CASES)
def test_stress_never_invents_pvalues_or_breaks_groups(case):
    day = synthetic_day(4, 5, 9)
    result = stress_market(day, 13, case)
    assert group_set(result) <= group_set(day)
    assert result.source["stress_case"] == case
    if case == "late_high_value":
        for t in range(36):
            def score(slot):
                return slot.values.sum(axis=1) / np.maximum(slot.participation_mask.sum(axis=1), 1)
            assert max(score(result.slots[t])) <= min(score(result.slots[t + 36]))


def test_budget_modes_preserve_original_heterogeneous_budgets_and_traffic_distinction():
    advertisers = synthetic_day(0).advertisers
    original, _ = randomize_budgets(advertisers, "0")
    assert original == advertisers and len(set(a.budget for a in original)) == 48
    controlled, meta = randomize_budgets(advertisers, "0", seed=7, mode="controlled")
    assert controlled[1:] == advertisers[1:] and controlled[0] != advertisers[0]
    assert .8 <= meta["realized_multipliers"]["0"] <= 1.2
    common, _ = randomize_budgets(advertisers, "0", seed=7, mode="common")
    assert common == randomize_budgets(advertisers, "0", seed=7, mode="common")[0]
    tight, _ = randomize_budgets(advertisers, "0", traffic_multiplier=2.)
    scaled, meta = randomize_budgets(advertisers, "0", traffic_multiplier=2., traffic_budget_mode="proportional")
    assert tight == advertisers
    assert all(b.budget == round(a.budget * 2, 6) for a, b in zip(advertisers, scaled))
    assert meta["traffic_budget_mode"] == "proportional"


def test_sampler_training_whitelist_and_exact_resume():
    days = [synthetic_day(1, 3, 2), synthetic_day(2, 3, 2)]
    with pytest.raises(ValueError, match="training split"):
        MarketSampler(days, allowed_day_ids=[days[0].day_id])
    sampler = MarketSampler(days, seed=44)
    sampler.sample()
    saved = sampler.state_dict()
    first, meta = sampler.sample()
    sampler.load_state_dict(saved)
    second, repeated = sampler.sample()
    assert first.fingerprint == second.fingerprint and meta == repeated
    a, info_a = sampler.sample(seed=18)
    b, info_b = sampler.sample(seed=18)
    assert a.fingerprint == b.fingerprint and info_a == info_b
    assert info_a["parent_day_id"] in {d.day_id for d in days}


def test_statistics_distinguish_absence_and_zero_values():
    day = synthetic_day(4, 3, 4)
    summary = distribution_summary(day)
    assert summary["request_count"] == 288
    assert sum(summary["participants_per_request_histogram"]) == 288
    expected = sum(int(s.participation_mask[:, 0].sum()) for s in day.slots)
    assert summary["per_player_opportunities"]["0"] == expected
    assert summary["competition_cooccurrence"][0][0] == expected


def test_opponent_randomization_default_is_exact_and_explicit_ranges_are_reproducible():
    specs = {"0": {"kind": "pid", "initial_alpha": 4., "kp": .5},
             "1": {"kind": "constant", "alpha": 8.}}
    exact, _ = randomize_opponents(specs, 3)
    assert exact == specs
    varied, meta = randomize_opponents(specs, 3, low=.8, high=1.2, gain_low=.9, gain_high=1.1)
    assert varied == randomize_opponents(specs, 3, low=.8, high=1.2, gain_low=.9, gain_high=1.1)[0]
    assert varied != specs and specs["0"]["initial_alpha"] == 4.
