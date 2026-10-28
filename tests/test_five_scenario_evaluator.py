"""Bounded integration checks; no retained market period or model is executed."""
from copy import deepcopy
from dataclasses import fields, replace
import json

import numpy as np
import pytest

from harmony_bidding.agent_api import Agent, Observation, OwnFeedback
from harmony_bidding.schema import Advertiser, fingerprint
from harmony_baselines.classical import spec
from harmony_baselines.data import MatrixDay, MatrixSlot
from harmony_baselines import evaluator as historical
from harmony_baselines.five_scenarios import evaluator as five


def tiny_day(players=48):
    """Few requests per slot, including ties, absent rows, genuine zero and empty slots."""
    rng = np.random.default_rng(800)
    slots = []
    for slot in range(72):
        values = rng.uniform(.001, .02, size=(4, players))
        mask = rng.random(values.shape) > .2
        values[~mask] = 0.
        values[0] = .01
        mask[0] = True
        values[1, 0], mask[1, 0] = 0., True
        if slot in (6, 43):
            values, mask = values[:0], mask[:0]
        count = len(values)
        slots.append(MatrixSlot(values, mask, np.array([f"request-{slot}-{i}" for i in range(count)]),
                                np.arange(count, dtype=np.int64)))
    advertisers = tuple(Advertiser(str(i), str(i // 8), .16 + .01 * (i % 5)) for i in range(players))
    return MatrixDay("five-synthetic", advertisers, tuple(slots), "synthetic-five-fingerprint", {})


def opponent_config():
    # Deliberately distinct specs; actual AuctionNetPID integration is below.
    return {"fixed": spec("FixedAlpha", alpha=3.),
            "pid": spec("LocalPID", initial_alpha=3., kp=.5, ki=.05, kd=.05),
            "auctionnet_pid": spec("LocalPID", initial_alpha=4., kp=.7, ki=.1, kd=.1)}


def test_canonical_names_do_not_silently_accept_historical_pid():
    assert five.SCENARIOS == ("Fixed", "LocalPID", "Mixed", "AuctionNetPID", "Mixed2")
    assert five.historical_scenario_name("PID") == "LocalPID"
    assert five.historical_scenario_name("Mixed") == "Mixed"
    with pytest.raises(ValueError, match="unknown scenario"):
        five.historical_scenario_name("unrecognized")
    with pytest.raises(ValueError, match="canonical scenario"):
        five.run_episode(tiny_day(3), "0", spec("FixedAlpha", alpha=3.), "PID", opponent_config())


@pytest.mark.parametrize("target", ["0", "17"])
@pytest.mark.parametrize("seed", [20261028, 20261029])
@pytest.mark.parametrize("scenario,old_name", [("Fixed", "Fixed"), ("LocalPID", "PID"), ("Mixed", "Mixed")])
def test_old_scenarios_business_metrics_and_assignment_are_exact(scenario, old_name, target, seed):
    day = tiny_day()
    opponents = opponent_config()
    old_opponents = {key: opponents[key] for key in ("fixed", "pid")}
    target_spec = spec("LocalPID", initial_alpha=4., kp=.5, ki=.05, kd=.05)
    before = deepcopy(opponents)
    old_row, old_slots, old_transitions = historical.run_episode(
        day, target, target_spec, old_name, old_opponents, seed=seed, collect=True)
    row, slots, transitions = five.run_episode(
        day, target, target_spec, scenario, opponents, seed=seed, collect=True)
    for key in ("total_value", "total_cost", "wins", "budget_utilization", "initial_budget",
                "assignment_fingerprint", "model_fingerprint", "dataset_fingerprint", "rules_version"):
        assert row[key] == old_row[key], key
    assert row["scenario"] == scenario
    assert row["legacy_opponent_fingerprint"] == old_row["opponent_fingerprint"]
    assert row["opponent_fingerprint"] != old_row["opponent_fingerprint"]
    assert len(slots) == len(old_slots) == len(transitions) == len(old_transitions) == 72
    for new_slot, old_slot in zip(slots, old_slots):
        for key in ("alpha", "cost", "value", "wins", "time_slot", "player_id"):
            assert new_slot[key] == old_slot[key]
        assert new_slot["slot"] == new_slot["time_slot"]
    for new, old in zip(transitions, old_transitions):
        assert new.keys() == old.keys()
        for key in new:
            if isinstance(new[key], np.ndarray):
                np.testing.assert_array_equal(new[key], old[key])
            else:
                assert new[key] == old[key]
    assert opponents == before


def test_mixed_assignments_preserve_old_rng_and_match_fixed_membership():
    ids = [str(i) for i in range(48)]
    opponents = opponent_config()
    old = historical.opponent_specs(ids, "17", "Mixed", opponents, 20261028 + 1000003, "day")
    mixed = five.opponent_specs(ids, "17", "Mixed", opponents, 20261028 + 1000003, "day")
    mixed2 = five.opponent_specs(ids, "17", "Mixed2", opponents, 20261028 + 1000003, "day")
    assert mixed == old
    fixed_ids = {p for p, s in mixed.items() if s == opponents["fixed"]}
    assert len(fixed_ids) == 23
    assert fixed_ids == {p for p, s in mixed2.items() if s == opponents["fixed"]}
    assert sum(s == opponents["auctionnet_pid"] for s in mixed2.values()) == 24
    assert "17" not in mixed2
    assert mixed2 == five.opponent_specs(ids, "17", "Mixed2", opponents, 20261028 + 1000003, "day")
    assert all(s == opponents["auctionnet_pid"] for s in
               five.opponent_specs(ids, "17", "AuctionNetPID", opponents, 11, "day").values())


@pytest.mark.parametrize("scenario", ["AuctionNetPID", "Mixed2"])
def test_new_scenarios_have_independent_instances_and_only_own_history(monkeypatch, scenario):
    created, calls = [], []
    original_engine = five.AuctionEngine
    class TrackedEngine(original_engine):
        def run_matrix_slot(self, values, mask, alphas, **kwargs):
            calls.append(kwargs["time_slot"])
            assert mask.dtype == np.bool_
            return super().run_matrix_slot(values, mask, alphas, **kwargs)
    class OwnOnlyAgent(Agent):
        def reset(self, own, public):
            self.own_config, self.public_config = own, public
            self.completed = 0
            self.observations = []
        def bidding(self, observation):
            assert type(observation) is Observation
            assert {f.name for f in fields(observation)} == {
                "own_config", "time_slot", "total_slots", "remaining_budget", "history"}
            assert observation.own_config is self.own_config
            assert len(observation.history) == self.completed == observation.time_slot
            assert all(type(feedback) is OwnFeedback for feedback in observation.history)
            self.observations.append(observation)
            return 2. + self.completed / 100
        def update(self, feedback):
            assert feedback.time_slot == self.completed
            self.completed += 1
    def factory(specification):
        agent = OwnOnlyAgent()
        created.append(agent)
        return agent
    monkeypatch.setattr(five, "make_agent", factory)
    monkeypatch.setattr(five, "AuctionEngine", TrackedEngine)
    row, slots, transitions = five.run_episode(tiny_day(), "0", spec("FixedAlpha", alpha=2.),
                                              scenario, opponent_config(), collect=True)
    assert len(created) == len({id(a) for a in created}) == 48
    assert all(a.completed == 72 and len(a.observations) == 72 for a in created)
    assert calls == list(range(72))
    assert row["completed"] and row["completed_slots"] == 72
    assert len(slots) == len(transitions) == 72
    assert slots[-1]["cumulative_value"] == row["total_value"]
    assert slots[-1]["cumulative_cost"] == row["total_cost"]
    assert slots[-1]["remaining_budget"] == pytest.approx(row["initial_budget"] - row["total_cost"])
    assert transitions[-1]["done"] is True and not any(t["done"] for t in transitions[:-1])
    assert transitions[-1]["next_state"][0] == 1.
    assert all(t["state"].shape == (18,) for t in transitions)


def test_direct_call_timing_excludes_validation_auction_and_json(monkeypatch, tmp_path):
    now = [0.]
    class TimedAgent(Agent):
        def reset(self, own, public):
            self.own_config = own
        def bidding(self, observation):
            now[0] += .002 if self.own_config.advertiser_id == "0" else .1
            return 2.
        def update(self, feedback):
            now[0] += .003 if self.own_config.advertiser_id == "0" else .2
    original_validate, original_engine, original_dumps = five.validate_alpha, five.AuctionEngine, json.dumps
    def slow_validate(value):
        now[0] += 7.
        return original_validate(value)
    class SlowEngine(original_engine):
        def run_matrix_slot(self, *args, **kwargs):
            now[0] += 11.
            return super().run_matrix_slot(*args, **kwargs)
    def slow_json(*args, **kwargs):
        now[0] += 13.
        return original_dumps(*args, **kwargs)
    monkeypatch.setattr(five, "make_agent", lambda _: TimedAgent())
    monkeypatch.setattr(five, "perf_counter", lambda: now[0])
    monkeypatch.setattr(five, "validate_alpha", slow_validate)
    monkeypatch.setattr(five, "AuctionEngine", SlowEngine)
    monkeypatch.setattr(five.json, "dumps", slow_json)
    path = tmp_path / "target.jsonl"
    row, slots, _ = five.run_episode(tiny_day(3), "0", spec("FixedAlpha", alpha=2.), "Mixed2",
                                    opponent_config(), diagnostic_path=path)
    for key, expected in [("bidding_p99_ms", 2.), ("update_p99_ms", 3.), ("paired_p99_ms", 5.)]:
        assert row[key] == pytest.approx(expected, abs=1e-7)
    assert slots[0]["bidding_ms"] == pytest.approx(2.)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 216
    feedback = [r for r in records if r["phase"] == "feedback"]
    assert [r["slot"] for r in feedback] == list(range(72))
    assert all({"alpha", "cost", "value", "wins", "remaining_budget"} <= r.keys() for r in feedback)


def test_policy_failure_keeps_completed_diagnostics_without_a_result(monkeypatch, tmp_path):
    class FailsAtSlotTwo(Agent):
        def reset(self, own, public):
            self.own_config = own
        def bidding(self, observation):
            if self.own_config.advertiser_id == "0" and observation.time_slot == 2:
                raise OverflowError("diagnostic failure")
            return 2.
        def update(self, feedback):
            pass
    monkeypatch.setattr(five, "make_agent", lambda _: FailsAtSlotTwo())
    path = tmp_path / "failed.jsonl"
    with pytest.raises(OverflowError, match="diagnostic failure"):
        five.run_episode(tiny_day(3), "0", spec("FixedAlpha", alpha=2.), "AuctionNetPID",
                         opponent_config(), diagnostic_path=path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["slot"] for r in records if r["phase"] == "feedback"] == [0, 1]
    assert records[-1]["phase"] == "before_action" and records[-1]["slot"] == 2
    # The handle was closed even though the episode raised.
    with path.open("a") as handle:
        handle.write("")


def test_core_rejects_nonzero_absent_values_and_short_periods():
    day = tiny_day(3)
    with pytest.raises(ValueError, match="exactly 72"):
        five.run_episode(replace(day, slots=day.slots[:71]), "0", spec("FixedAlpha", alpha=1.),
                         "Fixed", opponent_config())
    values = day.slots[0].values.copy()
    mask = day.slots[0].participation_mask.copy()
    values[0, 1], mask[0, 1] = 1., False
    bad = replace(day, slots=(replace(day.slots[0], values=values, participation_mask=mask),) + day.slots[1:])
    with pytest.raises(ValueError, match="absent"):
        five.run_episode(bad, "0", spec("FixedAlpha", alpha=1.), "Fixed", opponent_config())


def test_target_changes_do_not_change_mixed2_initial_assignment():
    first, _, _ = five.run_episode(tiny_day(3), "0", spec("FixedAlpha", alpha=2.), "Mixed2", opponent_config())
    second, _, _ = five.run_episode(tiny_day(3), "0", spec("FixedAlpha", alpha=5.), "Mixed2", opponent_config())
    assert first["assignment_fingerprint"] == second["assignment_fingerprint"]
    assert first["model_fingerprint"] != second["model_fingerprint"]


def test_collect_all_is_explicit_and_does_not_hide_terminal_or_empty_slots():
    row, slots, transitions = five.run_episode(tiny_day(3), "0", spec("FixedAlpha", alpha=3.),
        "Mixed2", opponent_config(), collect=True, collect_all=True)
    assert len(slots) == len(transitions) == 3 * 72
    for player in ("0", "1", "2"):
        records = [r for r in slots if r["player_id"] == player]
        assert len(records) == 72
        assert records[6]["wins"] == records[6]["cost"] == records[6]["value"] == 0
        assert records[-1]["remaining_budget"] >= 0
        assert sum(t["done"] for t in transitions if t["player_id"] == player) == 1
    assert row["assignment_fingerprint"] == fingerprint(five.opponent_specs(
        ["0", "1", "2"], "0", "Mixed2", opponent_config(), 20261028 + 1000003, "five-synthetic"))


@pytest.mark.parametrize("scenario", ["AuctionNetPID", "Mixed2"])
def test_actual_auctionnet_pid_runs_72_slots_and_resets_between_episodes(monkeypatch, scenario):
    from harmony_baselines.five_scenarios.auctionnet_pid import AuctionNetPIDAgent
    target = {"kind": "custom",
              "class_path": "harmony_baselines.five_scenarios.auctionnet_pid:AuctionNetPIDAgent",
              "kwargs": {"initial_alpha": 3., "lower_threshold": .7, "upper_threshold": 1.1,
                         "increase_factor": 1.2, "decrease_factor": .7}}
    opponents = {**opponent_config(), "auctionnet_pid": target}
    instances = []
    original_factory = five.make_agent
    def factory(specification):
        agent = original_factory(specification)
        instances.append(agent)
        return agent
    monkeypatch.setattr(five, "make_agent", factory)
    first, slots, _ = five.run_episode(tiny_day(), "0", target, scenario, opponents)
    second, repeated, _ = five.run_episode(tiny_day(), "0", target, scenario, opponents)
    assert len(instances) == len({id(agent) for agent in instances}) == 96
    expected = 96 if scenario == "AuctionNetPID" else 50
    assert sum(isinstance(agent, AuctionNetPIDAgent) for agent in instances) == expected
    for key in ("total_value", "total_cost", "wins", "budget_utilization", "assignment_fingerprint"):
        assert first[key] == second[key]
    assert len(slots) == len(repeated) == 72
    assert slots[0]["alpha"] == repeated[0]["alpha"] == 3.
    assert [s["alpha"] for s in slots] == [s["alpha"] for s in repeated]
    assert all(np.isfinite(s["alpha"]) and s["alpha"] >= 0 for s in slots)
