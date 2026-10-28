from copy import deepcopy
from dataclasses import asdict, FrozenInstanceError, replace
import json
import pytest
from harmony_bidding.agent_api import Observation, OwnConfig, PublicConfig, OwnFeedback
from harmony_bidding.agents import ConstantAlphaAgent, LocalPIDAgent
from harmony_bidding.runner import run_episode, EpisodeFailure, write_episode, business_result
from harmony_bidding.schema import Advertiser, DayData, Opportunity
from harmony_bidding.trajectory import export_trajectories, read_trajectories, validate_trajectory_file


def day(slots=3, budget=10):
    return DayData("2026-01-01", (Advertiser("A", "own", budget), Advertiser("B", "other", 100)),
        tuple((Opportunity("2026-01-01", t, f"r{t}", 0, (10.0, 5.0), (True, True)),) if t != 1 else ()
              for t in range(slots)))


class RecordingAgent(ConstantAlphaAgent):
    def reset(self, own_config, public_config):
        super().reset(own_config, public_config)
        self.inputs, self.feedbacks, self.online_weights, self.cache = [], [], [1.0], {}

    def bidding(self, observation):
        self.inputs.append(asdict(observation))
        return self.configured_alpha

    def update(self, feedback):
        super().update(feedback)
        self.feedbacks.append(asdict(feedback))
        self.online_weights[0] += feedback.value
        self.cache[feedback.time_slot] = feedback.value


def test_72_slots_lifecycle_empty_and_exhausted():
    agent = RecordingAgent(2)
    result = run_episode(day(72, 5), {}, agent_factories={"A": lambda: agent})
    assert len(agent.inputs) == len(agent.feedbacks) == 72
    assert result["summaries"]["A"]["cost"] <= 5
    assert agent.feedbacks[1]["cost"] == agent.feedbacks[1]["value"] == agent.feedbacks[1]["wins"] == 0
    assert all(row["remaining_budget"] >= 0 for row in result["slots"])
    trajectory = [r for r in result["trajectories"] if r["advertiser_id"] == "A"]
    assert len(trajectory) == 72
    assert sum(r["done"] for r in trajectory) == 1
    assert trajectory[-1]["next_observation"]["time_slot"] == 72
    assert trajectory[-1]["done"] is True


def test_all_bids_before_all_updates_and_zero_budget_utilization():
    events = []
    class Sequenced(RecordingAgent):
        def bidding(self, observation):
            events.append(("bid", self.own_config.advertiser_id, observation.time_slot))
            return 0
        def update(self, feedback):
            events.append(("update", self.own_config.advertiser_id, feedback.time_slot))
    result = run_episode(day(3, 0), {"rules": {"total_slots": 3}},
                         agent_factories={"A": Sequenced, "B": Sequenced})
    assert events[:4] == [("bid", "A", 0), ("bid", "B", 0), ("update", "A", 0), ("update", "B", 0)]
    assert result["summaries"]["A"]["budget_utilization"] == 0


def test_information_boundary_equivalent_decision_points():
    first_day = day()
    second_day = DayData(first_day.day_id,
        (first_day.advertisers[0], Advertiser("SECRET_OPPONENT", "private", 9999)),
        ((Opportunity(first_day.day_id, 0, "totally-different", 0, (999, 999), (True, True)),), (), ()))
    first, second = RecordingAgent(1.2), RecordingAgent(1.2)
    first_result = run_episode(first_day, {"rules": {"total_slots": 3}}, agent_factories={"A": lambda: first})
    second_result = run_episode(second_day, {"rules": {"total_slots": 3}}, agent_factories={"A": lambda: second})
    assert first.inputs[0] == second.inputs[0]
    assert first.feedbacks[0]["alpha"] == second.feedbacks[0]["alpha"]
    assert set(first.inputs[0]) == {"own_config", "time_slot", "total_slots", "remaining_budget", "history"}
    assert set(first.inputs[0]["own_config"]) == {"advertiser_id", "advertiser_type", "initial_budget", "random_seed"}
    assert first.inputs[0]["history"] == ()
    forbidden = ("opportunity", "pValues", "opponent", "request_count", "market_price", "least_winning")
    for token in forbidden:
        assert token not in json.dumps(first.inputs)


def test_snapshots_frozen():
    own = OwnConfig("A", "x", 10, 3)
    observation = Observation(own, 0, 72, 10, ())
    with pytest.raises(FrozenInstanceError):
        observation.time_slot = 3
    with pytest.raises(FrozenInstanceError):
        own.initial_budget = 999


def test_trajectory_matches_actual_policy_input_and_export(tmp_path):
    agent = RecordingAgent(1.2)
    result = run_episode(day(), {"rules": {"total_slots": 3}}, agent_factories={"A": lambda: agent})
    rows = [r for r in result["trajectories"] if r["advertiser_id"] == "A"]
    assert [r["observation_t"] for r in rows] == agent.inputs
    assert rows[0]["observation_t"]["history"] == ()
    assert rows[0]["next_observation"] == rows[1]["observation_t"]
    assert rows[0]["reward_t"] == agent.feedbacks[0]["value"]
    assert rows[0]["reward_t"] != rows[0]["reward_t"] - rows[0]["cost_t"]
    path = tmp_path / "trajectory.jsonl"
    assert export_trajectories([result], path) == 6
    loaded = list(read_trajectories(path))
    assert len(loaded) == 6
    assert validate_trajectory_file(path) == 6
    write_episode(result, tmp_path / "report")
    report = json.loads((tmp_path / "report/episode.json").read_text(encoding="utf-8"))
    assert "trajectories" not in report and "privileged_debug" not in report
    assert report["metadata"]["rules_version"] == "harmony-A-B-local-v2"


def test_reset_weights_cache_history_and_pid_integral():
    agent = RecordingAgent(1.2)
    cfg = {"rules": {"total_slots": 3}}
    original = run_episode(day(), cfg, agent_factories={"A": lambda: agent})
    assert agent.cache and agent.online_weights[0] > 1
    repeat = run_episode(day(), cfg, agent_factories={"A": lambda: agent})
    assert business_result(original) == business_result(repeat)
    assert len(agent.inputs) == 3
    pid = LocalPIDAgent()
    own, public = OwnConfig("A", "x", 10, 1), PublicConfig(3)
    pid.reset(own, public)
    pid.update(OwnFeedback(0, 1, 0, 0, 0, 10, 0, 0))
    assert pid.integral > 0
    pid.reset(own, public)
    assert pid.integral == pid.previous_error == pid.feedback_count == 0
    assert pid.alpha == 1


def test_illegal_action_has_failure_record():
    with pytest.raises(EpisodeFailure) as error:
        run_episode(day(), {"rules": {"total_slots": 3}}, agent_factories={"A": lambda: ConstantAlphaAgent(float("nan"))})
    report = error.value.report
    assert report["checks"]["legal"] is False
    assert report["exception_count"] == 1
    assert report["exceptions"][0]["stage"] == "bidding"


def test_model_config_not_mutated_and_no_state_carryover():
    cfg = {"rules": {"total_slots": 3}, "pid": {"initial_alpha": 1.2}}
    sealed = deepcopy(cfg)
    original = run_episode(day(), cfg)
    run_episode(replace(day(), day_id="different", slots=((), (), ())), cfg)
    assert cfg == sealed
    assert business_result(original) == business_result(run_episode(day(), cfg))


@pytest.mark.parametrize("curve", [[], [False, 0.5, 1], [0.5, 0.2, 1], [0.2, 0.3, 0.8]])
def test_bad_pid_target_curves_fail(curve):
    with pytest.raises(ValueError, match="target_curve"):
        LocalPIDAgent(target_curve=curve).reset(OwnConfig("A", "x", 10, 1), PublicConfig(3))


def test_declared_model_cannot_be_written_back_during_episode(tmp_path):
    model = tmp_path / "frozen-model.json"
    model.write_text('{"weight": 1}', encoding="utf-8")
    class WritesBack(ConstantAlphaAgent):
        def update(self, feedback):
            model.write_text('{"weight": 2}', encoding="utf-8")
    with pytest.raises(EpisodeFailure, match="sealed source/model/statistics changed") as error:
        run_episode(day(), {"rules": {"total_slots": 3}, "sealed_model_files": [str(model)]},
                    agent_factories={"A": WritesBack})
    assert error.value.report["exceptions"][0]["stage"] == "sealed_asset_integrity"


@pytest.mark.parametrize("alpha", [0, 1, 5, 40, 100, 1e6])
def test_uncapped_action_reaches_all_slots(alpha):
    result = run_episode(day(), {"rules": {"total_slots": 3}},
                         agent_factories={"A": lambda: ConstantAlphaAgent(alpha)})
    assert [r["alpha"] for r in result["slots"] if r["advertiser_id"] == "A"] == [alpha] * 3
    pid = LocalPIDAgent(initial_alpha=alpha)
    pid.reset(OwnConfig("A", "x", 10, 1), PublicConfig(3))
    assert pid.alpha == alpha


@pytest.mark.parametrize("bad", [True, "1", -1, float("nan"), float("inf"), [1]])
def test_last_advertiser_invalid_action_changes_no_ledger(bad):
    with pytest.raises(EpisodeFailure) as failure:
        run_episode(day(), {"rules": {"total_slots": 3}},
                    agent_factories={"A": lambda: ConstantAlphaAgent(100), "B": lambda: ConstantAlphaAgent(bad)})
    report = failure.value.report
    assert report["failure_state"]["remaining_units"] == {"A": 10000000, "B": 100000000}
    assert report["failure_state"]["cumulative_value"] == {"A": 0, "B": 0}
    assert report["slots"] == report["timing"] == []


def test_official_interface_unavailable_without_fake_fallback():
    from harmony_bidding.agent_api import OfficialBiddingStrategy, OfficialStrategyUnavailable
    from harmony_bidding.agents import make_agent
    assert OfficialBiddingStrategy.available is False
    with pytest.raises(OfficialStrategyUnavailable):
        make_agent({"kind": "official"})
    with pytest.raises(EpisodeFailure) as failure:
        run_episode(day(), {"rules": {"total_slots": 3}, "reference_strategy": "official"})
    assert failure.value.report["summaries"] == {}
    assert failure.value.report["exceptions"][0]["type"] == "OfficialStrategyUnavailable"


def valid_rows():
    result = run_episode(day(), {"rules": {"total_slots": 3}})
    return json.loads(json.dumps(result["trajectories"]))


@pytest.mark.parametrize("mutation", [
    lambda r: r.__setitem__("done", "false"),
    lambda r: r.__setitem__("alpha_t", True),
    lambda r: r.__setitem__("reward_t", True),
    lambda r: r.__setitem__("cost_t", True),
    lambda r: r.__setitem__("advertiser_id", "other"),
    lambda r: r.__setitem__("future_count", 1),
    lambda r: r["observation_t"].__setitem__("opponent_budget", 10),
    lambda r: r["observation_t"]["own_config"].__setitem__("future_count", 3),
    lambda r: r["next_observation"]["history"][0].__setitem__("request_count", 1),
    lambda r: r["next_observation"].__setitem__("remaining_budget", 999),
    lambda r: r.__setitem__("reward_t", r["reward_t"] + 1),
    lambda r: r.__setitem__("done", True),
])
def test_trajectory_rejects_illegal_fields_types_and_state(tmp_path, mutation):
    rows = valid_rows()
    mutation(rows[0])
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        list(read_trajectories(path))


@pytest.mark.parametrize("corruption", ["missing_tail", "duplicate", "reordered", "missing_start"])
def test_trajectory_rejects_incomplete_or_discontinuous_stream(tmp_path, corruption):
    rows = valid_rows()
    if corruption == "missing_tail": rows.pop()
    elif corruption == "duplicate": rows.insert(1, deepcopy(rows[0]))
    elif corruption == "reordered": rows[0], rows[2] = rows[2], rows[0]
    else: rows.pop(0)
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_trajectory_file(path)


def test_trajectory_duplicate_json_field_and_atomic_export(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    result = run_episode(day(), {"rules": {"total_slots": 3}})
    export_trajectories([result], path)
    original = path.read_bytes()
    result["trajectories"][0]["done"] = "false"
    with pytest.raises(ValueError):
        export_trajectories([result], path)
    assert path.read_bytes() == original
    path.write_text(original.decode().replace('"done": false', '"done": false, "done": false', 1))
    with pytest.raises(ValueError, match="duplicate JSON"):
        validate_trajectory_file(path)


def test_handwritten_zero_value_win_trajectory_is_impossible(tmp_path):
    own = {"advertiser_id": "A", "advertiser_type": "app", "initial_budget": 1.0, "random_seed": 7}
    before = {"own_config": own, "time_slot": 0, "total_slots": 1, "remaining_budget": 1.0, "history": []}
    # No engine-generated expected result: a positive bid requires positive
    # pvalue, so a win with total value zero cannot be a legal transition.
    feedback = {"time_slot": 0, "alpha": 1.0, "wins": 1, "cost": 0.0, "value": 0.0,
                "remaining_budget": 1.0, "cumulative_cost": 0.0, "cumulative_value": 0.0}
    after = {**before, "time_slot": 1, "history": [feedback]}
    row = {"observation_t": before, "alpha_t": 1.0, "reward_t": 0.0, "cost_t": 0.0,
           "next_observation": after, "done": True, "episode_id": "manual", "advertiser_id": "A",
           "day_id": "d", "run_id": "manual-run", "rules_version": "harmony-A-B-local-v2",
           "assumption_profile": "local-unified-v2", "trajectory_version": "harmony-trajectory-v2", "money_decimals": 6}
    path = tmp_path / "impossible.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="winning feedback must have positive value"):
        validate_trajectory_file(path)


def test_episode_records_actual_runtime_and_tie_protocol():
    import platform
    import sys
    metadata = run_episode(day(), {"rules": {"total_slots": 3}})["metadata"]
    assert metadata["runtime"] == {"python": sys.version, "implementation": platform.python_implementation(),
                                   "executable": sys.executable}
    assert metadata["tie_protocol"] == "sha256-json-request-round-candidates/python-random-randrange-v2"
