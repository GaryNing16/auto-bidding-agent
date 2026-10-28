"""Independent integration checks for the user-described six-field adapter.

All source rows used here are hand-authored schema fixtures, never organizer data.
Expected payments/rewards below are calculated directly, not by another engine.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict
from pathlib import Path

import pytest

from harmony_bidding.agents import ConstantAlphaAgent
from harmony_bidding.auction import AuctionEngine
from harmony_bidding.data import load_days
from harmony_bidding.organizer_data import SOURCE_FIELDS, import_organizer_csv
from harmony_bidding.runner import run_episode
from harmony_bidding.schema import Opportunity, Rules, load_config


ROOT = Path(__file__).resolve().parents[1]
PLAYER_IDS = ("001", "002", "010")
PLAYER_TYPES = {"001": "7", "002": "2", "010": "A"}


class CaptureAgent(ConstantAlphaAgent):
    def reset(self, own_config, public_config):
        super().reset(own_config, public_config)
        self.observations = []
        self.feedback = []

    def bidding(self, observation):
        self.observations.append(asdict(observation))
        return super().bidding(observation)

    def update(self, own_feedback):
        self.feedback.append(asdict(own_feedback))
        super().update(own_feedback)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOURCE_FIELDS)
        writer.writerows(rows)


def request_rows(request_id, values, time_step=1, indexes=(90, 500, 2)):
    return [[index, time_step, request_id, player, PLAYER_TYPES[player], value]
            for index, player, value in zip(indexes, PLAYER_IDS, values)]


def fixture_config(tmp_path, rows, name="fixture", budgets=None):
    source = tmp_path / f"{name}.csv"
    write_rows(source, rows)
    budgets = budgets or dict.fromkeys(PLAYER_IDS, 100)
    return {
        "data_dir": str(tmp_path / f"{name}_imported"),
        "rules": {"total_slots": 72, "assumption_profile": "local-organizer-schema-v2"},
        "expected_n": 3,
        "agent_seed": 1234,
        "organizer_import": {
            "source_kind": "schema_fixture",
            "sources": [{"path": str(source), "day_id": "2026-10-20"}],
            "time_step_base": 1,
            "request_id_scope": "day",
            "ordering": {"mode": "file_first_seen"},
            "players": {player: {"player_type": PLAYER_TYPES[player], "budget": budgets[player]}
                        for player in PLAYER_IDS},
        },
    }


def import_one(config):
    manifest = import_organizer_csv(config)
    days = list(load_days(config["data_dir"], Rules(**config["rules"]), expected_n=3))
    assert len(days) == 1
    runtime_config = copy.deepcopy(config)
    # This is exactly the CLI's evaluator-only metadata route.
    runtime_config["_dataset_import_metadata"] = manifest["import_metadata"]
    return days[0], runtime_config, manifest


def run_capture(day, config, alphas=(1.2, 1, 1)):
    agents = {player: CaptureAgent(alpha) for player, alpha in zip(PLAYER_IDS, alphas)}
    result = run_episode(day, config, episode_id="six-field-boundary", mode="test",
                         agent_factories={player: (lambda player=player: agents[player]) for player in PLAYER_IDS})
    return result, agents


def nested_keys(value):
    if isinstance(value, dict):
        yield from value
        for item in value.values():
            yield from nested_keys(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from nested_keys(item)


def test_repository_fixture_manual_second_price_and_full_72_slot_lifecycle(tmp_path):
    config = load_config(ROOT / "configs" / "organizer_example.yaml")
    config["data_dir"] = str(tmp_path / "repository_fixture")
    for source in config["organizer_import"]["sources"]:
        source["path"] = str(ROOT / source["path"])
    manifest = import_organizer_csv(config)
    config["_dataset_import_metadata"] = manifest["import_metadata"]
    days = list(load_days(config["data_dir"], Rules(**config["rules"]), expected_n=3))
    assert manifest["source_kind"] == "schema_fixture"
    assert [entry["split"] for entry in manifest["days"]] == ["train", "validation", "local_holdout"]
    assert [sum(map(len, day.slots)) for day in days] == [4, 3, 3]
    day = days[0]
    assert tuple(a.advertiser_id for a in day.advertisers) == PLAYER_IDS
    assert tuple(a.advertiser_type for a in day.advertisers) == ("7", "2", "A")
    assert [op.opportunity_id for op in day.slots[0]] == ["r20", "r03"]
    assert day.slots[0][0].values == (10, 8, 11)

    # A bids 12, B bids 8, C bids 11; one request gives A value 10 at cost 11.
    engine = AuctionEngine(day.advertisers, Rules(**config["rules"]))
    stats, debug = engine.run_slot((day.slots[0][0],), dict(zip(PLAYER_IDS, (1.2, 1, 1))))
    assert debug[0]["winner"] == "001"
    assert engine.money.amount(stats["001"]["cost_units"]) == 11
    assert stats["001"]["value"] == 10
    assert stats["001"]["wins"] == 1
    assert stats["002"] == {"wins": 0, "cost_units": 0, "value": 0, "opportunity_count": 1}
    assert stats["010"] == {"wins": 0, "cost_units": 0, "value": 0, "opportunity_count": 1}

    result, agents = run_capture(day, config)
    assert len(result["slots"]) == 3 * 72
    # r20: (cost 11, value 10); r03: (2,3); r02: (2,5).
    assert result["summaries"]["001"]["value"] == 18
    assert result["summaries"]["001"]["cost"] == 15
    assert result["summaries"]["001"]["wins"] == 3
    assert result["summaries"]["002"]["value"] == 9
    assert result["summaries"]["002"]["cost"] == 2
    assert result["summaries"]["010"]["value"] == 0
    for player, agent in agents.items():
        assert len(agent.observations) == len(agent.feedback) == 72
        assert [row["time_slot"] for row in agent.observations] == list(range(72))
        assert agent.feedback[1]["wins"] == agent.feedback[1]["cost"] == agent.feedback[1]["value"] == 0
        assert result["summaries"][player]["done"] is True
        transitions = [row for row in result["trajectories"] if row["advertiser_id"] == player]
        assert [row["observation_t"] for row in transitions] == agent.observations
        assert [row["done"] for row in transitions] == [False] * 71 + [True]
        assert transitions[-1]["next_observation"]["time_slot"] == 72


def test_current_and_future_source_values_counts_and_opponents_are_invisible(tmp_path):
    # Slot zero is empty in both worlds. At the next decision both visible
    # histories are equal, although the *current* and future source tables differ.
    first_rows = request_rows("current", (10, 8, 11), time_step=2)
    second_rows = (request_rows("current", (1000, 4, 2), time_step=2)
                   + request_rows("extra", (9, 3, 7), time_step=2)
                   + request_rows("future", (200, 90, 30), time_step=72))
    first_day, first_cfg, first_manifest = import_one(fixture_config(tmp_path, first_rows, "first"))
    second_day, second_cfg, second_manifest = import_one(fixture_config(
        tmp_path, second_rows, "second", {"001": 100, "002": 500, "010": 1}))
    assert first_manifest["import_metadata"]["request_count"] == 1
    assert second_manifest["import_metadata"]["request_count"] == 3
    first_result, first_agents = run_capture(first_day, first_cfg)
    second_result, second_agents = run_capture(second_day, second_cfg)
    for slot in (0, 1):
        assert first_agents["001"].observations[slot] == second_agents["001"].observations[slot]
        assert first_result["slots"][slot * 3]["alpha"] == second_result["slots"][slot * 3]["alpha"] == 1.2
    observation = first_agents["001"].observations[1]
    assert set(observation) == {"own_config", "time_slot", "total_slots", "remaining_budget", "history"}
    assert set(observation["own_config"]) == {"advertiser_id", "advertiser_type", "initial_budget", "random_seed"}
    assert set(observation["history"][0]) == {"time_slot", "alpha", "wins", "cost", "value", "remaining_budget", "cumulative_cost", "cumulative_value"}
    for result, manifest in ((first_result, first_manifest), (second_result, second_manifest)):
        assert result["metadata"]["data_import"] == manifest["import_metadata"]
        forbidden = {"data_import", "_dataset_import_metadata", "source_files", "row_count", "request_count",
                     "pvalue", "index", "request_id", "ordering", "organizer_import", "participation_mask", "opportunity_count"}
        assert not forbidden.intersection(nested_keys(result["trajectories"]))


def test_budget_outcome_tracks_first_seen_order_not_index_or_request_id(tmp_path):
    first = request_rows("z-first", (10, 5, 0), indexes=(90, 500, 8))
    second = request_rows("a-second", (100, 5, 0), indexes=(1, 2, 3))
    # Interleaved long-table rows: neither numeric index order nor request-ID
    # lexical order matches the request's first physical appearance.
    rows = [first[0], second[0], first[1], second[1], first[2], second[2]]
    budgets = {"001": 5, "002": 100, "010": 100}
    day, config, _ = import_one(fixture_config(tmp_path, rows, "forward", budgets))
    assert [op.opportunity_id for op in day.slots[0]] == ["z-first", "a-second"]
    assert [op.original_order for op in day.slots[0]] == [0, 1]
    result, _ = run_capture(day, config, alphas=(1, 1, 0))
    assert result["summaries"]["001"]["value"] == 10
    assert result["summaries"]["001"]["cost"] == 5
    assert result["summaries"]["001"]["remaining_budget"] == 0
    assert result["summaries"]["001"]["bidding_calls"] == 72

    reordered = [second[0], first[0], first[1], second[1], first[2], second[2]]
    reverse_day, reverse_config, _ = import_one(fixture_config(tmp_path, reordered, "reversed", budgets))
    reverse_result, _ = run_capture(reverse_day, reverse_config, alphas=(1, 1, 0))
    assert [op.opportunity_id for op in reverse_day.slots[0]] == ["a-second", "z-first"]
    assert reverse_result["summaries"]["001"]["value"] == 100
    assert reverse_result["summaries"]["001"]["cost"] == 5


def test_explicit_order_controls_budget_semantics_without_using_source_index(tmp_path):
    rows = request_rows("z-first", (10, 5, 0)) + request_rows("a-second", (100, 5, 0), indexes=(0, 1, 2))
    config = fixture_config(tmp_path, rows, budgets={"001": 5, "002": 100, "010": 100})
    order_csv = tmp_path / "local_order.csv"
    order_csv.write_text("day_id,time_step,request_id,original_order\n"
                         "2026-10-20,1,z-first,9\n2026-10-20,1,a-second,4\n", encoding="utf-8")
    config["organizer_import"]["ordering"] = {"mode": "explicit_csv", "path": str(order_csv)}
    day, runtime_config, manifest = import_one(config)
    assert [op.opportunity_id for op in day.slots[0]] == ["a-second", "z-first"]
    assert [op.original_order for op in day.slots[0]] == [4, 9]
    result, _ = run_capture(day, runtime_config, alphas=(1, 1, 0))
    assert result["summaries"]["001"]["value"] == 100
    assert result["summaries"]["001"]["cost"] == 5
    assert manifest["import_metadata"]["ordering"]["is_local_assumption"] is True


@pytest.mark.parametrize("missing_player", PLAYER_IDS)
def test_only_recorded_players_participate_in_same_request(tmp_path, missing_player):
    rows = [row for row in request_rows("incomplete", (10, 8, 11)) if row[3] != missing_player]
    config = fixture_config(tmp_path, rows)
    day, runtime, _ = import_one(config)
    opportunity = day.slots[0][0]
    index = PLAYER_IDS.index(missing_player)
    assert opportunity.values[index] == 0 and opportunity.participation_mask[index] is False
    stats, debug = AuctionEngine(day.advertisers, Rules(**runtime["rules"])).run_slot(
        (opportunity,), dict(zip(PLAYER_IDS, (1.2, 1, 1))))
    assert stats[missing_player]["opportunity_count"] == stats[missing_player]["wins"] == 0
    assert debug[0]["winner"] != missing_player


def test_sparse_csv_and_explicit_dense_mask_use_same_engine_and_manual_truth(tmp_path):
    rows = (request_rows("r1", (10, 8, 11))
            + request_rows("r2", (3, 2, 0))[:2]
            + request_rows("r3", (4, 0, 0))[:1])
    day, runtime, manifest = import_one(fixture_config(tmp_path, rows, budgets=dict.fromkeys(PLAYER_IDS, 20)))
    expected_masks = ((True, True, True), (True, True, False), (True, False, False))
    explicit = tuple(Opportunity(day.day_id, 0, request, order, values, mask)
                     for order, (request, values, mask) in enumerate(zip(
                         ("r1", "r2", "r3"), ((10., 8., 11.), (3., 2., 0.), (4., 0., 0.)), expected_masks)))
    assert day.slots[0] == explicit
    assert manifest["days"][0]["opportunity_counts_by_player"] == dict(zip(PLAYER_IDS, (3, 2, 1)))
    actions = dict(zip(PLAYER_IDS, (1.2, 1, 1)))
    imported_engine = AuctionEngine(day.advertisers, Rules(**runtime["rules"]))
    matrix_engine = AuctionEngine(day.advertisers, Rules(**runtime["rules"]))
    imported_result = imported_engine.run_slot(day.slots[0], actions)
    assert imported_result == matrix_engine.run_slot(explicit, actions)
    stats, debug = imported_result
    assert stats["001"]["value"] == 17 and stats["001"]["cost_units"] == 13_000_000
    assert [row["payment_units"] for row in debug] == [11_000_000, 2_000_000, 0]
    assert imported_engine.remaining == matrix_engine.remaining


def test_absent_all_day_role_and_sparse_future_are_invisible_to_actor(tmp_path):
    first = request_rows("current", (1, 0, 0), time_step=2)[:1]
    second = request_rows("current", (9, 0, 0), time_step=2)[:1] + request_rows("late", (50, 20, 0), time_step=72)[:2]
    day_a, cfg_a, _ = import_one(fixture_config(tmp_path, first, "sparse_a"))
    day_b, cfg_b, _ = import_one(fixture_config(tmp_path, second, "sparse_b"))
    result_a, agents_a = run_capture(day_a, cfg_a, alphas=(40, 100, 5))
    result_b, agents_b = run_capture(day_b, cfg_b, alphas=(40, 100, 5))
    for slot in (0, 1):
        assert agents_a["001"].observations[slot] == agents_b["001"].observations[slot]
    for result, agents in ((result_a, agents_a), (result_b, agents_b)):
        absent = agents["010"]
        assert len(absent.observations) == len(absent.feedback) == 72
        assert all(row["wins"] == row["cost"] == row["value"] == 0 for row in absent.feedback)
        assert result["summaries"]["010"]["done"] is True
        assert not {"participation_mask", "opportunity_count", "request_count"}.intersection(nested_keys(result["trajectories"]))
