"""Cross-layer acceptance using hand-calculated source rows and public examples."""
from copy import deepcopy
from pathlib import Path

from harmony_bidding.auction import AuctionEngine
from harmony_bidding.data import generate_synthetic, load_days
from harmony_bidding.organizer_data import import_organizer_csv
from harmony_bidding.runner import business_result, run_episode
from harmony_bidding.schema import Rules, load_config
from harmony_bidding.trajectory import export_trajectories, read_trajectories

ROOT = Path(__file__).resolve().parents[1]


def test_hand_authored_sparse_fixture_through_one_engine_and_72_slot_actor_contract(tmp_path):
    config = load_config(ROOT / "configs/sparse_example.yaml")
    config["data_dir"] = str(tmp_path / "dataset")
    for source in config["organizer_import"]["sources"]:
        source["path"] = str(ROOT / source["path"])
    import_organizer_csv(config)
    day = next(load_days(config["data_dir"], Rules(**config["rules"]), 3, "train"))
    assert [o.opportunity_id for o in day.slots[0]] == ["r100", "r050", "r003", "r900"]
    assert [o.values for o in day.slots[0]] == [(10, 8, 11), (3, 2, 0), (4, 0, 0), (0, 0, 0)]
    assert [o.participation_mask for o in day.slots[0]] == [
        (True, True, True), (True, True, False), (True, False, False), (False, False, True)]
    engine = AuctionEngine(day.advertisers, Rules(**config["rules"]))
    stats, requests = engine.run_slot(day.slots[0], {"A": 1.2, "B": 1, "C": 1})
    assert {a: row["opportunity_count"] for a, row in stats.items()} == {"A": 3, "B": 2, "C": 2}
    assert [r["winner"] for r in requests] == ["A", "A", "A", None]
    assert [engine.money.amount(r["payment_units"]) for r in requests] == [11, 2, 0, 0]
    settings = config["run"]
    result = run_episode(day, config, settings["assignments"], settings["teams"])
    a = result["summaries"]["A"]
    assert (a["wins"], a["cost"], a["value"], a["remaining_budget"]) == (3, 13, 17, 87)
    assert all(s["bidding_calls"] == s["update_calls"] == 72 and s["done"] for s in result["summaries"].values())
    assert len(result["slots"]) == 216
    path = tmp_path / "actor.jsonl"
    assert export_trajectories([result], path) == 216
    rows = list(read_trajectories(path))
    assert sum(row["done"] for row in rows) == 3
    for row in rows:
        for key in ("observation_t", "next_observation"):
            assert "participation_mask" not in row[key]
            assert "opportunity_count" not in row[key]
            assert "alpha_max" not in row[key]


def test_custom_example_uses_identical_rule_and_feedback_path_as_builtin_constant():
    config = {"rules": {"total_slots": 4}, "expected_n": 3,
              "synthetic": {"n": 3, "days": ["one"], "base_opportunities": 4,
                            "budget_min": 10.0, "budget_max": 30.0, "empty_slots": [], "burst_slots": []}}
    day = generate_synthetic(config)[0]
    assignments = {"advertiser_1": "example"}
    builtin = run_episode(day, config, assignments, {"example": {"kind": "constant", "alpha": 100.0}})
    custom = run_episode(day, deepcopy(config), assignments, {"example": {
        "kind": "custom", "class_path": "examples.constant_agent:ExampleConstantAgent", "kwargs": {"alpha": 100.0}}})
    assert business_result(custom) == business_result(builtin)
    assert custom["metadata"]["strategies"]["advertiser_1"]["class"] == "examples.constant_agent:ExampleConstantAgent"
