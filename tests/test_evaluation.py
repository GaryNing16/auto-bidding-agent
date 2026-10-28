"""Hand-calculated scores and independent evaluation lifecycle acceptance cases."""
import copy
import json
import math
from pathlib import Path

import pytest

from harmony_bidding.evaluation import (
    EvaluationError, UndefinedReferenceError, _run_fresh_episode,
    evaluate, relative_scores, rotation_schedule, score_ratios,
)
from harmony_bidding.runner import business_result
from harmony_bidding.schema import Advertiser, DayData, Opportunity


def example_day(day_id="2026-01-01", total_slots=2):
    advertisers = tuple(Advertiser(a, "local_fixture", 100.0) for a in ("a", "b", "c"))
    # Every role has a hand-visible unique winner, so each PID denominator is 10.
    values = ((10.0, 1.0, 1.0), (1.0, 10.0, 1.0), (1.0, 1.0, 10.0))
    first = tuple(Opportunity(day_id, 0, f"r{i}", i, v, (True,) * 3) for i, v in enumerate(values))
    return DayData(day_id, advertisers, (first,) + ((),) * (total_slots - 1))


def config(roles=("a", "b", "c"), teams=None, isolation="in_process"):
    return {
        "rules": {"total_slots": 2, "tie_seed": 123},
        "pid": {"kp": 0.2, "ki": 0.01, "kd": 0.02, "initial_alpha": 1.0},
        "evaluation": {
            "roles": list(roles), "isolation": isolation,
            "teams": teams if teams is not None else {"A": {"kind": "pid"}, "B": {"kind": "pid"}, "C": {"kind": "pid"}},
        },
    }


def game_business(report):
    return {(r["day_id"], r["team"], r["advertiser_id"]):
            (r["value"], r["cost"], r["pid_reference_value"], r["value_ratio"])
            for r in report["games"]}


def only_run(root):
    directories = [p for p in Path(root).iterdir() if p.is_dir()]
    assert len(directories) == 1
    return directories[0]


def test_hand_calculated_ratio_mean_and_relative_weight():
    result = score_ratios([120, 900], [100, 1000], expected_count=2)
    assert result["ratios"] == [1.2, 0.9]
    assert result["G"] == pytest.approx(1.05)
    assert result["G"] != pytest.approx(1020 / 1100)
    assert relative_scores({"ours": result["G"], "best": 1.20}, 40)["ours"] == pytest.approx(35)


def test_all_zero_distinct_from_zero_reference_missing_and_exception():
    assert score_ratios([0, 0], [100, 1000])["G"] == 0
    assert relative_scores({"A": 0, "B": 0}, 40) == {"A": 0, "B": 0}
    with pytest.raises(UndefinedReferenceError, match="normalization undefined"):
        score_ratios([0], [0])
    with pytest.raises(EvaluationError, match="missing games"):
        score_ratios([1], [1], expected_count=2)
    with pytest.raises(EvaluationError):
        score_ratios([1], [])
    with pytest.raises(EvaluationError):
        score_ratios([math.nan], [1])
    debug = score_ratios([0, 2], [0, 1], diagnostic_undefined=True)
    assert debug["ratios"] == [None, 2]
    assert debug["G"] is None and debug["valid"] is False
    assert relative_scores({"A": None, "B": 1}, 15) == {"A": None, "B": None}


@pytest.mark.parametrize("values,references", [([-1], [1]), ([1], [-1]), ([1], [math.inf]), ([True], [1]), (["1"], [1])])
def test_invalid_scoring_inputs_raise(values, references):
    with pytest.raises(EvaluationError):
        score_ratios(values, references)


def test_nonfinite_ratio_cannot_be_reported_as_valid():
    with pytest.raises(EvaluationError, match="nonfinite value ratio"):
        score_ratios([1e308], [1e-308])
    assert relative_scores({"A": 1e308}, 40) == {"A": 40}
    assert score_ratios([1e308, 1e308], [1, 1])["G"] == 1e308


def test_multi_rotation_exact_order_and_each_role_once():
    schedule = rotation_schedule(["a", "b", "c"], ["A", "B", "C"])
    assert schedule == [
        {"a": "A", "b": "B", "c": "C"},
        {"a": "C", "b": "A", "c": "B"},
        {"a": "B", "b": "C", "c": "A"},
    ]
    for team in ("A", "B", "C"):
        assert sorted(role for assignment in schedule for role, assigned in assignment.items() if assigned == team) == ["a", "b", "c"]
    assert rotation_schedule(["a"], ["A"]) == [{"a": "A"}]
    with pytest.raises(EvaluationError):
        rotation_schedule(["a"], ["A", "B"])
    with pytest.raises(EvaluationError):
        rotation_schedule(["a", "a"], ["A", "B"])


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_same_pid_replacement_equal_reference_and_readable_reports(mode, tmp_path):
    original_config = config()
    snapshot = copy.deepcopy(original_config)
    report = evaluate([example_day()], original_config, mode, tmp_path / mode)
    assert original_config == snapshot
    assert report["valid"] is True
    assert len(report["games"]) == 9
    assert report["references"][0]["values"] == {"a": 10, "b": 10, "c": 10}
    assert all(row["value_ratio"] == 1 for row in report["games"])
    assert all(team["G"] == 1 for team in report["teams"].values())
    assert all(team["local_relative_score"] == (40 if mode == "single" else 15) for team in report["teams"].values())
    directory = Path(report["report_directory"])
    reloaded = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    assert reloaded == report
    assert report["rules_version"] == "harmony-A-B-local-v2"
    assert report["assumption_profile"] == "local-unified-v2"
    for episode in report["episodes"]:
        result = json.loads((directory / episode["episode_id"] / "episode.json").read_text(encoding="utf-8"))
        assert len(result["slots"]) == 6
        assert result["checks"] == {"legal": True, "budget": True}
        assert (directory / episode["episode_id"] / "slots.csv").exists()


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_repeated_runs_and_date_order_permutation_do_not_carry_feedback(mode):
    days = [example_day("2026-01-01"), example_day("2026-01-02")]
    original_config = config()
    first = evaluate(days, original_config, mode)
    repeated = evaluate(days, original_config, mode)
    reordered = evaluate(list(reversed(days)), original_config, mode)
    assert game_business(first) == game_business(repeated) == game_business(reordered)
    assert first["teams"] == repeated["teams"] == reordered["teams"]


def test_single_team_match_order_permutation_is_invariant():
    teams = {"aggressive": {"kind": "constant", "alpha": 2}, "zero": {"kind": "zero"}, "pid": {"kind": "pid"}}
    forward = evaluate([example_day()], config(teams=teams), "single")
    backward = evaluate([example_day()], config(teams=dict(reversed(list(teams.items())))), "single")
    assert game_business(forward) == game_business(backward)
    assert forward["teams"] == backward["teams"]


def test_generator_dates_consumed_once_and_scored_completely():
    seen = []

    def dates():
        for day_id in ("2026-01-01", "2026-01-02"):
            seen.append(day_id)
            yield example_day(day_id)

    report = evaluate(dates(), config(["a"], {"A": {"kind": "pid"}}), "single")
    assert seen == ["2026-01-01", "2026-01-02"]
    assert report["dates_evaluated"] == 2
    assert report["teams"]["A"]["games_expected"] == 2
    assert report["teams"]["A"]["G"] == 1


def test_empty_generator_records_failure(tmp_path):
    with pytest.raises(EvaluationError, match="at least one complete date"):
        evaluate(iter(()), config(["a"], {"A": {"kind": "pid"}}), "single", tmp_path)
    failure = json.loads((only_run(tmp_path) / "evaluation_error.json").read_text(encoding="utf-8"))
    assert failure["valid"] is False
    assert failure["completed_games"] == []


def test_duplicate_day_in_generator_records_incomplete_failure(tmp_path):
    with pytest.raises(EvaluationError, match="duplicate day"):
        evaluate((example_day() for _ in range(2)), config(["a"], {"A": {"kind": "pid"}}), "single", tmp_path)
    failure = json.loads((only_run(tmp_path) / "evaluation_error.json").read_text(encoding="utf-8"))
    assert len(failure["completed_games"]) == 1
    assert failure["valid"] is False


@pytest.mark.parametrize("d", [1, 3])
def test_multi_boundaries_d_one_and_d_n(d):
    roles = ["a", "b", "c"][:d]
    teams = {f"team{i}": {"kind": "pid"} for i in range(d)}
    report = evaluate([example_day()], config(roles, teams), "multi")
    assert len(report["games"]) == d * d
    assert len(report["schedule"]) == d
    for team in teams:
        assert sorted(r["advertiser_id"] for r in report["games"] if r["team"] == team) == sorted(roles)


def test_zero_reference_strict_error_and_diagnostic_keeps_all_games(tmp_path):
    day = example_day()
    zero_day = DayData(day.day_id, day.advertisers, ((), ()))
    cfg = config(["a"], {"A": {"kind": "zero"}})
    with pytest.raises(UndefinedReferenceError):
        evaluate([zero_day], cfg, "single", tmp_path / "strict")
    strict_dir = only_run(tmp_path / "strict")
    failure = json.loads((strict_dir / "evaluation_error.json").read_text(encoding="utf-8"))
    assert failure["type"] == "UndefinedReferenceError"
    assert failure["valid"] is False
    assert json.loads((strict_dir / "evaluation.json").read_text(encoding="utf-8"))["valid"] is False
    assert (strict_dir / "single_date000_match000" / "episode.json").exists()
    cfg["evaluation"]["diagnostic_undefined"] = True
    debug = evaluate([zero_day], cfg, "single", tmp_path / "debug")
    assert debug["games"][0]["value_ratio"] is None
    assert debug["teams"]["A"]["G"] is None
    assert debug["teams"]["A"]["games_recorded"] == 1
    assert debug["teams"]["A"]["local_relative_score"] is None


def test_failed_episode_persists_failure_and_cannot_score(tmp_path):
    cfg = config(["a"], {"bad": {"kind": "constant", "alpha": -1}})
    with pytest.raises(Exception, match="bidding failed"):
        evaluate([example_day()], cfg, "single", tmp_path)
    directory = only_run(tmp_path)
    error = json.loads((directory / "evaluation_error.json").read_text(encoding="utf-8"))
    assert error["valid"] is False
    failed_episode = json.loads((directory / error["failed_episode_id"] / "episode.json").read_text(encoding="utf-8"))
    assert failed_episode["exception_count"] == 1
    assert failed_episode["checks"]["legal"] is False


def test_reused_output_directory_cannot_leave_old_valid_score(tmp_path):
    good_config = config(["a"], {"A": {"kind": "pid"}})
    good = evaluate([example_day()], good_config, "single", tmp_path)
    good_directory = Path(good["report_directory"])
    assert json.loads((good_directory / "evaluation.json").read_text(encoding="utf-8"))["valid"] is True
    bad_config = config(["a"], {"A": {"kind": "constant", "alpha": -1}})
    with pytest.raises(Exception):
        evaluate([example_day()], bad_config, "single", tmp_path)
    failed_directory = next(p for p in tmp_path.iterdir() if p != good_directory)
    current = json.loads((failed_directory / "evaluation.json").read_text(encoding="utf-8"))
    assert current["valid"] is False
    assert "teams" not in current
    assert current == json.loads((failed_directory / "evaluation_error.json").read_text(encoding="utf-8"))
    assert current["run_id"] != good["run_id"]
    assert json.loads((good_directory / "status.json").read_text())["status"] == "completed"
    assert json.loads((failed_directory / "status.json").read_text())["status"] == "failed"


def test_missing_slots_or_exception_cannot_silently_be_averaged(monkeypatch):
    import harmony_bidding.evaluation as evaluation
    real_run = evaluation._run_fresh_episode

    def broken(*args, **kwargs):
        result = real_run(*args, **kwargs)
        result["slots"].pop()
        return result

    monkeypatch.setattr(evaluation, "_run_fresh_episode", broken)
    with pytest.raises(EvaluationError, match="missing, duplicate, or unexpected slots"):
        evaluate([example_day()], config(["a"], {"A": {"kind": "pid"}}), "single")


def test_even_sub_nanounit_budget_overspend_is_rejected(monkeypatch):
    import harmony_bidding.evaluation as evaluation
    real_run = evaluation._run_fresh_episode

    def overspent(*args, **kwargs):
        result = real_run(*args, **kwargs)
        result["summaries"]["a"]["cost"] = result["summaries"]["a"]["budget"] + 1e-10
        return result

    monkeypatch.setattr(evaluation, "_run_fresh_episode", overspent)
    with pytest.raises(EvaluationError, match="budget exceeded"):
        evaluate([example_day()], config(["a"], {"A": {"kind": "pid"}}), "single")


def test_sealed_assets_cannot_change_between_episodes(monkeypatch, tmp_path):
    import harmony_bidding.evaluation as evaluation
    asset = tmp_path / "sealed_weights.json"
    asset.write_text('{"alpha": 1}', encoding="utf-8")
    cfg = config(["a"], {"A": {"kind": "pid"}})
    cfg["sealed_model_files"] = [str(asset)]
    real_run = evaluation._run_fresh_episode

    def alter_after_run(*args, **kwargs):
        result = real_run(*args, **kwargs)
        asset.write_text('{"alpha": 2}', encoding="utf-8")
        return result

    monkeypatch.setattr(evaluation, "_run_fresh_episode", alter_after_run)
    with pytest.raises(EvaluationError, match="sealed model/source assets changed"):
        evaluate([example_day()], cfg, "single", tmp_path / "results")
    failure = json.loads((only_run(tmp_path / "results") / "evaluation.json").read_text(encoding="utf-8"))
    assert failure["valid"] is False
    assert failure["sealed_assets"]


def test_pid_reference_kind_cannot_be_overridden():
    cfg = config(["a"], {"A": {"kind": "pid"}})
    cfg["pid"]["kind"] = "constant"
    with pytest.raises(EvaluationError, match="cannot override"):
        evaluate([example_day()], cfg, "single")


def test_process_isolation_resets_mutable_module_state(tmp_path, monkeypatch):
    # Intentional bad-reset plugin demonstrates why creating instances alone is
    # insufficient. Each process import starts with pristine offline module state.
    module = tmp_path / "mutable_evaluation_test_agent.py"
    module.write_text(
        "from harmony_bidding.agents import ConstantAlphaAgent\n"
        "counter = 0\n"
        "class MutableAgent(ConstantAlphaAgent):\n"
        "    def __init__(self):\n"
        "        global counter\n"
        "        counter += 1\n"
        "        super().__init__(counter)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = config(["a"], {"A": {"kind": "custom", "class_path": "mutable_evaluation_test_agent:MutableAgent"}}, isolation="process")
    specs = cfg["evaluation"]["teams"]
    first = _run_fresh_episode(example_day(), cfg, {"a": "A"}, specs, "same", "single")
    second = _run_fresh_episode(example_day(), cfg, {"a": "A"}, specs, "same", "single")
    assert business_result(first) == business_result(second)
    assert [r["alpha"] for r in first["slots"] if r["advertiser_id"] == "a"] == [1, 1]
    cfg["evaluation"]["isolation"] = "in_process"
    trusted_first = _run_fresh_episode(example_day(), cfg, {"a": "A"}, specs, "same", "single")
    trusted_second = _run_fresh_episode(example_day(), cfg, {"a": "A"}, specs, "same", "single")
    assert [r["alpha"] for r in trusted_first["slots"] if r["advertiser_id"] == "a"] == [1, 1]
    assert [r["alpha"] for r in trusted_second["slots"] if r["advertiser_id"] == "a"] == [2, 2]


def test_process_mode_preserves_failed_episode_report(tmp_path):
    cfg = config(["a"], {"bad": {"kind": "constant", "alpha": -1}}, isolation="process")
    with pytest.raises(EvaluationError, match="EpisodeFailure"):
        evaluate([example_day()], cfg, "single", tmp_path)
    directory = only_run(tmp_path)
    error = json.loads((directory / "evaluation_error.json").read_text(encoding="utf-8"))
    assert error["failed_episode_id"] == "single_date000_match000"
    episode = json.loads((directory / error["failed_episode_id"] / "episode.json").read_text(encoding="utf-8"))
    assert episode["exceptions"][0]["stage"] == "bidding"


def test_invalid_evaluation_roles_dates_and_modes():
    day = example_day()
    with pytest.raises(EvaluationError):
        evaluate([], config(), "single")
    with pytest.raises(EvaluationError, match="duplicate day"):
        evaluate([day, day], config(), "single")
    with pytest.raises(EvaluationError, match="unknown test role"):
        evaluate([day], config(["missing"], {"A": {"kind": "pid"}}), "single")
    with pytest.raises(EvaluationError):
        evaluate([day], config(), "official")


@pytest.mark.parametrize("broken", ["roles", "rules", "mode", "official_reference"])
def test_preflight_failure_is_isolated_and_unscorable(tmp_path, broken):
    from harmony_bidding.runner import read_completed_run
    good = evaluate([example_day()], config(["a"], {"A": {"kind": "pid"}}), "single", tmp_path)
    cfg = config(["a"], {"A": {"kind": "pid"}})
    mode = "single"
    if broken == "roles": cfg["evaluation"]["roles"] = []
    elif broken == "rules": cfg["rules"]["alpha_max"] = 5
    elif broken == "mode": mode = "official"
    else: cfg["evaluation"]["reference_strategy"] = "official"
    with pytest.raises(Exception):
        evaluate([example_day()], cfg, mode, tmp_path)
    bad_dir = next(p for p in tmp_path.iterdir() if p.name != good["run_id"])
    with pytest.raises(ValueError, match="incomplete, failed"):
        read_completed_run(bad_dir)
    assert read_completed_run(good["report_directory"])["valid"] is True
    error = json.loads((bad_dir / "evaluation.json").read_text())
    assert not error["valid"] and "teams" not in error
    assert error["run_id"] == bad_dir.name


def test_completed_reader_rejects_foreign_report(tmp_path):
    from harmony_bidding.runner import read_completed_run
    result = evaluate([example_day()], config(["a"], {"A": {"kind": "pid"}}), "single", tmp_path)
    directory = Path(result["report_directory"])
    result["config_fingerprint"] = "another-config"
    (directory / "evaluation.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprints"):
        read_completed_run(directory)


def test_direct_daydata_evaluation_rejects_cross_date_type_conflict(tmp_path):
    from dataclasses import replace
    first, second = example_day("2026-01-01"), example_day("2026-01-02")
    second = replace(second, advertisers=(replace(second.advertisers[0], advertiser_type="changed"),
                                         *second.advertisers[1:]))
    with pytest.raises(ValueError, match="player_type conflict across dates"):
        evaluate([first, second], config(["a"], {"A": {"kind": "pid"}}), "single", tmp_path)
    directory = only_run(tmp_path)
    failure = json.loads((directory / "evaluation.json").read_text())
    assert failure["valid"] is False and "teams" not in failure
    assert json.loads((directory / "status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_completed_reader_requires_evaluation_report(tmp_path, mode):
    from harmony_bidding.runner import read_completed_run
    result = evaluate([example_day()], config(["a"], {"A": {"kind": "pid"}}), mode, tmp_path)
    directory = Path(result["report_directory"])
    assert read_completed_run(directory)["valid"] is True
    (directory / "evaluation.json").unlink()
    with pytest.raises(ValueError, match="missing required report"):
        read_completed_run(directory)
