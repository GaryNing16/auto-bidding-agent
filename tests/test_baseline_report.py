import numpy as np
import pandas as pd
import pytest
import json

from harmony_baselines.report import aggregate_results, draw_figures, audit_complete_results, write_report, algorithm_status_summary
from harmony_baselines.data import sha256_file


def fixture_rows():
    rows = []
    for scenario in ["Fixed", "PID", "Mixed"]:
        for algorithm, values in [("LocalPID", [1., 10., 0.]), ("FixedAlpha", [2., 5., 1.])]:
            for player, value in enumerate(values):
                for seed in [1, 2]:
                    rows.append(dict(algorithm=algorithm, scenario=scenario, player_id=str(player), player_type=str(player),
                        day_id="test-period", seed=seed, training_seed=0, budget_scale=1., total_value=value*seed,
                        total_cost=50., initial_budget=100., budget_utilization=.5, wins=10,
                        bidding_p99_ms=1., update_p99_ms=.1, paired_p99_ms=1.1, rss_mb=500., peak_process_rss_mb=600.))
    return pd.DataFrame(rows)


def test_role_ratios_then_equal_mean_and_no_epsilon():
    tables = aggregate_results(fixture_rows())
    s = tables["all_player_summary"].query("algorithm == 'FixedAlpha' and scenario == 'Fixed'").iloc[0]
    assert s.mean_relative_performance == pytest.approx(1.25)
    assert s.valid_reference_roles == 2
    assert s.zero_reference_roles == 1
    assert s.mean_total_value == 4.
    assert s.relative_cross_player_std == pytest.approx(np.std([2., .5], ddof=1))
    undefined = tables["undefined_references"]
    assert undefined.relative_performance.isna().all()
    assert set(undefined.relative_status) == {"both_zero", "zero_reference_positive_candidate"}
    assert tables["inference_summary"].max_episode_peak_process_rss_mb.eq(600.).all()


def test_training_seeds_are_averaged_inside_role_before_ratio():
    frame = fixture_rows()
    extra = frame[frame.algorithm == "FixedAlpha"].copy()
    extra.training_seed = 7
    extra.total_value *= 3
    tables = aggregate_results(pd.concat([frame, extra]))
    result = tables["all_player_summary"].query("algorithm == 'FixedAlpha'")
    assert result.mean_relative_performance.eq(2.5).all()
    assert result.relative_training_seed_min.eq(1.25).all()
    assert result.relative_training_seed_max.eq(3.75).all()
    assert tables["per_training_seed_summary"].query("algorithm == 'FixedAlpha'").training_seed.nunique() == 2


def test_rejects_duplicate_episodes_and_missing_reference():
    frame = fixture_rows()
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_results(pd.concat([frame, frame.iloc[:1]]))
    with pytest.raises(ValueError, match="reference missing"):
        aggregate_results(frame[frame.algorithm != "LocalPID"])


def test_four_figures_are_csv_derived_png_and_svg(tmp_path):
    tables = aggregate_results(fixture_rows())
    paths = draw_figures(tables, tmp_path)
    assert len(paths) == 4
    for path in paths.values():
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.with_suffix(".svg").stat().st_size > 1000
    protocol = {"test_roles": ["0", "1", "2"], "training_seeds": [1, 2], "evaluation_seeds": [1, 2],
                "executed_split": {"train": ["train"], "validation": ["val"], "test": ["test"]}}
    report = write_report(tmp_path, tables, protocol,
        {"selected": [{"algorithm": "FixedAlpha"}, {"algorithm": "LocalPID"}]}, paths,
        report_path=tmp_path/"synthetic_fixture_only.md")
    text = report.read_text(encoding="utf-8")
    assert "Fixed α" in text and "不是两份独立市场样本" in text
    assert "三场景平均累计价值最高为 **LocalPID**（V=5.500）" in text
    assert "逐角色归一化R最高为 **Fixed α**（R=1.250）" in text
    assert "**5.500**" in text and "**1.250**" in text


def test_formal_report_requires_completion_and_identical_opponents(tmp_path):
    frame = fixture_rows()
    frame["completed"] = True
    for field in ["dataset_fingerprint", "rules_version", "opponent_fingerprint", "assignment_fingerprint"]:
        frame[field] = "shared"
    def dump(name, value):
        (tmp_path/name).write_text(json.dumps(value), encoding="utf-8")
    dump("opponents_frozen.json", {"fixed": "test"})
    (tmp_path/"online_lp_table.csv").write_text("test-table\n", encoding="utf-8")
    dump("selected_frozen.json", {"selected": [{"algorithm": "LocalPID"}, {"algorithm": "FixedAlpha"}],
        "test_used": False, "opponents_sha256": sha256_file(tmp_path/"opponents_frozen.json"),
        "lp_table_sha256": sha256_file(tmp_path/"online_lp_table.csv"), "checkpoints": {}})
    protocol = {"status": "completed", "selection_sha256": sha256_file(tmp_path/"selected_frozen.json"),
        "test_episodes": len(frame), "executed_split": {"train": ["train-period"], "validation": ["validation-period"], "test": ["test-period"]},
        "test_roles": ["0", "1", "2"], "evaluation_seeds": [1, 2], "budget_scales": [1.]}
    dump("protocol.json", protocol)
    frame.to_csv(tmp_path/"episode_results.csv", index=False)
    dump("episode_results.status.json", {"status": "completed", "episodes": len(frame),
        "csv_sha256": sha256_file(tmp_path/"episode_results.csv")})
    assert len(audit_complete_results(tmp_path)[0]) == len(frame)
    protocol["status"] = "frozen_before_test"
    dump("protocol.json", protocol)
    with pytest.raises(ValueError, match="completed"):
        audit_complete_results(tmp_path)
    protocol["status"] = "completed"
    dump("protocol.json", protocol)
    frame.loc[0, "assignment_fingerprint"] = "different"
    frame.to_csv(tmp_path/"episode_results.csv", index=False)
    dump("episode_results.status.json", {"status": "completed", "episodes": len(frame),
        "csv_sha256": sha256_file(tmp_path/"episode_results.csv")})
    with pytest.raises(ValueError, match="fairness"):
        audit_complete_results(tmp_path)


def test_validation_failures_remain_visible_without_inventing_test_scores(tmp_path):
    (tmp_path/"models").mkdir()
    training = pd.DataFrame([dict(algorithm="CQL", config_index=0, seed=1, step=200,
                                 parameter_count=100, training_seconds=1.)])
    (tmp_path/"models"/"candidates.json").write_text(training.to_json(orient="records"), encoding="utf-8")
    (tmp_path/"models"/"failures.json").write_text("[]", encoding="utf-8")
    validation = [dict(algorithm="CQL", candidate="CQL|0|200", status="failed", exception_type="OverflowError",
                      reason="math range error", phase="validation", test_used=False),
                  dict(algorithm="CQL", candidate="CQL|0|200", status="resource_incomplete",
                      reason="full episode exceeded deadline", timeout_seconds=180, phase="validation", test_used=False)]
    (tmp_path/"validation_failures.json").write_text(json.dumps(validation), encoding="utf-8")
    (tmp_path/"validation_skipped.json").write_text(json.dumps([dict(algorithm="CQL", status="not_run")]), encoding="utf-8")
    selected = {"selected": [{"algorithm": "LocalPID"}, {"algorithm": "FixedAlpha"}],
                "excluded_candidates": [{"candidate": "CQL|0|200", "reason": "validation incomplete"}]}
    status = algorithm_status_summary(tmp_path, selected, {"LocalPID", "FixedAlpha"}, training).set_index("algorithm")
    assert status.loc["CQL", "status"] == "已训练，验证失败/资源未完成，未排名"
    assert status.loc["CQL", "validation_failed_cases"] == 1
    assert status.loc["CQL", "validation_resource_incomplete_cases"] == 1
    assert status.loc["CQL", "validation_not_run_cases"] == 1
    assert status.loc["CQL", "excluded_candidate_count"] == 1
    tables = aggregate_results(fixture_rows())
    protocol = {"test_roles": ["0", "1", "2"], "training_seeds": [1, 2], "evaluation_seeds": [1, 2],
                "executed_split": {"train": ["train"], "validation": ["val"], "test": ["test"]}}
    figures = {k: tmp_path/(k+".png") for k in ["01_performance", "02_budget", "03_player_types", "04_robustness"]}
    report = write_report(tmp_path, tables, protocol, selected, figures, report_path=tmp_path/"fixture.md")
    content = report.read_text(encoding="utf-8")
    assert "已训练，验证失败/资源未完成，未排名" in content
    assert "验证异常1局（OverflowError×1）" in content
    assert "CQL" not in set(tables["all_player_summary"].algorithm)


@pytest.mark.parametrize("failed_algorithm", ["FixedAlpha", "LocalPID"])
def test_test_failure_excludes_entire_algorithm_without_replacing_checkpoint(tmp_path, failed_algorithm):
    frame = fixture_rows()
    frame["task_index"] = np.arange(len(frame))
    frame["completed"] = True
    for field in ["dataset_fingerprint", "rules_version", "opponent_fingerprint", "assignment_fingerprint"]:
        frame[field] = "shared"
    missing = frame[frame.algorithm == failed_algorithm].iloc[0]
    remaining = frame[frame.task_index != missing.task_index].copy()
    def dump(name, value):
        (tmp_path/name).write_text(json.dumps(value), encoding="utf-8")
    dump("opponents_frozen.json", {"fixed": "fixture"})
    (tmp_path/"online_lp_table.csv").write_text("fixture\n", encoding="utf-8")
    dump("selected_frozen.json", {"selected": [{"algorithm": "LocalPID"}, {"algorithm": "FixedAlpha"}],
        "test_used": False, "opponents_sha256": sha256_file(tmp_path/"opponents_frozen.json"),
        "lp_table_sha256": sha256_file(tmp_path/"online_lp_table.csv"), "checkpoints": {}})
    protocol = {"status": "completed_with_failures", "selection_sha256": sha256_file(tmp_path/"selected_frozen.json"),
        "test_episodes": len(remaining), "test_expected_episodes": len(frame), "test_failure_episodes": 1, "test_skipped_episodes": 0,
        "incomplete_algorithms": [failed_algorithm],
        "executed_split": {"train": ["train-period"], "validation": ["validation-period"], "test": ["test-period"]},
        "test_roles": ["0", "1", "2"], "evaluation_seeds": [1, 2], "budget_scales": [1.]}
    dump("protocol.json", protocol)
    remaining.to_csv(tmp_path/"episode_results.csv", index=False)
    dump("episode_results.status.json", {"status": "completed_with_failures", "episodes": len(remaining),
        "expected_episodes": len(frame), "failure_episodes": 1, "skipped_episodes": 0,
        "csv_sha256": sha256_file(tmp_path/"episode_results.csv")})
    failure = {"task_index": int(missing.task_index), "algorithm": failed_algorithm, "training_seed": int(missing.training_seed),
        "day_id": missing.day_id, "scenario": missing.scenario, "player_id": missing.player_id, "evaluation_seed": int(missing.seed),
        "status": "failed", "reason": "fixture failure", "phase": "test", "test_used": True}
    dump("evaluation_failures.json", [failure]); dump("evaluation_skipped.json", [])
    if failed_algorithm == "LocalPID":
        with pytest.raises(ValueError, match="reference incomplete"):
            audit_complete_results(tmp_path)
    else:
        ranked, audited, selection = audit_complete_results(tmp_path)
        assert set(ranked.algorithm) == {"LocalPID"}
        assert len(selection["selected"]) == 2
        assert len(pd.read_csv(tmp_path/"episode_results.csv")) == len(remaining)
        assert audited["_report_audit"]["coverage_by_algorithm"][failed_algorithm] == {"expected": 18, "completed": 17, "missing": 1}
        dump("evaluation_failures.json", [])
        with pytest.raises(ValueError, match="accounted for"):
            audit_complete_results(tmp_path)
