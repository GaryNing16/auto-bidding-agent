"""Statistical contract tests; synthetic fixtures are never report inputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harmony_baselines.five_scenarios.report import (
    ALGORITHMS, NEURAL, SCENARIOS, Sources, aggregate_phase, behavioral_statistics,
    paired_improvement, validate_phase,
    draw_figures, historical_mechanisms, training_evidence, selection_records,
    write_reports, OLD_ROOT,
    execution_evidence, resource_evidence, resource_paragraph, digest, research_findings,
    type_value_table, qa_evidence,
    gain_interpretation,
    failure_history, failure_history_paragraph,
    verify_recovery_manifest,
    recovery_prefix_evidence, recovery_prefix_paragraph,
    completed_test_evidence,
)


def test_completed_independent_audit_requires_valid_and_same_final_inputs(tmp_path):
    qa=tmp_path / "qa"; qa.mkdir()
    final=tmp_path / "independent_test"; final.mkdir()
    script=qa / "audit_independent_test_completed.py"; script.write_text("# synthetic",encoding="utf8")
    inputs={}
    for name in ("episode_results.status.json","recovery_manifest.json","episode_results.csv","slots.csv"):
        p=final / name; p.write_text("fixture",encoding="utf8"); inputs[str(p.resolve())]=digest(p)
    audit=dict(status="passed",valid=True,audit_script_sha256=digest(script),input_sha256=inputs,
        counts=dict(episodes=7680,original_successes=7534,recovered_cases=146,historical_timeouts=5,
            historical_skips=141,final_unresolved=0,policies=16,algorithms=10,scenarios=5,roles=48,
            evaluation_seeds=2,slots=552960,diagnostic_records=1658880))
    path=qa / "independent_test_completed_audit.json"
    path.write_text(json.dumps(audit),encoding="utf8")
    assert completed_test_evidence(tmp_path,Sources())["valid"] is True
    audit["valid"]=False; path.write_text(json.dumps(audit),encoding="utf8")
    with pytest.raises(ValueError,match="not passed and valid"):
        completed_test_evidence(tmp_path,Sources())
    audit["valid"]=True; path.write_text(json.dumps(audit),encoding="utf8")
    (final / "slots.csv").write_text("changed",encoding="utf8")
    with pytest.raises(ValueError,match="different final input"):
        completed_test_evidence(tmp_path,Sources())


def test_recovery_prefix_evidence_preserves_scope_and_rejects_changed_source(tmp_path):
    qa=tmp_path / "qa"; qa.mkdir()
    stage=tmp_path / "independent_test_recovery_001"; stage.mkdir()
    source=qa / "prefix_audit.py"; source.write_text("# synthetic audit fixture",encoding="utf8")
    plan=stage / "recovery_plan.json"; plan.write_text("{}",encoding="utf8")
    audit=dict(status="passed",recovery_executed_by_audit=False,maximum_numeric_difference=0,
        timing_fields_compared=False,audit_source_path=str(source),audit_source_sha256=digest(source),
        recovery_plan_sha256=digest(plan),original_failed_cases=5,original_prefix_records_compared=919,
        original_completed_feedback_slots_compared=303,original_action_records_compared=308,numeric_comparisons=2131)
    (qa / "independent_recovery_prefix_audit.json").write_text(json.dumps(audit),encoding="utf8")
    evidence=recovery_prefix_evidence(tmp_path,Sources())
    text=recovery_prefix_paragraph({"recovery_prefix_evidence":evidence})
    assert "2131" in text and "不能单独证明全部146局恢复完成" in text and "feedback没有被补造" in text
    source.write_text("# changed audit fixture",encoding="utf8")
    with pytest.raises(ValueError,match="source or frozen plan"):
        recovery_prefix_evidence(tmp_path,Sources())


def frame(scale=1., day="period-24"):
    rows = []
    for a in ALGORITHMS:
        for scenario in SCENARIOS:
            for player in ("0", "1"):
                for seed in (20261028, 20261029):
                    for train in ((20261028, 20261029) if a in NEURAL else (0,)):
                        value = (1 if player == "0" else 9) * scale
                        rows.append(dict(algorithm=a, scenario=scenario, day_id=day,
                            player_id=player, player_type=player, seed=seed, training_seed=train,
                            total_value=value, total_cost=50., initial_budget=100., budget_utilization=.5,
                            wins=10, completed=True, completed_slots=72, task_index=len(rows),
                            model_fingerprint=f"{a}-{train}-{scale}", policy_id=f"{a}-{train}-{scale}",
                            dataset_fingerprint=day, rules_version="harmony-A-B-local-v2",
                            assignment_fingerprint=f"{scenario}-{player}-{seed}", opponent_fingerprint="frozen",
                            provenance="new_execution"))
    return pd.DataFrame(rows)


def test_full_cartesian_coverage_rejects_missing_duplicate_and_mixed_policy():
    source = frame()
    assert len(validate_phase(source, days=("period-24",), roles=("0", "1"))) == 320
    with pytest.raises(ValueError, match="coverage"):
        validate_phase(source.iloc[:-1], days=("period-24",), roles=("0", "1"))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_phase(pd.concat([source, source.iloc[:1]]), days=("period-24",), roles=("0", "1"))
    changed = source.copy()
    changed.loc[0, "policy_id"] = "another-model"
    with pytest.raises(ValueError, match="Different policies"):
        validate_phase(changed, days=("period-24",), roles=("0", "1"))
    changed.loc[0, "provenance"] = "historical_exact_policy_reuse"
    with pytest.raises(ValueError, match="Different policies"):
        validate_phase(changed, days=("period-24",), roles=("0", "1"))


def test_localpid_scene_is_canonical_and_historical_label_is_rejected():
    source = frame()
    source.loc[source.scenario == "LocalPID", "scenario"] = "PID"
    with pytest.raises(ValueError, match="canonical LocalPID"):
        validate_phase(source, days=("period-24",), roles=("0", "1"))


def test_role_ratios_not_ratio_of_total_values_and_phase_own_reference():
    a = frame()
    a.loc[(a.algorithm == "BC") & (a.player_id == "0"), "total_value"] = 2
    tables = aggregate_phase(a)
    assert tables["overall"].set_index("algorithm").loc["BC", "mean_R"] == 1.5
    b = a.copy()
    b.loc[b.algorithm == "LocalPID", "total_value"] *= 2
    assert aggregate_phase(b)["overall"].set_index("algorithm").loc["BC", "mean_R"] == .75
    # The model did not improve merely because its internal reference changed.
    assert paired_improvement(a, b)["overall"].set_index("algorithm").loc["BC", "mean_delta_V"] == 0


def test_training_seeds_averaged_before_pairing_and_zero_reference_separate():
    a, b = frame(), frame()
    a.loc[(a.algorithm == "DT") & (a.training_seed == 20261028), "total_value"] = 0
    a.loc[(a.algorithm == "DT") & (a.training_seed == 20261029), "total_value"] = 2
    b.loc[b.algorithm == "DT", "total_value"] = 3
    result = paired_improvement(a, b)
    dt = result["overall"].set_index("algorithm").loc["DT"]
    assert dt.pairs == 20 and dt.mean_delta_V == 2 and dt.mean_paired_relative_gain == 2
    a.loc[a.algorithm == "CQL", "total_value"] = 0
    result = paired_improvement(a, b)
    cql = result["overall"].set_index("algorithm").loc["CQL"]
    assert cql.zero_reference_pairs == 20 and np.isnan(cql.mean_paired_relative_gain)
    assert cql.mean_delta_V == 5


def test_zero_localpid_is_undefined_not_epsilon():
    source = frame()
    source.loc[(source.algorithm == "LocalPID") & (source.player_id == "0"), "total_value"] = 0
    result = aggregate_phase(source)
    assert len(result["undefined_references"]) == len(ALGORITHMS) * len(SCENARIOS)
    assert result["per_role"].query("player_id == '0'").R.isna().all()
    assert np.isfinite(result["per_role"].query("player_id == '1'").R).all()


def test_main_R_equal_scenes_with_unequal_zero_reference_cells():
    source=frame()
    source.loc[(source.algorithm=="LocalPID") & (source.player_id=="0") & (source.scenario=="Fixed"),"total_value"]=0
    source.loc[(source.algorithm=="BC") & (source.player_id=="0"),"total_value"]=10.
    result=aggregate_phase(source)
    overall=result["overall"].set_index("algorithm").loc["BC"]
    # Fixed has only role1 R=1; each other scene has roles R=10 and1 =>5.5.
    assert overall.mean_R==4.6
    assert overall.mean_role_average_R==5.5
    assert overall.defined_R_role_scenario_cells==9 and overall.undefined_R_role_scenario_cells==1
    roles=result["per_role"].query("algorithm == 'BC'")
    assert overall.p10_R==roles.groupby("player_id").R.mean().quantile(.1)
    # A wholly undefined scenario is not silently dropped from the five-scene mean.
    source.loc[(source.algorithm=="LocalPID") & (source.scenario=="Fixed"),"total_value"]=0
    overall=aggregate_phase(source)["overall"].set_index("algorithm").loc["BC"]
    assert np.isnan(overall.mean_R) and overall.defined_R_scenarios==4
    assert overall.mean_defined_scenario_R==5.5


def test_pairing_rejects_market_assignment_or_budget_changes():
    a, b = frame(), frame(2)
    b["dataset_fingerprint"] = "different"
    with pytest.raises(ValueError, match="dataset"):
        paired_improvement(a, b)
    b = frame(2)
    b["assignment_fingerprint"] = "changed-assignment"
    with pytest.raises(ValueError, match="assignment"):
        paired_improvement(a, b)
    b = frame(2)
    b["initial_budget"] = 200
    with pytest.raises(ValueError, match="budgets"):
        paired_improvement(a, b)


def test_threshold_is_one_based_once_and_never_is_nan(tmp_path):
    episodes = frame().iloc[:2].copy()
    episodes["total_value"] = 72.
    episodes["total_cost"] = [96., 10.]
    episodes["budget_utilization"] = [.96, .1]
    rows = []
    for r in episodes.itertuples():
        paid = 0.
        for slot in range(72):
            cost = (96. if r.task_index == 0 else 10.) if slot == 2 else 0.
            paid += cost
            rows.append(dict(task_index=r.task_index, slot=slot, alpha=10., cost=cost,
                value=1., wins=1, remaining_budget=100-paid))
    pd.DataFrame(rows).to_csv(tmp_path / "slots.csv", index=False)
    result = behavioral_statistics(tmp_path, episodes, Sources())
    ep = result["episodes"].set_index("task_index")
    assert ep.loc[0, "slot95"] == 3
    assert np.isnan(ep.loc[1, "slot95"])
    assert result["overall"].iloc[0].slot95_median_reached == 3
    assert result["overall"].iloc[0].slot95_never == 1


def test_duplicate_or_incomplete_slots_are_rejected(tmp_path):
    episodes = frame().iloc[:1].copy()
    episodes["total_value"] = 0.
    episodes["total_cost"] = 0.
    episodes["budget_utilization"] = 0.
    rows = pd.DataFrame([dict(task_index=0, slot=i, alpha=0., cost=0., value=0., wins=0, remaining_budget=100.) for i in range(71)])
    rows.to_csv(tmp_path / "slots.csv", index=False)
    with pytest.raises(ValueError, match="Missing or extra"):
        behavioral_statistics(tmp_path, episodes, Sources())
    pd.concat([rows, rows.iloc[:1]]).to_csv(tmp_path / "slots.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        behavioral_statistics(tmp_path, episodes, Sources())


def test_overall_p10_order_is_explicit():
    source = frame()
    for i, scene in enumerate(SCENARIOS):
        source.loc[(source.algorithm == "BC") & (source.scenario == scene) & (source.player_id == str(i % 2)), "total_value"] *= 10
    result = aggregate_phase(source)
    r = result["overall"].set_index("algorithm").loc["BC"]
    roles = result["per_role"].query("algorithm == 'BC'")
    assert r.p10_R == roles.groupby("player_id").R.mean().quantile(.1)
    assert r.mean_scenario_p10_R == result["per_scenario"].query("algorithm == 'BC'").p10_R.mean()
    assert r.p10_R != r.mean_scenario_p10_R


def test_global_catalog_owns_config_identity_not_nested_attempt(tmp_path):
    models = tmp_path / "models"
    run = models / "runs" / "BC" / "config_1_seed_20261028" / "attempt_0" / "BC" / "config_0_seed_20261028"
    run.mkdir(parents=True)
    checkpoint = run / "step_600.pt"
    checkpoint.write_text("synthetic test placeholder; never loaded")
    record = dict(algorithm="BC", config_index=1, checkpoint=str(checkpoint), seed=20261028,
        step=600, config={"learning_rate": .001}, training_log=str(run / "training.json"))
    (models / "candidates.json").write_text(json.dumps([record]))
    (run.parent.parent / "candidates.json").write_text(json.dumps([{**record,"config_index":0}]))
    (run / "training.json").write_text(json.dumps(dict(config=record["config"], seconds=1., parameters=100,
        status="completed", history=[dict(step=1,losses=[.2]),dict(step=600,losses=[.1])])))
    result = training_evidence(tmp_path, Sources())
    assert result["runs"].config_index.tolist() == [1]
    assert result["losses"].config_index.tolist() == [1,1]
    assert result["checkpoints"].config_index.tolist() == [1]


def test_selection_phase_paths_and_exact_independent_freeze(tmp_path):
    records = [{"algorithm":a,"training_seed":seed,"spec":{}} for a in ALGORITHMS
        for seed in ((20261028,20261029) if a in NEURAL else (0,))]
    for phase in ("experiment_A","experiment_B","independent_test"):
        directory = tmp_path / phase
        directory.mkdir()
        (directory / "selected_frozen.json").write_text(json.dumps({"phase":"experiment_A" if phase=="experiment_A" else "experiment_B", "selected":records,"assets":{}}))
    result = selection_records(tmp_path,Sources())
    assert len(result)==2 and any("experiment_A" in name for name in result)
    path=tmp_path / "independent_test" / "selected_frozen.json"
    value=json.loads(path.read_text());value["selected"][0]["spec"]={"different":True};path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="exactly frozen B"):
        selection_records(tmp_path,Sources())


def test_audited_mechanisms_are_embedded_without_historical_final_tables():
    result=historical_mechanisms(Sources())
    assert set(result)==set(ALGORITHMS)
    assert "two-hot" in result["BCQ"] and "因果" in result["DT"]
    assert "r_up" in result["AuctionNetPID"]
    assert all("#### 最终效果" not in body for body in result.values())


def test_render_twelve_figures_from_explicitly_synthetic_fixture(tmp_path):
    a,b=frame(),frame(1.2)
    tables={"experiment_A":aggregate_phase(a),"experiment_B":aggregate_phase(b),"independent_test":aggregate_phase(frame(.8))}
    paired=paired_improvement(a,b)
    population=pd.DataFrame([dict(algorithm=algorithm,scenario=scene,slot_number=slot,
        cumulative_spend_mean=slot/100,cumulative_value_mean=slot/72*5)
        for algorithm in ALGORITHMS for scene in SCENARIOS for slot in range(1,73)])
    behavior={p:{"population_slots":population} for p in ("experiment_A","experiment_B")}
    losses=pd.DataFrame([dict(algorithm=algorithm,config_index=config,training_seed=seed,step=step,
        component=0,loss=(600/step)*(config+1)) for algorithm in NEURAL for config in (0,1)
        for seed in (20261028,20261029) for step in (1,600,1800)])
    validation=pd.DataFrame([dict(algorithm=algorithm,config_index=config,step=step,complete=True,
        mean_V=10+config+step/1800,training_seed_V_std=.1) for algorithm in NEURAL
        for config in (0,1) for step in (600,1800)])
    records=draw_figures(tmp_path / "SYNTHETIC_FIGURE_QA",tables,paired,behavior,{"losses":losses,"validation":validation})
    assert len(records)==12
    for record in records:
        assert Path(record["png"]).stat().st_size>1000
        assert Path(record["svg"]).stat().st_size>1000


def test_detailed_report_assembly_with_synthetic_scores(tmp_path):
    # This tests document integration only. Historical models are never loaded
    # or executed; output is isolated under the pytest SYNTHETIC directory.
    root=tmp_path / "SYNTHETIC_REPORT_QA"
    destination=root / "reporting"
    destination.mkdir(parents=True)
    episodes={"experiment_A":frame(),"experiment_B":frame(1.2),"independent_test":frame(.8)}
    for f in episodes.values():
        f["paired_p99_ms"]=1.
        f["peak_process_rss_mb"]=100.
    tables={phase:aggregate_phase(f) for phase,f in episodes.items()}
    paired=paired_improvement(episodes["experiment_A"],episodes["experiment_B"])
    overall=pd.DataFrame([dict(algorithm=a,episodes=32,first_slot_spend=.01,first6_spend=.06,
        alpha_median=10.,alpha_p90=11.,alpha_max=12.,slot95_median_reached=50.,slot95_never=0,late_value_share=.5)
        for a in ALGORITHMS])
    behavior={p:{"overall":overall} for p in episodes}
    sources=Sources()
    records=sources.json(OLD_ROOT / "selected_frozen.json")["selected"]
    records=records+[{"algorithm":"AuctionNetPID","spec":{"kwargs":{"initial_alpha":15.,"lower_threshold":.7}}}]
    selections={f"{p}/selected_frozen.json":records for p in ("experiment_A","experiment_B")}
    figures=[dict(name=name,description="SYNTHETIC test only",png=str(root / f"{name}.png")) for name in
        ("01_frozen_value","02_tuned_value","03_internal_relative_performance","04_overall_value_comparison",
         "05_paired_value_improvement","06_budget_before_after","07_cumulative_budget","08_cumulative_value",
         "09_player_types","10_cross_scenario_robustness","11_actual_learning_curves","12_independent_period")]
    training={"validation":pd.DataFrame([dict(algorithm=a,config_index=0,step=600,successful_episodes=60,
        expected_episodes=60,complete=True,mean_V=2.,training_seed_V_std=.1,failed_episodes=0,skipped_episodes=0) for a in NEURAL]),
        "run_receipts":pd.DataFrame([dict(algorithm=a,training_seconds=1.,call_wall_seconds=2.) for a in NEURAL])}
    context={"trajectory_provenance.json":dict(days=["testfixture"],episodes=1,transitions=72,reused_episodes=0,new_episodes=1,
        scenario_counts={},behavior_selection="synthetic",labels_are_actor_inputs=False,test_used=False),
        "models/training_status.json":{"status":"SYNTHETIC"},"models/training_metadata.json":{"action_scale":1.,"reward_scale":1.},
        "shared_feature_definition":"Synthetic integration fixture."}
    execution=pd.DataFrame([dict(stage="SYNTHETIC",status="completed",completed_episodes=1,expected_episodes=1,
        failed_episodes=0,skipped_episodes=0,workers=4,recorded_segment_seconds=1.)])
    reports=write_reports(root,destination,tables,paired,behavior,training,figures,episodes,selections,
        historical_mechanisms(sources),context,sources,execution)
    detailed=reports[1].read_text(encoding="utf8")
    assert "历史机制档案" in detailed and "本次五场景调优与冻结" in detailed
    assert "A：按类型代码的平均V" in detailed and "B：按类型代码的平均V" in detailed
    assert len(detailed)>10000
    assert len(reports)==2 and all(p.exists() for p in reports)


def test_all_real_runtime_segments_preserved_and_qa_excluded(tmp_path):
    def stage(name,segments,workers=None,completed=0):
        directory=tmp_path / name
        directory.mkdir(parents=True)
        (directory / "episode_results.status.json").write_text(json.dumps(dict(status="completed",valid=True,workers=workers,
            completed_episodes=completed,expected_episodes=completed)))
        for index,(seconds,new_completed,resumed) in enumerate(segments):
            (directory / f"runtime_segment_{index:03d}.json").write_text(json.dumps(dict(seconds=seconds,new_completed=new_completed,resumed=resumed,phase=name,failures=0)))
    stage("opponent_train_calibration",[(200.,240,0),(1.,0,240)],workers=8,completed=240)
    stage("resource_benchmark/workers_4",[(54.,48,0)],workers=4,completed=48)
    stage("resource_benchmark/workers_8",[(45.,48,0)],workers=8,completed=48)
    stage("experiment_A",[],completed=7680)
    stage("qa/root_preflight_4/test_fake",[(99999.,99999,0)],workers=999,completed=99999)
    stage("smoke_execution/test_fake",[(99999.,99999,0)],workers=999,completed=99999)
    stage("snapshots/original_stage",[(99999.,99999,0)],workers=999,completed=99999)
    result,segments=execution_evidence(tmp_path,Sources(),include_segments=True)
    assert len(result)==4 and len(segments)==4
    assert set(result.scope)=={"opponent_calibration","resource_benchmark","final_rollup"}
    calibration=result.set_index("stage").loc["opponent_train_calibration"]
    assert calibration.workers==8 and calibration.recorded_segment_seconds==201.
    assert calibration.recorded_segment_new_completed==240 and calibration.saved_runtime_segments==2
    assert segments.workers_recorded_in_segment.isna().all()
    assert result.query("scope == 'resource_benchmark'").recorded_segment_new_completed.sum()==96
    assert result.recorded_segment_seconds.sum()==300.


def test_applied_resources_override_recommendation_status_without_rewriting(tmp_path):
    selection=dict(status="completed",recommended_workers=8,test_used=False,validation_used=False,
        included_in_training_data=False,included_in_performance_rankings=False,
        groups=[dict(workers=4,episodes=48,seconds=54.),dict(workers=8,episodes=48,seconds=45.)],
        paired_episode_equality={"matched_episodes":48})
    path=tmp_path / "resource_selection.json"
    path.write_text(json.dumps(selection))
    benchmark=tmp_path / "resource_benchmark"
    benchmark.mkdir()
    recommended=benchmark / "operational_adjustment.json"
    recommended.write_text(json.dumps({"status":"recommended_not_applied"}))
    original_hash=digest(recommended)
    (tmp_path / "resource_application.json").write_text(json.dumps(dict(status="applied",initial_protocol_workers=4,
        actual_following_workers=8,selection_record_sha256=digest(path))))
    evidence=resource_evidence(tmp_path,Sources())
    assert evidence["applied_workers"]==8 and evidence["benchmark_episodes"]==96
    assert evidence["original_recommendation"]["status"]=="recommended_not_applied"
    assert "已采用8 workers" in resource_paragraph({"resource_evidence":evidence})
    assert digest(recommended)==original_hash
    path.write_text(json.dumps({**selection,"recommended_workers":4}))
    with pytest.raises(ValueError,match="not linked"):
        resource_evidence(tmp_path,Sources())


def test_equal_value_algorithms_share_rank():
    assert aggregate_phase(frame())["overall"].rank_V.eq(1).all()


def test_type_value_table_uses_observed_role_counts_and_equal_scenes():
    source=frame()
    source["player_type"]=source.player_type.map({"0":"7","1":"42"})
    for index,scene in enumerate(SCENARIOS):
        source.loc[(source.algorithm=="BC") & (source.player_id=="0") & (source.scenario==scene),"total_value"]=index+1
    table=type_value_table({"experiment_A":aggregate_phase(source)},"experiment_A")
    assert "代码 7（1角色）" in table and "代码 42（1角色）" in table
    assert "| BC | 3.0000 | 9.0000 |" in table
    assert "行业" not in table and "48角色" not in table


def test_qa_log_count_is_read_from_evidence_and_not_experiment_time(tmp_path):
    directory=tmp_path / "qa"
    directory.mkdir()
    path=directory / "root_integrated_final_1.log"
    path.write_text("... [100%]\n199 passed in 102.42s (0:01:42)\n",encoding="utf8")
    focused=directory / "report_focused_type_qa.log"
    focused.write_text("3 passed, 17 deselected in 2.50s\n",encoding="utf8")
    sources=Sources()
    records=qa_evidence(tmp_path,sources)
    assert [r["passed_tests"] for r in records]==[199,3]
    assert records[0]["qa_elapsed_seconds"]==102.42
    assert all(not r["included_in_rankings"] and not r["included_in_experiment_runtime"] for r in records)
    assert sources.hashes[str(path.resolve())]==digest(path)


def test_positive_paired_percent_does_not_overrule_lower_mean_value():
    a,b=frame(),frame()
    b.loc[(b.algorithm=="DT") & (b.player_id=="0"),"total_value"]=2.
    b.loc[(b.algorithm=="DT") & (b.player_id=="1"),"total_value"]=7.
    result=paired_improvement(a,b)
    dt=result["overall"].set_index("algorithm").loc["DT"]
    assert dt.A_mean_V==5. and dt.B_mean_V==4.5 and dt.mean_delta_V==-.5
    assert dt.ratio_of_mean_V_gain<0 and dt.mean_paired_relative_gain>0
    assert dt.relative_vs_mean_V_sign_reversal
    warnings,text=gain_interpretation(result)
    assert [r["algorithm"] for r in warnings]==["DT"]
    assert "主目标平均V仍然下降" in text and "不能" in text
    reverse=paired_improvement(b,a)
    assert reverse["overall"].set_index("algorithm").loc["DT"].relative_vs_mean_V_sign_reversal
    assert "主目标平均V仍然上升" in gain_interpretation(reverse)[1]


def test_recovery_success_does_not_erase_failures_or_count_snapshots_twice(tmp_path):
    def make(name,failed,skipped):
        directory=tmp_path / name
        directory.mkdir(parents=True)
        (directory / "task_protocol.json").write_text("{}")
        (directory / "failures.json").write_text(json.dumps(failed))
        (directory / "skipped.json").write_text(json.dumps(skipped))
    failed=[dict(task_index=5,algorithm="TD3BC",status="resource_incomplete",exception_type="Timeout",elapsed_seconds=1960.)]
    skipped=[dict(task_index=6,algorithm="TD3BC",status="not_run",reason="same algorithm failure")]
    make("independent_test_new",failed,skipped)
    make("independent_test_recovery",[],[])
    make("snapshots/original_stage",failed,skipped)
    make("qa/fixture",failed,skipped)
    histories=failure_history(tmp_path,Sources())
    assert len(histories["failures"])==1 and len(histories["skipped"])==1
    assert histories["by_stage"].iloc[0].resource_timeouts==1
    text=failure_history_paragraph(tmp_path,{"failure_history":{k:v.to_dict("records") for k,v in histories.items()}})
    assert "完整排名不等于零历史失败" in text and "不覆盖或改写原失败" in text
    assert "1960.000" in text


def test_final_recovery_manifest_requires_lineage_and_preserved_history(tmp_path):
    root=tmp_path
    original=root / "independent_test_new"
    recovery=root / "independent_test_recovery_001"
    final=root / "independent_test"
    frozen=root / "experiment_B"
    for directory in (original,recovery,final,frozen):
        directory.mkdir()
    def save(path,value):
        path.write_text(json.dumps(value),encoding="utf8")
        return digest(path)
    original_protocol_sha=save(original / "task_protocol.json",{})
    frozen_sha=save(frozen / "selected_frozen.json",{})
    failed=[dict(task_index=5,status="resource_incomplete",exception_type="Timeout")]
    skipped=[dict(task_index=6,status="not_run")]
    save(original / "failures.json",failed);save(original / "skipped.json",skipped)
    mapping=[dict(original_task_index=old,recovery_task_index=local,task_key=f"key{old}",
        terminal={"kind":"failure" if old==5 else "skipped","record":failed[0] if old==5 else skipped[0]}) for local,old in enumerate((5,6))]
    plan=dict(mapping=mapping,expected_recovery_episodes=2,historical_failed_episodes=1,historical_skipped_episodes=1,
        timeout_seconds=180.,workers=2,evidence_sha256={},original_task_protocol_sha256=original_protocol_sha,B_freeze_sha256=frozen_sha)
    plan_path=recovery / "recovery_plan.json"
    plan_sha=save(plan_path,plan)
    result_sha=save(recovery / "episode_results.csv",{"synthetic":"csv fixture bytes"})
    slots_sha=save(recovery / "slots.csv",{"synthetic":"slots fixture bytes"})
    receipt=dict(status="completed",completed_episodes=2,expected_episodes=2,failed_episodes=0,skipped_episodes=0,
        plan_sha256=plan_sha,result_sha256=result_sha,slots_sha256=slots_sha,runtime_segments=[])
    receipt_path=recovery / "recovery_receipt.json"
    receipt_sha=save(receipt_path,receipt)
    manifest=dict(plan_path=str(plan_path),plan_sha256=plan_sha,receipt_path=str(receipt_path),receipt_sha256=receipt_sha,
        original_stage=original.name,recovery_stage=recovery.name,original_completed_episodes=1,recovery_completed_episodes=2,
        historical_failed_episodes=1,historical_skipped_episodes=1,unresolved_episodes=0,mapping=mapping,
        original_failure_path=str(original / "failures.json"),original_skipped_path=str(original / "skipped.json"),
        final_export_sha256={"episode_results.csv":"finalCSV","slots.csv":"finalSlots"})
    manifest_sha=save(final / "recovery_manifest.json",manifest)
    status=dict(recovery_manifest_sha256=manifest_sha,historical_failed_episodes=1,historical_skipped_episodes=1,
        original_completed_episodes=1,recovery_completed_episodes=2,result_sha256="finalCSV",slots_sha256="finalSlots")
    rows=[dict(task_index=0,provenance="new_execution")]
    for local,old in enumerate((5,6)):
        rows.append(dict(task_index=old,original_task_index=old,recovery_task_index=local,recovery_stage=recovery.name,
            provenance="resource_recovery",task_key=f"key{old}",source_task=f"{recovery.name}:{local}",
            source_result_path=str(recovery / "results" / f"task_{local:05d}.json")))
    frame=pd.DataFrame(rows)
    result=verify_recovery_manifest(root,frame,status,Sources())
    assert result["verified_recovered_rows"]==2 and result["original_failed_events_preserved"]==1
    with pytest.raises(ValueError,match="hides or changes"):
        verify_recovery_manifest(root,frame,{**status,"historical_failed_episodes":0},Sources())
    frame.loc[1,"source_task"]="wrong-stage:0"
    with pytest.raises(ValueError,match="source identity"):
        verify_recovery_manifest(root,frame,status,Sources())
