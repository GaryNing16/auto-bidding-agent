"""Audit, aggregate and plot actual completed baseline evaluation CSVs.

Run only after selection is frozen and the full evaluation stage is completed:
    python -m harmony_baselines.report --root outputs/baseline_comparison_20261028
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "baseline_comparison_20261028"
SCENARIOS = ["Fixed", "PID", "Mixed"]
ORDER = ["FixedAlpha", "LocalPID", "OnlineLP", "BC", "IQL", "BCQ", "TD3BC", "CQL", "DT"]
COLORS = {"Fixed": "#2878b5", "PID": "#e07a30", "Mixed": "#269b82"}
LABELS = {"FixedAlpha": "Fixed α", "LocalPID": "LocalPID", "OnlineLP": "Online LP", "TD3BC": "TD3+BC"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _test_outcome_path(root, kind):
    canonical = Path(root)/f"evaluation_{kind}.json"
    return canonical if canonical.exists() else Path(root)/f"test_{kind}.json"


def _ordered(values):
    names = set(values)
    return [x for x in ORDER if x in names] + sorted(names-set(ORDER))


def _std(series):
    values = pd.Series(series).dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else np.nan


def _ratio(frame, references):
    keys = ["budget_scale", "scenario", "player_id"]
    if "seed" in frame:
        keys += ["seed"]
    ref = references[keys+["total_value"]].rename(columns={"total_value": "reference_value"})
    joined = frame.merge(ref, on=keys, how="left", validate="many_to_one")
    if joined.reference_value.isna().any():
        raise ValueError("LocalPID reference missing for one or more role/scenario/seed cells")
    joined["relative_performance"] = np.nan
    positive = joined.reference_value > 0
    joined.loc[positive, "relative_performance"] = joined.loc[positive, "total_value"] / joined.loc[positive, "reference_value"]
    joined["relative_status"] = np.where(positive, "defined", np.where(joined.total_value > 0,
                                                                      "zero_reference_positive_candidate", "both_zero"))
    return joined


def aggregate_results(frame, fixed_player="0"):
    """Role mean first, then role ratios; never replace zero references by eps.

    Every day, evaluation seed and training seed has equal weight inside a
    role. The formal gate checks rectangular coverage before calling this.
    """
    frame = frame.copy()
    required = ["algorithm", "scenario", "player_id", "player_type", "day_id", "seed", "training_seed",
                "budget_scale", "total_value", "total_cost", "initial_budget", "budget_utilization", "wins"]
    missing = set(required)-set(frame)
    if missing:
        raise ValueError(f"missing episode columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("no completed episodes")
    for column in ["player_id", "player_type", "day_id", "algorithm", "scenario"]:
        frame[column] = frame[column].astype(str)
    numbers = ["total_value", "total_cost", "initial_budget", "budget_utilization", "wins", "budget_scale"]
    if not np.isfinite(frame[numbers].to_numpy(dtype=float)).all() or (frame[numbers] < 0).any().any():
        raise ValueError("invalid episode metric")
    if (frame.total_cost > frame.initial_budget+1e-6).any() or (frame.budget_utilization > 1+1e-9).any():
        raise ValueError("budget invariant violated")
    if set(frame.scenario)-set(SCENARIOS):
        raise ValueError("unknown scenario")
    uniqueness = ["budget_scale", "algorithm", "training_seed", "day_id", "scenario", "player_id", "seed"]
    if frame.duplicated(uniqueness).any():
        raise ValueError("duplicate episode keys")
    if frame.groupby("player_id").player_type.nunique().max() != 1:
        raise ValueError("player_type changed across episodes")
    keys = ["budget_scale", "algorithm", "scenario", "player_id", "player_type"]
    role = frame.groupby(keys, as_index=False, sort=False)[numbers[:-1]].mean()
    counts = frame.groupby(keys).size().rename("episodes").reset_index()
    role = role.merge(counts, on=keys, validate="one_to_one")
    role = _ratio(role, role[role.algorithm == "LocalPID"])
    seed_keys = keys+["seed"]
    seed_role = frame.groupby(seed_keys, as_index=False, sort=False)[["total_value", "budget_utilization"]].mean()
    seed_role = _ratio(seed_role, seed_role[seed_role.algorithm == "LocalPID"])
    seed_means = seed_role.groupby(["budget_scale", "algorithm", "scenario", "seed"], as_index=False).agg(
        relative_performance=("relative_performance", "mean"), total_value=("total_value", "mean"),
        budget_utilization=("budget_utilization", "mean"), valid_roles=("relative_performance", "count"))
    training_role = frame.groupby(keys+["training_seed"], as_index=False, sort=False)[["total_value", "budget_utilization"]].mean()
    training_role = _ratio(training_role, role[role.algorithm == "LocalPID"])
    training_means = training_role.groupby(["budget_scale", "algorithm", "scenario", "training_seed"], as_index=False).agg(
        relative_performance=("relative_performance", "mean"), total_value=("total_value", "mean"),
        budget_utilization=("budget_utilization", "mean"), valid_roles=("relative_performance", "count"))
    summary_rows = []
    for (scale, algorithm, scenario), group in role.groupby(["budget_scale", "algorithm", "scenario"], sort=False):
        ratios = group.relative_performance.dropna()
        seed_group = seed_means[(seed_means.budget_scale == scale) & (seed_means.algorithm == algorithm) & (seed_means.scenario == scenario)]
        training_group = training_means[(training_means.budget_scale == scale) & (training_means.algorithm == algorithm) & (training_means.scenario == scenario)]
        summary_rows.append({"budget_scale": scale, "algorithm": algorithm, "scenario": scenario,
            "mean_total_value": group.total_value.mean(), "value_cross_player_std": _std(group.total_value),
            "value_median": group.total_value.median(), "value_p10": group.total_value.quantile(.1),
            "mean_budget_utilization": group.budget_utilization.mean(), "mean_wins": group.wins.mean(),
            "mean_relative_performance": ratios.mean(), "relative_cross_player_std": _std(ratios),
            "relative_median": ratios.median(), "relative_p10": ratios.quantile(.1),
            "relative_seed_std": _std(seed_group.relative_performance),
            "relative_seed_min": seed_group.relative_performance.min(), "relative_seed_max": seed_group.relative_performance.max(),
            "relative_training_seed_std": _std(training_group.relative_performance),
            "relative_training_seed_min": training_group.relative_performance.min(),
            "relative_training_seed_max": training_group.relative_performance.max(),
            "training_seeds": len(training_group),
            "value_seed_std": _std(seed_group.total_value), "budget_seed_std": _std(seed_group.budget_utilization),
            "valid_reference_roles": len(ratios), "total_roles": len(group),
            "zero_reference_roles": len(group)-len(ratios), "evaluation_seeds": len(seed_group)})
    summary = pd.DataFrame(summary_rows)
    types = role.groupby(["budget_scale", "algorithm", "scenario", "player_type"], as_index=False).agg(
        mean_relative_performance=("relative_performance", "mean"), mean_total_value=("total_value", "mean"),
        mean_budget_utilization=("budget_utilization", "mean"), mean_wins=("wins", "mean"),
        valid_reference_roles=("relative_performance", "count"), total_roles=("player_id", "count"))
    robustness = summary.groupby(["budget_scale", "algorithm"], as_index=False).agg(
        scenario_mean=("mean_relative_performance", "mean"), worst_scenario=("mean_relative_performance", "min"),
        best_scenario=("mean_relative_performance", "max"), scenario_std=("mean_relative_performance", _std),
        mean_role_p10=("relative_p10", "mean"), mean_value=("mean_total_value", "mean"),
        scenario_count=("scenario", "count"))
    timing_rows = []
    for (scale, algorithm), group in frame.groupby(["budget_scale", "algorithm"], sort=False):
        row = {"budget_scale": scale, "algorithm": algorithm, "episodes": len(group)}
        for column in ("bidding_p99_ms", "update_p99_ms", "paired_p99_ms", "rss_mb", "peak_process_rss_mb"):
            if column in group:
                row["max_episode_"+column] = group[column].max()
                row["mean_episode_"+column] = group[column].mean()
        timing_rows.append(row)
    return {"per_player": role, "all_player_summary": summary, "fixed_player_summary": role[role.player_id == str(fixed_player)],
            "by_player_type": types, "per_seed_summary": seed_means, "per_training_seed_summary": training_means, "robustness": robustness,
            "inference_summary": pd.DataFrame(timing_rows),
            "undefined_references": role[role.relative_status != "defined"]}


def audit_complete_results(root):
    root = Path(root)
    protocol = _read(root/"protocol.json")
    status = _read(root/"episode_results.status.json")
    selected = _read(root/"selected_frozen.json")
    path = root/"episode_results.csv"
    allowed_statuses = {"completed", "completed_with_failures"}
    if protocol.get("status") not in allowed_statuses or status.get("status") not in allowed_statuses:
        raise ValueError("formal report requires completed protocol and completed evaluation status")
    if protocol["status"] != status["status"]:
        raise ValueError("protocol and evaluation completion status disagree")
    splits = protocol["executed_split"]
    if (set(splits["train"]) & set(splits["validation"]) or set(splits["train"]) & set(splits["test"])
            or set(splits["validation"]) & set(splits["test"])):
        raise ValueError("train/validation/test periods overlap")
    if status.get("csv_sha256") != sha256_file(path):
        raise ValueError("evaluation CSV differs from completed status digest")
    if protocol.get("selection_sha256") != sha256_file(root/"selected_frozen.json"):
        raise ValueError("frozen selection changed after test")
    if selected.get("test_used") is not False:
        raise ValueError("selection must explicitly exclude test data")
    if selected.get("opponents_sha256") != sha256_file(root/"opponents_frozen.json"):
        raise ValueError("opponents changed after freezing")
    if selected.get("lp_table_sha256") != sha256_file(root/"online_lp_table.csv"):
        raise ValueError("OnlineLP table changed after freezing")
    for checkpoint, digest in selected.get("checkpoints", {}).items():
        if sha256_file(checkpoint) != digest:
            raise ValueError("checkpoint changed after test")
    for code_path, digest in selected.get("code_sha256", {}).items():
        if sha256_file(ROOT/code_path) != digest:
            raise ValueError(f"source code changed after freezing: {code_path}")
    frame = pd.read_csv(path, dtype={"player_id": str, "player_type": str})
    if len(frame) != status.get("episodes") or len(frame) != protocol.get("test_completed_episodes", protocol.get("test_episodes")):
        raise ValueError("evaluation count mismatch")
    if "completed" not in frame or not frame.completed.eq(True).all():
        raise ValueError("incomplete episodes cannot enter report")
    expected = set()
    for candidate in selected["selected"]:
        for scale in protocol.get("budget_scales", [1.]):
            for day in protocol["executed_split"]["test"]:
                for scenario in SCENARIOS:
                    for player in protocol["test_roles"]:
                        for seed in protocol["evaluation_seeds"]:
                            expected.add((float(scale), candidate["algorithm"], int(candidate.get("training_seed", 0)),
                                          day, scenario, str(player), int(seed)))
    actual = set(frame[["budget_scale", "algorithm", "training_seed", "day_id", "scenario", "player_id", "seed"]].itertuples(index=False, name=None))
    if actual-expected or len(actual) != len(frame):
        raise ValueError(f"evaluation coverage mismatch: missing={len(expected-actual)}, unexpected={len(actual-expected)}")
    missing = expected-actual
    incomplete_algorithms = {key[1] for key in missing}
    if status["status"] == "completed":
        if missing or protocol.get("incomplete_algorithms"):
            raise ValueError("completed evaluation has missing role/seed cases")
    else:
        if not missing or set(protocol.get("incomplete_algorithms", [])) != incomplete_algorithms:
            raise ValueError("incomplete algorithm declaration does not match actual missing cases")
        outcome_keys = set()
        if "task_index" not in frame or frame.task_index.duplicated().any():
            raise ValueError("test outcome coverage requires unique completed task_index values")
        outcome_indices = set(int(v) for v in frame.task_index)
        outcome_counts = {}
        for kind, allowed in [("failures", {"failed", "resource_incomplete"}), ("skipped", {"not_run"})]:
            records = _read(_test_outcome_path(root,kind))
            if not isinstance(records, list):
                raise ValueError("test outcome evidence must be a list")
            outcome_counts[kind] = len(records)
            for record in records:
                if record.get("status") not in allowed or record.get("phase") != "test" or record.get("test_used") is not True:
                    raise ValueError("invalid test failure/skipped provenance")
                key = (float(record.get("budget_scale", 1.)), record["algorithm"], int(record["training_seed"]),
                       record["day_id"], record["scenario"], str(record["player_id"]), int(record["evaluation_seed"]))
                if key in outcome_keys:
                    raise ValueError("duplicate test failure/skipped case")
                outcome_keys.add(key)
                if int(record["task_index"]) in outcome_indices:
                    raise ValueError("test task_index appears in more than one outcome")
                outcome_indices.add(int(record["task_index"]))
        if outcome_keys != missing:
            raise ValueError("missing test cases are not exactly accounted for by failure/skipped evidence")
        if outcome_indices != set(range(len(expected))):
            raise ValueError("test task_index outcomes do not cover the full task range")
        for container, expected_field, failure_field, skipped_field in [(status,"expected_episodes","failure_episodes","skipped_episodes"),
                (protocol,"test_expected_episodes","test_failure_episodes","test_skipped_episodes")]:
            if (container.get(expected_field) != len(expected) or container.get(failure_field) != outcome_counts["failures"]
                    or container.get(skipped_field) != outcome_counts["skipped"]):
                raise ValueError("test completion/failure/skipped counts disagree with evidence")
    if "LocalPID" in incomplete_algorithms:
        raise ValueError("LocalPID reference incomplete; cannot publish comparable normalized rankings")
    fairness = ["budget_scale", "day_id", "scenario", "player_id", "seed"]
    for column in ["initial_budget", "dataset_fingerprint", "rules_version", "opponent_fingerprint", "assignment_fingerprint"]:
        if frame.groupby(fairness)[column].nunique().max() != 1:
            raise ValueError(f"evaluation fairness mismatch: {column}")
    full_algorithms = set(frame.algorithm)-incomplete_algorithms
    protocol["_report_audit"] = {"expected_episodes": len(expected), "completed_raw_episodes": len(frame),
        "ranked_algorithms": _ordered(full_algorithms), "incomplete_algorithms": _ordered(incomplete_algorithms),
        "coverage_by_algorithm": {name: {"expected": sum(key[1] == name for key in expected),
            "completed": sum(key[1] == name for key in actual), "missing": sum(key[1] == name for key in missing)}
            for name in _ordered({key[1] for key in expected})}}
    return frame[frame.algorithm.isin(full_algorithms)].copy(), protocol, selected


def draw_figures(tables, destination, scale=1., qa_label=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "figure.dpi": 130})
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    summary = tables["all_player_summary"].query("budget_scale == @scale")
    names = _ordered(summary.algorithm)
    x = np.arange(len(names))
    paths = {}
    def save(fig, name):
        if qa_label:
            fig.text(.5, 1.03, qa_label, color="#a32626", weight="bold", ha="center", fontsize=12)
        for suffix in ("png", "svg"):
            fig.savefig(destination/(name+"."+suffix), dpi=180)
        paths[name] = destination/(name+".png")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4.3))
    for i, scenario in enumerate(SCENARIOS):
        group = summary[summary.scenario == scenario].set_index("algorithm").reindex(names)
        ax.bar(x+(i-1)*.25, group.mean_relative_performance, .23, color=COLORS[scenario], label=scenario,
               yerr=group.relative_seed_std, capsize=3, error_kw={"elinewidth": .9})
    ax.axhline(1., color="#64748b", linewidth=1, linestyle="--")
    ax.set_xticks(x, [LABELS.get(n,n) for n in names]); ax.set_ylabel("Mean per-player value / LocalPID value")
    ax.set_ylim(0, max(1.15, float((summary.mean_relative_performance+summary.relative_seed_std).max())*1.2))
    ax.set_title("All-player performance across opponent scenarios")
    ax.legend(ncol=3, frameon=False); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.text(.5, -.015, "Bars: ratio after within-player averaging. Error bars: ±1 SD of evaluation-seed role means.", ha="center", fontsize=9)
    save(fig, "01_performance")
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for i, scenario in enumerate(SCENARIOS):
        group = summary[summary.scenario == scenario].set_index("algorithm").reindex(names)
        ax.bar(x+(i-1)*.25, 100*group.mean_budget_utilization, .23, label=scenario,
               color=COLORS[scenario], yerr=100*group.budget_seed_std, capsize=2)
    ax.set_xticks(x, [LABELS.get(n,n) for n in names]); ax.set_ylabel("Mean budget utilization (%)")
    ax.set_ylim(0, max(115, ax.get_ylim()[1])); ax.axhline(100, color="#64748b", linewidth=1, linestyle="--")
    ax.set_title("Budget use under original budgets"); ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    save(fig, "02_budget")
    types = tables["by_player_type"].query("budget_scale == @scale")
    kinds = sorted(types.player_type.astype(str).unique(), key=lambda v: int(v) if v.isdigit() else v)
    matrices = [types[types.scenario == s].pivot(index="algorithm", columns="player_type", values="mean_relative_performance").reindex(index=names, columns=kinds).to_numpy() for s in SCENARIOS]
    finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
    lower, upper = min(0., float(finite.min())), max(1.05, float(finite.max()))
    norm = TwoSlopeNorm(vmin=lower, vcenter=1., vmax=upper)
    fig, axes = plt.subplots(1, 3, figsize=(13, max(4, len(names)*.45)), sharey=True, layout="constrained")
    for ax, scenario, matrix in zip(axes, SCENARIOS, matrices):
        im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_title(scenario, color=COLORS[scenario]); ax.set_xticks(range(len(kinds)), kinds); ax.set_xlabel("Player type")
        ax.set_yticks(range(len(names)), [LABELS.get(n,n) for n in names])
        for i in range(len(names)):
            for j in range(len(kinds)):
                ax.text(j, i, f"{matrix[i,j]:.2f}" if np.isfinite(matrix[i,j]) else "NA", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=list(axes), label="Mean per-player ratio to LocalPID", shrink=.8)
    save(fig, "03_player_types")
    robust = tables["robustness"].query("budget_scale == @scale").set_index("algorithm").reindex(names)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, robust.scenario_mean, color="#5f79a6", width=.6, label="Three-scenario mean",
           yerr=robust.scenario_std, capsize=4)
    ax.scatter(x, robust.worst_scenario, marker="D", s=32, color="#c74f45", label="Worst scenario", zorder=3)
    ax.scatter(x, robust.best_scenario, marker="_", s=80, color="#16866c", label="Best scenario", zorder=3)
    ax.axhline(1., color="#64748b", linewidth=1, linestyle="--")
    ax.set_xticks(x, [LABELS.get(n,n) for n in names]); ax.set_ylabel("Relative performance (LocalPID = 1)")
    ax.set_ylim(0, max(1.15, float((robust.scenario_mean+robust.scenario_std).max())*1.2))
    ax.set_title("Cross-scenario robustness: mean, worst case and variation")
    ax.legend(ncol=3, frameon=False); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.text(.5, -.015, "Whiskers: ±1 sample SD across three scenario means; descriptive, not a confidence interval.", ha="center", fontsize=9)
    save(fig, "04_robustness")
    return paths


def _fmt(value, digits=3):
    return "未定义" if pd.isna(value) else f"{value:.{digits}f}"


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(map(str,row)) + " |" for row in rows])


def _link(path):
    return Path(path).resolve().as_posix()


def algorithm_status_summary(root, selected, completed_names, training, protocol=None):
    """Keep trained-but-invalid algorithms visible without inventing scores.

    Validation failures and canceled/not-run cases are separate from network
    training failures. The former never become zero-value test episodes.
    """
    root = Path(root)
    def records(path):
        if not path.exists():
            return []
        value = _read(path)
        if not isinstance(value, list):
            raise ValueError(f"expected failure-record list: {path}")
        return value
    train_failures = records(root/"models"/"failures.json")
    validation = records(root/"validation_failures.json")
    skipped = records(root/"validation_skipped.json")
    test_failures = records(_test_outcome_path(root,"failures"))
    test_skipped = records(_test_outcome_path(root,"skipped"))
    excluded = selected.get("excluded_candidates", [])
    coverage = (protocol or {}).get("_report_audit", {}).get("coverage_by_algorithm", {})
    def algorithm(record):
        return str(record.get("algorithm") or record.get("candidate", "unknown")).split("|")[0]
    for record in validation:
        if record.get("status") not in {"failed", "resource_incomplete"}:
            raise ValueError("unknown validation failure status")
        if record.get("phase") != "validation" or record.get("test_used") is not False:
            raise ValueError("validation failure provenance must exclude test data")
    for record in skipped:
        if record.get("status") != "not_run":
            raise ValueError("validation skipped records must be explicitly not_run")
    names = set(completed_names)
    if not training.empty:
        names.update(training.algorithm)
    for source in (train_failures, validation, skipped, excluded, test_failures, test_skipped, selected.get("selected", [])):
        names.update(algorithm(r) for r in source)
    rows = []
    for name in _ordered(names):
        trained = training[training.algorithm == name] if not training.empty else pd.DataFrame()
        errors = [r for r in validation if algorithm(r) == name and r["status"] == "failed"]
        resource = [r for r in validation if algorithm(r) == name and r["status"] == "resource_incomplete"]
        canceled = [r for r in skipped if algorithm(r) == name]
        rejected = [r for r in excluded if algorithm(r) == name]
        train_errors = [r for r in train_failures if algorithm(r) == name]
        test_errors = [r for r in test_failures if algorithm(r) == name and r["status"] == "failed"]
        test_resource = [r for r in test_failures if algorithm(r) == name and r["status"] == "resource_incomplete"]
        test_canceled = [r for r in test_skipped if algorithm(r) == name]
        done = name in completed_names
        if done:
            status = "完成测试；部分候选验证失败/未完成" if errors or resource or rejected else "完成测试"
        elif test_errors or test_resource or test_canceled or coverage.get(name, {}).get("missing", 0):
            status = "已冻结，测试角色/seed未完成，整算法未排名"
        elif len(trained):
            status = "已训练，验证失败/资源未完成，未排名" if errors or resource else "已训练，验证覆盖未完成，未排名"
        else:
            status = "训练失败，未排名" if train_errors else "未完成，未排名"
        rows.append({"algorithm": name, "status": status, "completed_test": done,
            "saved_training_checkpoints": len(trained),
            "training_runs_with_checkpoint": len(trained[["config_index", "seed"]].drop_duplicates()) if len(trained) else 0,
            "training_failure_records": len(train_errors), "validation_failed_cases": len(errors),
            "validation_resource_incomplete_cases": len(resource), "validation_not_run_cases": len(canceled),
            "test_failed_cases": len(test_errors), "test_resource_incomplete_cases": len(test_resource),
            "test_not_run_cases": len(test_canceled), "test_expected_cases": coverage.get(name, {}).get("expected", 0),
            "test_completed_cases": coverage.get(name, {}).get("completed", 0),
            "test_missing_cases": coverage.get(name, {}).get("missing", 0),
            "excluded_candidate_count": len(rejected),
            "exception_types": json.dumps(dict(Counter(str(r.get("exception_type", "unknown")) for r in errors)), ensure_ascii=False),
            "validation_failure_reasons": json.dumps(dict(Counter(str(r.get("reason", "unspecified")) for r in errors+resource)), ensure_ascii=False),
            "exclusion_reasons": json.dumps(dict(Counter(str(r.get("reason", "unspecified")) for r in rejected)), ensure_ascii=False),
            "test_failure_reasons": json.dumps(dict(Counter(str(r.get("reason", "unspecified")) for r in test_errors+test_resource)), ensure_ascii=False),
            "timeout_seconds": json.dumps(sorted({r["timeout_seconds"] for r in resource+test_resource if "timeout_seconds" in r}))})
    return pd.DataFrame(rows)


def write_report(root, tables, protocol, selected, figures, report_path=None):
    root = Path(root)
    path = Path(report_path or ROOT.parent/"report_auctionnet_baselines_comparison.md")
    summary = tables["all_player_summary"].query("budget_scale == 1")
    robust = tables["robustness"].query("budget_scale == 1").sort_values("scenario_mean", ascending=False)
    names = _ordered(summary.algorithm)
    fixed = tables["fixed_player_summary"].query("budget_scale == 1")
    training_path = root/"models"/"candidates.json"
    training = pd.DataFrame(_read(training_path)) if training_path.exists() else pd.DataFrame()
    failed_path = root/"models"/"failures.json"
    failures = _read(failed_path) if failed_path.exists() else []
    statuses = algorithm_status_summary(root, selected, set(names), training, protocol)
    statuses.to_csv(root/"algorithm_status.csv", index=False)
    all_names = statuses.algorithm.tolist()
    status_index = statuses.set_index("algorithm")
    failure_notes = []
    for row in statuses.itertuples():
        if (row.training_failure_records or row.validation_failed_cases or row.validation_resource_incomplete_cases
                or row.validation_not_run_cases or row.excluded_candidate_count or not row.completed_test):
            reasons = ", ".join(f"{kind}×{count}" for kind,count in json.loads(row.exception_types).items())
            resource_note = f"，单局上限{'/'.join(map(str,json.loads(row.timeout_seconds)))}秒" if row.validation_resource_incomplete_cases else ""
            exclusion_reasons = json.loads(row.exclusion_reasons)
            missing_note = "；".join(f"{reason[:100]}×{count}" for reason,count in exclusion_reasons.items())
            failure_notes.append(f"**{LABELS.get(row.algorithm,row.algorithm)}**：训练失败{row.training_failure_records}条；"
                f"验证异常{row.validation_failed_cases}局（{reasons or '无'}），资源未完成{row.validation_resource_incomplete_cases}局{resource_note}，"
                f"取消未运行{row.validation_not_run_cases}局；排除{row.excluded_candidate_count}个候选"
                f"（{missing_note or '无'}）；{row.status}。")
        if row.test_failed_cases or row.test_resource_incomplete_cases or row.test_not_run_cases or row.test_missing_cases:
            test_reasons = "；".join(f"{reason[:120]}×{count}" for reason,count in json.loads(row.test_failure_reasons).items())
            failure_notes.append(f"**{LABELS.get(row.algorithm,row.algorithm)}测试未完成**：成功{row.test_completed_cases}/{row.test_expected_cases}局，"
                f"异常{row.test_failed_cases}局、资源未完成{row.test_resource_incomplete_cases}局、取消未运行{row.test_not_run_cases}局；"
                f"原因：{test_reasons or '见逐任务记录'}。保留成功原始行，但整个算法退出主排名，未更换冻结checkpoint。")
    validation_failure_text = (" ".join(failure_notes) + f" 失败没有按0分计入；确切原因见[验证失败记录]({_link(root/'validation_failures.json')})、"
                               f"[候选排除]({_link(root/'selected_frozen.json')})与[状态汇总]({_link(root/'algorithm_status.csv')})。"
                               + (f" [测试失败记录]({_link(_test_outcome_path(root,'failures'))})保留未完成角色及seed。" if _test_outcome_path(root,"failures").exists() else "")) if failure_notes else ""
    provenance = _read(root/"trajectory_provenance.json") if (root/"trajectory_provenance.json").exists() else {}
    calibration = _read(root/"calibration.json") if (root/"calibration.json").exists() else {}
    classical_search = _read(root/"classical_search.json") if (root/"classical_search.json").exists() else {}
    algorithm_multipliers = classical_search.get("algorithm_multipliers", {})
    multiplier_text = "；".join(f"{LABELS.get(n,n)}={values}" for n,values in algorithm_multipliers.items()) if algorithm_multipliers else str(classical_search.get("multipliers",[]))
    stability_path = root/"models"/"stability_tuning.json"
    stability_text = ""
    if stability_path.exists():
        stability = _read(stability_path)
        extension = _read(root/"models_cql_linear_run.json") if (root/"models_cql_linear_run.json").exists() else {}
        chosen_cql = [c for c in selected["selected"] if c["algorithm"] == "CQL"]
        descriptions = set()
        for candidate in chosen_cql:
            checkpoint = candidate["spec"]["kwargs"]["checkpoint"]
            matched = training[training.checkpoint == checkpoint]
            if len(matched) != 1:
                raise ValueError("selected CQL checkpoint missing or duplicated in training provenance")
            record = matched.iloc[0]
            config = record["config"]
            transform = config.get("action_transform", record.get("metadata", {}).get("action_transform", "log1p"))
            descriptions.add(f"config {int(record.config_index)} / step {int(record.step)} / {transform} / lr={config['learning_rate']:g}")
        selected_description = "最终冻结CQL为"+"；".join(sorted(descriptions))+"。" if descriptions else "CQL没有冻结的最终策略，追加配置不代表已成功入选。"
        stability_text = (f"**调参预算并非完全相同**：CQL因验证动作失稳额外增加一种稳定性配置"
            f"（{stability.get('action_transform','见记录')}，lr={stability.get('learning_rate','见记录')}），"
            f"{stability.get('extra_training_runs',0)}次训练、{len(stability.get('extra_records',[]))}个checkpoint"
            + (f"，追加墙钟{extension['seconds']:.3f}秒" if 'seconds' in extension else "")
            + f"；合并候选记录共{len(training)}条。{selected_description}所有追加仅依据训练/验证数据，"
              f"详见[稳定性补训]({_link(stability_path)})。")
    train_seconds = float(training.groupby(["algorithm", "config_index", "seed"]).training_seconds.max().sum()) if not training.empty else 0
    all_status = list(root.glob("*.status.json"))
    evaluation_seconds = sum(float(_read(p).get("seconds", 0)) for p in all_status)
    winner = robust.iloc[0]
    value_winner = robust.sort_values("mean_value", ascending=False).iloc[0]
    overall_relative_leaders = "、".join(LABELS.get(n,n) for n in robust.loc[robust.scenario_mean == winner.scenario_mean, "algorithm"])
    overall_value_leaders = "、".join(LABELS.get(n,n) for n in robust.loc[robust.mean_value == value_winner.mean_value, "algorithm"])
    worst = robust.sort_values(["worst_scenario", "scenario_mean"], ascending=False).iloc[0]
    lowtail = robust.sort_values(["mean_role_p10", "scenario_mean"], ascending=False).iloc[0]
    source_days = protocol["executed_split"]
    data_scope = ", ".join(f"{k}={','.join(v)}" for k,v in source_days.items())
    rows = []
    scenario_maxima = summary.groupby("scenario")[["mean_total_value", "mean_relative_performance"]].max()
    for name in names:
        group = summary[summary.algorithm == name].set_index("scenario")
        def triple(field, digits=3):
            cells = []
            for scenario in SCENARIOS:
                value = group.loc[scenario,field]
                cell = _fmt(value,digits)
                if field in scenario_maxima and pd.notna(value) and value == scenario_maxima.loc[scenario,field]:
                    cell = "**"+cell+"**"
                cells.append(cell)
            return " / ".join(cells)
        rows.append([LABELS.get(name,name), triple("mean_total_value"), triple("mean_relative_performance"),
                     " / ".join(f"{100*group.loc[s,'mean_budget_utilization']:.1f}%" for s in SCENARIOS),
                     triple("relative_cross_player_std")])
    fixed_rows = []
    for name in names:
        group = fixed[fixed.algorithm == name].set_index("scenario")
        fixed_rows.append([LABELS.get(name,name)] + [f"{group.loc[s,'total_value']:.3f} / {group.loc[s,'relative_performance']:.3f}" for s in SCENARIOS])
    algo_explanation = {
        "FixedAlpha": ("固定α；同片bid=α×pvalue", "数据尺度确定候选范围，验证集选择全角色共享α"),
        "LocalPID": ("依据自身历史预算偏差调节α", "调初值及P/I/D；每局reset，本地参考，非官方PID"),
        "OnlineLP": ("按剩余预算查询经验成本曲线", "复用上游查询；train7临时Fixed对手轨迹拟合72片曲线、验证倍率"),
        "BC": ("监督模仿离线动作", "上游BC网络及损失；两个学习率、两个训练seed、两个checkpoint"),
        "IQL": ("期望分位V与优势加权策略学习", "上游IQL训练；同一离线集及验证选择"),
        "BCQ": ("VAE生成数据附近动作并用Q筛选", "上游BCQ，缩小隐藏层并修复two-hot边界"),
        "TD3BC": ("双Q策略学习结合行为克隆约束", "上游TD3+BC；修复分布Q解码及two-hot，缩小隐藏层"),
        "CQL": ("对离线分布外动作施加保守Q约束", "上游CQL；同步目标网络初始化，同一数据选模"),
        "DT": ("以历史状态、动作、目标回报预测动作", "上游Transformer；72片序列，目标回报只由训练集确定")}
    algo_rows = [[LABELS.get(n,n), *algo_explanation.get(n,("见源码审计", "见配置")), status_index.loc[n,"status"]] for n in all_names]
    candidate_rows = []
    for name in all_names:
        chosen = [c for c in selected["selected"] if c["algorithm"] == name]
        matches = training[training.algorithm == name] if not training.empty else pd.DataFrame()
        params = str(int(matches.parameter_count.max())) if not matches.empty else "—"
        seconds = matches.groupby(["config_index", "seed"]).training_seconds.max().sum() if not matches.empty else 0
        choice = ", ".join(sorted({c.get("validation_choice", "冻结参数") for c in chosen})) if chosen else "无有效最终策略；未排名"
        training_scores = tables["per_training_seed_summary"].query("budget_scale == 1 and algorithm == @name").groupby("training_seed").relative_performance.mean()
        training_range = f"{training_scores.min():.3f}–{training_scores.max():.3f}" if len(training_scores)>1 else "—"
        candidate_rows.append([LABELS.get(name,name), choice.replace("|", " / "), params, f"{seconds:.1f}" if seconds else "参数搜索", training_range])
    scenario_winners = []
    for scenario in SCENARIOS:
        ranked = summary[summary.scenario == scenario].sort_values("mean_relative_performance", ascending=False)
        row = ranked.iloc[0]
        value_row = ranked.sort_values("mean_total_value", ascending=False).iloc[0]
        relative_leaders = set(ranked.loc[ranked.mean_relative_performance == row.mean_relative_performance, "algorithm"])
        value_leaders = set(ranked.loc[ranked.mean_total_value == value_row.mean_total_value, "algorithm"])
        relative_label = "、".join(LABELS.get(n,n) for n in _ordered(relative_leaders))
        value_label = "、".join(LABELS.get(n,n) for n in _ordered(value_leaders))
        if relative_leaders == value_leaders:
            scenario_winners.append(f"{scenario}：**{relative_label}**的V与R均最高（V={value_row.mean_total_value:.3f}，R={row.mean_relative_performance:.3f}）")
        else:
            scenario_winners.append(f"{scenario}：V最高为**{value_label}**（{value_row.mean_total_value:.3f}），R最高为**{relative_label}**（{row.mean_relative_performance:.3f}）")
    maximum_budget = summary.mean_budget_utilization.max()
    budget_text = ("原始预算仍有较多剩余，预算约束偏弱；本轮未追加预算压力排名。" if maximum_budget < .5 else
                   "至少部分策略显著消耗原始预算，预算尺度已对行为产生影响；各策略的剩余预算见图2，未混入缩放预算排名。")
    neural_names = {"BC", "IQL", "BCQ", "TD3BC", "CQL", "DT"}
    neural_best = robust[robust.algorithm.isin(neural_names)]
    classical_best = robust[~robust.algorithm.isin(neural_names)]
    comparative = "神经网络结果不足，不能比较复杂方法与经典基线。"
    if not neural_best.empty and not classical_best.empty:
        n, c = neural_best.iloc[0], classical_best.iloc[0]
        comparative = f"复杂方法中{LABELS.get(n.algorithm,n.algorithm)}的平均R为{n.scenario_mean:.3f}，经典方法中{LABELS.get(c.algorithm,c.algorithm)}为{c.scenario_mean:.3f}。仅一个测试周期和两个评测seed，不能据此宣称统计显著优于或长期优于。"
    inference = tables["inference_summary"].query("budget_scale == 1")
    max_p99 = inference.max_episode_paired_p99_ms.max() if "max_episode_paired_p99_ms" in inference else np.nan
    memory_column = "max_episode_peak_process_rss_mb" if "max_episode_peak_process_rss_mb" in inference else "max_episode_rss_mb"
    max_memory = inference[memory_column].max() if memory_column in inference else np.nan
    memory_label = "Windows进程生命周期峰值工作集最大值" if "max_episode_peak_process_rss_mb" in inference else "终局采样进程RSS最大值（不是连续峰值）"
    runtime_text = f"实际所有候选的最大单局bidding+update P99={max_p99:.2f} ms；{memory_label}={max_memory:.0f} MiB。"
    runtime_text += " 进程RSS包含Python、模型与映射数据，不能解释为独立Agent内存。"
    runtime_text += (" 本地采样未超过200 ms / 4 GiB阈值，但不等同官方环境认证。" if max_p99<=200 and max_memory<=4096 else
                     " 本地测量至少有一项超过200 ms / 4 GiB阈值，详见逐算法推理CSV。")
    undefined = int(summary.zero_reference_roles.sum())
    text = f"""# AuctionNet Baseline：72片完整周期、有限训练预算实验

本轮主排名只包含在 **Fixed / 本地PID / Mixed** 三种对手下完成全部 **{len(protocol['test_roles'])}** 个广告主轮换、全部seed的冻结策略；每局只替换一人，未完成算法另列。三场景平均累计价值最高为 **{overall_value_leaders}**（V={value_winner.mean_value:.3f}）；逐角色归一化R最高为 **{overall_relative_leaders}**（R={winner.scenario_mean:.3f}）。V直接对应赛题价值目标，R用于同角色参考比较，二者排名可能不同。这是所测数据与训练预算下的描述性结果，不代表算法已收敛或官方比赛排名。

## 实验口径

- **数据**：保留原pValue、原预算2,000–4,850，每周期约50万请求、48人，分布在72片。使用 **2训练 / 1验证 / 1测试** 完整周期；{data_scope}。全库共21周期，其余17周期未进入本轮比较；小样本288请求仅用于功能检查。原48→72中点映射是本地时间假设。
- **规则**：唯一Core AuctionEngine，单广告位二价、付不起重新竞价、稀疏mask、随机平局；α无比赛级上限；奖励仅为竞胜pValue和。观察仅含自身配置、时钟和已完成片反馈。
- **对手**：训练集逐角色调Fixed α，验证集选本地PID参数后冻结；Mixed每局seed固定分配23 Fixed+24 PID。待测FixedAlpha/LocalPID全角色共享α与控制参数，神经网络共享权重；只有对手使用角色α校准表，OnlineLP成本表按类别共享。所有候选在不同场景测试，不按测试角色重新选模。
- **随机与汇总**：评测seed={protocol['evaluation_seeds']}，神经网络训练seed={protocol['training_seeds']}。先在每角色内平均周期、评测seed与训练seed，再除以同角色同场景LocalPID均值，最后等权平均角色；不是总价值之比。零参考不加epsilon，本次未定义角色单元数={undefined}。误差线为评测seed角色均值的±1样本SD，不是置信区间。Fixed/PID的seed主要改变平局，Mixed还改变对手分配，BCQ另有候选动作采样；两seed不是两份独立市场样本。

## 算法与训练

{_table(['算法','核心思想','实际适配与选择','完成状态'], algo_rows)}

{validation_failure_text}

上游固定版本`{protocol.get('source_upstream_commit','见审计')}`。**上游Abid是依赖当前pValue的二次报价，不能原样满足片内固定α；Fixed Alpha没有冒称原始Abid。** 上游48片PID启发式未直接沿用，采用项目LocalPID。MBRL-MOPO/COMBOMICRO仅有wrapper，缺少所引用模型及训练模块，本轮未完成、不进入排名。神经网络结构压缩、数值修复及动作变换详见[源码审计]({_link(ROOT/'harmony_baselines'/'UPSTREAM_AUDIT.md')})。

OnlineLP只保留上游查表核心；成本表来自train7临时Fixed对手的校准轨迹，未直接覆盖PID/Mixed。其内部`1.5×fallback`边界仍保留并受验证倍率缩放，属于策略范围，不是比赛全局α上限；此结果不能代表完整原始OnlineLP训练器的上限。

离线数据全部由本轮统一环境生成：{provenance.get('episodes','见数据清单')}条72片轨迹、{provenance.get('transitions',0):,}个transition（{'含对手自身合法轨迹' if provenance.get('includes_opponents_own_legal_trajectories') else '仅保留被控角色自身轨迹'}）；没有回放旧CPA奖励或旧拍卖轨迹。训练时间合计约 **{train_seconds/60:.1f}分钟**，各实验阶段墙钟合计约{evaluation_seconds/60:.1f}分钟；CPU运行，具体配置及资源见[实验协议]({_link(root/'protocol.json')})。这是有限训练预算比较，不能据短训练判断算法理论上限。

{_table(['算法','验证集最终选择','参数数','训练秒数（全部候选）','训练seed的测试R范围'], candidate_rows)}

{stability_text}

最终参数、Checkpoint路径和SHA保存在[冻结选择]({_link(root/'selected_frozen.json')})。**对手**Fixed搜索α={', '.join(f'{x:.2f}' for x in calibration.get('alpha_grid',[]))}；**目标策略**共享初始α基数={_fmt(classical_search.get('shared_initial_alpha', np.nan),2)}，实际搜索倍率：{multiplier_text}。{'OnlineLP在初始网格下界胜出后，仅用验证集向更低倍率扩展；最终选择及全部真实分数见classical_search/refinement记录。' if classical_search.get('refinement') else ''} PID三组控制参数；神经网络学习率与checkpoint统一验证选择。表中训练seed范围是每seed跨三场景的测试平均R，非选模依据；[分训练seed完整CSV]({_link(root/'per_training_seed_summary.csv')})保留波动。

## 全广告主结果（主要比较）

![三种场景下全角色相对表现]({_link(figures['01_performance'])})

表内三元组顺序均为 **Fixed / PID / Mixed**；V为累计竞胜价值，R为相对LocalPID，σ为跨角色R的样本标准差。各场景最高V、最高R分别加粗，精确并列均标出。

{_table(['算法','平均V','平均R','平均预算使用率','跨角色σ(R)'], rows)}

各场景：{'；'.join(scenario_winners)}。全部算法的竞胜次数、中位数、P10、有效角色数与seed范围见[完整汇总CSV]({_link(root/'all_player_summary.csv')})。

![预算使用率]({_link(figures['02_budget'])})

{budget_text}

![按广告主类型的相对表现]({_link(figures['03_player_types'])})

逐player结果完整保存在[逐角色CSV]({_link(root/'per_player.csv')})。平均角色P10最高的是 **{LABELS.get(lowtail.algorithm,lowtail.algorithm)}**（{lowtail.mean_role_p10:.3f}），可作为角色间稳健性的一个指标；所有角色均有训练数据，这不是“未见过角色”的泛化实验。

![跨场景稳健性]({_link(figures['04_robustness'])})

按最低场景R，**{LABELS.get(worst.algorithm,worst.algorithm)}**最稳健（最低{worst.worst_scenario:.3f}，三场景均值{worst.scenario_mean:.3f}）；图中须同时看均值和最差值，单看波动小会把稳定的低性能策略误判为优秀。

## 固定角色辅助结果与结论

固定player={protocol.get('fixed_role','0')}；单元格为 **V / R**，仅辅助诊断，不用于主要排名。

{_table(['算法','Fixed','PID','Mixed'], fixed_rows)}

{comparative} **仅一个完整测试周期，不能据此判断跨日泛化。** 后续优先扩大完整验证/测试周期，并按角色低分位补足行为数据覆盖；再延长领先的1–2种方法训练、检查验证学习曲线，避免仅增加模型复杂度。

{runtime_text} [逐算法推理开销]({_link(root/'inference_summary.csv')})记录实际Baseline，未用LocalPID代测。

**可复现交付**：[实验配置]({_link(root/'protocol.json')})、[对手冻结参数]({_link(root/'opponents_frozen.json')})、[逐局原始CSV]({_link(root/'episode_results.csv')})、[数据范围]({_link(root/'data_audit.json')})。全部图由CSV生成，并保存PNG与SVG；在AuctionNet目录运行`.venv-baselines/Scripts/python.exe -m harmony_baselines.report`可重建本报告。图表与排名只含实际完成全角色测试的算法；训练失败、验证异常、资源未完成和未运行候选分别记录，不使用旧实验成绩替代。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def generate_report(root=DEFAULT_OUT, report_path=None):
    root = Path(root)
    frame, protocol, selected = audit_complete_results(root)
    (root/"report_coverage.json").write_text(json.dumps(protocol["_report_audit"],ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tables = aggregate_results(frame, protocol.get("fixed_role", "0"))
    for name, table in tables.items():
        table.to_csv(root/(name+".csv"), index=False)
    figures = draw_figures(tables, root/"figures", 1.)
    path = write_report(root, tables, protocol, selected, figures, report_path)
    output = {"status": "completed", "source_csv_sha256": sha256_file(root/"episode_results.csv"),
              "experiment_status": protocol["status"], "incomplete_algorithms": protocol["_report_audit"]["incomplete_algorithms"],
              "report": str(path.resolve()), "report_sha256": sha256_file(path),
              "failure_evidence_sha256": {str(p.name): sha256_file(p) for p in [root/"validation_failures.json", root/"validation_skipped.json", _test_outcome_path(root,"failures"), _test_outcome_path(root,"skipped"), root/"algorithm_status.csv", root/"report_coverage.json"] if p.exists()},
              "figures": {k: {"path": str(p), "sha256": sha256_file(p)} for k,p in figures.items()}}
    (root/"report.status.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_OUT))
    parser.add_argument("--report-path")
    args = parser.parse_args()
    print(json.dumps(generate_report(args.root,args.report_path),ensure_ascii=True),flush=True)
