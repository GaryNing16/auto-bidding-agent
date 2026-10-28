"""Independent five-scenario statistics and reports from completed real records.

Never imports a training/evaluation entry point and never rewrites historical
assets. A/B normalization is internal to each experiment. Paired improvements
compare value, averaging training seeds before pairing role/evaluation seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "outputs/baseline_five_scenarios_20261028"
OLD_ROOT = ROOT / "outputs/baseline_comparison_20261028"
ALGORITHMS = ("FixedAlpha", "LocalPID", "AuctionNetPID", "OnlineLP", "BC", "IQL", "BCQ", "TD3BC", "CQL", "DT")
NEURAL = ("BC", "IQL", "BCQ", "TD3BC", "CQL", "DT")
SCENARIOS = ("Fixed", "LocalPID", "Mixed", "AuctionNetPID", "Mixed2")
PHASES = ("experiment_A", "experiment_B", "independent_test")
EVAL_SEEDS = (20261028, 20261029)
TRAIN_SEEDS = (20261028, 20261029)
LABEL = {"FixedAlpha": "Fixed Alpha", "OnlineLP": "Online LP", "TD3BC": "TD3+BC"}
COLORS = dict(zip(ALGORITHMS, ("#64748b", "#111827", "#db2777", "#7c3aed", "#0891b2", "#2563eb", "#ea580c", "#a855f7", "#dc2626", "#059669")))
KEYS = ["algorithm", "scenario", "day_id", "player_id", "seed", "training_seed"]
ROLE_KEYS = ["algorithm", "scenario", "player_id", "player_type"]
VALUE_COLUMNS = ["total_value", "total_cost", "initial_budget", "budget_utilization", "wins"]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Sources:
    def __init__(self):
        self.hashes = {}
        self.recovery_evidence = {}

    def record(self, path):
        path = Path(path).resolve()
        self.hashes[str(path)] = digest(path)
        return path

    def json(self, path):
        path=Path(path).resolve()
        data=path.read_bytes()
        self.hashes[str(path)]=hashlib.sha256(data).hexdigest()
        return json.loads(data.decode("utf-8-sig"))

    def csv(self, path, **kwargs):
        return pd.read_csv(self.record(path), **kwargs)

    def text(self, path):
        path=Path(path).resolve()
        data=path.read_bytes()
        self.hashes[str(path)]=hashlib.sha256(data).hexdigest()
        return data.decode("utf-8-sig")


def canonical_scene(values):
    return values.replace({"PID": "LocalPID"})


def validate_phase(frame, *, days, roles=tuple(str(i) for i in range(48)), algorithms=ALGORITHMS,
                   evaluation_seeds=EVAL_SEEDS, training_seeds=TRAIN_SEEDS):
    """Check the required Cartesian coverage, not merely a plausible row count."""
    required = set(KEYS + VALUE_COLUMNS + ["player_type", "completed", "task_index", "model_fingerprint",
        "dataset_fingerprint", "rules_version", "assignment_fingerprint", "opponent_fingerprint"])
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing result fields: {sorted(missing)}")
    frame = frame.copy()
    frame["player_id"] = frame.player_id.astype(str)
    frame["player_type"] = frame.player_type.astype(str)
    if (frame.scenario == "PID").any():
        raise ValueError("Import historical PID as canonical LocalPID before reporting; preserve source_scenario")
    if frame.duplicated(KEYS).any() or frame.task_index.duplicated().any():
        raise ValueError("Duplicate case identity or stage task_index")
    complete = frame.completed.astype(str).str.lower().isin(["true", "1"])
    if not complete.all():
        raise ValueError("Incomplete episodes cannot enter ranked results")
    if "completed_slots" in frame and (frame.completed_slots.dropna() != 72).any():
        raise ValueError("Every completed episode must cover all 72 slots")
    if not np.isfinite(frame[VALUE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite business metrics")
    if (frame[VALUE_COLUMNS] < 0).any().any() or (frame.initial_budget <= 0).any():
        raise ValueError("Negative metric or nonpositive budget")
    if (frame.total_cost > frame.initial_budget + 1e-6).any():
        raise ValueError("Cost exceeds initial budget")
    if not np.allclose(frame.budget_utilization, frame.total_cost / frame.initial_budget, rtol=1e-10, atol=1e-10):
        raise ValueError("Budget utilization differs from actual cost / initial budget")
    expected = {(a, s, str(d), str(p), int(e), int(h)) for a in algorithms for s in SCENARIOS for d in days
                for p in roles for e in evaluation_seeds for h in (training_seeds if a in NEURAL else (0,))}
    actual = set(frame[KEYS].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"Incomplete protocol coverage: missing={len(expected-actual)}, unexpected={len(actual-expected)}")
    # Imported rows receive the same canonical spec+asset policy ID as new
    # executions. Different provenance never permits a different policy.
    fingerprint = "policy_id" if "policy_id" in frame else "model_fingerprint"
    if frame[fingerprint].isna().any() or frame.groupby(["algorithm", "training_seed"])[fingerprint].nunique().gt(1).any():
        raise ValueError("Different policies concatenated into one algorithm/training-seed result")
    if frame.groupby(["day_id"])["dataset_fingerprint"].nunique().gt(1).any():
        raise ValueError("Different market data within a test day")
    if frame.rules_version.nunique() != 1:
        raise ValueError("Different auction rules mixed in one phase")
    for column in ["assignment_fingerprint", "initial_budget", "player_type"]:
        group = ["scenario", "day_id", "player_id", "seed"] if column == "assignment_fingerprint" else ["day_id", "player_id"]
        if frame.groupby(group)[column].nunique().gt(1).any():
            raise ValueError(f"Unequal case conditions: {column}")
    return frame


def summarize_roles(group):
    return {"roles": len(group), "mean_V": group.total_value.mean(), "std_V": group.total_value.std(ddof=1),
        "median_V": group.total_value.median(), "p10_V": group.total_value.quantile(.1),
        "mean_R": group.R.mean(), "std_R": group.R.std(ddof=1), "median_R": group.R.median(),
        "p10_R": group.R.quantile(.1), "mean_budget_utilization": group.budget_utilization.mean(),
        "mean_wins": group.wins.mean(), "defined_R_roles": group.R.count(),
        "undefined_R_roles": int(group.R.isna().sum())}


def aggregate_phase(frame):
    role = frame.groupby(ROLE_KEYS, as_index=False)[VALUE_COLUMNS].mean()
    reference = role[role.algorithm == "LocalPID"][["scenario", "player_id", "total_value"]].rename(
        columns={"total_value": "reference_V"})
    role = role.merge(reference, on=["scenario", "player_id"], validate="many_to_one")
    role["R"] = np.divide(role.total_value, role.reference_V, out=np.full(len(role), np.nan), where=role.reference_V > 0)
    role["reference_status"] = np.where(role.reference_V > 0, "defined", "zero_reference")
    scenario = pd.DataFrame([dict(algorithm=a, scenario=s, **summarize_roles(g))
        for (a, s), g in role.groupby(["algorithm", "scenario"])])
    overall_role = role.groupby(["algorithm", "player_id", "player_type"], as_index=False)[VALUE_COLUMNS + ["R", "reference_V"]].mean()
    overall = pd.DataFrame([dict(algorithm=a, **summarize_roles(g)) for a, g in overall_role.groupby("algorithm")])
    # Main R gives every scenario equal weight after averaging its defined
    # roles. Role-distribution SD/median/P10 keep the separately declared order.
    overall["mean_role_average_R"]=overall["mean_R"]
    scenario_R=scenario.groupby("algorithm",as_index=False).agg(mean_defined_scenario_R=("mean_R","mean"),
        defined_R_scenarios=("mean_R","count"),defined_R_role_scenario_cells=("defined_R_roles","sum"),
        undefined_R_role_scenario_cells=("undefined_R_roles","sum"))
    overall=overall.merge(scenario_R,on="algorithm",validate="one_to_one")
    overall["mean_R"]=overall.mean_defined_scenario_R.where(overall.defined_R_scenarios==len(SCENARIOS))
    stability = scenario.groupby("algorithm", as_index=False).agg(worst_scenario_V=("mean_V", "min"),
        best_scenario_V=("mean_V", "max"), scenario_std_V=("mean_V", "std"),
        worst_scenario_R=("mean_R", "min"), best_scenario_R=("mean_R", "max"),
        scenario_std_R=("mean_R", "std"), mean_scenario_p10_R=("p10_R", "mean"))
    overall = overall.merge(stability, on="algorithm", validate="one_to_one").sort_values(["mean_V","algorithm"], ascending=[False,True])
    overall["rank_V"] = overall.mean_V.rank(method="min",ascending=False).astype(int)
    types = pd.DataFrame([dict(algorithm=a, scenario=s, player_type=t, **summarize_roles(g))
        for (a, s, t), g in role.groupby(["algorithm", "scenario", "player_type"])])
    per_seed = frame.groupby(ROLE_KEYS + ["seed"], as_index=False)[VALUE_COLUMNS].mean()
    per_training_seed = frame.groupby(ROLE_KEYS + ["training_seed"], as_index=False)[VALUE_COLUMNS].mean()
    seed_summary = per_seed.groupby(["algorithm", "scenario", "seed"], as_index=False)[VALUE_COLUMNS].mean()
    training_seed_summary = per_training_seed.groupby(["algorithm", "scenario", "training_seed"], as_index=False)[VALUE_COLUMNS].mean()
    inference_rows = []
    for (a, s), g in frame.groupby(["algorithm", "scenario"]):
        row = {"algorithm": a, "scenario": s, "episodes": len(g)}
        for column in ["bidding_p99_ms", "update_p99_ms", "paired_p99_ms", "rss_mb", "peak_process_rss_mb"]:
            if column in g:
                row["mean_episode_" + column] = g[column].mean()
                row["max_episode_" + column] = g[column].max()
        inference_rows.append(row)
    contrasts = []
    for before, after in [("LocalPID", "AuctionNetPID"), ("Mixed", "Mixed2"), ("Fixed", "AuctionNetPID")]:
        left = role[role.scenario == before]
        right = role[role.scenario == after]
        merged = left.merge(right,on=["algorithm","player_id","player_type"],suffixes=("_before","_after"),validate="one_to_one")
        merged["delta_V"] = merged.total_value_after - merged.total_value_before
        merged["relative_V_change"] = np.divide(merged.delta_V,merged.total_value_before,out=np.full(len(merged),np.nan),where=merged.total_value_before>0)
        for algorithm,g in merged.groupby("algorithm"):
            contrasts.append(dict(algorithm=algorithm,before_scenario=before,after_scenario=after,
                before_mean_V=g.total_value_before.mean(),after_mean_V=g.total_value_after.mean(),
                mean_delta_V=g.delta_V.mean(),mean_role_relative_change=g.relative_V_change.mean(),
                undefined_role_percentages=int(g.relative_V_change.isna().sum()),
                delta_budget_utilization=(g.budget_utilization_after-g.budget_utilization_before).mean()))
    return {"per_role": role, "per_scenario": scenario, "overall": overall, "per_type": types,
        "fixed_role0": role[role.player_id == "0"], "per_evaluation_seed": seed_summary,
        "per_training_seed": training_seed_summary, "inference": pd.DataFrame(inference_rows),
        "undefined_references": role[role.reference_V <= 0], "scenario_contrasts":pd.DataFrame(contrasts)}


def paired_improvement(frozen, tuned):
    """Mean training seeds first; pair identical role/market/evaluation seed.

    This does not compare internal R values, whose LocalPID denominators differ.
    A zero-V reference is undefined for percentage gain but its absolute delta
    is retained. Main percentages average individual paired percentages.
    """
    pair_keys = ["algorithm", "scenario", "day_id", "player_id", "player_type", "seed"]
    metrics = ["total_value", "budget_utilization", "wins", "initial_budget"]
    a = frozen.groupby(pair_keys, as_index=False)[metrics].mean()
    b = tuned.groupby(pair_keys, as_index=False)[metrics].mean()
    pairs = a.merge(b, on=pair_keys, suffixes=("_A", "_B"), how="outer", indicator=True, validate="one_to_one")
    if not (pairs._merge == "both").all():
        raise ValueError("A/B paired improvement requires identical case coverage")
    pairs = pairs.drop(columns="_merge")
    if not np.allclose(pairs.initial_budget_A, pairs.initial_budget_B, rtol=0, atol=1e-9):
        raise ValueError("A/B initial budgets differ")
    # The imported old A fingerprint uses its original two-opponent object;
    # assignment identity and actual data are invariant across the two phases.
    condition_keys = ["scenario", "day_id", "player_id", "seed"]
    for col in ["dataset_fingerprint", "rules_version", "assignment_fingerprint"]:
        ca = frozen[condition_keys + [col]].drop_duplicates()
        cb = tuned[condition_keys + [col]].drop_duplicates()
        joined = ca.merge(cb, on=condition_keys, suffixes=("_A", "_B"), validate="one_to_one")
        if not (joined[col + "_A"] == joined[col + "_B"]).all():
            raise ValueError("A/B paired conditions differ: " + col)
    pairs["delta_V"] = pairs.total_value_B - pairs.total_value_A
    pairs["relative_gain"] = np.divide(pairs.delta_V, pairs.total_value_A, out=np.full(len(pairs), np.nan), where=pairs.total_value_A > 0)
    pairs["relative_gain_status"] = np.where(pairs.total_value_A > 0, "defined", "zero_frozen_value")
    pairs["delta_budget_utilization"] = pairs.budget_utilization_B - pairs.budget_utilization_A
    def summary(g):
        relative_roles = g.groupby(["scenario", "player_id"]).relative_gain.mean()
        equal_role_scene_gain = relative_roles.groupby("scenario").mean().mean()
        return {"pairs": len(g), "A_mean_V": g.total_value_A.mean(), "B_mean_V": g.total_value_B.mean(),
            "mean_delta_V": g.delta_V.mean(), "median_delta_V": g.delta_V.median(),
            "p10_delta_V": g.delta_V.quantile(.1), "std_delta_V": g.delta_V.std(ddof=1),
            "mean_paired_relative_gain": equal_role_scene_gain, "median_paired_relative_gain": g.relative_gain.median(),
            "defined_relative_pairs": g.relative_gain.count(), "zero_reference_pairs": int(g.relative_gain.isna().sum()),
            "defined_relative_role_scenario_cells": int(relative_roles.notna().sum()),
            "ratio_of_mean_V_gain": g.total_value_B.mean() / g.total_value_A.mean() - 1 if g.total_value_A.mean() > 0 else np.nan,
            "mean_budget_change": g.delta_budget_utilization.mean(), "B_wins": int((g.delta_V > 1e-12).sum()),
            "A_wins": int((g.delta_V < -1e-12).sum()), "ties": int((g.delta_V.abs() <= 1e-12).sum())}
    scenario = pd.DataFrame([dict(algorithm=a, scenario=s, **summary(g)) for (a, s), g in pairs.groupby(["algorithm", "scenario"])])
    overall = pd.DataFrame([dict(algorithm=a, **summary(g)) for a, g in pairs.groupby("algorithm")])
    for table in (scenario,overall):
        table["relative_vs_mean_V_sign_reversal"]=(table.mean_delta_V.abs()>1e-12)&(table.mean_paired_relative_gain.abs()>1e-12)&(
            np.sign(table.mean_delta_V)!=np.sign(table.mean_paired_relative_gain))
    roles = pairs.groupby(["algorithm", "scenario", "player_id", "player_type"], as_index=False)[
        ["total_value_A", "total_value_B", "delta_V", "relative_gain", "delta_budget_utilization"]].mean()
    return {"pairs": pairs, "per_role": roles, "per_scenario": scenario, "overall": overall,
            "undefined_relative": pairs[pairs.relative_gain.isna()]}


def _load_slots(directory, episodes, sources):
    path = directory / "slots.csv"
    columns = ["task_index", "slot", "alpha", "cost", "value", "wins", "remaining_budget"]
    parts = []
    if path.exists():
        header = pd.read_csv(path, nrows=0).columns
        use = [c for c in columns if c in header]
        if "slot" not in use and "time_slot" in header:
            use.append("time_slot")
        part = sources.csv(path, usecols=use)
        if "slot" not in part:
            part = part.rename(columns={"time_slot": "slot"})
        if "wins" not in part:
            part["wins"] = np.nan
        part["slot_source"] = np.where(part.wins.isna(), "imported_historical_slots_wins_unavailable", "new_slots_csv")
        parts.append(part)
    present = set(parts[0].task_index.unique()) if parts else set()
    historical = []
    for row in episodes[~episodes.task_index.isin(present)].itertuples():
        if getattr(row, "provenance", "") not in {"historical_reuse", "historical_exact_policy_reuse"}:
            raise ValueError(f"Missing new slot trace for task {row.task_index}")
        old_index = getattr(row, "old_task_index", None)
        if old_index is None or pd.isna(old_index):
            source_task = str(getattr(row, "source_task", ""))
            old_index = int(source_task.split(":")[-1])
        path = OLD_ROOT / f"test_diagnostics/task_{int(old_index)}.jsonl"
        sources.record(path)
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for event in events:
            if event["phase"] == "feedback":
                historical.append({"task_index": row.task_index, "slot": event["slot"], "alpha": event["alpha"],
                    "cost": event["cost"], "value": event["value"], "wins": np.nan,
                    "remaining_budget": event["remaining_budget"], "slot_source": "historical_jsonl_wins_unavailable"})
    if historical:
        parts.append(pd.DataFrame(historical))
    if not parts:
        raise ValueError("No 72-slot evidence available")
    slots = pd.concat(parts, ignore_index=True)
    if slots.duplicated(["task_index", "slot"]).any():
        raise ValueError("Duplicate slot record")
    if set(slots.task_index.unique()) != set(episodes.task_index.unique()):
        raise ValueError("Slot task identity differs from result coverage")
    identity = episodes[["task_index"] + KEYS + ["player_type", "initial_budget", "total_value", "total_cost"]]
    slots = slots.merge(identity, on="task_index", validate="many_to_one")
    if len(slots) != len(episodes) * 72:
        raise ValueError("Missing or extra slot records")
    numeric = ["alpha", "cost", "value", "remaining_budget"]
    if not np.isfinite(slots[numeric].to_numpy()).all() or (slots[numeric] < 0).any().any():
        raise ValueError("Invalid slot metrics")
    counts = slots.groupby("task_index").slot.agg(["count", "min", "max", "nunique"])
    if not ((counts["count"] == 72) & (counts["min"] == 0) & (counts["max"] == 71) & (counts["nunique"] == 72)).all():
        raise ValueError("Each episode must contain exactly slots 0..71")
    sums = slots.groupby("task_index")[["cost", "value"]].sum().join(episodes.set_index("task_index")[["total_cost", "total_value"]])
    if not np.allclose(sums.cost, sums.total_cost, atol=1e-7, rtol=1e-11) or not np.allclose(sums.value, sums.total_value, atol=1e-10, rtol=1e-10):
        raise ValueError("Slot sums disagree with episode scores")
    slots=slots.sort_values(["task_index", "slot"])
    expected_remaining=slots.initial_budget-slots.groupby("task_index").cost.cumsum()
    if not np.allclose(slots.remaining_budget,expected_remaining,atol=2e-6,rtol=1e-11):
        raise ValueError("Slot remaining budget disagrees with cumulative costs")
    return slots


def behavioral_statistics(directory, episodes, sources):
    slots = _load_slots(directory, episodes, sources)
    slots["slot_number"] = slots.slot + 1
    slots["cumulative_spend_fraction"] = 1 - slots.remaining_budget / slots.initial_budget
    slots["cumulative_value"] = slots.groupby("task_index").value.cumsum()
    records = []
    for task, g in slots.groupby("task_index", sort=False):
        row = g.iloc[0]
        r = {c: row[c] for c in KEYS + ["player_type", "initial_budget", "total_value", "total_cost"]}
        r.update(task_index=task, first_slot_spend=g.cost.iloc[0] / row.initial_budget,
            first6_spend=g.cost.iloc[:6].sum() / row.initial_budget,
            first36_spend=g.cost.iloc[:36].sum() / row.initial_budget,
            alpha_median=g.alpha.median(), alpha_p90=g.alpha.quantile(.9), alpha_max=g.alpha.max(),
            late_value_share=g.value.iloc[36:].sum() / row.total_value if row.total_value > 0 else np.nan,
            final_budget_utilization=g.cumulative_spend_fraction.iloc[-1],
            exact_zero_balance=bool((g.remaining_budget == 0).any()), wins_available=bool(g.wins.notna().all()))
        for threshold in (50, 95, 99):
            hit = g.loc[g.cumulative_spend_fraction >= threshold / 100, "slot_number"]
            r[f"slot{threshold}"] = int(hit.iloc[0]) if len(hit) else np.nan
        records.append(r)
    episode = pd.DataFrame(records)
    population = slots.groupby(["algorithm", "scenario", "slot_number"], as_index=False).agg(
        episodes=("task_index", "size"), cumulative_spend_mean=("cumulative_spend_fraction", "mean"),
        cumulative_value_mean=("cumulative_value", "mean"), alpha_median=("alpha", "median"),
        alpha_p90=("alpha", lambda v: v.quantile(.9)), alpha_max=("alpha", "max"))
    rows = []
    for a, g in episode.groupby("algorithm"):
        alpha = slots.loc[slots.algorithm == a, "alpha"]
        row = dict(algorithm=a, episodes=len(g), first_slot_spend=g.first_slot_spend.mean(),
            first6_spend=g.first6_spend.mean(), first36_spend=g.first36_spend.mean(),
            alpha_median=alpha.median(), alpha_p90=alpha.quantile(.9), alpha_max=alpha.max(),
            late_value_share=g.late_value_share.mean(), exact_zero_episodes=int(g.exact_zero_balance.sum()),
            pooled_value_per_cost=g.total_value.sum() / g.total_cost.sum() if g.total_cost.sum() > 0 else np.nan,
            wins_unavailable_episodes=int((~g.wins_available).sum()))
        for threshold in (50, 95, 99):
            v = g[f"slot{threshold}"]
            row[f"slot{threshold}_median_reached"] = v.median()
            row[f"slot{threshold}_never"] = int(v.isna().sum())
        rows.append(row)
    return {"episodes": episode, "population_slots": population, "overall": pd.DataFrame(rows)}


def training_evidence(root, sources):
    """Discover actual candidate/training/validation records; never invent points."""
    # Inner attempt catalogs use local config_index=0. Only the global catalog
    # owns the final config identity; recursively merging it would mislabel LR.
    catalogs = [root / "models" / "candidates.json"]
    candidates, run_keys, losses, runs = [], set(), [], []
    for path in catalogs:
        value = sources.json(path)
        records = value if isinstance(value, list) else value.get("candidates", [])
        for record in records:
            if record.get("algorithm") not in NEURAL or "checkpoint" not in record:
                continue
            candidates.append({**record, "catalog_source": str(path)})
            log_path = Path(record.get("training_log", Path(record["checkpoint"]).parent / "training.json"))
            if log_path in run_keys or not log_path.exists():
                continue
            run_keys.add(log_path)
            log = sources.json(log_path)
            config = log.get("config", record.get("config", {}))
            common = {"algorithm": record["algorithm"], "config_index": record.get("config_index", -1),
                "training_seed": record.get("seed", log.get("seed")), "learning_rate": config.get("learning_rate"),
                "action_transform": config.get("action_transform", "log1p"), "source": str(log_path)}
            runs.append({**common, "status": log.get("status"), "seconds": log.get("seconds"),
                "parameters": log.get("parameters"), "logged_steps": len(log.get("history", [])),
                "last_logged_step": max((x["step"] for x in log.get("history", [])), default=0)})
            for row in log.get("history", []):
                for component, loss in enumerate(row["losses"]):
                    losses.append({**common, "step": row["step"], "component": component, "loss": loss})
    # Final catalog can contain multiple references to the same checkpoint.
    unique = {record["checkpoint"]: record for record in candidates}
    candidates = list(unique.values())
    checkpoint_rows = []
    for r in candidates:
        checkpoint_rows.append({k: r.get(k) for k in ("algorithm", "config_index", "seed", "step", "training_seconds", "parameter_count", "checkpoint", "checkpoint_sha256")})
    validations = []
    for directory in root.iterdir():
        if not directory.is_dir() or "valid" not in directory.name.lower():
            continue
        csv_path, protocol_path = directory / "episode_results.csv", directory / "task_protocol.json"
        if not csv_path.exists() or not protocol_path.exists():
            continue
        frame = sources.csv(csv_path, dtype={"player_id": str})
        tasks = sources.json(protocol_path)["tasks"]
        expected = {}
        for task in tasks:
            expected.setdefault(task["algorithm"], []).append(task)
        failures = sources.json(directory / "failures.json") if (directory / "failures.json").exists() else []
        skipped = sources.json(directory / "skipped.json") if (directory / "skipped.json").exists() else []
        for name, task_group in expected.items():
            parts = name.split("|")
            if len(parts) != 3 or parts[0] not in NEURAL:
                continue
            algorithm, config_index, step = parts
            group = frame[frame.algorithm == name]
            expected_cases = {(t["day_id"], t["scenario"], str(t["target"]), int(t["seed"]), int(t.get("training_seed", 0))) for t in task_group}
            actual_cases = set(group[["day_id", "scenario", "player_id", "seed", "training_seed"]].itertuples(index=False, name=None))
            complete = expected_cases == actual_cases and len(group) == len(expected_cases)
            seed_means = group.groupby("training_seed").total_value.mean()
            validations.append({"stage": directory.name, "candidate": name, "algorithm": algorithm,
                "config_index": int(config_index), "step": int(step), "expected_episodes": len(task_group),
                "successful_episodes": len(group), "complete": complete,
                "failed_episodes": sum(r.get("algorithm", r.get("candidate")) == name for r in failures),
                "skipped_episodes": sum(r.get("algorithm", r.get("candidate")) == name for r in skipped),
                "mean_V": group.total_value.mean() if complete else np.nan,
                "training_seed_V_std": seed_means.std(ddof=1) if complete else np.nan,
                "source": str(csv_path)})
    receipt_rows = []
    receipt_path = root / "models" / "training_runs.json"
    if receipt_path.exists():
        for receipt in sources.json(receipt_path):
            identity = receipt["identity"]
            receipt_rows.append({"algorithm": identity["algorithm"], "config_index": identity["config_index"],
                "training_seed": identity["seed"], "training_seconds": receipt["training_seconds"],
                "call_wall_seconds": receipt["call_wall_seconds"], "parameters": receipt["parameter_count"],
                "source": str(receipt_path), "training_log": receipt["training_log"], "attempt": receipt["attempt"]})
    search_rows = []
    for path in sorted(root.glob("*/candidate_scores.json")):
        for record in sources.json(path):
            search_rows.append({**record, "stage": path.parent.name, "source": str(path)})
    return {"runs": pd.DataFrame(runs), "losses": pd.DataFrame(losses),
        "checkpoints": pd.DataFrame(checkpoint_rows), "validation": pd.DataFrame(validations),
        "run_receipts": pd.DataFrame(receipt_rows), "search_scores": pd.DataFrame(search_rows)}


def draw_figures(destination, tables, improvements, behavior, training):
    import os
    os.environ["MPLCONFIGDIR"] = str(destination.parent / "matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.right": False, "axes.spines.top": False, "savefig.facecolor": "white"})
    destination.mkdir(parents=True, exist_ok=True)
    figures = []
    def save(fig, name, description, data):
        for extension in ("png", "svg"):
            fig.savefig(destination / f"{name}.{extension}", dpi=170, bbox_inches="tight")
        plt.close(fig)
        figures.append({"name": name, "description": description, "png": str((destination / f"{name}.png").resolve()),
            "svg": str((destination / f"{name}.svg").resolve()), "derived_sources": data})
    def heatmap(ax, data, title, *, vmin=None, vmax=None, norm=None, cmap="YlGnBu", fmt=".2f"):
        array = data.to_numpy(dtype=float)
        kwargs = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
        im = ax.imshow(array, aspect="auto", cmap=cmap, **kwargs)
        ax.set_xticks(range(len(data.columns)), data.columns, rotation=20, ha="right")
        ax.set_yticks(range(len(data.index)), [LABEL.get(x, str(x)) for x in data.index])
        ax.set_title(title)
        for row in range(array.shape[0]):
            for col in range(array.shape[1]):
                value = array[row, col]
                text = format(value, fmt) if np.isfinite(value) else "undefined"
                rgb = np.asarray(im.cmap(im.norm(value))[:3]) if np.isfinite(value) else np.ones(3)
                linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                luminance = float(linear @ np.array([.2126, .7152, .0722]))
                color = "white" if luminance < .179 else "black"
                ax.text(col, row, text, ha="center", va="center", fontsize=8, color=color)
        return im
    def matrix(phase, metric):
        return tables[phase]["per_scenario"].pivot(index="algorithm", columns="scenario", values=metric).reindex(index=ALGORITHMS, columns=SCENARIOS)
    for phase, name, caption in [("experiment_A", "01_frozen_value", "A: frozen policies across five opponents"),
                                  ("experiment_B", "02_tuned_value", "B: jointly tuned policies across five opponents")]:
        fig, ax = plt.subplots(figsize=(10, 7))
        im = heatmap(ax, matrix(phase, "mean_V"), caption)
        fig.colorbar(im, ax=ax, label="Mean won pValue V")
        save(fig, name, caption, [f"{phase}_per_scenario.csv"])
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    vmax = max(matrix(p, "mean_R").max().max() for p in PHASES[:2])
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=max(1.01, vmax))
    for ax, phase, title in zip(axes, PHASES[:2], ["A / own LocalPID reference", "B / own LocalPID reference"]):
        im = heatmap(ax, matrix(phase, "mean_R"), title, norm=norm, cmap="RdYlBu")
    fig.colorbar(im, ax=axes, label="Within-experiment R; denominators differ")
    save(fig, "03_internal_relative_performance", "A/B internal R uses a separate LocalPID denominator", [f"{p}_per_scenario.csv" for p in PHASES[:2]])
    fig, ax = plt.subplots(figsize=(12, 6))
    ranked_algorithms = tables["experiment_B"]["overall"].sort_values("mean_V",ascending=False).algorithm.tolist()
    x = np.arange(len(ranked_algorithms))
    for offset, phase, label, color in [(-.2, "experiment_A", "A frozen", "#94a3b8"), (.2, "experiment_B", "B tuned", "#2563eb")]:
        values = tables[phase]["overall"].set_index("algorithm").reindex(ranked_algorithms).mean_V
        ax.bar(x + offset, values, width=.38, label=label, color=color)
    ax.set_xticks(x, [LABEL.get(a, a) for a in ranked_algorithms], rotation=25, ha="right")
    ax.set_ylabel("Mean won pValue V"); ax.legend(); ax.set_title("Five-scene mean V: ordered by B tuned ranking")
    save(fig, "04_overall_value_comparison", "Five-scene equal-role mean V", ["experiment_A_overall.csv", "experiment_B_overall.csv"])
    gain = improvements["per_scenario"].pivot(index="algorithm", columns="scenario", values="mean_paired_relative_gain").reindex(index=ALGORITHMS, columns=SCENARIOS) * 100
    bound = max(1., float(np.nanmax(np.abs(gain.to_numpy()))))
    fig, ax = plt.subplots(figsize=(10, 7))
    im = heatmap(ax, gain, "Paired B versus A value gain (%)", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), cmap="RdBu", fmt=".1f")
    fig.colorbar(im, ax=ax, label="Mean paired gain (%); zero-A cases undefined")
    save(fig, "05_paired_value_improvement", "Paired relative-change diagnostic; positive percentages need not imply higher mean V", ["paired_per_scenario.csv"])
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    for ax, phase, title in zip(axes, PHASES[:2], ["A frozen budget utilization (%)", "B tuned budget utilization (%)"]):
        im = heatmap(ax, matrix(phase, "mean_budget_utilization") * 100, title, vmin=0, vmax=100, fmt=".1f")
    fig.colorbar(im, ax=axes, label="Mean cost / initial budget (%)")
    save(fig, "06_budget_before_after", "Budget usage is a diagnostic, not the objective", [f"{p}_per_scenario.csv" for p in PHASES[:2]])
    for metric, name, ylabel in [("cumulative_spend_mean", "07_cumulative_budget", "Mean cumulative cost / initial budget"),
                                  ("cumulative_value_mean", "08_cumulative_value", "Mean cumulative won pValue")]:
        fig, axes = plt.subplots(2, 5, figsize=(24, 9), constrained_layout=True)
        handles = None
        for r, phase in enumerate(PHASES[:2]):
            for c, scene in enumerate(SCENARIOS):
                ax = axes[r, c]; group = behavior[phase]["population_slots"]
                for a in ALGORITHMS:
                    g = group[(group.algorithm == a) & (group.scenario == scene)]
                    ax.plot(g.slot_number, g[metric], label=LABEL.get(a, a), color=COLORS[a], linewidth=1.5)
                ax.set_title(f"{'A frozen' if r == 0 else 'B tuned'} / {scene}")
                ax.set_xlim(1, 72); ax.set_xlabel("Slot number (1–72)"); ax.grid(alpha=.15)
                if c == 0: ax.set_ylabel(ylabel)
                if metric == "cumulative_spend_mean": ax.axhline(.95, ls=":", color="#64748b", alpha=.5)
                handles = ax.get_legend_handles_labels()
        fig.legend(*handles, loc="outside lower center", ncol=10, fontsize=9)
        save(fig, name, ylabel + "; all complete cases", [f"{p}_behavior_population_slots.csv" for p in PHASES[:2]])
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    type_max = max(tables[p]["per_type"].groupby(["algorithm", "player_type"]).mean_R.mean().max() for p in PHASES[:2])
    type_norm = TwoSlopeNorm(vmin=0,vcenter=1,vmax=max(1.01,type_max))
    for ax, phase, title in zip(axes, PHASES[:2], ["A frozen by player type", "B tuned by player type"]):
        data = tables[phase]["per_type"].groupby(["algorithm", "player_type"]).mean_R.mean().unstack().reindex(ALGORITHMS)
        im = heatmap(ax, data, title, norm=type_norm, cmap="RdYlBu")
    fig.colorbar(im, ax=axes, label="Within-phase R; five scenes equally averaged")
    save(fig, "09_player_types", "Type labels are source category codes", [f"{p}_per_type.csv" for p in PHASES[:2]])
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, phase, title in zip(axes, PHASES[:2], ["A frozen", "B tuned"]):
        data = tables[phase]["overall"].set_index("algorithm").reindex(ALGORITHMS)
        for i, a in enumerate(ALGORITHMS):
            r = data.loc[a]
            ax.plot([r.worst_scenario_V, r.best_scenario_V], [i, i], color=COLORS[a], linewidth=3)
            ax.scatter(r.mean_V, i, color=COLORS[a], s=50, zorder=3)
        ax.set_yticks(range(len(ALGORITHMS)), [LABEL.get(a, a) for a in ALGORITHMS]); ax.invert_yaxis()
        ax.set_xlabel("Mean V (dot); min–max of five scenario means (line)"); ax.set_title(title); ax.grid(axis="x", alpha=.15)
    save(fig, "10_cross_scenario_robustness", "High worst-scene V and small range are different criteria", [f"{p}_overall.csv" for p in PHASES[:2]])
    if training["losses"].empty or training["validation"].empty:
        raise ValueError("Actual training and validation evidence required for learning-curve figure")
    fig, axes = plt.subplots(2, 6, figsize=(24, 10), constrained_layout=True)
    for col, a in enumerate(NEURAL):
        loss = training["losses"].query("algorithm == @a and component == 0")
        for (cfg, seed), g in loss.groupby(["config_index", "training_seed"]):
            axes[0, col].plot(g.step, g.loss, marker=".", markersize=2, linewidth=1.2, label=f"cfg {cfg} / {seed}")
        axes[0, col].set_yscale("symlog", linthresh=1e-5); axes[0, col].set_title(a + " recorded loss[0]")
        axes[0, col].set_xlabel("Gradient step"); axes[0, col].legend(fontsize=6)
        val = training["validation"].query("algorithm == @a and complete")
        for cfg, g in val.groupby("config_index"):
            g = g.sort_values("step")
            axes[1, col].errorbar(g.step, g.mean_V, yerr=g.training_seed_V_std, marker="o", capsize=3, label=f"cfg {cfg}")
        axes[1, col].set_xlabel("Saved checkpoint step"); axes[1, col].set_title(a + " validation V"); axes[1, col].legend(fontsize=7)
        for ax in axes[:, col]: ax.grid(alpha=.15)
    axes[0, 0].set_ylabel("Loss definitions differ; no cross-algorithm loss ranking")
    axes[1, 0].set_ylabel("Validation mean V ± two-training-seed SD")
    save(fig, "11_actual_learning_curves", "Top: actual logged loss. Bottom: complete validation groups only", ["training_losses.csv", "training_validation.csv"])
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    for ax, metric, title in zip(axes, ["mean_V", "mean_R"], ["Independent period: V", "Independent period: own LocalPID R"]):
        im = heatmap(ax, matrix("independent_test", metric), title)
        fig.colorbar(im, ax=ax)
    save(fig, "12_independent_period", "Separate independent-period evidence; not merged into period-24 paired gains", ["independent_test_per_scenario.csv"])
    return figures


PRINCIPLES = {
    "FixedAlpha": ("以共享固定α将自身pValue换算成报价，不使用局内预算反馈。", "五场景验证重新搜索共享α；无神经网络训练。"),
    "LocalPID": ("用自身费用进度误差及P/I/D项调整下一片α，是本项目本地参考策略。", "保留控制器，重新搜索初值与实际支持的控制增益。"),
    "AuctionNetPID": ("复用AuctionNet源码的预算阈值与乘法调整启发式；名称中的PID不表示标准P/I/D公式。", "适配72片和自身反馈、完整reset与末片/余额边界；A按旧三场景口径选参，B按五场景V选参。"),
    "OnlineLP": ("依据经验累计成本响应表查询预算下的出价阈值，在线核复用上游查表实现。", "B重新拟合五场景成本表并选倍率；保留查表核心，不能将联合提升单独归因于表或倍率。"),
    "BC": ("回归离线行为策略的动作坐标，不使用critic或奖励反传。", "共享18维输入及原128宽actor；用五场景轨迹从头训练，验证选择学习率与保存步数。"),
    "IQL": ("expectile状态价值、双Q及优势加权动作似然；温度公式按保留源码为exp(Q−V)×temperature。", "保留原更新顺序和结构，换为覆盖五场景的新训练数据并联合选模。"),
    "BCQ": ("条件VAE生成动作，受限扰动actor修正，再按分布Q筛选；正式推理每片100个候选。", "继续使用紧凑网络和已验证的two-hot修复；新数据与较长训练共同变化，无单因素归因。"),
    "TD3BC": ("带行为回归项的延迟actor更新、双Q和目标动作噪声；本版本Q使用symlog桶分类。", "保持旧紧凑结构及Q1标量解码修复，重新训练/选择，未修改核心学习目标。"),
    "CQL": ("保留AuctionNet版本的熵策略、expectile Bellman误差及特定logsumexp保守项。", "本轮线性动作尺度候选与学习率以实际配置为准；数值稳定不等同预算分配有效。"),
    "DT": ("用目标剩余回报与最近10片return/state/action token做动作回归；不直接看到未来市场。", "保持64维三层因果Transformer和合法18维历史，使用五场景序列及实际保存步点验证。"),
}


def link(path, label=None):
    path = Path(path).resolve()
    return f"[{label or path.name}]({path.as_posix()})"


def markdown_table(headers, rows):
    def cell(value):
        return str(value).replace("|", r"\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |"] +
        ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def number(value, digits=4, percent=False):
    if value is None or not np.isfinite(float(value)):
        return "未定义"
    return f"{float(value):.{digits}%}" if percent else f"{float(value):.{digits}f}"


def selection_records(root, sources):
    result = {}
    frozen = {}
    for phase in PHASES:
        path = root / phase / "selected_frozen.json"
        value = sources.json(path)
        rows = value["selected"]
        expected = {(a, seed) for a in ALGORITHMS for seed in (TRAIN_SEEDS if a in NEURAL else (0,))}
        actual = {(r["algorithm"], int(r.get("training_seed", 0))) for r in rows}
        if len(rows) != len(expected) or actual != expected:
            raise ValueError("Frozen selection coverage differs: " + phase)
        for asset, expected_sha in value.get("assets", {}).items():
            if digest(asset) != expected_sha:
                raise ValueError("Frozen selected asset changed: " + asset)
        frozen[phase] = value
        if phase != "independent_test":
            result[str(path.relative_to(root))] = rows
    if frozen["independent_test"] != frozen["experiment_B"]:
        raise ValueError("Independent test must use exactly frozen B selection")
    return result


def historical_mechanisms(sources):
    """Inline audited mechanisms, excluding historical final-score sections.

    Historical search values and the DT worked example stay explicitly labeled
    as historical evidence; current choices are emitted separately below them.
    """
    text = sources.text(OLD_ROOT / "report_auctionnet_baselines_detailed.md")
    algorithms = ("FixedAlpha", "LocalPID", "OnlineLP", "BC", "IQL", "BCQ", "TD3BC", "CQL", "DT")
    result = {}
    for index, algorithm in enumerate(algorithms, 1):
        match = re.search(rf"^### 5\.{index} .+?\n(.*?)(?=^### 5\.{index+1} |^## 6\.)", text, re.M | re.S)
        if not match:
            raise ValueError("Missing audited historical mechanism: " + algorithm)
        body = re.split(r"^#### (?:\d+\. )?最终效果", match.group(1), maxsplit=1, flags=re.M)[0].strip()
        body = re.sub(r'<a id="[^"]+"></a>\s*', '', body)
        result[algorithm] = body
    pid = sources.text(ROOT / "harmony_baselines/five_scenarios/PID_ADAPTATION.md")
    pid = pid[pid.index("## 1."):pid.index("## 4.")].strip()
    result["AuctionNetPID"] = re.sub(r"^## ", "#### ", pid, flags=re.M)
    return result


def experiment_context(root, sources):
    names = ["protocol.json", "opponents_frozen.json", "opponents_provenance.json",
        "trajectory_provenance.json", "models/training_metadata.json", "models/training_status.json",
        "online_lp_provenance.json"]
    context = {name: sources.json(root / name) for name in names}
    status = context["models/training_status.json"]
    if status.get("status") != "completed" or status.get("completed_runs") != 24 or status.get("checkpoints") != 48:
        raise ValueError("Completed 24-run/48-checkpoint training evidence required")
    metadata=context["models/training_metadata.json"]
    trajectory=context["trajectory_provenance.json"]
    data_path=sources.record(root / "train_transitions.npz")
    if (metadata.get("source_sha256") != sources.hashes[str(data_path)] or trajectory.get("sha256") != sources.hashes[str(data_path)]
            or metadata.get("transitions") != 34560 or trajectory.get("transitions") != 34560
            or metadata.get("episodes") != 480 or trajectory.get("episodes") != 480):
        raise ValueError("Training data, metadata and provenance differ")
    old_text=sources.text(OLD_ROOT / "report_auctionnet_baselines_detailed.md")
    section=old_text.split("### 4.2 合法的 18 维观察、动作与奖励",1)[1].split("实现对 B=0",1)[0].strip()
    context["shared_feature_definition"]=section
    context["resource_evidence"]=resource_evidence(root,sources)
    for name in ("runtime_environment.json","runtime_concurrency_observation.json"):
        if (root / name).exists():
            context[name]=sources.json(root / name)
    context["qa_evidence"]=qa_evidence(root,sources)
    context["interruption_evidence"]=interruption_evidence(root,sources)
    context["recovery_prefix_evidence"]=recovery_prefix_evidence(root,sources)
    context["actual_input_audits"]={}
    for name in ("actual_training_data_audit.json","actual_lp_table_audit.json","actual_neural_training_audit.json",
                 "independent_test_protocol_audit.json"):
        path=root / "qa" / name
        if path.exists():
            audit=sources.json(path)
            if audit.get("status") != "passed":
                raise ValueError("Independent input/protocol audit did not pass: "+name)
            context["actual_input_audits"][name]=audit
    context["actual_input_audits"]["independent_test_completed_audit.json"]=completed_test_evidence(root,sources)
    return context


def completed_test_evidence(root,sources):
    """Require the completed full-result audit; protocol/prefix audits are narrower."""
    audit=sources.json(root / "qa/independent_test_completed_audit.json")
    if audit.get("status")!="passed" or audit.get("valid") is not True:
        raise ValueError("Completed independent-test audit is not passed and valid")
    expected={"episodes":7680,"original_successes":7534,"recovered_cases":146,
        "historical_timeouts":5,"historical_skips":141,"final_unresolved":0,
        "policies":16,"algorithms":10,"scenarios":5,"roles":48,"evaluation_seeds":2,
        "slots":552960,"diagnostic_records":1658880}
    if any(audit.get("counts",{}).get(k)!=v for k,v in expected.items()):
        raise ValueError("Completed independent-test audit coverage differs")
    script=sources.record(root / "qa/audit_independent_test_completed.py")
    if sources.hashes[str(script)]!=audit["audit_script_sha256"]:
        raise ValueError("Completed independent-test audit script changed")
    for name in ("episode_results.status.json","recovery_manifest.json","episode_results.csv","slots.csv"):
        source=sources.record(root / "independent_test" / name)
        if audit.get("input_sha256",{}).get(str(source))!=sources.hashes[str(source)]:
            raise ValueError("Completed independent-test audit describes different final input: "+name)
    return audit


def qa_evidence(root,sources):
    """QA evidence is cited separately and never enters experiment totals."""
    result=[]
    paths=[root / "qa/root_integrated_final_1.log"]
    paths.extend(sorted((root / "qa").glob("report_focused_*.log")))
    for path in paths:
        if not path.exists():
            continue
        log=sources.text(path)
        match=re.search(r"(?m)^(\d+) passed(?:, [^\n]*?)? in ([\d.]+)s",log)
        if not match:
            raise ValueError("QA log has no successful pytest summary: "+str(path))
        result.append({"source":str(path.resolve()),"passed_tests":int(match.group(1)),"qa_elapsed_seconds":float(match.group(2)),
            "kind":"integrated" if path.name=="root_integrated_final_1.log" else "focused_reporting",
            "included_in_rankings":False,"included_in_experiment_runtime":False})
    return result


def interruption_evidence(root,sources):
    directory=root / "qa/independent_interruption_20261028"
    diagnosis_path=directory / "incident_diagnosis.json"
    if not diagnosis_path.exists():
        return {}
    diagnosis=sources.json(diagnosis_path)
    events=sources.json(directory / "windows_power_events.json")
    system=diagnosis["system_evidence"]
    for side in ("enter","exit"):
        if not any(event["id"]==system[side+"_event_id"] and event["time_utc"]==system[side+"_utc"]
                   and event["provider"]==system["provider"] for event in events):
            raise ValueError("Standby diagnosis does not match preserved power event")
    delta=(datetime.fromisoformat(system["exit_utc"].replace("Z","+00:00"))-
           datetime.fromisoformat(system["enter_utc"].replace("Z","+00:00"))).total_seconds()
    if abs(delta-system["standby_interval_seconds"])>1e-6:
        raise ValueError("Standby duration does not match event timestamps")
    for finding in diagnosis["findings"]:
        path=directory / f"task_{finding['task_index']:05d}.jsonl"
        if digest(sources.record(path))!=finding["diagnostic_sha256"]:
            raise ValueError("Preserved timeout diagnostic changed")
    for name in ("original_failures.json","original_skipped.json","pipeline_log_prefix.txt"):
        sources.record(directory / name)
    user_path=root / "qa/timeout_user_report.json"
    user=sources.json(user_path) if user_path.exists() else None
    return {"diagnosis":diagnosis,"diagnosis_source":str(diagnosis_path.resolve()),
        "events_source":str((directory / "windows_power_events.json").resolve()),
        "user_report":user,"user_report_source":str(user_path.resolve()) if user else None,
        "power_event_timestamp_difference_seconds":delta}


def recovery_prefix_evidence(root,sources):
    """A completed prefix audit is not proof that every recovery task finished."""
    path=root / "qa/independent_recovery_prefix_audit.json"
    if not path.exists():
        return {}
    audit=sources.json(path)
    if (audit.get("status")!="passed" or audit.get("recovery_executed_by_audit") is not False
            or audit.get("maximum_numeric_difference")!=0 or audit.get("timing_fields_compared") is not False):
        raise ValueError("Recovery prefix audit did not confirm exact business-only equality")
    source=sources.record(audit["audit_source_path"])
    plan=sources.record(root / "independent_test_recovery_001/recovery_plan.json")
    if sources.hashes[str(source)]!=audit["audit_source_sha256"] or sources.hashes[str(plan)]!=audit["recovery_plan_sha256"]:
        raise ValueError("Recovery prefix audit source or frozen plan changed")
    return {"audit":audit,"source":str(path.resolve())}


def recovery_prefix_paragraph(context):
    evidence=context.get("recovery_prefix_evidence",{})
    if not evidence:
        return ""
    audit=evidence["audit"]
    return (f"另有局部恢复前缀审计：{audit['original_failed_cases']}个原超时局的"
        f"{audit['original_prefix_records_compared']}条原始记录，包含{audit['original_completed_feedback_slots_compared']}个已完成反馈片和"
        f"{audit['original_action_records_compared']}条动作；{audit['numeric_comparisons']}个业务数值与对应恢复记录逐值完全一致，最大绝对误差为0。"
        "比较范围为已记录的alpha、cost、value、wins及余额；原末片缺失的feedback没有被补造，计时字段没有作为相等比较对象。"
        "这只支持已记录业务前缀在恢复中重现，不能单独证明全部146局恢复完成，也不将用户反馈的断网认定为已证实的故障原因。来源："+
        link(evidence["source"])+"。")


def resource_evidence(root, sources):
    """Keep recommendation and later application as separate immutable facts."""
    evidence={}
    for key,name in (("selection","resource_selection.json"),("application","resource_application.json"),
                     ("original_recommendation","resource_benchmark/operational_adjustment.json")):
        if (root / name).exists():
            evidence[key]=sources.json(root / name)
            evidence[key+"_source"]=str((root / name).resolve())
    selection=evidence.get("selection",{})
    application=evidence.get("application",{})
    if application:
        if not selection or application.get("selection_record_sha256") != digest(root / "resource_selection.json"):
            raise ValueError("Resource application is not linked to the recorded benchmark")
    if selection:
        if (selection.get("test_used") is not False or selection.get("validation_used") is not False
                or selection.get("included_in_training_data") is not False
                or selection.get("included_in_performance_rankings") is not False):
            raise ValueError("Resource benchmark scope differs from declared auxiliary-only use")
        for group in selection["groups"]:
            if "timing_source" in group:
                timing_path=sources.record(group["timing_source"])
                if sources.hashes[str(timing_path)] != group["timing_sha256"]:
                    raise ValueError("Resource benchmark timing evidence changed")
        evidence["benchmark_episodes"]=sum(g["episodes"] for g in selection["groups"])
        evidence["benchmark_recorded_seconds"]=sum(g["seconds"] for g in selection["groups"])
    evidence["application_status"]=application.get("status","not_recorded")
    evidence["applied_workers"]=application.get("actual_following_workers") if application.get("status")=="applied" else None
    return evidence


def is_live_execution_path(path,root):
    """Archived copies and synthetic fixtures are evidence, not extra runs."""
    return not any(x.lower() in {"qa","tests","reporting"} or x.lower().startswith(("report_","test_","pytest","synthetic"))
        or any(token in x.lower() for token in ("smoke","snapshot","archive"))
        for x in path.relative_to(root).parts[:-1])


def failure_history(root,sources):
    columns=["stage","task_index","algorithm","status","exception_type","reason","elapsed_seconds",
        "day_id","scenario","player_id","seed","training_seed","task_key","diagnostic_path","source"]
    histories={"failures":[],"skipped":[]}
    for kind in histories:
        for path in sorted(root.glob(f"**/{kind}.json")):
            if not is_live_execution_path(path,root) or not (path.parent / "task_protocol.json").exists():
                continue
            rows=sources.json(path)
            if len({r["task_index"] for r in rows})!=len(rows):
                raise ValueError("Duplicate task event in failure history: "+str(path))
            for record in rows:
                task=record.get("task",{})
                row={k:record.get(k,task.get(k)) for k in columns}
                row.update(stage=path.parent.relative_to(root).as_posix(),source=str(path.resolve()))
                histories[kind].append(row)
    failures=pd.DataFrame(histories["failures"],columns=columns)
    skipped=pd.DataFrame(histories["skipped"],columns=columns)
    summaries=[]
    for stage in sorted(set(failures.stage)|set(skipped.stage)):
        f=failures[failures.stage==stage]
        s=skipped[skipped.stage==stage]
        summaries.append({"stage":stage,"historical_failure_events":len(f),"historical_skip_events":len(s),
            "resource_timeouts":int(((f.status=="resource_incomplete")&(f.exception_type=="Timeout")).sum()),
            "recorded_min_failure_elapsed_seconds":pd.to_numeric(f.elapsed_seconds,errors="coerce").min(),
            "recorded_max_failure_elapsed_seconds":pd.to_numeric(f.elapsed_seconds,errors="coerce").max()})
    return {"failures":failures,"skipped":skipped,"by_stage":pd.DataFrame(summaries)}


def execution_evidence(root, sources, *, include_segments=False):
    """Read every real execution segment, excluding test/QA fixtures.

    A latest status worker count is not retroactively assigned to older segments:
    the original runtime-segment schema has no worker field.
    """
    statuses=[p for p in root.glob("**/episode_results.status.json") if is_live_execution_path(p,root)]
    segments=[p for p in root.glob("**/runtime_segment_*.json") if is_live_execution_path(p,root)]
    directories=sorted({p.parent for p in statuses+segments})
    rows,segment_rows=[],[]
    for directory in directories:
        path=directory / "episode_results.status.json"
        value=sources.json(path) if path.exists() else {"status":"not_recorded"}
        stage=directory.relative_to(root).as_posix()
        stage_segments=[]
        for segment_path in sorted(directory.glob("runtime_segment_*.json")):
            segment=sources.json(segment_path)
            if not math.isfinite(segment["seconds"]) or segment["seconds"]<0:
                raise ValueError("Invalid runtime duration: "+str(segment_path))
            stage_segments.append(segment)
            segment_rows.append({"stage":stage,"segment":segment_path.name,**segment,
                "workers_recorded_in_segment":segment.get("workers"),"source":str(segment_path.resolve())})
        scope=("final_rollup" if directory.name in PHASES else "resource_benchmark" if "resource_benchmark" in directory.parts
            else "opponent_calibration" if directory.name.startswith("opponent_") else "model_validation" if directory.name.startswith("validation_")
            else "training_rollouts" if directory.name in {"trajectory_extension","lp_training"} else "performance_execution")
        rows.append({"stage":stage,"scope":scope,"aggregate_final_table":directory.name in PHASES,
            **{key:value.get(key) for key in ("status","valid","completed_episodes","expected_episodes","new_executed_episodes",
                "reused_episodes","resumed_completed","failed_episodes","skipped_episodes","workers","segment_seconds")},
            "saved_runtime_segments":len(stage_segments),
            "recorded_segment_seconds":sum(s["seconds"] for s in stage_segments) if stage_segments else np.nan,
            "recorded_segment_new_completed":sum(s["new_completed"] for s in stage_segments) if stage_segments else np.nan,
            "source":str(path.resolve()) if path.exists() else None})
    stages=pd.DataFrame(rows)
    return (stages,pd.DataFrame(segment_rows)) if include_segments else stages


def selected_summary(algorithm, selections, sources):
    """Readable actual freeze choices, while preserving full specs separately."""
    rows = []
    for source, records in selections.items():
        chosen = [r for r in records if r["algorithm"] == algorithm]
        if algorithm in NEURAL:
            common = chosen[0]
            checkpoint = Path(common["spec"]["kwargs"]["checkpoint"])
            log = sources.json(checkpoint.parent / "training.json")
            config = log["config"]
            text = (f"{common.get('validation_choice', '未记录')}；lr={config['learning_rate']:g}；"
                f"action={config.get('action_transform','log1p')}；训练seed=" + "/".join(str(r["training_seed"]) for r in chosen))
        else:
            kwargs = chosen[0]["spec"]["kwargs"]
            display = {k: v for k,v in kwargs.items() if k not in {"player_alphas", "table_path"} and v is not None}
            text = ", ".join(f"{k}={v:.6g}" if isinstance(v, (float,int)) else f"{k}={v}" for k,v in display.items())
            if "table_path" in kwargs:
                text += "；成本表=" + link(kwargs["table_path"])
        rows.append(["A" if "experiment_A" in source else "B", text])
    return rows


def type_value_table(tables,phase):
    """Equal five-scene mean V; type membership counts actual unique roles."""
    per_type=tables[phase]["per_type"]
    values=per_type.groupby(["algorithm","player_type"]).mean_V.mean().unstack("player_type").reindex(ALGORITHMS)
    roles=tables[phase]["per_role"][["player_id","player_type"]].drop_duplicates()
    if roles.player_id.duplicated().any():
        raise ValueError("A role has multiple type codes in one experiment")
    counts=roles.groupby("player_type").player_id.nunique()
    codes=sorted(values.columns,key=lambda x:(0,int(x)) if str(x).lstrip("-").isdigit() else (1,str(x)))
    values=values.reindex(columns=codes)
    table=markdown_table(["算法"]+[f"代码 {code}（{int(counts.loc[code])}角色）" for code in codes],
        [[LABEL.get(a,a)]+[number(values.loc[a,code]) for code in codes] for a in ALGORITHMS])
    return table


def resource_paragraph(context):
    evidence=context.get("resource_evidence",{})
    selection=evidence.get("selection",{})
    application=evidence.get("application",{})
    if not selection:
        return "未记录独立资源基准；实际worker以各阶段最后调用状态为准。"
    group_text="；".join(f"{g['workers']} workers：{g['episodes']}局/{number(g['seconds'],3)}秒" for g in selection["groups"])
    application_text=(f"后续正式阶段已采用{application['actual_following_workers']} workers，依据独立的resource_application.json。"
        if application.get("status")=="applied" else "资源推荐尚无已应用记录，不能将recommended当作实际执行设置。")
    return (f"资源基准仅在period-7重复同48个LocalPID目标/Fixed对手局，总计{evidence['benchmark_episodes']}个辅助局：{group_text}。"
        f"两组{selection['paired_episode_equality']['matched_episodes']}个配对局的业务与身份字段完全一致。{application_text}"
        "原protocol的4 workers和operational_adjustment中的recommended_not_applied保留为阶段性历史记录，"
        "后者不否定后续application的实际采用。4→8只测了一次且顺序固定，启动、文件缓存和系统负载没有被独立控制；"
        "不能声称普遍或稳定的并行加速比例。辅助局不进入A/B排名、训练数据或模型选择。")


def concurrency_paragraph(root,context):
    observed=context.get("runtime_concurrency_observation.json")
    if not observed:
        return "没有额外加载主机并发快照；计时仍依赖运行时资源条件。"
    other=[g for g in observed["process_groups"] if g["pid"]!=observed["this_pipeline_pid"]]
    jobs="、".join(f"{g['module']}（{g['direct_children']}个直接子进程）" for g in other)
    return (f"独立周期后半程存在同机CPU竞争的实际观察：{observed['observed_at_utc']}的1秒采样CPU为"
        f"{number(observed['system_cpu_percent_one_second'],1)}%，可用内存{number(observed['available_memory_GiB'],2)} GiB。"
        f"除本实验pipeline外，同机还有{jobs}。这是一份只读单次观察，不能外推整阶段CPU始终为100%，也不是受控benchmark。"
        "独立周期的阶段墙钟和目标推理延时受到同期主机负载影响，不能据此做纯算法速度对比。"
        "其他任务的案例和运行时间不计入本实验；本次观察未停止或调整任何进程。来源："+
        link(root / "runtime_concurrency_observation.json")+"。")


def failure_history_paragraph(root,context):
    history=context.get("failure_history")
    if history is None:
        return "执行事件账目未在本次文本组装中加载。"
    summaries=history["by_stage"]
    if not summaries:
        return "已读取的源执行事件文件未记录失败或跳过。"
    parts=["**最终成功覆盖与运行历史是两套账目：完整排名不等于零历史失败。**"]
    for row in summaries:
        parts.append(f"{row['stage']}保留{row['historical_failure_events']}次历史失败、{row['historical_skip_events']}次未运行/连带跳过；"
            f"其中资源超时为{row['resource_timeouts']}次。失败事件记录的elapsed范围为"
            f"{number(row['recorded_min_failure_elapsed_seconds'],3)}–{number(row['recorded_max_failure_elapsed_seconds'],3)}秒。")
    parts.append("原始failures/skipped和诊断保留；后续成功记录通过独立恢复来源补足缺失身份，不覆盖或改写原失败。"
        "正式结果只取成功完成全部72片的完整case，失败不按0分入榜。源incomplete阶段与恢复阶段的真实运行段分别计时，"
        "封存快照不视为新执行、不重复计数。事件索引见 "+link(root / "reporting/execution_failures.csv")+"、"+
        link(root / "reporting/execution_skipped.csv")+"。")
    return "\n\n".join(parts)


def interruption_paragraph(context):
    evidence=context.get("interruption_evidence",{})
    if not evidence:
        return "本次组装未加载额外的中断原因证据；不能从超时类型推断模型异常。"
    diagnosis=evidence["diagnosis"]
    system=diagnosis["system_evidence"]
    user=evidence.get("user_report")
    statement=("用户直接反馈“"+user["user_message"]+"”（"+link(evidence["user_report_source"])+"）。"
        if user else "用户报告断网，与系统事件证据分开记录。")
    return (statement+f"系统Kernel-Power {system['enter_event_id']}/{system['exit_event_id']}事件分别记录"
        f"{system['enter_utc']}进入Modern Standby、{system['exit_utc']}退出，间隔{number(system['standby_interval_seconds'],3)}秒。"
        "该区间与长进度间隔及资源超时记录重叠，是系统待机的直接证据；网络中断本身仍是用户反馈，二者不能混称。"
        "这些证据不证明每一局超时的唯一原因，也不提前证明各缺失局在正常活跃运行时一定小于180秒；后者必须看实际补跑记录。"
        "不能把这次墙钟超时当作策略数值异常。已有成功跨待机窗口的记录及其计时保留，不能为美化速度而剔除。来源："+
        link(evidence["diagnosis_source"])+"、"+link(evidence["events_source"])+"。")


def verify_recovery_manifest(root,frame,status,sources):
    """Verify final success lineage without erasing the original event history."""
    path=root / "independent_test/recovery_manifest.json"
    if not path.exists():
        raise ValueError("Recovered final results require a recovery manifest")
    manifest=sources.json(path)
    if status.get("recovery_manifest_sha256")!=sources.hashes[str(path.resolve())]:
        raise ValueError("Final recovery manifest hash differs")
    plan=sources.json(manifest["plan_path"])
    receipt=sources.json(manifest["receipt_path"])
    for name in ("plan","receipt"):
        if sources.hashes[str(Path(manifest[name+"_path"]).resolve())]!=manifest[name+"_sha256"]:
            raise ValueError("Recovery "+name+" hash differs")
    count=manifest["recovery_completed_episodes"]
    if (receipt.get("status")!="completed" or receipt.get("completed_episodes")!=count
            or receipt.get("expected_episodes")!=count or receipt.get("failed_episodes")!=0 or receipt.get("skipped_episodes")!=0
            or receipt.get("plan_sha256")!=manifest["plan_sha256"] or manifest.get("unresolved_episodes")!=0):
        raise ValueError("Recovery is not completely successful")
    if plan["mapping"]!=manifest["mapping"] or len(plan["mapping"])!=count:
        raise ValueError("Recovery mapping differs from frozen plan")
    for relative,key in ((manifest["original_stage"]+"/task_protocol.json","original_task_protocol_sha256"),
                         ("experiment_B/selected_frozen.json","B_freeze_sha256")):
        source=sources.record(root / relative)
        if sources.hashes[str(source)]!=plan[key]:
            raise ValueError("Recovery original protocol or B freeze changed")
    for key in ("historical_failed_episodes","historical_skipped_episodes","original_completed_episodes","recovery_completed_episodes"):
        if status.get(key)!=manifest[key]:
            raise ValueError("Final status hides or changes recovery/history counts")
    if (manifest["original_completed_episodes"]+count!=len(frame)
            or plan["historical_failed_episodes"]!=manifest["historical_failed_episodes"]
            or plan["historical_skipped_episodes"]!=manifest["historical_skipped_episodes"]
            or plan["expected_recovery_episodes"]!=count or plan["timeout_seconds"]!=180. or plan["workers"]!=2):
        raise ValueError("Recovery count or unchanged resource protocol differs")
    for source,expected in plan.get("evidence_sha256",{}).items():
        if digest(sources.record(source))!=expected:
            raise ValueError("Sealed recovery evidence differs")
    original_failure=sources.json(manifest["original_failure_path"])
    original_skipped=sources.json(manifest["original_skipped_path"])
    failed_ids={r["task_index"] for r in original_failure}
    skipped_ids={r["task_index"] for r in original_skipped}
    if (len(original_failure)!=manifest["historical_failed_episodes"] or len(original_skipped)!=manifest["historical_skipped_episodes"]
            or failed_ids & skipped_ids):
        raise ValueError("Original failed/skipped history was changed")
    mapping={r["original_task_index"]:r for r in manifest["mapping"]}
    if len(mapping)!=count or set(mapping)!=failed_ids|skipped_ids:
        raise ValueError("Recovery does not cover exactly original terminal missing cases")
    terminal={r["task_index"]:{"kind":kind,"record":r} for kind,records in (("failure",original_failure),("skipped",original_skipped)) for r in records}
    if any(item.get("terminal")!=terminal[old] for old,item in mapping.items()):
        raise ValueError("Recovery mapping changed its original terminal failure evidence")
    recovered=frame[frame.provenance=="resource_recovery"]
    if len(recovered)!=count or set(recovered.task_index)!=set(mapping):
        raise ValueError("Recovered result coverage differs from the original missing set")
    directory=root / manifest["recovery_stage"]
    for row in recovered.itertuples():
        item=mapping[row.task_index]
        local=item["recovery_task_index"]
        if (int(row.original_task_index)!=row.task_index or int(row.recovery_task_index)!=local
                or row.recovery_stage!=manifest["recovery_stage"] or row.task_key!=item["task_key"]
                or row.source_task!=f"{manifest['recovery_stage']}:{local}"
                or Path(row.source_result_path).resolve()!=(directory / "results" / f"task_{local:05d}.json").resolve()):
            raise ValueError("Recovered row source identity is not its planned original case")
    for name in ("episode_results.csv","slots.csv"):
        final_key="result_sha256" if name=="episode_results.csv" else "slots_sha256"
        if status[final_key]!=manifest["final_export_sha256"][name]:
            raise ValueError("Final merged export differs from recovery manifest")
        source=directory / name
        if digest(sources.record(source))!=receipt[final_key]:
            raise ValueError("Recovery source export changed")
    for source,expected in manifest.get("original_snapshot",{}).get("sealed_sha256",{}).items():
        if digest(sources.record(source))!=expected:
            raise ValueError("Original incomplete snapshot was changed")
    for segment in receipt.get("runtime_segments",[]):
        if digest(sources.record(segment["path"]))!=segment["sha256"]:
            raise ValueError("Recovery runtime segment changed")
    return {"manifest":manifest,"receipt":receipt,"plan_source":manifest["plan_path"],
        "plan_summary":{k:v for k,v in plan.items() if k not in {"tasks","identities"}},
        "manifest_source":str(path.resolve()),"verified_recovered_rows":len(recovered),
        "original_failed_events_preserved":len(original_failure),"original_skipped_events_preserved":len(original_skipped)}


def recovery_paragraph(root,context):
    evidence=context.get("recovery_evidence",{})
    if not evidence:
        return "本次文本未加载成功恢复的最终合并凭据；不能宣称原缺失局已补齐。" if context.get("failure_history",{}).get("by_stage") else ""
    manifest=evidence["manifest"]
    plan=evidence["plan_summary"]
    completed=context.get("actual_input_audits",{}).get("independent_test_completed_audit.json",{})
    segments=completed.get("runtime_segments",[])
    timing=""
    if segments:
        original=sum(s["seconds"] for s in segments if s["category"]=="original_runtime_segments")
        recovered=sum(s["seconds"] for s in segments if s["category"]=="recovery_runtime_segments")
        timing=(f"原执行段{number(original,4)}秒加恢复执行段{number(recovered,4)}秒，合计{number(original+recovered,4)}秒。"
            "原阶段与恢复阶段部分并行，该数值是运行段累计，不是独立测试端到端墙钟；包含待机时间，没有扣除或估算活跃执行时间。")
    return (f"恢复阶段{manifest['recovery_stage']}使用{plan['workers']} workers和原{number(plan['timeout_seconds'],0)}秒监督超时阈值，"
        f"只补原失败/跳过的{manifest['recovery_completed_episodes']}个身份；冻结模型、参数、市场与seed保持原计划。"
        f"最终完整集由{manifest['original_completed_episodes']}个原成功局和{manifest['recovery_completed_episodes']}个恢复成功局组成，"
        f"当前未解决数{manifest['unresolved_episodes']}；历史仍有{manifest['historical_failed_episodes']}次失败和"
        f"{manifest['historical_skipped_episodes']}次连带跳过。"
        "最终行task_index回到原始编号，source_task保留恢复阶段局部编号，source_result_path指向恢复结果，provenance=resource_recovery；"
        "恢复映射与原始incomplete表的封存SHA可核验，未拼接不同模型版本。来源："+
        link(evidence["manifest_source"])+"、"+link(evidence["plan_source"])+"、"+link(manifest["receipt_path"])+"。"+timing)


def gain_interpretation(improvements):
    """Warn explicitly when relative-case averages reverse the main V result."""
    overall=improvements["overall"]
    records=overall[overall.relative_vs_mean_V_sign_reversal].to_dict("records")
    text=["平均V（主目标）与平均配对相对变化使用不同权重。前者先平均V，再计算均值差或均值比；"
        "后者先对各配对除以自己的A基线，再等权汇总角色/场景，因此小基线上的同样绝对变化会形成更大的百分比。"
        "当所有配对基线为正且等权时，平均V变化比例=sum(ΔV)/sum(V_A)，等于以V_A为权重的配对百分比；"
        "平均配对百分比则不使用这种基线权重。零基线仍单列，不能填成0%。"]
    for row in records:
        direction="下降" if row["mean_delta_V"]<0 else "上升"
        text.append(f"**{row['algorithm']}出现符号反转：平均V从{number(row['A_mean_V'],6)}变为{number(row['B_mean_V'],6)}，"
            f"ΔV={number(row['mean_delta_V'],6)}，均值比变化{number(row['ratio_of_mean_V_gain'],4,True)}；"
            f"平均配对相对变化却为{number(row['mean_paired_relative_gain'],4,True)}。主目标平均V仍然{direction}，"
            "不能用相反符号的百分比诊断替代这一判断。**")
        pairs=improvements["pairs"]
        pairs=pairs[pairs.algorithm==row["algorithm"]]
        positive=pairs[pairs.delta_V>1e-12]
        negative=pairs[pairs.delta_V < -1e-12]
        row["positive_pairs"],row["negative_pairs"]=len(positive),len(negative)
        row["positive_group_mean_A_V"],row["negative_group_mean_A_V"]=positive.total_value_A.mean(),negative.total_value_A.mean()
        row["positive_sum_delta_V"],row["negative_sum_delta_V"]=positive.delta_V.sum(),negative.delta_V.sum()
        text.append(f"{row['algorithm']}在训练seed平均后的{len(pairs)}个配对中，{len(positive)}对价值增加、{len(negative)}对下降。"
            f"增加组的A平均V为{number(positive.total_value_A.mean(),6)}、累计ΔV为{number(positive.delta_V.sum(),6)}；"
            f"下降组的A平均V为{number(negative.total_value_A.mean(),6)}、累计ΔV为{number(negative.delta_V.sum(),6)}。"
            "这些真实分组描述了基线尺度与价值变化如何影响两种平均数，不是额外独立市场样本。")
    if not records:
        text.append("本组总体记录没有出现上述两种变化方向相反的算法；两项指标仍应分别解释。")
    return records,"\n\n".join(text)


def research_findings(tables, improvements, behavior):
    """Describe complete results; no data access, execution or selection here."""
    facts={"leaders":{},"pid_comparison":[],"budget_changes":{},"robustness":{}}
    paragraphs=["## 专门研究问题", "", "### DT与IQL是否仍领先"]
    for phase,title in zip(PHASES,["A冻结泛化","B联合调优","独立周期"]):
        ranked=tables[phase]["overall"].sort_values(["mean_V","algorithm"],ascending=[False,True])
        indexed=ranked.set_index("algorithm")
        dt,iql=indexed.loc["DT"],indexed.loc["IQL"]
        leaders=ranked[ranked.rank_V<=3].algorithm.tolist()
        facts["leaders"][phase]={"rank_at_most_3":leaders,"DT":dt.to_dict(),"IQL":iql.to_dict()}
        paragraphs += ["",f"{title}：V排名≤3的算法为{'、'.join(leaders)}；DT第{int(dt.rank_V)}（V={number(dt.mean_V)}），"
            f"IQL第{int(iql.rank_V)}（V={number(iql.mean_V)}），DT−IQL的平均V差={number(dt.mean_V-iql.mean_V)}。同V并列。"]
        pid,local=indexed.loc["AuctionNetPID"],indexed.loc["LocalPID"]
        facts["pid_comparison"].append({"phase":phase,"AuctionNetPID_V":pid.mean_V,"LocalPID_V":local.mean_V,
            "delta_V":pid.mean_V-local.mean_V,"AuctionNetPID_R":pid.mean_R,
            "delta_budget_percentage_points":100*(pid.mean_budget_utilization-local.mean_budget_utilization)})
    contrasts=tables["experiment_A"]["scenario_contrasts"]
    new_pid=contrasts.query("before_scenario == 'LocalPID' and after_scenario == 'AuctionNetPID'").sort_values("mean_delta_V")
    mixed=contrasts.query("before_scenario == 'Mixed' and after_scenario == 'Mixed2'")
    facts["A_new_opponent_changes"],facts["A_mixed_changes"]=new_pid.to_dict("records"),mixed.to_dict("records")
    paragraphs += ["", "### A面对新对手的变化", "",
        "保持目标策略冻结，将全LocalPID对手换为全AuctionNetPID。下表按ΔV由低到高排列，负值才表示下降；"
        "它衡量对手环境变化，与同场景A→B调优收益不同。", "",
        markdown_table(["算法","ΔV","平均角色百分比变化"],[[r.algorithm,number(r.mean_delta_V),number(r.mean_role_relative_change,2,True)] for r in new_pid.itertuples()]),
        "", "Mixed→Mixed2保留同一组23名Fixed，将另24个LocalPID换为AuctionNetPID。", "",
        markdown_table(["算法","Mixed V","Mixed2 V","ΔV"],[[r.algorithm,number(r.before_mean_V),number(r.after_mean_V),number(r.mean_delta_V)] for r in mixed.itertuples()])]
    paragraphs += ["", "### 跨场景稳健性"]
    for phase,title in [("experiment_A","A"),("experiment_B","B")]:
        overall=tables[phase]["overall"].copy()
        overall["scenario_V_cv"]=np.divide(overall.scenario_std_V,overall.mean_V,out=np.full(len(overall),np.nan),where=overall.mean_V>0)
        floor=overall.sort_values("worst_scenario_V",ascending=False).iloc[0]
        finite=overall.dropna(subset=["scenario_V_cv"])
        stable=finite.sort_values("scenario_V_cv").iloc[0] if len(finite) else None
        facts["robustness"][phase]={"highest_worst_scene_V":floor.to_dict(),"lowest_scene_V_cv":None if stable is None else stable.to_dict()}
        paragraphs += ["",f"{title}最差场景V最高者为{floor.algorithm}（{number(floor.worst_scenario_V)}）。"+
            (f"场景V变异系数最低者为{stable.algorithm}（场景标准差/五场景平均V={number(stable.scenario_V_cv)}，"
             f"其平均V={number(stable.mean_V)}）。" if stable is not None else "所有平均V为0，变异系数未定义。")+
            "价值下限高与波动小是两种不同标准；低波动不等于高价值。"]
    gains=improvements["overall"].set_index("algorithm")
    absolute=gains.sort_values("mean_delta_V",ascending=False).iloc[0]
    relative_defined=gains.dropna(subset=["mean_paired_relative_gain"])
    relative=relative_defined.sort_values("mean_paired_relative_gain",ascending=False).iloc[0] if len(relative_defined) else None
    facts["largest_B_gains"]={"absolute_algorithm":absolute.name,"absolute":absolute.to_dict(),
        "relative_algorithm":None if relative is None else relative.name,"relative":None if relative is None else relative.to_dict()}
    paragraphs += ["", "### B收益、预算改善与剩余短板", "",
        f"绝对平均ΔV最大为{absolute.name}（{number(absolute.mean_delta_V)}）。"+
        (f"平均配对百分比变化最大为{relative.name}（{number(relative.mean_paired_relative_gain,2,True)}）。" if relative is not None else "所有A参考V为0，百分比变化均未定义。")+
        "两种最大值的含义不同；主目标是否改善以平均V及其ΔV为准。"]
    facts["relative_vs_mean_V_sign_reversals"],gain_text=gain_interpretation(improvements)
    paragraphs += ["",gain_text]
    declines=gains[gains.mean_delta_V < -1e-12]
    facts["mean_V_declines"]=declines.reset_index().to_dict("records")
    if len(declines):
        paragraphs += ["", "按主目标平均V，B低于A的算法为："+"；".join(
            f"{a}（ΔV={number(r.mean_delta_V)}，均值比{number(r.ratio_of_mean_V_gain,2,True)}）" for a,r in declines.iterrows())+
            "。这些下降不能被更多训练步数、正的局部收益或某些百分比诊断抵消。"]
    for algorithm in ("BCQ","CQL","OnlineLP"):
        left=behavior["experiment_A"]["overall"].set_index("algorithm").loc[algorithm]
        right=behavior["experiment_B"]["overall"].set_index("algorithm").loc[algorithm]
        current=tables["experiment_B"]["overall"].set_index("algorithm").loc[algorithm]
        worst=tables["experiment_B"]["per_scenario"].query("algorithm == @algorithm").sort_values("mean_V").iloc[0]
        gain=gains.loc[algorithm]
        facts["budget_changes"][algorithm]={"A":left.to_dict(),"B":right.to_dict(),"gain":gain.to_dict(),"B_worst_scene":worst.to_dict()}
        paragraphs += ["",f"{algorithm}：A→B的ΔV={number(gain.mean_delta_V)}；前6片支出/B从{number(left.first6_spend,2,True)}变为{number(right.first6_spend,2,True)}；"
            f"最终预算使用率变化{number(gain.mean_budget_change*100,2)}个百分点。B内部R={number(current.mean_R)}，"
            f"最差场景为{worst.scenario}（V={number(worst.mean_V)}）。更晚消费与更高价值应分别判断，不能自动等同。"]
    paragraphs += ["", "OnlineLP同时换了成本响应表并搜索倍率；BCQ/CQL也同时改变数据覆盖、训练量和验证选择。"
        "以上只能报告联合变化，没有单因素消融证明某个学习目标、动作坐标或成本曲线单独造成收益。",
        "", "### AuctionNetPID是否优于LocalPID", "",
        markdown_table(["阶段","AuctionNetPID V","LocalPID V","ΔV","AuctionNetPID内部R","预算变化百分点"],
            [[r["phase"],number(r["AuctionNetPID_V"]),number(r["LocalPID_V"]),number(r["delta_V"]),
              number(r["AuctionNetPID_R"]),number(r["delta_budget_percentage_points"],2)] for r in facts["pid_comparison"]]),
        "", "正ΔV表示对应阶段的AuctionNetPID平均价值更高；这不是对标准PID理论的验证，AuctionNetPID本身没有P/I/D三项。"]
    return facts,"\n".join(paragraphs)


def write_reports(root, destination, tables, improvements, behavior, training, figures, episodes, selections, mechanisms, context, sources, execution):
    figure_by_name = {r["name"]: r for r in figures}
    def embed(name):
        r = figure_by_name[name]
        return f'![{r["description"]}]({Path(r["png"]).as_posix()})'
    def phase_table(phase):
        return markdown_table(["算法", "平均V", "内部R", "预算使用", "角色V P10", "最差场景V"],
            [[LABEL.get(r.algorithm, r.algorithm), number(r.mean_V), number(r.mean_R),
              number(r.mean_budget_utilization, 2, True), number(r.p10_V), number(r.worst_scenario_V)]
             for r in tables[phase]["overall"].itertuples()])
    improvement_table = markdown_table(["算法", "A平均V", "B平均V", "平均ΔV", "平均配对相对变化", "平均V变化比例", "预算变化百分点", "零参考配对"],
        [[LABEL.get(r.algorithm, r.algorithm), number(r.A_mean_V), number(r.B_mean_V), number(r.mean_delta_V),
          number(r.mean_paired_relative_gain, 2, True), number(r.ratio_of_mean_V_gain, 2, True),
          number(100 * r.mean_budget_change, 2), r.zero_reference_pairs]
         for r in improvements["overall"].itertuples()])
    new_a = episodes["experiment_A"].provenance.eq("new_execution").sum()
    reused_a = episodes["experiment_A"].provenance.isin(["historical_reuse", "historical_exact_policy_reuse"]).sum()
    a_leader = tables["experiment_A"]["overall"].iloc[0]
    b_leader = tables["experiment_B"]["overall"].iloc[0]
    independent_leader = tables["independent_test"]["overall"].iloc[0]
    trajectory = context["trajectory_provenance.json"]
    trajectory_display = {k: trajectory[k] for k in ["days", "episodes", "transitions", "reused_episodes", "new_episodes",
        "scenario_counts", "behavior_selection", "labels_are_actor_inputs", "test_used"]}
    input_audits=context.get("actual_input_audits",{})
    input_audit_links="、".join(link(root / "qa" / name) for name in input_audits) or "未在本次输入中加载"
    old_detail = OLD_ROOT / "report_auctionnet_baselines_detailed.md"
    setup = ("本轮将10类策略放入Fixed、LocalPID、Mixed、AuctionNetPID、Mixed2五场景。每局只替换一个目标，"
        "轮换48角色、两个评测seed；六种神经算法保留两个训练seed。A保持旧策略冻结，用新场景检验泛化；"
        "B在训练/验证阶段针对五场景重新调优，V为主要选择依据。A与B各使用本实验的LocalPID作为R分母。\n\n"
        f"A复用{int(reused_a)}局旧结果，新增{int(new_a)}局；A/B分别完成{len(episodes['experiment_A'])}/{len(episodes['experiment_B'])}局。"
        f"额外独立测试完成{len(episodes['independent_test'])}局，单独报告，不拼入A/B同日提升。"
        "period-24此前已公开分析，不能称全新盲测；额外周期的模型与参数在加载其市场前冻结。")
    method = ("先在角色内平均周期、评测seed和训练seed，再与同角色同场景LocalPID比较。主R先在各场景内平均可定义角色，"
        "再等权平均五场景；若某场景所有LocalPID参考均为0，五场景主R保持未定义，并另列可定义场景的诊断平均。"
        "A/B内部R的分母可能改变，R之差不是调优提升。配对提升先平均同配置的训练seed，再匹配同日/角色/场景/评测seed，"
        "计算ΔV及ΔV/V_A；每角色内汇总配对，再等权汇总角色与场景。V_A=0的比例提升保持未定义并另计，绝对ΔV仍保留。"
        "总体均值比提升单列，与平均配对百分比不同。所有P10/中位/标准差为角色分布描述，非独立市场置信区间。")
    selection_method="选模口径分开：A旧九类沿用原冻结选择；仅新增AuctionNetPID按原三场景、相对旧冻结LocalPID的平均R选参。B全10类统一按五场景验证平均V选参，R仅作诊断。"
    short = ["# AuctionNet：10算法、5对手场景的冻结泛化与联合调优", "", setup, "", selection_method, "", method,
        "", failure_history_paragraph(root,context),
        "", recovery_paragraph(root,context),
        "", "## 算法与本轮工作", "", markdown_table(["算法", "核心机制", "本轮工作"],
            [[LABEL.get(a, a), *PRINCIPLES[a]] for a in ALGORITHMS]),
        "", "A的旧九类参数/权重保持冻结。B的六种神经算法均从头训练；四种经典策略只拟合成本表或搜索参数。实际最终B配置如下：", "",
        markdown_table(["算法", "B冻结配置"], [[LABEL.get(a,a), selected_summary(a,selections,sources)[1][1]] for a in ALGORITHMS]),
        "", "## A：冻结策略泛化", "", phase_table("experiment_A"), "", embed("01_frozen_value"),
        "", "## B：五场景联合调优（主排名）", "", phase_table("experiment_B"), "", embed("02_tuned_value"),
        "", "## 调优前后的配对变化", "", improvement_table, "", gain_interpretation(improvements)[1], "", embed("05_paired_value_improvement"),
        "", "## 独立周期补充", "", phase_table("independent_test"), "", embed("12_independent_period"),
        "", f"A平均V领先者为{LABEL.get(a_leader.algorithm,a_leader.algorithm)}（{number(a_leader.mean_V)}）；"
        f"B为{LABEL.get(b_leader.algorithm,b_leader.algorithm)}（{number(b_leader.mean_V)}）；独立周期为"
        f"{LABEL.get(independent_leader.algorithm,independent_leader.algorithm)}（{number(independent_leader.mean_V)}）。"
        "这些是已完成记录的描述，不是统计显著性或官方竞赛结论。新数据、参数和训练步数共同变化，不能把联合收益归因于单一因素。",
        "", "完整场景、类型、训练/验证、预算路径与资源证据见 " + link(root / "report_auctionnet_baselines_detailed.md", "详细报告") + "。"]
    short_path = root / "report_auctionnet_baselines_comparison.md"
    short_path.write_text("\n".join(short), encoding="utf-8")
    details = ["# AuctionNet五场景：机制、训练、冻结与结果复盘", "", setup, "", selection_method, "", "## 统计与比较边界", "", method,
        "", "总体表的角色P10先对每个角色跨五场景取均值，再取48角色分位；`mean_scenario_p10_R`另保存逐场景P10再平均。"
        "R的标准差/中位数/P10描述角色跨场景均值分布，与主R的场景优先加权次序不同；零参考分布不均时两者均值可能不同，"
        "`mean_role_average_R`另列角色优先均值。"
        "分位数与均值不可交换。预算阈值仅从slot0..71转换一次为第1..72片；never保持缺失并另计。",
        "", "下文内嵌旧九算法的完整机制与适配说明，来源为 " + link(old_detail, "历史详细报告") +
        "；其中历史搜索值、旧数据尺度与DT轨迹例子会明确标记，不能当作本轮B的配置或成绩。",
        "", "## 原始与新增记录", "", "历史逐文件保护见 " + link(root / "historical_audit.json") + " 和 " + link(root / "historical_assets_manifest.json") +
        "。A历史行的source_scenario/source_task保留原身份，PID只在新导入副本规范化为LocalPID；旧文件未改变。",
        "", "## 场景、数据与训练协议", "",
        "Fixed为其余47人全部冻结Fixed；LocalPID为其余47人全部旧冻结LocalPID；AuctionNetPID为其余47人全部新校准冻结控制器。"
        "Mixed固定分配23名Fixed与24名LocalPID；Mixed2保留同seed、同角色的23名Fixed成员，将另24人替为AuctionNetPID。"
        "对手均使用独立实例并每局reset，所有目标算法比较同一场景分配；对手配置在A/B之间不重新校准。",
        "", "对手配置与来源：" + link(root / "opponents_frozen.json") + "、" + link(root / "opponents_provenance.json") + "。",
        "", "训练使用period-7/8，验证使用period-21。A的新AuctionNetPID按旧三场景相对旧LocalPID选参；"
        "B的所有目标按五场景验证平均V选参，R只是诊断。验证轮换角色0/8/16/24/32/40及一个评测seed，"
        "神经候选必须两个训练seed均完整。选择准则与测试排名有明确区分。",
        "", "实际训练轨迹来源与分布：", "", "```json", json.dumps(clean(trajectory_display),ensure_ascii=False,indent=2), "```",
        "", "完整样本清单与源SHA见 " + link(root / "trajectory_episodes.csv") + "、" + link(root / "trajectory_provenance.json") + "。",
        "", "各算法结构和输入保持旧适配：18维合法自身历史，无场景标签，标准化与动作/奖励尺度仅从新训练集求得并冻结。"
        "训练transition不是独立市场样本；48个角色都曾出现于训练。六模型均为两个学习率×两个训练seed，从头训练1800步，保存600/1800。"
        "CQL采用linear_scaled和1e-4/3e-4；其余采用log1p和3e-4/1e-3。普通批量64，DT每批32条至多10片序列；"
        "相同步数不代表相同反向传播工作量，也不证明收敛。",
        "", "实际训练状态：", "", "```json", json.dumps(clean(context["models/training_status.json"]),ensure_ascii=False,indent=2), "```",
        "", "### 共同输入、动作与奖励", "", context["shared_feature_definition"],
        "", "每维仅以本轮训练数据的均值μ和总体标准差σ标准化，σ<1e-6替换为1。设s为该18维状态，u为网络动作，"
        "a为实际alpha，v为本片原始竞胜价值，d只在第72片为1。离线奖励r=v/S，不额外惩罚成本；耗尽也完整走72片。"
        "常规算法u=log1p(a/s_a)，部署a=s_a·expm1(max(0,u))；CQL采用u=a/s_a，部署a=s_a·max(0,u)。"
        "两种坐标均无统一alpha上限，非有限动作直接失败。",
        "", f"本轮B真实训练动作尺度s_a={number(context['models/training_metadata.json']['action_scale'],8)}，"
        f"奖励尺度S={number(context['models/training_metadata.json']['reward_scale'],8)}；"
        "A旧神经模型各自保留其checkpoint中的旧尺度。完整μ/σ与数据SHA见 " + link(root / "models/training_metadata.json") + "。",
        "", "## A与B总体结果", "", "### A冻结", "", phase_table("experiment_A"), "", "### B联合调优", "", phase_table("experiment_B"),
        "", embed("04_overall_value_comparison"), "", embed("03_internal_relative_performance"),
        "", "## 十类算法：机制、本轮选择与逐场景结果"]
    selection_path = destination / "selection_records.json"
    dump(selection_path, selections)
    for a in ALGORITHMS:
        details += ["", "### " + LABEL.get(a, a), "", PRINCIPLES[a][0], "", PRINCIPLES[a][1], ""]
        if a == "AuctionNetPID":
            details += ["以下为新增控制器原始机制与72片适配；不将上游名称解释为标准P/I/D。", "", mechanisms[a], ""]
        else:
            details += ["**历史机制档案。以下原理/接口/网络说明保持有效；其中‘本轮’、搜索数值、训练尺度和轨迹例子均指旧三场景实验。"
                "其历史最终效果表没有复制。当前A/B选择与五场景成绩紧接档案之后给出。**", "", mechanisms[a], ""]
        details += ["#### 本次五场景调优与冻结", "", PRINCIPLES[a][1], "", markdown_table(["实验", "实际冻结选择"], selected_summary(a,selections,sources)), ""]
        selected = []
        for source, records in selections.items():
            for record in records:
                if record.get("algorithm") == a:
                    selected.append({"source": source, "training_seed": record.get("training_seed", 0),
                        "validation_choice": record.get("validation_choice"), "spec": record.get("spec", record.get("config"))})
        details += ["实际冻结配置（原样摘取，A/B来源分开）：", "", "```json", json.dumps(clean(selected), ensure_ascii=False, indent=2), "```", ""]
        for phase, phase_name in zip(PHASES, ["A冻结", "B联合调优", "独立周期"]):
            g = tables[phase]["per_scenario"].query("algorithm == @a").set_index("scenario").reindex(SCENARIOS)
            details += [f"{phase_name}：", "", markdown_table(["场景", "V", "内部R", "预算", "wins", "σ(V)", "V中位", "V P10", "R P10"],
                [[s, number(r.mean_V), number(r.mean_R), number(r.mean_budget_utilization, 2, True), number(r.mean_wins, 1),
                  number(r.std_V), number(r.median_V), number(r.p10_V), number(r.p10_R)] for s, r in g.iterrows()]), ""]
        delta = improvements["overall"].set_index("algorithm").loc[a]
        details += [f"记录中的联合变化：平均ΔV={number(delta.mean_delta_V)}，平均配对相对变化={number(delta.mean_paired_relative_gain,2,True)}，"
            f"预算变化={number(delta.mean_budget_change*100,2)}个百分点。新训练分布、参数及训练量一起改变，以上仅表示联合效果；没有单因素消融证明哪项修改主导。"]
        ba=behavior["experiment_A"]["overall"].set_index("algorithm").loc[a]
        bb=behavior["experiment_B"]["overall"].set_index("algorithm").loc[a]
        b_scenes=tables["experiment_B"]["per_scenario"].query("algorithm == @a").sort_values("mean_V")
        details += ["", f"可观察的行为变化：前6片支出/初始预算从{number(ba.first6_spend,2,True)}变为{number(bb.first6_spend,2,True)}；"
            f"到达95%预算的片号中位数（只计达到者）从第{number(ba.slot95_median_reached,1)}片变为第{number(bb.slot95_median_reached,1)}片，"
            f"未达到者分别{int(ba.slot95_never)}/{int(ba.episodes)}与{int(bb.slot95_never)}/{int(bb.episodes)}。"
            f"B平均V最低场景为{b_scenes.iloc[0].scenario}（{number(b_scenes.iloc[0].mean_V)}），最高为{b_scenes.iloc[-1].scenario}（{number(b_scenes.iloc[-1].mean_V)}）。"
            "这些路径可以核实是否更早花钱或预算未用完；它们不能单独识别网络估值、行为覆盖或市场反馈中哪一个因素导致成绩。"]
    details += ["", research_findings(tables,improvements,behavior)[1],
        "", "## 配对提升与对手泛化", "", improvement_table, "", embed("05_paired_value_improvement"),
        "", "原三场景与新增两场景的差异体现对手环境变化；A到B在同场景的配对变化体现联合调优结果，二者不能混称。"
        "Mixed与Mixed2采用各自固定分配；同场景同目标同seed的A/B分配一致，但不同场景的实际成交和PID反馈路径会变化。",
        "", "A保持目标策略不变时，更换对手的具体差异如下。百分比先平均seed得到每角色V，再算角色百分比并等权平均；"
        "这与同场景A→B调优提升是不同对比。", "",
        markdown_table(["算法","对手场景变化","起点V","终点V","ΔV","平均角色变化"],
            [[r.algorithm,f"{r.before_scenario}→{r.after_scenario}",number(r.before_mean_V),number(r.after_mean_V),
              number(r.mean_delta_V),number(r.mean_role_relative_change,2,True)] for r in tables["experiment_A"]["scenario_contrasts"].itertuples()
             if r.before_scenario != "Fixed"]),
        "", "## 72片预算与价值路径", "", embed("06_budget_before_after"), "", embed("07_cumulative_budget"), "", embed("08_cumulative_value")]
    for phase, title in zip(PHASES[:2], ["A冻结", "B联合调优"]):
        g = behavior[phase]["overall"].set_index("algorithm").reindex(ALGORITHMS)
        details += ["", title, "", markdown_table(["算法", "首片/B", "前6片/B", "α中位/P90/最大", "95%中位片", "未达95%", "后半价值份额"],
            [[LABEL.get(a,a), number(r.first_slot_spend,2,True), number(r.first6_spend,2,True),
              "/".join(number(r[c],2) for c in ["alpha_median","alpha_p90","alpha_max"]),
              number(r.slot95_median_reached,1), f"{int(r.slot95_never)}/{int(r.episodes)}", number(r.late_value_share,4,True)] for a,r in g.iterrows()])]
    details += ["", "达到95%或99%不等于余额严格为0。旧A的JSONL未记录逐片wins，旧片wins保持缺失；逐局总wins仍来自完整成绩。"
        "α统计包含全部72片，无人工裁剪；早消费和低价值的共现只支持行为关联，不能直接证明Q或VAE的因果失效。",
        "", "## 类型、跨场景与独立周期", "", embed("09_player_types"),
        "", "下表是每种算法在实际player_type代码下的平均V：先取该类型各场景的角色平均，再等权平均五场景。"
        "列标题的角色数由本实验真实player_id/type映射去重计算，不把五场景、两个评测seed或训练seed重复计为新角色。"
        "类型代码仅沿用源数据，不臆测其对应的行业名称。A与B分别展示，未合并排名。",
        "", "### A：按类型代码的平均V", "", type_value_table(tables,"experiment_A"),
        "", "### B：按类型代码的平均V", "", type_value_table(tables,"experiment_B"),
        "", embed("10_cross_scenario_robustness"),
        "", phase_table("independent_test"), "", embed("12_independent_period"),
        "", "## 实际训练与验证证据", "", embed("11_actual_learning_curves"),
        "", "上排只画training.json实际日志点的loss[0]，不同算法的loss定义不相同；下排只画完整候选的验证平均V。"
        "误差线为两个训练seed的描述性标准差，不是跨独立市场置信区间；失败/不完整候选没有补成0分。",
        "", markdown_table(["算法","配置","步数","完整局数/要求","完整","验证平均V","训练seed标准差","失败/跳过"],
            [[r.algorithm,r.config_index,r.step,f"{r.successful_episodes}/{r.expected_episodes}",r.complete,
              number(r.mean_V),number(r.training_seed_V_std),f"{r.failed_episodes}/{r.skipped_episodes}"] for r in training["validation"].itertuples()]),
        "", link(destination / "training_runs.csv", "逐run训练计时与参数") + "、" + link(destination / "training_checkpoints.csv", "保存步数与checkpoint") + "、" + link(destination / "training_validation.csv", "完整候选验证") + "。",
        "", "训练数据、LP表重建、神经权重、独立周期协议及最终结果的只读审计来源：" + input_audit_links + "。"
        "各审计只支持其声明范围；独立周期协议审计不代表最终结果完整性验收。训练曲线仅保留step1与每50步的实际日志，"
        "每步有限性检查与每步完整日志保存是不同事实，不能声称保存了所有1800步的loss。",
        "", ("独立周期最终结果另经全量审计：7680局、552960个时间片与1658880条诊断记录均核对，"
            "7534个原成功局和146个恢复局的来源、整数金额账本与完整覆盖通过；历史仍保留5次超时和141次跳过。"
            "该审计不执行新比赛；最终JSON必须status=passed且valid=true，其脚本和最终输入SHA同时匹配才被本报告接受。"
            if "independent_test_completed_audit.json" in input_audits else "本次组装未加载独立周期最终全量审计。"),
        "", "训练计时分开保留：原training.json的seconds覆盖该训练器定义的范围；call_wall_seconds覆盖外层整次调用。"
        "这里求和表示串行CPU调用的累计耗时，不替代包含生成轨迹和所有评测的端到端墙钟时间。", "",
        markdown_table(["算法","实际run数","训练器累计秒","外层调用累计秒"],
            [[a,len(g),number(g.training_seconds.sum(),3),number(g.call_wall_seconds.sum(),3)]
             for a,g in training["run_receipts"].groupby("algorithm")]),
        "", "经典策略、对手校准与神经候选的全部实际验证分数（包含不完整标记）见 " + link(destination / "training_search_scores.csv") + "。",
        "", "### 增量执行和资源记录", "",
        failure_history_paragraph(root,context), "",
        recovery_paragraph(root,context), "",
        interruption_paragraph(context), "",
        recovery_prefix_paragraph(context), "",
        resource_paragraph(context), "",
        markdown_table(["阶段","状态","成功/要求","失败/跳过","最新调用worker","有记录运行段累计秒"],
            [[r.stage,r.status,f"{r.completed_episodes}/{r.expected_episodes}",
              f"{r.failed_episodes if pd.notna(r.failed_episodes) else 0}/{r.skipped_episodes if pd.notna(r.skipped_episodes) else 0}",
              int(r.workers) if pd.notna(r.workers) else "汇总表",number(r.recorded_segment_seconds,2)] for r in execution.itertuples()]),
        "", "A/B/独立测试最终目录是合并汇总，不能与其_new执行目录再相加。有记录运行段累计秒来自runtime_segment文件求和；"
        "没有运行段文件的汇总目录保持未定义，不用最近一段状态秒数冒充完整累计。"
        "并行或重叠阶段的段耗时相加不等于端到端墙钟；原独立阶段与恢复阶段的重叠不被扣除。"
        "旧runtime段文件没有记录worker数，因此阶段最新status中的worker不反推给旧段；所有原段都单独保留来源。"
        "历史复用行不计为新执行。若存在4/8 worker资源基准，它只使用训练周期，计入辅助任务，"
        "不进入排名、训练轨迹或模型选择。完整阶段与运行段来源见 " + link(destination / "execution_stages.csv") + "、" + link(destination / "execution_segments.csv") + "。",
        "", "资源建议与实际采用分别见 " + link(root / "resource_selection.json") + "、" + link(root / "resource_application.json") + "；"
        "原始推荐状态保留在 " + link(root / "resource_benchmark/operational_adjustment.json") + "。",
        "", "运行环境清单（操作系统/硬件容量/软件版本/配置线程数，不是峰值内存或性能基准）：", "", "```json",
        json.dumps(clean(context.get("runtime_environment.json",{"status":"not_recorded"})),ensure_ascii=False,indent=2), "```",
        "", concurrency_paragraph(root,context),
        "", "## 实际推理资源"]
    for phase in PHASES:
        inf = tables[phase]["inference"]
        records = []
        for a, g in inf.groupby("algorithm"):
            records.append([LABEL.get(a,a), number(g.mean_episode_paired_p99_ms.mean(),3), number(g.max_episode_paired_p99_ms.max(),3),
                number(g.max_episode_peak_process_rss_mb.max(),1) if "max_episode_peak_process_rss_mb" in g else "未记录"])
        details += ["", phase, "", markdown_table(["算法", "平均单局paired P99 ms", "最大单局paired P99 ms", "进程峰值MiB"], records)]
    details += ["", "计时范围以各行timing_scope为准：新阶段直接记录目标bidding/update，不含核心结算和诊断写盘；A旧行保留历史测量。"
        "内存包含解释器、市场、模型及缓存，不能视为单个网络独占内存；硬件和并发负载不同，A/B延迟差不应当作纯算法加速。",
        "", "## 新增实现的QA验收证据", "",
        markdown_table(["范围","通过测试数","QA运行秒","原始日志"],
            [[r["kind"],r["passed_tests"],number(r["qa_elapsed_seconds"],2),link(r["source"])] for r in context.get("qa_evidence",[])]),
        "", "本轮完整集成验收记录为199项通过，包含全部图形渲染测试；该次QA与正式A并行执行。"
        "报告后续修改的focused检查单列。QA模拟数据、QA秒数和试绘均不进入任何算法排名、训练数据或实验耗时，"
        "也不是机器吞吐benchmark；不能用历史旧测试数量替代本轮新日志。",
        "", "## 限制与复现", "", "市场周期数量有限，角色和seed重放不是独立市场。验证同时用于多个开发选择；更多场景与较长训练增加了搜索预算。"
        "本轮只报告预冻结策略的真实执行结果，不依据测试调参，不声称算法收敛、统计显著或官方认证。",
        "", "在AuctionNet目录执行：", "", "```powershell", ".\\.venv-baselines\\Scripts\\python.exe -X utf8 -m harmony_baselines.five_scenarios.report", "```",
        "", "该命令只读已完成实验并写本轮reporting与简详报告；所有输入SHA和派生图表索引见 " + link(destination / "report_facts.json") + "。"]
    detail_path = root / "report_auctionnet_baselines_detailed.md"
    detail_path.write_text("\n".join(details), encoding="utf-8")
    return [short_path, detail_path]


def load_completed_phase(root, phase, sources):
    directory = root / phase
    status = sources.json(directory / "episode_results.status.json")
    if status.get("status") != "completed" or status.get("valid") is not True:
        raise ValueError(f"{phase}: no valid completed status; cannot publish ranking")
    frame = sources.csv(directory / "episode_results.csv", dtype={"player_id": str, "player_type": str, "day_id": str})
    if status.get("result_sha256") and status["result_sha256"] != sources.hashes[str((directory / "episode_results.csv").resolve())]:
        raise ValueError("CSV hash differs from completed status: " + phase)
    if status.get("slots_sha256") and status["slots_sha256"] != digest(directory / "slots.csv"):
        raise ValueError("Slot evidence hash differs from completed status: " + phase)
    if "phase" in frame and not frame.phase.eq(phase).all():
        raise ValueError("Wrong experiment phase in result file")
    if not {"source_task", "provenance", "source_result_path", "policy_id"} <= set(frame):
        raise ValueError("Missing explicit source/policy provenance")
    days = ("period-25",) if phase == "independent_test" else ("period-24",)
    frame = validate_phase(frame, days=days)
    selection = sources.json(directory / "selected_frozen.json")
    policy_ids = selection["policy_ids"]
    if any(policy_ids.get(f"{r.algorithm}:{int(r.training_seed)}") != r.policy_id for r in frame.itertuples()):
        raise ValueError("Result policy IDs differ from frozen selection: " + phase)
    if status.get("completed_episodes", len(frame)) != len(frame) or status.get("expected_episodes", len(frame)) != len(frame):
        raise ValueError("Status coverage differs from result CSV")
    if any(status.get(k, 0) != 0 for k in ["failed_episodes", "skipped_episodes"]):
        raise ValueError("Failed/skipped cases cannot be hidden inside completed ranking")
    if phase=="independent_test" and (frame.provenance.eq("resource_recovery").any() or status.get("historical_failed_episodes",0)>0):
        sources.recovery_evidence=verify_recovery_manifest(root,frame,status,sources)
    return frame


def generate(root=DEFAULT_ROOT, *, check_only=False):
    root = Path(root).resolve()
    if root == OLD_ROOT.resolve() or OLD_ROOT.resolve() in root.parents:
        raise ValueError("Refusing to write into historical experiment")
    sources = Sources()
    episodes = {phase: load_completed_phase(root, phase, sources) for phase in PHASES}
    paired = paired_improvement(episodes["experiment_A"], episodes["experiment_B"])
    if check_only:
        return {"status": "passed", "phase_rows": {p: len(f) for p, f in episodes.items()}, "paired_cases": len(paired["pairs"])}
    from .historical_audit import verify
    integrity = verify(root)
    if integrity["status"] != "passed":
        raise ValueError("Historical assets changed: " + json.dumps(integrity["mismatches"][:5]))
    destination = root / "reporting"
    destination.mkdir(parents=True, exist_ok=True)
    dump(destination / "report.status.json", {"status":"generating","valid":False})
    tables, behavior = {}, {}
    for phase, frame in episodes.items():
        tables[phase] = aggregate_phase(frame)
        for name, table in tables[phase].items():
            table.to_csv(destination / f"{phase}_{name}.csv", index=False)
        behavior[phase] = behavioral_statistics(root / phase, frame, sources)
        for name, table in behavior[phase].items():
            table.to_csv(destination / f"{phase}_behavior_{name}.csv", index=False)
    for name, table in paired.items():
        table.to_csv(destination / f"paired_{name}.csv", index=False)
    training = training_evidence(root, sources)
    expected_groups={(a,c,s) for a in NEURAL for c in (0,1) for s in (600,1800)}
    actual_groups=set(training["validation"][["algorithm","config_index","step"]].itertuples(index=False,name=None))
    if (len(training["runs"]) != 24 or len(training["checkpoints"]) != 48 or len(training["run_receipts"]) != 24
            or len(training["validation"]) != 24 or actual_groups != expected_groups):
        raise ValueError("Actual training/catalog/validation coverage differs from declared protocol")
    for name, table in training.items():
        table.to_csv(destination / f"training_{name}.csv", index=False)
    selections = selection_records(root, sources)
    mechanisms = historical_mechanisms(sources)
    context = experiment_context(root, sources)
    context["recovery_evidence"]=sources.recovery_evidence
    execution,execution_segments = execution_evidence(root,sources,include_segments=True)
    execution.to_csv(destination / "execution_stages.csv",index=False)
    execution_segments.to_csv(destination / "execution_segments.csv",index=False)
    failures=failure_history(root,sources)
    for kind,table in failures.items():
        table.to_csv(destination / ("execution_"+kind+".csv"),index=False)
    context["failure_history"]={kind:table.to_dict("records") for kind,table in failures.items()}
    figures = draw_figures(destination / "figures", tables, paired, behavior, training)
    reports = write_reports(root, destination, tables, paired, behavior, training, figures, episodes, selections, mechanisms, context, sources, execution)
    changed_inputs = [p for p, expected in sources.hashes.items() if digest(p) != expected]
    if changed_inputs:
        raise ValueError("Inputs changed during reporting: " + str(changed_inputs[:5]))
    facts = {"status": "completed", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "completed frozen A, jointly tuned B and separate independent test",
        "phase_rows": {p: len(f) for p, f in episodes.items()}, "input_sha256": sources.hashes,
        "historical_integrity": integrity, "algorithm_order": ALGORITHMS, "scenario_order": SCENARIOS,
        "normalization": "mean seeds within role; R uses same-experiment LocalPID; defined roles equally within each scene then exactly five equal scenes; wholly undefined scene makes main R undefined",
        "paired_improvement": "mean training seeds before same-role/day/scene/evaluation-seed pairing; zero A undefined; equal role/scene mean",
        "threshold_slots": "one-based first end-of-slot crossing; never separate; no double +1",
        "overall_dispersion": "first average each role across 5 scenes, then 48-role SD/median/P10",
        "phase_overall": {p: t["overall"].to_dict("records") for p,t in tables.items()},
        "phase_scenario_contrasts": {p: t["scenario_contrasts"].to_dict("records") for p,t in tables.items()},
        "phase_behavior": {p:b["overall"].to_dict("records") for p,b in behavior.items()},
        "training_status":context["models/training_status.json"],
        "execution_stages":execution.to_dict("records"),
        "execution_segments":execution_segments.to_dict("records"),
        "failure_history":context["failure_history"],
        "interruption_evidence":context.get("interruption_evidence",{}),
        "recovery_evidence":context.get("recovery_evidence",{}),
        "recovery_prefix_evidence":context.get("recovery_prefix_evidence",{}),
        "resource_evidence":context.get("resource_evidence",{}),
        "runtime_environment":context.get("runtime_environment.json"),
        "runtime_concurrency_observation":context.get("runtime_concurrency_observation.json"),
        "qa_evidence":context.get("qa_evidence",[]),
        "actual_input_audits":context.get("actual_input_audits",{}),
        "research_findings":research_findings(tables,paired,behavior)[0],
        "paired_overall": paired["overall"].to_dict("records"), "figures": figures,
        "reports": [str(p) for p in reports], "script_sha256": digest(Path(__file__))}
    dump(destination / "report_facts.json", facts)
    outputs=list(destination.glob("*.csv"))+[Path(r[format]) for r in figures for format in ("png","svg")]+reports+[destination / "report_facts.json"]
    dump(destination / "output_manifest.json", {str(p.resolve()):digest(p) for p in outputs})
    dump(root / "historical_integrity_verification.json", integrity)
    dump(destination / "report.status.json", {"status": "completed", "valid": True, "figures": len(figures), "reports": [str(p) for p in reports]})
    return {"status": "completed", "figures": len(figures), "reports": [str(p) for p in reports]}


def audit_bookkeeping(root=DEFAULT_ROOT):
    """An intermediate status/time snapshot, never a performance analysis."""
    root=Path(root).resolve()
    if root==OLD_ROOT.resolve() or OLD_ROOT.resolve() in root.parents:
        raise ValueError("Refusing to write into historical experiment")
    sources=Sources()
    resources=resource_evidence(root,sources)
    stages,segments=execution_evidence(root,sources,include_segments=True)
    environment=sources.json(root / "runtime_environment.json") if (root / "runtime_environment.json").exists() else None
    record={"status":"intermediate_bookkeeping_snapshot","generated_utc":datetime.now(timezone.utc).isoformat(),
        "not_a_completed_result_audit":True,"ranking_or_test_values_read":False,
        "resource_evidence":resources,"runtime_environment":environment,
        "stages":stages.to_dict("records"),"segments":segments.to_dict("records"),"input_snapshot_sha256":sources.hashes,
        "notes":["QA and smoke fixtures excluded", "final rollups must not be added to execution stages",
                 "latest status worker count is not inferred for old segment files", "an active unfinished segment has no final runtime file yet"]}
    destination=root / "qa"
    destination.mkdir(parents=True,exist_ok=True)
    path=destination / "report_bookkeeping_audit.json"
    dump(path,record)
    return {"status":record["status"],"stages":len(stages),"segments":len(segments),
        "applied_workers":resources.get("applied_workers"),"benchmark_auxiliary_episodes":resources.get("benchmark_episodes"),"path":str(path)}


def audit_ab_comparison(root=DEFAULT_ROOT):
    """Read complete A/B only. Never touches independent-test files."""
    root=Path(root).resolve()
    if root==OLD_ROOT.resolve() or OLD_ROOT.resolve() in root.parents:
        raise ValueError("Refusing to write into historical experiment")
    sources=Sources()
    episodes={phase:load_completed_phase(root,phase,sources) for phase in PHASES[:2]}
    tables={phase:aggregate_phase(frame) for phase,frame in episodes.items()}
    paired=paired_improvement(episodes["experiment_A"],episodes["experiment_B"])
    warnings,explanation=gain_interpretation(paired)
    reconciliation={}
    for phase,suffix in (("experiment_A","A"),("experiment_B","B")):
        a=tables[phase]["overall"].set_index("algorithm").mean_V
        b=paired["overall"].set_index("algorithm")[suffix+"_mean_V"]
        reconciliation[phase]=float((a-b).abs().max())
        if reconciliation[phase]>1e-10:
            raise ValueError("Paired mean values do not reconcile with phase means")
    decomposition={}
    for warning in warnings:
        group=paired["pairs"][paired["pairs"].algorithm==warning["algorithm"]]
        rows=[]
        for label,mask in (("positive_delta",group.delta_V>1e-12),("negative_delta",group.delta_V < -1e-12),("tie",group.delta_V.abs()<=1e-12)):
            subset=group[mask]
            rows.append({"group":label,"pairs":len(subset),"sum_delta_V":subset.delta_V.sum(),
                "mean_A_value":subset.total_value_A.mean(),"mean_B_value":subset.total_value_B.mean(),
                "mean_pair_relative_change":subset.relative_gain.mean()})
        decomposition[warning["algorithm"]]=rows
    destination=root / "qa"
    destination.mkdir(parents=True,exist_ok=True)
    paired["overall"].to_csv(destination / "AB_comparison_pre_audit.csv",index=False)
    record={"status":"passed_complete_AB_pre_audit","not_final_three_phase_report":True,
        "independent_test_read":False,"model_or_selection_changed":False,
        "phase_rows":{p:len(f) for p,f in episodes.items()},"input_sha256":sources.hashes,
        "phase_overall":{p:t["overall"].to_dict("records") for p,t in tables.items()},
        "paired_overall":paired["overall"].to_dict("records"),"phase_vs_paired_mean_max_abs_error":reconciliation,
        "sign_reversals":warnings,"sign_reversal_pair_decomposition":decomposition,"interpretation":explanation}
    dump(destination / "AB_comparison_pre_audit.json",record)
    return {"status":record["status"],"phase_rows":record["phase_rows"],
        "sign_reversal_algorithms":[r["algorithm"] for r in warnings],"independent_test_read":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--audit-bookkeeping", action="store_true", help="Snapshot only status/runtime/resources; no scores, ranking, or formal report")
    parser.add_argument("--audit-ab-comparison", action="store_true", help="Audit complete A/B only without reading independent test or generating reports")
    args = parser.parse_args()
    result=(audit_ab_comparison(args.root) if args.audit_ab_comparison else audit_bookkeeping(args.root)
        if args.audit_bookkeeping else generate(args.root, check_only=args.check_inputs))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
