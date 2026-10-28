"""Evidence-based comparisons: missing experiments stay explicitly missing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from harmony_baselines.data import sha256_file
from harmony_baselines.five_scenarios.common import ALGORITHMS, SCENARIOS, ROLES, SEEDS, read, dump
from harmony_baselines.five_scenarios.report import aggregate_phase
from .evaluate import VARIANTS, KEYS, BASELINE_ROOT, audit_baseline


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def complete_algorithms(frame, protocol):
    """A partial algorithm must never be ranked against full Baselines."""
    if frame.empty:
        frame = pd.DataFrame(columns=KEYS)
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate frozen PPO evaluation cell")
    actual = set(frame[KEYS].itertuples(index=False, name=None))
    expected = {(t["algorithm"], t["scenario"], t["day_id"], str(t["target"]), int(t["seed"]), int(t["training_seed"]))
                for t in protocol["tasks"]}
    if not actual <= expected:
        raise ValueError("Evaluation contains cases outside the frozen protocol")
    labels = list(dict.fromkeys(list(VARIANTS) + [c["algorithm"] for c in protocol["candidates"]]))
    records, valid = [], []
    for algorithm in labels:
        need = {k for k in expected if k[0] == algorithm}
        have = {k for k in actual if k[0] == algorithm}
        completed = bool(need) and need == have
        records.append(dict(algorithm=algorithm, status="completed" if completed else "未完成",
                            expected_episodes=len(need), completed_episodes=len(have), missing_episodes=len(need-have)))
        if completed:
            valid.append(algorithm)
    return valid, pd.DataFrame(records)


def paired_gains(frame):
    """Average training seeds, then pair the identical market/role/eval seed."""
    keys = ["scenario", "day_id", "player_id", "seed"]
    means = frame.groupby(["algorithm"] + keys, as_index=False).total_value.mean()
    rows = []
    for algorithm in set(frame.algorithm) - set(ALGORITHMS):
        candidate = means[means.algorithm == algorithm]
        for baseline in ALGORITHMS:
            reference = means[means.algorithm == baseline]
            joined = candidate.merge(reference, on=keys, suffixes=("", "_reference"), validate="one_to_one")
            if len(joined) != len(candidate) or len(joined) != len(reference):
                raise ValueError("Unpaired algorithm coverage")
            joined["baseline"] = baseline
            joined["delta_V"] = joined.total_value - joined.total_value_reference
            joined["relative_gain"] = np.divide(joined.delta_V, joined.total_value_reference,
                out=np.full(len(joined), np.nan), where=joined.total_value_reference > 0)
            rows.append(joined.drop(columns=["algorithm_reference"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _verify_pair_conditions(frame):
    for column, keys in [("dataset_fingerprint", ["day_id"]), ("initial_budget", ["day_id", "player_id"]),
                         ("assignment_fingerprint", ["scenario", "day_id", "player_id", "seed"]),
                         ("rules_version", ["day_id"]), ("opponent_fingerprint", ["day_id"])]:
        if frame.groupby(keys)[column].nunique().gt(1).any():
            raise ValueError("Incomparable result conditions: " + column)


def _plots(tables, slots, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    destination.mkdir(exist_ok=True)
    files = []
    scenario = tables["per_scenario"]
    pivot = scenario.pivot(index="algorithm", columns="scenario", values="mean_V").reindex(columns=SCENARIOS)
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot.bar(ax=ax, width=.85, rot=35)
    ax.set(xlabel="", ylabel="Mean full-period value V", title="Frozen policies / original budgets / five scenarios")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "01_value_comparison.png", dpi=150)
    plt.close(fig)
    files.append("01_value_comparison.png")
    roles = tables["per_role"]
    fig, ax = plt.subplots(figsize=(12, 5))
    names = tables["overall"].algorithm.tolist()
    ax.boxplot([roles.loc[roles.algorithm == name, "R"].dropna() for name in names], tick_labels=names, showfliers=False)
    ax.axhline(1, color="#64748b", linestyle="--")
    ax.set(ylabel="Role/scenario relative value R", title="Distribution across advertiser roles and scenarios")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(destination / "02_role_distribution.png", dpi=150)
    plt.close(fig)
    files.append("02_role_distribution.png")
    if not slots.empty:
        # Representative case is chosen by identity, never by its score.
        chosen = slots.sort_values(["scenario", "day_id", "player_id", "seed", "training_seed"]).iloc[0]
        selected = slots[(slots.scenario == chosen.scenario) & (slots.day_id == chosen.day_id) &
                         (slots.player_id == chosen.player_id) & (slots.seed == chosen.seed)]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for algorithm, group in selected.groupby("algorithm"):
            mean = group.groupby("time_slot", as_index=False)[["alpha", "cumulative_cost", "cumulative_value", "remaining_budget"]].mean()
            for ax, col in zip(axes.flat, ["alpha", "cumulative_cost", "cumulative_value", "remaining_budget"]):
                ax.plot(mean.time_slot + 1, mean[col], label=algorithm)
                ax.set(xlabel="Time slot", ylabel=col.replace("_", " "))
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Identity-selected case: {chosen.day_id}, role {chosen.player_id}, {chosen.scenario}, seed {chosen.seed}")
        fig.tight_layout()
        fig.savefig(destination / "03_budget_behavior.png", dpi=150)
        plt.close(fig)
        files.append("03_budget_behavior.png")
    return files


def _committed_training_records(path):
    """Never plot optimizer/log tails beyond the durable resume boundary."""
    path = Path(path)
    last_path = path.parent / "last.pt"
    evidence = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    if last_path.exists():
        import torch
        committed = torch.load(last_path, map_location="cpu", weights_only=False)
        evidence.update(checkpoint_sha256=sha256_file(last_path), committed_episodes=committed["episode_count"],
                        committed_update=committed["policy_version"])
        if path.name == "training_history.json":
            # last.pt is authoritative even when JSON writing was interrupted.
            records = committed["history"]
            evidence["plotted_source"] = "committed_checkpoint_history"
            return records, evidence
    else:
        committed = None
    records = (read(path) if path.suffix == ".json" else
               [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    if committed is not None:
        records = [r for r in records if r.get("episodes", 0) <= committed["episode_count"] and
                   r.get("update", 0) <= committed["policy_version"]]
    evidence["plotted_source"] = "log_with_committed_boundary" if committed is not None else "recorded_log"
    return records, evidence


def _training_figures(training_root, destination):
    """Plot numeric logs actually present; never synthesize learning curves."""
    if training_root is None:
        return [], []
    import matplotlib.pyplot as plt
    root = Path(training_root)
    frames, evidence = [], []
    paths = (sorted(root.glob("**/metrics.jsonl")) + sorted(root.glob("**/training_metrics.jsonl"))
             + sorted(root.glob("**/training_history.json")))
    for path in paths:
        records, source = _committed_training_records(path)
        if records:
            frame = pd.DataFrame(records)
            frame["run"] = str(path.parent.relative_to(root))
            frames.append(frame)
            evidence.append({**source, "records": len(records)})
    if not frames:
        return [], evidence
    logs = pd.concat(frames, ignore_index=True)
    logs.to_csv(destination.parent / "training_metrics.csv", index=False)
    possible = [("mean_return", "episode_return", "mean_raw_return"), ("actor_loss", "policy_loss"),
                ("critic_loss", "value_loss"), ("approx_kl", "kl"), ("clip_fraction",), ("lstm_loss", "return_prediction_loss")]
    fields = [next((c for c in choices if c in logs and pd.to_numeric(logs[c], errors="coerce").notna().any()), None) for choices in possible]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for ax, field in zip(axes.flat, fields):
        if field is None:
            ax.text(.5, .5, "Metric not recorded", ha="center", transform=ax.transAxes)
            continue
        for name, group in logs.groupby("run"):
            x = group["episodes"] if "episodes" in group else np.arange(len(group))
            ax.plot(x, pd.to_numeric(group[field], errors="coerce"), label=name)
        ax.set(xlabel="Completed episodes / update index", ylabel=field)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(destination / "04_training_metrics.png", dpi=150)
    plt.close(fig)
    return ["04_training_metrics.png"], evidence


def _training_records(training_root, candidates=()):
    """Cross-check human-readable logs against the actual committed checkpoints."""
    import torch
    from .config import VARIANTS as EXPERIMENTS
    records = []
    if training_root is None:
        return records
    for path in sorted(Path(training_root).glob("**/training_status.json")):
        status = read(path)
        config = read(path.parent / "config.json")
        selection = read(path.parent / "selection.json") if (path.parent / "selection.json").exists() else {}
        errors = []
        evidence = {}
        try:
            last_path, best_path = path.parent / "last.pt", path.parent / "best.pt"
            last = torch.load(last_path, map_location="cpu", weights_only=False)
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            canonical = lambda x: json.dumps(x, sort_keys=True)
            resume_mutable = {"episodes", "workers", "max_seconds"}
            best_immutable = {k: v for k, v in best["config"].items() if k not in resume_mutable}
            current_immutable = {k: v for k, v in config.items() if k not in resume_mutable}
            if canonical(config) != canonical(last["config"]) or canonical(current_immutable) != canonical(best_immutable):
                errors.append("configuration differs from saved model")
            if config.get("smoke") or last["provenance"].get("smoke"):
                errors.append("synthetic smoke is not a formal training run")
            if last["provenance"] != read(path.parent / "provenance.json"):
                errors.append("training provenance file differs from checkpoint")
            if (status["completed_episodes"] != last["episode_count"] or
                    status["environment_steps"] != last["episode_count"] * 72 or
                    status["ppo_updates"] != last["policy_version"] or
                    status["training_seed"] != last["training_seed"] or
                    status["algorithm"] != EXPERIMENTS[config["variant"]][0]):
                errors.append("training counters or identity differ from committed checkpoint")
            if status.get("training_requests") != sum(h["requests"] for h in last["history"]):
                errors.append("training request count differs from committed history")
            if (selection.get("checkpoint_sha256") != sha256_file(best_path) or
                    selection.get("test_used") is not False or
                    selection.get("episodes") != best["episode_count"] or
                    selection.get("update") != best["policy_version"]):
                errors.append("selected checkpoint does not match validation selection")
            matching = [c for c in candidates if c["algorithm"] == status["algorithm"] and
                        int(c.get("training_seed", 0)) == int(status["training_seed"])]
            if len(matching) != 1:
                errors.append("no unique evaluated candidate for this training run")
            else:
                evaluated_path = matching[0]["spec"]["kwargs"]["checkpoint"]
                if sha256_file(evaluated_path) != selection.get("checkpoint_sha256"):
                    errors.append("evaluated model is not the selected checkpoint")
            if not status.get("complete") or not status.get("validated"):
                errors.append("training or validation incomplete")
            evidence = {"last_checkpoint_sha256": sha256_file(last_path),
                        "config_sha256": sha256_file(path.parent / "config.json"),
                        "provenance_sha256": sha256_file(path.parent / "provenance.json"),
                        "selection_sha256": sha256_file(path.parent / "selection.json"),
                        "actor_critic_parameters": sum(t.numel() for t in last.get("model_state", {}).values()) or None,
                        "lstm_parameters_stored": sum(t.numel() for t in last.get("lstm_state", {}).values()) or None,
                        "lstm_parameters_active": (sum(t.numel() for t in last.get("lstm_state", {}).values()) or None)
                            if EXPERIMENTS[config["variant"]][2] else 0,
                        "sampling_seconds": sum(h.get("sampling_seconds", 0.) for h in last["history"]),
                        "ppo_seconds": sum(h.get("ppo_seconds", 0.) for h in last["history"]),
                        "lstm_seconds": sum(h.get("lstm_seconds", 0.) for h in last["history"]),
                        "validation_seconds": sum(h.get("seconds", 0.) for h in last.get("validation_history", [])),
                        "validation_episodes": sum(h.get("episodes_evaluated", 0) for h in last.get("validation_history", []))}
        except (KeyError, OSError, ValueError, TypeError) as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))
        records.append({**status, "variant": config["variant"], "batch_episodes": config["batch_episodes"],
            "workers": config["workers"], "gamma": config["ppo"]["gamma"], "gae_lambda": config["ppo"]["gae_lambda"],
            "learning_rate": config["learning_rate"], "selected_at_episode": selection.get("episodes"),
            "config_path": str((path.parent / "config.json").resolve()),
            "checkpoint_sha256": selection.get("checkpoint_sha256"),
            "training_status_sha256": sha256_file(path), "verified_config": config,
            "training_evidence_verified": not errors, "training_evidence_errors": errors, **evidence})
    return records


def _ablation_status(left, right, records, candidates):
    """Require equal committed interaction budgets and paired training seeds."""
    seeds = {a: {int(c.get("training_seed", 0)) for c in candidates if c["algorithm"] == a}
             for a in (left, right)}
    if not seeds[left] or seeds[left] != seeds[right]:
        return "训练seed不匹配"
    grouped = {a: [r for r in records if r["algorithm"] == a] for a in (left, right)}
    for a in (left, right):
        if (len(grouped[a]) != len(seeds[a]) or
                {int(r["training_seed"]) for r in grouped[a]} != seeds[a] or
                not all(r.get("training_evidence_verified") for r in grouped[a])):
            return "训练来源未核验"
    matched_fields = ("train_days", "validation_days", "test_days", "validation_roles", "scenarios",
                      "batch_episodes", "hidden", "learning_rate", "ppo", "budget_mode", "budget_range",
                      "opponent_randomization", "initial_alpha", "validate_every", "early_stopping_patience")
    for seed in seeds[left]:
        before = next(r for r in grouped[left] if int(r["training_seed"]) == seed)
        after = next(r for r in grouped[right] if int(r["training_seed"]) == seed)
        if (before["completed_episodes"] != after["completed_episodes"] or
                before["environment_steps"] != after["environment_steps"]):
            return "训练预算不匹配"
        for field in matched_fields:
            a, b = before["verified_config"].get(field), after["verified_config"].get(field)
            if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                return "非消融配置不匹配"
    return "matched"


def _reference_slots(slots):
    """Read only the identity-selected case from existing sealed Baseline traces."""
    if slots.empty:
        return slots
    chosen = slots.sort_values(["scenario", "day_id", "player_id", "seed", "training_seed"]).iloc[0]
    columns = ["algorithm", "scenario", "day_id", "player_id", "seed", "training_seed", "time_slot",
               "alpha", "cumulative_cost", "cumulative_value", "remaining_budget"]
    reference = []
    for chunk in pd.read_csv(BASELINE_ROOT / "experiment_B/slots.csv", usecols=columns,
                             dtype={"player_id": str, "day_id": str}, chunksize=65536):
        found = chunk[(chunk.algorithm == "LocalPID") & (chunk.scenario == chosen.scenario) &
                      (chunk.day_id == chosen.day_id) & (chunk.player_id == chosen.player_id) &
                      (chunk.seed == chosen.seed)]
        if len(found):
            reference.append(found)
    return pd.concat([slots] + reference, ignore_index=True) if reference else slots


def _diagnostic_evidence(root, destination, skip_variants=()):
    """Include only actually generated diagnostics; keep estimates labelled."""
    if root is None or not Path(root).is_dir():
        return [], [], []
    records, figures, sources = [], [], []
    for path in sorted(Path(root).glob("**/diagnostics.json")):
        value = read(path)
        if value["config"]["variant"] in skip_variants:
            continue
        if (value.get("policy_frozen") is not True or value.get("diagnostics_only") is not True or
                sha256_file(value["checkpoint"]) != value["checkpoint_sha256"]):
            raise ValueError("Budget diagnostic is not linked to its unchanged frozen checkpoint")
        cf = value.get("counterfactual") or {}
        critic = cf.get("critic") or {}
        redistributed = value.get("redistribution") or {}
        records.append(dict(variant=value["config"]["variant"], checkpoint=value["checkpoint"],
            total_value=value["metrics"]["total_value"], player_id=value["metrics"]["player_id"],
            scenario=value["metrics"]["scenario"], time_slot=cf.get("time_slot"),
            predicted_marginal=critic.get("marginal_estimate"), empirical_marginal=cf.get("empirical_marginal_value"),
            raw_value_interpretation=critic.get("raw_value_interpretation"),
            return_conservation_error=redistributed.get("return_conservation_error")))
        if redistributed:
            raw = np.asarray(value["raw_rewards"], dtype=float)
            moved = np.asarray(redistributed["redistributed_rewards"], dtype=float)
            predicted = np.asarray(redistributed["prefix_predictions"], dtype=float)
            positive = np.flatnonzero(raw > 0)
            records[-1].update(prefix_at_8=float(predicted[7]),
                middle_prefix_min=float(predicted[8:71].min()), middle_prefix_max=float(predicted[8:71].max()),
                raw_value_first_8=float(raw[:8].sum()), redistributed_first_8=float(moved[:8].sum()),
                corrected_last_reward=float(moved[-1]),
                first_positive_raw_slot=int(positive[0]+1) if len(positive) else None)
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        for name in ("budget_behavior.png", "reward_redistribution.png"):
            source = path.parent / name
            if source.exists():
                target_name = f"diagnostic_{value['config']['variant']}_{len(records)}_{name}"
                shutil.copy2(source, destination / target_name)
                figures.append(target_name)
    for path in sorted(Path(root).glob("**/market_distribution_audit.json")):
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        source = path.parent / "market_distributions.png"
        if source.exists():
            target_name = f"market_distribution_{len(figures)}.png"
            shutil.copy2(source, destination / target_name)
            figures.append(target_name)
    return records, figures, sources


def _system_evidence(root, output):
    """Capture exact observed resource/test/profile inputs, including live logs."""
    root = Path(root)
    sources, values = [], {}
    target = output / "evidence"
    target.mkdir(exist_ok=True)
    for name in ("acceptance_tests.log", "final_rl_tests.log", "final_rl_tests_latest.log", "core_profile.json", "resources.jsonl"):
        path = root / name
        if not path.exists():
            continue
        data = path.read_bytes()
        (target / name).write_bytes(data)
        sources.append(dict(path=str(path.resolve()), snapshot=str((target/name).resolve()),
                            sha256=hashlib.sha256(data).hexdigest()))
        if name in ("acceptance_tests.log", "final_rl_tests.log", "final_rl_tests_latest.log"):
            values["acceptance" if name == "acceptance_tests.log" else "final_rl_tests"] = data.decode("utf-8-sig").strip()
        elif name == "core_profile.json":
            values["profile"] = json.loads(data.decode("utf-8-sig"))
        else:
            records = []
            for line in data.decode("utf-8-sig").splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # A live final partial line is not a complete measurement.
            if records:
                values["resources"] = dict(samples=len(records),
                    observed_peak_process_rss_sum_mb=max(r["total_rss_mb"] for r in records),
                    mean_system_cpu_percent=float(np.mean([r["system_cpu_percent"] for r in records])),
                    sampled_seconds=max(r["elapsed_seconds"] for r in records),
                    interpretation="Observed samples, not an exact continuous peak; shared memory may be counted in multiple process RSS values")
    market = root / "market_audit/market_distribution_audit.json"
    if market.exists():
        values["markets"] = read(market)
        sources.append(dict(path=str(market.resolve()), sha256=sha256_file(market)))
        source = market.parent / "market_distributions.png"
        if source.exists():
            shutil.copy2(source, output / "figures/market_distribution_audit.png")
    return values, sources


def _prefix_evidence(root, training_records, destination):
    summaries, sources, figures = [], [], []
    if not Path(root).is_dir():
        return summaries, sources, figures
    for path in sorted(Path(root).glob("**/prefix_validation.json")):
        value = read(path)
        matches = [r for r in training_records if r.get("checkpoint_sha256") == value.get("checkpoint_sha256")]
        if len(matches) != 1 or not matches[0].get("training_evidence_verified"):
            raise ValueError("Prefix diagnostic is not linked to a verified selected training checkpoint")
        run = matches[0]
        cfg = run["verified_config"]
        if (value.get("data_split") != "validation" or value.get("actor_updated") is not False or
                value.get("lstm_updated") is not False or value["checkpoint_sha256"] != sha256_file(value["checkpoint"])):
            raise ValueError("Prefix diagnostic changed model or used the wrong split")
        expected = {(d, str(p), s) for d in cfg["validation_days"] for p in cfg["validation_roles"] for s in cfg["scenarios"]}
        actual = {(r["market_id"], str(r["player_id"]), r["scenario"]) for r in value["records"]}
        if len(actual) != len(value["records"]) or not actual <= expected:
            raise ValueError("Prefix diagnostic contains duplicated or unexpected validation cases")
        complete = actual == expected
        if bool(value["complete"]) != complete or value["expected"] != len(expected) or value["completed"] != len(actual):
            raise ValueError("Prefix diagnostic completion differs from exact validation coverage")
        row = dict(variant=run["variant"], training_seed=run["training_seed"], completed=len(actual),
                   expected=len(expected), status="completed" if complete else "未完成", data_split="validation",
                   checkpoint_sha256=value["checkpoint_sha256"])
        if value["records"]:
            error = np.asarray([np.asarray(r["prefix_predictions"], dtype=float) - r["realized_return"] for r in value["records"]])
            if error.shape != (len(actual), 72) or not np.isfinite(error).all():
                raise ValueError("Invalid causal prefix prediction data")
            mse, bias = np.mean(error**2, axis=0), np.mean(error, axis=0)
            if not np.allclose(mse, value["mse_by_prefix"]) or not np.allclose(bias, value["bias_by_prefix"]):
                raise ValueError("Prefix error summary does not match raw validation records")
            for prefix in (1, 36, 72):
                row[f"mse_{prefix}"] = float(mse[prefix-1]) if complete else None
                row[f"bias_{prefix}"] = float(bias[prefix-1]) if complete else None
        summaries.append(row)
        sources.append(dict(path=str(path.resolve()), sha256=sha256_file(path)))
        image = path.parent / "prefix_validation.png"
        if image.exists():
            name = f"prefix_validation_{run['variant']}_{run['training_seed']}.png"
            shutil.copy2(image, destination / name)
            figures.append(name)
    return summaries, sources, figures


def generate_report(evaluation_dir, *, training_root=None, output=None, diagnostics_root=None):
    evaluation_dir = Path(evaluation_dir).resolve()
    output = Path(output).resolve() if output else evaluation_dir / "report"
    output.mkdir(parents=True, exist_ok=True)
    protocol, status = read(evaluation_dir / "protocol.json"), read(evaluation_dir / "status.json")
    for name, field in [("episode_results.csv", "result_sha256"), ("slots.csv", "slots_sha256"), ("protocol.json", "protocol_sha256")]:
        if sha256_file(evaluation_dir / name) != status[field]:
            raise ValueError("Evaluation evidence changed: " + name)
    dtype = {"player_id": str, "player_type": str, "day_id": str}
    frame = pd.read_csv(evaluation_dir / "episode_results.csv", dtype=dtype) if status["completed_episodes"] else pd.DataFrame(columns=KEYS)
    valid, coverage = complete_algorithms(frame, protocol)
    baseline, _, audit = audit_baseline()
    baseline = baseline[baseline.day_id.isin(protocol["days"]) & baseline.player_id.isin(protocol["roles"]) &
                        baseline.scenario.isin(protocol["scenarios"]) & baseline.seed.isin(protocol["seeds"])]
    if baseline.empty:
        raise ValueError("No matching Baseline result for this evaluation protocol")
    complete = pd.concat([baseline, frame[frame.algorithm.isin(valid)]], ignore_index=True)
    _verify_pair_conditions(complete)
    tables = aggregate_phase(complete)
    runtime_segments = [read(p) for p in sorted(evaluation_dir.glob("runtime_*.json"))]
    if runtime_segments:
        pd.DataFrame([{k: row.get(k) for k in ("seconds", "workers", "max_seconds", "resumed_episodes",
            "new_completed", "completed_episodes", "expected_episodes", "deadline_reached")}
            for row in runtime_segments]).to_csv(output / "evaluation_runtime_segments.csv", index=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    seed_cells = complete.groupby(["algorithm", "scenario", "player_id", "seed"], as_index=False).total_value.mean()
    seed_means = seed_cells.groupby(["algorithm", "seed"], as_index=False).total_value.mean()
    seed_means.groupby("algorithm").total_value.agg(["count", "mean", "std", "min", "max"]).to_csv(
        output / "evaluation_seed_statistics.csv")
    coverage.to_csv(output / "ppo_completeness.csv", index=False)
    gains = paired_gains(complete)
    gains.to_csv(output / "paired_baseline_gains.csv", index=False)
    scenario = tables["per_scenario"].copy()
    scenario["rank_V"] = scenario.groupby("scenario").mean_V.rank(method="min", ascending=False)
    scenario.to_csv(output / "scenario_ranks.csv", index=False)
    summary = scenario.pivot(index="algorithm", columns="scenario", values="mean_V").reindex(columns=SCENARIOS)
    overall = tables["overall"].set_index("algorithm")
    summary["mean_V"] = overall.mean_V
    summary["mean_R"] = overall.mean_R
    summary = summary.reindex(list(ALGORITHMS) + list(VARIANTS) + [a for a in valid if a not in VARIANTS])
    summary["status"] = ["completed" if a in set(complete.algorithm) else "未完成" for a in summary.index]
    summary.to_csv(output / "main_comparison.csv")
    slots = pd.read_csv(evaluation_dir / "slots.csv", dtype=dtype) if status["completed_episodes"] else pd.DataFrame()
    files = _plots(tables, _reference_slots(slots), output / "figures")
    more, training_evidence = _training_figures(training_root, output / "figures")
    files += more
    training_records = _training_records(training_root, protocol["candidates"])
    pd.DataFrame(training_records).to_csv(output / "training_runs.csv", index=False)
    artifact_root = evaluation_dir.parent.parent
    if not any((artifact_root / name).exists() for name in ("resources.jsonl", "market_audit", "acceptance_tests.log")):
        artifact_root = evaluation_dir.parent
    if diagnostics_root is None:
        diagnostics_root = artifact_root / "diagnostics"
    diagnostic_records, diagnostic_figures, diagnostic_sources = _diagnostic_evidence(diagnostics_root, output / "figures")
    fallback = evaluation_dir.parent / "diagnostics"
    if fallback.resolve() != Path(diagnostics_root).resolve() and fallback.is_dir():
        existing = {r["variant"] for r in diagnostic_records}
        extra_records, extra_figures, extra_sources = _diagnostic_evidence(fallback, output / "figures", existing)
        diagnostic_records += extra_records
        diagnostic_figures += extra_figures
        diagnostic_sources += extra_sources
    files += diagnostic_figures
    pd.DataFrame(diagnostic_records).to_csv(output / "diagnostic_summary.csv", index=False)
    system_evidence, system_sources = _system_evidence(artifact_root, output)
    prefix_records, prefix_sources, prefix_figures = _prefix_evidence(diagnostics_root, training_records, output / "figures")
    files += prefix_figures
    pd.DataFrame(prefix_records).to_csv(output / "prefix_validation_summary.csv", index=False)
    if (output / "figures/market_distribution_audit.png").exists():
        files.append("market_distribution_audit.png")
    full = (set(protocol["roles"]) == set(ROLES) and set(protocol["scenarios"]) == set(SCENARIOS)
            and set(protocol["seeds"]) == set(SEEDS))
    training_seeds = {a: sorted({int(c.get("training_seed", 0)) for c in protocol["candidates"] if c["algorithm"] == a})
                      for a in {c["algorithm"] for c in protocol["candidates"]}}
    best_baseline = overall.loc[list(ALGORITHMS)].sort_values("mean_V", ascending=False).iloc[0]
    best_name = best_baseline.name
    lines = ["# PPO + LSTM 统一实验报告", "",
             f"本报告只使用实际已提交结果。PPO 本次完成 {status['completed_episodes']}/{status['expected_episodes']} 局；"
             f"复用经文件、模型、代码、市场、预算和对手核验的十种 Baseline，原完整结果为 {audit['episodes']} 局。",
             "", f"[完整系统设计与一键命令]({(Path(__file__).parent/'README.md').as_posix()})；"
             f"[十种Baseline、规则与资产审计]({(Path(__file__).parent/'AUDIT.md').as_posix()})；"
             f"[算法与评测代码审查]({(Path(__file__).parent/'REVIEW.md').as_posix()})。",
             "", "评测口径：单局只替换一名广告主，72片完整周期，原始不同角色预算，累计 pvalue 为唯一目标；"
             "五场景采用相同动态对手及分配种子。先平均每角色的训练seed、评测seed和日期，再等权平均角色。"
             "R 分母为本轮五场景联合调优的 LocalPID 同角色同场景均值；零分母保持未定义。", "",
             f"覆盖：{'完整48角色×5场景×2评测seed' if full else '所声明子集，不能声称全48角色正式比较'}。"
             f"日期={protocol['days']}。period-24 是已展示过的可比性测试；不声称新的盲测。", "",
             "| 算法 | Fixed V | Local PID V | Mixed V | AuctionNet PID V | Mixed2 V | 平均 V | 平均 R | 状态 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|" ]
    for algorithm, row in summary.iterrows():
        cells = ["未完成" if pd.isna(row[c]) else f"{row[c]:.4f}" for c in list(SCENARIOS) + ["mean_V", "mean_R"]]
        lines.append("| " + " | ".join([algorithm] + cells + [row.status]) + " |")
    lines += ["", "PPO 训练seed：" + "；".join(f"{a}={s}" for a, s in sorted(training_seeds.items())) + "。"]
    if any(len(s) < 2 for s in training_seeds.values()):
        lines += ["本轮至少部分 PPO 只有1个训练seed，而神经 Baseline 保留2个训练seed的均值。"
                  "两个评测seed主要改变平局与混合对手分配，不能替代独立训练重复；本轮不能据此声称 LSTM 或增强带来稳定训练收益。"]
    lines += ["评测seed均值、样本标准差与范围另存 evaluation_seed_statistics.csv。只有两个随机种子且共用同一测试周期，"
              "不据此构造跨市场泛化置信区间。"]
    if runtime_segments:
        lines += [f"冻结评测共记录{len(runtime_segments)}个执行段，累计墙钟 {sum(r['seconds'] for r in runtime_segments):.1f} 秒；"
                  "各段时限、复用局数和新增局数见 evaluation_runtime_segments.csv。墙钟包含等待及系统暂停，不等同活跃CPU计算时间。"]
    if training_records:
        lines += ["", "实际训练与验证选择：", "",
                  "| 算法/训练seed | 完整轨迹 | 环境step | 请求数 | PPO更新 | LSTM版本 | 秒数 | 选模episode | 验证V |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for record in training_records:
            score = record.get("best_validation_value")
            score = "未验证" if score is None else f"{score:.4f}"
            lines.append(f"| {record['algorithm']}/{record['training_seed']} | {record['completed_episodes']} | "
                f"{record['environment_steps']} | {record['training_requests']} | {record['ppo_updates']} | "
                f"{record['lstm_versions']} | {record['elapsed_seconds']:.1f} | {record.get('selected_at_episode')} | {score} |")
        lines += ["", "各run的最终配置、选中Checkpoint路径与SHA保存在 training_runs.csv；停止原因和实际完成数量保留，"
                  "不把请求上限当作已完成训练。验证按period-21预先声明的角色/场景网格的原始V均值选择；测试不回流选模。"
                  "详细状态、动作概率、GAE、归一化、LSTM版本和恢复语义见项目 harmony_rl/README.md。"]
        earlier = [r for r in training_records if r.get("selected_at_episode") is not None and
                   r["selected_at_episode"] < r["completed_episodes"]]
        if earlier:
            lines += ["相同交互预算按总完成训练轨迹衡量；部署Checkpoint可能较早被Validation选中。实际为：" +
                      "；".join(f"{r['variant']}选中{r['selected_at_episode']}ep、共训练{r['completed_episodes']}ep" for r in earlier) +
                      "。因此同预算不等于所选权重经历相同更新次数，此区别保留在上表。"]
        lines += ["", "模型规模与实际计算时间分项（参数量直接由Checkpoint张量numel求和）：", "",
                  "| 算法/训练seed | Actor+Critic参数 | 启用LSTM参数 | 采样秒 | PPO秒 | LSTM秒 | 验证秒 | 验证局 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for record in training_records:
            fmt = lambda value: "未记录" if value is None else f"{value:.2f}"
            lines.append(f"| {record['algorithm']}/{record['training_seed']} | {record.get('actor_critic_parameters')} | "
                f"{record.get('lstm_parameters_active')} | {fmt(record.get('sampling_seconds'))} | "
                f"{fmt(record.get('ppo_seconds'))} | {fmt(record.get('lstm_seconds'))} | "
                f"{fmt(record.get('validation_seconds'))} | {record.get('validation_episodes')} |")
        lines += ["", "分项是日志中的墙钟计时；并发采样不把worker时间相加冒称墙钟。"
                  "分项不覆盖所有进程启动、序列化与Checkpoint I/O，因此不要求与总耗时相等。"
                  "Critic/Actor/LSTM loss仅报告实际值与曲线，不据下降趋势声称收敛。"]
    lines += ["", f"当前可比 Baseline 中平均 V 最高为 **{best_name}**（{best_baseline.mean_V:.4f}）。"]
    if valid:
        winner = overall.loc[valid].sort_values("mean_V", ascending=False).iloc[0]
        delta = winner.mean_V / best_baseline.mean_V - 1 if best_baseline.mean_V > 0 else np.nan
        winner_scenes = scenario[scenario.algorithm == winner.name].sort_values("mean_V", ascending=False)
        lines += [f"已完整评测的 PPO 变体中平均 V 最高为 **{winner.name}**（{winner.mean_V:.4f}），"
                  f"相对最佳 Baseline 的整体均值变化为 {delta:+.2%}。这不是按角色配对百分比的均值；后者保存在 paired_baseline_gains.csv。",
                  f"该变体最好场景为 **{winner_scenes.iloc[0].scenario}**（V={winner.best_scenario_V:.4f}），"
                  f"最差场景为 **{winner_scenes.iloc[-1].scenario}**（V={winner.worst_scenario_V:.4f}）。"]
    else:
        lines += ["尚无达到声明全覆盖的 PPO 变体，不能回答 PPO 是否超过最强 Baseline。"]
    lines += ["", "消融只比较已完成、训练来源已核验且训练交互预算相同的实际模型。"
              "每项核对选中/已提交Checkpoint、有效配置、相同训练seed、completed_episodes和environment_steps；"
              "不相符时不输出消融差。以下通过核验的差值仍是描述性测试差异，不能证明稳定因果收益：", "",
              "| 比较 | 平均 V 差（后者−前者） |", "|---|---:|"]
    ablations = []
    for left, right in [("PPO-Raw", "PPO-Aug"), ("PPO-Raw", "PPO-LSTM"), ("PPO-Aug", "PPO-LSTM-Aug"),
                        ("PPO-Raw", "Residual-PPO"), ("Residual-PPO", "Residual-PPO-LSTM")]:
        check = _ablation_status(left, right, training_records, protocol["candidates"]) if left in valid and right in valid else "未完成"
        difference = float(overall.loc[right, 'mean_V']-overall.loc[left, 'mean_V']) if check == "matched" else None
        value = f"{difference:+.4f}" if difference is not None else check
        ablations.append(dict(before=left, after=right, status=check, mean_value_difference=difference))
        lines.append(f"| {left} → {right} | {value} |")
    pd.DataFrame(ablations).to_csv(output / "verified_ablations.csv", index=False)
    lines += ["", "LSTM 预测变化只代表前缀回报预测差，不能解释为动作的真实因果贡献。"
              "机会成本有限差分和反事实样本需要单独查看诊断证据；平均测试价值不足以证明早期预算决策改善。",
              "未完成的独立日期测试、更多训练seed、收敛训练和长期鲁棒性实验不填估计值。"
              "正式比赛方案选择应结合验证集、预算分配曲线与部署耗时，不能用本报告测试分数继续调参。", ""]
    if diagnostic_records:
        lines += ["实际预算干预诊断：", "",
                  "| 变体 | 场景/角色 | Critic预算敏感性 | 模拟新增价值/新增预算 | 可解释为原始未来价值 | 回报守恒误差 |",
                  "|---|---|---:|---:|---|---:|"]
        for record in diagnostic_records:
            fmt = lambda v: "未完成" if v is None else f"{v:.6g}"
            conservation = "不适用（Raw）" if record["raw_value_interpretation"] else fmt(record['return_conservation_error'])
            lines.append(f"| {record['variant']} | {record['scenario']}/{record['player_id']} | "
                f"{fmt(record['predicted_marginal'])} | {fmt(record['empirical_marginal'])} | "
                f"{'是，仍为估计' if record['raw_value_interpretation'] else '否，重分配return敏感性'} | "
                f"{conservation} |")
        lines += ["", "上述是指定角色/市场/seed的单次预算干预，不能证明稳定机会成本准确性；"
                  "完整源记录及散列保存于 report_manifest.json。LSTM行的Critic列是重分配return敏感性，"
                  "模拟列是原始未来价值增量，二者不是同一个物理量，不能计算为预测误差或准确率；"
                  "只有Raw Critic行作同量比较。", ""]
        raw_diagnostics = [r for r in diagnostic_records if r["raw_value_interpretation"] and r["predicted_marginal"]]
        for record in raw_diagnostics:
            if record["empirical_marginal"] is not None:
                ratio = record["empirical_marginal"] / record["predicted_marginal"]
                lines += [f"{record['variant']} 的该次模拟边际/预测边际={ratio:.3g}；预测偏差需结合符号和样本解释。"
                          "这种未经验证的敏感性不能作为reward惩罚使用。", ""]
        for record in diagnostic_records:
            if record["variant"] == "E3" and "prefix_at_8" in record:
                lines += [f"E3的这个训练市场诊断样本最终原始V={record['total_value']:.4f}；"
                    f"前8片LSTM预测达到{record['prefix_at_8']:.4f}，第9–71片预测范围"
                    f"[{record['middle_prefix_min']:.4f}, {record['middle_prefix_max']:.4f}]。"
                    f"前8片原始价值合计{record['raw_value_first_8']:.4f}，重分配奖励合计{record['redistributed_first_8']:.4f}，"
                    f"最后一片校正奖励为{record['corrected_last_reward']:.4f}；第一片实际正价值出现在片{record['first_positive_raw_slot']}。"
                    "在这个样本中，前缀预测早期上升后趋于平台，奖励集中到前段和终点校正；"
                    "这更像回报基线与reward形状的学习，不能证明早期预算决策改善，也不能据单样本作因果结论。", ""]
    else:
        lines += ["机会成本/反事实诊断尚未导入本报告，不依据训练或测试均值推断其解释能力。", ""]
    if prefix_records:
        lines += ["冻结LSTM的Validation前缀预测检查：", "",
                  "| 变体/训练seed | 验证局完成 | 前缀1 MSE | 前缀36 MSE | 前缀72 MSE | 状态 |",
                  "|---|---:|---:|---:|---:|---|"]
        for record in prefix_records:
            fmt = lambda v: "未完成" if v is None else f"{v:.5g}"
            lines.append(f"| {record['variant']}/{record['training_seed']} | {record['completed']}/{record['expected']} | "
                f"{fmt(record.get('mse_1'))} | {fmt(record.get('mse_36'))} | {fmt(record.get('mse_72'))} | {record['status']} |")
        lines += ["", "这里Actor已由Validation选择，故不是完全未参与模型选择的独立泛化测试；"
                  "LSTM参数固定，Validation return未用于拟合，Actor也未更新。每项核验原模型SHA、"
                  "6角色×5场景的完整验证覆盖，并从逐局预测重算MSE/bias。单训练seed不能证明稳定泛化；"
                  "bias与全部前缀曲线见 prefix_validation_summary.csv、图和源JSON。", ""]
    if system_evidence:
        lines += ["执行和验证证据：", ""]
        if "acceptance" in system_evidence:
            lines += ["```text", system_evidence["acceptance"], "```", "",
                      "上述为本轮保存的测试输出快照，不能用旧项目通过数量替代本轮验收。"]
        if "final_rl_tests" in system_evidence:
            lines += ["", "后续RL专属回归测试：", "", "```text", system_evidence["final_rl_tests"], "```", "",
                      "两份测试记录有重叠，不将通过数量相加冒称独立测试总数。"]
        profile = system_evidence.get("profile")
        if profile:
            lines += [f"统一核心profile：{profile['requests']:,}请求，完整episode {profile['episode']['elapsed_seconds']:.3f}秒，"
                      f"{profile['requests_per_second']:,.0f}请求/秒；数据加载 {profile['load_seconds']:.3f}秒。"
                      "这是一次带profile开销的测量，不能替代整批训练吞吐。"]
        resources = system_evidence.get("resources")
        if resources:
            lines += [f"资源日志 {resources['samples']}个完整采样，覆盖约{resources['sampled_seconds']:.1f}秒；"
                      f"采样进程RSS总和最高 {resources['observed_peak_process_rss_sum_mb']:.1f} MiB，"
                      f"系统CPU采样均值 {resources['mean_system_cpu_percent']:.1f}%。RSS会重复计算共享页，"
                      "这不是整机实际内存峰值，也不是连续监测保证。运行环境为CPU训练。"]
        markets = system_evidence.get("markets")
        if markets:
            lines += ["", "训练市场增强的实际统计（完整参与关系、分位数和共现矩阵见原JSON）：", "",
                      "| 市场 | 请求数 | pvalue均值 | 生效总预算/请求 |", "|---|---:|---:|---:|"]
            for label, record in markets["records"].items():
                ratio = record.get("effective_budget_per_request", record.get("budget_per_request"))
                lines.append(f"| {label} | {record['request_count']} | {record['pvalue_mean']:.6g} | {ratio:.6g} |")
        lines += ["", "资源与测试快照位于本报告 evidence/；全部源路径及SHA列于 report_manifest.json。", ""]
    lines += ["对十个研究问题的回答：", ""]
    if valid:
        better = winner.mean_V > best_baseline.mean_V
        lines += [f"1. 已完成PPO中平均V最佳：{winner.name}。",
                  f"2. {'已在本轮比较均值上超过' if better else '没有超过'}最强Baseline {best_name}；不外推为收敛或独立日期结论。",
                  f"3. 最佳PPO的最好场景：{winner_scenes.iloc[0].scenario}。",
                  f"4. 最佳PPO的最差场景：{winner_scenes.iloc[-1].scenario}。"]
    else:
        lines += ["1–4. PPO完整覆盖不足，最佳变体、是否超过Baseline及最好/最差场景均未完成。"]
    lines += ["5. LSTM是否有稳定收益：单训练seed不能证明。上表给出实际消融差；reward形状改变或回报守恒本身不能证明早期预算决策改善。",
              "6. 增强是否提高鲁棒性：五场景描述性差异见消融和最差场景表；独立测试市场、压力测试泛化及更多训练seed验证尚未完成。"]
    if ("PPO-Raw" in valid and "Residual-PPO" in valid and
            _ablation_status("PPO-Raw", "Residual-PPO", training_records, protocol["candidates"]) == "matched"):
        residual_delta = overall.loc["Residual-PPO", "mean_V"]-overall.loc["PPO-Raw", "mean_V"]
        lines += [f"7. Residual相对直接PPO本轮平均V差={residual_delta:+.4f}；是否普遍更优尚不能确定。"]
    else:
        lines += ["7. Residual与直接PPO的等预算、来源可追溯消融比较尚未通过完整核验，不能解释为方法效果。"]
    lines += ["8. 机会成本指标：若上文有反事实记录，仅可据该具体样本检查敏感性；偏差大的预测不能解释为真实预算价值，更不能直接作惩罚。",
              "9. 当前主要限制：每变体仅有限完整市场轨迹、角色访问次数少、单训练seed，Critic与前缀回报预测可能未充分拟合；未进行收敛或跨独立市场证据验证。"]
    if valid and winner.mean_V <= best_baseline.mean_V:
        lines += [f"10. 正式比赛有限训练时间内，本轮证据支持优先保留现有冻结的 {best_name} Baseline；"
                  "PPO/LSTM继续作为研究候选，不因理论复杂度替换实测更强方法。"]
    elif valid:
        lines += [f"10. {winner.name}可作为下一轮验证候选；先补独立日期、训练seed和部署资源验证，再决定是否替换现有Baseline。"]
    else:
        lines += ["10. 目前保留现有已验证Baseline，PPO评测完整后再比较。"]
    lines += [""]
    for name in files:
        lines += [f"![{name}](figures/{name})", ""]
    lines += ["详细可复现证据：main_comparison.csv、per_role.csv、per_evaluation_seed.csv、per_training_seed.csv、"
              "scenario_ranks.csv、paired_baseline_gains.csv、ppo_completeness.csv 和父目录 protocol.json/status.json。"]
    report = output / "research_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"status": "completed", "report": str(report), "full_evaluation_scope": full,
              "complete_ppo_algorithms": valid, "missing_ppo_algorithms": [a for a in VARIANTS if a not in valid],
              "ppo_training_seeds": training_seeds,
              "baseline_reuse": audit, "training_log_evidence": training_evidence,
              "training_runs": training_records, "diagnostic_sources": diagnostic_sources,
              "verified_ablations": ablations,
              "system_evidence": system_evidence, "system_sources": system_sources,
              "prefix_validation": prefix_records, "prefix_sources": prefix_sources,
              "figures": files, "sources": {str(evaluation_dir / n): sha256_file(evaluation_dir / n)
                                             for n in ("protocol.json", "status.json", "episode_results.csv", "slots.csv")}}
    dump(output / "report_manifest.json", _clean(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--diagnostics-root", type=Path)
    args = parser.parse_args(argv)
    result = generate_report(args.evaluation, training_root=args.training_root, output=args.output,
                             diagnostics_root=args.diagnostics_root)
    print(json.dumps({k: result[k] for k in ("status", "report", "complete_ppo_algorithms", "missing_ppo_algorithms")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
