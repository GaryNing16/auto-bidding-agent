"""Matched local PID references, independent episodes, and ratio-first scoring.

These scores are development diagnostics. The organizer's PID and complete
competition participant set are unavailable. Process isolation prevents ordinary
Python module state from carrying feedback between episodes; it is not a security
sandbox for hostile strategy code.
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .schema import DayData


class EvaluationError(ValueError):
    """An incomplete, invalid, or failed evaluation must not become a score."""

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report


class UndefinedReferenceError(EvaluationError):
    """A zero PID reference has no confirmed official normalization rule."""


def rotation_schedule(roles: list[str], teams: list[str]) -> list[dict[str, str]]:
    """Return d right rotations: ABC, CAB, BCA when d is three."""
    if not roles or len(roles) != len(teams):
        raise EvaluationError("multi requires exactly d teams for d nonempty roles")
    if len(set(roles)) != len(roles) or len(set(teams)) != len(teams):
        raise EvaluationError("roles and team names must be unique")
    return [
        {role: teams[(i - shift) % len(teams)] for i, role in enumerate(roles)}
        for shift in range(len(teams))
    ]


def score_ratios(
    values: Iterable[float],
    references: Iterable[float],
    *,
    expected_count: int | None = None,
    diagnostic_undefined: bool = False,
) -> dict[str, Any]:
    """Compute an equal mean of per-game ratios, never a ratio of sums.

    Diagnostic mode retains every slot and returns a null aggregate if any
    reference is zero. Invalid values and missing games always raise.
    """
    values, references = list(values), list(references)
    if not values or len(values) != len(references):
        raise EvaluationError("missing games: values/references must have equal nonzero length")
    if expected_count is not None and len(values) != expected_count:
        raise EvaluationError(f"missing games: expected {expected_count}, got {len(values)}")
    ratios: list[float | None] = []
    diagnostics: list[dict[str, Any]] = []
    for index, (value, reference) in enumerate(zip(values, references)):
        if isinstance(value, bool) or isinstance(reference, bool):
            raise EvaluationError("game values must be finite nonnegative numbers")
        if not isinstance(value, (int, float)) or not isinstance(reference, (int, float)):
            raise EvaluationError("game values must be finite nonnegative numbers")
        if not math.isfinite(value) or not math.isfinite(reference) or value < 0 or reference < 0:
            raise EvaluationError("game values must be finite nonnegative numbers")
        if reference == 0:
            message = "PID reference value is zero: normalization undefined / awaiting official rule"
            if not diagnostic_undefined:
                raise UndefinedReferenceError(f"game {index}: {message}")
            ratios.append(None)
            diagnostics.append({"game_index": index, "type": "undefined_reference", "message": message})
        else:
            ratio = value / reference
            if not math.isfinite(ratio):
                raise EvaluationError(f"game {index}: nonfinite value ratio")
            ratios.append(ratio)
    aggregate = None if any(r is None for r in ratios) else math.fsum(r / len(ratios) for r in ratios)
    return {"ratios": ratios, "G": aggregate, "valid": aggregate is not None, "diagnostics": diagnostics}


def relative_scores(aggregates: dict[str, float | None], weight: float) -> dict[str, float | None]:
    """Local-relative diagnostics; nulls never disappear from the cohort."""
    if not aggregates or not math.isfinite(weight) or weight < 0:
        raise EvaluationError("nonempty team set and finite nonnegative score weight required")
    if any(value is None for value in aggregates.values()):
        return {team: None for team in aggregates}
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in aggregates.values()):
        raise EvaluationError("G values must be finite and nonnegative")
    maximum = max(aggregates.values())
    return {team: (0.0 if maximum == 0 else weight * (value / maximum)) for team, value in aggregates.items()}


def _worker(connection: Any, payload: tuple) -> None:
    """Top-level spawn target: import the runner only inside the clean process."""
    try:
        from .runner import run_episode

        day, config, assignments, team_specs, episode_id, mode, run_id = payload
        result = run_episode(day, config, assignments, team_specs, episode_id, mode, run_id=run_id)
        connection.send((True, result))
    except BaseException as exc:
        connection.send((False, {"type": type(exc).__name__, "message": str(exc),
                                 "traceback": traceback.format_exc(), "report": getattr(exc, "report", None)}))
    finally:
        connection.close()


def _run_fresh_episode(
    day: Any,
    config: dict,
    assignments: dict[str, str],
    team_specs: dict[str, dict],
    episode_id: str,
    mode: str,
    run_id: str | None = None,
) -> dict:
    isolated_config = copy.deepcopy(config)
    isolated_specs = copy.deepcopy(team_specs)
    isolation = config.get("evaluation", {}).get("isolation", "process")
    if isolation == "in_process":
        from .runner import run_episode

        return run_episode(day, isolated_config, dict(assignments), isolated_specs, episode_id, mode, run_id=run_id)
    if isolation != "process":
        raise EvaluationError("evaluation.isolation must be process or in_process")
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(sender, (day, isolated_config, dict(assignments), isolated_specs, episode_id, mode, run_id)),
    )
    process.start()
    sender.close()
    timeout = float(config.get("evaluation", {}).get("episode_timeout_seconds", 120))
    try:
        if not receiver.poll(timeout):
            raise EvaluationError(f"episode {episode_id} timed out after {timeout} seconds")
        try:
            success, result = receiver.recv()
        except EOFError as exc:
            raise EvaluationError(f"episode {episode_id} worker exited without a result") from exc
        process.join(timeout=10)
        if process.is_alive():
            raise EvaluationError(f"episode {episode_id} worker did not exit after sending result")
        if not success:
            raise EvaluationError(f"episode {episode_id} failed: {result['type']}: {result['message']}\n{result['traceback']}", result.get("report"))
        if process.exitcode != 0:
            raise EvaluationError(f"episode {episode_id} worker exit code {process.exitcode}")
        return result
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        process.close()


def _advertiser_ids(day: Any) -> list[str]:
    advertisers = day.advertisers
    if isinstance(advertisers, dict):
        return list(advertisers)
    return [advertiser.advertiser_id for advertiser in advertisers]


def _validate_episode(result: dict, advertiser_ids: list[str], episode_id: str, total_slots: int) -> None:
    checks = result.get("checks", {})
    if checks.get("legal") is not True or checks.get("budget") is not True:
        raise EvaluationError(f"episode {episode_id} failed legality/budget checks")
    summaries = result.get("summaries", {})
    if set(summaries) != set(advertiser_ids):
        raise EvaluationError(f"episode {episode_id} has missing or unexpected advertisers")
    if result.get("exception_count", 0) or result.get("exceptions", []):
        raise EvaluationError(f"episode {episode_id} contains recorded exceptions")
    expected_slots = {(advertiser_id, slot) for advertiser_id in advertiser_ids for slot in range(total_slots)}
    actual_slots = [(row.get("advertiser_id"), row.get("time_slot")) for row in result.get("slots", [])]
    if len(actual_slots) != len(expected_slots) or set(actual_slots) != expected_slots:
        raise EvaluationError(f"episode {episode_id} has missing, duplicate, or unexpected slots")
    for advertiser_id, summary in summaries.items():
        for field in ("value", "cost", "budget", "remaining_budget"):
            value = summary.get(field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise EvaluationError(f"episode {episode_id}, {advertiser_id}: invalid {field}")
        if summary["cost"] > summary["budget"]:
            raise EvaluationError(f"episode {episode_id}, {advertiser_id}: budget exceeded")
        if not math.isclose(summary["cost"] + summary["remaining_budget"], summary["budget"], abs_tol=1e-8):
            raise EvaluationError(f"episode {episode_id}, {advertiser_id}: inconsistent budget accounting")
        if summary.get("done") is not True or any(summary.get(field) != total_slots for field in ("bidding_calls", "update_calls")):
            raise EvaluationError(f"episode {episode_id}, {advertiser_id}: incomplete lifecycle")


def evaluate(days: Iterable[DayData], config: dict, mode: str, output_dir: Path | None = None, *, run_record=None) -> dict:
    """Public entry: even preflight failures belong to a new recorded attempt."""
    from .runner import RunRecord, write_json
    record = run_record or RunRecord(output_dir, f"evaluate-{mode}")
    try:
        record.bind(config)
        report = _evaluate(days, config, mode, record.directory, record.run_id)
        report.update(record.identity())
        if record.directory is not None:
            write_json(record.directory / "evaluation.json", report)
        record.complete({"mode": mode, "teams": report["teams"], **record.identity()}, report["valid"])
        return report
    except Exception as exc:
        record.fail(exc)
        if record.directory is not None:
            path = record.directory / "evaluation.json"
            failure = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
                "mode": mode, "valid": False, "type": type(exc).__name__, "message": str(exc),
                "completed_games": [], "completed_episodes": [], "score_status": "invalid; no aggregate score"}
            failure.update(record.identity())
            write_json(path, failure)
            write_json(record.directory / "evaluation_error.json", failure)
        raise


def _evaluate(days, config, mode, output_dir, run_id):
    """Evaluate local teams against independent matched all-PID daily references.

    Every episode receives a deep copy of configuration and strategy specifications.
    The default spawn isolation also resets process-level strategy state. In-process
    mode is for trusted development and cannot guarantee reset of mutable globals.
    Dates are consumed once from an iterable and only the current date is retained.
    Data, budgets, rule profile, and request-keyed tie seed remain matched across
    reference/replacement games. No historical PID bids are replayed.
    """
    if mode not in {"single", "multi"}:
        raise EvaluationError("mode must be single or multi")
    from .schema import Rules
    from .provenance import capture_assets

    rules = Rules(**config.get("rules", {}))
    settings = config.get("evaluation", {})
    from .agent_api import OfficialStrategyUnavailable
    if settings.get("reference_strategy", config.get("reference_strategy", "local_pid")) != "local_pid":
        raise OfficialStrategyUnavailable("Official reference strategy unavailable; local_pid is only a local baseline")
    if "kind" in config.get("pid", {}):
        raise EvaluationError("pid configuration cannot override the LocalPIDAgent kind")
    roles = list(settings.get("roles", []))
    team_specs = copy.deepcopy(settings.get("teams", {}))
    for team, spec in team_specs.items():
        if not isinstance(spec, dict):
            raise EvaluationError(f"team {team}: constructor specification must be a mapping")
        if spec.get("kind", "pid") == "pid":
            team_specs[team] = {"kind": "pid", **copy.deepcopy(config.get("pid", {})), **spec}
    teams = list(team_specs)
    if not roles or len(roles) != len(set(roles)) or not teams:
        raise EvaluationError("nonempty unique roles and nonempty teams required")
    if any(not isinstance(team, str) or not team for team in teams):
        raise EvaluationError("team names must be nonempty strings")
    if mode == "multi":
        schedule = rotation_schedule(roles, teams)
    else:
        schedule = [{role: team} for team in teams for role in roles]
    diagnostic = settings.get("diagnostic_undefined", False)
    if type(diagnostic) is not bool:
        raise EvaluationError("diagnostic_undefined must be boolean")
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    games: list[dict] = []
    references: list[dict] = []
    episode_records: list[dict] = []
    seen_day_ids: set[str] = set()
    type_registry: dict[str, str] = {}
    day_count = 0
    sealed_assets: dict | None = None

    def check_sealed_assets() -> None:
        if capture_assets(config) != sealed_assets:
            raise EvaluationError("sealed model/source assets changed during evaluation")

    try:
        sealed_assets = capture_assets(config)
        for day_index, day in enumerate(days):
            # Public callers may pass DayData directly, bypassing CSV loaders.
            # Reuse the same dataset-scoped identity contract at this boundary.
            from .data import _register_types
            _register_types(day.advertisers, type_registry)
            day_id = str(day.day_id)
            if day_id in seen_day_ids:
                raise EvaluationError("duplicate day IDs are not independent evaluation dates")
            seen_day_ids.add(day_id)
            day_count += 1
            advertiser_ids = _advertiser_ids(day)
            if not set(roles).issubset(advertiser_ids):
                raise EvaluationError(f"unknown test role in date {day.day_id}")
            reference_id = f"{mode}_date{day_index:03d}_pid_reference"
            check_sealed_assets()
            reference = _run_fresh_episode(day, config, {}, {}, reference_id, "pid_reference", run_id)
            check_sealed_assets()
            _validate_episode(reference, advertiser_ids, reference_id, rules.total_slots)
            references.append({
                "day_id": str(day.day_id), "episode_id": reference_id,
                "values": {role: reference["summaries"][role]["value"] for role in roles},
                "metadata": reference.get("metadata", {}),
            })
            if output_dir is not None:
                from .runner import write_episode

                write_episode(reference, output_dir / reference_id)
            episode_records.append({"episode_id": reference_id, "kind": "pid_reference", "checks": reference["checks"]})
            for match_index, assignments in enumerate(schedule):
                episode_id = f"{mode}_date{day_index:03d}_match{match_index:03d}"
                check_sealed_assets()
                episode = _run_fresh_episode(day, config, assignments, team_specs, episode_id, mode, run_id)
                check_sealed_assets()
                _validate_episode(episode, advertiser_ids, episode_id, rules.total_slots)
                # Persist actual gameplay even when normalization below is undefined.
                if output_dir is not None:
                    write_episode(episode, output_dir / episode_id)
                rows = []
                for role, team in assignments.items():
                    summary = episode["summaries"][role]
                    reference_value = reference["summaries"][role]["value"]
                    normalized = score_ratios([summary["value"]], [reference_value], diagnostic_undefined=diagnostic)
                    row = {
                        "run_id": run_id,
                        "episode_id": episode_id, "mode": mode, "day_id": str(day.day_id),
                        "team": team, "advertiser_id": role, "value": summary["value"],
                        "cost": summary["cost"], "budget": summary["budget"],
                        "budget_utilization": summary.get("budget_utilization"), "wins": summary.get("wins"),
                        "pid_reference_value": reference_value, "value_ratio": normalized["ratios"][0],
                        "valid": normalized["valid"], "diagnostics": normalized["diagnostics"],
                    }
                    rows.append(row)
                    games.append(row)
                    summary["pid_reference_value"] = reference_value
                    summary["value_ratio"] = normalized["ratios"][0]
                episode["evaluation"] = {"reference_episode_id": reference_id, "per_game": rows}
                if output_dir is not None:
                    write_episode(episode, output_dir / episode_id)
                episode_records.append({"episode_id": episode_id, "kind": mode, "assignments": dict(assignments), "checks": episode["checks"]})
        if day_count == 0:
            raise EvaluationError("evaluation requires at least one complete date")
        expected = day_count * len(roles)
        team_results: dict[str, dict] = {}
        for team in teams:
            rows = [row for row in games if row["team"] == team]
            normalized = score_ratios(
                [row["value"] for row in rows], [row["pid_reference_value"] for row in rows],
                expected_count=expected, diagnostic_undefined=diagnostic,
            )
            team_results[team] = {
                "G": normalized["G"], f"G_{mode}": normalized["G"],
                "games_expected": expected, "games_recorded": len(rows),
                "valid": normalized["valid"], "diagnostics": normalized["diagnostics"],
            }
        local_scores = relative_scores({team: result["G"] for team, result in team_results.items()}, 40.0 if mode == "single" else 15.0)
        for team in teams:
            team_results[team]["local_relative_score"] = local_scores[team]
        report = {
            "mode": mode,
            "rules_version": rules.rules_version,
            "assumption_profile": rules.assumption_profile,
            "local_assumptions": asdict(rules),
            "sealed_assets": sealed_assets,
            "budget_policy_notice": "LOCAL assumption: exclude unaffordable candidate and reauction this request",
            "isolation": settings.get("isolation", "process"),
            "local_only": True,
            "reference_strategy": "LocalPIDAgent", "reference_origin": "local",
            "official_strategy_available": False,
            "score_status": "local relative scores; official complete participant set and official PID unavailable",
            "matching_protocol": "same date, advertiser, budgets, rule config and request-keyed tie seed; opponents rerun online",
            "roles": roles, "schedule": schedule, "references": references,
            "dates_evaluated": day_count,
            "games": games, "teams": team_results, "episodes": episode_records,
            "valid": all(result["valid"] for result in team_results.values()),
        }
        if output_dir is not None:
            from .runner import write_json
            write_json(output_dir / "evaluation.json", report)
        return report
    except Exception as exc:
        if output_dir is not None:
            failure = {
                "mode": mode, "valid": False, "type": type(exc).__name__, "message": str(exc),
                "rules_version": rules.rules_version,
                "assumption_profile": rules.assumption_profile,
                "sealed_assets": sealed_assets,
                "completed_games": games, "completed_episodes": episode_records,
                "score_status": "invalid/incomplete evaluation; no aggregate score",
            }
            episode_failure = getattr(exc, "report", None)
            if episode_failure is not None:
                from .runner import write_episode

                failure_id = episode_failure.get("metadata", {}).get("episode_id", "failed_episode")
                write_episode(episode_failure, output_dir / failure_id)
                failure["failed_episode_id"] = failure_id
            from .runner import write_json
            write_json(output_dir / "evaluation_error.json", failure)
            # A reused output directory must never retain an apparently current
            # successful aggregate after this run failed. Preserve episode logs.
            write_json(output_dir / "evaluation.json", failure)
        raise
