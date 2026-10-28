"""Synchronous scalar decisions, sequential auctions, private feedback, fresh episodes."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import csv
import json
import os
import platform
from pathlib import Path
import sys
import tempfile
from time import perf_counter_ns
from uuid import uuid4
from .agent_api import OwnConfig, PublicConfig, Observation, OwnFeedback, OfficialStrategyUnavailable
from .agents import make_agent
from .auction import AuctionEngine, validate_alpha
from .schema import Rules, fingerprint
from .provenance import capture_assets

UPSTREAM_URL = "https://github.com/alimama-tech/AuctionNet"
UPSTREAM_COMMIT = "c20de631a40e24669d1d6a8e2acefd8f492f172d"


class EpisodeFailure(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def code_fingerprint():
    root = Path(__file__).resolve().parent
    files = sorted(root.glob("*.py")) + [root.parent / "simul_bidding_env/strategy/base_bidding_strategy.py"]
    h = sha256()
    for path in files:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def run_episode(day, config, assignments=None, team_specs=None, episode_id="episode", mode="demo",
                agent_factories=None, run_id=None):
    """Evaluator-only entry. No argument here is given to policies except typed snapshots.

    factories are a test/embedding hook; built-in CLI uses sealed constructor specs.
    Stronger reset isolation across episodes is provided by evaluation process mode.
    """
    from .data import validate_day
    run_id = run_id or uuid4().hex
    metadata = {"run_id": run_id, "episode_id": episode_id, "mode": mode,
                "day_id": getattr(day, "day_id", None), "local_only": True,
                "reference_strategy": "LocalPIDAgent", "reference_origin": "local",
                "official_strategy_available": False,
                "runtime": {"python": sys.version, "implementation": platform.python_implementation(),
                            "executable": sys.executable},
                "tie_protocol": "sha256-json-request-round-candidates/python-random-randrange-v2"}
    report = {"metadata": metadata, "summaries": {}, "slots": [], "timing": [],
              "initialization_ms": {}, "checks": {"legal": True, "budget": True},
              "exceptions": [], "exception_count": 0, "trajectories": []}
    engine = None
    stage, active_id, slot = "preflight", None, None
    try:
        config = deepcopy(config)
        rules = Rules(**config.get("rules", {}))
        if config.get("reference_strategy", "local_pid") != "local_pid":
            raise OfficialStrategyUnavailable("Only the explicitly local PID reference is available")
        validate_day(day, rules, config.get("expected_n"))
        assignments, team_specs = dict(assignments or {}), deepcopy(team_specs or {})
        ids = tuple(a.advertiser_id for a in day.advertisers)
        if set(assignments) - set(ids):
            raise ValueError("unknown assigned advertiser")
        if any(team not in team_specs for team in assignments.values()):
            raise ValueError("unknown team")
        engine = AuctionEngine(day.advertisers, rules)
        public = PublicConfig(rules.total_slots)
        if "kind" in config.get("pid", {}):
            raise ValueError("pid configuration cannot override kind; all-PID reference must use LocalPIDAgent")
        pid = {**config.get("pid", {}), "kind": "pid"}
        specs = {a: deepcopy(team_specs[assignments[a]] if a in assignments else pid) for a in ids}
        sealed_assets = capture_assets(config, specs.values())
        metadata.update({
        "dataset_fingerprint": day.fingerprint, "config_fingerprint": fingerprint(config),
        "dataset_source_schema_version": getattr(day, "source_schema_version", None),
        "dataset_source_fingerprint": getattr(day, "source_fingerprint", None),
        "code_fingerprint": code_fingerprint(), "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT, "rules_version": rules.rules_version,
        "assumption_profile": rules.assumption_profile, "local_assumptions": asdict(rules),
        "budget_policy_notice": "LOCAL assumption: exclude unaffordable candidate and reauction this request",
        "assignments": assignments, "sealed_assets": sealed_assets,
        "model_fingerprint": fingerprint({"specifications": specs, "assets": sealed_assets}),
        "random_protocol": "stable request-keyed local uniform ties; role/day agent seed; separate traffic RNG"})
        if "_dataset_import_metadata" in config:
            metadata["data_import"] = deepcopy(config["_dataset_import_metadata"])
        agents, histories, own_configs = {}, {a: [] for a in ids}, {}
        metadata["strategies"] = {}
        stage = "initialization"
        for advertiser in day.advertisers:
            a = active_id = advertiser.advertiser_id
            seed = int(fingerprint([config.get("agent_seed", 1234), day.day_id, a])[:16], 16)
            own_configs[a] = OwnConfig(a, advertiser.advertiser_type,
                                      engine.money.amount(engine.initial[a]), seed)
            start = perf_counter_ns()
            agent = agent_factories[a]() if agent_factories and a in agent_factories else make_agent(specs[a])
            agent.reset(deepcopy(own_configs[a]), deepcopy(public))
            report["initialization_ms"][a] = (perf_counter_ns() - start) / 1e6
            agents[a] = agent
            metadata["strategies"][a] = {"class": f"{type(agent).__module__}:{type(agent).__qualname__}",
                                          "specification": deepcopy(specs[a])}

        def observation(a, time_slot):
            return Observation(own_configs[a], time_slot, rules.total_slots,
                               engine.money.amount(engine.remaining[a]), tuple(histories[a]))

        for slot in range(rules.total_slots):
            # Build all pre-slot observations before invoking any policy.
            observations = {a: observation(a, slot) for a in ids}
            actions, durations = {}, {}
            stage = "bidding"
            for a in ids:
                active_id = a
                agent_input = deepcopy(observations[a])
                start = perf_counter_ns()
                raw_action = agents[a].bidding(agent_input)
                elapsed = (perf_counter_ns() - start) / 1e6
                actions[a] = validate_alpha(raw_action)
                durations[a] = elapsed
            stage, active_id = "auction", None
            stats, debug = engine.run_slot(day.slots[slot], actions)
            if config.get("debug_privileged", False):
                report.setdefault("privileged_debug", []).append({"time_slot": slot, "requests": debug,
                    "opportunity_count": {a: stats[a].get("opportunity_count") for a in ids}})
            feedbacks = {}
            for a in ids:
                cumulative_cost = engine.initial[a] - engine.remaining[a]
                feedbacks[a] = OwnFeedback(slot, actions[a], stats[a]["wins"],
                    engine.money.amount(stats[a]["cost_units"]), stats[a]["value"],
                    engine.money.amount(engine.remaining[a]), engine.money.amount(cumulative_cost),
                    engine.cumulative_value[a])
            stage = "update"
            for a in ids:
                active_id = a
                private_feedback = deepcopy(feedbacks[a])
                start = perf_counter_ns()
                agents[a].update(private_feedback)
                elapsed = (perf_counter_ns() - start) / 1e6
                report["timing"].append({"advertiser_id": a, "time_slot": slot,
                    "run_id": run_id,
                    "rules_version": rules.rules_version, "assumption_profile": rules.assumption_profile,
                    "bidding_ms": durations[a], "update_ms": elapsed})
            for a in ids:
                feedback = feedbacks[a]
                histories[a].append(feedback)
                row = {"run_id": run_id, "episode_id": episode_id, "day_id": day.day_id, "mode": mode,
                       "rules_version": rules.rules_version, "assumption_profile": rules.assumption_profile,
                       "advertiser_id": a, "team": assignments.get(a, "LocalPIDReference"),
                       **asdict(feedback)}
                report["slots"].append(row)
                report["trajectories"].append({"observation_t": asdict(observations[a]),
                    "run_id": run_id, "trajectory_version": "harmony-trajectory-v2", "money_decimals": rules.money_decimals,
                    "alpha_t": actions[a], "reward_t": feedback.value, "cost_t": feedback.cost,
                    "next_observation": asdict(observation(a, slot + 1)),
                    "done": slot + 1 == rules.total_slots, "episode_id": episode_id,
                    "advertiser_id": a, "day_id": day.day_id,
                    "rules_version": rules.rules_version, "assumption_profile": rules.assumption_profile})

        for a in ids:
            budget, remaining = engine.money.amount(engine.initial[a]), engine.money.amount(engine.remaining[a])
            total_cost_units = sum(engine.money.units(f.cost) for f in histories[a])
            if total_cost_units != engine.initial[a] - engine.remaining[a]:
                raise AssertionError("cost accounting inconsistent")
            report["summaries"][a] = {"advertiser_id": a, "team": assignments.get(a, "LocalPIDReference"),
                "budget": budget, "cost": engine.money.amount(total_cost_units), "remaining_budget": remaining,
                "value": engine.cumulative_value[a], "wins": sum(f.wins for f in histories[a]),
                "budget_utilization": engine.money.amount(total_cost_units) / budget if budget else 0.0,
                "bidding_calls": rules.total_slots, "update_calls": rules.total_slots,
                "pid_reference_value": None, "value_ratio": None, "done": True}
            if mode in {"all_pid_reference", "pid_reference"}:
                report["summaries"][a]["pid_reference_value"] = engine.cumulative_value[a]
                report["summaries"][a]["value_ratio"] = 1.0 if engine.cumulative_value[a] > 0 else None
        stage, active_id = "sealed_asset_integrity", None
        if capture_assets(config, specs.values()) != sealed_assets:
            raise ValueError("sealed source/model/statistics changed during episode; evaluation invalid")
        return report
    except Exception as exc:
        report["exceptions"].append({"type": type(exc).__name__, "message": str(exc),
                                     "stage": stage, "advertiser_id": active_id, "time_slot": slot})
        report["exception_count"] = len(report["exceptions"])
        report["checks"] = {"legal": False, "budget": engine is not None and all(v >= 0 for v in engine.remaining.values())}
        if engine is not None:
            report["failure_state"] = {"remaining_units": dict(engine.remaining),
                                       "cumulative_value": dict(engine.cumulative_value)}
        raise EpisodeFailure(f"{stage} failed for {active_id} at slot {slot}: {exc}", report) from exc


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_episode(result, output_dir):
    """All-advertiser evaluator reports are separate from actor training files."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "episode.json", {k: v for k, v in result.items()
                                           if k not in {"trajectories", "privileged_debug"}})
    for key in ("slots", "timing"):
        with (directory / f"{key}.csv").open("w", newline="", encoding="utf-8") as handle:
            if result[key]:
                writer = csv.DictWriter(handle, fieldnames=list(result[key][0]))
                writer.writeheader()
                writer.writerows(result[key])
    if "privileged_debug" in result:
        write_json(directory / "PRIVILEGED_DEBUG.json", result["privileged_debug"])


def business_result(result):
    """Comparison excludes timing, init/load cost and runtime fingerprints."""
    business = deepcopy({k: result[k] for k in ("summaries", "slots", "checks", "exceptions")})
    for row in business["slots"]:
        row.pop("run_id", None)
    return business


class RunRecord:
    """A unique attempt, independent of earlier successful output directories.

    Only completed + valid records may be consumed as successful results.
    Abrupt termination leaves running status, never an apparently valid result.
    """
    def __init__(self, output_root=None, command="run", run_id=None):
        self.run_id = run_id or uuid4().hex
        self.directory = Path(output_root) / self.run_id if output_root is not None else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=False)
        self.state = {"run_id": self.run_id, "command": command, "status": "running", "valid": False,
                      "report_directory": str(self.directory.resolve()) if self.directory else None,
                      "config_fingerprint": None, "code_fingerprint": code_fingerprint()}
        self.persist()

    def persist(self):
        if self.directory is not None:
            write_json(self.directory / "status.json", self.state)

    def bind(self, config):
        rules = Rules(**config.get("rules", {}))
        self.state.update(config_fingerprint=fingerprint(config), rules_version=rules.rules_version,
                          assumption_profile=rules.assumption_profile)
        self.persist()

    def complete(self, summary, valid=True):
        self.state.update(status="completed", valid=bool(valid), summary=summary)
        self.persist()

    def fail(self, exc):
        self.state.update(status="failed", valid=False, exception_type=type(exc).__name__, message=str(exc))
        self.state.pop("summary", None)
        if self.directory is not None:
            write_json(self.directory / "ERROR.json", self.state)
        self.persist()

    def identity(self):
        return {k: self.state[k] for k in ("run_id", "report_directory", "config_fingerprint", "code_fingerprint")}


def read_completed_run(directory):
    """Explicit reader guard: never infer success from a leftover results file."""
    directory = Path(directory)
    state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if state.get("status") != "completed" or state.get("valid") is not True or state.get("run_id") != directory.name:
        raise ValueError("run is incomplete, failed, invalid or belongs to another run_id")
    command = state.get("command")
    required_report = ("evaluation.json" if command in {"evaluate-single", "evaluate-multi"}
                       else "benchmark.json" if command == "benchmark" else None)
    if required_report is not None and not (directory / required_report).is_file():
        raise ValueError(f"completed run is missing required report: {required_report}")
    for filename in ("evaluation.json", "benchmark.json"):
        path = directory / filename
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            if any(report.get(key) != state.get(key) for key in ("run_id", "config_fingerprint", "code_fingerprint")):
                raise ValueError("report identity/fingerprints do not match completed run")
            if report.get("valid") is not True:
                raise ValueError(f"{filename} report is invalid")
    return state
