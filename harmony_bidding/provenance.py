"""Content fingerprints for declared offline assets (integrity checks, not a sandbox)."""
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path


def capture_assets(config, extra_specs=()):
    files = config.get("sealed_model_files", [])
    if not isinstance(files, list) or any(not isinstance(p, str) for p in files):
        raise ValueError("sealed_model_files must list every offline model/statistics file path")
    paths = {Path(p).resolve() for p in files}
    package = Path(__file__).resolve().parent
    paths.update(package.glob("*.py"))
    paths.add(package.parent / "simul_bidding_env/strategy/base_bidding_strategy.py")
    specs = (list(config.get("evaluation", {}).get("teams", {}).values())
             + list(config.get("run", {}).get("teams", {}).values()) + list(extra_specs))
    if "agent" in config.get("benchmark", {}):
        specs.append(config["benchmark"]["agent"])
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("agent specifications must be mappings")
        if spec.get("kind") == "custom":
            module_name = spec["class_path"].split(":")[0]
            module = find_spec(module_name)
            if module is None or not module.origin or not Path(module.origin).is_file():
                raise ValueError(f"custom strategy must have inspectable source: {module_name}")
            paths.add(Path(module.origin).resolve())
    result = {}
    for path in sorted(paths):
        digest = sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[str(path)] = digest.hexdigest()
    return result
