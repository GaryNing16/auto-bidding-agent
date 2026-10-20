"""Fetch the exact reference upstream checkout; does not overwrite existing work."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

UPSTREAM = "https://github.com/alimama-tech/AuctionNet.git"
COMMIT = "c20de631a40e24669d1d6a8e2acefd8f492f172d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("external/AuctionNet"))
    args = parser.parse_args()
    target = args.destination.resolve()
    if target.exists():
        parser.error(f"destination already exists; inspect it or choose a new path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", UPSTREAM, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", COMMIT], check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f"unexpected upstream commit: {actual}")
    print(f"Reference checkout ready: {target}\nCommit: {actual}")


if __name__ == "__main__":
    main()
