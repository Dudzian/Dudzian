#!/usr/bin/env python
"""Prove that a native desktop dependency plan is contained in an exact lock."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def locked_versions(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        pins = [
            specifier.version for specifier in requirement.specifier if specifier.operator == "=="
        ]
        if len(pins) != 1:
            raise ValueError(f"non-exact lock entry: {line}")
        result[canonicalize_name(requirement.name)] = pins[0]
    return result


def validate_report(lock: Path, report: Path, *, ignore_requested_targets: bool = False) -> None:
    """Reject every dependency missing from, or version-mismatched with, the lock."""
    locked = locked_versions(lock)
    planned = json.loads(report.read_text(encoding="utf-8")).get("install", [])
    escaped: list[str] = []
    for item in planned:
        # For a project target such as .[desktop], the root project is expected
        # not to occur in its dependency lock. Only that explicitly requested
        # target is excluded; every dependency remains subject to the lock.
        if ignore_requested_targets and item.get("requested", False):
            continue
        metadata = item["metadata"]
        name = canonicalize_name(metadata["name"])
        version = metadata["version"]
        if locked.get(name) != version:
            escaped.append(f"{name}=={version} (lock: {locked.get(name, 'missing')})")
    if escaped:
        raise SystemExit("pip resolution escaped canonical lock: " + ", ".join(sorted(escaped)))


def build_plan_command(python: str, lock: Path, report: Path, target: str) -> list[str]:
    return [
        python,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--report",
        str(report),
        "--constraint",
        str(lock),
        target,
    ]


def validate_target_resolution(
    lock: Path,
    target: str = ".[desktop]",
    *,
    python: str = sys.executable,
    runner: Callable[..., object] = subprocess.run,
) -> None:
    """Resolve an actual project target, then prove all dependencies are locked."""
    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "pip-report.json"
        runner(build_plan_command(python, lock, report, target), check=True)
        validate_report(lock, report, ignore_requested_targets=True)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    parser.add_argument("--target", default=".[desktop]")
    args = parser.parse_args(argv)
    validate_target_resolution(args.lock, args.target)
    print(f"Native {args.target} dependency plan is contained in {args.lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
