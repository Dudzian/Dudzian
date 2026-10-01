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

if __package__:
    from scripts.ci.bootstrap_dependency import (
        canonicalize_package_name,
        locked_versions,
        project_requirements,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bootstrap_dependency import (
        canonicalize_package_name,
        locked_versions,
        project_requirements,
    )


def parse_pip_report(report: Path) -> dict[str, str]:
    """Read a pip installation report, rejecting incomplete or ambiguous metadata."""
    payload = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("install"), list):
        raise ValueError(f"malformed pip report: {report}")

    planned: dict[str, str] = {}
    for item in payload["install"]:
        if not isinstance(item, dict):
            raise ValueError(f"malformed pip report item: {report}")
        requested = item.get("requested", False)
        if not isinstance(requested, bool):
            raise ValueError(f"malformed requested flag in pip report: {report}")
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"missing pip report metadata: {report}")
        raw_name = metadata.get("name")
        version = metadata.get("version")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"missing package name in pip report: {report}")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"missing package version in pip report: {report}")
        name = canonicalize_package_name(raw_name.strip())
        if name in planned:
            raise ValueError(f"duplicate package in pip report: {name}")
        planned[name] = version.strip()
    return planned


def validate_reports(active_lock_report: Path, desktop_report: Path) -> None:
    """Prove that the actual desktop plan is a subset of the native-active lock plan."""
    active_lock = parse_pip_report(active_lock_report)
    desktop = parse_pip_report(desktop_report)
    escaped = [
        f"{name}=={version} (native lock: {active_lock.get(name, 'missing')})"
        for name, version in desktop.items()
        if active_lock.get(name) != version
    ]
    if escaped:
        raise SystemExit(
            "pip resolution escaped native-active canonical lock: " + ", ".join(escaped)
        )


def build_active_lock_command(python: str, lock: Path, report: Path) -> list[str]:
    return [
        python,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--report",
        str(report),
        "--requirement",
        str(lock),
    ]


def build_target_plan_command(
    python: str, lock: Path, report: Path, requirements: Sequence[str]
) -> list[str]:
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
        *requirements,
    ]


def validate_target_resolution(
    lock: Path,
    target: str = ".[desktop]",
    *,
    python: str = sys.executable,
    project: Path = Path("pyproject.toml"),
    runner: Callable[..., object] = subprocess.run,
) -> None:
    """Resolve an actual project target, then prove all dependencies are locked."""
    if target != ".[desktop]":
        raise ValueError(f"unsupported project dependency target: {target}")
    # Keep structural validation independent from pip's marker evaluation and
    # read project metadata without invoking its PEP 517 backend.
    locked_versions(lock)
    requirements = project_requirements(project, ("desktop",))
    with tempfile.TemporaryDirectory() as directory:
        active_lock_report = Path(directory) / "native-lock-plan.json"
        desktop_report = Path(directory) / "desktop-plan.json"
        runner(build_active_lock_command(python, lock, active_lock_report), check=True)
        runner(build_target_plan_command(python, lock, desktop_report, requirements), check=True)
        validate_reports(active_lock_report, desktop_report)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    parser.add_argument("--target", default=".[desktop]")
    args = parser.parse_args(argv)
    validate_target_resolution(args.lock, args.target)
    print(f"Native {args.target} dependency plan is contained in the active plan for {args.lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
