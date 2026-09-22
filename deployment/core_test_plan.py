"""Execute and attest the complete canonical cross-platform core test plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Callable

from deployment.host_identity import canonical_host_os

MANIFEST = Path(__file__).with_name("core_required_suites_v1.json")
MARKER_SCHEMA_VERSION = 1


class CorePlanError(RuntimeError):
    pass


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_plan_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def selectors_for_platform(manifest: dict[str, Any], runner_os: str) -> list[str]:
    if runner_os not in manifest["required_platforms"]:
        raise CorePlanError(f"unsupported core runner: {runner_os}")
    selectors: list[str] = []
    for suite in manifest["suites"]:
        requirement = suite["platform_requirement"]
        if requirement == "REQUIRED_ON_ALL_THREE" or requirement == f"{runner_os.upper()}_ONLY":
            selectors.extend(suite["selectors"])
    if len(selectors) != len(set(selectors)):
        raise CorePlanError("duplicate selector in core manifest")
    return selectors


def marker_document(
    manifest: dict[str, Any], source_revision: str, runner_os: str,
    ci_run_id: str, ci_provider: str,
) -> dict[str, Any]:
    return {
        "schema_version": MARKER_SCHEMA_VERSION,
        "source_revision": source_revision,
        "runner_os": runner_os,
        "test_plan_id": manifest["test_plan_id"],
        "test_plan_digest": canonical_plan_digest(manifest),
        "ci_run_id": ci_run_id,
        "ci_provider": ci_provider,
    }


def execute_plan(
    *, manifest_path: Path, runner_os: str, source_revision: str, ci_run_id: str,
    ci_provider: str, output: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    manifest = load_manifest(manifest_path)
    if runner_os != canonical_host_os():
        raise CorePlanError("runner OS does not match executing host")
    selectors = selectors_for_platform(manifest, runner_os)
    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "core-plan.xml"
        completed = runner(
            [sys.executable, "-m", "pytest", "-q", "--strict-markers",
             "-o", "xfail_strict=true", f"--junitxml={report}", *selectors],
            check=False, text=True,
        )
        if completed.returncode != manifest["acceptance_policy"]["pytest_exit_code"]:
            raise CorePlanError("required core pytest plan failed")
        root = ET.parse(report).getroot()
        totals = {name: sum(int(node.attrib.get(name, 0)) for node in root.iter("testsuite"))
                  for name in ("failures", "errors", "skipped")}
        if totals != {"failures": 0, "errors": 0, "skipped": 0}:
            raise CorePlanError(f"core result violates zero-failure/error/skip policy: {totals}")
    output.write_text(json.dumps(marker_document(
        manifest, source_revision, runner_os, ci_run_id, ci_provider,
    ), indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run canonical CORE_REQUIRED_SUITES_V1")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--runner-os", required=True, choices=("Linux", "Windows", "macOS"))
    parser.add_argument("--source-revision", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--ci-run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--ci-provider", default=os.environ.get("GITHUB_SERVER_URL"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.source_revision or not args.ci_run_id or not args.ci_provider:
        parser.error("source revision, CI run id and CI provider are required")
    try:
        execute_plan(
            manifest_path=args.manifest, runner_os=args.runner_os,
            source_revision=args.source_revision, ci_run_id=args.ci_run_id,
            ci_provider=args.ci_provider, output=args.output,
        )
    except (CorePlanError, OSError, json.JSONDecodeError, ET.ParseError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
