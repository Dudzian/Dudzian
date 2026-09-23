"""Revision-bound execution evidence producers for deployment release gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

from deployment.core_test_plan import MANIFEST, canonical_plan_digest, load_manifest

SCHEMA_VERSION = 1
SCM_ITEMS = (
    "WINDOWS_SERVICE_INSTALLATION",
    "WINDOWS_SERVICE_START",
    "WINDOWS_GRACEFUL_STOP",
    "WINDOWS_MANUAL_RESTART",
)


class EvidenceProductionError(RuntimeError):
    pass


def _identity(revision: str | None) -> tuple[str, str, str]:
    source_revision = revision or os.environ.get("GITHUB_SHA", "")
    if not source_revision:
        raise EvidenceProductionError("source revision is required")
    return (
        source_revision,
        os.environ.get("GITHUB_SERVER_URL", "LOCAL_REVIEWED_EXECUTION"),
        os.environ.get("GITHUB_RUN_ID", "LOCAL"),
    )


def evidence_document(
    *, platform_name: str, source_revision: str, ci_provider: str, ci_run_id: str,
    runner_os: str, runner_arch: str, results: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "platform": platform_name,
        "source_revision": source_revision,
        "ci_provider": ci_provider,
        "ci_run_id": ci_run_id,
        "runner_os": runner_os,
        "runner_arch": runner_arch,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "results": results,
    }


def aggregate_core_markers(
    revision: str | None, markers: list[Path], output: Path, *,
    manifest_path: Path = MANIFEST, ci_run_id: str | None = None,
    ci_provider: str | None = None,
) -> None:
    """Emit one core result only after all three revision-bound runner markers exist."""
    source_revision, default_provider, default_run_id = _identity(revision)
    provider = ci_provider or default_provider
    run_id = ci_run_id or default_run_id
    manifest = load_manifest(manifest_path)
    expected_plan_id = manifest["test_plan_id"]
    expected_digest = canonical_plan_digest(manifest)
    if len(markers) != 3:
        raise EvidenceProductionError("exactly three core markers are required")
    observed: dict[str, Path] = {}
    for marker in markers:
        value = json.loads(marker.read_text(encoding="utf-8"))
        required = {
            "schema_version", "source_revision", "runner_os", "test_plan_id",
            "test_plan_digest", "ci_run_id", "ci_provider",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise EvidenceProductionError(f"malformed core marker: {marker}")
        if (
            value["schema_version"] != 1
            or value["source_revision"] != source_revision
            or value["test_plan_id"] != expected_plan_id
            or value["test_plan_digest"] != expected_digest
            or value["ci_run_id"] != run_id
            or value["ci_provider"] != provider
        ):
            raise EvidenceProductionError(f"invalid or stale core marker: {marker}")
        runner_os = value["runner_os"]
        if runner_os in observed:
            raise EvidenceProductionError(f"duplicate core runner marker: {runner_os}")
        observed[runner_os] = marker
    if set(observed) != {"Linux", "Windows", "macOS"}:
        raise EvidenceProductionError(f"incomplete core runner set: {sorted(observed)}")
    result = {"item": "CORE_REQUIRED_SUITES", "status": "PASS",
              "evidence_class": "CROSS_OS_CI_MATRIX", "test_or_probe": "core CI matrix",
              "details": f"{expected_plan_id}:{expected_digest}; Linux, Windows and macOS passed"}
    output.write_text(json.dumps(evidence_document(
        platform_name="CROSS_PLATFORM_CORE", source_revision=source_revision,
        ci_provider=provider, ci_run_id=run_id, runner_os="Windows",
        runner_arch=platform.machine(), results=[result],
    ), indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Executable deployment evidence producer")
    sub = parser.add_subparsers(dest="command", required=True)
    aggregate = sub.add_parser("aggregate-core")
    aggregate.add_argument("--source-revision")
    aggregate.add_argument("--marker", action="append", type=Path, required=True)
    aggregate.add_argument("--manifest", type=Path, default=MANIFEST)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        aggregate_core_markers(
            args.source_revision, args.marker, args.output,
            manifest_path=args.manifest,
        )
    except (EvidenceProductionError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
