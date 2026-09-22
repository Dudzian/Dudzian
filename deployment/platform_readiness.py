"""Evaluate independent platform gates from the canonical deployment contract."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

CONTRACT = Path(__file__).with_name("platform_readiness.json")
PASS = "PASS"
REQUIRES_EXECUTION_EVIDENCE = {
    "LIVE_WINDOWS_INTEGRATION",
    "LIVE_LINUX_INTEGRATION",
    "LIVE_MACOS_INTEGRATION",
    "CROSS_OS_CI_MATRIX",
}


class EvidenceValidationError(ValueError):
    pass


def load_contract(path: Path = CONTRACT) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_results(
    data: dict[str, Any], evidence: list[dict[str, Any]], current_revision: str,
    expected_ci_provider: str, expected_ci_run_id: str,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    expected_os = {
        "LIVE_WINDOWS_INTEGRATION": ("WINDOWS", "Windows"),
        "LIVE_LINUX_INTEGRATION": ("LINUX", "Linux"),
        "LIVE_MACOS_INTEGRATION": ("MACOS", "macOS"),
        "CROSS_OS_CI_MATRIX": ("CROSS_PLATFORM_CORE", None),
    }
    for document in evidence:
        required = {
            "schema_version", "platform", "source_revision", "ci_provider", "ci_run_id",
            "runner_os", "runner_arch", "generated_at_utc", "results",
        }
        if not isinstance(document, dict) or set(document) != required:
            raise EvidenceValidationError("malformed evidence document")
        if document["schema_version"] != 1 or document["source_revision"] != current_revision:
            raise EvidenceValidationError("unsupported schema or stale source revision")
        if document["ci_provider"] != expected_ci_provider:
            raise EvidenceValidationError("untrusted CI provider")
        if document["ci_run_id"] != expected_ci_run_id:
            raise EvidenceValidationError("stale CI run")
        if not all(isinstance(document[name], str) and document[name] for name in required - {"schema_version", "results"}):
            raise EvidenceValidationError("empty evidence identity field")
        try:
            if not document["generated_at_utc"].endswith("Z"):
                raise ValueError
            timestamp = datetime.fromisoformat(
                document["generated_at_utc"].replace("Z", "+00:00")
            )
            if timestamp.utcoffset() is None:
                raise ValueError
        except ValueError as exc:
            raise EvidenceValidationError("invalid evidence timestamp") from exc
        if not isinstance(document["results"], list):
            raise EvidenceValidationError("results must be a list")
        for result in document["results"]:
            result_fields = {"item", "status", "evidence_class", "test_or_probe", "details"}
            if not isinstance(result, dict) or set(result) != result_fields:
                raise EvidenceValidationError("malformed evidence result")
            if not all(isinstance(result[name], str) and result[name] for name in result_fields):
                raise EvidenceValidationError("empty evidence result field")
            item = result["item"]
            if item not in data["acceptance"]:
                raise EvidenceValidationError(f"unknown evidence item: {item}")
            if item in results:
                raise EvidenceValidationError(f"duplicate evidence item: {item}")
            evidence_class = result["evidence_class"]
            if evidence_class not in REQUIRES_EXECUTION_EVIDENCE:
                raise EvidenceValidationError(f"unsupported execution evidence class: {evidence_class}")
            if evidence_class != data["acceptance"][item]["evidence_class"]:
                raise EvidenceValidationError(f"evidence class mismatch: {item}")
            expected_platform, runner_os = expected_os[evidence_class]
            if document["platform"] != expected_platform:
                raise EvidenceValidationError(f"platform mismatch: {item}")
            if runner_os is not None and document["runner_os"] != runner_os:
                raise EvidenceValidationError(f"runner OS mismatch: {item}")
            if result["status"] not in data["allowed_statuses"]:
                raise EvidenceValidationError(f"invalid evidence status: {item}")
            results[item] = result
    return results


def production_ready(
    platform: str, contract: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None, current_revision: str | None = None,
    expected_ci_provider: str = "test", expected_ci_run_id: str = "1",
) -> bool:
    data = load_contract() if contract is None else contract
    gates = data["release_gates"]
    items = data["acceptance"]
    required = gates[f"{platform}_PRODUCTION_READY"]
    try:
        results = _validated_results(
            data, evidence or [], current_revision or "",
            expected_ci_provider, expected_ci_run_id,
        )
    except EvidenceValidationError:
        return False
    for item in required:
        evidence_class = items[item]["evidence_class"]
        if evidence_class == "STATIC_EXECUTABLE_CONTRACT":
            if items[item]["status"] != PASS:
                return False
        elif evidence_class in REQUIRES_EXECUTION_EVIDENCE:
            if item not in results or results[item]["status"] != PASS:
                return False
        else:
            return False
    return True


def blocking_items(
    platform: str, contract: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None, current_revision: str | None = None,
    expected_ci_provider: str = "test", expected_ci_run_id: str = "1",
) -> list[str]:
    """Return required items without PASS evidence for the selected platform."""
    data = load_contract() if contract is None else contract
    required = data["release_gates"][f"{platform}_PRODUCTION_READY"]
    try:
        results = _validated_results(
            data, evidence or [], current_revision or "",
            expected_ci_provider, expected_ci_run_id,
        )
    except EvidenceValidationError as exc:
        return [f"INVALID_EVIDENCE:{exc}"]
    blockers = []
    for name in required:
        item = data["acceptance"][name]
        if item["evidence_class"] == "STATIC_EXECUTABLE_CONTRACT":
            if item["status"] != PASS:
                blockers.append(name)
        elif name not in results or results[name]["status"] != PASS:
            blockers.append(name)
    return blockers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed platform release gate")
    parser.add_argument("--platform", required=True, choices=("WINDOWS", "LINUX", "MACOS"))
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--evidence", action="append", type=Path, default=[])
    parser.add_argument("--current-revision", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--expected-ci-provider", default=os.environ.get("GITHUB_SERVER_URL"))
    parser.add_argument("--expected-ci-run-id", default=os.environ.get("GITHUB_RUN_ID"))
    args = parser.parse_args(argv)
    data = load_contract(args.contract)
    if not args.current_revision or not args.expected_ci_provider or not args.expected_ci_run_id:
        parser.error("current revision, CI provider and CI run id are required")
    try:
        evidence = [json.loads(path.read_text(encoding="utf-8")) for path in args.evidence]
    except (OSError, json.JSONDecodeError) as exc:
        print(f"invalid evidence file: {exc}")
        return 1
    blockers = blocking_items(
        args.platform, data, evidence, args.current_revision,
        args.expected_ci_provider, args.expected_ci_run_id,
    )
    if blockers:
        for name in blockers:
            if name.startswith("INVALID_EVIDENCE:"):
                print(name)
                continue
            item = data["acceptance"][name]
            status = item["status"] if item["evidence_class"] == "STATIC_EXECUTABLE_CONTRACT" else "MISSING_CURRENT_EVIDENCE"
            print(f"{name}={status} [{item['evidence_class']}]")
        print(f"{args.platform}_PRODUCTION_READY=NOT_READY")
        return 1
    print(f"{args.platform}_PRODUCTION_READY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
