"""Deterministic ownership transition for the reviewed Windows SCM acceptance probe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

INTENT_CREATED = "INTENT_CREATED"
SERVICE_PROVEN = "SERVICE_PROVEN"


class OwnershipQualificationError(RuntimeError):
    """The observed SCM object is not proven to have been created by this run."""


@dataclass(frozen=True)
class ServiceObservation:
    name: str
    start_name: str
    path_name: str
    service_host: str
    service_sid: str


@dataclass(frozen=True)
class CleanupPolicy:
    remove_service: bool
    remove_health_marker: bool
    ownership_failure: bool


def service_cleanup_allowed(record: dict[str, Any]) -> bool:
    """Only the terminal creation-proof phase authorizes destructive SCM cleanup."""
    return record.get("ownership_phase") == SERVICE_PROVEN


def health_marker_cleanup_allowed(record: dict[str, Any]) -> bool:
    """A marker is owned only after the run has proven its service lifecycle."""
    return record.get("ownership_phase") == SERVICE_PROVEN


def cleanup_policy(
    record: dict[str, Any], *, service_exists: bool, health_marker_exists: bool,
) -> CleanupPolicy:
    """Plan destructive cleanup without adopting resources during the intent phase."""
    proven = service_cleanup_allowed(record)
    return CleanupPolicy(
        remove_service=proven and service_exists,
        remove_health_marker=proven and health_marker_exists,
        ownership_failure=not proven and (service_exists or health_marker_exists),
    )


def qualify_service_creation(
    record: dict[str, Any], *, run_token: str, python_executable: str, harness: str,
    strict_create_result: str, observation: ServiceObservation,
) -> dict[str, Any]:
    """Return a SERVICE_PROVEN record only after successful install and exact qualification."""
    required_record = {
        "run_token": run_token,
        "service_name": "CryptoHunterBackend",
        "service_identity": r"NT SERVICE\CryptoHunterBackend",
        "python_executable": python_executable,
        "harness": harness,
        "ownership_phase": INTENT_CREATED,
    }
    if any(record.get(key) != value for key, value in required_record.items()):
        raise OwnershipQualificationError("ownership intent record does not match this run")
    if strict_create_result != "CREATED":
        raise OwnershipQualificationError("strict create did not prove service creation")
    if observation.name != required_record["service_name"]:
        raise OwnershipQualificationError("SCM service name mismatch")
    if observation.start_name != required_record["service_identity"]:
        raise OwnershipQualificationError("SCM service identity mismatch")
    if not observation.path_name or not observation.service_host:
        raise OwnershipQualificationError("SCM service path or host is missing")
    if not observation.service_sid:
        raise OwnershipQualificationError("SCM service SID is missing")
    proven = dict(record)
    proven.update({
        "ownership_phase": SERVICE_PROVEN,
        "strict_create_result": strict_create_result,
        "service_path_name": observation.path_name,
        "service_host": observation.service_host,
        "service_sid": observation.service_sid,
    })
    return proven


def write_record(path: Path, record: dict[str, Any]) -> None:
    """Replace a record only after the complete proven document has been serialized."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged_path = path.with_name(f"{path.name}.{record['run_token']}.prove.tmp")
    try:
        staged_path.write_text(
            json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8",
        )
        staged_path.replace(path)
    finally:
        staged_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prove current-run Windows SCM ownership")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--strict-create-result", required=True)
    parser.add_argument("--observed-name", required=True)
    parser.add_argument("--observed-start-name", required=True)
    parser.add_argument("--observed-path-name", required=True)
    parser.add_argument("--observed-service-host", required=True)
    parser.add_argument("--observed-service-sid", required=True)
    args = parser.parse_args(argv)
    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
        proven = qualify_service_creation(
            record, run_token=args.run_token, python_executable=args.python_executable,
            harness=args.harness, strict_create_result=args.strict_create_result,
            observation=ServiceObservation(
                name=args.observed_name, start_name=args.observed_start_name,
                path_name=args.observed_path_name, service_host=args.observed_service_host,
                service_sid=args.observed_service_sid,
            ),
        )
        write_record(args.record, proven)
    except (OSError, json.JSONDecodeError, OwnershipQualificationError) as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
