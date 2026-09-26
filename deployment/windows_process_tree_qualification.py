"""Strictly read-only qualification of an existing Stage-5 child Job Object."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from deployment.windows_process_tree import MARKER_NAME, SCHEMA_VERSION, _pids, job_name


class ProcessTreeQualificationError(RuntimeError):
    pass


def qualify(marker_path: Path, service_pid: int, *, win32job: Any) -> dict[str, str]:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    required = {"schema_version", "service_pid", "child_pid", "grandchild_pid", "job_name"}
    if set(marker) != required or marker["schema_version"] != SCHEMA_VERSION:
        raise ProcessTreeQualificationError("process-tree marker schema mismatch")
    values = (service_pid, marker["service_pid"], marker["child_pid"], marker["grandchild_pid"])
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
        raise ProcessTreeQualificationError("process-tree PIDs must be positive")
    if marker["service_pid"] != service_pid or len(set(values[1:])) != 3:
        raise ProcessTreeQualificationError("process-tree PID identity mismatch")
    expected_name = job_name(service_pid)
    if marker["job_name"] != expected_name:
        raise ProcessTreeQualificationError("process-tree job name mismatch")
    job = win32job.OpenJobObject(win32job.JOB_OBJECT_QUERY, False, expected_name)
    try:
        limits = win32job.QueryInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation)
        if limits["BasicLimitInformation"]["LimitFlags"] != (
                win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE):
            raise ProcessTreeQualificationError("KILL_ON_JOB_CLOSE limit mismatch")
        observed = _pids(win32job.QueryInformationJobObject(
            job, win32job.JobObjectBasicProcessIdList))
    finally:
        job.Close()
    expected = {marker["child_pid"], marker["grandchild_pid"]}
    if service_pid in observed or observed != expected:
        raise ProcessTreeQualificationError(
            f"job PID set mismatch expected={sorted(expected)} observed={sorted(observed)}")
    return {"WINDOWS_PROCESS_TREE": "PASS"}


def qualify_absent(service_pid: int, *, win32job: Any) -> dict[str, str]:
    """Prove the old named object cannot be opened, without creating it."""
    try:
        handle = win32job.OpenJobObject(win32job.JOB_OBJECT_QUERY, False, job_name(service_pid))
    except Exception as exc:
        if getattr(exc, "winerror", None) == 2:  # ERROR_FILE_NOT_FOUND
            return {"JOB_ABSENT": "PASS"}
        raise ProcessTreeQualificationError("old Job Object absence could not be proven") from exc
    handle.Close()
    raise ProcessTreeQualificationError("old Job Object remains openable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--service-pid", required=True, type=int)
    parser.add_argument("--expect-absent", action="store_true")
    args = parser.parse_args(argv)
    if os.name != "nt":
        print("native Windows Job Objects are required")
        return 1
    import win32job  # type: ignore[import-not-found]
    try:
        result = (qualify_absent(args.service_pid, win32job=win32job)
                  if args.expect_absent else
                  qualify(args.runtime / MARKER_NAME, args.service_pid, win32job=win32job))
    except (OSError, ValueError, TypeError, json.JSONDecodeError,
            ProcessTreeQualificationError) as exc:
        print(str(exc))
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
