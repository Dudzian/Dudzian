"""Administrator-owned provision/cleanup for the independent Stage-5 Logs boundary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from deployment.windows_dacl_provision import _save_record, _validate_record
from deployment.windows_dacl_qualification import (
    ADMINISTRATORS_SID, ACE_FLAGS, DELETE, FILE_ALL_ACCESS, FILE_GENERIC_EXECUTE,
    FILE_GENERIC_READ, FILE_GENERIC_WRITE, SYSTEM_SID, _is_reparse,
    qualify_native_paths, windows_path_identity,
)
from deployment.windows_persistent_logging import ALLOWED_LOG_FILES

SENTINEL = ".stage5-logging-ownership.json"


class WindowsLoggingProvisionError(RuntimeError):
    pass


def expected_logging_aces(service_sid: str) -> set[tuple[str, int, int]]:
    return {(ADMINISTRATORS_SID, FILE_ALL_ACCESS, ACE_FLAGS),
            (SYSTEM_SID, FILE_ALL_ACCESS, ACE_FLAGS),
            (service_sid, FILE_GENERIC_READ | FILE_GENERIC_WRITE |
             FILE_GENERIC_EXECUTE | DELETE, ACE_FLAGS)}


def provision(record_path: Path, run_token: str, *, win32api: Any, win32file: Any,
              win32security: Any) -> None:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    sid = _validate_record(record, run_token, win32security)
    logs = qualify_native_paths(win32api=win32api, win32file=win32file).logs
    if logs.exists():
        raise WindowsLoggingProvisionError("Stage-5 Logs target pre-exists")
    canonical = windows_path_identity(logs, win32api)
    record["logging_security_plan"] = {
        "role": "LOGS", "canonical_path": canonical, "service_sid": sid}
    _save_record(record_path, record)
    logs.mkdir()
    sentinel = {"run_token": run_token, "canonical_path": canonical,
                "service_sid": sid, "role": "LOGS"}
    (logs / SENTINEL).write_text(json.dumps(sentinel, separators=(",", ":")), encoding="utf-8")
    dacl = win32security.ACL()
    for principal, mask, flags in expected_logging_aces(sid):
        obj = win32security.ConvertStringSidToSid(principal)
        dacl.AddAccessAllowedAceEx(win32security.ACL_REVISION_DS, flags, mask, obj)
    owner = win32security.ConvertStringSidToSid(ADMINISTRATORS_SID)
    info = (win32security.OWNER_SECURITY_INFORMATION |
            win32security.DACL_SECURITY_INFORMATION |
            win32security.PROTECTED_DACL_SECURITY_INFORMATION)
    win32security.SetNamedSecurityInfo(str(logs), win32security.SE_FILE_OBJECT,
                                      info, owner, None, dacl, None)


def cleanup(record_path: Path, run_token: str, *, win32api: Any, win32file: Any) -> None:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    plan = record.get("logging_security_plan")
    if (record.get("run_token") != run_token or
            record.get("ownership_phase") != "SERVICE_PROVEN" or not isinstance(plan, dict) or
            plan.get("service_sid") != record.get("service_sid")):
        raise WindowsLoggingProvisionError("logging cleanup authority mismatch")
    logs = Path(plan.get("canonical_path", ""))
    if windows_path_identity(logs, win32api) != plan.get("canonical_path"):
        raise WindowsLoggingProvisionError("logging cleanup path mismatch")
    if logs.exists():
        if _is_reparse(logs, win32file) or not logs.is_dir():
            raise WindowsLoggingProvisionError("Logs is not an owned regular directory")
        expected_sentinel = {"run_token": run_token, "canonical_path": plan["canonical_path"],
                             "service_sid": plan["service_sid"], "role": "LOGS"}
        sentinel = logs / SENTINEL
        if json.loads(sentinel.read_text(encoding="utf-8")) != expected_sentinel:
            raise WindowsLoggingProvisionError("logging sentinel mismatch")
        names = {item.name for item in logs.iterdir()}
        if not names <= (set(ALLOWED_LOG_FILES) | {SENTINEL}):
            raise WindowsLoggingProvisionError("foreign content in Logs")
        for item in logs.iterdir():
            if item.name != SENTINEL:
                if _is_reparse(item, win32file) or not item.is_file():
                    raise WindowsLoggingProvisionError("log family contains non-regular file")
        for item in logs.iterdir():
            item.unlink()
        logs.rmdir()
    record.pop("logging_security_plan")
    _save_record(record_path, record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("provision", "cleanup"))
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--run-token", required=True)
    args = parser.parse_args(argv)
    if os.name != "nt": return 1
    import win32api  # type: ignore[import-not-found]
    import win32file  # type: ignore[import-not-found]
    import win32security  # type: ignore[import-not-found]
    try:
        if args.command == "provision":
            provision(args.record, args.run_token, win32api=win32api,
                      win32file=win32file, win32security=win32security)
        else:
            cleanup(args.record, args.run_token, win32api=win32api, win32file=win32file)
    except Exception as exc:
        print(str(exc)); return 1
    print(json.dumps({"LOGS": "PASS"})); return 0


if __name__ == "__main__": raise SystemExit(main())
