"""Administrator-only Stage-4 protected-DACL provisioner and owned cleanup."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from deployment.windows_dacl_qualification import (
    ACE_FLAGS, ADMINISTRATORS_SID, expected_aces, qualify_native_paths,
    windows_path_identity, _is_reparse,
)
from deployment.windows_service_ownership import SERVICE_PROVEN

SERVICE_NAME = "CryptoHunterBackend"
SERVICE_IDENTITY = r"NT SERVICE\CryptoHunterBackend"
SENTINEL = ".stage4-ownership.json"


class WindowsDaclProvisionError(RuntimeError):
    """Provisioning or cleanup could not prove exclusive current-run ownership."""


def _validate_record(record: dict[str, Any], run_token: str, win32security: Any) -> str:
    required = {"run_token": run_token, "ownership_phase": SERVICE_PROVEN,
                "strict_create_result": "CREATED", "service_name": SERVICE_NAME,
                "service_identity": SERVICE_IDENTITY}
    if any(record.get(key) != value for key, value in required.items()):
        raise WindowsDaclProvisionError("Stage-4 requires exact SERVICE_PROVEN ownership")
    service_sid = record.get("service_sid")
    if not isinstance(service_sid, str) or not service_sid:
        raise WindowsDaclProvisionError("ownership record lacks service_sid")
    sid, _, _ = win32security.LookupAccountName(None, SERVICE_IDENTITY)
    if win32security.ConvertSidToStringSid(sid) != service_sid:
        raise WindowsDaclProvisionError("live service SID mismatches ownership record")
    return service_sid


def _plan(paths: Any, service_sid: str, win32api: Any) -> dict[str, Any]:
    targets = [{"role": role, "path": windows_path_identity(path, win32api)} for role, path in (
        ("CONFIG", paths.configuration), ("STATE", paths.state), ("RUNTIME", paths.runtime))]
    return {"targets": targets, "service_sid": service_sid, "owner_sid": ADMINISTRATORS_SID,
            "config_policy": [list(ace) for ace in sorted(expected_aces("CONFIG", service_sid))],
            "state_policy": [list(ace) for ace in sorted(expected_aces("STATE", service_sid))]}


def _save_record(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".stage4.tmp")
    temporary.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def provision(record_path: Path, run_token: str, *, win32api: Any, win32file: Any,
              win32security: Any) -> dict[str, str]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    service_sid = _validate_record(record, run_token, win32security)
    paths = qualify_native_paths(win32api=win32api, win32file=win32file)
    targets = (("CONFIG", paths.configuration), ("STATE", paths.state),
               ("RUNTIME", paths.runtime))
    machine = paths.state.parent
    if _is_reparse(machine, win32file):
        raise WindowsDaclProvisionError("machine root is a reparse point")
    for _, target in targets:
        if target.exists():
            if _is_reparse(target, win32file):
                raise WindowsDaclProvisionError("pre-existing target is a reparse point")
            raise WindowsDaclProvisionError(f"Stage-4 target pre-exists: {target}")
    record["path_security_plan"] = _plan(paths, service_sid, win32api)
    _save_record(record_path, record)  # plan is durable before the first mutation
    owner = win32security.ConvertStringSidToSid(ADMINISTRATORS_SID)
    sid_objects = {sid: win32security.ConvertStringSidToSid(sid)
                   for sid in (ADMINISTRATORS_SID, "S-1-5-18", service_sid)}
    for role, target in targets:
        target.mkdir()  # atomic create; deliberately no exist_ok
        sentinel = {"run_token": run_token, "canonical_path": windows_path_identity(target, win32api),
                    "service_sid": service_sid, "role": role}
        (target / SENTINEL).write_text(json.dumps(sentinel, separators=(",", ":")), encoding="utf-8")
        dacl = win32security.ACL()
        for sid, mask, flags in expected_aces(role, service_sid):
            dacl.AddAccessAllowedAceEx(
                win32security.ACL_REVISION_DS, flags, mask, sid_objects[sid]
            )
        info = (win32security.OWNER_SECURITY_INFORMATION |
                win32security.DACL_SECURITY_INFORMATION |
                win32security.PROTECTED_DACL_SECURITY_INFORMATION)
        win32security.SetNamedSecurityInfo(str(target), win32security.SE_FILE_OBJECT,
                                           info, owner, None, dacl, None)
    return {
        "CONFIG_PROVISIONED": "PASS",
        "STATE_PROVISIONED": "PASS",
        "RUNTIME_PROVISIONED": "PASS",
    }


def cleanup(record_path: Path, run_token: str, *, win32api: Any, win32file: Any) -> None:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    plan = record.get("path_security_plan")
    if (record.get("run_token") != run_token or record.get("ownership_phase") != SERVICE_PROVEN
            or not isinstance(plan, dict) or plan.get("service_sid") != record.get("service_sid")):
        raise WindowsDaclProvisionError("Stage-4 cleanup ownership record mismatch")
    paths = qualify_native_paths(win32api=win32api, win32file=win32file)
    if plan != _plan(paths, record["service_sid"], win32api):
        raise WindowsDaclProvisionError("Stage-4 cleanup plan differs from native paths")
    targets = plan.get("targets")
    if not isinstance(targets, list) or {item.get("role") for item in targets} != {
        "CONFIG", "STATE", "RUNTIME"}:
        raise WindowsDaclProvisionError("Stage-4 cleanup plan mismatch")
    # Validate every directory before deleting any of them.
    checked: list[tuple[Path, Path]] = []
    for item in targets:
        target = Path(item["path"])
        if windows_path_identity(target, win32api) != item["path"]:
            raise WindowsDaclProvisionError("Stage-4 cleanup target mismatch")
        if not target.exists():
            continue  # planned but not yet created after an interrupted provision
        if _is_reparse(target, win32file):
            raise WindowsDaclProvisionError("Stage-4 cleanup target is a reparse point")
        if not target.is_dir():
            raise WindowsDaclProvisionError("Stage-4 cleanup target is not a directory")
        sentinel_path = target / SENTINEL
        if {child.name for child in target.iterdir()} != {SENTINEL}:
            raise WindowsDaclProvisionError("Stage-4 cleanup found foreign content")
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
        expected = {"run_token": run_token, "canonical_path": item["path"],
                    "service_sid": record["service_sid"], "role": item["role"]}
        if sentinel != expected:
            raise WindowsDaclProvisionError("Stage-4 cleanup sentinel mismatch")
        checked.append((target, sentinel_path))
    for target, sentinel in checked:
        sentinel.unlink()
        target.rmdir()
    record.pop("path_security_plan")
    _save_record(record_path, record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("provision", "cleanup"))
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--run-token", required=True)
    args = parser.parse_args(argv)
    if os.name != "nt":
        print("native Windows security APIs are required")
        return 1
    import win32api  # type: ignore[import-not-found]
    import win32file  # type: ignore[import-not-found]
    import win32security  # type: ignore[import-not-found]
    try:
        if args.command == "provision":
            print(json.dumps(provision(args.record, args.run_token, win32api=win32api,
                                       win32file=win32file, win32security=win32security)))
        else:
            cleanup(args.record, args.run_token, win32api=win32api, win32file=win32file)
    except (OSError, ValueError, json.JSONDecodeError, WindowsDaclProvisionError) as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
