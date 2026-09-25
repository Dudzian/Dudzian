"""Read-only native Windows path and Stage-4 DACL qualification."""

from __future__ import annotations

import argparse
import json
import ntpath
import os
import stat
from pathlib import Path
from typing import Any

from deployment.platforms.windows import WindowsDeploymentNotQualified, resolve_paths

ADMINISTRATORS_SID = "S-1-5-32-544"
SYSTEM_SID = "S-1-5-18"
ACE_FLAGS = 0x01 | 0x02  # OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE
FILE_ALL_ACCESS = 0x1F01FF
FILE_GENERIC_READ = 0x120089
FILE_GENERIC_WRITE = 0x120116
FILE_GENERIC_EXECUTE = 0x1200A0
DELETE = 0x00010000
SERVICE_NAME = "CryptoHunterBackend"
SERVICE_IDENTITY = rf"NT SERVICE\{SERVICE_NAME}"
SERVICE_PROVEN = "SERVICE_PROVEN"


class WindowsDaclQualificationError(RuntimeError):
    """Native path or security descriptor differs from the frozen contract."""


def windows_path_identity(path: str | Path, win32api: Any) -> str:
    """Return the case-insensitive native identity of a fully expanded path."""
    return ntpath.normcase(win32api.GetFullPathName(str(path))).rstrip("\\")


def is_safe_descendant(path: str | Path, root: str | Path, win32api: Any) -> bool:
    """Check descent with a separator boundary (not a vulnerable prefix check)."""
    child = windows_path_identity(path, win32api)
    parent = windows_path_identity(root, win32api)
    return child.startswith(parent + "\\")


def expected_aces(role: str, service_sid: str) -> set[tuple[str, int, int]]:
    service_mask = FILE_GENERIC_READ | FILE_GENERIC_EXECUTE
    if role in {"STATE", "RUNTIME"}:
        service_mask |= FILE_GENERIC_WRITE | DELETE
    return {
        (ADMINISTRATORS_SID, FILE_ALL_ACCESS, ACE_FLAGS),
        (SYSTEM_SID, FILE_ALL_ACCESS, ACE_FLAGS),
        (service_sid, service_mask, ACE_FLAGS),
    }


def qualify_native_paths(*, win32api: Any, win32file: Any) -> Any:
    """Resolve and prove the live known-folder layout without environment overrides."""
    paths = resolve_paths()
    roots = {
        "ProgramFiles": os.environ.get("ProgramFiles", ""),
        "ProgramData": os.environ.get("ProgramData", ""),
        "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", ""),
    }
    for name, root in roots.items():
        if not root or not ntpath.isabs(root) or not ntpath.splitdrive(root)[0]:
            raise WindowsDaclQualificationError(f"{name} is not fully qualified")
        if "/" in root or not Path(root).is_dir():
            raise WindowsDaclQualificationError(f"{name} is not an existing native directory")
    machine = Path(roots["ProgramData"]) / "CryptoHunter"
    expected = (
        Path(roots["ProgramFiles"]) / "CryptoHunter", machine / "State",
        machine / "Config", machine / "Logs", machine / "Runtime",
        machine / "Updates", Path(roots["LOCALAPPDATA"]) / "CryptoHunter",
    )
    actual = (paths.install, paths.state, paths.configuration, paths.logs, paths.runtime,
              paths.update_staging, paths.gui_user_state)
    if any(windows_path_identity(a, win32api) != windows_path_identity(e, win32api)
           for a, e in zip(actual, expected, strict=True)):
        raise WindowsDaclQualificationError("resolved deployment layout differs from contract")
    if any(not is_safe_descendant(target, machine, win32api)
           for target in (paths.state, paths.configuration, paths.logs, paths.runtime,
                          paths.update_staging)):
        raise WindowsDaclQualificationError("machine path escaped its separator boundary")
    if _is_reparse(machine, win32file):
        raise WindowsDaclQualificationError("machine root is a reparse point")
    return paths


def _is_reparse(path: str | Path, win32file: Any) -> bool:
    return bool(win32file.GetFileAttributes(str(path)) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def qualify_directory_dacl(
    path: str | Path, role: str, service_sid: str, *, win32security: Any,
) -> None:
    """Read and compare one descriptor; this function has no mutation API."""
    info = win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION
    descriptor = win32security.GetNamedSecurityInfo(
        str(path), win32security.SE_FILE_OBJECT, info,
    )
    owner = descriptor.GetSecurityDescriptorOwner()
    if win32security.ConvertSidToStringSid(owner) != ADMINISTRATORS_SID:
        raise WindowsDaclQualificationError(f"{role} owner SID differs")
    control, _ = descriptor.GetSecurityDescriptorControl()
    if not control & win32security.SE_DACL_PRESENT:
        raise WindowsDaclQualificationError(f"{role} DACL is missing")
    if not control & win32security.SE_DACL_PROTECTED:
        raise WindowsDaclQualificationError(f"{role} DACL is not protected")
    dacl = descriptor.GetSecurityDescriptorDacl()
    if dacl is None:
        raise WindowsDaclQualificationError(f"{role} has a null DACL")
    observed: set[tuple[str, int, int]] = set()
    for index in range(dacl.GetAceCount()):
        header, mask, sid = dacl.GetAce(index)
        ace_type, flags = int(header[0]), int(header[1])
        if ace_type != win32security.ACCESS_ALLOWED_ACE_TYPE:
            raise WindowsDaclQualificationError(f"{role} contains a non-ALLOW ACE")
        if flags & win32security.INHERITED_ACE:
            raise WindowsDaclQualificationError(f"{role} contains an inherited ACE")
        observed.add((win32security.ConvertSidToStringSid(sid), int(mask), flags))
    if observed != expected_aces(role, service_sid) or len(observed) != dacl.GetAceCount():
        raise WindowsDaclQualificationError(f"{role} DACL is not the exact allow-list")


def qualify_stage4(paths: Any, service_sid: str, *, win32security: Any) -> dict[str, str]:
    for role, path in (("CONFIG", paths.configuration), ("STATE", paths.state),
                       ("RUNTIME", paths.runtime)):
        qualify_directory_dacl(path, role, service_sid, win32security=win32security)
    return {"CONFIG_DACL": "PASS", "STATE_DACL": "PASS", "RUNTIME_DACL": "PASS"}


def validate_authority(
    record: dict[str, Any], *, run_token: str, win32security: Any,
) -> str:
    """Re-prove the exact service SID from the current-run ownership authority."""
    expected = {
        "run_token": run_token,
        "ownership_phase": SERVICE_PROVEN,
        "strict_create_result": "CREATED",
        "service_name": SERVICE_NAME,
        "service_identity": SERVICE_IDENTITY,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise WindowsDaclQualificationError("read-only qualification requires SERVICE_PROVEN")
    service_sid = record.get("service_sid")
    if not isinstance(service_sid, str) or not service_sid:
        raise WindowsDaclQualificationError("ownership record lacks service_sid")
    sid, _, _ = win32security.LookupAccountName(None, SERVICE_IDENTITY)
    if win32security.ConvertSidToStringSid(sid) != service_sid:
        raise WindowsDaclQualificationError("live service SID mismatches ownership record")
    return service_sid


def qualify_record(
    record: dict[str, Any], *, run_token: str, win32api: Any, win32file: Any,
    win32security: Any,
) -> dict[str, str]:
    """Execute the complete, read-only authority, path and DACL proof."""
    service_sid = validate_authority(
        record, run_token=run_token, win32security=win32security,
    )
    paths = qualify_native_paths(win32api=win32api, win32file=win32file)
    result = qualify_stage4(paths, service_sid, win32security=win32security)
    return {"NATIVE_PATHS": "PASS", **result, "READ_ONLY_QUALIFIER": "PASS"}


def qualify_acl(service_sid: str | None = None) -> dict[str, str]:
    """Public read-only platform boundary using an explicitly proven service SID."""
    if os.name != "nt":
        raise WindowsDeploymentNotQualified("native Windows DACL qualification requires Windows")
    if not service_sid:
        raise WindowsDeploymentNotQualified(
            "proven service SID is required for native DACL qualification"
        )
    import win32api  # type: ignore[import-not-found]
    import win32file  # type: ignore[import-not-found]
    import win32security  # type: ignore[import-not-found]

    paths = qualify_native_paths(win32api=win32api, win32file=win32file)
    return qualify_stage4(paths, service_sid, win32security=win32security)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Windows Stage-4 qualifier")
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
        record = json.loads(args.record.read_text(encoding="utf-8"))
        result = qualify_record(
            record, run_token=args.run_token, win32api=win32api,
            win32file=win32file, win32security=win32security,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError,
            WindowsDaclQualificationError) as exc:
        print(str(exc))
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
