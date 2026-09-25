"""Live proof of the pwsh -> Python -> Windows PowerShell ACL boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def main() -> int:
    if os.environ.get("WINDOWS_ACL_REGRESSION_PARENT_EDITION") != "Core":
        raise SystemExit("ACL regression must be launched by pwsh")
    script = Path(__file__).with_name("windows_native_acl_regression.ps1")
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode:
        if completed.stderr:
            print(completed.stderr, end="")
        return completed.returncode
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    if payload.get("child_edition") != "Desktop":
        raise SystemExit("ACL child is not native Windows PowerShell")
    if payload.get("get_acl_command_type") != "Cmdlet":
        raise SystemExit("Get-Acl is not a cmdlet")
    if payload.get("get_acl_module_name") != "Microsoft.PowerShell.Security":
        raise SystemExit("Get-Acl has an unexpected module name")
    if payload.get("get_acl_implementing_type") != "Microsoft.PowerShell.Commands.GetAclCommand":
        raise SystemExit("Get-Acl has an unexpected implementing type")
    expected_assembly = (
        "Microsoft.PowerShell.Security, Version=3.0.0.0, Culture=neutral, "
        "PublicKeyToken=31bf3856ad364e35"
    )
    if payload.get("get_acl_assembly_full_name") != expected_assembly:
        raise SystemExit("Get-Acl has an unexpected strong assembly identity")
    if not payload.get("get_acl_assembly_location"):
        raise SystemExit("Get-Acl assembly location is unavailable")
    if payload.get("get_acl_assembly_file_exists") is not True:
        raise SystemExit("Get-Acl assembly location is not file-backed")
    if payload.get("get_acl_global_assembly_cache") is not True:
        raise SystemExit("Get-Acl assembly is not in the Global Assembly Cache")
    if payload.get("security_executable_authority_qualified") is not True:
        raise SystemExit("native Security executable authority is not qualified")
    required = {"before": False, "after_grant": True, "after_remove": False}
    if any(payload.get(key) is not value for key, value in required.items()):
        raise SystemExit("native ACL read/grant/remove proof failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
