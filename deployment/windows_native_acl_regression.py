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
    required = {"before": False, "after_grant": True, "after_remove": False}
    if any(payload.get(key) is not value for key, value in required.items()):
        raise SystemExit("native ACL read/grant/remove proof failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
