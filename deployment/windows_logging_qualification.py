"""Read-only native qualification of the separate Stage-5 Logs boundary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from deployment.windows_dacl_qualification import (
    _is_reparse, qualify_directory_dacl, qualify_native_paths, validate_authority,
    windows_path_identity,
)


class WindowsLoggingQualificationError(RuntimeError):
    pass


def qualify_record(record: dict[str, Any], *, run_token: str, win32api: Any,
                   win32file: Any, win32security: Any) -> dict[str, str]:
    service_sid = validate_authority(record, run_token=run_token, win32security=win32security)
    paths = qualify_native_paths(win32api=win32api, win32file=win32file)
    plan = record.get("logging_security_plan")
    canonical = windows_path_identity(paths.logs, win32api)
    if not isinstance(plan, dict) or plan != {
        "role": "LOGS", "canonical_path": canonical, "service_sid": service_sid,
    }:
        raise WindowsLoggingQualificationError("logging security plan mismatch")
    if not paths.logs.is_dir() or _is_reparse(paths.logs, win32file):
        raise WindowsLoggingQualificationError("Logs is not a regular native directory")
    qualify_directory_dacl(paths.logs, "LOGS", service_sid, win32security=win32security)
    return {"LOGS_DACL": "PASS", "READ_ONLY_QUALIFIER": "PASS"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
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
        result = qualify_record(json.loads(args.record.read_text(encoding="utf-8")),
                                run_token=args.run_token, win32api=win32api,
                                win32file=win32file, win32security=win32security)
    except Exception as exc:
        print(str(exc))
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
