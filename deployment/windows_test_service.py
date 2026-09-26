"""Minimal pywin32 SCM lifecycle harness; Windows CI only, never production runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from deployment.windows_strict_service_create import (
    StrictCreateFailure,
    strict_create_service,
)
from deployment.windows_persistent_logging import close_service_logger, configure_service_logger
from deployment.windows_process_tree import (
    ChildJob,
    GATE_NAME,
    MARKER_NAME,
    reviewed_python_executable,
)

if os.name != "nt":
    raise RuntimeError("SCM test service is Windows-only")

import servicemanager  # type: ignore[import-not-found]
import win32api  # type: ignore[import-not-found]
import win32con  # type: ignore[import-not-found]
import win32event  # type: ignore[import-not-found]
import win32job  # type: ignore[import-not-found]
import win32service  # type: ignore[import-not-found]
import win32serviceutil  # type: ignore[import-not-found]


class CryptoHunterBackendTestService(win32serviceutil.ServiceFramework):
    _svc_name_ = "CryptoHunterBackend"
    _svc_display_name_ = "CryptoHunter Backend SCM Acceptance Harness"

    def __init__(self, args: list[str]) -> None:
        super().__init__(args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.marker = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "CryptoHunter" / "scm-health.txt"
        self.machine_root = self.marker.parent
        self.ownership_record = self.machine_root / "windows-acceptance-ownership.json"
        self.tree: ChildJob | None = None
        self.logger = None

    def SvcStop(self) -> None:
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)

    def SvcDoRun(self) -> None:
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        self.marker.write_text(str(os.getpid()), encoding="ascii")
        self.logger = configure_service_logger(self.machine_root / "Logs")
        self.logger.info("SERVICE_START")
        try:
            child_python = reviewed_python_executable(self.ownership_record)
            self.tree = ChildJob(self.machine_root / "Runtime", python_executable=child_python,
                                 win32api=win32api,
                                 win32con=win32con, win32job=win32job)
            self.tree.start()
            self.logger.info("PROCESS_TREE_READY")
            servicemanager.LogInfoMsg("CryptoHunter SCM acceptance harness running")
            win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)
            self.logger.info("SERVICE_STOP")
        finally:
            if self.tree is not None:
                self.tree.close()
            for name in (MARKER_NAME, GATE_NAME):
                (self.machine_root / "Runtime" / name).unlink(missing_ok=True)
            self.marker.unlink(missing_ok=True)
            if self.logger is not None:
                close_service_logger(self.logger)


def acceptance_install_strict(argv: list[str]) -> int:
    """Install once through pywin32 InstallService, with no update fallback."""
    parser = argparse.ArgumentParser(description="Strict create-only SCM acceptance install")
    parser.add_argument("--username", required=True)
    args = parser.parse_args(argv)
    try:
        result = strict_create_service(lambda: win32serviceutil.InstallService(
            win32serviceutil.GetServiceClassString(CryptoHunterBackendTestService),
            CryptoHunterBackendTestService._svc_name_,
            CryptoHunterBackendTestService._svc_display_name_,
            startType=win32service.SERVICE_AUTO_START,
            userName=args.username,
        ))
    except StrictCreateFailure as exc:
        print(json.dumps({"strict_create_result": exc.result, "detail": exc.detail}))
        return 1
    print(json.dumps({"strict_create_result": result}))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "acceptance-install-strict":
        raise SystemExit(acceptance_install_strict(sys.argv[2:]))
    win32serviceutil.HandleCommandLine(CryptoHunterBackendTestService)
