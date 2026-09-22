"""Minimal pywin32 SCM lifecycle harness; Windows CI only, never production runtime."""

from __future__ import annotations

import os
from pathlib import Path

if os.name != "nt":
    raise RuntimeError("SCM test service is Windows-only")

import servicemanager  # type: ignore[import-not-found]
import win32event  # type: ignore[import-not-found]
import win32service  # type: ignore[import-not-found]
import win32serviceutil  # type: ignore[import-not-found]


class CryptoHunterBackendTestService(win32serviceutil.ServiceFramework):
    _svc_name_ = "CryptoHunterBackend"
    _svc_display_name_ = "CryptoHunter Backend SCM Acceptance Harness"

    def __init__(self, args: list[str]) -> None:
        super().__init__(args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.marker = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "CryptoHunter" / "scm-health.txt"

    def SvcStop(self) -> None:
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)

    def SvcDoRun(self) -> None:
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        self.marker.write_text(str(os.getpid()), encoding="ascii")
        servicemanager.LogInfoMsg("CryptoHunter SCM acceptance harness running")
        win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)
        self.marker.unlink(missing_ok=True)


if __name__ == "__main__":
    win32serviceutil.HandleCommandLine(CryptoHunterBackendTestService)
