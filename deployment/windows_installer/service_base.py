"""Production-only pywin32 SCM lifecycle boundary."""

from __future__ import annotations
from importlib import resources
import logging
from pathlib import Path
import threading
from typing import Callable, Protocol

from deployment.windows_installer.dependency_contract import (
    AI_DEFAULTS_PACKAGE,
    AI_DEFAULTS_RESOURCE,
    AI_DEFAULTS_SMOKE_MARKER,
    REQUIRED_WIN32_MODULES,
)


def qualify_ai_defaults_resource() -> None:
    """Read and parse the canonical defaults without accepting a fallback."""
    import importlib
    import yaml

    importlib.import_module(AI_DEFAULTS_PACKAGE)
    resource = resources.files(AI_DEFAULTS_PACKAGE).joinpath(AI_DEFAULTS_RESOURCE)
    with resource.open("r", encoding="utf8") as stream:
        defaults = yaml.safe_load(stream)
    if not isinstance(defaults, dict) or not defaults:
        raise RuntimeError(
            f"canonical AI defaults are invalid: {AI_DEFAULTS_PACKAGE}/{AI_DEFAULTS_RESOURCE}"
        )


def build_smoke_requested(argv: list[str] | None = None) -> bool:
    """Run the dependency closure only; never contact SCM or mutate the host."""
    import sys

    arguments = sys.argv[1:] if argv is None else argv
    if arguments != ["--build-smoke"]:
        return False
    import importlib

    import numpy  # noqa: F401

    for module in REQUIRED_WIN32_MODULES:
        importlib.import_module(module)

    # This direct proof cannot silently succeed through config_loader's fallback.
    qualify_ai_defaults_resource()
    print(AI_DEFAULTS_SMOKE_MARKER)
    print("BUILD_SMOKE = PASS")
    return True


class StatusReporter(Protocol):
    def __call__(self, status: int, *, wait_hint: int = 0) -> None: ...


Worker = Callable[[threading.Event, StatusReporter], None]


def run_service(name: str, worker: Worker) -> None:
    """Host a worker which alone decides when SCM may observe RUNNING."""
    import servicemanager  # type: ignore[import-not-found]
    import win32event  # type: ignore[import-not-found]
    import win32service  # type: ignore[import-not-found]
    import win32serviceutil  # type: ignore[import-not-found]

    class ProductionService(win32serviceutil.ServiceFramework):
        _svc_name_ = name
        _svc_display_name_ = name

        def __init__(self, args: list[str]) -> None:
            super().__init__(args)
            self.stop_event = threading.Event()
            self.stopped = win32event.CreateEvent(None, 1, 0, None)

        def report(self, status: int, *, wait_hint: int = 0) -> None:
            self.ReportServiceStatus(status, waitHint=wait_hint)

        def SvcRun(self) -> None:
            # ServiceFramework's default reports RUNNING before SvcDoRun.  The
            # production boundary deliberately keeps START_PENDING instead.
            self.report(win32service.SERVICE_START_PENDING, wait_hint=60_000)
            try:
                worker(self.stop_event, self.report)
            finally:
                self.report(win32service.SERVICE_STOPPED)

        def SvcStop(self) -> None:
            self.report(win32service.SERVICE_STOP_PENDING, wait_hint=30_000)
            self.stop_event.set()
            win32event.SetEvent(self.stopped)

        def SvcShutdown(self) -> None:
            self.SvcStop()

        def SvcDoRun(self) -> None:
            raise RuntimeError("SvcRun owns the explicit readiness lifecycle")

    servicemanager.Initialize()
    servicemanager.PrepareToHostSingle(ProductionService)
    servicemanager.StartServiceCtrlDispatcher()


def machine_root() -> Path:
    import os

    root = os.environ.get("ProgramData")
    if not root:
        raise RuntimeError("ProgramData is unavailable")
    return Path(root) / "CryptoHunter"


def service_logger(name: str) -> logging.Logger:
    from deployment.windows_persistent_logging import configure_service_logger

    logger = configure_service_logger(machine_root() / "Logs")
    logger.name = name
    return logger
