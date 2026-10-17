"""Production-only pywin32 SCM lifecycle boundary."""

from __future__ import annotations
from contextlib import contextmanager
from importlib import resources
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Iterator, Protocol

from deployment.windows_installer.dependency_contract import (
    AI_DEFAULTS_PACKAGE,
    AI_DEFAULTS_RESOURCE,
    AI_DEFAULTS_SMOKE_MARKER,
    REQUIRED_WIN32_MODULES,
)
from deployment.windows_installer.contract import CONTRACT


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


def _confirmed_service_evidence_enrollment() -> None:
    """Require the still-missing service-local committed enrollment adapter."""
    from deployment.windows_installer.corehost_composition import (
        WindowsProvisioningAdapterUnavailable,
        load_windows_external_provisioning_handoff,
        resolve_accepted_handoff,
    )

    provisioning = load_windows_external_provisioning_handoff()
    resolve_accepted_handoff(provisioning)
    # An accepted external claim alone cannot attest committed enrollment to
    # both service identities. The administrator's ownership/finalization and
    # ProductionTrust authorities are not service-readable admission evidence.
    # Do not broaden those ACLs or use local markers. Remain BLOCKED until an
    # approved runtime admission integration provides this missing authority.
    raise WindowsProvisioningAdapterUnavailable(
        "SERVICE_LOCAL_COMMITTED_ENROLLMENT_AUTHORITY_UNAVAILABLE"
    )


def _serve_service_evidence(name: str, stop: threading.Event) -> None:
    """Contain evidence failures without changing the product's readiness."""
    try:
        if stop.is_set():
            return
        _confirmed_service_evidence_enrollment()
        if stop.is_set():
            return
        # Static imports keep the frozen service dependency closure discoverable.
        from deployment.windows_stage9_evidence_pipe import serve
        from deployment.windows_stage9_service_sql import collect_service_sql

        @contextmanager
        def collect(request: dict[str, Any]) -> Iterator[dict[str, Any]]:
            # Re-resolve authority for each request; an earlier endpoint start
            # is not continuing legal enrollment evidence.
            _confirmed_service_evidence_enrollment()
            with collect_service_sql(name, request, stop) as evidence:
                yield evidence

        serve(name, stop, collect)
    except Exception:
        # Exceptions may contain credentials, paths or request material.
        logging.getLogger(name).warning("Stage 9 service evidence endpoint BLOCKED")


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
            self.evidence_stop = threading.Event()
            self.evidence_thread: threading.Thread | None = None
            self.evidence_started = False
            self.stopped = win32event.CreateEvent(None, 1, 0, None)

        def report(self, status: int, *, wait_hint: int = 0) -> None:
            self.ReportServiceStatus(status, waitHint=wait_hint)
            if (
                status == win32service.SERVICE_RUNNING
                and name in (CONTRACT.backend_service, CONTRACT.verifier_service)
                and not self.evidence_started
                and not self.stop_event.is_set()
            ):
                self.evidence_started = True
                thread = threading.Thread(
                    target=_serve_service_evidence,
                    args=(name, self.evidence_stop),
                    name=f"{name}-stage9-evidence",
                    daemon=False,
                )
                try:
                    thread.start()
                except Exception:
                    logging.getLogger(name).warning("Stage 9 service evidence endpoint BLOCKED")
                else:
                    self.evidence_thread = thread

        def SvcRun(self) -> None:
            # ServiceFramework's default reports RUNNING before SvcDoRun.  The
            # production boundary deliberately keeps START_PENDING instead.
            self.report(win32service.SERVICE_START_PENDING, wait_hint=60_000)
            try:
                worker(self.stop_event, self.report)
            finally:
                self.evidence_stop.set()
                if self.evidence_thread is not None:
                    # The native pipe is cancellable; SQL has a five-second
                    # connect/query bound and checks cancellation between calls.
                    self.evidence_thread.join(timeout=7.0)
                    if self.evidence_thread.is_alive():
                        raise RuntimeError("Stage 9 service evidence endpoint cleanup unavailable")
                self.report(win32service.SERVICE_STOPPED)

        def SvcStop(self) -> None:
            self.report(win32service.SERVICE_STOP_PENDING, wait_hint=30_000)
            self.stop_event.set()
            self.evidence_stop.set()
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
