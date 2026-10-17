"""Emulated lifecycle boundaries; these tests do not qualify Windows services."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from deployment.windows_installer import service_base
from deployment.windows_installer.contract import CONTRACT


START_PENDING, RUNNING, STOP_PENDING, STOPPED = 2, 4, 3, 1


def emulated_service(monkeypatch: pytest.MonkeyPatch, name: str, worker: Any) -> Any:
    hosted: list[Any] = []

    class ServiceFramework:
        def __init__(self, args: list[str]) -> None:
            self.statuses: list[int] = []

        def ReportServiceStatus(self, status: int, *, waitHint: int) -> None:
            self.statuses.append(status)

    modules = {
        "servicemanager": SimpleNamespace(
            Initialize=lambda: None,
            PrepareToHostSingle=hosted.append,
            StartServiceCtrlDispatcher=lambda: None,
        ),
        "win32event": SimpleNamespace(
            CreateEvent=lambda *args: threading.Event(), SetEvent=lambda event: event.set()
        ),
        "win32service": SimpleNamespace(
            SERVICE_START_PENDING=START_PENDING,
            SERVICE_RUNNING=RUNNING,
            SERVICE_STOP_PENDING=STOP_PENDING,
            SERVICE_STOPPED=STOPPED,
        ),
        "win32serviceutil": SimpleNamespace(ServiceFramework=ServiceFramework),
    }
    for module_name, module in modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)
    service_base.run_service(name, worker)
    return hosted[0]([])


@pytest.mark.parametrize("name", [CONTRACT.backend_service, CONTRACT.verifier_service])
def test_endpoint_starts_only_after_running_and_cleans_up_before_stopped(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    entered, cleaned = threading.Event(), threading.Event()
    observed: list[Any] = []

    def endpoint(service_name: str, stop: threading.Event) -> None:
        observed.append(
            (
                service_name,
                list(service.statuses),
                threading.current_thread().name,
                threading.current_thread().daemon,
            )
        )
        entered.set()
        assert stop.wait(1.0)
        cleaned.set()

    def worker(stop: threading.Event, report: Any) -> None:
        assert not entered.is_set()
        assert service.statuses == [START_PENDING]
        report(RUNNING)
        assert entered.wait(1.0)
        report(RUNNING)

    monkeypatch.setattr(service_base, "_serve_service_evidence", endpoint)
    service = emulated_service(monkeypatch, name, worker)
    report_status = service.ReportServiceStatus

    def report_after_cleanup(status: int, *, waitHint: int) -> None:
        if status == STOPPED:
            assert cleaned.is_set()
        report_status(status, waitHint=waitHint)

    service.ReportServiceStatus = report_after_cleanup
    service.SvcRun()
    assert cleaned.is_set()
    assert service.statuses == [START_PENDING, RUNNING, RUNNING, STOPPED]
    assert observed == [(name, [START_PENDING, RUNNING], f"{name}-stage9-evidence", False)]
    assert service.evidence_thread is not None and not service.evidence_thread.is_alive()
    assert service.evidence_stop.is_set()


@pytest.mark.parametrize("name", [CONTRACT.backend_service, CONTRACT.verifier_service])
def test_start_pending_never_exposes_endpoint(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    def forbidden(*args: Any) -> None:
        pytest.fail("endpoint started before worker readiness")

    monkeypatch.setattr(service_base, "_serve_service_evidence", forbidden)
    service = emulated_service(monkeypatch, name, lambda stop, report: None)
    service.SvcRun()
    assert service.statuses == [START_PENDING, STOPPED]
    assert service.evidence_thread is None


@pytest.mark.parametrize("name", [CONTRACT.postgresql_service, "UnrelatedService"])
def test_noncanonical_service_never_starts_endpoint(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setattr(
        service_base,
        "_serve_service_evidence",
        lambda *args: pytest.fail("noncanonical service must not expose endpoint"),
    )
    service = emulated_service(monkeypatch, name, lambda stop, report: report(RUNNING))
    service.SvcRun()
    assert service.statuses == [START_PENDING, RUNNING, STOPPED]
    assert service.evidence_thread is None


@pytest.mark.parametrize("stop_method", ["SvcStop", "SvcShutdown"])
def test_stop_and_shutdown_cancel_both_workers(
    monkeypatch: pytest.MonkeyPatch, stop_method: str
) -> None:
    service = emulated_service(monkeypatch, CONTRACT.backend_service, lambda stop, report: None)
    getattr(service, stop_method)()
    assert service.stop_event.is_set() and service.evidence_stop.is_set()
    assert service.stopped.is_set()
    assert service.statuses == [STOP_PENDING]


def test_worker_failure_cancels_endpoint_before_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, cleaned = threading.Event(), threading.Event()

    def endpoint(name: str, stop: threading.Event) -> None:
        entered.set()
        assert stop.wait(1.0)
        cleaned.set()

    def worker(stop: threading.Event, report: Any) -> None:
        report(RUNNING)
        assert entered.wait(1.0)
        raise ValueError("worker failure")

    monkeypatch.setattr(service_base, "_serve_service_evidence", endpoint)
    service = emulated_service(monkeypatch, CONTRACT.backend_service, worker)
    with pytest.raises(ValueError, match="worker failure"):
        service.SvcRun()
    assert cleaned.is_set()
    assert service.statuses[-1] == STOPPED


def test_unclean_endpoint_does_not_report_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    joined: list[float] = []
    thread = SimpleNamespace(join=lambda *, timeout: joined.append(timeout), is_alive=lambda: True)
    service = emulated_service(monkeypatch, CONTRACT.backend_service, lambda stop, report: None)
    service.evidence_thread = thread
    with pytest.raises(RuntimeError, match="cleanup unavailable"):
        service.SvcRun()
    assert joined == [7.0]
    assert service.evidence_stop.is_set()
    assert service.statuses == [START_PENDING]


class EmulatedAdmissionUnavailable(RuntimeError):
    """Test-only adapter failure; this is not a production admission authority."""


@pytest.mark.parametrize("failure_stage", ["load", "resolve"])
def test_absent_or_rejected_external_authority_blocks_service_admission(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    events: list[str] = []
    handoff = object()

    def load() -> object:
        events.append("load")
        if failure_stage == "load":
            raise EmulatedAdmissionUnavailable("external authority unavailable")
        return handoff

    def resolve(received: object) -> None:
        assert received is handoff
        events.append("resolve")
        raise EmulatedAdmissionUnavailable("external authority rejected")

    monkeypatch.setitem(
        sys.modules,
        "deployment.windows_installer.corehost_composition",
        SimpleNamespace(
            WindowsProvisioningAdapterUnavailable=EmulatedAdmissionUnavailable,
            load_windows_external_provisioning_handoff=load,
            resolve_accepted_handoff=resolve,
        ),
    )
    with pytest.raises(EmulatedAdmissionUnavailable, match="external authority"):
        service_base._confirmed_service_evidence_enrollment()
    assert events == (["load"] if failure_stage == "load" else ["load", "resolve"])


def test_emulated_accepted_handoff_still_needs_service_local_committed_authority(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    events: list[str] = []
    handoff = object()

    def load() -> object:
        events.append("load")
        return handoff

    def resolve(received: object) -> None:
        assert received is handoff
        events.append("resolve")

    def protected_files_forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("service admission must not read administrator-owned enrollment files")

    monkeypatch.setattr(service_base, "Path", protected_files_forbidden)
    monkeypatch.setattr(service_base, "machine_root", protected_files_forbidden)
    monkeypatch.setitem(
        sys.modules,
        "deployment.windows_installer.corehost_composition",
        SimpleNamespace(
            WindowsProvisioningAdapterUnavailable=EmulatedAdmissionUnavailable,
            load_windows_external_provisioning_handoff=load,
            resolve_accepted_handoff=resolve,
        ),
    )
    with pytest.raises(
        EmulatedAdmissionUnavailable,
        match="^SERVICE_LOCAL_COMMITTED_ENROLLMENT_AUTHORITY_UNAVAILABLE$",
    ):
        service_base._confirmed_service_evidence_enrollment()
    assert events == ["load", "resolve"]
    service_base._serve_service_evidence(CONTRACT.backend_service, threading.Event())
    assert events == ["load", "resolve", "load", "resolve"]
    assert "endpoint BLOCKED" in caplog.text


def test_unavailable_external_authority_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def unavailable() -> None:
        raise RuntimeError("secret credential and request detail")

    monkeypatch.setitem(
        sys.modules,
        "deployment.windows_installer.corehost_composition",
        SimpleNamespace(
            WindowsProvisioningAdapterUnavailable=EmulatedAdmissionUnavailable,
            load_windows_external_provisioning_handoff=unavailable,
            resolve_accepted_handoff=lambda handoff: pytest.fail("missing authority resolved"),
        ),
    )
    caplog.set_level(logging.WARNING)
    service_base._serve_service_evidence(CONTRACT.backend_service, threading.Event())
    assert "endpoint BLOCKED" in caplog.text
    assert "secret credential" not in caplog.text


def test_endpoint_rechecks_legal_authority_before_each_collection(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    events: list[str] = []
    stop = threading.Event()
    request = {"test": "emulated authenticated request"}

    def confirm() -> None:
        events.append("authority")

    @contextmanager
    def sql(name: str, received: dict[str, Any], received_stop: threading.Event):
        assert name == CONTRACT.verifier_service and received is request and received_stop is stop
        events.append("sql_open")
        try:
            yield {"emulated": True}
        finally:
            events.append("sql_close")

    def serve(name: str, received_stop: threading.Event, handler: Any) -> None:
        assert name == CONTRACT.verifier_service and received_stop is stop
        events.append("endpoint")
        for _ in range(2):
            with handler(request) as response:
                assert response == {"emulated": True}
                events.append("acknowledgement")

    monkeypatch.setattr(service_base, "_confirmed_service_evidence_enrollment", confirm)
    monkeypatch.setitem(
        sys.modules, "deployment.windows_stage9_evidence_pipe", SimpleNamespace(serve=serve)
    )
    monkeypatch.setitem(
        sys.modules,
        "deployment.windows_stage9_service_sql",
        SimpleNamespace(collect_service_sql=sql),
    )
    service_base._serve_service_evidence(CONTRACT.verifier_service, stop)
    assert events == [
        "authority",
        "endpoint",
        "authority",
        "sql_open",
        "acknowledgement",
        "sql_close",
        "authority",
        "sql_open",
        "acknowledgement",
        "sql_close",
    ]
    assert "BLOCKED" not in caplog.text


def test_revoked_authority_blocks_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    checks: list[bool] = []

    def confirm() -> None:
        checks.append(True)
        if len(checks) > 1:
            raise RuntimeError("authority no longer available")

    def serve(name: str, stop: threading.Event, handler: Any) -> None:
        with handler({}):
            pytest.fail("revoked authority entered collection")

    monkeypatch.setattr(service_base, "_confirmed_service_evidence_enrollment", confirm)
    monkeypatch.setitem(
        sys.modules, "deployment.windows_stage9_evidence_pipe", SimpleNamespace(serve=serve)
    )
    monkeypatch.setitem(
        sys.modules,
        "deployment.windows_stage9_service_sql",
        SimpleNamespace(
            collect_service_sql=lambda *args: pytest.fail("revoked authority used SQL")
        ),
    )
    service_base._serve_service_evidence(CONTRACT.backend_service, threading.Event())
    assert checks == [True, True]


def test_already_stopping_service_never_resolves_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = threading.Event()
    stop.set()
    monkeypatch.setattr(
        service_base,
        "_confirmed_service_evidence_enrollment",
        lambda: pytest.fail("stopping service resolved authority"),
    )
    service_base._serve_service_evidence(CONTRACT.backend_service, stop)


def test_blocked_evidence_gate_does_not_change_worker_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = threading.Event()
    readiness: list[str] = []

    def unavailable() -> None:
        blocked.set()
        raise RuntimeError("accepted production authority unavailable")

    def worker(stop: threading.Event, report: Any) -> None:
        report(RUNNING)
        readiness.append("product work and readiness still available")
        assert blocked.wait(1.0)

    monkeypatch.setattr(service_base, "_confirmed_service_evidence_enrollment", unavailable)
    service = emulated_service(monkeypatch, CONTRACT.backend_service, worker)
    service.SvcRun()
    assert readiness == ["product work and readiness still available"]
    assert service.statuses == [START_PENDING, RUNNING, STOPPED]


def test_endpoint_thread_start_failure_does_not_fail_worker(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def start() -> None:
        raise RuntimeError("private runtime detail")

    monkeypatch.setattr(
        service_base.threading, "Thread", lambda **kwargs: SimpleNamespace(start=start)
    )
    service = emulated_service(
        monkeypatch, CONTRACT.backend_service, lambda stop, report: report(RUNNING)
    )
    service.SvcRun()
    assert service.statuses == [START_PENDING, RUNNING, STOPPED]
    assert service.evidence_thread is None
    assert "endpoint BLOCKED" in caplog.text
    assert "private runtime detail" not in caplog.text
