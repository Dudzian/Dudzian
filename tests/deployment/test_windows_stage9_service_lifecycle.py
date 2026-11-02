"""Emulated lifecycle boundaries; these tests do not qualify Windows services."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from deployment.windows_installer import service_base
from deployment.windows_installer.contract import CONTRACT


START_PENDING, RUNNING, STOP_PENDING, STOPPED = 2, 4, 3, 1
# pywin32 b312's PythonService.cpp native host maps an unhandled SvcRun exception to these
# values. This is an explicit model of that boundary, not a native SCM test.
ERROR_SERVICE_SPECIFIC_ERROR, PY_E_UNHANDLED_EXCEPTION = 1066, 0x20000001


def emulated_service(monkeypatch: pytest.MonkeyPatch, name: str, worker: Any) -> Any:
    hosted: list[Any] = []

    class ServiceFramework:
        def __init__(self, args: list[str]) -> None:
            self.ssh = self
            self.checkPoint = 0
            self.statuses: list[int] = []
            self.status_records: list[tuple[int, int, int, int, int, int, int]] = []
            self.native_statuses: list[tuple[int, int, int]] = []
            self.scm_submissions: list[tuple[str, tuple[int, int, int]]] = []
            self.scm_status: tuple[int, int, int] | None = None

        def ReportServiceStatus(self, status: int, *, waitHint: int) -> None:
            # Mirror the pinned host's checkpoint behavior, including an
            # increment before the native call (which may then fail).
            checkpoint = 0
            if status not in (RUNNING, STOPPED):
                self.checkPoint += 1
                checkpoint = self.checkPoint
            modules["win32service"].SetServiceStatus(
                self.ssh,
                (
                    16,
                    status,
                    0 if status == START_PENDING else self.GetAcceptedControls(),
                    0,
                    0,
                    checkpoint,
                    waitHint,
                ),
            )

        def GetAcceptedControls(self) -> int:
            return 5  # SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN

        def submit_status(self, owner: str, value: tuple[int, int, int]) -> None:
            self.scm_submissions.append((owner, value))
            # A terminal SCM status ends this service's reporting lifetime.
            # A Python STOPPED success must not hide the host's later error.
            if self.scm_status is None or self.scm_status[0] != STOPPED:
                self.scm_status = value

    def set_service_status(
        handle: ServiceFramework, status: tuple[int, int, int, int, int, int, int]
    ) -> None:
        assert handle.ssh is handle
        handle.status_records.append(status)
        handle.statuses.append(status[1])
        handle.submit_status("python", (status[1], status[3], status[4]))

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
            SERVICE_WIN32_OWN_PROCESS=16,
            SERVICE_START_PENDING=START_PENDING,
            SERVICE_RUNNING=RUNNING,
            SERVICE_STOP_PENDING=STOP_PENDING,
            SERVICE_STOPPED=STOPPED,
            SetServiceStatus=set_service_status,
        ),
        "win32serviceutil": SimpleNamespace(ServiceFramework=ServiceFramework),
    }
    for module_name, module in modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)
    service_base.run_service(name, worker)
    return hosted[0]([])


def emulated_native_host(service: Any) -> BaseException | None:
    """Run SvcRun, then model the separate native host's final SetServiceStatus."""
    failure: BaseException | None = None
    try:
        service.SvcRun()
    except BaseException as exc:
        failure = exc
    final = (
        STOPPED,
        ERROR_SERVICE_SPECIFIC_ERROR if failure is not None else 0,
        PY_E_UNHANDLED_EXCEPTION if failure is not None else 0,
    )
    service.native_statuses.append(final)
    service.submit_status("native_host", final)
    return failure


def assert_final_host_status(service: Any, *, failed: bool) -> None:
    expected = (
        STOPPED,
        ERROR_SERVICE_SPECIFIC_ERROR if failed else 0,
        PY_E_UNHANDLED_EXCEPTION if failed else 0,
    )
    assert STOPPED not in service.statuses
    assert service.native_statuses == [expected]
    assert service.scm_status == expected
    assert [owner for owner, status in service.scm_submissions if status[0] == STOPPED] == [
        "native_host"
    ]


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
    submit_status = service.submit_status

    def report_after_cleanup(owner: str, status: tuple[int, int, int]) -> None:
        if status[0] == STOPPED:
            assert cleaned.is_set()
        submit_status(owner, status)

    service.submit_status = report_after_cleanup
    assert emulated_native_host(service) is None
    assert cleaned.is_set()
    assert service.statuses == [START_PENDING, RUNNING, RUNNING]
    assert_final_host_status(service, failed=False)
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
    assert service.statuses == [START_PENDING]
    assert service.native_statuses == []
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
    assert service.statuses == [START_PENDING, RUNNING]
    assert service.native_statuses == []
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
    assert service.statuses == [START_PENDING, RUNNING]
    assert service.native_statuses == []


@pytest.mark.parametrize("name", [CONTRACT.backend_service, CONTRACT.verifier_service])
@pytest.mark.parametrize("stop_method", ["SvcStop", "SvcShutdown"])
def test_native_host_reports_normal_stop_only_after_endpoint_cleanup(
    monkeypatch: pytest.MonkeyPatch, name: str, stop_method: str
) -> None:
    entered, cleaned = threading.Event(), threading.Event()

    def endpoint(service_name: str, stop: threading.Event) -> None:
        assert service_name == name
        entered.set()
        assert stop.wait(1.0)
        cleaned.set()

    def worker(stop: threading.Event, report: Any) -> None:
        report(RUNNING)
        assert entered.wait(1.0)
        getattr(service, stop_method)()
        assert stop.is_set() and service.evidence_stop.is_set()
        assert service.stopped.is_set()

    monkeypatch.setattr(service_base, "_serve_service_evidence", endpoint)
    service = emulated_service(monkeypatch, name, worker)
    submit = service.submit_status

    def submit_after_cleanup(owner: str, status: tuple[int, int, int]) -> None:
        if status[0] == STOPPED:
            assert cleaned.is_set()
            assert service.evidence_thread is not None
            assert not service.evidence_thread.is_alive()
        submit(owner, status)

    service.submit_status = submit_after_cleanup
    assert emulated_native_host(service) is None
    assert service.statuses == [START_PENDING, RUNNING, STOP_PENDING]
    assert_final_host_status(service, failed=False)


@pytest.mark.parametrize("worker_failed", [False, True])
@pytest.mark.parametrize("cleanup_failure", ["join", "alive_check", "still_alive"])
def test_native_host_receives_cleanup_failure_without_masking_primary_worker_error(
    monkeypatch: pytest.MonkeyPatch, worker_failed: bool, cleanup_failure: str
) -> None:
    original = ValueError("primary worker failure")
    cleanup_error = OSError("private endpoint cleanup detail")
    attempts: list[object] = []

    def worker(stop: threading.Event, report: Any) -> None:
        if worker_failed:
            raise original

    def join(*, timeout: float) -> None:
        attempts.append(("join", timeout))
        if cleanup_failure == "join":
            raise cleanup_error

    def is_alive() -> bool:
        attempts.append("alive_check")
        if cleanup_failure == "alive_check":
            raise cleanup_error
        return True

    service = emulated_service(monkeypatch, CONTRACT.backend_service, worker)
    service.evidence_thread = SimpleNamespace(join=join, is_alive=is_alive)
    failure = emulated_native_host(service)
    assert service.evidence_stop.is_set()
    assert attempts[0] == ("join", 7.0)
    if worker_failed:
        assert failure is original
        assert any("cleanup" in note for note in original.__notes__)
        assert "private endpoint cleanup detail" not in " ".join(original.__notes__)
    elif cleanup_failure == "still_alive":
        assert isinstance(failure, RuntimeError)
        assert "cleanup unavailable" in str(failure)
    else:
        assert failure is cleanup_error
    assert service.statuses == [START_PENDING]
    assert_final_host_status(service, failed=True)


@pytest.mark.parametrize("cleanup_failure", [None, "thread", "job"])
@pytest.mark.parametrize(
    "outcome",
    [
        "exit_zero",
        "exit_nonzero",
        "bad_wait",
        "wait_error",
        "exit_code_error",
        "SvcStop",
        "SvcShutdown",
        "SvcStop_report_error",
        "SvcStop_event_error",
        "SvcStop_both_errors",
        "SvcShutdown_report_error",
        "SvcShutdown_event_error",
        "SvcShutdown_both_errors",
    ],
)
def test_actual_postgresql_worker_reaches_native_host_with_truthful_final_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, cleanup_failure: str | None
) -> None:
    """Exercise the real _run/monitor loop with modeled Win32 process handles."""
    from deployment.windows_installer import postgresql_service

    events: list[object] = []
    diagnostics: list[tuple[object, ...]] = []
    state = SimpleNamespace(alive=True, ready=False)
    handle_error = OSError("private process handle detail")
    close_error = OSError("private process cleanup detail")
    report_error = OSError("private stop reporting detail")
    native_event_error = OSError("private native cancellation detail")
    control_errors: list[BaseException] = []
    stop_method = next(
        (method for method in ("SvcStop", "SvcShutdown") if outcome.startswith(method)), None
    )
    report_failed = "report_error" in outcome or "both_errors" in outcome
    native_event_failed = "event_error" in outcome or "both_errors" in outcome

    class Handle:
        def __init__(self, name: str) -> None:
            self.name = name

        def Close(self) -> None:
            events.append(("close", self.name))
            if self.name == cleanup_failure:
                raise close_error

    process, thread, job = Handle("process"), Handle("thread"), Handle("job")
    service = emulated_service(monkeypatch, CONTRACT.postgresql_service, postgresql_service._run)
    event_api = sys.modules["win32event"]
    original_report = service.ReportServiceStatus
    original_set_event = event_api.SetEvent

    def report_status(status: int, *, waitHint: int) -> None:
        if status == STOP_PENDING and report_failed:
            raise report_error
        original_report(status, waitHint=waitHint)

    def set_native_event(event: Any) -> None:
        assert event is service.stopped
        if native_event_failed:
            raise native_event_error
        original_set_event(event)

    service.ReportServiceStatus = report_status
    event_api.SetEvent = set_native_event

    def wait(observed: Any, timeout: int) -> int:
        assert observed is process
        events.append(("wait", timeout))
        if not state.ready:
            assert timeout == 0
            return 258
        if not state.alive:
            return 0
        if outcome == "bad_wait":
            return -1
        if outcome == "wait_error":
            raise handle_error
        if stop_method is not None:
            assert timeout == 100

            def control_handler() -> None:
                # Native SCM controls run separately from SvcRun. A handler
                # exception must not be injected into the worker's wait API.
                try:
                    getattr(service, stop_method)()
                except BaseException as exc:
                    control_errors.append(exc)

            control = threading.Thread(target=control_handler)
            control.start()
            control.join(timeout=1.0)
            assert not control.is_alive()
            return 258
        state.alive = False
        return 0

    def exit_code(observed: Any) -> int:
        assert observed is process
        if outcome == "exit_code_error":
            raise handle_error
        return 31 if outcome == "exit_nonzero" else 0

    event_api.WAIT_OBJECT_0 = 0
    event_api.WAIT_TIMEOUT = 258
    event_api.WaitForSingleObject = wait
    for name, module in (
        ("win32api", SimpleNamespace()),
        ("win32con", SimpleNamespace()),
        ("win32job", SimpleNamespace()),
        ("win32process", SimpleNamespace(GetExitCodeProcess=exit_code)),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    install_root, machine = tmp_path / "install", tmp_path / "machine"
    bindir, pgdata = install_root / "PostgreSQL" / "bin", machine / "PostgreSQL" / "Data"
    bindir.mkdir(parents=True)
    pgdata.mkdir(parents=True)
    for path in (
        bindir / "postgres.exe",
        bindir / "pg_ctl.exe",
        bindir / "pg_isready.exe",
        pgdata / "PG_VERSION",
    ):
        path.touch()

    def ready(_path: Path, observed: Any, stop: threading.Event, **kwargs: Any) -> None:
        assert observed is process and stop is service.stop_event
        state.ready = True

    def stop_postgresql(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        assert stop_method is not None
        assert service.stop_event.is_set()
        assert command[1:] == ["stop", "-D", str(pgdata), "-m", "fast", "-w", "-t", "30"]
        assert kwargs == {"check": True, "timeout": 35}
        events.append("normal-stop")
        state.alive = False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(postgresql_service, "_install_root", lambda: install_root)
    monkeypatch.setattr(postgresql_service, "machine_root", lambda: machine)
    monkeypatch.setattr(
        postgresql_service,
        "create_suspended_in_job",
        lambda *_args, **_kwargs: (job, process, thread, 41),
    )
    monkeypatch.setattr(postgresql_service, "wait_ready", ready)
    monkeypatch.setattr(
        postgresql_service, "try_write_startup_diagnostic", lambda *args: diagnostics.append(args)
    )
    monkeypatch.setattr(postgresql_service.subprocess, "run", stop_postgresql)

    submit = service.submit_status

    def submit_after_cleanup(owner: str, status: tuple[int, int, int]) -> None:
        if status[0] == STOPPED:
            assert events[-3:] == [("close", "thread"), ("close", "process"), ("close", "job")]
            assert service.evidence_stop.is_set()
        submit(owner, status)

    service.submit_status = submit_after_cleanup
    failure = emulated_native_host(service)
    normal_stop = stop_method is not None
    if normal_stop:
        assert service.statuses == [START_PENDING, RUNNING] + (
            [] if report_failed else [STOP_PENDING]
        )
        assert "normal-stop" in events
        assert failure is (None if cleanup_failure is None else close_error)
        assert service.stop_event.is_set() and service.evidence_stop.is_set()
        assert service.stopped.is_set() is not native_event_failed
        if report_failed or native_event_failed:
            assert control_errors == [report_error if report_failed else native_event_error]
            if report_failed and native_event_failed:
                assert any("native cancellation" in note for note in report_error.__notes__)
                assert "private native cancellation detail" not in " ".join(report_error.__notes__)
        else:
            assert control_errors == []
    else:
        assert service.statuses == [START_PENDING, RUNNING]
        assert not service.stop_event.is_set()
        assert "normal-stop" not in events
        assert diagnostics[-1][0] == "MONITOR_PROCESS"
        if outcome in {"wait_error", "exit_code_error"}:
            assert failure is handle_error
        else:
            assert isinstance(failure, RuntimeError)
            expected = (
                "unexpected PostgreSQL wait result"
                if outcome == "bad_wait"
                else f"exited while running: exit_code={31 if outcome == 'exit_nonzero' else 0}"
            )
            assert expected in str(failure)
        if cleanup_failure is not None:
            assert any(f"{cleanup_failure} cleanup failed" in note for note in failure.__notes__)
            assert "private process cleanup detail" not in " ".join(failure.__notes__)
    assert_final_host_status(service, failed=not normal_stop or cleanup_failure is not None)


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
    assert service.statuses == [START_PENDING, RUNNING]


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
    assert service.statuses == [START_PENDING, RUNNING]
    assert service.evidence_thread is None
    assert "endpoint BLOCKED" in caplog.text
    assert "private runtime detail" not in caplog.text
