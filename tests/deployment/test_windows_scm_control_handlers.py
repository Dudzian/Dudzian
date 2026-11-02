"""Deterministic SCM control models; no Windows service is installed or started."""

from __future__ import annotations

import sys
import threading
from typing import Any, Callable

import pytest

from deployment.windows_installer import service_base
from deployment.windows_installer.contract import CONTRACT
from tests.deployment.test_windows_stage9_service_lifecycle import (
    RUNNING,
    START_PENDING,
    STOP_PENDING,
    STOPPED,
    assert_final_host_status,
    emulated_native_host,
    emulated_service,
)


def _threaded(action: Callable[[], Any]) -> tuple[threading.Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            action()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    return thread, errors


def _joined(thread: threading.Thread, errors: list[BaseException]) -> None:
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "modeled control handler did not finish"
    assert errors == []


@pytest.mark.parametrize(
    "status,wait_hint", [(START_PENDING, 60_000), (RUNNING, 0), (STOP_PENDING, 30_000)]
)
def test_interrogate_replays_only_the_confirmed_state_and_wait_hint(
    monkeypatch: pytest.MonkeyPatch, status: int, wait_hint: int
) -> None:
    service = emulated_service(monkeypatch, CONTRACT.postgresql_service, lambda stop, report: None)
    submitted: list[tuple[int, int, int, int, int, int, int]] = []
    scm = sys.modules["win32service"]
    report = scm.SetServiceStatus

    def record(handle: Any, value: tuple[int, int, int, int, int, int, int]) -> None:
        assert handle is service.ssh
        submitted.append(value)
        report(handle, value)

    monkeypatch.setattr(scm, "SetServiceStatus", record)
    service.report(status, wait_hint=wait_hint)
    confirmed = submitted[0]
    service.SvcInterrogate()
    assert submitted == [confirmed, confirmed]
    assert confirmed[1] == status and confirmed[6] == wait_hint
    assert STOPPED not in service.statuses


@pytest.mark.parametrize("status,wait_hint", [(START_PENDING, 60_000), (STOP_PENDING, 30_000)])
def test_repeated_interrogate_preserves_full_pending_snapshot_without_inventing_progress(
    monkeypatch: pytest.MonkeyPatch, status: int, wait_hint: int
) -> None:
    service = emulated_service(monkeypatch, CONTRACT.postgresql_service, lambda stop, report: None)
    service.report(status, wait_hint=wait_hint)
    confirmed = service.status_records[-1]
    checkpoint = service.checkPoint
    assert confirmed == (
        16,
        status,
        0 if status == START_PENDING else 5,
        0,
        0,
        checkpoint,
        wait_hint,
    )
    assert checkpoint == 1
    for _ in range(3):
        service.SvcInterrogate()
    assert service.status_records == [confirmed] * 4
    assert service.checkPoint == checkpoint
    # A real worker/state report may advance progress; interrogation may not.
    service.report(status, wait_hint=wait_hint)
    assert service.checkPoint == checkpoint + 1
    progressed = (*confirmed[:5], checkpoint + 1, wait_hint)
    assert service.status_records[-1] == progressed
    service.SvcInterrogate()
    assert service.status_records[-2:] == [progressed, progressed]
    assert service.checkPoint == checkpoint + 1


def test_failed_pending_report_replays_exact_last_acknowledged_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = emulated_service(monkeypatch, CONTRACT.postgresql_service, lambda stop, report: None)
    service.report(START_PENDING, wait_hint=60_000)
    confirmed = service.status_records[-1]
    scm = sys.modules["win32service"]
    report = scm.SetServiceStatus
    original = OSError("private pending report failure")
    failed = False

    def deny_once(handle: Any, value: tuple[int, int, int, int, int, int, int]) -> None:
        nonlocal failed
        assert handle is service.ssh
        if not failed:
            failed = True
            assert value[5] == confirmed[5] + 1
            assert value[6] == 45_000
            raise original
        report(handle, value)

    monkeypatch.setattr(scm, "SetServiceStatus", deny_once)
    with pytest.raises(OSError) as raised:
        service.report(START_PENDING, wait_hint=45_000)
    assert raised.value is original
    assert service.checkPoint == confirmed[5] + 1
    assert service.status_records == [confirmed]
    service.SvcInterrogate()
    assert service.status_records == [confirmed, confirmed]
    assert service.checkPoint == confirmed[5] + 1


def test_failed_raw_interrogate_does_not_change_snapshot_or_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = emulated_service(monkeypatch, CONTRACT.postgresql_service, lambda stop, report: None)
    service.report(START_PENDING, wait_hint=60_000)
    confirmed = service.status_records[-1]
    checkpoint = service.checkPoint
    scm = sys.modules["win32service"]
    report = scm.SetServiceStatus
    original = OSError("private interrogation failure")
    attempts: list[Any] = []

    def deny_once(handle: Any, value: tuple[int, int, int, int, int, int, int]) -> None:
        assert handle is service.ssh
        attempts.append(value)
        if len(attempts) == 1:
            raise original
        report(handle, value)

    monkeypatch.setattr(scm, "SetServiceStatus", deny_once)
    with pytest.raises(OSError) as raised:
        service.SvcInterrogate()
    assert raised.value is original
    assert service.status_records == [confirmed]
    assert service.checkPoint == checkpoint
    service.SvcInterrogate()
    assert attempts == [confirmed, confirmed]
    assert service.status_records == [confirmed, confirmed]
    assert service.checkPoint == checkpoint


def test_interrogate_before_any_successful_report_cannot_invent_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = emulated_service(monkeypatch, CONTRACT.backend_service, lambda stop, report: None)
    service.SvcInterrogate()
    assert service.statuses == []


def test_failed_readiness_report_does_not_replace_confirmed_start_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = emulated_service(monkeypatch, CONTRACT.postgresql_service, lambda stop, report: None)
    service.report(START_PENDING, wait_hint=60_000)
    report = service.ReportServiceStatus
    original = OSError("private readiness report detail")

    def deny_running(status: int, *, waitHint: int) -> None:
        if status == RUNNING:
            raise original
        report(status, waitHint=waitHint)

    monkeypatch.setattr(service, "ReportServiceStatus", deny_running)
    with pytest.raises(OSError) as raised:
        service.report(RUNNING)
    assert raised.value is original
    service.SvcInterrogate()
    assert service.statuses == [START_PENDING, START_PENDING]


@pytest.mark.parametrize("name", [CONTRACT.backend_service, CONTRACT.verifier_service])
def test_interrogate_never_starts_the_evidence_endpoint(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    started: list[str] = []

    class EndpointThread:
        def __init__(self, *, name: str, **kwargs: Any) -> None:
            self.name = name

        def start(self) -> None:
            started.append(self.name)

    monkeypatch.setattr(service_base.threading, "Thread", EndpointThread)
    service = emulated_service(monkeypatch, name, lambda stop, report: None)
    service.SvcInterrogate()
    service.report(START_PENDING, wait_hint=60_000)
    service.SvcInterrogate()
    assert started == []
    service.report(RUNNING)
    assert started == [f"{name}-stage9-evidence"]
    service.SvcInterrogate()
    service.SvcStop()
    service.SvcInterrogate()
    assert started == [f"{name}-stage9-evidence"]
    assert service.statuses == [
        START_PENDING,
        START_PENDING,
        RUNNING,
        RUNNING,
        STOP_PENDING,
        STOP_PENDING,
    ]


@pytest.mark.parametrize("first_control", ["interrogate", "stop"])
def test_concurrent_interrogate_stop_and_late_readiness_cannot_regress_state(
    monkeypatch: pytest.MonkeyPatch, first_control: str
) -> None:
    service = emulated_service(monkeypatch, CONTRACT.postgresql_service, lambda stop, report: None)
    service.report(START_PENDING, wait_hint=60_000)
    entered, release = threading.Event(), threading.Event()
    second_started, readiness_started = threading.Event(), threading.Event()
    scm = sys.modules["win32service"]
    report = scm.SetServiceStatus
    first_status = START_PENDING if first_control == "interrogate" else STOP_PENDING
    blocked_once = False

    def block_first(handle: Any, status: tuple[int, int, int, int, int, int, int]) -> None:
        nonlocal blocked_once
        assert handle is service.ssh
        if status[1] == first_status and not blocked_once:
            blocked_once = True
            entered.set()
            assert release.wait(2.0)
        report(handle, status)

    monkeypatch.setattr(scm, "SetServiceStatus", block_first)
    first = service.SvcInterrogate if first_control == "interrogate" else service.SvcStop
    second = service.SvcStop if first_control == "interrogate" else service.SvcInterrogate
    first_thread, first_errors = _threaded(first)
    assert entered.wait(2.0)

    def second_control() -> None:
        second_started.set()
        second()

    second_thread, second_errors = _threaded(second_control)
    assert second_started.wait(2.0)
    readiness_thread = None
    readiness_errors: list[BaseException] = []
    if first_control == "stop":

        def late_readiness() -> None:
            readiness_started.set()
            service.report(RUNNING)

        readiness_thread, readiness_errors = _threaded(late_readiness)
        assert readiness_started.wait(2.0)
    release.set()
    _joined(first_thread, first_errors)
    _joined(second_thread, second_errors)
    if readiness_thread is not None:
        _joined(readiness_thread, readiness_errors)
    service.report(RUNNING)
    service.report(START_PENDING, wait_hint=60_000)
    service.SvcInterrogate()
    stopped_at = service.statuses.index(STOP_PENDING)
    assert set(service.statuses[stopped_at:]) == {STOP_PENDING}
    assert RUNNING not in service.statuses
    assert STOPPED not in service.statuses
    assert service.stop_event.is_set() and service.evidence_stop.is_set()
    assert service.stopped.is_set()


@pytest.mark.parametrize("stop_method", ["SvcStop", "SvcShutdown"])
def test_interrogate_retries_failed_stop_without_replaying_running(
    monkeypatch: pytest.MonkeyPatch, stop_method: str
) -> None:
    service = emulated_service(monkeypatch, CONTRACT.postgresql_service, lambda stop, report: None)
    service.report(RUNNING)
    scm = sys.modules["win32service"]
    report = scm.SetServiceStatus
    original = OSError("private stop report detail")
    failed = False

    def fail_once(handle: Any, status: tuple[int, int, int, int, int, int, int]) -> None:
        nonlocal failed
        assert handle is service.ssh
        if status[1] == STOP_PENDING and not failed:
            failed = True
            raise original
        if status[1] == STOP_PENDING:
            assert status[6] == 30_000
        report(handle, status)

    monkeypatch.setattr(scm, "SetServiceStatus", fail_once)
    with pytest.raises(OSError) as raised:
        getattr(service, stop_method)()
    assert raised.value is original
    assert service.stop_event.is_set() and service.evidence_stop.is_set()
    assert service.stopped.is_set()
    service.report(RUNNING)
    service.SvcInterrogate()
    assert service.statuses == [RUNNING, STOP_PENDING]
    confirmed = service.status_records[-1]
    assert confirmed[5] == service.checkPoint == 2
    service.SvcInterrogate()
    assert service.status_records[-2:] == [confirmed, confirmed]
    assert service.checkPoint == 2


@pytest.mark.parametrize("stop_method", ["SvcStop", "SvcShutdown"])
@pytest.mark.parametrize(
    "failures",
    [
        (),
        ("report",),
        ("stop_event",),
        ("evidence_stop",),
        ("native_event",),
        ("report", "native_event"),
        ("stop_event", "evidence_stop"),
        ("report", "stop_event", "evidence_stop", "native_event"),
    ],
)
def test_stop_attempts_every_cancellation_and_preserves_the_first_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stop_method: str,
    failures: tuple[str, ...],
) -> None:
    service = emulated_service(monkeypatch, CONTRACT.postgresql_service, lambda stop, report: None)
    attempts: list[str] = []
    errors = {
        "report": OSError("private-report-password"),
        "stop_event": RuntimeError("private-worker-password"),
        "evidence_stop": ValueError("private-evidence-password"),
        "native_event": LookupError("private-native-password"),
    }
    report = service.ReportServiceStatus
    worker_set = service.stop_event.set
    evidence_set = service.evidence_stop.set
    native_set = sys.modules["win32event"].SetEvent

    def attempt(label: str, action: Callable[[], Any]) -> None:
        attempts.append(label)
        if label in failures:
            raise errors[label]
        action()

    def stop_report(status: int, *, waitHint: int) -> None:
        assert (status, waitHint) == (STOP_PENDING, 30_000)
        attempt("report", lambda: report(status, waitHint=waitHint))

    monkeypatch.setattr(service, "ReportServiceStatus", stop_report)
    monkeypatch.setattr(service.stop_event, "set", lambda: attempt("stop_event", worker_set))
    monkeypatch.setattr(
        service.evidence_stop, "set", lambda: attempt("evidence_stop", evidence_set)
    )
    monkeypatch.setattr(
        sys.modules["win32event"],
        "SetEvent",
        lambda event: attempt("native_event", lambda: native_set(event)),
    )
    if failures:
        with pytest.raises(BaseException) as raised:
            getattr(service, stop_method)()
        primary = next(label for label in errors if label in failures)
        assert raised.value is errors[primary]
        notes = getattr(raised.value, "__notes__", [])
        assert len(notes) == len(failures) - 1
        for label in failures:
            assert str(errors[label]) not in " ".join(notes)
            assert str(errors[label]) not in caplog.text
        for label in failures:
            if label != primary:
                assert type(errors[label]).__name__ in " ".join(notes)
    else:
        getattr(service, stop_method)()
    assert attempts == ["report", "stop_event", "evidence_stop", "native_event"]
    assert service.stop_event.is_set() is ("stop_event" not in failures)
    assert service.evidence_stop.is_set() is ("evidence_stop" not in failures)
    assert service.stopped.is_set() is ("native_event" not in failures)
    assert STOPPED not in service.statuses


@pytest.mark.parametrize("stop_method", ["SvcStop", "SvcShutdown"])
@pytest.mark.parametrize("failure", [None, "report", "native_event"])
def test_handler_cancellation_finishes_worker_and_endpoint_before_sole_native_stop(
    monkeypatch: pytest.MonkeyPatch, stop_method: str, failure: str | None
) -> None:
    ready, endpoint_entered, endpoint_cleaned = (threading.Event() for _ in range(3))
    worker_cancelled = threading.Event()
    host_result: list[BaseException | None] = []

    def endpoint(name: str, stop: threading.Event) -> None:
        endpoint_entered.set()
        assert stop.wait(2.0)
        endpoint_cleaned.set()

    def worker(stop: threading.Event, report: Any) -> None:
        report(RUNNING)
        assert endpoint_entered.wait(2.0)
        ready.set()
        assert stop.wait(2.0)
        worker_cancelled.set()

    monkeypatch.setattr(service_base, "_serve_service_evidence", endpoint)
    service = emulated_service(monkeypatch, CONTRACT.backend_service, worker)
    report = service.ReportServiceStatus
    submit = service.submit_status
    original = OSError("private control handler failure")

    def report_with_failure(status: int, *, waitHint: int) -> None:
        if status == STOP_PENDING and failure == "report":
            raise original
        report(status, waitHint=waitHint)

    def submit_after_cleanup(owner: str, status: tuple[int, int, int]) -> None:
        if status[0] == STOPPED:
            assert worker_cancelled.is_set() and endpoint_cleaned.is_set()
        submit(owner, status)

    monkeypatch.setattr(service, "ReportServiceStatus", report_with_failure)
    monkeypatch.setattr(service, "submit_status", submit_after_cleanup)
    if failure == "native_event":

        def fail_native(event: Any) -> None:
            raise original

        monkeypatch.setattr(sys.modules["win32event"], "SetEvent", fail_native)
    thread, errors = _threaded(lambda: host_result.append(emulated_native_host(service)))
    assert ready.wait(2.0)
    if failure is None:
        getattr(service, stop_method)()
    else:
        with pytest.raises(OSError) as raised:
            getattr(service, stop_method)()
        assert raised.value is original
    _joined(thread, errors)
    assert host_result == [None]
    assert_final_host_status(service, failed=False)
    submitted = list(service.scm_submissions)
    service.SvcInterrogate()
    service.report(RUNNING)
    assert service.scm_submissions == submitted


@pytest.mark.parametrize("initial_report_failed", [False, True])
def test_interrogate_after_service_exit_cannot_reopen_the_native_status_lifetime(
    monkeypatch: pytest.MonkeyPatch, initial_report_failed: bool
) -> None:
    calls: list[str] = []
    service = emulated_service(
        monkeypatch, CONTRACT.postgresql_service, lambda stop, report: calls.append("worker")
    )
    original = OSError("private initial status failure")
    report = service.ReportServiceStatus

    def initial_report(status: int, *, waitHint: int) -> None:
        if initial_report_failed and status == START_PENDING:
            raise original
        report(status, waitHint=waitHint)

    monkeypatch.setattr(service, "ReportServiceStatus", initial_report)
    result = emulated_native_host(service)
    assert result is (original if initial_report_failed else None)
    assert calls == ([] if initial_report_failed else ["worker"])
    assert_final_host_status(service, failed=initial_report_failed)
    submitted = list(service.scm_submissions)
    service.SvcInterrogate()
    service.report(RUNNING)
    service.report(START_PENDING, wait_hint=60_000)
    assert service.scm_submissions == submitted
