"""Programmatic readiness cancellation models, without SCM STOP or native services.

These tests inject the worker's cancellation event. They do not claim that SCM
accepts an ordinary STOP control while the service reports START_PENDING.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from deployment.windows_installer import postgresql_service
from deployment.windows_installer.contract import CONTRACT


class ModelClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class CancellationEvent:
    def __init__(self, clock: ModelClock) -> None:
        self.clock = clock
        self.cancelled = False
        self.waits: list[float] = []
        self.on_wait: Callable[[], None] = lambda: None

    def is_set(self) -> bool:
        return self.cancelled

    def set(self) -> None:
        self.cancelled = True

    def wait(self, timeout: float) -> bool:
        self.waits.append(timeout)
        self.clock.now += timeout
        self.on_wait()
        return self.cancelled


class ReadinessModel:
    WAIT_OBJECT_0 = 0
    WAIT_TIMEOUT = 258

    def __init__(self, executable: Path) -> None:
        self.executable = executable
        self.process = object()
        self.clock = ModelClock()
        self.stop = CancellationEvent(self.clock)
        self.native_waits: list[tuple[object, int]] = []
        self.exit_queries: list[object] = []
        self.probes: list[list[str]] = []
        self.wait_results: list[int | BaseException] = []
        self.exit_result: int | BaseException = 0
        self.on_probe: Callable[[], Any] = lambda: SimpleNamespace(returncode=1)

    def WaitForSingleObject(self, process: object, timeout: int) -> int:
        assert process is self.process
        assert timeout == 0
        self.native_waits.append((process, timeout))
        result = self.wait_results.pop(0) if self.wait_results else self.WAIT_TIMEOUT
        if isinstance(result, BaseException):
            raise result
        return result

    def GetExitCodeProcess(self, process: object) -> int:
        assert process is self.process
        self.exit_queries.append(process)
        if isinstance(self.exit_result, BaseException):
            raise self.exit_result
        return self.exit_result

    def run(self, command: list[str], **kwargs: Any) -> Any:
        assert command == [
            str(self.executable),
            "-h",
            CONTRACT.postgresql_host,
            "-p",
            str(CONTRACT.postgresql_port),
            "-d",
            "freshness_gate",
        ]
        assert kwargs == {"capture_output": True, "timeout": 5}
        self.probes.append(command)
        return self.on_probe()

    def wait_ready(self, *, timeout: float = 1.0) -> bool:
        return postgresql_service.wait_ready(
            self.executable,
            self.process,
            self.stop,
            timeout=timeout,
            clock=self.clock,
            run=self.run,
            win32event=self,
            win32process=self,
        )


@pytest.fixture
def model(tmp_path: Path) -> ReadinessModel:
    return ReadinessModel(tmp_path / "pg_isready.exe")


def test_programmatic_cancel_before_first_probe_skips_native_process_reads(
    model: ReadinessModel,
) -> None:
    model.stop.set()
    model.wait_results = [PermissionError("unreachable native call")]
    assert model.wait_ready() is False
    assert model.native_waits == model.exit_queries == model.probes == model.stop.waits == []


def test_programmatic_cancel_after_alive_read_prevents_the_first_probe(
    model: ReadinessModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    wait = model.WaitForSingleObject

    def cancel_after_wait(process: object, timeout: int) -> int:
        result = wait(process, timeout)
        model.stop.set()
        return result

    monkeypatch.setattr(model, "WaitForSingleObject", cancel_after_wait)
    assert model.wait_ready() is False
    assert model.native_waits == [(model.process, 0)]
    assert model.exit_queries == model.probes == model.stop.waits == []


def test_programmatic_cancel_during_backoff_is_not_a_readiness_timeout(
    model: ReadinessModel,
) -> None:
    def cancel_at_deadline() -> None:
        model.clock.now = 10.0
        model.stop.set()

    model.stop.on_wait = cancel_at_deadline
    assert model.wait_ready(timeout=1.0) is False
    assert len(model.probes) == 1
    assert len(model.native_waits) == 2
    assert model.exit_queries == []
    assert model.stop.waits == [0.1]


@pytest.mark.parametrize("probe_returncode", [0, 1])
def test_programmatic_cancel_during_probe_prevents_ready_even_when_probe_succeeds(
    model: ReadinessModel, probe_returncode: int
) -> None:
    def cancel_probe() -> Any:
        model.stop.set()
        return SimpleNamespace(returncode=probe_returncode)

    model.on_probe = cancel_probe
    assert model.wait_ready() is False
    assert len(model.probes) == 1
    # Cancellation does not replace the final read of the original handle.
    assert len(model.native_waits) == 2
    assert model.exit_queries == model.stop.waits == []


def test_readiness_success_is_explicit_and_requires_original_process_alive_after_probe(
    model: ReadinessModel,
) -> None:
    model.on_probe = lambda: SimpleNamespace(returncode=0)
    assert model.wait_ready() is True
    assert len(model.probes) == 1
    assert model.native_waits == [(model.process, 0), (model.process, 0)]
    assert model.exit_queries == model.stop.waits == []


def test_actual_readiness_deadline_still_raises_without_cancellation(
    model: ReadinessModel,
) -> None:
    with pytest.raises(postgresql_service.PostgreSQLReadinessTimeout, match="readiness deadline"):
        model.wait_ready(timeout=0.25)
    assert not model.stop.is_set()
    assert model.clock.now >= 0.25
    assert model.probes
    assert len(model.native_waits) == 2 * len(model.probes)
    assert model.exit_queries == []
    assert all(timeout == 0.1 for timeout in model.stop.waits)


def test_successful_probe_after_deadline_does_not_claim_ready(model: ReadinessModel) -> None:
    def too_late() -> Any:
        model.clock.now = 2.0
        return SimpleNamespace(returncode=0)

    model.on_probe = too_late
    with pytest.raises(postgresql_service.PostgreSQLReadinessTimeout):
        model.wait_ready(timeout=1.0)
    assert not model.stop.is_set()
    assert len(model.native_waits) == 2


@pytest.mark.parametrize("after_probe", [False, True])
@pytest.mark.parametrize("exit_code", [0, 31])
def test_spontaneous_process_exit_before_readiness_remains_failure(
    model: ReadinessModel, after_probe: bool, exit_code: int
) -> None:
    model.wait_results = (
        [model.WAIT_TIMEOUT, model.WAIT_OBJECT_0] if after_probe else [model.WAIT_OBJECT_0]
    )
    model.exit_result = exit_code
    model.on_probe = lambda: SimpleNamespace(returncode=0)
    with pytest.raises(RuntimeError, match=f"exited before readiness: exit_code={exit_code}"):
        model.wait_ready()
    assert not model.stop.is_set()
    assert model.exit_queries == [model.process]
    assert len(model.probes) == int(after_probe)


@pytest.mark.parametrize("exit_code", [0, 31])
@pytest.mark.parametrize("probe_returncode", [0, 1])
def test_exit_during_probe_is_not_hidden_by_simultaneous_programmatic_cancel(
    model: ReadinessModel, exit_code: int, probe_returncode: int
) -> None:
    model.wait_results = [model.WAIT_TIMEOUT, model.WAIT_OBJECT_0]
    model.exit_result = exit_code

    def cancel_probe() -> Any:
        model.stop.set()
        return SimpleNamespace(returncode=probe_returncode)

    model.on_probe = cancel_probe
    with pytest.raises(RuntimeError, match=f"exited before readiness: exit_code={exit_code}"):
        model.wait_ready()
    assert model.stop.is_set()
    assert model.exit_queries == [model.process]
    assert len(model.native_waits) == 2
    assert model.stop.waits == []


@pytest.mark.parametrize("after_probe", [False, True])
def test_unexpected_native_wait_result_remains_failure(
    model: ReadinessModel, after_probe: bool
) -> None:
    model.wait_results = [model.WAIT_TIMEOUT, -1] if after_probe else [-1]
    with pytest.raises(RuntimeError, match="unexpected PostgreSQL wait result: -1"):
        model.wait_ready()
    assert model.exit_queries == []
    assert len(model.probes) == int(after_probe)


@pytest.mark.parametrize("operation", ["wait", "get_exit_code", "probe"])
@pytest.mark.parametrize("during_cancel", [False, True])
def test_real_api_failures_preserve_original_exception_including_during_cancel(
    model: ReadinessModel, operation: str, during_cancel: bool
) -> None:
    original = OSError("private native readiness detail")

    def probe() -> Any:
        if during_cancel:
            model.stop.set()
        if operation == "probe":
            raise original
        return SimpleNamespace(returncode=0)

    model.on_probe = probe
    if operation == "wait":
        model.wait_results = [model.WAIT_TIMEOUT, original] if during_cancel else [original]
    elif operation == "get_exit_code":
        model.wait_results = (
            [model.WAIT_TIMEOUT, model.WAIT_OBJECT_0] if during_cancel else [model.WAIT_OBJECT_0]
        )
        model.exit_result = original
    with pytest.raises(OSError) as raised:
        model.wait_ready()
    assert raised.value is original
    assert model.stop.is_set() is during_cancel
    assert model.stop.waits == []


def test_probe_timeout_is_not_reclassified_as_readiness_cancel(model: ReadinessModel) -> None:
    original = subprocess.TimeoutExpired(str(model.executable), timeout=5)

    def probe() -> Any:
        model.stop.set()
        raise original

    model.on_probe = probe
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        model.wait_ready()
    assert raised.value is original
    assert model.stop.is_set()
    assert len(model.native_waits) == 1
