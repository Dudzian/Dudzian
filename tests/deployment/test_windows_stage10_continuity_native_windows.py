"""Prawdziwe API Windows na własnych procesach; bez produkcyjnego evidence.

Hostowany runner nie instaluje ani nie uruchamia CryptoHunter. Krótki przedział
pomocnika sprawdza uchwyt, tożsamość i odmowy, a nie produkcyjne wymaganie 24h.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from contextlib import contextmanager
from threading import Event, Timer
from typing import Any, Callable, Iterator
from uuid import uuid4

import pytest

from deployment import windows_stage10_continuity_observer as observer
from deployment import windows_stage9_evidence_pipe as pipe

pytestmark = [
    pytest.mark.security,
    pytest.mark.qualification,
    pytest.mark.skipif(
        os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8,
        reason="wymagane rzeczywiste API Windows x64, bez emulacji",
    ),
]

HELPER = "import sys; print('ready', flush=True); sys.stdin.buffer.read(1)"


@contextmanager
def child_process() -> Iterator[subprocess.Popen[bytes]]:
    child = subprocess.Popen(
        [sys.executable, "-I", "-S", "-u", "-c", HELPER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().rstrip(b"\r\n") == b"ready"
        yield child
    finally:
        if child.stdin is not None and not child.stdin.closed:
            child.stdin.close()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
        if child.stdout is not None:
            child.stdout.close()
        if child.stderr is not None:
            child.stderr.close()


def terminate_child(child: subprocess.Popen[bytes]) -> None:
    assert child.stdin is not None
    child.stdin.close()
    assert child.wait(timeout=5) == 0


def assert_handle_closed(raw: int) -> None:
    kernel: Any = getattr(ctypes, "WinDLL")("kernel32.dll", use_last_error=True, winmode=0x00000800)
    kernel.GetHandleInformation.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.GetHandleInformation.restype = ctypes.c_int
    flags = ctypes.c_uint32()
    assert not kernel.GetHandleInformation(ctypes.c_void_p(raw), ctypes.byref(flags))
    assert getattr(ctypes, "get_last_error")() == 6  # ERROR_INVALID_HANDLE


def capture_lease(
    monkeypatch: pytest.MonkeyPatch,
    on_first_wait: Callable[[], None] | None = None,
) -> list[tuple[Any, int, list[int]]]:
    original = observer._HelperProcessLease
    captured: list[tuple[Any, int, list[int]]] = []

    def create(child: subprocess.Popen[bytes]) -> Any:
        lease = original(child)
        handles: list[int] = []
        original_wait = lease.wait
        triggered = False

        def wait(milliseconds: int) -> bool:
            nonlocal triggered
            handles.append(int(lease.process))
            if milliseconds > 0 and not triggered:
                triggered = True
                if on_first_wait is not None:
                    on_first_wait()
            return original_wait(milliseconds)

        monkeypatch.setattr(lease, "wait", wait)
        captured.append((lease, int(lease.process), handles))
        return lease

    monkeypatch.setattr(observer, "_HelperProcessLease", create)
    return captured


def test_rzeczywisty_uchwyt_czas_utworzenia_i_sciezka() -> None:
    with child_process() as child:
        lease = observer._HelperProcessLease(child)
        raw = int(lease.process)
        try:
            native = pipe._win32()
            assert raw > 0
            assert lease.identity["process_id"] == child.pid
            assert (
                lease.identity["process_started"]
                == native.process.GetProcessTimes(lease.process)["CreationTime"].isoformat()
            )
            assert lease.identity["process_path"] == native.process.QueryFullProcessImageName(
                lease.process, 0
            )
            assert not lease.wait(1)
            lease.assert_current()
            assert int(lease.process) == raw
        finally:
            lease.close()
        assert_handle_closed(raw)
        lease.close()


def test_krotka_obserwacja_zachowuje_ten_sam_rzeczywisty_uchwyt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = capture_lease(monkeypatch)
    with child_process() as child:
        result = observer.observe_test_process(child, seconds=0.6)
        assert result.state is observer.ObservationState.CONTINUOUS
        assert result.observed_100ns >= 6_000_000
        assert result.scope == "TEST_ONLY"
        assert result.duration_24h == "BLOCKED"
        assert result.process_identity is not None
        assert result.process_identity["process_id"] == child.pid
        assert len(captured) == 1
        assert len(captured[0][2]) >= 3
        assert set(captured[0][2]) == {captured[0][1]}
        assert_handle_closed(captured[0][1])


def test_rzeczywisty_sygnal_zakonczenia_i_odmowa_dalszej_obserwacji() -> None:
    with child_process() as child:
        lease = observer._HelperProcessLease(child)
        raw = int(lease.process)
        try:
            assert not lease.wait(0)
            terminate_child(child)
            assert lease.wait(1_000)
            assert lease.wait(0)
            with pytest.raises(observer._ProcessExited):
                lease.assert_current()
        finally:
            lease.close()
        assert_handle_closed(raw)


def test_nowa_instancja_nie_zastepuje_zakonczonego_procesu() -> None:
    with child_process() as first:
        old = observer._HelperProcessLease(first)
        old_identity = dict(old.identity)
        raw = int(old.process)
        try:
            terminate_child(first)
            with child_process() as second:
                new = observer._HelperProcessLease(second)
                try:
                    assert new.identity != old_identity
                    assert old.identity == old_identity
                    assert old.wait(0)
                    assert not new.wait(0)
                    with pytest.raises(observer._ProcessExited):
                        old.assert_current()
                finally:
                    new.close()
        finally:
            old.close()
        assert_handle_closed(raw)


def test_obserwator_zwraca_zakonczenie_i_zamyka_uchwyt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with child_process() as child:
        assert child.stdin is not None
        timer = Timer(0.15, child.stdin.close)
        captured = capture_lease(monkeypatch, timer.start)
        try:
            result = observer.observe_test_process(child, seconds=2)
        finally:
            timer.cancel()
            if timer.ident is not None:
                timer.join(timeout=5)
        assert result.state is observer.ObservationState.PROCESS_EXITED
        assert result.duration_24h == "BLOCKED"
        assert len(captured) == 1
        assert_handle_closed(captured[0][1])


def test_anulowanie_natywnego_oczekiwania_zamyka_uchwyt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop = Event()
    with child_process() as child:
        timer = Timer(0.1, stop.set)
        captured = capture_lease(monkeypatch, timer.start)
        try:
            result = observer.observe_test_process(child, seconds=2, stop=stop)
        finally:
            timer.cancel()
            if timer.ident is not None:
                timer.join(timeout=5)
        assert result.state is observer.ObservationState.OBSERVATION_LOST
        assert len(captured) == 1
        assert_handle_closed(captured[0][1])
        assert child.poll() is None


def test_utrata_rzeczywistego_uchwytu_jest_bledem_win32() -> None:
    with child_process() as child:
        lease = observer._HelperProcessLease(child)
        lease.close()
        with pytest.raises(observer.ObservationLost):
            lease.wait(0)


def test_natywny_openprocess_odmawia_nieprawidlowego_pid() -> None:
    native = pipe._win32()
    with pytest.raises(Exception) as caught:
        native.api.OpenProcess(0x1000 | 0x100000, False, 0)
    assert getattr(caught.value, "winerror", None) in {5, 87}


def test_natywny_scm_odmawia_nieistniejacej_uslugi() -> None:
    native = pipe._win32()
    manager = native.service.OpenSCManager(None, None, 1)
    try:
        with pytest.raises(Exception) as caught:
            native.service.OpenService(manager, f"CryptoHunter-TestAbsent-{uuid4().hex}", 1 | 4)
        assert getattr(caught.value, "winerror", None) == 1060
    finally:
        native.service.CloseServiceHandle(manager)


def test_brak_zatwierdzonej_uslugi_nie_wytwarza_evidence() -> None:
    native = pipe._win32()
    manager = native.service.OpenSCManager(None, None, 1)
    try:
        try:
            service = native.service.OpenService(manager, pipe.CONTRACT.backend_service, 1 | 4)
        except Exception as exc:
            assert getattr(exc, "winerror", None) == 1060
        else:
            native.service.CloseServiceHandle(service)
            pytest.skip("host zawiera CryptoHunter; test nie obserwuje rzeczywistych usług")
    finally:
        native.service.CloseServiceHandle(manager)
    result = observer.observe_backend(seconds=0.1)
    assert result.state is observer.ObservationState.NOT_OBSERVED
    assert result.observed_100ns == 0
    assert result.duration_24h == "BLOCKED"
    assert result.process_identity is None


def test_rzeczywiste_natywne_zegary_maja_uporzadkowane_probki() -> None:
    clock = observer._NativeClock()
    first = clock.sample()
    second = clock.sample()
    assert first.biased_before <= first.biased_after
    assert second.biased_before <= second.biased_after
    assert second.biased_before >= first.biased_before > 0
    assert second.unbiased >= first.unbiased > 0


def test_sam_pid_nie_jest_obiektem_wlasnego_pomocnika() -> None:
    result = observer.observe_test_process(os.getpid(), seconds=0.1)
    assert result.state is observer.ObservationState.NOT_OBSERVED
    assert result.scope == "TEST_ONLY"
    assert result.duration_24h == "BLOCKED"
