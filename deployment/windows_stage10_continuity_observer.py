"""Natywna, nieautoryzująca obserwacja jednego uchwytu procesu Windows.

Wynik jest diagnostyką, nigdy dowodem Stage 10. Nie ma wejścia pozwalającego
odtworzyć obserwację z pliku ani wznowić ją po restarcie kwalifikatora.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from enum import Enum
import math
import os
import subprocess
from threading import Event
from typing import Any, Callable, Protocol

from deployment.windows_installer.contract import CONTRACT
from deployment import windows_stage9_evidence_pipe as pipe

TICKS_PER_SECOND = 10_000_000
MINIMUM_LIFECYCLE_SECONDS = 86_400
MAX_GAP_100NS = 50_000_000
MAX_CLOCK_SKEW_100NS = 200_000
_WAIT_MS = 250


class ObservationState(str, Enum):
    CONTINUOUS = "CONTINUOUS"
    OBSERVATION_LOST = "OBSERVATION_LOST"
    PROCESS_EXITED = "PROCESS_EXITED"
    IDENTITY_CHANGED = "IDENTITY_CHANGED"
    NOT_OBSERVED = "NOT_OBSERVED"


@dataclass(frozen=True)
class Observation:
    state: ObservationState
    scope: str
    reason: str
    observed_100ns: int = 0
    process_identity: dict[str, Any] | None = None
    # Nawet 86400 s nie zastępuje osobnego autorytetu suspend/resume i produktu.
    duration_24h: str = field(default="BLOCKED", init=False)

    @property
    def elapsed_seconds(self) -> float:
        return self.observed_100ns / TICKS_PER_SECOND


class IdentityChanged(ValueError):
    """Zachowany uchwyt nie odpowiada już oczekiwanej tożsamości SCM."""


class ObservationLost(ValueError):
    """Nie można dalej potwierdzać procesu albo ciągłości zegara."""


class _ProcessExited(ValueError):
    """Pierwotny uchwyt pozostał sygnalizowany; nie wolno otworzyć następcy."""


@dataclass(frozen=True)
class ClockSample:
    biased_before: int
    unbiased: int
    biased_after: int


class _Clock(Protocol):
    def sample(self) -> ClockSample: ...


class _Lease(Protocol):
    identity: dict[str, Any]

    def wait(self, milliseconds: int) -> bool: ...

    def assert_current(self) -> None: ...

    def close(self) -> None: ...


class _NativeClock:
    """100 ns: interrupt time obejmuje sen; unbiased interrupt time go wyklucza.

    API Precise wymagają Windows 10 / Server 2016. Brak API oznacza odmowę.
    Trzy odczyty ograniczają błąd parowania; nie zastępują powiadomień zasilania.
    """

    def __init__(self) -> None:
        if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise ObservationLost("Wymagany jest natywny Windows x64")
        loader = getattr(ctypes, "WinDLL", None)
        if loader is None:
            raise ObservationLost("Natywny loader Win32 jest niedostępny")
        kernel = loader("kernel32.dll", use_last_error=True, winmode=0x00000800)
        self._biased = kernel.QueryInterruptTimePrecise
        self._unbiased = kernel.QueryUnbiasedInterruptTimePrecise
        for function in (self._biased, self._unbiased):
            function.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
            function.restype = None

    def sample(self) -> ClockSample:
        before, active, after = (ctypes.c_ulonglong() for _ in range(3))
        self._biased(ctypes.byref(before))
        self._unbiased(ctypes.byref(active))
        self._biased(ctypes.byref(after))
        return ClockSample(before.value, active.value, after.value)


def _wait(w: Any, process: Any, milliseconds: int) -> bool:
    result = w.event.WaitForSingleObject(process, milliseconds)
    if result == 0:  # WAIT_OBJECT_0: zakończenie pozostaje sygnalizowane.
        return True
    if result != 258:  # WAIT_TIMEOUT jest jedyną dopuszczalną odpowiedzią żywego procesu.
        raise ObservationLost("Oczekiwanie na zachowanym uchwycie nie powiodło się")
    return False


class _ServiceLease:
    """Wykorzystuje sprawdzony pin SCM Stage 9 bez zmiany jego kontraktu."""

    def __init__(self) -> None:
        self._pinned = pipe._PinnedService(CONTRACT.backend_service, None)
        self.process = self._pinned.process
        self.identity = dict(self._pinned.identity)

    def wait(self, milliseconds: int) -> bool:
        return _wait(self._pinned.w, self.process, milliseconds)

    def assert_current(self) -> None:
        try:
            self._pinned.assert_current()
        except pipe.EvidencePipeError as exc:
            if self.wait(0):
                raise _ProcessExited("Zachowany proces zakończył się") from exc
            raise IdentityChanged(str(exc)) from exc

    def close(self) -> None:
        self._pinned.close()


class _HelperProcessLease:
    """Wyłącznie własny proces pomocniczy; nigdy tożsamość usługi produkcyjnej."""

    def __init__(self, child: subprocess.Popen[Any]) -> None:
        self.w = pipe._win32()
        self.process: Any = None
        self.child = child
        self.pid = child.pid
        try:
            if child.poll() is not None:
                raise ObservationLost("Proces pomocniczy zakończył się przed otwarciem")
            self.process = self.w.api.OpenProcess(0x1000 | 0x100000, False, self.pid)
            self.identity = self._identity()
            self.assert_current()
        except BaseException:
            self.close()
            raise

    def _identity(self) -> dict[str, Any]:
        return {
            "process_id": self.pid,
            "process_started": self.w.process.GetProcessTimes(self.process)[
                "CreationTime"
            ].isoformat(),
            "process_path": self.w.process.QueryFullProcessImageName(self.process, 0),
        }

    def wait(self, milliseconds: int) -> bool:
        if self.process is None:
            raise ObservationLost("Uchwyt procesu pomocniczego został zamknięty")
        return _wait(self.w, self.process, milliseconds)

    def assert_current(self) -> None:
        if self.child.poll() is not None and self.wait(0):
            raise _ProcessExited("Zachowany proces pomocniczy zakończył się")
        if (
            self.child.pid != self.pid
            or self.child.poll() is not None
            or self._identity() != self.identity
        ):
            raise IdentityChanged("Tożsamość procesu pomocniczego uległa zmianie")

    def close(self) -> None:
        if self.process is not None:
            handle, self.process = self.process, None
            handle.Close()


def _validate_sample(sample: ClockSample) -> None:
    if (
        any(
            type(value) is not int or value < 0
            for value in (sample.biased_before, sample.unbiased, sample.biased_after)
        )
        or sample.biased_after < sample.biased_before
        or sample.biased_after - sample.biased_before > MAX_CLOCK_SKEW_100NS
        or sample.unbiased > sample.biased_after
    ):
        raise ObservationLost("Niepoprawna lub niespójna próbka zegarów natywnych")


def _check_clock(first: ClockSample, previous: ClockSample, current: ClockSample) -> None:
    _validate_sample(current)
    if (
        current.unbiased <= previous.unbiased
        or current.biased_before < previous.biased_after
        or current.biased_after - previous.biased_before > MAX_GAP_100NS
    ):
        raise ObservationLost("Cofnięcie zegara lub przerwa w obserwacji przekraczająca 5 s")
    # Przedziały różnicy zegarów muszą pozostać zgodne z początkiem próby.
    # Sen/hibernacja zwiększa różnicę; niewyjaśnione zmniejszenie też odmawia.
    if (
        current.biased_before - current.unbiased
        > first.biased_after - first.unbiased + MAX_CLOCK_SKEW_100NS
        or current.biased_after - current.unbiased
        < first.biased_before - first.unbiased - MAX_CLOCK_SKEW_100NS
    ):
        raise ObservationLost("Suspend/resume lub niewyjaśniona rozbieżność zegarów")


def _observe(
    lease_factory: Callable[[], _Lease],
    clock_factory: Callable[[], _Clock],
    *,
    seconds: float,
    stop: Event | None = None,
    scope: str,
) -> Observation:
    """Jednorazowy właściciel zasobów; żadnego wznowienia ani otwierania następcy."""
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or not 0 < seconds <= MINIMUM_LIFECYCLE_SECONDS
    ):
        return Observation(ObservationState.NOT_OBSERVED, scope, "Niepoprawny czas obserwacji")
    if stop is not None and stop.is_set():
        return Observation(ObservationState.NOT_OBSERVED, scope, "Anulowano przed otwarciem")
    required_ticks = math.ceil(seconds * TICKS_PER_SECOND)
    lease = None
    identity = None
    elapsed = 0
    result = Observation(ObservationState.NOT_OBSERVED, scope, "Obserwacja nie rozpoczęła się")
    try:
        lease = lease_factory()
        identity = dict(lease.identity)
        clock = clock_factory()
        if lease.wait(0):
            raise _ProcessExited("Proces zakończony przed początkiem pomiaru")
        lease.assert_current()
        first = previous = clock.sample()
        _validate_sample(first)
        while True:
            if stop is not None and stop.is_set():
                raise ObservationLost("Anulowano aktywną obserwację")
            wait_ms = min(_WAIT_MS, max(1, math.ceil((required_ticks - elapsed) / 10_000)))
            if lease.wait(wait_ms):
                result = Observation(
                    ObservationState.PROCESS_EXITED,
                    scope,
                    "Zachowany proces zakończył się",
                    elapsed,
                    identity,
                )
                break
            current = clock.sample()
            _check_clock(first, previous, current)
            lease.assert_current()
            # Ostatni wait wiąże koniec pomiaru z nadal żywym, pierwotnym uchwytem.
            if lease.wait(0):
                result = Observation(
                    ObservationState.PROCESS_EXITED,
                    scope,
                    "Proces zakończony na końcu pomiaru",
                    elapsed,
                    identity,
                )
                break
            if stop is not None and stop.is_set():
                raise ObservationLost("Anulowano aktywną obserwację")
            elapsed = current.unbiased - first.unbiased
            previous = current
            if elapsed >= required_ticks:
                result = Observation(
                    ObservationState.CONTINUOUS,
                    scope,
                    "Potwierdzono ograniczony zakres jednego uchwytu; kwalifikacja BLOCKED",
                    elapsed,
                    identity,
                )
                break
    except _ProcessExited as exc:
        result = Observation(ObservationState.PROCESS_EXITED, scope, str(exc), elapsed, identity)
    except IdentityChanged as exc:
        result = Observation(ObservationState.IDENTITY_CHANGED, scope, str(exc), elapsed, identity)
    except KeyboardInterrupt:
        result = Observation(
            ObservationState.OBSERVATION_LOST, scope, "Przerwano kwalifikator", elapsed, identity
        )
    except Exception as exc:
        state = (
            ObservationState.OBSERVATION_LOST
            if lease is not None
            else ObservationState.NOT_OBSERVED
        )
        result = Observation(state, scope, f"{type(exc).__name__}: {exc}", elapsed, identity)
    finally:
        if lease is not None:
            try:
                lease.close()
            except Exception as exc:
                result = Observation(
                    ObservationState.OBSERVATION_LOST,
                    scope,
                    f"Błąd zamknięcia uchwytów: {exc}",
                    elapsed,
                    identity,
                )
    return result


def observe_backend(*, seconds: float, stop: Event | None = None) -> Observation:
    """Obserwuj wyłącznie kanoniczny backend SCM. Nie kwalifikuje produktu ani Stage 10."""
    return _observe(_ServiceLease, _NativeClock, seconds=seconds, stop=stop, scope="SCM_CONTRACT")


def observe_test_process(
    child: subprocess.Popen[Any], *, seconds: float, stop: Event | None = None
) -> Observation:
    """Izolowana próba własnego helpera, zawsze TEST_ONLY, bez emisji evidence."""
    return _observe(
        lambda: _HelperProcessLease(child),
        _NativeClock,
        seconds=seconds,
        stop=stop,
        scope="TEST_ONLY",
    )
