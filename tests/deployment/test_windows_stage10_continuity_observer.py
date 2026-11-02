"""Syntetyczne odmowy obserwatora; atrapy nie kwalifikują usług Windows."""

from __future__ import annotations

import dataclasses
import inspect
from datetime import datetime, timezone
from threading import Event
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock

import pytest

from deployment import windows_stage10_continuity_observer as observer

pytestmark = pytest.mark.security

TICK = 2_500_000
START = 100_000_000


class Lease:
    def __init__(self) -> None:
        self.identity = {
            "process_id": 123,
            "process_started": "2026-11-02T12:00:00+00:00",
            "process_path": r"C:\Program Files\CryptoHunter\CryptoHunterBackend.exe",
        }
        self.waits: list[int] = []
        self.checks = 0
        self.closes = 0
        self.on_wait: Callable[[], bool] = lambda: False
        self.on_check: Callable[[], None] = lambda: None
        self.close_error: BaseException | None = None

    def wait(self, milliseconds: int) -> bool:
        self.waits.append(milliseconds)
        return self.on_wait()

    def assert_current(self) -> None:
        self.checks += 1
        self.on_check()

    def close(self) -> None:
        self.closes += 1
        if self.close_error is not None:
            raise self.close_error


class Clock:
    def __init__(self, samples: list[Any] | None = None) -> None:
        self.samples = samples
        self.calls = 0

    def sample(self) -> observer.ClockSample:
        index = self.calls
        self.calls += 1
        if self.samples is not None:
            value = self.samples[min(index, len(self.samples) - 1)]
            if isinstance(value, BaseException):
                raise value
            return value
        value = START + index * TICK
        return observer.ClockSample(value, value, value)


def observe(
    lease: Lease,
    clock: Clock | None = None,
    *,
    seconds: float = 0.5,
    stop: Event | None = None,
) -> observer.Observation:
    return observer._observe(
        lambda: lease,
        lambda: clock or Clock(),
        seconds=seconds,
        stop=stop,
        scope="SYNTHETIC_TEST_ONLY",
    )


def test_przedzial_wymaga_jednego_uchwytu_i_sprawdzenia_tozsamosci() -> None:
    lease = Lease()
    opened: list[Lease] = []

    def factory() -> Lease:
        opened.append(lease)
        return lease

    result = observer._observe(factory, Clock, seconds=0.5, stop=None, scope="SYNTHETIC_TEST_ONLY")
    assert result.state is observer.ObservationState.CONTINUOUS
    assert result.observed_100ns >= 5_000_000
    assert result.process_identity == lease.identity
    assert result.scope == "SYNTHETIC_TEST_ONLY"
    assert result.duration_24h == "BLOCKED"
    assert opened == [lease]
    assert lease.checks >= 2
    assert lease.waits and all(0 <= timeout <= 250 for timeout in lease.waits)
    assert lease.closes == 1


@pytest.mark.parametrize(
    "seconds",
    [0, -1, float("nan"), float("inf"), float("-inf"), 86_401, True, False, 10**400, -(10**400)],
    ids=[
        "zero",
        "ujemny",
        "nan",
        "inf",
        "minus-inf",
        "ponad-limit",
        "true",
        "false",
        "duzy-int",
        "duzy-ujemny-int",
    ],
)
@pytest.mark.parametrize("public_entry", [False, True], ids=["wewnetrzne", "publiczne"])
def test_niepoprawny_czas_nie_rozpoczyna_obserwacji(
    seconds: float, public_entry: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease_factory = Mock(side_effect=AssertionError("Nie wolno otwierać SCM ani procesu"))
    clock_factory = Mock(side_effect=AssertionError("Nie wolno inicjalizować zegara Windows"))
    if public_entry:
        monkeypatch.setattr(observer, "_ServiceLease", lease_factory)
        monkeypatch.setattr(observer, "_NativeClock", clock_factory)
        result = observer.observe_backend(seconds=seconds)
    else:
        result = observer._observe(
            lease_factory, clock_factory, seconds=seconds, scope="SYNTHETIC_TEST_ONLY"
        )
    assert result.state is observer.ObservationState.NOT_OBSERVED
    assert result.observed_100ns == 0
    lease_factory.assert_not_called()
    clock_factory.assert_not_called()


@pytest.mark.parametrize("seconds", [1, 1.0, 0.5], ids=["int", "float", "ulamek"])
def test_poprawny_czas_int_i_float_nadal_rozpoczyna_obserwacje(seconds: float) -> None:
    result = observe(Lease(), seconds=seconds)
    assert result.state is observer.ObservationState.CONTINUOUS
    assert result.observed_100ns >= seconds * observer.TICKS_PER_SECOND


@pytest.mark.parametrize("error", [OSError(5, "SCM niedostępny"), RuntimeError("brak API")])
def test_brak_zrodla_procesu_nie_jest_sukcesem(error: Exception) -> None:
    def unavailable() -> Lease:
        raise error

    result = observer._observe(
        unavailable, Clock, seconds=1, stop=None, scope="SYNTHETIC_TEST_ONLY"
    )
    assert result.state is observer.ObservationState.NOT_OBSERVED
    assert result.observed_100ns == 0
    assert result.duration_24h == "BLOCKED"
    assert result.reason


def test_zakonczenie_jest_terminalne_bez_otwarcia_nastepnej_instancji() -> None:
    lease = Lease()
    lease.on_wait = lambda: True
    opened: list[Lease] = []

    def factory() -> Lease:
        opened.append(lease)
        return lease

    result = observer._observe(factory, Clock, seconds=1, stop=None, scope="SYNTHETIC_TEST_ONLY")
    assert result.state is observer.ObservationState.PROCESS_EXITED
    assert opened == [lease]
    assert lease.closes == 1
    assert result.duration_24h == "BLOCKED"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("process_id", 456),
        ("process_started", "2026-11-02T13:00:00+00:00"),
        ("process_path", r"C:\Temp\podmieniony.exe"),
    ],
)
def test_zmiana_pid_czasu_lub_obrazu_przerywa_obserwacje(field: str, replacement: Any) -> None:
    lease = Lease()
    original = dict(lease.identity)

    def check() -> None:
        if lease.checks > 1:
            actual = dict(original)
            actual[field] = replacement
            if actual != original:
                raise observer.IdentityChanged(f"zmieniono {field}")

    lease.on_check = check
    result = observe(lease)
    assert result.state is observer.ObservationState.IDENTITY_CHANGED
    assert lease.closes == 1


def test_ponowne_uzycie_pid_nie_zastepuje_pierwotnego_uchwytu() -> None:
    lease = Lease()

    def check() -> None:
        if lease.checks > 1:
            raise observer.IdentityChanged("ten sam PID, inny CreationTime")

    lease.on_check = check
    result = observe(lease)
    assert result.state is observer.ObservationState.IDENTITY_CHANGED
    assert lease.closes == 1


@pytest.mark.parametrize("operation", ["wait", "SCM", "clock"])
def test_utrata_api_odbiera_mozliwosc_potwierdzenia(operation: str) -> None:
    lease, clock = Lease(), Clock()

    def fail() -> Any:
        raise OSError(6, "utrata uchwytu lub odczytu")

    if operation == "wait":
        lease.on_wait = fail
    elif operation == "SCM":
        lease.on_check = fail
    else:
        clock.samples = [observer.ClockSample(START, START, START), OSError(6, "zegar")]
    result = observe(lease, clock)
    assert result.state is observer.ObservationState.OBSERVATION_LOST
    assert lease.closes == 1
    assert result.duration_24h == "BLOCKED"


@pytest.mark.parametrize(
    "bad",
    [
        observer.ClockSample(START - 1, START + TICK, START - 1),
        observer.ClockSample(START + TICK, START - 1, START + TICK),
        observer.ClockSample(START + TICK + 1, START + TICK, START + TICK),
        observer.ClockSample(START + TICK, START + TICK, START + TICK + 200_001),
        observer.ClockSample(START + 60_000_000, START + 60_000_000, START + 60_000_000),
        observer.ClockSample(START + 30_000_000, START + TICK, START + 30_000_000),
        observer.ClockSample(-1, -1, -1),
        observer.ClockSample(True, True, True),
        observer.ClockSample(START, START, START),
    ],
    ids=[
        "cofniecie-inclusive",
        "cofniecie-unbiased",
        "odwrocony-przedzial",
        "niepewne-probkowanie",
        "utrata-obserwacji-6s",
        "suspend-resume",
        "ujemny-zegar",
        "bool-zamiast-licznika",
        "zatrzymany-zegar",
    ],
)
def test_nieciaglosci_zegara_i_suspend_resume_odmawiaja(bad: observer.ClockSample) -> None:
    lease = Lease()
    clock = Clock([observer.ClockSample(START, START, START), bad])
    result = observe(lease, clock)
    assert result.state is observer.ObservationState.OBSERVATION_LOST
    assert result.duration_24h == "BLOCKED"
    assert lease.closes == 1


def test_brak_natywnego_zegara_zamyka_juz_otwarty_uchwyt() -> None:
    lease = Lease()

    def unavailable() -> Clock:
        raise OSError("API zegara niedostępne")

    result = observer._observe(
        lambda: lease, unavailable, seconds=1, stop=None, scope="SYNTHETIC_TEST_ONLY"
    )
    assert result.state is not observer.ObservationState.CONTINUOUS
    assert result.observed_100ns == 0
    assert lease.closes == 1


def test_anulowanie_w_trakcie_obserwacji_zamyka_uchwyt() -> None:
    lease, stop = Lease(), Event()

    def cancel() -> bool:
        stop.set()
        return False

    lease.on_wait = cancel
    result = observe(lease, stop=stop)
    assert result.state is observer.ObservationState.OBSERVATION_LOST
    assert lease.closes == 1


def test_zadanie_anulowania_przed_startem_nie_potwierdza_ciaglosci() -> None:
    stop = Event()
    stop.set()
    result = observe(Lease(), stop=stop)
    assert result.state is not observer.ObservationState.CONTINUOUS
    assert result.observed_100ns == 0


def test_keyboardinterrupt_odmawia_i_zamyka_uchwyt() -> None:
    lease = Lease()

    def interrupt() -> bool:
        raise KeyboardInterrupt

    lease.on_wait = interrupt
    result = observe(lease)
    assert result.state is observer.ObservationState.OBSERVATION_LOST
    assert lease.closes == 1


def test_systemexit_zamyka_uchwyt_i_propaguje_przerwanie() -> None:
    lease = Lease()

    def interrupt() -> bool:
        raise SystemExit(9)

    lease.on_wait = interrupt
    with pytest.raises(SystemExit):
        observe(lease)
    assert lease.closes == 1


def test_blad_cleanup_nie_pozostawia_sukcesu() -> None:
    lease = Lease()
    lease.close_error = OSError(6, "CloseHandle")
    result = observe(lease)
    assert result.state is observer.ObservationState.OBSERVATION_LOST
    assert lease.closes == 1


def test_utc_nie_wplywa_na_pomiar(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    def utc_unavailable() -> float:
        raise AssertionError("obserwacja nie może używać zegara UTC")

    with monkeypatch.context() as limited:
        limited.setattr(time, "time", utc_unavailable)
        result = observe(Lease())
    assert result.state is observer.ObservationState.CONTINUOUS
    assert result.observed_100ns >= 5_000_000


def test_stary_json_i_pid_nie_sa_wejsciem_zatwierdzajacym() -> None:
    parameters = inspect.signature(observer.observe_backend).parameters
    assert set(parameters) == {"seconds", "stop"}
    previous = dataclasses.asdict(observe(Lease()))
    with pytest.raises(TypeError):
        observer.observe_backend(seconds=1, previous=previous)
    with pytest.raises(TypeError):
        observer.observe_backend(seconds=1, pid=123)
    assert "PASS" not in previous.values()
    assert previous["duration_24h"] == "BLOCKED"


def test_nowy_obserwator_zaczyna_wlasny_przedzial_od_zera() -> None:
    first, second = observe(Lease()), observe(Lease())
    assert second.observed_100ns == first.observed_100ns
    assert second.observed_100ns < 86_400 * 10_000_000
    assert second.duration_24h == "BLOCKED"


def test_blad_cleanup_nie_maskuje_poczatkowego_zakonczenia_jako_sukcesu() -> None:
    lease = Lease()
    lease.on_wait = lambda: True
    lease.close_error = OSError(6, "CloseHandle")
    result = observe(lease)
    assert result.state is observer.ObservationState.OBSERVATION_LOST
    assert lease.closes == 1


class NativeHandle:
    def __init__(self) -> None:
        self.closes = 0

    def Close(self) -> None:
        self.closes += 1


@pytest.fixture
def native_service(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Prawdziwy pin Stage 9 z atrapami wyłącznie na granicy wywołań Win32."""
    pipe = observer.pipe
    name = pipe.CONTRACT.backend_service
    image = rf"C:\Program Files\CryptoHunter\{name}.exe"
    sid = "S-1-5-80-111"
    state: dict[str, Any] = {
        "process_id": 123,
        "state": 4,
        "account": rf"NT SERVICE\{name}",
        "command_line": f'"{image}"',
        "service_type": 16,
        "token_sid": sid,
        "process_path": image,
        "process_started": datetime(2026, 11, 2, tzinfo=timezone.utc),
        "alive": 258,
    }
    opened: list[tuple[Any, ...]] = []
    process, token = NativeHandle(), NativeHandle()
    closed: list[Any] = []
    waited: list[tuple[Any, int]] = []

    def wait(handle: Any, milliseconds: int) -> int:
        waited.append((handle, milliseconds))
        return state["alive"]

    native = SimpleNamespace(
        api=SimpleNamespace(OpenProcess=lambda *args: (opened.append(args), process)[1]),
        event=SimpleNamespace(WaitForSingleObject=wait),
        process=SimpleNamespace(
            QueryFullProcessImageName=lambda *_: state["process_path"],
            GetProcessTimes=lambda *_: {"CreationTime": state["process_started"]},
        ),
        security=SimpleNamespace(
            OpenProcessToken=lambda *_: token,
            LookupAccountName=lambda *_: (sid, None, None),
            ConvertSidToStringSid=str,
            GetTokenInformation=lambda *_: (state["token_sid"], 0),
        ),
        service=SimpleNamespace(
            OpenSCManager=lambda *_: "manager",
            OpenService=lambda *_: "service",
            QueryServiceStatusEx=lambda *_: {
                "ProcessId": state["process_id"],
                "CurrentState": state["state"],
            },
            QueryServiceConfig=lambda *_: (
                state["service_type"],
                0,
                0,
                state["command_line"],
                0,
                0,
                0,
                state["account"],
            ),
            CloseServiceHandle=closed.append,
        ),
    )
    monkeypatch.setattr(pipe, "_win32", lambda: native)
    monkeypatch.setattr(pipe, "_canonical_image", lambda _: image)
    return SimpleNamespace(
        state=state,
        native=native,
        process=process,
        token=token,
        closed=closed,
        opened=opened,
        waited=waited,
    )


def assert_service_closed(native: SimpleNamespace) -> None:
    assert native.process.closes == native.token.closes == 1
    assert native.closed == ["service", "manager"]


def test_adapter_utrzymuje_pin_stage9_z_minimalnymi_prawami(
    native_service: SimpleNamespace,
) -> None:
    lease = observer._ServiceLease()
    result = observe(lease)
    assert result.state is observer.ObservationState.CONTINUOUS
    assert native_service.opened == [(0x101000, False, 123)]
    assert native_service.waited
    assert all(handle is native_service.process for handle, _ in native_service.waited)
    assert_service_closed(native_service)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("process_id", 999),
        ("process_started", datetime(2026, 11, 3, tzinfo=timezone.utc)),
        ("process_path", r"C:\Temp\podmieniony.exe"),
        ("account", "LocalSystem"),
        ("command_line", r"C:\Temp\podmieniony.exe"),
        ("token_sid", "S-1-5-18"),
        ("state", 1),
        ("service_type", 32),
    ],
    ids=["PID", "PID-reuse", "obraz", "konto", "SCM-program", "token", "stan", "typ"],
)
def test_rzeczywisty_pin_odmawia_podmiany_tozsamosci_i_zamyka_zasoby(
    native_service: SimpleNamespace, field: str, replacement: Any
) -> None:
    lease = observer._ServiceLease()
    native_service.state[field] = replacement
    result = observe(lease)
    assert result.state is observer.ObservationState.IDENTITY_CHANGED
    assert native_service.opened == [(0x101000, False, 123)]
    assert_service_closed(native_service)


def test_adapter_klasyfikuje_zakonczenie_zamiast_podmiany(
    native_service: SimpleNamespace,
) -> None:
    lease = observer._ServiceLease()
    native_service.state["alive"] = 0
    result = observe(lease)
    assert result.state is observer.ObservationState.PROCESS_EXITED
    assert_service_closed(native_service)


def test_adapter_wykrywa_zakonczenie_pomiedzy_wait_i_assert(
    native_service: SimpleNamespace,
) -> None:
    lease = observer._ServiceLease()
    native_service.state["alive"] = 0
    try:
        with pytest.raises(observer._ProcessExited):
            lease.assert_current()
    finally:
        lease.close()
    assert_service_closed(native_service)


@pytest.mark.parametrize("status", [0xFFFFFFFF, 128, 123])
def test_adapter_odmawia_nieznanej_odpowiedzi_waitforsingleobject(
    native_service: SimpleNamespace, status: int
) -> None:
    lease = observer._ServiceLease()
    native_service.state["alive"] = status
    result = observe(lease)
    assert result.state is observer.ObservationState.OBSERVATION_LOST
    assert_service_closed(native_service)


def test_blad_scm_po_otwarciu_uchwytu_odbiera_obserwacje(
    native_service: SimpleNamespace,
) -> None:
    lease = observer._ServiceLease()

    def fail(*_: Any) -> Any:
        raise OSError(5, "QueryServiceStatusEx")

    native_service.native.service.QueryServiceStatusEx = fail
    result = observe(lease)
    assert result.state is observer.ObservationState.OBSERVATION_LOST
    assert_service_closed(native_service)


@pytest.mark.parametrize("field", ["state", "service_type", "account"])
def test_niezgodny_scm_przed_startem_nie_ustanawia_obserwacji(
    native_service: SimpleNamespace, field: str
) -> None:
    native_service.state[field] = "niezgodne"
    result = observer._observe(
        observer._ServiceLease, Clock, seconds=0.5, scope="SYNTHETIC_TEST_ONLY"
    )
    assert result.state is observer.ObservationState.NOT_OBSERVED
    assert native_service.opened == []
    assert native_service.closed == ["service", "manager"]


def test_cleanup_probuje_wszystkich_zasobow_po_bledzie_zamkniecia_tokenu(
    native_service: SimpleNamespace,
) -> None:
    lease = observer._ServiceLease()

    def fail() -> None:
        native_service.token.closes += 1
        raise OSError(6, "token.Close")

    native_service.token.Close = fail
    result = observe(lease)
    assert result.state is observer.ObservationState.OBSERVATION_LOST
    assert_service_closed(native_service)


def test_helper_exit_pomiedzy_timeout_i_poll_jest_zakonczeniem(
    native_service: SimpleNamespace,
) -> None:
    state: dict[str, Any] = {"exit": None}
    child = SimpleNamespace(pid=123, poll=lambda: state["exit"])
    lease = observer._HelperProcessLease(child)

    def timeout_then_exit(_handle: Any, milliseconds: int) -> int:
        if milliseconds > 0:
            state["exit"] = 0
            native_service.state["alive"] = 0
            return 258
        return native_service.state["alive"]

    native_service.native.event.WaitForSingleObject = timeout_then_exit
    result = observe(lease)
    assert result.state is observer.ObservationState.PROCESS_EXITED
    assert native_service.opened == [(0x101000, False, 123)]
    assert native_service.process.closes == 1


def test_helper_zamyka_uchwyt_po_bledzie_odczytu_tozsamosci(
    native_service: SimpleNamespace,
) -> None:
    def unavailable(*_: Any) -> Any:
        raise OSError(6, "GetProcessTimes")

    native_service.native.process.GetProcessTimes = unavailable
    child = SimpleNamespace(pid=123, poll=lambda: None)
    with pytest.raises(OSError):
        observer._HelperProcessLease(child)
    assert native_service.process.closes == 1


@pytest.mark.parametrize("read_error", [False, True])
def test_helper_exit_w_trakcie_odczytu_tozsamosci(
    native_service: SimpleNamespace, read_error: bool
) -> None:
    child = SimpleNamespace(pid=123, poll=lambda: None)
    lease = observer._HelperProcessLease(child)

    def image(*_: Any) -> str:
        native_service.state["alive"] = 0
        if read_error:
            raise OSError(6, "Proces zakończył się podczas QueryFullProcessImageName")
        return r"C:\Temp\nieaktualny.exe"

    native_service.native.process.QueryFullProcessImageName = image
    result = observe(lease)
    assert result.state is observer.ObservationState.PROCESS_EXITED
    assert native_service.process.closes == 1


def test_po_otwarciu_helper_jest_potwierdzany_uchwytem_bez_poll(
    native_service: SimpleNamespace,
) -> None:
    child = SimpleNamespace(pid=123, poll=lambda: None)
    lease = observer._HelperProcessLease(child)

    def unavailable() -> None:
        raise AssertionError("Źródłem ciągłości po otwarciu jest zachowany uchwyt")

    child.poll = unavailable
    result = observe(lease)
    assert result.state is observer.ObservationState.CONTINUOUS
    assert native_service.process.closes == 1


def test_scm_blad_odczytu_po_zakonczeniu_zachowuje_stan_zakonczenia(
    native_service: SimpleNamespace,
) -> None:
    lease = observer._ServiceLease()

    def ended(*_: Any) -> Any:
        native_service.state["alive"] = 0
        raise OSError(6, "Usługa zakończyła się podczas QueryServiceStatusEx")

    native_service.native.service.QueryServiceStatusEx = ended
    result = observe(lease)
    assert result.state is observer.ObservationState.PROCESS_EXITED
    assert_service_closed(native_service)


def test_helper_nie_otwiera_procesu_zakonczonego_przed_startem(
    native_service: SimpleNamespace,
) -> None:
    child = SimpleNamespace(pid=123, poll=lambda: 0)
    with pytest.raises(observer.ObservationLost):
        observer._HelperProcessLease(child)
    assert native_service.opened == []


@pytest.mark.parametrize("changed", ["pid", "process_started", "process_path"])
def test_helper_nie_przejmuje_nastepnej_tozsamosci(
    native_service: SimpleNamespace, changed: str
) -> None:
    child = SimpleNamespace(pid=123, poll=lambda: None)
    lease = observer._HelperProcessLease(child)
    if changed == "pid":
        child.pid = 456
    elif changed == "process_started":
        native_service.state[changed] = datetime(2026, 11, 3, tzinfo=timezone.utc)
    else:
        native_service.state[changed] = r"C:\Temp\nastepca.exe"
    result = observe(lease)
    assert result.state is observer.ObservationState.IDENTITY_CHANGED
    assert native_service.opened == [(0x101000, False, 123)]
    assert native_service.process.closes == 1


def test_publiczny_helper_zawsze_pozostaje_test_only(
    native_service: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observer, "_NativeClock", Clock)
    child = SimpleNamespace(pid=123, poll=lambda: None)
    result = observer.observe_test_process(child, seconds=0.5)
    assert result.state is observer.ObservationState.CONTINUOUS
    assert result.scope == "TEST_ONLY"
    assert result.duration_24h == "BLOCKED"
    assert native_service.process.closes == 1


def test_publiczny_backend_korzysta_wylacznie_z_kontraktu_scm(
    native_service: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observer, "_NativeClock", Clock)
    result = observer.observe_backend(seconds=0.5)
    assert result.state is observer.ObservationState.CONTINUOUS
    assert result.scope == "SCM_CONTRACT"
    assert result.duration_24h == "BLOCKED"
    assert result.process_identity is not None
    assert result.process_identity["name"] == observer.CONTRACT.backend_service
    assert_service_closed(native_service)
