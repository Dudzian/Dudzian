"""Pomocnicze stuby wykorzystywane w testach serwisów telemetrycznych."""
from __future__ import annotations

from typing import Any, Mapping


class RouterStub:
    """Minimalna implementacja routera alertów."""

    def __init__(self, audit_log: Any) -> None:
        self.audit_log = audit_log
        self.registered: list[Any] = []

    def register(self, channel: Any) -> None:
        self.registered.append(channel)


class MemoryAuditStub:
    """Niewielka atrapa pamięciowego audytu alertów."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class SinkStub:
    """Domyślny sink telemetryczny zapisujący przekazane argumenty."""

    def __init__(self, router: Any, **kwargs: Any) -> None:
        self.router = router
        self.kwargs = kwargs


class UiSinkStub(SinkStub):
    """Lekka atrapa sinka UI wykorzystywana w wielu scenariuszach CLI."""

    def __init__(self, router: Any, **kwargs: Any) -> None:
        super().__init__(router, **kwargs)
        self.jsonl_path = kwargs.get("jsonl_path")


class RuntimeServerStub:
    """Prosty serwer gRPC wykorzystywany w testach CLI."""

    def __init__(self, address: str = "127.0.0.1:6000") -> None:
        self.address = address
        self.started = False
        self.stop_calls: list[Any] = []
        self.wait_calls: list[Any] = []

    def start(self) -> None:
        self.started = True

    def wait_for_termination(self, timeout: Any = None) -> bool:
        self.wait_calls.append(timeout)
        return True

    def stop(self, grace: Any = None) -> None:
        self.stop_calls.append(grace)


class MetricsServerStub(RuntimeServerStub):
    """Rozszerzenie serwera o dodatkowe metadane telemetryczne."""

    def __init__(
        self,
        *,
        address: str,
        history_size: int,
        tls_enabled: bool,
        require_client_auth: bool,
        sinks: tuple[Any, ...],
        runtime_metadata: Mapping[str, object],
    ) -> None:
        super().__init__(address)
        self._sinks = sinks
        self.history_size = history_size
        self.tls_enabled = tls_enabled
        self.tls_client_auth_required = require_client_auth
        self.runtime_metadata = runtime_metadata

    @property
    def sinks(self) -> tuple[Any, ...]:  # noqa: D401 - zachowanie interfejsu
        return self._sinks


def get_security_section(payload: Mapping[str, object]) -> Mapping[str, object]:
    """Wyodrębnia sekcję zabezpieczeń z wygenerowanego planu."""

    section = payload.get("security")
    assert isinstance(section, Mapping)
    fail_section = section.get("fail_on_security_warnings")
    assert isinstance(fail_section, Mapping)
    return fail_section
