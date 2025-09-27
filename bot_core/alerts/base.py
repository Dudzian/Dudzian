"""Interfejsy kanałów alertów oraz audytu."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Mapping, MutableSequence, Protocol


class AlertDeliveryError(RuntimeError):
    """Sygnalizuje niepowodzenie podczas wysyłki alertu na kanał."""


@dataclass(slots=True)
class AlertMessage:
    """Reprezentuje sformatowaną treść alertu wraz z metadanymi."""

    category: str
    title: str
    body: str
    severity: str
    context: Mapping[str, str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


class AlertChannel(abc.ABC):
    """Każdy kanał powiadomień musi implementować ten interfejs."""

    name: str

    @abc.abstractmethod
    def send(self, message: AlertMessage) -> None:
        """Publikuje alert na danym kanale."""

    @abc.abstractmethod
    def health_check(self) -> Mapping[str, str]:
        """Zwraca stan kanału wykorzystywany w raportach SLO."""


class AlertAuditLog(Protocol):
    """Minimalny kontrakt repozytorium audytowego."""

    def append(self, message: AlertMessage, *, channel: str) -> None:
        ...

    def export(self) -> Iterable[Mapping[str, str]]:
        ...


class AlertRouter(abc.ABC):
    """Centralna szyna, która będzie sterować dostarczaniem alertów."""

    channels: MutableSequence[AlertChannel]

    @abc.abstractmethod
    def register(self, channel: AlertChannel) -> None:
        ...

    @abc.abstractmethod
    def dispatch(self, message: AlertMessage) -> None:
        ...
