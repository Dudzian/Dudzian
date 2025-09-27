"""Repozytoria audytu wykorzystywane przez system alertów."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Mapping

from bot_core.alerts.base import AlertAuditLog, AlertMessage


@dataclass(slots=True)
class AlertAuditEntry:
    """Pojedynczy zapis audytowy odpowiadający wysłanemu komunikatowi."""

    channel: str
    message: AlertMessage
    created_at: datetime

    def as_dict(self) -> Mapping[str, str]:
        """Eksportuje wpis w formacie przyjaznym serializacji."""

        payload: dict[str, str] = {
            "channel": self.channel,
            "category": self.message.category,
            "title": self.message.title,
            "severity": self.message.severity,
            "timestamp": self.message.timestamp.isoformat(),
            "created_at": self.created_at.isoformat(),
        }
        payload.update({f"ctx_{k}": v for k, v in self.message.context.items()})
        payload["body"] = self.message.body
        return payload


class InMemoryAlertAuditLog(AlertAuditLog):
    """Prosta implementacja audytu na potrzeby środowisk deweloperskich."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: List[AlertAuditEntry] = []

    def append(self, message: AlertMessage, *, channel: str) -> None:
        entry = AlertAuditEntry(channel=channel, message=message, created_at=message.timestamp)
        self._entries.append(entry)

    def export(self) -> Iterable[Mapping[str, str]]:
        return tuple(entry.as_dict() for entry in self._entries)


__all__ = ["AlertAuditEntry", "InMemoryAlertAuditLog"]

