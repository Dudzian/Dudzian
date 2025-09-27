"""Domyślna implementacja routera alertów."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, MutableSequence

from bot_core.alerts.base import AlertChannel, AlertMessage, AlertRouter, AlertAuditLog, AlertDeliveryError


@dataclass(slots=True)
class DefaultAlertRouter(AlertRouter):
    """Zarządza kanałami powiadomień i rejestruje zdarzenia w audycie."""

    audit_log: AlertAuditLog
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("bot_core.alerts"))
    stop_on_error: bool = False
    channels: MutableSequence[AlertChannel] = field(default_factory=list)

    def register(self, channel: AlertChannel) -> None:
        if any(existing.name == channel.name for existing in self.channels):
            raise ValueError(f"Kanał o nazwie '{channel.name}' został już zarejestrowany")
        self.channels.append(channel)

    def dispatch(self, message: AlertMessage) -> None:
        failures: Dict[str, str] = {}
        for channel in list(self.channels):
            try:
                channel.send(message)
            except AlertDeliveryError as exc:  # pragma: no cover - defensive guard
                self.logger.error("Nie udało się wysłać alertu", extra={"channel": channel.name, "error": str(exc)})
                failures[channel.name] = str(exc)
                if self.stop_on_error:
                    raise
            except Exception as exc:  # noqa: BLE001
                error_msg = f"Nieznany błąd kanału {channel.name}: {exc}"
                self.logger.exception(error_msg)
                failures[channel.name] = str(exc)
                if self.stop_on_error:
                    raise AlertDeliveryError(error_msg) from exc
            else:
                self.audit_log.append(message, channel=channel.name)

        if failures and not self.stop_on_error:
            summary = ", ".join(f"{name}: {reason}" for name, reason in failures.items())
            self.logger.warning("Część kanałów zgłosiła błędy: %s", summary)

    def health_snapshot(self) -> Dict[str, Dict[str, str]]:
        snapshot: Dict[str, Dict[str, str]] = {}
        now = datetime.now(timezone.utc).isoformat()
        for channel in self.channels:
            data = {"checked_at": now}
            try:
                data.update(channel.health_check())
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Błąd podczas health-check kanału %s", channel.name)
                data.update({"status": "error", "detail": str(exc)})
            snapshot[channel.name] = data
        return snapshot


__all__ = ["DefaultAlertRouter"]

