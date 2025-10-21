"""Wspólne atrapy kanałów alertowych używane w testach runtime i pipeline."""

from __future__ import annotations

from typing import Mapping

from bot_core.alerts.base import AlertChannel, AlertMessage


class CollectingChannel(AlertChannel):
    """Kanał zbierający wiadomości i raportujący podstawowy stan."""

    name = "collector"

    def __init__(self, *, health_overrides: Mapping[str, object] | None = None) -> None:
        self.messages: list[AlertMessage] = []
        self._health: dict[str, str] = {"status": "ok"}
        if health_overrides:
            self._health.update({str(key): str(value) for key, value in health_overrides.items()})

    def send(self, message: AlertMessage) -> None:
        self.messages.append(message)

    def health_check(self) -> Mapping[str, str]:
        return dict(self._health)


__all__ = ["CollectingChannel"]
