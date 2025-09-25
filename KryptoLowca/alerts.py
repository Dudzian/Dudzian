"""Centralny dispatcher alertów i wspólne wyjątki dla całego bota."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class AlertSeverity(str, Enum):
    """Stopnie ważności alertów przekazywanych do dashboardu."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(slots=True)
class AlertEvent:
    """Struktura pojedynczego alertu przekazywanego do słuchaczy."""

    message: str
    severity: AlertSeverity
    source: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    exception: Optional[BaseException] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.context)
        payload.update(
            {
                "message": self.message,
                "severity": self.severity.value,
                "source": self.source,
                "timestamp": self.timestamp,
            }
        )
        if self.exception is not None:
            payload.setdefault("exception", repr(self.exception))
        return payload


class BotError(Exception):
    """Bazowy wyjątek domenowy – mapowany na alerty."""

    severity: AlertSeverity = AlertSeverity.ERROR
    source: str = "core"

    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.context = context or {}

    def to_alert(self) -> AlertEvent:
        return AlertEvent(
            message=str(self),
            severity=self.severity,
            source=self.source,
            context=dict(self.context),
            exception=self,
        )


class AlertDispatcher:
    """Prosty dispatcher przekazujący alerty do zarejestrowanych słuchaczy."""

    def __init__(self) -> None:
        self._listeners: Dict[str, Callable[[AlertEvent], None]] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def register(self, listener: Callable[[AlertEvent], None], *, name: Optional[str] = None) -> str:
        """Zarejestruj słuchacza i zwróć jego identyfikator."""

        with self._lock:
            token = name or f"listener-{self._counter}"  # prosty identyfikator
            while token in self._listeners:
                self._counter += 1
                token = f"listener-{self._counter}"
            self._listeners[token] = listener
            self._counter += 1
            return token

    def unregister(self, token: str) -> None:
        """Usuń słuchacza – brak błędu gdy nie istnieje."""

        with self._lock:
            self._listeners.pop(token, None)

    def dispatch(self, event: AlertEvent) -> None:
        listeners: Dict[str, Callable[[AlertEvent], None]]
        with self._lock:
            listeners = dict(self._listeners)
        for name, listener in listeners.items():
            try:
                listener(event)
            except Exception:  # pragma: no cover - logujemy, ale nie przerywamy
                logger.exception("Alert listener '%s' zgłosił wyjątek", name)

    def clear(self) -> None:
        """Usuń wszystkich słuchaczy (używane w testach)."""

        with self._lock:
            self._listeners.clear()


_DISPATCHER = AlertDispatcher()


def get_alert_dispatcher() -> AlertDispatcher:
    """Zwróć globalny dispatcher alertów."""

    return _DISPATCHER


def emit_alert(
    message: str,
    *,
    severity: AlertSeverity = AlertSeverity.WARNING,
    source: str = "core",
    context: Optional[Dict[str, Any]] = None,
    exception: Optional[BaseException] = None,
) -> AlertEvent:
    """Zbuduj i roześlij alert do wszystkich słuchaczy."""

    event = AlertEvent(
        message=message,
        severity=severity,
        source=source,
        context=dict(context or {}),
        exception=exception,
    )
    _DISPATCHER.dispatch(event)
    return event


__all__ = [
    "AlertDispatcher",
    "AlertEvent",
    "AlertSeverity",
    "BotError",
    "emit_alert",
    "get_alert_dispatcher",
]
