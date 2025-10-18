"""Lightweight alert dispatcher for in-process listeners."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

LOGGER = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Severity levels understood by the legacy UI surfaces."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(slots=True)
class AlertEvent:
    """Payload dispatched to registered listeners."""

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
    """Domain exception automatically translated into an alert."""

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
    """Simple dispatcher delivering alerts to registered listeners."""

    def __init__(self) -> None:
        self._listeners: Dict[str, Callable[[AlertEvent], None]] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def register(self, listener: Callable[[AlertEvent], None], *, name: Optional[str] = None) -> str:
        """Register a listener and return its identifier."""

        with self._lock:
            token = name or f"listener-{self._counter}"
            while token in self._listeners:
                self._counter += 1
                token = f"listener-{self._counter}"
            self._listeners[token] = listener
            self._counter += 1
            return token

    def unregister(self, token: str) -> None:
        """Remove a listener; silently ignore missing entries."""

        with self._lock:
            self._listeners.pop(token, None)

    def dispatch(self, event: AlertEvent) -> None:
        listeners: Dict[str, Callable[[AlertEvent], None]]
        with self._lock:
            listeners = dict(self._listeners)
        for name, listener in listeners.items():
            try:
                listener(event)
            except Exception:  # pragma: no cover - listeners are third-party callbacks
                LOGGER.exception("Alert listener '%s' raised an exception", name)

    def clear(self) -> None:
        """Remove every registered listener (mostly used in tests)."""

        with self._lock:
            self._listeners.clear()


_DISPATCHER = AlertDispatcher()


def get_alert_dispatcher() -> AlertDispatcher:
    """Return the process-global dispatcher instance."""

    return _DISPATCHER


def emit_alert(
    message: str,
    *,
    severity: AlertSeverity = AlertSeverity.WARNING,
    source: str = "core",
    context: Optional[Dict[str, Any]] = None,
    exception: Optional[BaseException] = None,
) -> AlertEvent:
    """Create an alert event and dispatch it to listeners."""

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
