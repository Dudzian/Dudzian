"""Lightweight alert dispatcher for in-process listeners."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

LOGGER = logging.getLogger(__name__)
_OFFLINE_LOGGER = logging.getLogger("bot_core.alerts.offline")


class AlertSeverity(str, Enum):
    """Severity levels understood by the legacy UI surfaces."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


_SEVERITY_ORDER = {
    AlertSeverity.INFO: 0,
    AlertSeverity.WARNING: 1,
    AlertSeverity.ERROR: 2,
    AlertSeverity.CRITICAL: 3,
}


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


class OfflineAlertSink:
    """Odbiorca alertów zapisujący zdarzenia do lokalnych logów."""

    def __init__(self, *, min_severity: AlertSeverity = AlertSeverity.WARNING) -> None:
        self._min_severity = min_severity

    def handle(self, event: AlertEvent) -> None:
        severity = event.severity
        if not isinstance(severity, AlertSeverity):
            candidate = getattr(event.severity, "value", event.severity)
            try:
                severity = AlertSeverity(str(candidate).lower())
            except Exception:  # pragma: no cover - defensywne
                severity = AlertSeverity.INFO
        if _SEVERITY_ORDER.get(severity, 0) < _SEVERITY_ORDER.get(self._min_severity, 1):
            return
        message = "%s | %s" % (event.source, event.message)
        if severity is AlertSeverity.CRITICAL:
            _OFFLINE_LOGGER.critical(message, extra=event.to_dict())
        elif severity is AlertSeverity.ERROR:
            _OFFLINE_LOGGER.error(message, extra=event.to_dict())
        elif severity is AlertSeverity.WARNING:
            _OFFLINE_LOGGER.warning(message, extra=event.to_dict())
        else:
            _OFFLINE_LOGGER.info(message, extra=event.to_dict())


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
_OFFLINE_SINK_TOKEN: str | None = None


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


def ensure_offline_logging_sink(
    *,
    dispatcher: AlertDispatcher | None = None,
    min_severity: AlertSeverity = AlertSeverity.WARNING,
) -> str:
    """Rejestruje domyślnego słuchacza logującego alerty offline."""

    global _OFFLINE_SINK_TOKEN
    target = dispatcher or _DISPATCHER
    if _OFFLINE_SINK_TOKEN:
        return _OFFLINE_SINK_TOKEN
    sink = OfflineAlertSink(min_severity=min_severity)
    token = target.register(sink.handle, name="offline-logging-sink")
    _OFFLINE_SINK_TOKEN = token
    return token


__all__ = [
    "AlertDispatcher",
    "AlertEvent",
    "AlertSeverity",
    "BotError",
    "emit_alert",
    "get_alert_dispatcher",
    "OfflineAlertSink",
    "ensure_offline_logging_sink",
]
