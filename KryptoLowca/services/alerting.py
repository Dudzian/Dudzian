"""Rozszerzone zarządzanie alertami (email, Slack, webhook) dla bota."""
from __future__ import annotations

import json
import logging
import smtplib
import ssl
import threading
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

import requests

from bot_core.alerts import AlertEvent, AlertSeverity, get_alert_dispatcher

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class AlertSink(Protocol):
    """Interfejs docelowego kanału powiadomień."""

    def send(self, event: AlertEvent) -> None:  # pragma: no cover - interfejs
        ...

    def close(self) -> None:  # pragma: no cover - opcjonalne
        ...


@dataclass(slots=True)
class EmailAlertSink:
    """Proste wysyłanie alertów przez SMTP (np. konta alertowe)."""

    host: str
    port: int
    username: Optional[str]
    password: Optional[str]
    sender: str
    recipients: Sequence[str]
    use_tls: bool = True
    subject_prefix: str = "[KryptoLowca]"
    timeout: float = 10.0

    def send(self, event: AlertEvent) -> None:
        if not self.recipients:
            logger.debug("Brak odbiorców alertu email – pomijam wysyłkę")
            return

        subject = f"{self.subject_prefix} {event.severity.value.upper()} - {event.source}"
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        body = [event.message, ""]
        if event.context:
            body.append("Kontekst:")
            body.append(json.dumps(event.context, indent=2, ensure_ascii=False))
        msg.set_content("\n".join(body))

        try:
            smtp_cls: Callable[..., smtplib.SMTP] = smtplib.SMTP
            context = ssl.create_default_context() if self.use_tls else None
            with smtp_cls(self.host, self.port, timeout=self.timeout) as smtp:
                if self.use_tls:
                    smtp.starttls(context=context)
                if self.username and self.password:
                    smtp.login(self.username, self.password)
                smtp.send_message(msg)
        except Exception:
            logger.exception("Nie udało się wysłać alertu email")

    def close(self) -> None:  # pragma: no cover - nie utrzymujemy połączenia
        return None


@dataclass(slots=True)
class SlackWebhookSink:
    """Wysyłanie alertów do kanału Slack poprzez webhook."""

    webhook_url: str
    timeout: float = 5.0

    def send(self, event: AlertEvent) -> None:
        payload = {
            "text": f"*{event.severity.value.upper()}* ({event.source}) {event.message}",
            "attachments": [
                {
                    "color": self._color(event.severity),
                    "fields": [
                        {"title": key, "value": str(value), "short": True}
                        for key, value in sorted(event.context.items())
                    ],
                }
            ]
            if event.context
            else [],
        }
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except Exception:
            logger.exception("Błąd wysyłki alertu Slack")

    @staticmethod
    def _color(severity: AlertSeverity) -> str:
        mapping = {
            AlertSeverity.INFO: "#439FE0",
            AlertSeverity.WARNING: "#FFCC00",
            AlertSeverity.ERROR: "#FF3300",
            AlertSeverity.CRITICAL: "#CC0000",
        }
        return mapping.get(severity, "#439FE0")

    def close(self) -> None:  # pragma: no cover - brak zasobów do zwolnienia
        return None


@dataclass(slots=True)
class WebhookAlertSink:
    """Ogólny webhook HTTP POST (np. integracje własne)."""

    url: str
    timeout: float = 5.0
    headers: Optional[Dict[str, str]] = None

    def send(self, event: AlertEvent) -> None:
        payload = {
            "message": event.message,
            "severity": event.severity.value,
            "source": event.source,
            "timestamp": event.timestamp,
            "context": event.context,
        }
        try:
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception:
            logger.exception("Błąd wysyłki alertu webhook")

    def close(self) -> None:  # pragma: no cover - brak zasobów do zwolnienia
        return None


class AlertManager:
    """Rejestruje kanały docelowe i nasłuchuje na dispatcherze alertów."""

    def __init__(self, sinks: Iterable[AlertSink]) -> None:
        self._sinks: List[AlertSink] = list(sinks)
        self._lock = threading.Lock()
        self._dispatcher = get_alert_dispatcher()
        self._token: Optional[str] = None
        self._register()

    def _register(self) -> None:
        if self._token is not None:
            return

        def _listener(event: AlertEvent) -> None:
            self._handle_event(event)

        self._token = self._dispatcher.register(_listener, name="alert-manager")

    def _handle_event(self, event: AlertEvent) -> None:
        sinks: List[AlertSink]
        with self._lock:
            sinks = list(self._sinks)
        for sink in sinks:
            try:
                sink.send(event)
            except Exception:
                logger.exception("Kanał alertowy %s zgłosił wyjątek", type(sink).__name__)

    def add_sink(self, sink: AlertSink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def remove_sink(self, predicate: Callable[[AlertSink], bool]) -> None:
        with self._lock:
            self._sinks = [sink for sink in self._sinks if not predicate(sink)]

    def close(self) -> None:
        if self._token is not None:
            self._dispatcher.unregister(self._token)
            self._token = None
        sinks: List[AlertSink]
        with self._lock:
            sinks = list(self._sinks)
            self._sinks.clear()
        for sink in sinks:
            try:
                sink.close()
            except Exception:
                logger.exception("Błąd zamykania kanału alertów %s", type(sink).__name__)

    @classmethod
    def from_config(cls, config: Sequence[Dict[str, Any]]) -> "AlertManager":
        """Zbuduj menedżera na podstawie listy konfiguracji kanałów."""

        sinks: List[AlertSink] = []
        for entry in config:
            if not isinstance(entry, dict):
                continue
            channel_type = str(entry.get("type") or "").lower()
            if channel_type == "email":
                recipients = entry.get("recipients") or []
                sinks.append(
                    EmailAlertSink(
                        host=str(entry.get("host")),
                        port=int(entry.get("port", 587)),
                        username=entry.get("username"),
                        password=entry.get("password"),
                        sender=str(entry.get("sender")),
                        recipients=[str(r) for r in recipients],
                        use_tls=bool(entry.get("use_tls", True)),
                        subject_prefix=str(entry.get("subject_prefix", "[KryptoLowca]")),
                        timeout=float(entry.get("timeout", 10.0)),
                    )
                )
            elif channel_type == "slack":
                sinks.append(
                    SlackWebhookSink(
                        webhook_url=str(entry.get("webhook_url")),
                        timeout=float(entry.get("timeout", 5.0)),
                    )
                )
            elif channel_type == "webhook":
                headers = entry.get("headers")
                sinks.append(
                    WebhookAlertSink(
                        url=str(entry.get("url")),
                        timeout=float(entry.get("timeout", 5.0)),
                        headers={str(k): str(v) for k, v in (headers or {}).items()},
                    )
                )
            else:
                logger.warning("Nieznany typ kanału alertów: %s", channel_type)
        return cls(sinks)


__all__ = [
    "AlertManager",
    "AlertSink",
    "EmailAlertSink",
    "SlackWebhookSink",
    "WebhookAlertSink",
]

