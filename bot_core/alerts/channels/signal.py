"""Adapter wysyłający alerty przez usługę Signal."""
from __future__ import annotations

import json
import logging
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Protocol, Sequence
from urllib import request

from bot_core.alerts.base import AlertChannel, AlertDeliveryError, AlertMessage


class _SignalHttpOpener(Protocol):
    def __call__(
        self,
        req: request.Request,
        *,
        timeout: float,
        context: ssl.SSLContext | None,
    ) -> request.addinfourl:
        ...


def _default_signal_opener(
    req: request.Request,
    *,
    timeout: float,
    context: ssl.SSLContext | None,
) -> request.addinfourl:
    return request.urlopen(req, timeout=timeout, context=context)  # noqa: S310 - kontrolujemy docelowy serwer


@dataclass(slots=True)
class SignalChannel(AlertChannel):
    """Integracja z lokalną instancją ``signal-cli`` lub kompatybilnym API."""

    service_url: str
    sender_number: str
    recipients: Sequence[str]
    auth_token: str | None = None
    verify_tls: bool = True
    name: str = "signal"
    timeout: float = 10.0
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("bot_core.alerts.signal"))
    _opener: _SignalHttpOpener = field(default=_default_signal_opener, repr=False)
    _ssl_context: ssl.SSLContext | None = field(default=None, init=False, repr=False)
    _last_success: datetime | None = field(default=None, init=False, repr=False)
    _last_error: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ssl_context = self._build_ssl_context()

    def send(self, message: AlertMessage) -> None:
        url = f"{self.service_url.rstrip('/')}/v2/send"
        payload = {
            "message": self._format_body(message),
            "number": self.sender_number,
            "recipients": list(self.recipients),
        }
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)

        try:
            with self._opener(req, timeout=self.timeout, context=self._ssl_context) as response:
                status = getattr(response, "status", response.getcode())
                raw = response.read()
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            self.logger.exception("Błąd wysyłki przez Signal")
            raise AlertDeliveryError(f"Signal: nie udało się wysłać powiadomienia ({exc})") from exc

        if status >= 400:
            detail = raw.decode("utf-8", errors="ignore") if raw else ""
            self._last_error = detail or str(status)
            self.logger.error("Signal zwrócił status %s: %s", status, detail)
            raise AlertDeliveryError(f"Signal zwrócił status {status}: {detail}")

        if raw:
            try:
                parsed: Dict[str, object] = json.loads(raw)
            except json.JSONDecodeError:  # pragma: no cover - brak JSON traktujemy jako sukces
                parsed = {"status": "ok"}
            error = parsed.get("error") if isinstance(parsed, dict) else None
            if error:
                self._last_error = str(error)
                self.logger.error("Signal zgłosił błąd aplikacyjny: %s", error)
                raise AlertDeliveryError(f"Signal zgłosił błąd aplikacyjny: {error}")

        self._last_success = datetime.now(tz=timezone.utc)
        self._last_error = None

    def health_check(self) -> Dict[str, str]:
        status = "ok" if self._last_error is None else "error"
        data: Dict[str, str] = {"status": status, "service_url": self.service_url}
        if self._last_success:
            data["last_success"] = self._last_success.isoformat()
        if self._last_error:
            data["last_error"] = self._last_error
        return data

    def _format_body(self, message: AlertMessage) -> str:
        context = "\n".join(f"- {key}: {value}" for key, value in sorted(message.context.items()))
        parts = [f"[{message.severity.upper()}] {message.title}", message.body]
        if context:
            parts.append(context)
        parts.append(f"Kategoria: {message.category}")
        return "\n\n".join(part for part in parts if part)

    def _build_ssl_context(self) -> ssl.SSLContext | None:
        if self.service_url.lower().startswith("http://"):
            return None
        if self.verify_tls:
            return ssl.create_default_context()
        return ssl._create_unverified_context()  # type: ignore[attr-defined]  # pragma: no cover


__all__ = ["SignalChannel"]
