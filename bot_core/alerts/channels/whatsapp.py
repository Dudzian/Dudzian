"""Adapter do wysyłki alertów przez WhatsApp Business API."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Protocol, Sequence
from urllib import request

from bot_core.alerts.base import AlertChannel, AlertDeliveryError, AlertMessage


class _HttpOpener(Protocol):
    def __call__(self, req: request.Request, *, timeout: float) -> request.addinfourl:
        ...


def _default_opener(req: request.Request, *, timeout: float) -> request.addinfourl:
    return request.urlopen(req, timeout=timeout)  # noqa: S310 - korzystamy z zaufanego Graph API


@dataclass(slots=True)
class WhatsAppChannel(AlertChannel):
    """Integracja z Graph API dla WhatsApp Business."""

    phone_number_id: str
    access_token: str
    recipients: Sequence[str]
    api_base_url: str = "https://graph.facebook.com"
    api_version: str = "v16.0"
    name: str = "whatsapp"
    timeout: float = 10.0
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("bot_core.alerts.whatsapp"))
    _opener: _HttpOpener = field(default=_default_opener, repr=False)
    _last_success: datetime | None = field(default=None, init=False, repr=False)
    _last_error: str | None = field(default=None, init=False, repr=False)

    def send(self, message: AlertMessage) -> None:
        url = (
            f"{self.api_base_url.rstrip('/')}/{self.api_version}/{self.phone_number_id}/messages"
        )
        payload = {
            "messaging_product": "whatsapp",
            "type": "text",
            "text": {"body": self._format_body(message)},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

        for recipient in self.recipients:
            payload["to"] = recipient
            req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            self._send_request(req)

        self._last_success = datetime.now(tz=timezone.utc)
        self._last_error = None

    def _send_request(self, req: request.Request) -> None:
        try:
            with self._opener(req, timeout=self.timeout) as response:
                status = getattr(response, "status", response.getcode())
                raw = response.read()
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            self.logger.exception("Błąd wysyłki przez WhatsApp")
            raise AlertDeliveryError(f"WhatsApp: nie udało się wysłać powiadomienia ({exc})") from exc

        if status >= 400:
            detail = raw.decode("utf-8", errors="ignore") if raw else ""
            self._last_error = detail or str(status)
            self.logger.error("WhatsApp zwrócił status %s: %s", status, detail)
            raise AlertDeliveryError(f"WhatsApp zwrócił status {status}: {detail}")

        if raw:
            try:
                parsed: Dict[str, object] = json.loads(raw)
            except json.JSONDecodeError:  # pragma: no cover
                parsed = {"status": "ok"}
            if isinstance(parsed, dict) and parsed.get("error"):
                error = parsed["error"]
                self._last_error = str(error)
                self.logger.error("WhatsApp zwrócił błąd aplikacyjny: %s", error)
                raise AlertDeliveryError(f"WhatsApp zwrócił błąd aplikacyjny: {error}")

    def health_check(self) -> Dict[str, str]:
        status = "ok" if self._last_error is None else "error"
        data: Dict[str, str] = {"status": status, "phone_number_id": self.phone_number_id}
        if self._last_success:
            data["last_success"] = self._last_success.isoformat()
        if self._last_error:
            data["last_error"] = self._last_error
        return data

    def _format_body(self, message: AlertMessage) -> str:
        context = "\n".join(f"• {key}: {value}" for key, value in sorted(message.context.items()))
        parts = [f"{message.title} ({message.severity.upper()})", message.body]
        if context:
            parts.append(context)
        parts.append(f"Kategoria: {message.category}")
        return "\n\n".join(part for part in parts if part)


__all__ = ["WhatsAppChannel"]
