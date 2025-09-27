"""Adapter SMS obsługujący dostawców z API HTTP (np. Twilio i operatorów lokalnych)."""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Protocol, Sequence
from urllib import parse, request

from bot_core.alerts.base import AlertChannel, AlertDeliveryError, AlertMessage
from bot_core.alerts.channels.providers import SmsProviderConfig


class _SmsHttpOpener(Protocol):
    def __call__(self, req: request.Request, *, timeout: float) -> request.addinfourl:
        ...


def _default_sms_opener(req: request.Request, *, timeout: float) -> request.addinfourl:
    return request.urlopen(req, timeout=timeout)  # noqa: S310 - kontrolujemy docelowe API


@dataclass(slots=True)
class SMSChannel(AlertChannel):
    """Wysyła wiadomości SMS poprzez API kompatybilne z modelem Twilio."""

    account_sid: str
    auth_token: str
    from_number: str
    recipients: Sequence[str]
    provider: SmsProviderConfig | None = None
    provider_base_url: str = "https://api.twilio.com"
    name: str = "sms"
    timeout: float = 10.0
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("bot_core.alerts.sms"))
    _opener: _SmsHttpOpener = field(default=_default_sms_opener, repr=False)
    _last_error: str | None = field(default=None, init=False, repr=False)
    _last_success: datetime | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.provider is not None:
            # Normalizujemy bazowy adres do postaci bez końcowego slasha.
            base = self.provider.api_base_url.rstrip("/")
            self.provider_base_url = base
            self.logger.debug(
                "SMSChannel skonfigurowany dla dostawcy", extra={"provider": self.provider.provider_id}
            )

    def send(self, message: AlertMessage) -> None:
        for recipient in self.recipients:
            self._send_single(recipient, message)
        self._last_success = datetime.now(tz=timezone.utc)
        self._last_error = None

    def _send_single(self, recipient: str, message: AlertMessage) -> None:
        url = f"{self.provider_base_url}/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        payload = parse.urlencode({
            "To": recipient,
            "From": self.from_number,
            "Body": self._build_body(message),
        }).encode("utf-8")
        req = request.Request(url, data=payload, method="POST")
        auth_header = base64.b64encode(f"{self.account_sid}:{self.auth_token}".encode("utf-8")).decode("ascii")
        req.add_header("Authorization", f"Basic {auth_header}")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")

        try:
            with self._opener(req, timeout=self.timeout) as response:
                status = getattr(response, "status", response.getcode())
                raw_body = response.read()
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            self.logger.exception("Błąd wysyłki SMS", extra={"recipient": recipient})
            raise AlertDeliveryError(f"Błąd wysyłki SMS do {recipient}: {exc}") from exc

        if status >= 400:
            detail = raw_body.decode("utf-8", errors="ignore") if raw_body else ""
            self._last_error = detail or str(status)
            self.logger.error(
                "Dostawca SMS zwrócił kod %s: %s",
                status,
                detail,
                extra={"recipient": recipient, "provider": self.provider.provider_id if self.provider else None},
            )
            raise AlertDeliveryError(f"Dostawca SMS zwrócił kod {status}: {detail}")

        if raw_body:
            try:
                decoded: Dict[str, object] = json.loads(raw_body)
            except json.JSONDecodeError:  # pragma: no cover - brak body JSON nie jest błędem krytycznym
                decoded = {"status": "unknown"}
            error_msg = decoded.get("message") if isinstance(decoded, dict) else None
            if error_msg:
                self._last_error = str(error_msg)
                self.logger.error(
                    "Dostawca SMS zgłosił błąd: %s",
                    error_msg,
                    extra={"recipient": recipient, "provider": self.provider.provider_id if self.provider else None},
                )
                raise AlertDeliveryError(f"Dostawca SMS zgłosił błąd: {error_msg}")

    def health_check(self) -> Dict[str, str]:
        status = "ok" if self._last_error is None else "error"
        data: Dict[str, str] = {"status": status}
        if self.provider is not None:
            data["provider"] = self.provider.provider_id
            data["country"] = self.provider.iso_country_code
        if self._last_success:
            data["last_success"] = self._last_success.isoformat()
        if self._last_error:
            data["last_error"] = self._last_error
        return data

    def _build_body(self, message: AlertMessage) -> str:
        context = "; ".join(f"{key}={value}" for key, value in sorted(message.context.items()))
        return f"{message.title}: {message.body} [{context}]"


__all__ = ["SMSChannel"]
