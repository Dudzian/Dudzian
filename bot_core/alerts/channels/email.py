"""Adapter kanału e-mail bazujący na SMTP."""
from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Callable, Sequence

from bot_core.alerts.base import AlertChannel, AlertDeliveryError, AlertMessage


@dataclass(slots=True)
class EmailChannel(AlertChannel):
    """Wysyła powiadomienia korzystając z serwera SMTP."""

    host: str
    port: int
    from_address: str
    recipients: Sequence[str]
    username: str | None = None
    password: str | None = None
    use_tls: bool = True
    timeout: float = 10.0
    name: str = "email"
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("bot_core.alerts.email"))
    _smtp_factory: Callable[[str, int, float], smtplib.SMTP] = field(default=smtplib.SMTP, repr=False)
    _last_error: str | None = field(default=None, init=False, repr=False)
    _last_success: str | None = field(default=None, init=False, repr=False)

    def send(self, message: AlertMessage) -> None:
        email = self._build_email(message)
        try:
            with self._smtp_factory(self.host, self.port, timeout=self.timeout) as client:
                client.ehlo()
                if self.use_tls:
                    client.starttls()
                    client.ehlo()
                if self.username and self.password:
                    client.login(self.username, self.password)
                client.send_message(email)
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            self.logger.exception("Błąd wysyłki e-maila")
            raise AlertDeliveryError(f"Nie udało się wysłać e-maila: {exc}") from exc

        self._last_error = None
        self._last_success = message.timestamp.isoformat()

    def health_check(self) -> dict[str, str]:
        status = "ok" if self._last_error is None else "error"
        data: dict[str, str] = {"status": status}
        if self._last_success:
            data["last_success"] = self._last_success
        if self._last_error:
            data["last_error"] = self._last_error
        return data

    def _build_email(self, message: AlertMessage) -> EmailMessage:
        email = EmailMessage()
        email["From"] = self.from_address
        email["To"] = ", ".join(self.recipients)
        email["Subject"] = f"[{message.severity.upper()}] {message.title}"
        body_lines = [
            message.body,
            "",
            "Kontekst:",
            *[f"- {key}: {value}" for key, value in sorted(message.context.items())],
            "",
            f"Kategoria: {message.category}",
            f"Znacznik czasu: {message.timestamp.isoformat()}",
        ]
        email.set_content("\n".join(body_lines))
        return email


__all__ = ["EmailChannel"]

