"""Adapter kanału powiadomień dla Telegrama."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict
from urllib import request

from bot_core.alerts.base import AlertChannel, AlertDeliveryError, AlertMessage
from bot_core.alerts.channels._http import HttpOpener, default_opener

@dataclass(slots=True)
class TelegramChannel(AlertChannel):
    """Publikuje alerty wykorzystując oficjalne API Telegram Bot."""

    bot_token: str
    chat_id: str
    parse_mode: str = "MarkdownV2"
    name: str = "telegram"
    timeout: float = 10.0
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("bot_core.alerts.telegram"))
    _opener: HttpOpener = field(default=default_opener, repr=False)
    _last_success: datetime | None = field(default=None, init=False, repr=False)
    _last_error: str | None = field(default=None, init=False, repr=False)

    def send(self, message: AlertMessage) -> None:
        payload = {
            "chat_id": self.chat_id,
            "text": self._format_message(message),
            "disable_web_page_preview": True,
        }
        if self.parse_mode:
            payload["parse_mode"] = self.parse_mode

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})

        try:
            with self._opener(req, timeout=self.timeout) as response:
                status = getattr(response, "status", response.getcode())
                data = response.read()
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            self.logger.exception("Błąd wysyłki do Telegrama")
            raise AlertDeliveryError(f"Błąd wysyłki do Telegrama: {exc}") from exc

        if status >= 400:
            body = data.decode("utf-8", errors="ignore") if data else ""
            self._last_error = body or str(status)
            self.logger.error("Telegram zwrócił błąd HTTP %s: %s", status, body)
            raise AlertDeliveryError(f"Telegram zwrócił błąd HTTP {status}: {body}")

        try:
            parsed: Dict[str, object] = json.loads(data) if data else {"ok": True}
        except json.JSONDecodeError as exc:  # pragma: no cover - powinno się udać dla poprawnych odpowiedzi
            self._last_error = str(exc)
            raise AlertDeliveryError("Niepoprawna odpowiedź Telegrama") from exc

        if not bool(parsed.get("ok", True)):
            description = str(parsed.get("description", "Nieznany błąd"))
            self._last_error = description
            self.logger.error("Telegram odrzucił wiadomość: %s", description)
            raise AlertDeliveryError(f"Telegram odrzucił wiadomość: {description}")

        self._last_success = datetime.now(tz=timezone.utc)
        self._last_error = None

    def health_check(self) -> Dict[str, str]:
        status = "ok" if self._last_error is None else "error"
        data: Dict[str, str] = {"status": status}
        if self._last_success:
            data["last_success"] = self._last_success.isoformat()
        if self._last_error:
            data["last_error"] = self._last_error
        return data

    def _format_message(self, message: AlertMessage) -> str:
        header = f"*{message.title}*" if self.parse_mode == "MarkdownV2" else message.title
        context_lines = "\n".join(f"• {key}: {value}" for key, value in sorted(message.context.items()))
        parts = [header, message.body]
        if context_lines:
            parts.append(context_lines)
        parts.append(f"Kategoria: {message.category} | Poziom: {message.severity}")
        return "\n\n".join(part for part in parts if part)


__all__ = ["TelegramChannel"]

