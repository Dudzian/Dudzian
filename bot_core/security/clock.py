"""Monotoniczny zegar dla walidacji licencji offline."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Callable

LOGGER = logging.getLogger(__name__)


class ClockService:
    """Przechowuje monotoniczną datę efektywną dla licencji offline."""

    DEFAULT_STATE_PATH = Path("var/security/license_state.json")

    def __init__(
        self,
        *,
        state_path: str | Path | None = None,
        today_provider: Callable[[], date] | None = None,
    ) -> None:
        self._state_path = Path(state_path) if state_path else self.DEFAULT_STATE_PATH
        self._today = today_provider or date.today

    def effective_today(self, license_id: str | None = None) -> date:
        """Zwraca monotoniczną datę, nie cofając się względem poprzednich uruchomień."""

        today = self._today()
        last_seen = self._load_last_seen(license_id)
        if last_seen and last_seen > today:
            LOGGER.debug("Zachowano monotoniczność daty (last_seen=%s)", last_seen)
            return last_seen
        return today

    def remember(self, license_id: str | None, value: date) -> None:
        """Utrwala datę efektywną dla danej licencji."""

        path = self._state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        document: dict[str, object] = {"last_seen_date": value.isoformat()}
        if license_id:
            document["license_id"] = license_id
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset(self) -> None:
        """Czyści zapamiętany stan (przydatne w testach)."""

        try:
            self._state_path.unlink()
        except FileNotFoundError:
            return

    def _load_last_seen(self, license_id: str | None) -> date | None:
        path = self._state_path
        if not path.exists():
            return None
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Plik stanu licencji ma niepoprawny format – ignoruję.")
            return None
        if license_id and document.get("license_id") not in (None, license_id):
            return None
        date_str = document.get("last_seen_date")
        if not isinstance(date_str, str):
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return None


__all__ = ["ClockService"]
