"""Dostawca fingerprintu sprzętowego wykorzystywanego przez licencje offline."""
from __future__ import annotations

import logging
from typing import Callable

from bot_core.security.fingerprint import get_local_fingerprint

LOGGER = logging.getLogger(__name__)


class HwIdProviderError(RuntimeError):
    """Błąd podczas uzyskiwania lokalnego fingerprintu."""


class HwIdProvider:
    """Zapewnia zgodny z ``bot_core.security.fingerprint`` odcisk sprzętu."""

    def __init__(self, *, fingerprint_reader: Callable[[], str] | None = None) -> None:
        self._fingerprint_reader = fingerprint_reader or get_local_fingerprint

    def read(self) -> str:
        """Zwraca oczyszczony fingerprint lub zgłasza błąd, jeśli nie można go ustalić."""

        try:
            value = self._fingerprint_reader()
        except Exception as exc:  # pragma: no cover - defensywne logowanie
            LOGGER.debug("HwIdProvider: wyjątek podczas odczytu fingerprintu", exc_info=True)
            raise HwIdProviderError("Nie udało się pobrać fingerprintu urządzenia.") from exc

        if value is None:
            raise HwIdProviderError("Funkcja fingerprintu zwróciła pustą wartość.")

        hwid = str(value).strip()
        if not hwid:
            raise HwIdProviderError("Fingerprint urządzenia jest pusty.")

        return hwid


__all__ = ["HwIdProvider", "HwIdProviderError"]
