"""Pomocnicze funkcje kontroli dostępu do modułu AI Signals."""

from __future__ import annotations

import logging

from bot_core.security.guards import get_capability_guard

_LOGGER = logging.getLogger(__name__)

_AI_SIGNALS_MODULE = "ai_signals"


def ensure_ai_signals_enabled(feature: str | None = None) -> None:
    """Wymusza posiadanie modułu AI Signals dla wskazanej funkcji."""

    guard = get_capability_guard()
    if guard is None:
        _LOGGER.debug(
            "Pomijam weryfikację modułu AI Signals dla %s – strażnik nie został zainstalowany.",
            feature or "funkcji AI",
        )
        return

    if feature:
        message = (
            f"Funkcja '{feature}' wymaga modułu AI Signals. Skontaktuj się z opiekunem licencji."
        )
    else:
        message = "Moduł AI Signals jest wymagany do korzystania z tej funkcji."
    guard.require_module(_AI_SIGNALS_MODULE, message=message)


__all__ = ["ensure_ai_signals_enabled"]
