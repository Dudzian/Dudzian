"""Mechanizmy tłumienia nadmiarowych alertów."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping

from bot_core.alerts.base import AlertMessage


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_mapping(values: Mapping[str, str]) -> str:
    if not values:
        return ""
    return ",".join(f"{key}={values[key]}" for key in sorted(values))


@dataclass(slots=True)
class AlertThrottle:
    """Kontroluje częstotliwość wysyłki powtarzających się alertów."""

    window: timedelta
    clock: Callable[[], datetime] = _utc_now
    exclude_severities: frozenset[str] = frozenset({"critical"})
    exclude_categories: frozenset[str] = frozenset()
    max_entries: int = 2048
    _last_sent: dict[str, datetime] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window.total_seconds() <= 0:
            raise ValueError("Okno throttlingu musi być dodatnie.")
        self.exclude_severities = frozenset(value.lower() for value in self.exclude_severities)
        self.exclude_categories = frozenset(value.lower() for value in self.exclude_categories)
        if self.max_entries <= 0:
            raise ValueError("max_entries musi być większe od zera")

    def allow(self, message: AlertMessage) -> bool:
        """Sprawdza, czy alert może zostać wysłany w tym momencie."""

        if self._is_exempt(message):
            return True

        key = self._build_key(message)
        now = self.clock()
        last_sent = self._last_sent.get(key)
        if last_sent is None:
            return True
        return (now - last_sent) >= self.window

    def record(self, message: AlertMessage) -> None:
        """Zapisuje informację o wysłaniu alertu (po udanej wysyłce)."""

        if self._is_exempt(message):
            return

        key = self._build_key(message)
        now = self.clock()
        self._last_sent[key] = now
        self._prune(now)

    def remaining_seconds(self, message: AlertMessage) -> float:
        """Zwraca liczbę sekund pozostałych do kolejnej wysyłki danego alertu."""

        if self._is_exempt(message):
            return 0.0
        key = self._build_key(message)
        last_sent = self._last_sent.get(key)
        if last_sent is None:
            return 0.0
        elapsed = (self.clock() - last_sent).total_seconds()
        remaining = self.window.total_seconds() - elapsed
        return max(0.0, remaining)

    def reset(self) -> None:
        """Czyści historię throttlingu (przydatne w testach lub po zmianie konfiguracji)."""

        self._last_sent.clear()

    def _is_exempt(self, message: AlertMessage) -> bool:
        severity = message.severity.lower()
        category = message.category.lower()
        return severity in self.exclude_severities or category in self.exclude_categories

    def _build_key(self, message: AlertMessage) -> str:
        context = _normalize_mapping({str(k): str(v) for k, v in message.context.items()})
        body = " ".join(message.body.split())
        title = " ".join(message.title.split())
        return "|".join(
            (
                message.category.lower(),
                message.severity.lower(),
                title,
                body,
                context,
            )
        )

    def _prune(self, now: datetime) -> None:
        if len(self._last_sent) <= self.max_entries:
            threshold = now - self.window
            to_remove = [key for key, timestamp in self._last_sent.items() if timestamp < threshold]
            for key in to_remove:
                self._last_sent.pop(key, None)
            return

        # Jeśli liczba wpisów przekracza max_entries, zachowaj najnowsze.
        sorted_items = sorted(self._last_sent.items(), key=lambda item: item[1], reverse=True)
        self._last_sent = {key: ts for key, ts in sorted_items[: self.max_entries]}


__all__ = ["AlertThrottle"]
