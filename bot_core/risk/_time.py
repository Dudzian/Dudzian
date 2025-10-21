"""Wspólne funkcje czasowe dla modułów ryzyka."""

from __future__ import annotations

from datetime import datetime, timezone


def now_utc() -> datetime:
    """Zwraca bieżący czas w strefie UTC."""

    return datetime.now(timezone.utc)


__all__ = ["now_utc"]
