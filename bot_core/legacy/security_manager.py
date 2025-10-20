"""Warstwa kompatybilności dla legacy ``SecurityManager`` GUI."""

from __future__ import annotations

try:
    from KryptoLowca.security_manager import SecurityError, SecurityManager
except Exception as exc:  # pragma: no cover - brak zależności legacy
    raise ImportError(
        "Pakiet legacy SecurityManager nie jest dostępny – zainstaluj komponenty KryptoLowca."
    ) from exc

__all__ = ["SecurityManager", "SecurityError"]

