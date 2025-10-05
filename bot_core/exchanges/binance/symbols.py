"""Mapowanie i normalizacja symboli dla adaptera Binance Spot."""
from __future__ import annotations

from typing import Iterable, Mapping

# Docelowy koszyk instrumentów obsługiwanych przez pierwszą wersję adaptera.
_CANONICAL_TO_EXCHANGE: Mapping[str, str] = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
    "BNB/USDT": "BNBUSDT",
    "XRP/USDT": "XRPUSDT",
    "ADA/USDT": "ADAUSDT",
    "LTC/USDT": "LTCUSDT",
    "MATIC/USDT": "MATICUSDT",
    "BTC/EUR": "BTCEUR",
    "ETH/EUR": "ETHEUR",
    "BTC/PLN": "BTCPLN",
    "ETH/PLN": "ETHPLN",
}

_EXCHANGE_TO_CANONICAL: Mapping[str, str] = {
    exchange: canonical for canonical, exchange in _CANONICAL_TO_EXCHANGE.items()
}


def _normalise_delimiters(symbol: str) -> str:
    """Ujednolica separatory w symbolu (/, -, _)."""

    return symbol.replace("-", "/").replace("_", "/")


def supported_symbols() -> tuple[str, ...]:
    """Zwraca krotkę wszystkich wspieranych symboli w notacji BASE/QUOTE."""

    return tuple(_CANONICAL_TO_EXCHANGE.keys())


def normalize_symbol(symbol: str | None) -> str | None:
    """Konwertuje symbol giełdy lub domenowy do notacji BASE/QUOTE."""

    if not symbol:
        return None

    cleaned = _normalise_delimiters(symbol.strip().upper())
    if not cleaned:
        return None

    if cleaned in _CANONICAL_TO_EXCHANGE:
        return cleaned

    canonical = _EXCHANGE_TO_CANONICAL.get(cleaned.replace("/", ""))
    if canonical:
        return canonical

    return None


def to_exchange_symbol(symbol: str | None) -> str | None:
    """Zwraca symbol w notacji Binance (np. BTCUSDT) lub ``None`` jeśli nieobsługiwany."""

    canonical = normalize_symbol(symbol)
    if canonical is None:
        return None
    return _CANONICAL_TO_EXCHANGE.get(canonical)


def is_supported(symbol: str | None) -> bool:
    """Sprawdza, czy symbol jest wspierany przez adapter."""

    return normalize_symbol(symbol) is not None


def filter_supported_exchange_symbols(raw_symbols: Iterable[str]) -> tuple[str, ...]:
    """Filtruje listę symboli zwróconą przez API Binance do wspieranej domeny."""

    canonical: list[str] = []
    seen: set[str] = set()
    for entry in raw_symbols:
        normalized = normalize_symbol(entry)
        if not normalized or normalized in seen:
            continue
        canonical.append(normalized)
        seen.add(normalized)
    return tuple(canonical)


__all__ = [
    "filter_supported_exchange_symbols",
    "is_supported",
    "normalize_symbol",
    "supported_symbols",
    "to_exchange_symbol",
]

