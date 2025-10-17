"""Mapowanie symboli dla adaptera nowa_gielda."""
from __future__ import annotations

from typing import Iterable

_INTERNAL_TO_EXCHANGE = {
    "BTC_USDT": "BTC-USDT",
    "ETH_USDT": "ETH-USDT",
    "SOL_USDT": "SOL-USDT",
    "ADA_USDT": "ADA-USDT",
}

_EXCHANGE_TO_INTERNAL = {value: key for key, value in _INTERNAL_TO_EXCHANGE.items()}


def to_exchange_symbol(internal_symbol: str) -> str:
    """Zwraca symbol giełdowy dla wewnętrznego oznaczenia."""
    try:
        return _INTERNAL_TO_EXCHANGE[internal_symbol.upper()]
    except KeyError as exc:  # pragma: no cover - defensywnie
        raise KeyError(f"Nieobsługiwany symbol wewnętrzny: {internal_symbol}") from exc


def to_internal_symbol(exchange_symbol: str) -> str:
    """Zwraca wewnętrzny symbol na podstawie oznaczenia giełdowego."""
    normalized = exchange_symbol.upper().replace("/", "-")
    try:
        return _EXCHANGE_TO_INTERNAL[normalized]
    except KeyError as exc:  # pragma: no cover - defensywnie
        raise KeyError(f"Nieobsługiwany symbol giełdowy: {exchange_symbol}") from exc


def supported_internal_symbols() -> Iterable[str]:
    """Zwraca iterowalny zbiór wewnętrznych symboli obsługiwanych przez adapter."""
    return tuple(_INTERNAL_TO_EXCHANGE.keys())


def supported_exchange_symbols() -> Iterable[str]:
    """Zwraca iterowalny zbiór symboli giełdowych obsługiwanych przez adapter."""
    return tuple(_EXCHANGE_TO_INTERNAL.keys())


__all__ = [
    "supported_exchange_symbols",
    "supported_internal_symbols",
    "to_exchange_symbol",
    "to_internal_symbol",
]
