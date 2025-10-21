"""Pomocnicze funkcje współdzielone przez moduły kosztowe TCO."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

_DECIMAL_QUANT = Decimal("0.000001")


def quantize_decimal(value: Decimal) -> Decimal:
    """Normalizuje wartości kosztów do stałej precyzji."""

    return value.quantize(_DECIMAL_QUANT, rounding=ROUND_HALF_UP)


__all__ = ["quantize_decimal"]
