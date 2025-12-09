"""Wspólne helpery parametrów rynku dla strategii spot/futures.

Moduł udostępnia spójny opis opłat oraz wielkości lotów, tak aby strategie
nie przechowywały własnych, rozproszonych stałych. Parametry są wykorzystywane
do wyliczania wielkości zleceń oraz raportowania kosztów transakcyjnych.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MarketParams:
    """Parametry egzekucji charakterystyczne dla danego rynku."""

    market: str
    taker_fee_rate: float
    maker_fee_rate: float
    lot_size: float


DEFAULT_SPOT_PARAMS = MarketParams(
    market="spot",
    taker_fee_rate=0.0004,  # 4 bps
    maker_fee_rate=0.0002,  # 2 bps
    lot_size=0.001,
)

DEFAULT_FUTURES_PARAMS = MarketParams(
    market="futures",
    taker_fee_rate=0.0005,  # 5 bps
    maker_fee_rate=0.0002,  # 2 bps
    lot_size=0.000001,
)


def _apply_fee(notional: float, fee_rate: float) -> float:
    notional = max(0.0, float(notional))
    fee_rate = max(0.0, float(fee_rate))
    return notional * (1.0 - fee_rate)


def quantity_from_notional(notional: float, price: float, *, params: MarketParams) -> float:
    """Przelicza nominal na ilość z uwzględnieniem fee i wielkości lota.

    Rounding jest wykonywany w dół do wielokrotności lota, a minimalna wielkość
    to dokładnie ``lot_size`` – dzięki temu strategie futures/spot mają
    wspólne zachowanie.
    """

    effective_notional = _apply_fee(abs(notional), params.taker_fee_rate)
    if price <= 0:
        return params.lot_size

    raw_quantity = effective_notional / float(price)
    lot = max(params.lot_size, 0.0)
    if lot <= 0:
        return raw_quantity

    steps = max(1, math.floor(raw_quantity / lot))
    return steps * lot


__all__ = [
    "MarketParams",
    "DEFAULT_SPOT_PARAMS",
    "DEFAULT_FUTURES_PARAMS",
    "quantity_from_notional",
]
