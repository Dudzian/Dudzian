import math

import pytest

from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.futures_spread import FuturesSpreadSettings, FuturesSpreadStrategy
from bot_core.strategies.market_params import (
    DEFAULT_FUTURES_PARAMS,
    DEFAULT_SPOT_PARAMS,
    quantity_from_notional,
)


def _expected_quantity(notional: float, price: float, *, lot_size: float, fee_rate: float) -> float:
    raw_qty = max(0.0, notional * (1 - fee_rate)) / max(price, 1e-12)
    steps = max(1, math.floor(raw_qty / lot_size))
    return steps * lot_size


def test_quantity_from_notional_spot_respects_fee_and_lot_size() -> None:
    params = DEFAULT_SPOT_PARAMS
    qty = quantity_from_notional(1_000.0, 25_000.0, params=params)

    assert qty == pytest.approx(
        _expected_quantity(
            1_000.0, 25_000.0, lot_size=params.lot_size, fee_rate=params.taker_fee_rate
        )
    )
    assert qty >= params.lot_size


def test_quantity_from_notional_futures_supports_micro_lots() -> None:
    params = DEFAULT_FUTURES_PARAMS
    qty = quantity_from_notional(1.0, 20_000.0, params=params)

    assert qty == pytest.approx(
        _expected_quantity(1.0, 20_000.0, lot_size=params.lot_size, fee_rate=params.taker_fee_rate)
    )
    assert qty >= params.lot_size


def test_futures_spread_injects_market_params_into_signals() -> None:
    settings = FuturesSpreadSettings()
    strategy = FuturesSpreadStrategy(settings)
    snapshot = MarketSnapshot(
        symbol="BTCUSDT",
        timestamp=0,
        open=0.0,
        high=0.0,
        low=0.0,
        close=20_000.0,
        volume=0.0,
        indicators={
            "spread_zscore": settings.entry_z + 0.1,
            "basis": 0.0,
            "funding_rate": 0.0,
            "front_contract": "BTC-1",  # pragma: allowlist secret
            "back_contract": "BTC-2",
            "front_price": 20_000.0,
            "back_price": 20_100.0,
        },
    )

    signals = strategy.on_data(snapshot)

    assert signals, "strategia futures spread powinna wygenerować sygnał wejścia"
    payload = signals[0]
    assert payload.metadata["taker_fee_rate"] == settings.market.taker_fee_rate
    assert payload.metadata["lot_size"] == settings.market.lot_size
    assert payload.legs[0].quantity and payload.legs[0].quantity >= settings.market.lot_size
