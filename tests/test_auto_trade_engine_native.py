from __future__ import annotations

import time

from bot_core.events import EmitterAdapter, EventType
from bot_core.trading.auto_trade import AutoTradeConfig, AutoTradeEngine


def test_auto_trade_engine_generates_orders_and_signals() -> None:
    adapter = EmitterAdapter()
    orders: list[tuple[str, float]] = []
    signals: list[float] = []
    statuses: list[str] = []

    def _collect_signals(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            signals.append(float(evt.payload["direction"]))

    adapter.subscribe(EventType.SIGNAL, _collect_signals)

    adapter.subscribe(
        EventType.AUTOTRADE_STATUS,
        lambda evt: statuses.extend([ev.payload["status"] for ev in (evt if isinstance(evt, list) else [evt])]),
    )

    cfg = AutoTradeConfig(
        symbol="BTCUSDT",
        qty=0.5,
        regime_window=6,
        activation_threshold=0.0,
        breakout_window=3,
        mean_reversion_window=4,
        mean_reversion_z=0.5,
    )
    engine = AutoTradeEngine(adapter, lambda side, qty: orders.append((side, qty)), cfg)
    engine.apply_params({"fast": 2, "slow": 5})

    closes = [10, 9, 8, 7, 6, 7, 8, 9, 10, 9, 8, 7, 6]
    for px in closes:
        adapter.push_market_tick("BTCUSDT", price=px)
    time.sleep(0.2)

    assert orders == [("buy", 0.5), ("sell", 0.5)]
    assert any(sig > 0 for sig in signals)
    assert any(sig < 0 for sig in signals)
    assert "params_applied" in statuses
