from __future__ import annotations

import math
from dataclasses import replace

from bot_core.backtest.trend_following import (
    BacktestConfig,
    EntryParams,
    ExitParams,
    StrategyParams,
    TrendBacktestEngine,
)


class StubExchange:
    def __init__(self, candles, *, min_notional=0.0, precision=3):
        self._candles = candles
        self._min_notional = min_notional
        self._precision = precision
        self.loaded = False

    def load_markets(self) -> None:
        self.loaded = True

    def fetch_ohlcv(self, symbol: str, timeframe: str, *, limit: int = 3000):
        return self._candles[:limit]

    def quantize_amount(self, symbol: str, amount: float) -> float:
        if amount <= 0:
            return 0.0
        factor = 10 ** self._precision
        return math.floor(amount * factor) / factor

    def min_notional(self, symbol: str) -> float | None:
        return self._min_notional or None


def _make_trend_bars(count: int = 120, *, start_price: float = 100.0):
    base_ts = 1_600_000_000_000
    candles = []
    price = start_price
    for idx in range(count):
        open_price = price
        if idx < 20:
            close_price = price - 0.5
        else:
            close_price = price + 1.0
        high_price = max(open_price, close_price) + 0.2
        low_price = min(open_price, close_price) - 0.2
        volume = 100 + idx
        candles.append([base_ts + idx * 60_000, open_price, high_price, low_price, close_price, volume])
        price = close_price
    return candles


def test_trend_engine_executes_trade_with_take_profits():
    symbol = "BTC/USDT"
    candles = _make_trend_bars()
    exchange = StubExchange(candles)
    engine = TrendBacktestEngine(exchange)

    strategy = StrategyParams(
        timeframe="1m",
        ema_fast=3,
        ema_slow=5,
        atr_len=3,
        min_atr_pct=0.1,
        execution="next_open",
    )
    entry = EntryParams(capital_usdt=1_000.0, risk_pct=0.1, k_sl_atr=1.0, fee_rate=0.0)
    exit_params = ExitParams(
        k_tp1_atr=0.5,
        k_tp2_atr=1.0,
        k_tp3_atr=1.5,
        p1=0.5,
        p2=0.3,
        p3=0.2,
        move_sl_to_be_after_tp1=True,
        trail_activate_pct=0.02,
        trail_dist_pct=0.01,
    )
    cfg = BacktestConfig(symbols=[symbol], strategy=strategy, entry=entry, exit=exit_params, start_index_pad=8)

    trades, summary = engine.run_symbol(symbol, cfg)

    assert exchange.loaded is True
    assert summary == {"symbol": symbol, "trades": 1}
    assert len(trades) == 1
    trade = trades[0]
    tags = {fill.tag for fill in trade.fills}
    assert {"ENTRY", "TP1", "TP2", "TP3"}.issubset(tags)
    assert trade.pnl_usdt is not None and trade.pnl_usdt > 0
    assert trade.exit_ts is not None


def test_trend_engine_returns_note_when_not_enough_data():
    symbol = "ETH/USDT"
    candles = _make_trend_bars(count=20)
    exchange = StubExchange(candles)
    engine = TrendBacktestEngine(exchange)
    cfg = BacktestConfig(symbols=[symbol])

    trades, summary = engine.run_symbol(symbol, cfg)

    assert trades == []
    assert summary["symbol"] == symbol
    assert "Za mało" in summary.get("note", "")


def test_risk_sizing_honours_min_notional():
    symbol = "SOL/USDT"
    candles = _make_trend_bars()
    exchange = StubExchange(candles, min_notional=100.0, precision=2)
    engine = TrendBacktestEngine(exchange)

    cfg = BacktestConfig(symbols=[symbol])
    cfg = replace(cfg, entry=EntryParams(capital_usdt=100.0, risk_pct=0.01, k_sl_atr=1.5, fee_rate=0.0))

    qty, sl_price, sl_pct = engine._risk_size_long(symbol, price=10.0, atr=0.2, cfg=cfg)

    assert qty >= 10.0  # aby spełnić minNotional 100 przy cenie 10
    assert sl_price < 10.0
    assert sl_pct >= 0.03
