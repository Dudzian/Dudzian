# tests/test_trading_engine.py
# -*- coding: utf-8 -*-
"""Testy jednostkowe dla modułu core.trading_engine."""

import asyncio
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core.trading_engine import TradingEngine, TradingError
from trading_strategies import EngineConfig, TradingParameters


class DummyExchange:
    async def fetch_balance(self):
        # Symuluje strukturę jak w CCXT
        return {"total": {"USDT": 10_000.0}}


class DummyAI:
    ai_threshold_bps = 5.0


class DummyRisk:
    def calculate_position_size(self, symbol, signal, market_data, portfolio):
        # Zwracamy strukturę zgodną z RiskManagement.PositionSizing
        return {"recommended_size": 0.1, "risk_adjusted_size": 0.08}


class DummyDB:
    def __init__(self):
        self.logged_messages = []
        self.positions = []

    async def ensure_user(self, email):
        return 1

    async def log(self, user_id, level, msg, category="general", context=None):
        self.logged_messages.append((user_id, level, category, msg))

    async def get_positions(self, user_id):
        return list(self.positions)


@pytest.fixture
def engine():
    db = DummyDB()
    engine = TradingEngine(db_manager=db)
    asyncio.run(engine.configure(DummyExchange(), DummyAI(), DummyRisk()))
    params = TradingParameters()
    config = EngineConfig(min_data_points=20, max_position_size=5, enable_shorting=True)
    engine.set_parameters(params, config)
    return engine


def _build_df(rows: int = 30, price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="T"),
            "open": np.full(rows, price),
            "high": np.full(rows, price * 1.01),
            "low": np.full(rows, price * 0.99),
            "close": np.full(rows, price),
            "volume": np.full(rows, 10.0),
        }
    )


def test_execute_live_tick_generates_plan(engine):
    df = _build_df()
    preds = pd.Series(np.full(len(df), 10.0), index=df.index)

    plan = asyncio.run(engine.execute_live_tick("BTC/USDT", df, preds))

    assert plan is not None
    assert plan["symbol"] == "BTC/USDT"
    assert plan["side"] == "buy"
    # 10% kapitału (10 000) przy cenie 100 -> 10 jednostek
    assert plan["qty_hint"] == pytest.approx(10.0, rel=1e-3)
    assert plan["price_ref"] == pytest.approx(100.0)
    assert plan["strength"] >= 1.0
    assert plan["stop_loss"] < plan["price_ref"]
    assert plan["take_profit"] > plan["price_ref"]


def test_execute_live_tick_no_signal(engine):
    df = _build_df()
    preds = pd.Series(np.full(len(df), 1.0), index=df.index)  # poniżej progu

    plan = asyncio.run(engine.execute_live_tick("BTC/USDT", df, preds))

    assert plan is None


def test_execute_live_tick_max_positions(engine):
    # Przygotuj konfigurację z maksymalnie 1 pozycją
    engine.set_parameters(engine.tp, replace(engine.ec, max_position_size=1))
    engine.db_manager.positions = [{}]

    df = _build_df()
    preds = pd.Series(np.full(len(df), 10.0), index=df.index)

    plan = asyncio.run(engine.execute_live_tick("BTC/USDT", df, preds))

    assert plan is None


def test_execute_live_tick_shorting_disabled(engine):
    engine.set_parameters(engine.tp, replace(engine.ec, enable_shorting=False))

    df = _build_df()
    preds = pd.Series(np.full(len(df), -10.0), index=df.index)

    plan = asyncio.run(engine.execute_live_tick("BTC/USDT", df, preds))

    assert plan is None


def test_execute_live_tick_invalid_input(engine):
    df = _build_df(rows=10)  # mniej niż min_data_points
    preds = pd.Series(np.full(len(df), 10.0), index=df.index)

    with pytest.raises(ValueError):
        asyncio.run(engine.execute_live_tick("BTC/USDT", df, preds))

    with pytest.raises(ValueError):
        asyncio.run(engine.execute_live_tick("", _build_df(), preds))


def test_execute_live_tick_missing_dependencies():
    engine = TradingEngine()
    df = _build_df()
    preds = pd.Series(np.full(len(df), 10.0), index=df.index)

    with pytest.raises(TradingError):
        asyncio.run(engine.execute_live_tick("BTC/USDT", df, preds))
