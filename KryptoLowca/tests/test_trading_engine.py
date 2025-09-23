# tests/test_trading_engine.py
# -*- coding: utf-8 -*-
"""
Unit tests for trading_engine.py.
"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.trading_engine import TradingEngine, TradingError
from trading_strategies import TradingParameters, EngineConfig, TradingStrategies

class MockExchange:
    async def fetch_balance(self):
        return {"USDT": 10000.0}

class MockAI:
    async def predict_series(self, symbol, df):
        return pd.Series(np.full(len(df), 10.0), index=df.index)
    ai_threshold_bps = 5.0

class MockRisk:
    def calculate_position_size(self, symbol, signal, market_data, portfolio):
        return 0.1

class MockDB:
    async def ensure_user(self, email):
        return 1
    async def log(self, user_id, level, msg, category="general", context=None):
        pass
    async def get_positions(self, user_id):
        return []

@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr("core.trading_engine.ExchangeManager", MockExchange)
    monkeypatch.setattr("core.trading_engine.AIManager", MockAI)
    monkeypatch.setattr("core.trading_engine.RiskManager", MockRisk)
    monkeypatch.setattr("core.trading_engine.DatabaseManager", MockDB)
    monkeypatch.setattr(asyncio, "create_task", lambda coro: asyncio.run(coro))
    engine = TradingEngine(db_manager=MockDB())
    asyncio.run(engine.configure(MockExchange(), MockAI(), MockRisk()))
    engine.set_parameters(TradingParameters(), EngineConfig())
    return engine

def test_execute_live_tick(engine):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-08-21", periods=252, freq="T"),
        "open": np.full(252, 100.0),
        "high": np.full(252, 101.0),
        "low": np.full(252, 99.0),
        "close": np.full(252, 100.5),
        "volume": np.full(252, 10.0)
    })
    preds = pd.Series(np.full(252, 10.0), index=df.index)
    plan = asyncio.run(engine.execute_live_tick("BTC/USDT", df, preds))
    assert plan is not None
    assert plan["symbol"] == "BTC/USDT"
    assert plan["side"] == "buy"
    assert plan["qty_hint"] == 0.1
    assert plan["price_ref"] == 100.5

def test_no_signal(engine, monkeypatch):
    monkeypatch.setattr(engine.strategies, "run_strategy", lambda *args: ({}, [], pd.Series()))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-08-21", periods=252, freq="T"),
        "open": np.full(252, 100.0),
        "high": np.full(252, 101.0),
        "low": np.full(252, 99.0),
        "close": np.full(252, 100.5),
        "volume": np.full(252, 10.0)
    })
    preds = pd.Series(np.full(252, 0.0), index=df.index)
    plan = asyncio.run(engine.execute_live_tick("BTC/USDT", df, preds))
    assert plan is None

def test_max_positions(engine, monkeypatch):
    monkeypatch.setattr(engine.db_manager, "get_positions", AsyncMock(return_value=[{}]*1))
    engine.tp = TradingParameters(max_position_size=1)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-08-21", periods=252, freq="T"),
        "open": np.full(252, 100.0),
        "high": np.full(252, 101.0),
        "low": np.full(252, 99.0),
        "close": np.full(252, 100.5),
        "volume": np.full(252, 10.0)
    })
    preds = pd.Series(np.full(252, 10.0), index=df.index)
    plan = asyncio.run(engine.execute_live_tick("BTC/USDT", df, preds))
    assert plan is None

def test_invalid_input(engine):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-08-21", periods=10, freq="T"),
        "open": np.full(10, 100.0),
        "high": np.full(10, 101.0),
        "low": np.full(10, 99.0),
        "close": np.full(10, 100.5),
        "volume": np.full(10, 10.0)
    })
    preds = pd.Series(np.full(10, 10.0), index=df.index)
    with pytest.raises(ValueError):
        asyncio.run(engine.execute_live_tick("", df, preds))
    with pytest.raises(ValueError):
        asyncio.run(engine.execute_live_tick("BTC/USDT", df, preds))