# tests/test_trading_engine.py
# -*- coding: utf-8 -*-
"""Unit tests for :mod:`core.trading_engine`."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.trading_engine import TradingEngine
from managers.exchange_core import Mode, OrderDTO, OrderSide, OrderStatus, OrderType
from trading_strategies import EngineConfig, TradingParameters


class MockExchange:
    def __init__(self) -> None:
        self.mode = Mode.PAPER
        self.created_orders = []

    async def fetch_balance(self):
        return {"USDT": 10_000.0, "free": {"USDT": 10_000.0}}

    def quantize_amount(self, symbol: str, amount: float) -> float:
        return round(float(amount), 6)

    def quantize_price(self, symbol: str, price: float) -> float:
        return round(float(price), 2)

    def min_notional(self, symbol: str) -> float:
        return 10.0

    def create_order(
        self,
        symbol: str,
        side: str,
        type: str,
        quantity: float,
        price: float | None = None,
        client_order_id: str | None = None,
    ) -> OrderDTO:
        self.created_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "type": type,
                "quantity": quantity,
                "price": price,
                "client_order_id": client_order_id,
            }
        )
        return OrderDTO(
            id=123,
            client_order_id=client_order_id,
            symbol=symbol,
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            type=OrderType.MARKET if type.upper() == "MARKET" else OrderType.LIMIT,
            quantity=quantity,
            price=price or 100.5,
            status=OrderStatus.FILLED,
            mode=self.mode,
        )


class MockAI:
    ai_threshold_bps = 5.0


class MockRisk:
    def calculate_position_size(self, symbol, signal, market_data, portfolio):
        return 0.1


class MockDB:
    def __init__(self) -> None:
        self.orders = []
        self.order_updates = []
        self.logs = []
        self.positions = []

    async def ensure_user(self, email: str) -> int:
        return 1

    async def log(self, user_id, level, message, category="general", context=None):
        self.logs.append((level, message, category, context))

    async def get_positions(self, user_id):
        return list(self.positions)

    async def record_order(self, order):
        self.orders.append(order)
        return len(self.orders)

    async def update_order_status(
        self,
        *,
        order_id=None,
        client_order_id=None,
        status=None,
        price=None,
        exchange_order_id=None,
        extra=None,
    ):
        self.order_updates.append(
            {
                "order_id": order_id,
                "client_order_id": client_order_id,
                "status": status,
                "price": price,
                "exchange_order_id": exchange_order_id,
                "extra": extra,
            }
        )


@pytest.fixture
def engine():
    exchange = MockExchange()
    db = MockDB()
    engine = TradingEngine(db_manager=db)

    async def _setup():
        await engine.configure(exchange, MockAI(), MockRisk())
        engine.set_parameters(TradingParameters(), EngineConfig())

    asyncio.run(_setup())
    return engine


def _sample_df(size: int = 252) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-08-21", periods=size, freq="T"),
            "open": np.full(size, 100.0),
            "high": np.full(size, 101.0),
            "low": np.full(size, 99.0),
            "close": np.full(size, 100.5),
            "volume": np.full(size, 10.0),
        }
    )


def test_execute_live_tick(engine):
    df = _sample_df()
    preds = pd.Series(np.full(len(df), 10.0), index=df.index)

    plan = asyncio.run(engine.execute_live_tick("BTC/USDT", df, preds))

    assert plan is not None
    assert plan["symbol"] == "BTC/USDT"
    assert plan["side"] == "buy"
    assert "execution" in plan
    assert plan["execution"]["status"] == "FILLED"
    assert engine.ex_mgr.created_orders  # type: ignore[attr-defined]
    assert engine.db_manager.orders  # type: ignore[attr-defined]


def test_no_signal(engine):
    df = _sample_df()
    preds = pd.Series(np.zeros(len(df)), index=df.index)

    plan = asyncio.run(engine.execute_live_tick("BTC/USDT", df, preds))

    assert plan is None


def test_max_positions(engine, monkeypatch):
    monkeypatch.setattr(engine.db_manager, "positions", [{"symbol": "BTC/USDT"}])
    monkeypatch.setattr(engine.db_manager, "get_positions", AsyncMock(return_value=[{"symbol": "BTC/USDT"}]))
    engine.tp = TradingParameters(max_position_size=1)

    df = _sample_df()
    preds = pd.Series(np.full(len(df), 10.0), index=df.index)

    plan = asyncio.run(engine.execute_live_tick("BTC/USDT", df, preds))

    assert plan is None


def test_invalid_input(engine):
    df_short = _sample_df(size=10)
    preds_short = pd.Series(np.full(len(df_short), 10.0), index=df_short.index)

    with pytest.raises(ValueError):
        asyncio.run(engine.execute_live_tick("", df_short, preds_short))
    with pytest.raises(ValueError):
        asyncio.run(engine.execute_live_tick("BTC/USDT", df_short, preds_short))
