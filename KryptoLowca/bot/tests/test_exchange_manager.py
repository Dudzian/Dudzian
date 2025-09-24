# test_exchange_manager.py
# -*- coding: utf-8 -*-
"""
Unit tests for exchange_manager.py.
"""
import pytest
import asyncio
import pandas as pd
from unittest.mock import AsyncMock, MagicMock
from exchange_manager import ExchangeManager, ExchangeError, AuthenticationError, OrderResult
from config_manager import ExchangeConfig
from database_manager import DatabaseManager

@pytest.fixture
async def exchange_manager(monkeypatch):
    class MockExchange:
        def __init__(self, config):
            self.config = config
            self.markets = {"BTC/USDT": {}, "ETH/USDT": {}}
            self.load_markets = AsyncMock(return_value=self.markets)
            self.fetch_ohlcv = AsyncMock(return_value=[
                [1700000000000, 100.0, 101.0, 99.0, 100.5, 10.0],
                [1700000060000, 100.5, 101.5, 99.5, 101.0, 11.0]
            ])
            self.create_market_order = AsyncMock(return_value={"id": "123", "symbol": "BTC/USDT", "side": "buy", "amount": 0.1, "price": 100.0, "status": "filled", "timestamp": "2025-08-21T18:00:00Z"})
            self.fetch_balance = AsyncMock(return_value={"total": {"USDT": 1000.0, "BTC": 0.1}})
            self.fetch_open_orders = AsyncMock(return_value=[{"id": "124", "symbol": "BTC/USDT", "side": "sell", "amount": 0.05, "price": 101.0, "status": "open"}])
            self.cancel_order = AsyncMock(return_value=True)
            self.close = AsyncMock()

    class MockDB:
        async def ensure_user(self, email):
            return 1
        async def log(self, user_id, level, msg, category="general", context=None):
            pass
        async def insert_trade(self, user_id, trade):
            pass

    class MockSecurity:
        def load_encrypted_keys(self, pwd):
            return {"testnet": {"key": "test_key", "secret": "test_secret"}, "live": {"key": "live_key", "secret": "live_secret"}}

    monkeypatch.setattr("ccxt.asyncio.binance", MockExchange)
    config = ExchangeConfig(exchange_name="binance", testnet=True, api_key="test_key", api_secret="test_secret")
    db_manager = MockDB()
    security_manager = MockSecurity()
    manager = await ExchangeManager.create(config, db_manager, security_manager)
    yield manager
    await manager.close()

@pytest.mark.asyncio
async def test_initialize(exchange_manager):
    assert exchange_manager.exchange is not None
    assert exchange_manager.exchange.config["apiKey"] == "test_key"
    assert exchange_manager._user_id == 1

@pytest.mark.asyncio
async def test_load_markets(exchange_manager):
    markets = await exchange_manager.load_markets()
    assert len(markets) == 2
    assert "BTC/USDT" in markets
    exchange_manager.exchange.load_markets.assert_called_once()

@pytest.mark.asyncio
async def test_fetch_ohlcv(exchange_manager):
    data = await exchange_manager.fetch_ohlcv("BTC/USDT", "1m", 2)
    assert len(data) == 2
    assert data[0] == [1700000000000, 100.0, 101.0, 99.0, 100.5, 10.0]
    exchange_manager.exchange.fetch_ohlcv.assert_called_with("BTC/USDT", "1m", limit=2)

@pytest.mark.asyncio
async def test_place_market_order(exchange_manager):
    order = await exchange_manager.place_market_order("BTC/USDT", "buy", 0.1)
    assert isinstance(order, OrderResult)
    assert order.id == "123"
    assert order.symbol == "BTC/USDT"
    assert order.side == "buy"
    assert order.qty == 0.1
    exchange_manager.exchange.create_market_order.assert_called_with("BTC/USDT", "buy", 0.1)

@pytest.mark.asyncio
async def test_fetch_balance(exchange_manager):
    balance = await exchange_manager.fetch_balance()
    assert balance == {"USDT": 1000.0, "BTC": 0.1}
    exchange_manager.exchange.fetch_balance.assert_called_once()

@pytest.mark.asyncio
async def test_fetch_open_orders(exchange_manager):
    orders = await exchange_manager.fetch_open_orders("BTC/USDT")
    assert len(orders) == 1
    assert orders[0].id == "124"
    assert orders[0].symbol == "BTC/USDT"
    exchange_manager.exchange.fetch_open_orders.assert_called_with("BTC/USDT")

@pytest.mark.asyncio
async def test_cancel_order(exchange_manager):
    result = await exchange_manager.cancel_order("124", "BTC/USDT")
    assert result is True
    exchange_manager.exchange.cancel_order.assert_called_with("124", "BTC/USDT")

@pytest.mark.asyncio
async def test_invalid_input(exchange_manager):
    with pytest.raises(ValueError):
        await exchange_manager.fetch_ohlcv("", "1m", 100)
    with pytest.raises(ValueError):
        await exchange_manager.place_market_order("BTC/USDT", "invalid", 0.1)
    with pytest.raises(ValueError):
        await exchange_manager.place_market_order("BTC/USDT", "buy", -0.1)
    with pytest.raises(ValueError):
        await exchange_manager.cancel_order("", "BTC/USDT")

@pytest.mark.asyncio
async def test_retry_on_failure(exchange_manager, monkeypatch):
    call_count = 0
    def failing_fetch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise Exception("Temporary failure")
        return [{"id": "125", "symbol": "BTC/USDT", "side": "sell", "amount": 0.05}]
    exchange_manager.exchange.fetch_open_orders = AsyncMock(side_effect=failing_fetch)
    orders = await exchange_manager.fetch_open_orders("BTC/USDT")
    assert len(orders) == 1
    assert call_count == 2

@pytest.mark.asyncio
async def test_authentication_error(monkeypatch):
    class MockExchange:
        def __init__(self, config):
            raise ccxt.AuthenticationError("Invalid credentials")
    monkeypatch.setattr("ccxt.asyncio.binance", MockExchange)
    config = ExchangeConfig(exchange_name="binance", testnet=True, api_key="wrong_key", api_secret="wrong_secret")
    with pytest.raises(ExchangeError):
        await ExchangeManager.create(config)
