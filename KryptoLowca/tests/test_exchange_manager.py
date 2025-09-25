"""Unit tests for :mod:`KryptoLowca.exchange_manager`."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

from typing import Any, Dict, List

import pytest

import ccxt  # type: ignore

from KryptoLowca.alerts import get_alert_dispatcher
from KryptoLowca.config_manager import ExchangeConfig
from KryptoLowca.exchange_manager import (
    AuthenticationError,
    ExchangeError,
    ExchangeManager,
    OrderResult,
)


@pytest.fixture()
async def exchange_manager(monkeypatch):
    class MockExchange:
        def __init__(self, config):
            self.config = config
            self.rateLimit = 50
            self.markets = {"BTC/USDT": {}, "ETH/USDT": {}}
            self.load_markets = AsyncMock(return_value=self.markets)
            self.fetch_ohlcv = AsyncMock(
                return_value=[
                    [1700000000000, 100.0, 101.0, 99.0, 100.5, 10.0],
                    [1700000060000, 100.5, 101.5, 99.5, 101.0, 11.0],
                ]
            )
            self.create_market_order = AsyncMock(
                return_value={
                    "id": "123",
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "amount": 0.1,
                    "price": 100.0,
                    "status": "filled",
                    "timestamp": "2025-08-21T18:00:00Z",
                }
            )
            self.fetch_balance = AsyncMock(return_value={"total": {"USDT": 1000.0, "BTC": 0.1}})
            self.fetch_open_orders = AsyncMock(
                return_value=[
                    {
                        "id": "124",
                        "symbol": "BTC/USDT",
                        "side": "sell",
                        "amount": 0.05,
                        "price": 101.0,
                        "status": "open",
                    }
                ]
            )
            self.cancel_order = AsyncMock(return_value=True)
            self.fetch_ticker = AsyncMock(return_value={"last": 100.0})
            self.close = AsyncMock()

    class MockDB:
        def __init__(self):
            self.metrics: List[Dict[str, Any]] = []
            self.rate_limits: List[Dict[str, Any]] = []
            self.logs: List[Dict[str, Any]] = []

        async def ensure_user(self, email):
            return 1

        async def log(self, user_id, level, msg, category="general", context=None):
            self.logs.append({
                "user_id": user_id,
                "level": level,
                "message": msg,
                "category": category,
                "context": context or {},
            })
            return 1

        async def log_performance_metric(self, payload):
            self.metrics.append(dict(payload))
            return len(self.metrics)

        async def log_rate_limit_snapshot(self, payload):
            self.rate_limits.append(dict(payload))
            return len(self.rate_limits)

    class MockSecurity:
        def load_encrypted_keys(self, pwd):
            return {
                "testnet": {"key": "test_key", "secret": "test_secret"},
                "live": {"key": "live_key", "secret": "live_secret"},
            }

    monkeypatch.setattr("ccxt.asyncio.binance", MockExchange)
    config = ExchangeConfig(
        exchange_name="binance",
        testnet=True,
        api_key="test_key",
        api_secret="test_secret",
        rate_limit_per_minute=120,
        rate_limit_window_seconds=1.0,
        rate_limit_alert_threshold=0.8,
        error_alert_threshold=3,
    )
    db_manager = MockDB()
    security_manager = MockSecurity()
    manager = await ExchangeManager.create(config, db_manager, security_manager)
    manager.set_metrics_log_interval(0.01)
    manager._retry_delay = 0.0
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
async def test_retry_on_failure(exchange_manager):
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
    config = ExchangeConfig(
        exchange_name="binance",
        testnet=True,
        api_key="wrong_key",
        api_secret="wrong_secret",
    )
    with pytest.raises(ExchangeError):
        await ExchangeManager.create(config)


@pytest.mark.asyncio
async def test_metrics_snapshot_logged(exchange_manager):
    await exchange_manager.fetch_balance()
    await asyncio.sleep(0.05)
    db = exchange_manager._db_manager
    assert db.metrics, "Performance metrics should be stored"
    assert any(entry["metric"] == "exchange_total_calls" for entry in db.metrics)
    assert db.rate_limits, "Rate limit snapshots should be stored"


@pytest.mark.asyncio
async def test_rate_limit_alert_triggers_snapshot(exchange_manager):
    dispatcher = get_alert_dispatcher()
    events: List[Any] = []

    def _listener(event):
        events.append(event)

    token = dispatcher.register(_listener, name="test-rate-limit")
    try:
        exchange_manager.configure_rate_limits(per_minute=1, window_seconds=1.0)
        await exchange_manager.fetch_balance()
        await asyncio.sleep(0.05)
        db = exchange_manager._db_manager
        assert db.rate_limits, "Rate limit snapshots should be captured"
        assert any(entry["context"]["limit_triggered"] for entry in db.rate_limits)
        assert events, "An alert should be emitted when rate limit threshold is reached"
    finally:
        dispatcher.unregister(token)


@pytest.mark.asyncio
async def test_api_metrics_tracking(exchange_manager):
    await exchange_manager.load_markets()
    await exchange_manager.fetch_balance()

    metrics = exchange_manager.get_api_metrics()
    assert metrics["total_calls"] >= 2
    assert "fetch_balance" in metrics["endpoints"]


@pytest.mark.asyncio
async def test_rate_limit_alert(exchange_manager, caplog):
    exchange_manager.reset_api_metrics()
    exchange_manager._max_calls_per_window = 2
    exchange_manager._rate_limit_window = 0.05
    exchange_manager._alert_usage_threshold = 0.5
    exchange_manager._min_interval = 0.0

    with caplog.at_level(logging.CRITICAL):
        await exchange_manager.fetch_balance()
        await exchange_manager.fetch_balance()

    assert any("[ALERT]" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_error_alert(exchange_manager, caplog):
    exchange_manager.reset_api_metrics()
    exchange_manager._error_alert_threshold = 2
    exchange_manager.exchange.fetch_balance = AsyncMock(side_effect=Exception("boom"))

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(Exception):
            await exchange_manager.fetch_balance()

    assert any("[ALERT]" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_multi_symbol_retry_and_logging():
    class FakeExchange:
        rateLimit = None

        def __init__(self) -> None:
            self.markets = {
                "BTC/USDT": {"limits": {"amount": {"step": 0.001}, "price": {"step": 0.1}}},
                "ETH/USDT": {"limits": {"amount": {"step": 0.001}, "price": {"step": 0.1}}},
            }
            self._fail_once = {"ETH/USDT": True}

        async def load_markets(self):
            return self.markets

        async def create_order(self, symbol, order_type, side, amount, price=None, params=None):
            if self._fail_once.get(symbol):
                self._fail_once[symbol] = False
                raise RuntimeError("temporary outage")
            return {
                "id": f"{symbol}-order",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price or 100.0,
                "status": "filled",
            }

        async def fetch_balance(self):
            return {"total": {"USDT": 1000.0}}

        async def fetch_ticker(self, symbol):
            return {"last": 100.0}

    class FakeDB:
        def __init__(self) -> None:
            self.entries: List[Dict[str, Any]] = []

        async def log(self, user_id, level, message, category="general", context=None):
            self.entries.append(
                {
                    "user_id": user_id,
                    "level": level,
                    "message": message,
                    "category": category,
                    "context": context or {},
                }
            )

    fake_exchange = FakeExchange()
    manager = ExchangeManager(fake_exchange)
    manager._db_manager = FakeDB()
    manager._user_id = 42
    manager.set_retry_policy(attempts=2, delay=0.0)
    manager.configure_rate_limits(
        per_minute=None,
        window_seconds=1.0,
        buckets=[{"name": "burst", "capacity": 2, "window_seconds": 0.1}],
    )
    manager._alert_usage_threshold = 0.5
    manager._alert_cooldown_seconds = 0.0
    alerts: List[Dict[str, Any]] = []

    def _alert_handler(message: str, context: Dict[str, Any]) -> None:
        alerts.append({"message": message, "context": context})

    manager.register_alert_handler(_alert_handler)

    await manager.load_markets()

    # pierwsze zlecenie – sukces
    result1 = await manager.create_order("BTC/USDT", "buy", "market", 0.01)
    assert result1.status == "filled"

    # drugie zlecenie – pierwsze podejście rzuca wyjątek, drugie powinno się powieść
    result2 = await manager.create_order("ETH/USDT", "buy", "limit", 0.02, price=100.1)
    assert result2.status == "filled"

    # dodatkowe wywołanie aby przekroczyć próg limitu i wygenerować alert + log DB
    await manager.fetch_balance()
    await asyncio.sleep(0)  # pozwól logom z DB się wykonać

    metrics = manager.get_api_metrics()
    assert metrics["total_calls"] >= 3
    assert metrics["rate_limit_buckets"], "Powinny istnieć metryki kubełków limitów"
    assert alerts, "Alert o zbliżającym się limicie powinien zostać wygenerowany"
    assert any(entry["category"] == "exchange" for entry in manager._db_manager.entries)
