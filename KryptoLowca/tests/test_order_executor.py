from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot_core.exchanges import Mode, OrderDTO, OrderSide, OrderStatus, OrderType

from KryptoLowca.core.order_executor import OrderExecutor  # type: ignore


class DummyExchange:
    def __init__(self) -> None:
        self.mode = Mode.PAPER
        self.created_orders: list[dict[str, object]] = []
        self._attempts = 0
        self._fail_once = False

    def enable_fail_once(self) -> None:
        self._fail_once = True

    async def fetch_balance(self) -> dict[str, float]:
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
        self._attempts += 1
        if self._fail_once and self._attempts == 1:
            raise ConnectionError("temporary network glitch")

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
            price=price or 100.0,
            status=OrderStatus.FILLED,
            mode=self.mode,
        )


class DummyDB:
    def __init__(self) -> None:
        self.orders: list[dict[str, object]] = []
        self.order_updates: list[dict[str, object]] = []
        self.logs: list[tuple[str, str]] = []

    async def log(self, user_id, level, message, category="general", context=None):
        self.logs.append((level, message))

    async def record_order(self, payload: dict[str, object]) -> int:
        self.orders.append(payload)
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
    ) -> None:
        self.order_updates.append(
            {
                "order_id": order_id,
                "client_order_id": client_order_id,
                "status": status,
                "price": price,
                "exchange_order_id": exchange_order_id,
                "extra": extra or {},
            }
        )


def test_execute_plan_respects_fraction_and_records_order():
    exchange = DummyExchange()
    db = DummyDB()
    executor = OrderExecutor(exchange, db, max_fraction=0.4)
    executor.set_user(42)

    plan = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "order_type": "market",
        "qty_hint": 0.35,
        "price_ref": 100.0,
        "capital": 1_000.0,
        "portfolio": {"positions": []},
        "allow_short": False,
        "max_fraction": 0.25,
        "risk": {"recommended_size": 0.35},
    }

    result = asyncio.run(executor.execute_plan(plan))

    assert result.status == "FILLED"
    expected_fraction = plan["max_fraction"] / (1.0 + executor.fee_buffer)
    assert plan["applied_fraction"] == pytest.approx(expected_fraction, rel=1e-6)
    assert db.orders, "powinien zostać zapisany rekord zamówienia"
    extra = db.orders[0]["extra"]
    assert extra["applied_fraction"] == pytest.approx(expected_fraction, rel=1e-6)
    assert extra["max_fraction"] == 0.25
    assert exchange.created_orders, "zamówienie powinno trafić do giełdy"


def test_execute_plan_retries_on_transient_error():
    exchange = DummyExchange()
    exchange.enable_fail_once()
    db = DummyDB()
    executor = OrderExecutor(exchange, db, max_fraction=0.5, max_retries=1, retry_delay=0.0)

    plan = {
        "symbol": "ETH/USDT",
        "side": "buy",
        "order_type": "market",
        "qty_hint": 0.2,
        "price_ref": 50.0,
        "capital": 500.0,
        "portfolio": {"positions": []},
        "allow_short": False,
        "max_fraction": 0.3,
        "risk": {"recommended_size": 0.2},
    }

    result = asyncio.run(executor.execute_plan(plan))

    assert result.status == "FILLED"
    assert exchange._attempts == 2
    assert db.order_updates[-1]["status"] == "FILLED"


def test_execute_plan_fails_when_capital_below_min_notional():
    exchange = DummyExchange()
    db = DummyDB()
    executor = OrderExecutor(exchange, db, max_fraction=0.5)

    plan = {
        "symbol": "SOL/USDT",
        "side": "buy",
        "order_type": "market",
        "qty_hint": 0.5,
        "price_ref": 20.0,
        "capital": 5.0,
        "portfolio": {"positions": []},
        "allow_short": False,
        "max_fraction": 0.5,
        "risk": {"recommended_size": 0.5},
    }

    result = asyncio.run(executor.execute_plan(plan))

    assert result.status == "FAILED"
    assert "capital" in (result.error or "").lower()
    assert not db.orders, "zamówienie nie powinno zostać zapisane przy błędzie przygotowania"
