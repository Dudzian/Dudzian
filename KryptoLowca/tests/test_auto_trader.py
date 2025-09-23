import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_emitter_adapter import EventEmitter
from auto_trader import AutoTrader
from managers.exchange_core import (
    Mode,
    OrderDTO,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionDTO,
)


class StubExchange:
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 256):
        base = 100.0
        data = []
        for i in range(limit):
            ts = 1_700_000_000_000 + i * 60_000
            o = base + 0.1 * i
            h = o * 1.01
            l = o * 0.99
            c = o
            v = 10.0 + i
            data.append([ts, o, h, l, c, v])
        return data


class StubExecutionService:
    def __init__(self):
        self.mode = Mode.PAPER
        self.balance = 1_000.0
        self.orders: list[OrderDTO] = []
        self.positions: list[PositionDTO] = []

    def quote_market(self, symbol: str, side: str, amount=None, fallback_bps: float = 5.0, limit: int = 50):
        return 100.0, fallback_bps

    def calculate_quantity(self, symbol: str, notional: float, price: float) -> float:
        if price <= 0:
            return 0.0
        return round(max(0.0, notional / price), 6)

    def quantize_amount(self, symbol: str, amount: float) -> float:
        return round(max(0.0, amount), 6)

    def execute_market(self, symbol: str, side: str, quantity: float, *, client_order_id=None):
        order = OrderDTO(
            id=len(self.orders) + 1,
            client_order_id=client_order_id,
            symbol=symbol,
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            type=OrderType.MARKET,
            quantity=quantity,
            price=100.0,
            status=OrderStatus.FILLED,
            mode=self.mode,
        )
        self.orders.append(order)
        if order.side == OrderSide.BUY:
            self.balance -= quantity * 100.0
            self.positions = [
                PositionDTO(
                    symbol=symbol,
                    side="LONG",
                    quantity=quantity,
                    avg_price=100.0,
                    unrealized_pnl=0.0,
                    mode=self.mode,
                )
            ]
        else:
            self.balance += quantity * 100.0
            self.positions = []
        return order

    def fetch_balance(self):
        return {"free": {"USDT": self.balance}, "total": {"USDT": self.balance}}

    def list_positions(self, symbol: str | None = None):
        if symbol:
            return [p for p in self.positions if p.symbol == symbol]
        return list(self.positions)


class StubAI:
    def __init__(self, value: float):
        self.value = value
        self.ai_threshold_bps = 10.0

    def predict_series(self, df: pd.DataFrame, feature_cols: list[str]):
        return pd.Series([self.value] * len(df), index=df.index)


class StubRisk:
    def __init__(self, fraction: float = 0.1):
        self.fraction = fraction
        self.calls = []

    def calculate_position_size(self, symbol, signal, market_data, portfolio):
        self.calls.append((symbol, signal, portfolio))
        return self.fraction


class DummyGUI:
    def __init__(self, exec_service, exchange, ai, risk):
        self.exec_service = exec_service
        self.ex_mgr = exchange
        self.ai_mgr = ai
        self.risk_mgr = risk
        self.bridge_called = False

    def _bridge_execute_trade(self, *args, **kwargs):
        self.bridge_called = True


@pytest.fixture
def emitter():
    return EventEmitter()


def test_auto_trader_executes_buy_without_gui(emitter):
    exchange = StubExchange()
    exec_service = StubExecutionService()
    ai = StubAI(0.02)
    risk = StubRisk(0.2)

    trader = AutoTrader(
        emitter,
        gui=None,
        symbol_getter=lambda: "BTC/USDT",
        metrics_window=5,
        execution_service=exec_service,
        exchange_manager=exchange,
        ai_manager=ai,
        risk_manager=risk,
    )

    assert trader._auto_trade_once() is True
    assert len(exec_service.orders) == 1
    order = exec_service.orders[0]
    assert order.side == OrderSide.BUY
    assert risk.calls, "Risk manager should be invoked"


def test_auto_trader_executes_sell_when_position_present(emitter):
    exchange = StubExchange()
    exec_service = StubExecutionService()
    ai = StubAI(0.02)
    risk = StubRisk(0.1)

    trader = AutoTrader(
        emitter,
        gui=None,
        symbol_getter=lambda: "BTC/USDT",
        metrics_window=5,
        execution_service=exec_service,
        exchange_manager=exchange,
        ai_manager=ai,
        risk_manager=risk,
    )

    assert trader._auto_trade_once() is True  # open long
    ai.value = -0.02
    assert trader._auto_trade_once() is True  # close long
    assert len(exec_service.orders) == 2
    assert exec_service.orders[-1].side == OrderSide.SELL
    assert exec_service.positions == []


def test_auto_trader_uses_execution_service_from_gui(emitter):
    exchange = StubExchange()
    exec_service = StubExecutionService()
    ai = StubAI(0.02)
    risk = StubRisk(0.1)
    gui = DummyGUI(exec_service, exchange, ai, risk)

    trader = AutoTrader(
        emitter,
        gui=gui,
        symbol_getter=lambda: "BTC/USDT",
        metrics_window=5,
    )

    assert trader._auto_trade_once() is True
    assert len(exec_service.orders) == 1
    assert gui.bridge_called is False
