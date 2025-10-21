"""Testy fasady ExchangeManager opartej na `bot_core`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import pytest

from bot_core.exchanges.core import (
    MarketRules,
    Mode,
    OrderDTO,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionDTO,
)
from bot_core.exchanges.manager import ExchangeManager


class StubPublicBackend:
    """Backend publiczny CCXT zastępowany w testach."""

    def __init__(self) -> None:
        self.load_calls = 0
        self.ohlcv_requests: List[Tuple[str, str, int]] = []
        self.ticker_requests: List[str] = []
        self.order_book_requests: List[Tuple[str, int]] = []
        self._rules: Dict[str, MarketRules] = {
            "BTC/USDT": MarketRules(
                symbol="BTC/USDT",
                price_step=0.1,
                amount_step=0.001,
                min_notional=10.0,
                min_amount=0.001,
            ),
            "ETH/USDT": MarketRules(
                symbol="ETH/USDT",
                price_step=0.05,
                amount_step=0.01,
                min_notional=5.0,
                min_amount=0.01,
            ),
        }

    def load_markets(self) -> Dict[str, MarketRules]:
        self.load_calls += 1
        return self._rules

    def get_market_rules(self, symbol: str) -> Optional[MarketRules]:
        return self._rules.get(symbol, None)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        self.ohlcv_requests.append((symbol, timeframe, limit))
        return [[1_700_000_000_000, 100.0, 101.0, 99.5, 100.5, 12.0]]

    def fetch_ticker(self, symbol: str):
        self.ticker_requests.append(symbol)
        return {"last": 100.0}

    def fetch_order_book(self, symbol: str, limit: int = 50):
        self.order_book_requests.append((symbol, limit))
        return {
            "asks": [[100.0, 0.3], [100.5, 0.3], [101.0, 0.4]],
            "bids": [[99.5, 0.3], [99.0, 0.4]],
        }


class StubPrivateBackend:
    """Backend prywatny CCXT imitujący połączenie live/testnet."""

    def __init__(self, public: StubPublicBackend) -> None:
        self._public = public
        self.created_orders: List[OrderDTO] = []
        self.cancelled: List[Tuple[str, str]] = []
        self._positions: List[PositionDTO] = [
            PositionDTO(
                symbol="BTC/USDT",
                side="LONG",
                quantity=0.3,
                avg_price=100.0,
                unrealized_pnl=15.0,
                mode=Mode.FUTURES,
            )
        ]

    def load_markets(self) -> Dict[str, MarketRules]:
        return self._public.load_markets()

    def fetch_balance(self) -> Dict[str, dict]:
        return {
            "free": {"USDT": "900.0", "BTC": 0.2},
            "total": {"USDT": 1_000.0, "BTC": 0.2},
            "used": {"USDT": 100.0},
            "info": {"note": "demo"},
        }

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderDTO:
        order = OrderDTO(
            id=1,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            client_order_id=client_order_id,
            status=OrderStatus.FILLED,
            mode=Mode.SPOT,
        )
        self.created_orders.append(order)
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        self.cancelled.append((order_id, symbol))
        return True

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderDTO]:
        return [
            OrderDTO(
                id=2,
                symbol=symbol or "BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                quantity=0.1,
                price=99.5,
                status=OrderStatus.OPEN,
                mode=Mode.SPOT,
            )
        ]

    def fetch_positions(self, symbol: Optional[str] = None) -> List[PositionDTO]:
        if symbol:
            return [pos for pos in self._positions if pos.symbol == symbol]
        return list(self._positions)


class StubPaperBackend:
    """Backend paper tradingu używany w trybie symulacyjnym."""

    def __init__(self, public: StubPublicBackend) -> None:
        self._public = public
        self.orders: List[OrderDTO] = []
        self._positions: List[PositionDTO] = [
            PositionDTO(
                symbol="ETH/USDT",
                side="LONG",
                quantity=1.2,
                avg_price=1_500.0,
                unrealized_pnl=25.0,
                mode=Mode.PAPER,
            )
        ]

    def load_markets(self) -> Dict[str, MarketRules]:
        return self._public.load_markets()

    def fetch_balance(self) -> Dict[str, dict]:
        return {"total": {"USDT": 5_000.0}}

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderDTO:
        order = OrderDTO(
            id=3,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.FILLED,
            mode=Mode.PAPER,
            client_order_id=client_order_id,
        )
        self.orders.append(order)
        return order

    def fetch_positions(self, symbol: Optional[str] = None) -> List[PositionDTO]:
        if symbol:
            return [pos for pos in self._positions if pos.symbol == symbol]
        return list(self._positions)


class StubDatabase:
    """Prosty magazyn pozycji zwracający przygotowaną listę."""

    def __init__(self, positions: Iterable[dict]) -> None:
        self.positions = [dict(p) for p in positions]
        self.sync = SimpleNamespace(
            init_db=lambda: None,
            get_open_positions=self._get_positions,
        )

    def _get_positions(self, mode: str) -> List[dict]:
        return [dict(p) for p in self.positions]


@pytest.fixture()
def manager_factory():
    def _build(*, mode: Mode = Mode.PAPER, db_positions: Optional[Iterable[dict]] = None):
        manager = ExchangeManager(exchange_id="stub", db_url="sqlite+aiosqlite:///stub.db")
        if mode is Mode.PAPER:
            manager.set_mode(paper=True)
        elif mode is Mode.SPOT:
            manager.set_mode(spot=True)
        else:
            manager.set_mode(futures=True, testnet=True)

        public = StubPublicBackend()
        private = StubPrivateBackend(public)
        paper = StubPaperBackend(public)

        manager._public = public
        manager._private = private
        manager._paper = paper

        if db_positions is not None:
            manager._db = StubDatabase(db_positions)
        return manager, public, private, paper

    return _build


def test_load_markets_exposes_rules(manager_factory):
    manager, public, _, _ = manager_factory(mode=Mode.PAPER)
    markets = manager.load_markets()

    assert "BTC/USDT" in markets
    assert markets["BTC/USDT"].min_notional == pytest.approx(10.0)
    assert public.load_calls >= 1
    assert manager.get_market_rules("ETH/USDT").amount_step == pytest.approx(0.01)


def test_fetch_ohlcv_and_ticker_use_public_backend(manager_factory):
    manager, public, _, _ = manager_factory(mode=Mode.PAPER)
    candles = manager.fetch_ohlcv("BTC/USDT", "1h", 3)

    assert candles[0][1:] == [100.0, 101.0, 99.5, 100.5, 12.0]
    assert public.ohlcv_requests == [("BTC/USDT", "1h", 3)]

    ticker = manager.fetch_ticker("ETH/USDT")
    assert ticker == {"last": 100.0}
    assert public.ticker_requests == ["ETH/USDT"]


def test_simulate_vwap_price_calculates_slippage(manager_factory):
    manager, public, _, _ = manager_factory(mode=Mode.PAPER)
    price, slip = manager.simulate_vwap_price("BTC/USDT", "buy", amount=0.5)

    assert price == pytest.approx(100.2)
    assert slip == pytest.approx(20.0)
    assert public.order_book_requests == [("BTC/USDT", 50)]


def test_create_order_in_paper_mode_uses_paper_backend(manager_factory):
    manager, _, _, paper = manager_factory(mode=Mode.PAPER)

    order = manager.create_order("BTC/USDT", "buy", "market", 0.25)

    assert order.mode is Mode.PAPER
    assert paper.orders
    stored = paper.orders[-1]
    assert stored.symbol == "BTC/USDT"
    assert stored.quantity == pytest.approx(0.25)


def test_fetch_balance_spot_normalizes_totals(manager_factory):
    manager, _, private, _ = manager_factory(mode=Mode.SPOT)
    manager.set_credentials("key", "secret")
    manager._private = private

    balance = manager.fetch_balance()

    assert balance["BTC"] == pytest.approx(0.2)
    assert balance["total"]["USDT"] == pytest.approx(1_000.0)
    assert balance["free"]["USDT"] == pytest.approx(900.0)


def test_fetch_balance_paper_returns_snapshot(manager_factory):
    manager, _, _, paper = manager_factory(mode=Mode.PAPER)
    balance = manager.fetch_balance()

    assert balance["total"]["USDT"] == pytest.approx(5_000.0)
    assert paper.orders == []


def test_cancel_order_forwards_to_private_backend(manager_factory):
    manager, _, private, _ = manager_factory(mode=Mode.SPOT)
    manager.set_credentials("key", "secret")
    manager._private = private

    assert manager.cancel_order("abc", "BTC/USDT") is True
    assert private.cancelled == [("abc", "BTC/USDT")]


def test_fetch_positions_prefers_database_snapshot(manager_factory):
    db_positions = [
        {
            "symbol": "BTC/USDT",
            "side": "LONG",
            "quantity": 0.8,
            "avg_price": 98.5,
            "unrealized_pnl": 12.0,
            "mode": Mode.SPOT.value,
        }
    ]
    manager, _, private, _ = manager_factory(mode=Mode.SPOT, db_positions=db_positions)
    manager.set_credentials("key", "secret")
    manager._private = private

    positions = manager.fetch_positions()

    assert len(positions) == 1
    position = positions[0]
    assert position.symbol == "BTC/USDT"
    assert position.quantity == pytest.approx(0.8)
    assert position.mode is Mode.SPOT


def test_fetch_positions_falls_back_to_balance(manager_factory):
    manager, public, private, _ = manager_factory(mode=Mode.SPOT, db_positions=[])
    manager.set_credentials("key", "secret")
    manager._private = private

    # zapewnij dostępne rynki i kwotowania
    public.load_markets()
    positions = manager.fetch_positions()

    assert positions
    derived = positions[0]
    assert derived.symbol == "BTC/USDT"
    assert derived.quantity == pytest.approx(0.2)
    assert derived.mode is Mode.SPOT


def test_fetch_positions_in_paper_mode_returns_backend_state(manager_factory):
    manager, _, _, paper = manager_factory(mode=Mode.PAPER)
    positions = manager.fetch_positions()

    assert positions
    assert positions[0].mode is Mode.PAPER
    assert positions[0].symbol == "ETH/USDT"


