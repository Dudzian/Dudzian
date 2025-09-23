# test_exchange_manager.py
# -*- coding: utf-8 -*-
"""Testy jednostkowe fasady ExchangeManager (synchronizacja Fazą 0)."""

from __future__ import annotations

import pytest

import sys
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from managers.exchange_manager import ExchangeManager
from managers.exchange_core import (
    MarketRules,
    Mode,
    OrderDTO,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionDTO,
)


class DummyPublic:
    def __init__(self) -> None:
        self.rules = {
            "BTC/USDT": MarketRules(
                symbol="BTC/USDT",
                price_step=0.1,
                amount_step=0.001,
                min_notional=10.0,
            ),
            "ETH/USDT": MarketRules(
                symbol="ETH/USDT",
                price_step=0.05,
                amount_step=0.01,
                min_notional=5.0,
            ),
        }
        self.order_book = {
            "asks": [[100.0, 1.0], [101.0, 2.0], [102.0, 4.0]],
            "bids": [[99.5, 1.5], [99.0, 3.0], [98.5, 6.0]],
        }
        self.ohlcv = [
            [1_700_000_000_000, 100.0, 101.0, 99.0, 100.5, 10.0],
            [1_700_000_060_000, 100.5, 102.0, 99.5, 101.0, 12.0],
        ]

    def load_markets(self):
        return self.rules

    def get_market_rules(self, symbol: str):
        return self.rules.get(symbol)

    def fetch_ticker(self, symbol: str):
        return {"last": 100.0, "close": 100.0}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        return self.ohlcv[:limit]

    def fetch_order_book(self, symbol: str, limit: int = 50):
        return self.order_book


class DummyPaperBackend:
    def __init__(self, public: DummyPublic) -> None:
        self.public = public
        self.created = []

    def load_markets(self):
        return self.public.load_markets()

    def get_market_rules(self, symbol: str):
        return self.public.get_market_rules(symbol)

    def fetch_ticker(self, symbol: str):
        return self.public.fetch_ticker(symbol)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        return self.public.fetch_ohlcv(symbol, timeframe, limit)

    def fetch_order_book(self, symbol: str, limit: int = 50):
        return self.public.fetch_order_book(symbol, limit)

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        type_: OrderType,
        quantity: float,
        price=None,
        client_order_id=None,
    ) -> OrderDTO:
        self.created.append((symbol, side, type_, quantity, price, client_order_id))
        return OrderDTO(
            id=123,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            type=type_,
            quantity=quantity,
            price=price,
            status=OrderStatus.FILLED,
            mode=Mode.PAPER,
        )

    def fetch_balance(self):
        return {"free": {"USDT": 1_000.0}, "total": {"USDT": 1_000.0}}

    def fetch_positions(self, symbol: str | None = None):
        pos = PositionDTO(
            symbol="BTC/USDT",
            side="LONG",
            quantity=0.5,
            avg_price=100.0,
            unrealized_pnl=5.0,
            mode=Mode.PAPER,
        )
        if symbol and symbol != pos.symbol:
            return []
        return [pos]


class DummyPrivateBackend(DummyPaperBackend):
    def __init__(self, public: DummyPublic) -> None:
        super().__init__(public)
        self.mode = Mode.SPOT

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        type_: OrderType,
        quantity: float,
        price=None,
        client_order_id=None,
    ) -> OrderDTO:
        self.created.append((symbol, side, type_, quantity, price, client_order_id))
        return OrderDTO(
            id=456,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            type=type_,
            quantity=quantity,
            price=price,
            status=OrderStatus.OPEN,
            mode=self.mode,
        )

    def fetch_balance(self):
        return {"free": {"USDT": 500.0}, "total": {"USDT": 600.0}}

    def fetch_positions(self, symbol: str | None = None):
        pos = PositionDTO(
            symbol="ETH/USDT",
            side="LONG",
            quantity=1.0,
            avg_price=50.0,
            unrealized_pnl=2.5,
            mode=Mode.SPOT,
        )
        if symbol and symbol != pos.symbol:
            return []
        return [pos]


@pytest.fixture
def manager() -> ExchangeManager:
    mgr = ExchangeManager(exchange_id="dummy", paper_initial_cash=5_000.0)
    public = DummyPublic()
    paper = DummyPaperBackend(public)
    private = DummyPrivateBackend(public)

    def _ensure_public(self):
        return public

    def _ensure_paper(self):
        self.mode = Mode.PAPER
        return paper

    def _ensure_private(self):
        self.mode = Mode.SPOT
        return private

    mgr._public = public  # type: ignore[attr-defined]
    mgr._paper = paper  # type: ignore[attr-defined]
    mgr._private = private  # type: ignore[attr-defined]
    mgr._api_key = "test"  # type: ignore[attr-defined]
    mgr._secret = "secret"  # type: ignore[attr-defined]
    mgr._ensure_public = types.MethodType(_ensure_public, mgr)  # type: ignore[attr-defined]
    mgr._ensure_paper = types.MethodType(_ensure_paper, mgr)  # type: ignore[attr-defined]
    mgr._ensure_private = types.MethodType(_ensure_private, mgr)  # type: ignore[attr-defined]
    return mgr


def test_load_markets_and_quantization(manager: ExchangeManager):
    markets = manager.load_markets()
    assert "BTC/USDT" in markets
    amount = manager.quantize_amount("BTC/USDT", 0.123456)
    assert amount == pytest.approx(0.123)
    price = manager.quantize_price("BTC/USDT", 101.337)
    assert price == pytest.approx(101.3)


def test_simulate_vwap_price(manager: ExchangeManager):
    price, slip = manager.simulate_vwap_price("BTC/USDT", "buy", amount=2.5)
    assert price == pytest.approx(100.6, rel=1e-6)
    assert slip > 0


def test_fetch_balance_paper(manager: ExchangeManager):
    manager.mode = Mode.PAPER
    balance = manager.fetch_balance()
    assert balance["free"]["USDT"] == pytest.approx(1_000.0)


def test_fetch_balance_spot(manager: ExchangeManager):
    manager.mode = Mode.SPOT
    balance = manager.fetch_balance()
    assert balance["free"]["USDT"] == pytest.approx(500.0)
    assert balance["total"]["USDT"] == pytest.approx(600.0)


def test_create_order_paper(manager: ExchangeManager):
    manager.mode = Mode.PAPER
    order = manager.create_order("BTC/USDT", "BUY", "MARKET", 0.25)
    assert order.mode == Mode.PAPER
    assert manager._paper.created  # type: ignore[attr-defined]


def test_create_order_spot(manager: ExchangeManager):
    manager.mode = Mode.SPOT
    order = manager.create_order("ETH/USDT", "SELL", "LIMIT", 0.5, price=52.0)
    assert order.mode == Mode.SPOT
    created = manager._private.created  # type: ignore[attr-defined]
    assert created[-1][0] == "ETH/USDT"


def test_fetch_positions_by_mode(manager: ExchangeManager):
    manager.mode = Mode.PAPER
    paper_positions = manager.fetch_positions()
    assert paper_positions and paper_positions[0].mode == Mode.PAPER

    manager.mode = Mode.SPOT
    spot_positions = manager.fetch_positions()
    assert spot_positions and spot_positions[0].mode == Mode.SPOT


def test_set_mode_resets_backends(manager: ExchangeManager):
    manager._paper = DummyPaperBackend(DummyPublic())  # type: ignore[attr-defined]
    manager._private = DummyPrivateBackend(DummyPublic())  # type: ignore[attr-defined]
    manager.set_mode(paper=True)
    assert manager._paper is None  # type: ignore[attr-defined]
    assert manager._private is None  # type: ignore[attr-defined]


def test_fetch_ohlcv_pass_through(manager: ExchangeManager):
    data = manager.fetch_ohlcv("BTC/USDT", timeframe="1m", limit=1)
    assert data and data[0][0] == 1_700_000_000_000
