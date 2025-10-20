import datetime as dt
from typing import Iterable, Tuple

import pytest

from bot_core.exchanges.core import MarketRules, Mode, OrderSide, OrderStatus, OrderType
from KryptoLowca.managers.exchange_manager import ExchangeManager


class DummyDB:
    """Minimalny stub DatabaseManager.sync wykorzystywany w testach."""

    def __init__(self) -> None:
        self.orders: list[dict] = []
        self.order_updates: list[dict] = []
        self.trades: list[dict] = []
        self.positions: list[dict] = []
        self.equity: list[dict] = []
        self.closed: list[str] = []
        self.sync = self

    # --- API kompatybilne z PaperBackend ---
    def init_db(self) -> None:
        return None

    def record_order(self, payload: dict) -> int:
        order_id = len(self.orders) + 1
        entry = dict(payload)
        entry["id"] = order_id
        self.orders.append(entry)
        return order_id

    def update_order_status(self, **payload: object) -> None:
        data = dict(payload)
        self.order_updates.append(data)
        order_id = data.get("order_id")
        status = data.get("status")
        if order_id is None or status is None:
            return
        for entry in self.orders:
            if entry.get("id") == order_id:
                entry["status"] = status
                break

    def record_trade(self, payload: dict) -> None:
        self.trades.append(dict(payload))

    def upsert_position(self, payload: dict) -> None:
        self.positions.append(dict(payload))

    def close_position(self, symbol: str) -> None:
        self.closed.append(symbol)

    def log_equity(self, payload: dict) -> None:
        self.equity.append(dict(payload))


class FakeFeed:
    def __init__(self) -> None:
        self.last_price = 100.0
        self._rules = {
            "BTC/USDT": MarketRules(
                symbol="BTC/USDT",
                price_step=0.1,
                amount_step=0.001,
                min_notional=10.0,
            )
        }

    def load_markets(self) -> dict[str, MarketRules]:
        return self._rules

    def get_market_rules(self, symbol: str) -> MarketRules:
        return self._rules[symbol]

    def fetch_ticker(self, symbol: str) -> dict[str, float]:
        return {"last": float(self.last_price)}

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> Iterable[Iterable[float]]:
        return []


@pytest.fixture()
def paper_manager() -> Tuple[ExchangeManager, FakeFeed, DummyDB]:
    manager = ExchangeManager()
    dummy_db = DummyDB()
    feed = FakeFeed()
    # wstrzykujemy stuby, aby ExchangeManager nie próbował inicjalizować CCXT ani prawdziwej bazy
    manager._db = dummy_db  # type: ignore[attr-defined]
    manager._db_failed = False  # type: ignore[attr-defined]
    manager._public = feed  # type: ignore[attr-defined]
    manager.set_mode(paper=True)
    manager.set_paper_balance(10_000.0, asset="USDT")
    manager.set_paper_fee_rate(0.0)
    manager.load_markets()
    manager.process_paper_tick("BTC/USDT", feed.last_price, timestamp=dt.datetime.utcnow())
    return manager, feed, dummy_db


def test_market_order_flow(paper_manager: Tuple[ExchangeManager, FakeFeed, DummyDB]) -> None:
    manager, feed, db = paper_manager
    feed.last_price = 100.0
    manager.process_paper_tick("BTC/USDT", feed.last_price, timestamp=dt.datetime.utcnow())

    order = manager.create_order(
        "BTC/USDT",
        OrderSide.BUY.value,
        OrderType.MARKET.value,
        0.5,
    )

    assert order.status == OrderStatus.FILLED
    positions = manager.fetch_positions("BTC/USDT")
    assert positions and positions[0].quantity == pytest.approx(0.5)
    assert db.orders[0]["status"] == OrderStatus.FILLED.value
    assert db.order_updates[-1]["status"] == OrderStatus.FILLED.value
    assert db.trades


def test_limit_order_requires_tick(
    paper_manager: Tuple[ExchangeManager, FakeFeed, DummyDB]
) -> None:
    manager, feed, db = paper_manager

    order = manager.create_order(
        "BTC/USDT",
        OrderSide.BUY.value,
        OrderType.LIMIT.value,
        0.25,
        price=99.5,
    )
    assert order.status == OrderStatus.OPEN
    assert db.order_updates[-1]["status"] == OrderStatus.OPEN.value

    for price in (100.0, 99.8, 99.5, 99.0):
        feed.last_price = price
        manager.process_paper_tick("BTC/USDT", price, timestamp=dt.datetime.utcnow())
        if db.order_updates[-1]["status"] == OrderStatus.FILLED.value:
            break

    assert db.order_updates[-1]["status"] == OrderStatus.FILLED.value
    positions = manager.fetch_positions("BTC/USDT")
    assert positions and positions[0].quantity == pytest.approx(0.25)


def test_equity_logging(paper_manager: Tuple[ExchangeManager, FakeFeed, DummyDB]) -> None:
    manager, feed, db = paper_manager

    feed.last_price = 101.0
    manager.process_paper_tick("BTC/USDT", feed.last_price, timestamp=dt.datetime.utcnow())
    feed.last_price = 102.0
    manager.process_paper_tick("BTC/USDT", feed.last_price, timestamp=dt.datetime.utcnow())

    assert db.equity
    last_entry = db.equity[-1]
    assert set(last_entry.keys()) >= {"equity", "balance", "pnl", "mode"}
    assert last_entry["mode"] == Mode.PAPER.value


def test_paper_manager_with_real_database(tmp_path) -> None:
    db_path = tmp_path / "paper_smoke.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    manager = ExchangeManager(db_url=db_url)
    feed = FakeFeed()
    manager._public = feed  # type: ignore[attr-defined]
    manager.set_mode(paper=True)
    manager.set_paper_balance(5_000.0, asset="USDT")
    manager.set_paper_fee_rate(0.0)
    manager.load_markets()
    manager.process_paper_tick("BTC/USDT", 100.0, timestamp=dt.datetime.utcnow())

    order = manager.create_order(
        "BTC/USDT",
        OrderSide.BUY.value,
        OrderType.MARKET.value,
        0.25,
    )
    assert order.status == OrderStatus.FILLED

    trades = manager._ensure_db().sync.fetch_trades(symbol="BTC/USDT", mode=Mode.PAPER.value)  # type: ignore[attr-defined]
    assert trades

    positions = manager.fetch_positions("BTC/USDT")
    assert positions and positions[0].quantity == pytest.approx(0.25)
