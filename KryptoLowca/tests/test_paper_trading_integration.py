"""Testy integracyjne warstwy paper trading."""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from KryptoLowca.managers.database_manager import DatabaseManager
from KryptoLowca.managers.paper_exchange import BUY, LIMIT, MARKET, PaperExchange


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

    # --- API wykorzystywane przez PaperExchange ---
    def record_order(self, payload: dict) -> int:
        order_id = len(self.orders) + 1
        entry = dict(payload)
        entry["id"] = order_id
        self.orders.append(entry)
        return order_id

    def update_order_status(self, **payload: object) -> None:
        self.order_updates.append(dict(payload))

    def record_trade(self, payload: dict) -> None:
        self.trades.append(dict(payload))

    def upsert_position(self, payload: dict) -> None:
        self.positions.append(dict(payload))

    def close_position(self, symbol: str) -> None:
        self.closed.append(symbol)

    def log_equity(self, payload: dict) -> None:
        self.equity.append(dict(payload))


@pytest.fixture()
def paper_exchange() -> PaperExchange:
    db = DummyDB()
    exchange = PaperExchange(
        db,
        symbol="BTC/USDT",
        starting_balance=10_000.0,
        fee_rate=0.0,
        slippage_bps=0,
    )
    # ustawiamy ostatnią cenę, aby zlecenia MARKET wypełniały się natychmiast
    exchange.process_tick(100.0, ts=dt.datetime.utcnow())
    return exchange


def test_market_order_flow(paper_exchange: PaperExchange) -> None:
    order_id = paper_exchange.create_order(side=BUY, type=MARKET, quantity=0.5)

    assert order_id == 1
    assert paper_exchange.get_position()["quantity"] == pytest.approx(0.5)
    assert paper_exchange.db.sync.orders[0]["status"] == "NEW"  # type: ignore[attr-defined]
    # ostatnia aktualizacja statusu powinna oznaczać FILLED
    assert paper_exchange.db.sync.order_updates[-1]["status"] == "FILLED"  # type: ignore[attr-defined]
    assert paper_exchange.db.sync.trades  # type: ignore[attr-defined]


def test_limit_order_requires_tick(paper_exchange: PaperExchange) -> None:
    order_id = paper_exchange.create_order(side=BUY, type=LIMIT, quantity=0.25, price=99.5)
    assert order_id == 1
    # brak fill przed osiągnięciem ceny
    assert paper_exchange.db.sync.order_updates[-1]["status"] == "OPEN"  # type: ignore[attr-defined]

    # cena spada poniżej limitu -> zlecenie powinno się stopniowo wypełniać
    for price in (99.0, 98.8, 98.5, 98.0):
        paper_exchange.process_tick(price, ts=dt.datetime.utcnow())
        if paper_exchange.db.sync.order_updates[-1]["status"] == "FILLED":  # type: ignore[attr-defined]
            break

    assert paper_exchange.db.sync.order_updates[-1]["status"] in {"PARTIALLY_FILLED", "FILLED"}  # type: ignore[attr-defined]

    filled = any(
        update["status"] == "FILLED"
        for update in paper_exchange.db.sync.order_updates  # type: ignore[attr-defined]
    )
    if not filled:
        for _ in range(10):
            paper_exchange.process_tick(98.0, ts=dt.datetime.utcnow())
            if paper_exchange.db.sync.order_updates[-1]["status"] == "FILLED":  # type: ignore[attr-defined]
                filled = True
                break

    assert filled
    assert paper_exchange.get_position()["quantity"] == pytest.approx(0.25)


def test_equity_logging(paper_exchange: PaperExchange) -> None:
    paper_exchange.process_tick(101.0, ts=dt.datetime.utcnow())
    paper_exchange.process_tick(102.0, ts=dt.datetime.utcnow())

    assert paper_exchange.db.sync.equity  # type: ignore[attr-defined]
    last_entry = paper_exchange.db.sync.equity[-1]  # type: ignore[attr-defined]
    assert set(last_entry.keys()) >= {"equity", "balance", "pnl", "mode"}


def test_paper_exchange_with_real_database(tmp_path) -> None:
    """Zapewnia, że PaperExchange współpracuje z prawdziwym DatabaseManagerem."""

    db_path = tmp_path / "paper_smoke.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    db = DatabaseManager(db_url=db_url)
    asyncio.run(db.init_db(create=True))

    try:
        exchange = PaperExchange(
            db,
            symbol="BTC/USDT",
            starting_balance=5_000.0,
            fee_rate=0.0,
            slippage_bps=0,
        )
        exchange.process_tick(100.0, ts=dt.datetime.utcnow())

        order_id = exchange.create_order(side=BUY, type=MARKET, quantity=0.25)
        assert order_id == 1

        # W bazie powinien pojawić się wpis o transakcji
        trades = asyncio.run(db.fetch_trades(symbol="BTC/USDT"))
        assert trades, "Oczekiwano zarejestrowanych transakcji w bazie"

        position = exchange.get_position()
        assert position["quantity"] == pytest.approx(0.25)
        assert position["balance_quote"] < 5_000.0
    finally:
        if hasattr(db, "_state") and db._state.engine is not None:  # type: ignore[attr-defined]
            asyncio.run(db._state.engine.dispose())
