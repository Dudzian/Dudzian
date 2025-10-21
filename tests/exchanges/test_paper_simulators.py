import datetime as dt

import pytest

from bot_core.exchanges.core import MarketRules, OrderSide, OrderType
from bot_core.exchanges.paper_simulator import PaperFuturesSimulator, PaperMarginSimulator


class _DummyFeed:
    def __init__(self, price: float = 100.0) -> None:
        self.price = float(price)
        self.rules = MarketRules(symbol="BTC/USDT", price_step=0.1, amount_step=0.001, min_notional=10.0)

    def load_markets(self):
        return {self.rules.symbol: self.rules}

    def get_market_rules(self, symbol: str):
        if symbol == self.rules.symbol:
            return self.rules
        return None

    def fetch_ticker(self, symbol: str):
        if symbol != self.rules.symbol:
            raise ValueError("unknown symbol")
        return {"last": self.price}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500):  # pragma: no cover - unused
        raise NotImplementedError


class _DummyDB:
    class _Sync:
        def __init__(self) -> None:
            self._order_id = 0
            self.orders: dict[int, dict[str, object]] = {}
            self.positions: dict[str, dict[str, object]] = {}
            self.trades: list[dict[str, object]] = []
            self.equity_log: list[dict[str, object]] = []

        def init_db(self) -> None:
            return None

        def record_order(self, payload):
            self._order_id += 1
            self.orders[self._order_id] = dict(payload)
            return self._order_id

        def update_order_status(self, order_id: int, status: str) -> None:
            if order_id in self.orders:
                self.orders[order_id]["status"] = status

        def record_trade(self, payload) -> None:
            self.trades.append(dict(payload))

        def upsert_position(self, payload) -> None:
            self.positions[payload["symbol"]] = dict(payload)

        def log_equity(self, payload) -> None:
            self.equity_log.append(dict(payload))

    def __init__(self) -> None:
        self.sync = self._Sync()


def _make_margin_simulator(price: float = 100.0, **kwargs) -> PaperMarginSimulator:
    feed = _DummyFeed(price)
    database = _DummyDB()
    simulator = PaperMarginSimulator(
        feed,
        database=database,
        leverage_limit=kwargs.get("leverage_limit", 3.0),
        maintenance_margin_ratio=kwargs.get("maintenance_margin_ratio", 0.15),
        funding_rate=kwargs.get("funding_rate", 0.0),
        funding_interval_seconds=kwargs.get("funding_interval_seconds"),
    )
    simulator.load_markets()
    return simulator


def test_margin_simulator_generates_snapshot_and_events():
    simulator = _make_margin_simulator(price=20_000.0)
    simulator.create_order("BTC/USDT", OrderSide.BUY, OrderType.MARKET, 0.1)
    simulator.create_order("BTC/USDT", OrderSide.SELL, OrderType.MARKET, 0.05)

    snapshot = simulator.fetch_account_snapshot()
    assert snapshot.total_equity > 0
    assert "BTC/USDT_position" in snapshot.balances

    events = list(simulator.fetch_margin_events())
    assert any(event["type"] == "leverage_change" for event in events)


def test_futures_simulator_applies_funding_and_reports_exposure():
    simulator = PaperFuturesSimulator(_DummyFeed(25_000.0), database=_DummyDB(), funding_rate=0.001)
    simulator.load_markets()
    simulator.create_order("BTC/USDT", OrderSide.SELL, OrderType.MARKET, 0.2)

    # force funding accrual before next trade
    simulator._margin_state.last_funding -= dt.timedelta(hours=8)  # type: ignore[attr-defined]
    simulator.create_order("BTC/USDT", OrderSide.BUY, OrderType.MARKET, 0.1)

    snapshot = simulator.fetch_account_snapshot()
    assert "futures_exposure" in snapshot.balances
    events = list(simulator.fetch_margin_events())
    assert any(event["type"] == "funding" for event in events)


def test_simulator_describe_configuration_reports_runtime_values():
    simulator = _make_margin_simulator(
        leverage_limit=7.0,
        maintenance_margin_ratio=0.2,
        funding_rate=0.0004,
        funding_interval_seconds=7_200,
    )

    config = simulator.describe_configuration()

    assert config["leverage_limit"] == pytest.approx(7.0)
    assert config["maintenance_margin_ratio"] == pytest.approx(0.2)
    assert config["funding_rate"] == pytest.approx(0.0004)
    assert config["funding_interval_seconds"] == pytest.approx(7_200.0)


def test_margin_simulator_respects_funding_interval() -> None:
    simulator = _make_margin_simulator(
        price=30_000.0,
        funding_rate=0.001,
        funding_interval_seconds=3_600,
    )
    simulator.create_order("BTC/USDT", OrderSide.BUY, OrderType.MARKET, 0.1)
    simulator._margin_events.clear()
    base = simulator._margin_state.last_funding  # type: ignore[attr-defined]
    initial_cash = simulator._cash_balance  # type: ignore[attr-defined]

    simulator._apply_funding(base + dt.timedelta(minutes=30))  # type: ignore[attr-defined]

    assert simulator._margin_state.last_funding == base  # type: ignore[attr-defined]
    assert not any(event["type"] == "funding" for event in simulator.fetch_margin_events())

    simulator._apply_funding(base + dt.timedelta(hours=3))  # type: ignore[attr-defined]

    events = list(simulator.fetch_margin_events())
    assert events and events[-1]["type"] == "funding"
    payload = events[-1]["payload"]
    assert payload["periods"] == 3
    assert payload["interval_seconds"] == pytest.approx(3_600.0)
    assert simulator._margin_state.last_funding == base + dt.timedelta(hours=3)  # type: ignore[attr-defined]
    assert simulator._cash_balance < initial_cash  # type: ignore[attr-defined]
