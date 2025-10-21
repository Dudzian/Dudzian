from __future__ import annotations

import datetime as dt

from bot_core.exchanges.base import AccountSnapshot
from bot_core.exchanges.core import MarketRules
from bot_core.exchanges.manager import ExchangeManager, Mode, register_native_adapter


class _SmokeAdapter:
    def __init__(self, credentials, *, environment, settings=None, watchdog=None):
        self.credentials = credentials
        self.environment = environment
        self.settings = dict(settings or {})
        self.watchdog = watchdog

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 5_000.0},
            total_equity=5_000.0,
            available_margin=4_500.0,
            maintenance_margin=250.0,
        )


register_native_adapter(exchange_id="smokex", mode=Mode.MARGIN, factory=_SmokeAdapter)
register_native_adapter(exchange_id="smokex", mode=Mode.FUTURES, factory=_SmokeAdapter)


class _FakePublic:
    def __init__(self) -> None:
        self.rules = {
            "BTC/USDT": MarketRules(
                symbol="BTC/USDT",
                price_step=0.5,
                amount_step=0.001,
                min_notional=5.0,
            )
        }

    def load_markets(self):
        return self.rules

    def get_market_rules(self, symbol: str):
        return self.rules.get(symbol)

    def fetch_ticker(self, symbol: str):
        return {"last": 100.0, "close": 100.0}

    def fetch_order_book(self, symbol: str, limit: int = 50):
        return {"bids": [[99.5, 2.0]], "asks": [[100.5, 2.0]]}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        now = dt.datetime.utcnow()
        return [[now.timestamp(), 99.0, 101.0, 98.5, 100.0, 10.0]]


def _prepare_manager(monkeypatch, mode: Mode) -> ExchangeManager:
    manager = ExchangeManager("smokex")
    fake_public = _FakePublic()
    monkeypatch.setattr(manager, "_ensure_public", lambda: fake_public)
    if mode is Mode.PAPER:
        manager.set_mode(paper=True)
    elif mode is Mode.MARGIN:
        manager.set_mode(margin=True, testnet=True)
        manager.set_credentials("key", "secret")
    else:
        manager.set_mode(futures=True, testnet=True)
        manager.set_credentials("key", "secret")
    manager.load_markets()
    return manager


def test_smoke_paper_margin_pipeline(monkeypatch):
    manager = _prepare_manager(monkeypatch, Mode.PAPER)
    manager.set_paper_variant("margin")
    manager.configure_paper_simulator(leverage_limit=4.0)
    price, slippage = manager.simulate_vwap_price("BTC/USDT", "buy", 0.5)
    balance = manager.fetch_balance()
    assert price is not None
    assert "USDT" in balance


def test_smoke_margin_pipeline(monkeypatch):
    manager = _prepare_manager(monkeypatch, Mode.MARGIN)
    balance = manager.fetch_balance()
    assert balance["total_equity"] == 5_000.0
    price, slippage = manager.simulate_vwap_price("BTC/USDT", "buy", 0.1)
    assert price is not None


def test_smoke_futures_pipeline(monkeypatch):
    manager = _prepare_manager(monkeypatch, Mode.FUTURES)
    balance = manager.fetch_balance()
    assert balance["total_equity"] == 5_000.0
    price, slippage = manager.simulate_vwap_price("BTC/USDT", "sell", 0.2)
    assert price is not None
