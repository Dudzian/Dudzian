from __future__ import annotations

import types
from urllib.error import URLError

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.binance.spot import BinanceSpotAdapter, _MAX_RETRIES
from bot_core.exchanges.errors import ExchangeNetworkError
from bot_core.exchanges import manager as manager_module
from bot_core.exchanges.manager import ExchangeManager


def _build_fake_ccxt_module() -> types.SimpleNamespace:
    network_error = type("NetworkError", (Exception,), {})

    class FakeExchange:
        def __init__(self, options: dict[str, object]):
            self._options = options
            self.urls = {}
            self._ticker_calls = 0

        def setSandboxMode(self, enabled: bool) -> None:  # pragma: no cover - used indirectly
            self.sandbox_enabled = enabled

        def load_markets(self) -> dict[str, dict[str, object]]:
            return {
                "BTC/USDT": {
                    "limits": {"amount": {"min": 0.001}},
                    "precision": {"price": 2, "amount": 2},
                }
            }

        def fetch_ticker(self, symbol: str) -> dict[str, object]:
            self._ticker_calls += 1
            if self._ticker_calls == 1:
                raise network_error("temporary outage")
            return {"symbol": symbol, "last": 123.0}

        def fetch_ohlcv(self, *args, **kwargs):  # pragma: no cover - not used
            raise network_error("unreachable")

        def fetch_order_book(self, *args, **kwargs):
            raise network_error("unreachable")

    return types.SimpleNamespace(NetworkError=network_error, binance=FakeExchange)


def test_exchange_manager_network_error_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ccxt = _build_fake_ccxt_module()
    monkeypatch.setattr(manager_module, "ccxt", fake_ccxt)

    manager = ExchangeManager("binance")
    manager.load_markets()

    assert manager.fetch_ticker("BTC/USDT") is None  # pierwsza próba kończy się błędem sieci
    counts = manager.get_network_error_counts()
    assert counts.get("fetch_ticker") == 1

    ticker = manager.fetch_ticker("BTC/USDT")
    assert ticker is not None
    assert ticker["last"] == pytest.approx(123.0)

    assert manager.fetch_order_book("BTC/USDT") is None
    counts = manager.get_network_error_counts()
    assert counts.get("fetch_order_book") == 1


def test_binance_spot_adapter_reports_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def handler(endpoint: str, exc: Exception) -> None:
        events.append(endpoint)

    credentials = ExchangeCredentials(key_id="demo", environment=Environment.TESTNET)
    adapter = BinanceSpotAdapter(credentials, network_error_handler=handler)

    def failing_urlopen(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr("bot_core.exchanges.binance.spot.urlopen", failing_urlopen)
    monkeypatch.setattr("bot_core.exchanges.binance.spot.time.sleep", lambda *_: None)

    with pytest.raises(ExchangeNetworkError):
        adapter._public_request("/api/v3/ping")

    assert len(events) == _MAX_RETRIES
    assert all(endpoint == "/api/v3/ping" for endpoint in events)
