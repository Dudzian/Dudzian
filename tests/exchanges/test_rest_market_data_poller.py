from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pytest
import yaml

from bot_core.exchanges import manager as manager_module
from bot_core.exchanges.manager import Mode
from bot_core.runtime import market_data_service
from bot_core.runtime.market_data_service import RestMarketDataPoller


class _FakeRules:
    def __init__(self) -> None:
        self.price_step = 0.1
        self.amount_step = 0.01
        self.min_notional = 5.0
        self.min_amount = 0.001
        self.max_amount = 5.0
        self.min_price = 10.0
        self.max_price = 20_000.0


class _FakePublic:
    def __init__(self) -> None:
        self._markets = {
            "BTC/USDT": {"base": "BTC", "quote": "USDT", "id": "BTCUSDT"},
        }


class _FakeManager:
    def __init__(self) -> None:
        self._public = _FakePublic()

    def load_markets(self):
        return {"BTC/USDT": _FakeRules()}


def test_rest_market_data_poller_collects_snapshot() -> None:
    manager = _FakeManager()
    poller = RestMarketDataPoller(["binance"], manager_lookup={"BINANCE": manager}, interval=0.01)

    poller.refresh_now()
    snapshot = poller.snapshot("binance")

    assert snapshot
    entry = snapshot[0]
    assert entry["instrument"]["symbol"] == "BTC/USDT"
    assert entry["price_step"] == 0.1
    assert entry["max_amount"] == 5.0


def test_rest_market_data_poller_applies_environment_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    applied: list[tuple[str, str | None, Any | None]] = []

    class _Manager:
        def __init__(self, exchange_id: str) -> None:
            self.exchange_id = exchange_id
            self._public = _FakePublic()

        def apply_environment_profile(
            self,
            name: str,
            *,
            config_dir: str | None = None,
            overrides: Any | None = None,
        ) -> None:
            applied.append((name, config_dir, overrides))

        def load_markets(self):  # noqa: D401 - kompatybilność z pollerem
            return {}

    monkeypatch.setattr(market_data_service, "ExchangeManager", _Manager)

    poller = RestMarketDataPoller(["binance"], interval=0.01, profile="paper")
    poller.refresh_now()

    assert applied == [("paper", None, None)]


def _collect_env_names(payload: object) -> Iterable[str]:
    if isinstance(payload, str):
        if payload.startswith("${") and payload.endswith("}") and len(payload) > 3:
            yield payload[2:-1]
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from _collect_env_names(value)
    elif isinstance(payload, (list, tuple, set)):
        for item in payload:
            yield from _collect_env_names(item)


class _StubCCXTClient:
    def __init__(self, exchange_id: str, options: dict[str, Any]) -> None:
        self.exchange_id = exchange_id
        self.options = options
        self.sandbox_requests: list[bool] = []
        self.urls = {
            "api": "https://prod",
            "test": "https://sandbox",
        }

    def setSandboxMode(self, enabled: bool) -> None:  # noqa: N802 - konwencja CCXT
        self.sandbox_requests.append(bool(enabled))

    def load_markets(self) -> dict[str, dict[str, Any]]:
        return {
            "BTC/USDT": {
                "limits": {
                    "amount": {"min": 0.001, "max": 250.0, "step": 0.001},
                    "price": {"min": 5.0, "max": 250000.0, "step": 0.01},
                    "cost": {"min": 10.0},
                },
                "precision": {"amount": 3, "price": 2},
                "symbol": "BTC/USDT",
                "base": "BTC",
                "quote": "USDT",
                "id": "BTCUSDT",
            }
        }

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        return {"symbol": symbol, "last": 25_000.0, "close": 25_000.0}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500, **_: Any):
        base = 1_700_000_000_000
        return [
            [base, 25_000.0, 25_050.0, 24_950.0, 25_010.0, 10.0],
            [base + 60_000, 25_010.0, 25_100.0, 25_000.0, 25_050.0, 11.0],
        ][:limit]

    def fetch_order_book(self, symbol: str, limit: int = 50):
        return {
            "symbol": symbol,
            "bids": [[25_000.0, 1.2]] * min(limit, 1),
            "asks": [[25_010.0, 1.0]] * min(limit, 1),
        }


class _StubCCXTModule:
    NetworkError = (RuntimeError,)

    def __init__(self) -> None:
        self.created: dict[str, list[_StubCCXTClient]] = {}

    def __getattr__(self, name: str):  # noqa: D401 - zachowanie modułu CCXT
        def _factory(options: dict[str, Any]) -> _StubCCXTClient:
            client = _StubCCXTClient(name, options)
            self.created.setdefault(name, []).append(client)
            return client

        return _factory


class _StubDatabaseManager:
    def __init__(self, url: str) -> None:
        self.url = url
        self.sync = self

    def init_db(self) -> None:  # pragma: no cover - brak logiki
        return None


class _StubPaperBackend:
    created: list["_StubPaperBackend"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.price_feed_backend = args[0] if args else None
        self.loaded = False
        _StubPaperBackend.created.append(self)

    def load_markets(self) -> dict[str, dict[str, Any]]:
        self.loaded = True
        if self.price_feed_backend and hasattr(self.price_feed_backend, "load_markets"):
            return self.price_feed_backend.load_markets()
        return {}


@pytest.mark.parametrize("profile", ["testnet", "paper"])
def test_rest_market_data_poller_integrates_with_exchange_profiles(
    profile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = Path("config/exchanges")
    exchanges: list[str] = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        for env_name in _collect_env_names(payload):
            monkeypatch.setenv(env_name, f"stub-{env_name.lower()}")
        exchanges.append(config_path.stem)

    stub_ccxt = _StubCCXTModule()
    monkeypatch.setattr(manager_module, "ccxt", stub_ccxt)
    monkeypatch.setattr(manager_module, "DatabaseManager", _StubDatabaseManager)
    monkeypatch.setattr(manager_module, "PaperBackend", _StubPaperBackend)
    monkeypatch.setattr(manager_module, "PaperMarginSimulator", _StubPaperBackend)
    monkeypatch.setattr(manager_module, "PaperFuturesSimulator", _StubPaperBackend)

    _StubPaperBackend.created.clear()
    manager_module._EXCHANGE_PROFILE_CACHE.clear()

    poller = RestMarketDataPoller(
        exchanges,
        profile=profile,
        config_dir=config_dir,
        interval=0.01,
    )
    poller.refresh_now()

    for exchange in exchanges:
        snapshot = poller.snapshot(exchange)
        assert snapshot, f"brak danych dla {exchange}"
        instrument = snapshot[0]["instrument"]
        assert instrument["exchange"] == exchange.upper()
        assert instrument["symbol"], f"brak symbolu dla {exchange}"

        manager = poller._manager_provider(exchange.upper())  # pylint: disable=protected-access
        assert manager is not None
        if profile == "paper":
            assert manager.mode is Mode.PAPER
        else:
            assert manager.mode is not Mode.PAPER

        if profile == "paper":
            backend = manager._ensure_paper()  # type: ignore[attr-defined]  # pylint: disable=protected-access
            assert isinstance(backend, _StubPaperBackend)
            assert backend.loaded is True
        elif profile == "testnet":
            clients = stub_ccxt.created.get(exchange)
            assert clients is not None and clients[0].sandbox_requests == [True]
