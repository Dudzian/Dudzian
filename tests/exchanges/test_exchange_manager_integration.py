from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pytest
import yaml

from bot_core.exchanges import manager as manager_module
from bot_core.exchanges.manager import ExchangeManager, Mode


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


@dataclass
class _StubCCXTClient:
    exchange_id: str
    options: dict[str, Any]

    def __post_init__(self) -> None:
        self.sandbox_requests: list[bool] = []
        self.urls = {
            "api": "https://prod",  # pragma: no mutate - stabilne URL-e do remapowania
            "test": "https://sandbox",
            "fapi": "https://futures",
            "fapiTest": "https://futures-test",
        }

    def setSandboxMode(self, enabled: bool) -> None:  # noqa: N802 - kompatybilność z CCXT
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

    def create_order(self, symbol: str, *_: Any, **__: Any) -> dict[str, Any]:
        return {"id": f"{self.exchange_id}-{symbol}-order", "status": "open", "price": 25_000.0}

    def cancel_order(self, order_id: str, symbol: str | None = None, **_: Any) -> dict[str, Any]:
        return {"order_id": order_id, "symbol": symbol, "status": "cancelled"}


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
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.price_feed_backend = args[0] if args else None
        self.loaded = False

    def load_markets(self) -> dict[str, dict[str, Any]]:
        self.loaded = True
        if self.price_feed_backend and hasattr(self.price_feed_backend, "load_markets"):
            return self.price_feed_backend.load_markets()
        return {}


@pytest.mark.parametrize("config_path", sorted(Path("config/exchanges").glob("*.yaml")))
def test_exchange_profiles_spawn_backends(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for env_name in _collect_env_names(payload):
        monkeypatch.setenv(env_name, f"stub-{env_name.lower()}")

    stub_ccxt = _StubCCXTModule()
    monkeypatch.setattr(manager_module, "ccxt", stub_ccxt)
    monkeypatch.setattr(manager_module, "DatabaseManager", _StubDatabaseManager)
    monkeypatch.setattr(manager_module, "PaperBackend", _StubPaperBackend)
    monkeypatch.setattr(manager_module, "PaperMarginSimulator", _StubPaperBackend)
    monkeypatch.setattr(manager_module, "PaperFuturesSimulator", _StubPaperBackend)

    manager_module._EXCHANGE_PROFILE_CACHE.clear()
    exchange_id = config_path.stem

    testnet_manager = ExchangeManager(exchange_id)
    testnet_manager.apply_environment_profile("testnet", config_dir=config_path.parent)

    assert testnet_manager.mode in {Mode.SPOT, Mode.MARGIN, Mode.FUTURES}
    assert testnet_manager._testnet is True  # pylint: disable=protected-access

    public = testnet_manager._ensure_public()  # pylint: disable=protected-access
    rules = public.load_markets()
    assert "BTC/USDT" in rules

    created_clients = stub_ccxt.created.get(exchange_id)
    assert created_clients is not None and created_clients[0].sandbox_requests == [True]

    private = testnet_manager._ensure_private()  # pylint: disable=protected-access
    assert private.get_market_rules("BTC/USDT") is not None
    assert len(stub_ccxt.created[exchange_id]) >= 2
    assert stub_ccxt.created[exchange_id][1].sandbox_requests == [True]

    paper_manager = ExchangeManager(exchange_id)
    paper_manager.apply_environment_profile("paper", config_dir=config_path.parent)
    assert paper_manager.mode is Mode.PAPER

    paper_backend = paper_manager._ensure_paper()  # pylint: disable=protected-access
    assert isinstance(paper_backend, _StubPaperBackend)
    assert paper_backend.loaded is True
