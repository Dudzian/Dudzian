from __future__ import annotations

from typing import Any

import pytest

from bot_core.exchanges import manager


class _ClientWithSandbox:
    def __init__(self) -> None:
        self.enabled: list[bool] = []

    def setSandboxMode(self, enabled: bool) -> None:  # noqa: N802 - zgodne z CCXT
        self.enabled.append(bool(enabled))


class _ClientWithUrls:
    def __init__(self) -> None:
        self.urls = {
            "api": "https://prod",
            "test": "https://sandbox",
            "fapi": "https://futures",
            "fapiTest": "https://futures-test",
        }


def test_enable_sandbox_mode_prefers_method_call() -> None:
    client = _ClientWithSandbox()

    assert manager._enable_sandbox_mode(client) is True
    assert client.enabled == [True]


def test_enable_sandbox_mode_remaps_urls_when_method_missing() -> None:
    client = _ClientWithUrls()

    assert manager._enable_sandbox_mode(client) is True
    assert client.urls["api"] == "https://sandbox"
    assert client.urls["fapi"] == "https://futures-test"


def test_public_feed_uses_sandbox_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[Any] = []

    class _Module:
        def binance(self, options: dict[str, Any]) -> _ClientWithSandbox:  # type: ignore[override]
            client = _ClientWithSandbox()
            client.options = options  # type: ignore[attr-defined]
            created.append(client)
            return client

    monkeypatch.setattr(manager, "ccxt", _Module())

    feed = manager._CCXTPublicFeed(exchange_id="binance", testnet=True)

    assert created and created[0].enabled == [True]
    assert created[0].options["options"]["defaultType"] == "spot"
    # upewnij się, że klient został ustawiony na feedzie
    assert feed.client is created[0]


def test_public_feed_accepts_margin_default_type(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[Any] = []

    class _Client:
        def __init__(self, options: dict[str, Any]) -> None:
            self.options = options

    class _Module:
        def binance(self, options: dict[str, Any]) -> _Client:  # type: ignore[override]
            created.append(options)
            return _Client(options)

    monkeypatch.setattr(manager, "ccxt", _Module())

    feed = manager._CCXTPublicFeed(exchange_id="binance", market_type="margin")

    assert created
    assert created[0]["options"]["defaultType"] == "margin"
    assert feed.market_type == "margin"
    assert feed.futures is False


def test_private_backend_reenables_sandbox_after_reinit(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[_ClientWithSandbox] = []

    class _Module:
        def __init__(self) -> None:
            self.calls = 0

        def binance(self, options: dict[str, Any]) -> _ClientWithSandbox:  # type: ignore[override]
            self.calls += 1
            client = _ClientWithSandbox()
            client.options = options  # type: ignore[attr-defined]
            created.append(client)
            return client

    module = _Module()
    monkeypatch.setattr(manager, "ccxt", module)

    backend = manager._CCXTPrivateBackend(
        exchange_id="binance",
        testnet=True,
        futures=True,
        api_key="key",
        secret="secret",
    )

    # pierwszy klient pochodzi z publicznego konstruktora, drugi to prywatny backend
    assert module.calls == 2
    assert created[0].enabled == [True]
    assert created[1].enabled == [True]
    assert created[0].options["options"]["defaultType"] == "future"
    assert created[1].options["options"]["defaultType"] == "future"
    assert backend.client is created[1]

def test_private_backend_accepts_margin_default_type(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, object]] = []

    class _Client:
        def __init__(self, options: dict[str, object]) -> None:
            self.options = options

    class _Module:
        def binance(self, options: dict[str, object]) -> _Client:  # type: ignore[override]
            created.append(options)
            return _Client(options)

    monkeypatch.setattr(manager, "ccxt", _Module())

    backend = manager._CCXTPrivateBackend(
        exchange_id="binance",
        market_type="margin",
        api_key="key",
        secret="secret",
    )

    assert created
    assert created[-1]["options"]["defaultType"] == "margin"
    assert backend.market_type == "margin"
    assert backend.mode is manager.Mode.MARGIN

