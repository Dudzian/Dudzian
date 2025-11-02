"""Regresje strażnika sieci dla adapterów CCXT."""
from __future__ import annotations

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.ccxt_adapter import CCXTSpotAdapter
from bot_core.exchanges.errors import ExchangeNetworkError
from bot_core.exchanges.network_guard import NetworkAccessViolation


class _StubClient:
    """Minimalny klient CCXT używany w testach strażnika."""

    def __init__(self) -> None:
        self.load_markets_calls = 0

    def load_markets(self):  # noqa: D401 - prosty stub
        self.load_markets_calls += 1
        return {"BTC/USDT": {}}


def _build_adapter(client: _StubClient | None = None) -> CCXTSpotAdapter:
    credentials = ExchangeCredentials(key_id="key", secret="secret")
    return CCXTSpotAdapter(
        credentials,
        exchange_id="stub",
        environment=Environment.PAPER,
        settings={"sandbox_mode": False},
        client=client or _StubClient(),
    )


def test_ccxt_adapter_requires_configure_network() -> None:
    adapter = _build_adapter()

    with pytest.raises(ExchangeNetworkError) as excinfo:
        adapter.fetch_symbols()

    assert isinstance(excinfo.value.reason, NetworkAccessViolation)
    assert excinfo.value.reason.reason == "network_not_configured"


def test_ccxt_adapter_allows_calls_after_configure() -> None:
    client = _StubClient()
    adapter = _build_adapter(client)
    adapter.configure_network(ip_allowlist=())

    symbols = adapter.fetch_symbols()

    assert symbols == ("BTC/USDT",)
    assert client.load_markets_calls == 1


def test_ccxt_adapter_detects_proxy_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _StubClient()
    adapter = _build_adapter(client)

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:8080")
    adapter.configure_network(ip_allowlist=("127.0.0.1",))

    # Zmiana konfiguracji proxy po configure_network powinna zostać wykryta.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9090")

    with pytest.raises(ExchangeNetworkError) as excinfo:
        adapter.fetch_symbols()

    assert isinstance(excinfo.value.reason, NetworkAccessViolation)
    assert excinfo.value.reason.reason == "proxy_configuration_changed"

    # Przywrócenie pierwotnej konfiguracji umożliwia dalsze połączenia.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:8080")
    symbols = adapter.fetch_symbols()

    assert symbols == ("BTC/USDT",)
    assert client.load_markets_calls == 1

