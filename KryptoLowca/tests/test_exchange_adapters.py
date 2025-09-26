from __future__ import annotations

import types

import pytest

from KryptoLowca.exchanges import AdapterError, create_exchange_adapter


@pytest.mark.asyncio
async def test_ccxt_adapter_connects(monkeypatch):
    class DummyExchange:
        def __init__(self, options):
            self.options = options
            self.closed = False
            self.sandbox = False

        async def set_sandbox_mode(self, enabled: bool) -> None:
            self.sandbox = enabled

        async def close(self) -> None:
            self.closed = True

    module = types.SimpleNamespace(binance=DummyExchange)
    monkeypatch.setattr("KryptoLowca.exchanges.adapters.ccxt_async", module, raising=False)

    adapter = create_exchange_adapter("binance", api_key="k", api_secret="s", sandbox=True)
    client = await adapter.connect()

    assert isinstance(client, DummyExchange)
    assert client.options["apiKey"] == "k"
    assert client.sandbox is True

    await adapter.close()
    assert client.closed is True


def test_unknown_adapter_raises():
    with pytest.raises(AdapterError):
        create_exchange_adapter("nonexistent")
