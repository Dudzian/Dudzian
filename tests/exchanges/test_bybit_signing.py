from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Mapping

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.bybit import BybitSpotAdapter


class _StubBybitClient:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.sandbox_enabled = False
        self.fixed_timestamp = "1700000000000"
        self.recv_window = str(config.get("recvWindow", "5000"))

    def set_sandbox_mode(self, enabled: bool) -> None:
        self.sandbox_enabled = bool(enabled)

    def sign(
        self,
        path: str,
        api: str,
        method: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        del api, headers  # nieużywane w stubie
        params = params or {}
        body = body or {}
        method_upper = method.upper()
        if method_upper == "GET":
            serialized = "&".join(
                f"{key}={value}" for key, value in sorted(params.items(), key=lambda item: item[0])
            )
            payload = serialized
        else:
            payload = json.dumps(body or params, separators=(",", ":"), sort_keys=True)
        message = (
            f"{self.fixed_timestamp}{self.config.get('apiKey')}" f"{self.recv_window}{payload or ''}"
        )
        signature = hmac.new(
            str(self.config["secret"]).encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "url": f"https://api.bybit.com{path}",
            "method": method_upper,
            "body": payload if method_upper != "GET" else None,
            "headers": {
                "X-BAPI-API-KEY": self.config.get("apiKey"),
                "X-BAPI-TIMESTAMP": self.fixed_timestamp,
                "X-BAPI-RECV-WINDOW": self.recv_window,
                "X-BAPI-SIGN": signature,
            },
        }


class _CCXTModule:
    def __init__(self) -> None:
        self.created_clients: list[_StubBybitClient] = []

    def bybit(self, config: Mapping[str, Any]) -> _StubBybitClient:  # type: ignore[override]
        client = _StubBybitClient(config)
        self.created_clients.append(client)
        return client


@pytest.fixture()
def _patch_ccxt(monkeypatch: pytest.MonkeyPatch) -> _CCXTModule:
    module = _CCXTModule()
    monkeypatch.setattr("bot_core.exchanges.ccxt_adapter.ccxt", module)
    return module


def test_bybit_adapter_populates_credentials_and_signs_request(
    _patch_ccxt: _CCXTModule,
) -> None:
    credentials = ExchangeCredentials(
        key_id="bybit-key",
        secret="super-secret",  # API sekrety Bybit nie są kodowane base64
        environment=Environment.PAPER,
        permissions=("trade", "read"),
    )

    adapter = BybitSpotAdapter(credentials=credentials, environment=Environment.PAPER)

    assert _patch_ccxt.created_clients, "Adapter powinien zainicjować klienta CCXT"
    client = _patch_ccxt.created_clients[-1]

    assert client.config["apiKey"] == "bybit-key"
    assert client.config["secret"] == "super-secret"
    assert client.sandbox_enabled is True
    assert client.recv_window == "5000"

    payload = {
        "category": "spot",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderType": "Limit",
        "qty": "0.1",
        "price": "27000",
    }
    client.fixed_timestamp = "1700001234567"

    signed = client.sign("/v5/order/create", "private", "POST", params={}, body=payload)

    assert signed["headers"]["X-BAPI-API-KEY"] == "bybit-key"
    assert signed["headers"]["X-BAPI-RECV-WINDOW"] == "5000"
    assert signed["method"] == "POST"

    canonical_body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    expected_message = f"{client.fixed_timestamp}bybit-key5000{canonical_body}"
    expected_signature = hmac.new(
        b"super-secret",
        expected_message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert signed["headers"]["X-BAPI-SIGN"] == expected_signature
