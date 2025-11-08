from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any, Mapping

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.okx import OKXSpotAdapter


class _StubOKXClient:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.sandbox_enabled = False
        self.fixed_timestamp = "2024-02-03T04:05:06.789Z"

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
        query_fragment = ""
        if params:
            serialized_params = "&".join(
                f"{key}={value}" for key, value in sorted(params.items(), key=lambda item: item[0])
            )
            query_fragment = f"?{serialized_params}"
        body_payload = json.dumps(body, separators=(",", ":"), sort_keys=True) if body else ""
        message = f"{self.fixed_timestamp}{method_upper}{path}{query_fragment}{body_payload}"
        signature = base64.b64encode(
            hmac.new(
                str(self.config["secret"]).encode("utf-8"),
                message.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        return {
            "url": f"https://www.okx.com{path}{query_fragment}",
            "method": method_upper,
            "body": body_payload or None,
            "headers": {
                "OK-ACCESS-KEY": self.config.get("apiKey"),
                "OK-ACCESS-PASSPHRASE": self.config.get("password"),
                "OK-ACCESS-TIMESTAMP": self.fixed_timestamp,
                "OK-ACCESS-SIGN": signature,
            },
        }


class _CCXTModule:
    def __init__(self) -> None:
        self.created_clients: list[_StubOKXClient] = []

    def okx(self, config: Mapping[str, Any]) -> _StubOKXClient:  # type: ignore[override]
        client = _StubOKXClient(config)
        self.created_clients.append(client)
        return client


@pytest.fixture()
def _patch_ccxt(monkeypatch: pytest.MonkeyPatch) -> _CCXTModule:
    module = _CCXTModule()
    monkeypatch.setattr("bot_core.exchanges.ccxt_adapter.ccxt", module)
    return module


def test_okx_adapter_populates_credentials_and_signs_request(
    _patch_ccxt: _CCXTModule,
) -> None:
    credentials = ExchangeCredentials(
        key_id="okx-key",
        secret="super-secret",
        passphrase="okx-pass",
        environment=Environment.PAPER,
        permissions=("trade", "read"),
    )

    adapter = OKXSpotAdapter(credentials=credentials, environment=Environment.PAPER)

    assert _patch_ccxt.created_clients, "Adapter powinien zainicjować klienta CCXT"
    client = _patch_ccxt.created_clients[-1]

    assert client.config["apiKey"] == "okx-key"
    assert client.config["secret"] == "super-secret"
    assert client.config["password"] == "okx-pass"
    assert client.sandbox_enabled is True

    payload = {
        "instId": "BTC-USDT",
        "tdMode": "cash",
        "side": "buy",
        "ordType": "limit",
        "sz": "0.10",
        "px": "27000",
    }
    client.fixed_timestamp = "2024-06-01T10:11:12.123Z"

    signed = client.sign("/api/v5/trade/order", "private", "POST", params={}, body=payload)

    assert signed["headers"]["OK-ACCESS-KEY"] == "okx-key"
    assert signed["headers"]["OK-ACCESS-PASSPHRASE"] == "okx-pass"
    assert signed["method"] == "POST"

    canonical_body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    expected_message = f"{client.fixed_timestamp}POST/api/v5/trade/order{canonical_body}"
    expected_signature = base64.b64encode(
        hmac.new(b"super-secret", expected_message.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")
    assert signed["headers"]["OK-ACCESS-SIGN"] == expected_signature
