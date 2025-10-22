"""Testy kontraktowe porównujące adaptery z oczekiwaniami CCXT."""
from __future__ import annotations
import base64
import hashlib
import hmac
import json
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable

import pytest

from ccxt import binance as ccxt_binance
from ccxt import bitstamp as ccxt_bitstamp
from ccxt import bybit as ccxt_bybit
from ccxt import kraken as ccxt_kraken
from ccxt import okx as ccxt_okx
from ccxt import zonda as ccxt_zonda

from KryptoLowca.exchanges.binance import BinanceTestnetAdapter
from KryptoLowca.exchanges.bitstamp import BitstampAdapter
from KryptoLowca.exchanges.bybit import BybitSpotAdapter
from KryptoLowca.exchanges.kraken import KrakenDemoAdapter
from KryptoLowca.exchanges.okx import OKXDerivativesAdapter, OKXMarginAdapter
from KryptoLowca.exchanges.zonda import ZondaAdapter
from KryptoLowca.exchanges.interfaces import ExchangeCredentials, OrderRequest


@dataclass
class _MockResponse:
    payload: Dict[str, Any]
    status_code: int = 200

    async def json(self) -> Dict[str, Any]:
        return self.payload

    @property
    def text(self) -> str:  # pragma: no cover - helper dla logów
        return json.dumps(self.payload)


class _MockHTTPClient:
    def __init__(self, responses: Iterable[_MockResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[Dict[str, Any]] = []

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: Dict[str, Any] | None = None,
        data: Any = None,
        headers: Dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _MockResponse:
        self.requests.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "data": data,
                "headers": headers,
            }
        )
        if not self._responses:
            raise AssertionError("Nieoczekiwane wywołanie HTTP")
        return self._responses.pop(0)

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter_factory, symbol, sample, extractor, ccxt_factory",
    [
        (
            lambda client: BinanceTestnetAdapter(http_client=client),
            "BTCUSDT",
            {
                "symbol": "BTCUSDT",
                "bidPrice": "20191.87000000",
                "bidQty": "0.23200000",
                "askPrice": "20191.88000000",
                "askQty": "0.31500000",
            },
            lambda payload: payload,
            ccxt_binance,
        ),
        (
            lambda client: BitstampAdapter(http_client=client),
            "btcusd",
            {
                "symbol": "btcusd",
                "bid": "20190.0",
                "ask": "20195.0",
                "last": "20192.0",
                "volume": "123.45",
            },
            lambda payload: payload,
            ccxt_bitstamp,
        ),
        (
            lambda client: BybitSpotAdapter(http_client=client),
            "BTCUSDT",
            {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [
                        {
                            "symbol": "BTCUSDT",
                            "bid1Price": "20191.87",
                            "ask1Price": "20191.88",
                            "lastPrice": "20191.86",
                        }
                    ]
                },
            },
            lambda payload: payload["result"]["list"][0],
            ccxt_bybit,
        ),
        (
            lambda client: OKXMarginAdapter(http_client=client),
            "BTC-USDT",
            {
                "code": "0",
                "data": [
                    {
                        "instId": "BTC-USDT",
                        "bidPx": "20190.5",
                        "askPx": "20191.0",
                        "last": "20190.8",
                    }
                ],
            },
            lambda payload: payload["data"][0],
            ccxt_okx,
        ),
        (
            lambda client: OKXDerivativesAdapter(http_client=client),
            "BTC-USDT-SWAP",
            {
                "code": "0",
                "data": [
                    {
                        "instId": "BTC-USDT-SWAP",
                        "bidPx": "20188.5",
                        "askPx": "20189.0",
                        "last": "20188.7",
                    }
                ],
            },
            lambda payload: payload["data"][0],
            ccxt_okx,
        ),
        (
            lambda client: KrakenDemoAdapter(http_client=client),
            "XBT/USDT",
            {
                "error": [],
                "result": {
                    "XBT/USDT": {
                        "a": ["20191.9", "1", "1"],
                        "b": ["20191.7", "1", "1"],
                        "c": ["20191.8", "0.1"],
                    }
                },
            },
            lambda payload: {
                "symbol": "XBT/USDT",
                "bid": float(payload["result"]["XBT/USDT"]["b"][0]),
                "ask": float(payload["result"]["XBT/USDT"]["a"][0]),
                "last": float(payload["result"]["XBT/USDT"]["c"][0]),
            },
            ccxt_kraken,
        ),
        (
            lambda client: ZondaAdapter(http_client=client, enable_streaming=False),
            "BTC-PLN",
            {
                "status": "Ok",
                "ticker": {
                    "highestBid": "20190.1",
                    "lowestAsk": "20191.5",
                    "rate": "20190.9",
                },
            },
            lambda payload: {
                "symbol": "BTC-PLN",
                "bid": float(payload["ticker"]["highestBid"]),
                "ask": float(payload["ticker"]["lowestAsk"]),
                "last": float(payload["ticker"]["rate"]),
            },
            ccxt_zonda,
        ),
    ],
)
async def test_fetch_market_data_matches_ccxt(
    adapter_factory: Callable[[Any], Any],
    symbol: str,
    sample: Dict[str, Any],
    extractor: Callable[[Dict[str, Any]], Dict[str, Any]],
    ccxt_factory: Callable[[], Any],
) -> None:
    http_client = _MockHTTPClient([_MockResponse(sample)])
    adapter = adapter_factory(http_client)
    result = await adapter.fetch_market_data(symbol)

    extracted = extractor(sample)
    expected = ccxt_factory().parse_ticker(extracted, symbol)

    assert result["bid"] == pytest.approx(expected["bid"])
    assert result["ask"] == pytest.approx(expected["ask"])
    assert result["last"] == pytest.approx(expected["last"])
    assert result["raw"] == sample

@pytest.mark.asyncio
async def test_bitstamp_signed_request_adds_headers() -> None:
    sample = {"id": "1", "status": "ok", "filled": "0"}
    client = _MockHTTPClient([_MockResponse(sample)])
    adapter = BitstampAdapter(http_client=client)
    await adapter.authenticate(ExchangeCredentials(api_key="key", api_secret="secret"))
    request = OrderRequest(symbol="btcusd", side="buy", quantity=0.1, order_type="limit", price=20000)
    await adapter.submit_order(request)
    sent_headers = client.requests[0]["headers"]
    assert sent_headers is not None
    assert "X-Auth" in sent_headers and sent_headers["X-Auth"].startswith("BITSTAMP")
    sent_url = client.requests[0]["url"]
    assert sent_url.endswith("/buy/btcusd/")
    encoded_body = client.requests[0]["data"]
    assert isinstance(encoded_body, str)
    parsed_body = urllib.parse.parse_qs(encoded_body)
    assert parsed_body["amount"] == ["0.1"]
    assert parsed_body["price"] == ["20000"]
    assert parsed_body["type"] == ["limit"]


@pytest.mark.asyncio
async def test_bybit_submit_order_signs_request(monkeypatch) -> None:
    sample = {
        "retCode": 0,
        "result": {"orderId": "123", "orderStatus": "NEW", "cumExecQty": "0", "orderQty": "0.1"},
    }
    client = _MockHTTPClient([_MockResponse(sample)])
    adapter = BybitSpotAdapter(http_client=client)
    await adapter.authenticate(ExchangeCredentials(api_key="key", api_secret="secret"))
    fixed_time = 1_700_000_000.123
    monkeypatch.setattr("KryptoLowca.exchanges.bybit.time.time", lambda: fixed_time)

    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.1,
        order_type="limit",
        price=20000,
        client_order_id="abc123",
    )
    await adapter.submit_order(request)

    assert client.requests, "Żądanie HTTP nie zostało wysłane"
    sent = client.requests[0]
    assert sent["url"].endswith("/v5/order/create")
    headers = sent["headers"] or {}
    assert headers["X-BAPI-API-KEY"] == "key"
    assert headers["Content-Type"] == "application/json"
    timestamp = str(int(fixed_time * 1000))
    assert headers["X-BAPI-TIMESTAMP"] == timestamp

    body = json.loads(sent["data"])
    assert body == {
        "category": "spot",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "orderType": "LIMIT",
        "qty": "0.1",
        "price": "20000",
        "orderLinkId": "abc123",
    }
    serialized_body = json.dumps(body, separators=(",", ":"))
    expected_signature = hmac.new(
        b"secret",
        f"{timestamp}key5000{serialized_body}".encode(),
        hashlib.sha256,
    ).hexdigest()
    assert headers["X-BAPI-SIGN"] == expected_signature

@pytest.mark.asyncio
async def test_okx_submit_order_signs_request(monkeypatch) -> None:
    sample = {"code": "0", "data": [{"ordId": "1", "state": "live", "accFillSz": "0"}]}
    client = _MockHTTPClient([_MockResponse(sample)])
    adapter = OKXMarginAdapter(http_client=client)
    await adapter.authenticate(
        ExchangeCredentials(api_key="key", api_secret="secret", passphrase="phrase")
    )

    fixed_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    class _FixedDatetime:
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            assert tz == timezone.utc
            return fixed_dt

    monkeypatch.setattr("KryptoLowca.exchanges.okx.datetime", _FixedDatetime)

    request = OrderRequest(
        symbol="BTC-USDT",
        side="buy",
        quantity=0.1,
        order_type="limit",
        price=20000,
    )
    await adapter.submit_order(request)

    assert client.requests, "Żądanie HTTP nie zostało wysłane"
    sent = client.requests[0]
    assert sent["url"].endswith("/api/v5/trade/order")
    headers = sent["headers"] or {}
    assert headers["OK-ACCESS-KEY"] == "key"
    assert headers["OK-ACCESS-PASSPHRASE"] == "phrase"
    assert headers["Content-Type"] == "application/json"
    timestamp = fixed_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    assert headers["OK-ACCESS-TIMESTAMP"] == timestamp

    body = json.loads(sent["data"])
    assert body == {
        "instId": "BTC-USDT",
        "tdMode": "cross",
        "side": "buy",
        "ordType": "limit",
        "sz": "0.1",
        "px": "20000",
    }
    serialized_body = json.dumps(body, separators=(",", ":"))
    message = f"{timestamp}POST/api/v5/trade/order{serialized_body}"
    expected_signature = base64.b64encode(
        hmac.new(b"secret", message.encode(), hashlib.sha256).digest()
    ).decode()
    assert headers["OK-ACCESS-SIGN"] == expected_signature
