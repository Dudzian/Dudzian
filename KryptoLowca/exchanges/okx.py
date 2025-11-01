"""OKX adapters using REST endpoints and long-polling."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import urllib.parse
from datetime import datetime, timezone
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Dict, Optional

from .interfaces import (
    ExchangeCredentials,
    MarketPayload,
    MarketStreamHandle,
    MarketSubscription,
    OrderRequest,
    OrderStatus,
    RESTStreamingAdapter,
)
from .streaming import LongPollSubscription


logger = logging.getLogger(__name__)


class _OKXBaseAdapter(RESTStreamingAdapter):
    """Bazowa implementacja adapterów OKX z long-pollingiem."""

    def __init__(
        self,
        *,
        name: str,
        inst_type: str,
        default_td_mode: str,
        demo_mode: bool,
        http_client=None,
        stream_poll_interval: float = 1.0,
        compliance_ack: bool,
    ) -> None:
        super().__init__(
            name=name,
            base_url="https://www.okx.com",
            demo_mode=demo_mode,
            http_client=http_client,
            compliance_ack=compliance_ack,
        )
        if stream_poll_interval <= 0:
            raise ValueError("stream_poll_interval musi być dodatni")
        self._stream_poll_interval = float(stream_poll_interval)
        self._inst_type = inst_type
        self._default_td_mode = default_td_mode

    async def authenticate(self, credentials: ExchangeCredentials) -> None:
        if not credentials.api_key or not credentials.api_secret or not credentials.passphrase:
            raise ValueError("OKX wymaga klucza, sekretu oraz passphrase")
        await super().authenticate(credentials)

    async def fetch_market_data(self, symbol: str) -> MarketPayload:
        params = {"instId": symbol, "instType": self._inst_type}
        response: Dict[str, Any] = await self._request("GET", "/api/v5/market/ticker", params=params)
        if str(response.get("code")) not in ("0", "None", ""):
            raise RuntimeError(f"OKX API error {response.get('code')}: {response.get('msg')}")
        data = response.get("data") or []
        ticker = data[0] if data else {}
        bid = float(ticker.get("bidPx", 0.0))
        ask = float(ticker.get("askPx", 0.0))
        last = float(ticker.get("last", 0.0))
        payload: MarketPayload = {
            "symbol": ticker.get("instId", symbol),
            "bid": bid,
            "ask": ask,
            "last": last,
            "raw": response,
        }
        payload.update(response)
        payload["bid"] = bid
        payload["ask"] = ask
        payload["last"] = last
        return payload

    async def stream_market_data(
        self, subscriptions: Iterable[MarketSubscription], callback: Callable[[MarketPayload], Awaitable[None]]
    ) -> MarketStreamHandle:
        return LongPollSubscription(
            self,
            subscriptions,
            callback,
            default_interval=self._stream_poll_interval,
        )

    async def submit_order(self, order: OrderRequest) -> OrderStatus:
        body: Dict[str, Any] = {
            "instId": order.symbol,
            "tdMode": order.extra_params.get("tdMode", self._default_td_mode),
            "side": order.side.lower(),
            "ordType": order.order_type.lower(),
            "sz": str(order.quantity),
        }
        if order.price is not None:
            body["px"] = str(order.price)
        if order.client_order_id:
            body["clOrdId"] = order.client_order_id
        body.update({k: v for k, v in order.extra_params.items() if k not in {"tdMode"}})
        response = await self._private_request("POST", "/api/v5/trade/order", data=body)
        order_payload = (response.get("data") or [{}])[0]
        return self._parse_order(order_payload, default_status="NEW", raw=response)

    async def fetch_order_status(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        params = {"ordId": order_id}
        if symbol:
            params["instId"] = symbol
        response = await self._private_request("GET", "/api/v5/trade/order", params=params)
        order_payload = (response.get("data") or [{}])[0]
        if not order_payload:
            order_payload = {"ordId": order_id, "state": "unknown"}
        return self._parse_order(order_payload, default_status="UNKNOWN", raw=response)

    async def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        body: Dict[str, Any] = {"ordId": order_id}
        if symbol:
            body["instId"] = symbol
        response = await self._private_request("POST", "/api/v5/trade/cancel-order", data=body)
        order_payload = (response.get("data") or [{}])[0]
        if not order_payload:
            order_payload = {"ordId": order_id, "state": "canceled"}
        return self._parse_order(order_payload, default_status="CANCELED", raw=response)

    async def _private_request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._credentials:
            raise RuntimeError("Brak poświadczeń – wywołaj authenticate()")
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        body = ""
        if data and method.upper() != "GET":
            body = json.dumps(data, separators=(",", ":"))
        query = ""
        if params:
            query = "?" + urllib.parse.urlencode(params, doseq=True)
        prehash = f"{timestamp}{method.upper()}{path}{query}{body}"
        signature = base64.b64encode(
            hmac.new(
                self._credentials.api_secret.encode(), prehash.encode(), hashlib.sha256
            ).digest()
        ).decode()
        headers = {
            "OK-ACCESS-KEY": self._credentials.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self._credentials.passphrase or "",
            "Content-Type": "application/json",
        }
        response: Dict[str, Any] = await self._request(
            method,
            path + (query if method.upper() == "GET" and query else ""),
            data=body if body else None,
            headers=headers,
        )
        if str(response.get("code")) not in ("0", "None", ""):
            raise RuntimeError(f"OKX API error {response.get('code')}: {response.get('msg')}")
        return response

    def _parse_order(
        self,
        payload: Dict[str, Any],
        *,
        default_status: str,
        raw: Dict[str, Any],
    ) -> OrderStatus:
        order_id = str(payload.get("ordId") or payload.get("orderId") or "")
        status = str(payload.get("state") or default_status).upper()
        filled = float(payload.get("accFillSz", 0.0))
        qty = float(payload.get("sz", filled))
        avg_price = payload.get("avgPx") or payload.get("px")
        return OrderStatus(
            order_id=order_id,
            status=status,
            filled_quantity=filled,
            remaining_quantity=max(qty - filled, 0.0),
            average_price=float(avg_price) if avg_price not in (None, "") else None,
            raw=raw,
        )


class OKXMarginAdapter(_OKXBaseAdapter):
    """Adapter obsługujący rynek margin OKX."""

    def __init__(
        self,
        *,
        demo_mode: bool = True,
        http_client=None,
        stream_poll_interval: float = 1.0,
        compliance_ack: bool = False,
    ) -> None:
        super().__init__(
            name="okx-margin",
            inst_type="MARGIN",
            default_td_mode="cross",
            demo_mode=demo_mode,
            http_client=http_client,
            stream_poll_interval=stream_poll_interval,
            compliance_ack=compliance_ack,
        )


class OKXDerivativesAdapter(_OKXBaseAdapter):
    """Adapter obsługujący instrumenty pochodne (perpetual/futures) OKX."""

    def __init__(
        self,
        *,
        demo_mode: bool = True,
        http_client=None,
        stream_poll_interval: float = 1.0,
        compliance_ack: bool = False,
        inst_type: str = "SWAP",
    ) -> None:
        super().__init__(
            name="okx-derivatives",
            inst_type=inst_type,
            default_td_mode="cross",
            demo_mode=demo_mode,
            http_client=http_client,
            stream_poll_interval=stream_poll_interval,
            compliance_ack=compliance_ack,
        )


__all__ = ["OKXMarginAdapter", "OKXDerivativesAdapter"]
