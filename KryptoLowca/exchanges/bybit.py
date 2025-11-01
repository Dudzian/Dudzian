"""Bybit spot adapter using REST endpoints and long-polling."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import urllib.parse
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


class BybitSpotAdapter(RESTStreamingAdapter):
    """Adapter REST dla rynku spot Bybit v5 oparty o long-poll."""

    def __init__(
        self,
        *,
        demo_mode: bool = True,
        http_client=None,
        stream_poll_interval: float = 1.0,
        compliance_ack: bool = False,
    ) -> None:
        super().__init__(
            name="bybit-spot",
            base_url="https://api.bybit.com",
            demo_mode=demo_mode,
            http_client=http_client,
            compliance_ack=compliance_ack,
        )
        if stream_poll_interval <= 0:
            raise ValueError("stream_poll_interval musi być dodatni")
        self._stream_poll_interval = float(stream_poll_interval)

    async def authenticate(self, credentials: ExchangeCredentials) -> None:
        if not credentials.api_key or not credentials.api_secret:
            raise ValueError("Bybit wymaga klucza API i sekretu")
        await super().authenticate(credentials)

    async def fetch_market_data(self, symbol: str) -> MarketPayload:
        params = {"category": "spot", "symbol": symbol}
        response: Dict[str, Any] = await self._request("GET", "/v5/market/tickers", params=params)
        if response.get("retCode") not in (0, "0", None):
            raise RuntimeError(
                f"Bybit API error {response.get('retCode')}: {response.get('retMsg')}"
            )
        result = response.get("result", {})
        tickers = result.get("list") or []
        ticker = tickers[0] if tickers else {}
        bid = float(ticker.get("bid1Price", 0.0))
        ask = float(ticker.get("ask1Price", 0.0))
        last = float(ticker.get("lastPrice", 0.0))
        payload: MarketPayload = {
            "symbol": ticker.get("symbol", symbol),
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
            "category": "spot",
            "symbol": order.symbol,
            "side": order.side.upper(),
            "orderType": order.order_type.upper(),
            "qty": str(order.quantity),
        }
        if order.price is not None:
            body["price"] = str(order.price)
        if order.client_order_id:
            body["orderLinkId"] = order.client_order_id
        if order.time_in_force:
            body["timeInForce"] = order.time_in_force
        body.update({k: v for k, v in order.extra_params.items()})
        response = await self._private_request("POST", "/v5/order/create", data=body)
        result = response.get("result", {})
        order_id = result.get("orderId") or ""
        status = result.get("orderStatus") or response.get("retExtInfo", {}).get("orderStatus") or "NEW"
        avg_price = result.get("avgPrice") or result.get("price")
        filled = float(result.get("cumExecQty", 0.0))
        size = float(result.get("orderQty", filled))
        return OrderStatus(
            order_id=order_id,
            status=str(status).upper(),
            filled_quantity=filled,
            remaining_quantity=max(size - filled, 0.0),
            average_price=float(avg_price) if avg_price not in (None, "") else None,
            raw=response,
        )

    async def fetch_order_status(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        body: Dict[str, Any] = {"category": "spot", "orderId": order_id}
        if symbol:
            body["symbol"] = symbol
        response = await self._private_request("POST", "/v5/order/list", data=body)
        result = response.get("result", {})
        orders = result.get("list") or []
        order_payload = orders[0] if orders else {}
        avg_price = order_payload.get("avgPrice") or order_payload.get("price")
        filled = float(order_payload.get("cumExecQty", 0.0))
        qty = float(order_payload.get("orderQty", filled))
        return OrderStatus(
            order_id=order_payload.get("orderId", order_id),
            status=str(order_payload.get("orderStatus", "UNKNOWN")).upper(),
            filled_quantity=filled,
            remaining_quantity=max(qty - filled, 0.0),
            average_price=float(avg_price) if avg_price not in (None, "") else None,
            raw=response,
        )

    async def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        body: Dict[str, Any] = {"category": "spot", "orderId": order_id}
        if symbol:
            body["symbol"] = symbol
        response = await self._private_request("POST", "/v5/order/cancel", data=body)
        result = response.get("result", {})
        status = result.get("orderStatus") or "CANCELED"
        filled = float(result.get("cumExecQty", 0.0))
        qty = float(result.get("orderQty", filled))
        avg_price = result.get("avgPrice") or result.get("price")
        return OrderStatus(
            order_id=result.get("orderId", order_id),
            status=str(status).upper(),
            filled_quantity=filled,
            remaining_quantity=max(qty - filled, 0.0),
            average_price=float(avg_price) if avg_price not in (None, "") else None,
            raw=response,
        )

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
        timestamp = str(int(time.time() * 1000))
        recv_window = str(int(self._credentials.metadata.get("recv_window", 5000)))
        serialized_params = ""
        if params:
            serialized_params = urllib.parse.urlencode(params, doseq=True)
        serialized_body = ""
        if method.upper() != "GET":
            serialized_body = json.dumps(data or {}, separators=(",", ":"))
        message = (
            f"{timestamp}{self._credentials.api_key}{recv_window}"
            f"{serialized_params if method.upper() == 'GET' else serialized_body}"
        )
        signature = hmac.new(
            self._credentials.api_secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        headers = {
            "X-BAPI-API-KEY": self._credentials.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }
        response: Dict[str, Any] = await self._request(
            method,
            path,
            params=params,
            data=serialized_body if serialized_body else None,
            headers=headers,
        )
        if response.get("retCode") not in (0, "0"):
            raise RuntimeError(
                f"Bybit API error {response.get('retCode')}: {response.get('retMsg')}"
            )
        return response


__all__ = ["BybitSpotAdapter"]
