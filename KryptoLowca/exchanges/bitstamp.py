"""Bitstamp adapter relying on REST endpoints and long-polling."""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
import uuid
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


class BitstampAdapter(RESTStreamingAdapter):
    """Adapter REST dla Bitstamp z obsługą streamingu przez long-poll."""

    def __init__(
        self,
        *,
        demo_mode: bool = True,
        http_client=None,
        stream_poll_interval: float = 1.0,
        compliance_ack: bool = False,
    ) -> None:
        super().__init__(
            name="bitstamp",
            base_url="https://www.bitstamp.net/api/v2",
            demo_mode=demo_mode,
            http_client=http_client,
            compliance_ack=compliance_ack,
        )
        if stream_poll_interval <= 0:
            raise ValueError("stream_poll_interval musi być dodatni")
        self._stream_poll_interval = float(stream_poll_interval)

    async def authenticate(self, credentials: ExchangeCredentials) -> None:
        if not credentials.api_key or not credentials.api_secret:
            raise ValueError("Bitstamp wymaga klucza API i sekretu")
        await super().authenticate(credentials)

    async def fetch_market_data(self, symbol: str) -> MarketPayload:
        normalized_symbol = symbol.lower()
        path = f"/ticker/{normalized_symbol}/"
        response: Dict[str, Any] = await self._request("GET", path)
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(f"Bitstamp API error: {response.get('error')}")
        bid = float(response.get("bid", 0.0))
        ask = float(response.get("ask", 0.0))
        last = float(response.get("last", 0.0))
        volume = float(response.get("volume", 0.0))
        payload: MarketPayload = {
            "symbol": response.get("pair", symbol).upper(),
            "bid": bid,
            "ask": ask,
            "last": last,
            "volume": volume,
            "raw": response,
        }
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
        side = order.side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError("Bitstamp wspiera strony 'buy' oraz 'sell'")
        endpoint = f"/{side}/{order.symbol.lower()}/"
        data: Dict[str, Any] = {
            "amount": str(order.quantity),
            "type": order.order_type.lower(),
        }
        if order.price is not None:
            data["price"] = str(order.price)
        if order.client_order_id:
            data["client_order_id"] = order.client_order_id
        if order.time_in_force:
            data["time_in_force"] = order.time_in_force
        data.update({k: v for k, v in order.extra_params.items()})
        response = await self._signed_request("POST", endpoint, data=data)
        payload = self._extract_order_payload(response)
        return self._order_status_from_payload(payload, raw=response, default_status="NEW")

    async def fetch_order_status(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        del symbol
        data = {"id": order_id}
        response = await self._signed_request("POST", "/order_status/", data=data)
        payload = self._extract_order_payload(response)
        return self._order_status_from_payload(payload, raw=response, default_status="UNKNOWN")

    async def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        del symbol
        data = {"id": order_id}
        response = await self._signed_request("POST", "/cancel_order/", data=data)
        payload = self._extract_order_payload(response)
        return self._order_status_from_payload(payload, raw=response, default_status="CANCELED")

    async def _signed_request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._credentials:
            raise RuntimeError("Brak poświadczeń – wywołaj authenticate()")
        body = urllib.parse.urlencode(data or {}, doseq=True)
        timestamp = str(int(time.time() * 1000))
        nonce = uuid.uuid4().hex
        content_type = "application/x-www-form-urlencoded"
        payload_hash = hashlib.sha256(body.encode()).hexdigest()
        host = urllib.parse.urlparse(self.base_url).netloc
        message = (
            f"BITSTAMP {self._credentials.api_key}{method.upper()}{host}{path}"
            f"{content_type}{timestamp}{nonce}v2{payload_hash}"
        )
        signature = hmac.new(
            self._credentials.api_secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        headers = {
            "X-Auth": f"BITSTAMP {self._credentials.api_key}",
            "X-Auth-Signature": signature,
            "X-Auth-Nonce": nonce,
            "X-Auth-Timestamp": timestamp,
            "X-Auth-Version": "v2",
            "X-Auth-Content-Type": content_type,
            "X-Auth-Content-Hash": payload_hash,
            "Content-Type": content_type,
        }
        response: Dict[str, Any] = await self._request(
            method,
            path,
            params=params,
            data=body if body else None,
            headers=headers,
        )
        if isinstance(response, dict) and response.get("status") == "error":
            raise RuntimeError(
                f"Bitstamp API error: {response.get('reason') or response.get('message')}"
            )
        return response

    def _extract_order_payload(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if "data" in response and isinstance(response["data"], dict):
            return response["data"]
        return response

    def _order_status_from_payload(
        self,
        payload: Dict[str, Any],
        *,
        default_status: str,
        raw: Dict[str, Any],
    ) -> OrderStatus:
        order_id = str(payload.get("id") or payload.get("order_id") or payload.get("orderId") or "")
        status = str(payload.get("status") or default_status).upper()
        filled = float(
            payload.get("filled")
            or payload.get("filled_amount")
            or payload.get("amount_filled")
            or 0.0
        )
        amount = float(payload.get("amount") or payload.get("original_amount") or filled)
        price = payload.get("price") or payload.get("avg_price")
        average_price = float(price) if price not in (None, "") else None
        remaining = max(amount - filled, 0.0)
        return OrderStatus(
            order_id=order_id,
            status=status,
            filled_quantity=filled,
            remaining_quantity=remaining,
            average_price=average_price,
            raw=raw,
        )


__all__ = ["BitstampAdapter"]
