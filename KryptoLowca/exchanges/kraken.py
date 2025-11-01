"""Kraken demo adapter using REST endpoints and long-polling."""
from __future__ import annotations

import base64
import hashlib
import hmac
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


class KrakenDemoAdapter(RESTStreamingAdapter):
    """Implementacja podstawowych operacji Kraken Demo bez websocketów."""

    def __init__(
        self,
        *,
        demo_mode: bool = True,
        http_client=None,
        stream_poll_interval: float = 1.0,
        compliance_ack: bool = False,
    ) -> None:
        base_url = "https://api.demo.kraken.com"
        super().__init__(
            name="kraken-demo",
            base_url=base_url,
            demo_mode=demo_mode,
            http_client=http_client,
            compliance_ack=compliance_ack,
        )
        if stream_poll_interval <= 0:
            raise ValueError("stream_poll_interval musi być dodatni")
        self._stream_poll_interval = float(stream_poll_interval)

    async def authenticate(self, credentials: ExchangeCredentials) -> None:
        if not credentials.api_key or not credentials.api_secret:
            raise ValueError("Kraken wymaga kluczy API")
        await super().authenticate(credentials)

    async def fetch_market_data(self, symbol: str) -> MarketPayload:
        response: Dict[str, Any] = await self._request(
            "GET", "/0/public/Ticker", params={"pair": symbol}
        )
        errors = response.get("error")
        if errors:
            raise RuntimeError(f"Kraken API error: {errors}")
        result = response.get("result", {})
        symbol_key = next(iter(result.keys()), symbol)
        ticker = result.get(symbol_key, {})
        ask = ticker.get("a", [None])[0]
        bid = ticker.get("b", [None])[0]
        last = ticker.get("c", [None])[0]
        bid_value = float(bid) if bid is not None else 0.0
        ask_value = float(ask) if ask is not None else 0.0
        last_value = float(last) if last is not None else 0.0
        payload: MarketPayload = {
            "symbol": symbol,
            "bid": bid_value,
            "ask": ask_value,
            "last": last_value,
            "raw": response,
        }
        payload.update(response)
        payload["bid"] = bid_value
        payload["ask"] = ask_value
        payload["last"] = last_value
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

    def _private_headers(self, path: str, data: Dict[str, Any]) -> Dict[str, str]:
        if not self._credentials:
            raise RuntimeError("Brak poświadczeń")
        nonce = str(int(time.time() * 1000))
        data["nonce"] = nonce
        postdata = urllib.parse.urlencode(data)
        message = (nonce + postdata).encode()
        sha256 = hashlib.sha256(message).digest()
        try:
            secret = base64.b64decode(self._credentials.api_secret)
        except Exception:
            secret = self._credentials.api_secret.encode()
        mac = hmac.new(secret, path.encode() + sha256, hashlib.sha512)
        signature = base64.b64encode(mac.digest()).decode()
        return {"API-Key": self._credentials.api_key, "API-Sign": signature}

    async def submit_order(self, order: OrderRequest) -> OrderStatus:
        data = {
            "ordertype": order.order_type.lower(),
            "pair": order.symbol,
            "type": order.side.lower(),
            "volume": str(order.quantity),
        }
        if order.price is not None:
            data["price"] = str(order.price)
        data.update({k: str(v) for k, v in order.extra_params.items()})
        headers = self._private_headers("/0/private/AddOrder", data)
        response = await self._request(
            "POST",
            "/0/private/AddOrder",
            data=data,
            headers=headers,
        )
        txid = "".join(response.get("result", {}).get("txid", []))
        descr = response.get("result", {}).get("descr", {})
        return OrderStatus(
            order_id=txid,
            status="OPEN",
            filled_quantity=0.0,
            remaining_quantity=order.quantity,
            average_price=float(descr.get("price", 0.0)) if descr.get("price") else None,
            raw=response,
        )

    async def fetch_order_status(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        del symbol
        data = {"txid": order_id}
        headers = self._private_headers("/0/private/QueryOrders", data)
        response = await self._request(
            "POST",
            "/0/private/QueryOrders",
            data=data,
            headers=headers,
        )
        order_info = response.get("result", {}).get(order_id, {})
        status = order_info.get("status", "unknown").upper()
        filled = float(order_info.get("vol_exec", 0.0))
        total = float(order_info.get("vol", filled))
        price = order_info.get("price") or order_info.get("pricec")
        avg_price = float(price) if price else None
        return OrderStatus(
            order_id=order_id,
            status=status,
            filled_quantity=filled,
            remaining_quantity=max(total - filled, 0.0),
            average_price=avg_price,
            raw=response,
        )

    async def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        del symbol
        data = {"txid": order_id}
        headers = self._private_headers("/0/private/CancelOrder", data)
        response = await self._request(
            "POST",
            "/0/private/CancelOrder",
            data=data,
            headers=headers,
        )
        result = response.get("result", {})
        status = "CANCELED" if result.get("count", 0) else "UNKNOWN"
        return OrderStatus(
            order_id=order_id,
            status=status,
            filled_quantity=0.0,
            remaining_quantity=0.0,
            average_price=None,
            raw=response,
        )


__all__ = ["KrakenDemoAdapter"]
