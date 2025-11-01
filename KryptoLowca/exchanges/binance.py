"""Adapter Binance Testnet korzystający z REST + long-polling."""
from __future__ import annotations

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


class BinanceTestnetAdapter(RESTStreamingAdapter):
    """Adapter obsługujący podstawowe operacje dla Binance Testnet."""

    def __init__(
        self,
        *,
        demo_mode: bool = True,
        http_client=None,
        stream_poll_interval: float = 1.0,
        compliance_ack: bool = False,
    ) -> None:
        super().__init__(
            name="binance-testnet",
            base_url="https://testnet.binance.vision",
            demo_mode=demo_mode,
            http_client=http_client,
            compliance_ack=compliance_ack,
        )
        if stream_poll_interval <= 0:
            raise ValueError("stream_poll_interval musi być dodatni")
        self._stream_poll_interval = float(stream_poll_interval)

    async def authenticate(self, credentials: ExchangeCredentials) -> None:
        if not credentials.api_key or not credentials.api_secret:
            raise ValueError("Wymagane są klucze API dla Binance")
        await super().authenticate(credentials)

    def _signed_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._credentials:
            raise RuntimeError("Brak poświadczeń – wywołaj authenticate()")
        payload = dict(params)
        payload.setdefault("recvWindow", 5_000)
        payload["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(payload, doseq=True)
        signature = hmac.new(
            self._credentials.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        payload["signature"] = signature
        return payload

    def _auth_headers(self) -> Dict[str, str]:
        if not self._credentials:
            raise RuntimeError("Brak poświadczeń – wywołaj authenticate()")
        return {"X-MBX-APIKEY": self._credentials.api_key}

    def _validate_response(self, payload: Dict[str, Any]) -> None:
        code = payload.get("code")
        if code not in (None, 0):
            raise RuntimeError(f"Binance API error {code}: {payload.get('msg')}")

    async def fetch_market_data(self, symbol: str) -> MarketPayload:
        response: Dict[str, Any] = await self._request(
            "GET", "/api/v3/ticker/bookTicker", params={"symbol": symbol}
        )
        self._validate_response(response)
        bid = float(response.get("bidPrice", 0.0))
        ask = float(response.get("askPrice", 0.0))
        bid_volume = float(response.get("bidQty", 0.0))
        ask_volume = float(response.get("askQty", 0.0))
        last_raw = response.get("lastPrice")
        last_price = float(last_raw) if last_raw is not None else 0.0
        payload: MarketPayload = {
            "symbol": response.get("symbol", symbol),
            "bid": bid,
            "ask": ask,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "last": last_price,
            "raw": response,
        }
        payload.update(response)
        payload["bid"] = bid
        payload["ask"] = ask
        payload["bid_volume"] = bid_volume
        payload["ask_volume"] = ask_volume
        payload["last"] = last_price
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
        params: Dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side.upper(),
            "type": order.order_type,
            "quantity": order.quantity,
        }
        if order.price is not None:
            params["price"] = order.price
        if order.time_in_force:
            params["timeInForce"] = order.time_in_force
        if order.client_order_id:
            params["newClientOrderId"] = order.client_order_id
        params.update(order.extra_params)
        signed = self._signed_params(params)
        response: Dict[str, Any] = await self._request(
            "POST",
            "/api/v3/order",
            params=signed,
            headers=self._auth_headers(),
        )
        self._validate_response(response)
        return OrderStatus(
            order_id=response.get("orderId", ""),
            status=response.get("status", "UNKNOWN"),
            filled_quantity=float(response.get("executedQty", 0.0)),
            remaining_quantity=float(response.get("origQty", 0.0))
            - float(response.get("executedQty", 0.0)),
            average_price=float(response.get("price", 0.0)) if response.get("price") else None,
            raw=response,
        )

    async def fetch_order_status(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        params = {"orderId": order_id}
        if symbol:
            params["symbol"] = symbol
        signed = self._signed_params(params)
        response: Dict[str, Any] = await self._request(
            "GET",
            "/api/v3/order",
            params=signed,
            headers=self._auth_headers(),
        )
        self._validate_response(response)
        return OrderStatus(
            order_id=str(response.get("orderId", order_id)),
            status=response.get("status", "UNKNOWN"),
            filled_quantity=float(response.get("executedQty", 0.0)),
            remaining_quantity=float(response.get("origQty", 0.0))
            - float(response.get("executedQty", 0.0)),
            average_price=float(response.get("price", 0.0)) if response.get("price") else None,
            raw=response,
        )

    async def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        params = {"orderId": order_id}
        if symbol:
            params["symbol"] = symbol
        signed = self._signed_params(params)
        response: Dict[str, Any] = await self._request(
            "DELETE",
            "/api/v3/order",
            params=signed,
            headers=self._auth_headers(),
        )
        self._validate_response(response)
        return OrderStatus(
            order_id=str(response.get("orderId", order_id)),
            status=response.get("status", "CANCELED"),
            filled_quantity=float(response.get("executedQty", 0.0)),
            remaining_quantity=float(response.get("origQty", 0.0))
            - float(response.get("executedQty", 0.0)),
            average_price=float(response.get("price", 0.0)) if response.get("price") else None,
            raw=response,
        )

__all__ = ["BinanceTestnetAdapter"]
