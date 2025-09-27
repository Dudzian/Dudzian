"""Adapter Binance Testnet zgodny z RESTWebSocketAdapter."""
from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Dict, Optional

from .interfaces import (
    ExchangeCredentials,
    MarketPayload,
    MarketSubscription,
    OrderRequest,
    OrderStatus,
    RESTWebSocketAdapter,
    WebSocketSubscription,
)


class _DummySubscription:
    """Prosta implementacja kontekstu WebSocket dla testów."""

    def __init__(self, callback: Callable[[], None] | None = None) -> None:
        self._callback = callback

    async def __aenter__(self) -> "_DummySubscription":
        if self._callback:
            self._callback()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:  # type: ignore[override]
        return None


class BinanceTestnetAdapter(RESTWebSocketAdapter):
    """Adapter obsługujący podstawowe operacje dla Binance Testnet."""

    def __init__(
        self,
        *,
        demo_mode: bool = True,
        http_client=None,
        ws_factory: Optional[
            Callable[[Iterable[MarketSubscription], Callable[[MarketPayload], Awaitable[None]]], WebSocketSubscription]
        ] = None,
        compliance_ack: bool = False,
    ) -> None:
        super().__init__(
            name="binance-testnet",
            base_url="https://testnet.binance.vision",
            demo_mode=demo_mode,
            http_client=http_client,
            compliance_ack=compliance_ack,
        )
        self._ws_factory = ws_factory or self._default_ws_factory

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

    async def fetch_market_data(self, symbol: str) -> MarketPayload:
        response = await self._request("GET", "/api/v3/ticker/bookTicker", params={"symbol": symbol})
        return response

    async def stream_market_data(
        self, subscriptions: Iterable[MarketSubscription], callback: Callable[[MarketPayload], Awaitable[None]]
    ) -> WebSocketSubscription:
        return self._ws_factory(subscriptions, callback)

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
        response = await self._request(
            "POST",
            "/api/v3/order",
            params=signed,
            headers=self._auth_headers(),
        )
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
        response = await self._request(
            "GET",
            "/api/v3/order",
            params=signed,
            headers=self._auth_headers(),
        )
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
        response = await self._request(
            "DELETE",
            "/api/v3/order",
            params=signed,
            headers=self._auth_headers(),
        )
        return OrderStatus(
            order_id=str(response.get("orderId", order_id)),
            status=response.get("status", "CANCELED"),
            filled_quantity=float(response.get("executedQty", 0.0)),
            remaining_quantity=float(response.get("origQty", 0.0))
            - float(response.get("executedQty", 0.0)),
            average_price=float(response.get("price", 0.0)) if response.get("price") else None,
            raw=response,
        )

    def _default_ws_factory(
        self,
        subscriptions: Iterable[MarketSubscription],
        callback: Callable[[MarketPayload], Awaitable[None]],
    ) -> WebSocketSubscription:
        # Domyślnie zwracamy atrapę – umożliwia wstrzyknięcie mocków w testach.
        return _DummySubscription()


__all__ = ["BinanceTestnetAdapter"]
