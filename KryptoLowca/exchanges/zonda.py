"""Adapter REST/WebSocket dla giełdy Zonda."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
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
    """Prosty kontekst WebSocket wykorzystywany w testach kontraktowych."""

    def __init__(self, callback: Callable[[], None] | None = None) -> None:
        self._callback = callback

    async def __aenter__(self) -> "_DummySubscription":
        if self._callback:
            self._callback()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:  # type: ignore[override]
        return None


class ZondaAdapter(RESTWebSocketAdapter):
    """Implementacja podstawowych operacji REST/WS dla API Zonda."""

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
            name="zonda",
            base_url="https://api.zondacrypto.exchange/rest",
            demo_mode=demo_mode,
            http_client=http_client,
            compliance_ack=compliance_ack,
        )
        self._ws_factory = ws_factory or self._default_ws_factory

    async def authenticate(self, credentials: ExchangeCredentials) -> None:
        if not credentials.api_key or not credentials.api_secret:
            raise ValueError("Wymagane są klucze API dla Zonda")
        await super().authenticate(credentials)

    async def fetch_market_data(self, symbol: str) -> MarketPayload:
        path = f"/trading/ticker/{symbol}"
        return await self._request("GET", path)

    async def stream_market_data(
        self, subscriptions: Iterable[MarketSubscription], callback: Callable[[MarketPayload], Awaitable[None]]
    ) -> WebSocketSubscription:
        return self._ws_factory(subscriptions, callback)

    async def submit_order(self, order: OrderRequest) -> OrderStatus:
        payload: Dict[str, Any] = {
            "market": order.symbol,
            "side": order.side.lower(),
            "type": order.order_type.lower(),
            "amount": order.quantity,
        }
        if order.price is not None:
            payload["price"] = order.price
        if order.time_in_force:
            payload["timeInForce"] = order.time_in_force
        if order.client_order_id:
            payload["clientOrderId"] = order.client_order_id
        if order.extra_params:
            payload.update(order.extra_params)

        response = await self._private_request("POST", "/trading/offer", data=payload)
        order_payload = self._extract_order_payload(response)
        return self._order_status_from_payload(order_payload or response, default_status="NEW", raw=response)

    async def fetch_order_status(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        del symbol  # nie jest wymagane przez API Zonda
        path = f"/trading/order/{order_id}"
        response = await self._private_request("GET", path)
        order_payload = self._extract_order_payload(response)
        return self._order_status_from_payload(order_payload or response, default_status="UNKNOWN", raw=response)

    async def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        del symbol
        path = f"/trading/order/{order_id}"
        response = await self._private_request("DELETE", path)
        order_payload = self._extract_order_payload(response)
        return self._order_status_from_payload(order_payload or response, default_status="CANCELED", raw=response)

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
        body = self._prepare_body(data)
        headers = self._signed_headers(method, path, body)
        request_data = body if body else None
        if request_data:
            headers.setdefault("Content-Type", "application/json")
        return await self._request(
            method,
            path,
            params=params,
            data=request_data,
            headers=headers,
        )

    def _prepare_body(self, data: Optional[Dict[str, Any]]) -> str:
        if not data:
            return ""
        return json.dumps(data, separators=(",", ":"), sort_keys=True)

    def _signed_headers(self, method: str, path: str, body: str) -> Dict[str, str]:
        if not self._credentials:
            raise RuntimeError("Brak poświadczeń – wywołaj authenticate()")
        timestamp = str(int(time.time() * 1000))
        payload = f"{timestamp}{method.upper()}{path}{body}"
        signature = hmac.new(
            self._credentials.api_secret.encode(), payload.encode(), hashlib.sha512
        ).hexdigest()
        return {
            "API-Key": self._credentials.api_key,
            "API-Hash": signature,
            "Request-Timestamp": timestamp,
        }

    def _order_status_from_payload(
        self,
        payload: Dict[str, Any],
        *,
        default_status: str,
        raw: Dict[str, Any],
    ) -> OrderStatus:
        order_id = str(
            payload.get("orderId")
            or payload.get("offerId")
            or payload.get("id")
            or payload.get("clientOrderId")
            or ""
        )
        status = str(payload.get("status", default_status)).upper()
        filled = self._to_float(
            payload.get("filled")
            or payload.get("filledAmount")
            or payload.get("amountFilled")
            or payload.get("executed")
            or 0
        )
        remaining = self._to_float(
            payload.get("remaining")
            or payload.get("remainingAmount")
            or payload.get("amountRemaining")
            or payload.get("left")
            or 0
        )
        avg_price_value = (
            payload.get("avgPrice")
            or payload.get("averagePrice")
            or payload.get("price")
            or payload.get("limit")
        )
        average_price = self._to_float(avg_price_value) if avg_price_value is not None else None
        return OrderStatus(
            order_id=order_id,
            status=status,
            filled_quantity=filled,
            remaining_quantity=remaining,
            average_price=average_price,
            raw=raw,
        )

    def _extract_order_payload(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for key in ("order", "offer", "result", "data"):
            value = response.get(key)
            if isinstance(value, dict):
                return value
        return None

    def _to_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _default_ws_factory(
        self,
        subscriptions: Iterable[MarketSubscription],
        callback: Callable[[MarketPayload], Awaitable[None]],
    ) -> WebSocketSubscription:
        del subscriptions, callback  # API Zonda wymaga zewnętrznej implementacji w runtime
        return _DummySubscription()


__all__ = ["ZondaAdapter"]
