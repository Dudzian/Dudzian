"""Zonda adapter providing REST operations and long-poll streaming."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
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


class ZondaAdapter(RESTStreamingAdapter):
    """Implementacja podstawowych operacji REST dla API Zonda."""

    def __init__(
        self,
        *,
        demo_mode: bool = True,
        http_client=None,
        stream_poll_interval: float = 1.0,
        compliance_ack: bool = False,
        enable_streaming: bool = False,
    ) -> None:
        super().__init__(
            name="zonda",
            base_url="https://api.zondacrypto.exchange/rest",
            demo_mode=demo_mode,
            http_client=http_client,
            compliance_ack=compliance_ack,
        )
        if stream_poll_interval <= 0:
            raise ValueError("stream_poll_interval musi być dodatni")
        self._stream_poll_interval = float(stream_poll_interval)
        self._enable_streaming = enable_streaming

    async def authenticate(self, credentials: ExchangeCredentials) -> None:
        if not credentials.api_key or not credentials.api_secret:
            raise ValueError("Wymagane są klucze API dla Zonda")
        await super().authenticate(credentials)

    async def fetch_market_data(self, symbol: str) -> MarketPayload:
        path = f"/trading/ticker/{symbol}"
        response: Dict[str, Any] = await self._request("GET", path)
        status = response.get("status") or response.get("code")
        if status not in ("Ok", "OK", 0, None):
            raise RuntimeError(f"Zonda API error: {status}")
        ticker = response.get("ticker") or response.get("items") or response.get("data")
        bid = 0.0
        ask = 0.0
        last = 0.0
        if isinstance(ticker, dict):
            bid = self._to_float(ticker.get("highestBid") or ticker.get("bid"))
            ask = self._to_float(ticker.get("lowestAsk") or ticker.get("ask"))
            last = self._to_float(ticker.get("rate") or ticker.get("last"))
        payload: MarketPayload = {
            "symbol": symbol,
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
        if not self._enable_streaming:
            raise RuntimeError(
                "Streaming danych rynkowych Zonda jest wyłączony – domyślnie korzystamy z odpytywania REST."
            )
        return LongPollSubscription(
            self,
            subscriptions,
            callback,
            default_interval=self._stream_poll_interval,
        )

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
        del symbol
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

    _STATUS_ALIASES = {
        "ok": "OK",
        "new": "NEW",
        "active": "ACTIVE",
        "pending": "PENDING",
        "waiting": "PENDING",
        "post_only": "PENDING",
        "postonly": "PENDING",
        "partially_filled": "PARTIALLY_FILLED",
        "partiallyfilled": "PARTIALLY_FILLED",
        "partially-filled": "PARTIALLY_FILLED",
        "partially": "PARTIALLY_FILLED",
        "cancelled": "CANCELLED",
        "canceled": "CANCELED",
        "filled": "FILLED",
        "closed": "FILLED",
        "completed": "FILLED",
        "rejected": "REJECTED",
        "expired": "EXPIRED",
        "failed": "REJECTED",
    }

    def _normalize_status(self, status: Any, default_status: str) -> str:
        if not status:
            return default_status
        text = str(status).strip()
        if not text:
            return default_status
        key = text.replace(" ", "_").replace("-", "_").lower()
        return self._STATUS_ALIASES.get(key, text.upper())

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
        status = self._normalize_status(payload.get("status"), default_status)
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

    @staticmethod
    def _to_float(value: Any) -> float:
        if value in (None, ""):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - dane niekonwertowalne
            return 0.0


__all__ = ["ZondaAdapter"]
