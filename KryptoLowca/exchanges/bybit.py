"""Adapter Bybit Spot implementujący RESTWebSocketAdapter."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from collections.abc import Awaitable, Callable, Iterable, Sequence
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

try:  # pragma: no cover - zależność opcjonalna
    import websockets
except Exception:  # pragma: no cover - środowiska bez websocketów
    websockets = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_BYBIT_WS_ENDPOINT = "wss://stream.bybit.com/v5/public/spot"
_BYBIT_WS_INITIAL_BACKOFF = 1.0
_BYBIT_WS_MAX_BACKOFF = 20.0


class _BybitWebSocketSubscription(WebSocketSubscription):
    """Obsługa subskrypcji kanałów Bybit WebSocket v5."""

    def __init__(
        self,
        *,
        endpoint: str,
        channels: Sequence[str],
        callback: Callable[[MarketPayload], Awaitable[None]],
        initial_backoff: float = _BYBIT_WS_INITIAL_BACKOFF,
        max_backoff: float = _BYBIT_WS_MAX_BACKOFF,
    ) -> None:
        if websockets is None:  # pragma: no cover - zabezpieczenie
            raise RuntimeError("Pakiet 'websockets' jest wymagany do streamingu Bybit")
        if not channels:
            raise ValueError("Bybit WS wymaga co najmniej jednej subskrypcji")
        self._endpoint = endpoint
        self._channels = list(dict.fromkeys(channels))
        self._callback = callback
        self._initial_backoff = max(0.1, initial_backoff)
        self._max_backoff = max(self._initial_backoff, max_backoff)
        self._task: Optional[asyncio.Task[None]] = None
        self._ws: Any = None
        self._closed = False

    async def __aenter__(self) -> "_BybitWebSocketSubscription":
        self._task = asyncio.create_task(self._runner(), name="bybit-ws-runner")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:  # type: ignore[override]
        self._closed = True
        if self._ws:
            with contextlib.suppress(Exception):
                await self._send_unsubscribe()
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        return None

    async def _runner(self) -> None:
        attempt = 0
        while not self._closed:
            try:
                self._ws = await websockets.connect(self._endpoint, ping_interval=20)
                try:
                    await self._send_subscribe()
                    attempt = 0
                    async for message in self._ws:
                        await self._handle_message(message)
                finally:
                    if self._ws:
                        with contextlib.suppress(Exception):
                            await self._ws.close()
                    self._ws = None
            except asyncio.CancelledError:  # pragma: no cover - zamykanie
                raise
            except Exception as exc:
                if self._closed:
                    break
                attempt += 1
                wait_for = min(self._initial_backoff * (2 ** (attempt - 1)), self._max_backoff)
                logger.debug("Bybit WS reconnect in %.2fs after error: %s", wait_for, exc)
                await asyncio.sleep(wait_for)
            else:
                if not self._closed:
                    attempt += 1
                    wait_for = min(self._initial_backoff * (2 ** (attempt - 1)), self._max_backoff)
                    logger.debug("Bybit WS reconnect in %.2fs after close", wait_for)
                    await asyncio.sleep(wait_for)

    async def _send_subscribe(self) -> None:
        if not self._ws:
            return
        payload = {"op": "subscribe", "args": self._channels}
        await self._ws.send(json.dumps(payload))

    async def _send_unsubscribe(self) -> None:
        if not self._ws:
            return
        payload = {"op": "unsubscribe", "args": self._channels}
        await self._ws.send(json.dumps(payload))

    async def _handle_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8", "ignore")
        try:
            payload: MarketPayload = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("Bybit WS otrzymał nieprawidłowy JSON: %s", message)
            return
        try:
            await self._callback(payload)
        except Exception as exc:  # pragma: no cover - callback użytkownika
            logger.exception("Błąd callbacka Bybit WS: %s", exc)


def _resolve_channels(subscriptions: Iterable[MarketSubscription]) -> list[str]:
    channels: list[str] = []
    for subscription in subscriptions:
        base = subscription.params.get("topic") if subscription.params else None
        base = base or subscription.channel
        symbols = list(subscription.symbols) or [None]
        for symbol in symbols:
            if symbol is None:
                channels.append(base)
            elif "{symbol}" in base:
                channels.append(base.format(symbol=symbol))
            else:
                channels.append(f"{base}.{symbol}")
    return channels


class BybitSpotAdapter(RESTWebSocketAdapter):
    """Adapter REST/WS dla rynku spot Bybit v5."""

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
            name="bybit-spot",
            base_url="https://api.bybit.com",
            demo_mode=demo_mode,
            http_client=http_client,
            compliance_ack=compliance_ack,
        )
        self._ws_factory = ws_factory or self._default_ws_factory

    async def authenticate(self, credentials: ExchangeCredentials) -> None:
        if not credentials.api_key or not credentials.api_secret:
            raise ValueError("Bybit wymaga klucza API i sekretu")
        await super().authenticate(credentials)

    async def fetch_market_data(self, symbol: str) -> MarketPayload:
        params = {"category": "spot", "symbol": symbol}
        response: Dict[str, Any] = await self._request("GET", "/v5/market/tickers", params=params)
        if response.get("retCode") not in (0, "0", None):
            raise RuntimeError(f"Bybit API error {response.get('retCode')}: {response.get('retMsg')}")
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
    ) -> WebSocketSubscription:
        return self._ws_factory(subscriptions, callback)

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
            raise RuntimeError(f"Bybit API error {response.get('retCode')}: {response.get('retMsg')}")
        return response

    def _default_ws_factory(
        self,
        subscriptions: Iterable[MarketSubscription],
        callback: Callable[[MarketPayload], Awaitable[None]],
    ) -> WebSocketSubscription:
        channels = _resolve_channels(subscriptions)
        return _BybitWebSocketSubscription(
            endpoint=_BYBIT_WS_ENDPOINT,
            channels=channels,
            callback=callback,
            initial_backoff=_BYBIT_WS_INITIAL_BACKOFF,
            max_backoff=_BYBIT_WS_MAX_BACKOFF,
        )


__all__ = ["BybitSpotAdapter"]

