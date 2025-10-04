"""Adapter REST/WebSocket dla giełdy Zonda."""
from __future__ import annotations
from typing import Any, Mapping, cast

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import time
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


try:  # pragma: no cover - zależność opcjonalna w środowisku runtime
    import websockets
except Exception:  # pragma: no cover - środowiska bez websocketów
    websockets = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_ZONDA_WS_ENDPOINT = "wss://api.zondacrypto.exchange/websocket"
_ZONDA_WS_INITIAL_BACKOFF = 1.0
_ZONDA_WS_MAX_BACKOFF = 20.0


class _ZondaWebSocketSubscription(WebSocketSubscription):
    """Zarządza cyklem życia połączenia WS do publicznych kanałów Zondy."""

    def __init__(
        self,
        *,
        endpoint: str,
        subscribe_messages: Sequence[Dict[str, Any]],
        unsubscribe_messages: Sequence[Dict[str, Any]],
        callback: Callable[[MarketPayload], Awaitable[None]],
        initial_backoff: float = _ZONDA_WS_INITIAL_BACKOFF,
        max_backoff: float = _ZONDA_WS_MAX_BACKOFF,
    ) -> None:
        if websockets is None:  # pragma: no cover - ochrona przed brakiem zależności
            raise RuntimeError("Pakiet 'websockets' jest wymagany do streamingu Zonda")
        if not subscribe_messages:
            raise ValueError("Wymagana jest co najmniej jedna subskrypcja Zonda")
        self._endpoint = endpoint
        self._subscribe_messages = list(subscribe_messages)
        self._unsubscribe_messages = list(unsubscribe_messages)
        self._callback = callback
        self._initial_backoff = max(0.1, initial_backoff)
        self._max_backoff = max(self._initial_backoff, max_backoff)
        self._task: Optional[asyncio.Task[None]] = None
        self._ws: Any = None
        self._closed = False

    async def __aenter__(self) -> "_ZondaWebSocketSubscription":
        self._task = asyncio.create_task(self._runner(), name="zonda-ws-runner")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:  # type: ignore[override]
        self._closed = True
        if self._ws:
            with contextlib.suppress(Exception):
                await self._send_messages(self._unsubscribe_messages)
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
                    await self._send_messages(self._subscribe_messages)
                    attempt = 0
                    async for message in self._ws:
                        await self._handle_message(message)
                finally:
                    if self._ws:
                        with contextlib.suppress(Exception):
                            await self._ws.close()
                    self._ws = None
            except asyncio.CancelledError:  # pragma: no cover - kontrola zamknięcia
                raise
            except Exception as exc:
                if self._closed:
                    break
                attempt += 1
                wait_for = min(self._initial_backoff * (2 ** (attempt - 1)), self._max_backoff)
                logger.debug("Zonda WS reconnect in %.2fs after error: %s", wait_for, exc)
                await asyncio.sleep(wait_for)
            else:
                if not self._closed:
                    attempt += 1
                    wait_for = min(self._initial_backoff * (2 ** (attempt - 1)), self._max_backoff)
                    logger.debug("Zonda WS reconnect in %.2fs after close", wait_for)
                    await asyncio.sleep(wait_for)

    async def _send_messages(self, messages: Sequence[Dict[str, Any]]) -> None:
        if not self._ws:
            return
        for message in messages:
            await self._ws.send(json.dumps(message, separators=(",", ":")))

    async def _handle_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8", "ignore")
        try:
            payload: MarketPayload = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("Zonda WS otrzymał nieprawidłowy JSON: %s", message)
            return
        try:
            await self._callback(payload)
        except Exception as exc:  # pragma: no cover - callback użytkownika
            logger.exception("Błąd callbacka Zonda WS: %s", exc)


def _build_ws_messages(subscriptions: Iterable[MarketSubscription]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    subscribe: list[Dict[str, Any]] = []
    unsubscribe: list[Dict[str, Any]] = []
    for subscription in subscriptions:
        params = dict(subscription.params.get("params", {})) if subscription.params else {}
        action = subscription.params.get("action", "subscribe-public") if subscription.params else "subscribe-public"
        unsubscribe_action = (
            subscription.params.get("unsubscribe_action", "unsubscribe")
            if subscription.params
            else "unsubscribe"
        )
        module = subscription.params.get("module") if subscription.params else None
        path = subscription.params.get("path") if subscription.params else None
        channel = subscription.channel.strip("/")
        if not module or not path:
            parts = channel.split("/", 1)
            module = module or parts[0]
            path = path or (parts[1] if len(parts) > 1 else parts[0])
        template = {
            "module": module,
            "path": path,
            "params": params,
        }
        symbols = list(subscription.symbols)
        targets = symbols or []
        for symbol in targets:
            payload = {
                "action": action,
                **template,
                "params": dict(cast(Mapping[str, Any], template["params"])),
            }
            unsubscribe_payload = {
                "action": unsubscribe_action,
                **template,
                "params": dict(cast(Mapping[str, Any], template["params"])),
            }
            if symbol is not None:
                payload["params"]["symbol"] = symbol
                unsubscribe_payload["params"]["symbol"] = symbol
            subscribe.append(payload)
            unsubscribe.append(unsubscribe_payload)
    return subscribe, unsubscribe


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
        subscribe, unsubscribe = _build_ws_messages(subscriptions)
        return _ZondaWebSocketSubscription(
            endpoint=_ZONDA_WS_ENDPOINT,
            subscribe_messages=subscribe,
            unsubscribe_messages=unsubscribe,
            callback=callback,
        )


__all__ = ["ZondaAdapter"]
