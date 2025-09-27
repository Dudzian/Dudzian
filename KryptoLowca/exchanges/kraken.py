"""Adapter demo dla Kraken API."""
from __future__ import annotations

import asyncio
import contextlib
import base64
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
    from websockets.protocol import State as _WSState
except Exception:  # pragma: no cover - środowisko bez websocketów
    websockets = None  # type: ignore[assignment]
    _WSState = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


_KRAKEN_WS_ENDPOINT = "wss://beta-ws.kraken.com/"
_KRAKEN_WS_INITIAL_BACKOFF = 1.0
_KRAKEN_WS_MAX_BACKOFF = 30.0


class _KrakenWebSocketSubscription(WebSocketSubscription):
    """Obsługa websocketów Kraken Demo z reconnect/backoff."""

    def __init__(
        self,
        *,
        endpoint: str,
        subscriptions: Sequence[tuple[Dict[str, Any], Dict[str, Any]]],
        callback: Callable[[MarketPayload], Awaitable[None]],
        initial_backoff: float = _KRAKEN_WS_INITIAL_BACKOFF,
        max_backoff: float = _KRAKEN_WS_MAX_BACKOFF,
    ) -> None:
        if not subscriptions:
            raise ValueError("Wymagana jest co najmniej jedna subskrypcja Kraken")
        if websockets is None:  # pragma: no cover - brak zależności
            raise RuntimeError("Pakiet 'websockets' jest wymagany do streamingu Kraken")
        self._endpoint = endpoint
        self._subscriptions = list(subscriptions)
        self._callback = callback
        self._initial_backoff = max(0.1, initial_backoff)
        self._max_backoff = max(self._initial_backoff, max_backoff)
        self._task: Optional[asyncio.Task[None]] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._closed = False

    async def __aenter__(self) -> "_KrakenWebSocketSubscription":
        self._task = asyncio.create_task(self._runner(), name="kraken-ws-runner")
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
            except asyncio.CancelledError:  # pragma: no cover - zamykanie kontekstu
                raise
            except Exception as exc:
                if self._closed:
                    break
                attempt += 1
                wait_for = min(self._initial_backoff * (2 ** (attempt - 1)), self._max_backoff)
                logger.debug("Kraken WS reconnect in %.2fs after error: %s", wait_for, exc)
                await asyncio.sleep(wait_for)
            else:
                if not self._closed:
                    attempt += 1
                    wait_for = min(self._initial_backoff * (2 ** (attempt - 1)), self._max_backoff)
                    logger.debug("Kraken WS reconnect in %.2fs after close", wait_for)
                    await asyncio.sleep(wait_for)

    async def _send_subscribe(self) -> None:
        if not self._ws:
            return
        for subscribe_payload, _ in self._subscriptions:
            await self._ws.send(json.dumps(subscribe_payload))

    async def _send_unsubscribe(self) -> None:
        if not self._ws:
            return
        state = getattr(self._ws, "state", None)
        name = getattr(state, "name", None)
        if name == "CLOSED" or (state is not None and _WSState is not None and state == getattr(_WSState, "CLOSED", None)):
            return
        for _, unsubscribe_payload in self._subscriptions:
            await self._ws.send(json.dumps(unsubscribe_payload))

    async def _handle_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8", "ignore")
        try:
            payload: MarketPayload = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("Kraken WS otrzymał nieprawidłowy JSON: %s", message)
            return
        try:
            await self._callback(payload)
        except Exception as exc:  # pragma: no cover - callback użytkownika
            logger.exception("Błąd callbacka Kraken WS: %s", exc)


def _prepare_kraken_payloads(subscriptions: Iterable[MarketSubscription]) -> list[tuple[Dict[str, Any], Dict[str, Any]]]:
    payloads: list[tuple[Dict[str, Any], Dict[str, Any]]] = []
    for subscription in subscriptions:
        subscribe: Dict[str, Any] = {"event": "subscribe"}
        unsubscribe: Dict[str, Any] = {"event": "unsubscribe"}

        pairs = list(subscription.symbols)
        if pairs:
            subscribe["pair"] = pairs
            unsubscribe["pair"] = pairs

        base_subscription = dict(subscription.params.get("subscription", {}))
        base_subscription.setdefault("name", subscription.channel)
        subscribe["subscription"] = dict(base_subscription)
        unsubscribe["subscription"] = dict(base_subscription)

        extra_fields = {k: v for k, v in subscription.params.items() if k != "subscription"}
        subscribe.update(extra_fields)
        unsubscribe.update(extra_fields)

        payloads.append((subscribe, unsubscribe))
    return payloads


class KrakenDemoAdapter(RESTWebSocketAdapter):
    """Implementacja podstawowych operacji Kraken Demo."""

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
        base_url = "https://api.demo.kraken.com"
        super().__init__(
            name="kraken-demo",
            base_url=base_url,
            demo_mode=demo_mode,
            http_client=http_client,
            compliance_ack=compliance_ack,
        )
        self._ws_factory = ws_factory or self._default_ws_factory

    async def authenticate(self, credentials: ExchangeCredentials) -> None:
        if not credentials.api_key or not credentials.api_secret:
            raise ValueError("Kraken wymaga kluczy API")
        await super().authenticate(credentials)

    async def fetch_market_data(self, symbol: str) -> MarketPayload:
        response = await self._request("GET", "/0/public/Ticker", params={"pair": symbol})
        return response

    async def stream_market_data(
        self, subscriptions: Iterable[MarketSubscription], callback: Callable[[MarketPayload], Awaitable[None]]
    ) -> WebSocketSubscription:
        return self._ws_factory(subscriptions, callback)

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
        except Exception:  # pragma: no cover - fallback dla kluczy w formacie plain
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

    def _default_ws_factory(
        self,
        subscriptions: Iterable[MarketSubscription],
        callback: Callable[[MarketPayload], Awaitable[None]],
    ) -> WebSocketSubscription:
        payloads = _prepare_kraken_payloads(subscriptions)
        return _KrakenWebSocketSubscription(
            endpoint=_KRAKEN_WS_ENDPOINT,
            subscriptions=payloads,
            callback=callback,
            initial_backoff=_KRAKEN_WS_INITIAL_BACKOFF,
            max_backoff=_KRAKEN_WS_MAX_BACKOFF,
        )


__all__ = ["KrakenDemoAdapter"]
