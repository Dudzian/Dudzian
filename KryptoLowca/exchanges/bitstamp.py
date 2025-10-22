"""Adapter Bitstamp zgodny z kontraktem RESTWebSocketAdapter."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import time
import uuid
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

_BITSTAMP_WS_ENDPOINT = "wss://ws.bitstamp.net"
_BITSTAMP_WS_INITIAL_BACKOFF = 1.0
_BITSTAMP_WS_MAX_BACKOFF = 20.0


class _BitstampWebSocketSubscription(WebSocketSubscription):
    """Obsługuje subskrypcje kanałów Bitstamp WebSocket v2."""

    def __init__(
        self,
        *,
        endpoint: str,
        channels: Sequence[str],
        callback: Callable[[MarketPayload], Awaitable[None]],
        initial_backoff: float = _BITSTAMP_WS_INITIAL_BACKOFF,
        max_backoff: float = _BITSTAMP_WS_MAX_BACKOFF,
    ) -> None:
        if websockets is None:  # pragma: no cover - zabezpieczenie
            raise RuntimeError("Pakiet 'websockets' jest wymagany do streamingu Bitstamp")
        if not channels:
            raise ValueError("Bitstamp WS wymaga co najmniej jednego kanału")
        self._endpoint = endpoint
        self._channels = list(dict.fromkeys(channels))
        self._callback = callback
        self._initial_backoff = max(0.1, initial_backoff)
        self._max_backoff = max(self._initial_backoff, max_backoff)
        self._task: Optional[asyncio.Task[None]] = None
        self._ws: Any = None
        self._closed = False

    async def __aenter__(self) -> "_BitstampWebSocketSubscription":
        self._task = asyncio.create_task(self._runner(), name="bitstamp-ws-runner")
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
                logger.debug("Bitstamp WS reconnect in %.2fs after error: %s", wait_for, exc)
                await asyncio.sleep(wait_for)
            else:
                if not self._closed:
                    attempt += 1
                    wait_for = min(self._initial_backoff * (2 ** (attempt - 1)), self._max_backoff)
                    logger.debug("Bitstamp WS reconnect in %.2fs after close", wait_for)
                    await asyncio.sleep(wait_for)

    async def _send_subscribe(self) -> None:
        if not self._ws:
            return
        for channel in self._channels:
            payload = {"event": "bts:subscribe", "data": {"channel": channel}}
            await self._ws.send(json.dumps(payload))

    async def _send_unsubscribe(self) -> None:
        if not self._ws:
            return
        for channel in self._channels:
            payload = {"event": "bts:unsubscribe", "data": {"channel": channel}}
            await self._ws.send(json.dumps(payload))

    async def _handle_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8", "ignore")
        try:
            payload: MarketPayload = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("Bitstamp WS otrzymał nieprawidłowy JSON: %s", message)
            return
        try:
            await self._callback(payload)
        except Exception as exc:  # pragma: no cover - callback użytkownika
            logger.exception("Błąd callbacka Bitstamp WS: %s", exc)


def _resolve_channels(subscriptions: Iterable[MarketSubscription]) -> list[str]:
    channels: list[str] = []
    for subscription in subscriptions:
        template = subscription.params.get("channel") if subscription.params else None
        template = template or subscription.channel
        symbols = list(subscription.symbols) or [None]
        for symbol in symbols:
            if symbol is not None:
                channel = template.format(symbol=symbol.lower()) if "{symbol}" in template else f"{template}_{symbol.lower()}"
            else:
                channel = template
            channels.append(channel)
    return channels


class BitstampAdapter(RESTWebSocketAdapter):
    """Adapter REST/WS dla Bitstamp."""

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
            name="bitstamp",
            base_url="https://www.bitstamp.net/api/v2",
            demo_mode=demo_mode,
            http_client=http_client,
            compliance_ack=compliance_ack,
        )
        self._ws_factory = ws_factory or self._default_ws_factory

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
    ) -> WebSocketSubscription:
        return self._ws_factory(subscriptions, callback)

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
            raise RuntimeError(f"Bitstamp API error: {response.get('reason') or response.get('message')}")
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
        filled = float(payload.get("filled") or payload.get("filled_amount") or payload.get("amount_filled") or 0.0)
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

    def _default_ws_factory(
        self,
        subscriptions: Iterable[MarketSubscription],
        callback: Callable[[MarketPayload], Awaitable[None]],
    ) -> WebSocketSubscription:
        channels = _resolve_channels(subscriptions)
        return _BitstampWebSocketSubscription(
            endpoint=_BITSTAMP_WS_ENDPOINT,
            channels=channels,
            callback=callback,
            initial_backoff=_BITSTAMP_WS_INITIAL_BACKOFF,
            max_backoff=_BITSTAMP_WS_MAX_BACKOFF,
        )


__all__ = ["BitstampAdapter"]

