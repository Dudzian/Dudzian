"""Adapter Binance Testnet zgodny z RESTWebSocketAdapter."""
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

try:  # pragma: no cover - zależność opcjonalna w środowisku runtime
    import websockets
    from websockets.protocol import State as _WSState
except Exception:  # pragma: no cover - środowiska bez websocketów
    websockets = None  # type: ignore[assignment]
    _WSState = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


_BINANCE_WS_ENDPOINT = "wss://testnet.binance.vision/ws"
_BINANCE_WS_INITIAL_BACKOFF = 1.0
_BINANCE_WS_MAX_BACKOFF = 30.0


class _BinanceWebSocketSubscription(WebSocketSubscription):
    """Zarządza subskrypcjami websocket Binance z automatycznym reconnectem."""

    def __init__(
        self,
        *,
        endpoint: str,
        subscriptions: Sequence[str],
        callback: Callable[[MarketPayload], Awaitable[None]],
        initial_backoff: float = _BINANCE_WS_INITIAL_BACKOFF,
        max_backoff: float = _BINANCE_WS_MAX_BACKOFF,
    ) -> None:
        if not subscriptions:
            raise ValueError("Wymagana jest co najmniej jedna subskrypcja Binance")
        if websockets is None:  # pragma: no cover - ochrona przed brakiem zależności
            raise RuntimeError("Pakiet 'websockets' jest wymagany do streamingu Binance")
        self._endpoint = endpoint
        self._subscriptions = list(dict.fromkeys(subscriptions))
        self._callback = callback
        self._initial_backoff = max(0.1, initial_backoff)
        self._max_backoff = max(self._initial_backoff, max_backoff)
        self._task: Optional[asyncio.Task[None]] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._closed = False
        self._id_counter = 1

    async def __aenter__(self) -> "_BinanceWebSocketSubscription":
        self._task = asyncio.create_task(self._runner(), name="binance-ws-runner")
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
            except asyncio.CancelledError:  # pragma: no cover - kontrola zamknięcia
                raise
            except Exception as exc:
                if self._closed:
                    break
                attempt += 1
                wait_for = min(self._initial_backoff * (2 ** (attempt - 1)), self._max_backoff)
                logger.debug("Binance WS reconnect in %.2fs after error: %s", wait_for, exc)
                await asyncio.sleep(wait_for)
            else:
                if not self._closed:
                    # Po naturalnym zamknięciu również próbujemy wznowić połączenie
                    attempt += 1
                    wait_for = min(self._initial_backoff * (2 ** (attempt - 1)), self._max_backoff)
                    logger.debug("Binance WS reconnect in %.2fs after close", wait_for)
                    await asyncio.sleep(wait_for)

    async def _send_subscribe(self) -> None:
        if not self._ws:
            return
        payload = {"method": "SUBSCRIBE", "params": self._subscriptions, "id": self._next_id()}
        await self._ws.send(json.dumps(payload))

    async def _send_unsubscribe(self) -> None:
        if not self._ws:
            return
        state = getattr(self._ws, "state", None)
        name = getattr(state, "name", None)
        if name == "CLOSED" or (state is not None and _WSState is not None and state == getattr(_WSState, "CLOSED", None)):
            return
        payload = {"method": "UNSUBSCRIBE", "params": self._subscriptions, "id": self._next_id()}
        await self._ws.send(json.dumps(payload))

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    async def _handle_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8", "ignore")
        try:
            payload: MarketPayload = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("Binance WS otrzymał nieprawidłowy JSON: %s", message)
            return
        try:
            await self._callback(payload)
        except Exception as exc:  # pragma: no cover - callback użytkownika
            logger.exception("Błąd callbacka Binance WS: %s", exc)


def _build_binance_streams(subscriptions: Iterable[MarketSubscription]) -> list[str]:
    streams: list[str] = []
    for subscription in subscriptions:
        if subscription.params.get("stream"):
            template = subscription.params["stream"]
            if "{symbol}" in template:
                for symbol in subscription.symbols or [""]:
                    streams.append(template.format(symbol=symbol.lower()))
            else:
                streams.append(str(template))
            continue
        if not subscription.symbols:
            streams.append(subscription.channel)
            continue
        for symbol in subscription.symbols:
            streams.append(f"{symbol.lower()}@{subscription.channel}")
    return streams


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

    def _default_ws_factory(
        self,
        subscriptions: Iterable[MarketSubscription],
        callback: Callable[[MarketPayload], Awaitable[None]],
    ) -> WebSocketSubscription:
        streams = _build_binance_streams(subscriptions)
        return _BinanceWebSocketSubscription(
            endpoint=_BINANCE_WS_ENDPOINT,
            subscriptions=streams,
            callback=callback,
            initial_backoff=_BINANCE_WS_INITIAL_BACKOFF,
            max_backoff=_BINANCE_WS_MAX_BACKOFF,
        )


__all__ = ["BinanceTestnetAdapter"]
