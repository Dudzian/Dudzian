"""Adapter OKX Margin/Derivatives spełniające kontrakt RESTWebSocketAdapter."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, Dict, Optional
import urllib.parse

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

_OKX_PUBLIC_WS = "wss://ws.okx.com:8443/ws/v5/public"
_OKX_WS_INITIAL_BACKOFF = 1.0
_OKX_WS_MAX_BACKOFF = 20.0


class _OKXWebSocketSubscription(WebSocketSubscription):
    """Obsługa subskrypcji publicznych kanałów OKX."""

    def __init__(
        self,
        *,
        endpoint: str,
        subscriptions: Sequence[Dict[str, Any]],
        callback: Callable[[MarketPayload], Awaitable[None]],
        initial_backoff: float = _OKX_WS_INITIAL_BACKOFF,
        max_backoff: float = _OKX_WS_MAX_BACKOFF,
    ) -> None:
        if websockets is None:  # pragma: no cover - zabezpieczenie
            raise RuntimeError("Pakiet 'websockets' jest wymagany do streamingu OKX")
        if not subscriptions:
            raise ValueError("OKX WS wymaga co najmniej jednej subskrypcji")
        self._endpoint = endpoint
        self._subscriptions = list(subscriptions)
        self._callback = callback
        self._initial_backoff = max(0.1, initial_backoff)
        self._max_backoff = max(self._initial_backoff, max_backoff)
        self._task: Optional[asyncio.Task[None]] = None
        self._ws: Any = None
        self._closed = False

    async def __aenter__(self) -> "_OKXWebSocketSubscription":
        self._task = asyncio.create_task(self._runner(), name="okx-ws-runner")
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
                logger.debug("OKX WS reconnect in %.2fs after error: %s", wait_for, exc)
                await asyncio.sleep(wait_for)
            else:
                if not self._closed:
                    attempt += 1
                    wait_for = min(self._initial_backoff * (2 ** (attempt - 1)), self._max_backoff)
                    logger.debug("OKX WS reconnect in %.2fs after close", wait_for)
                    await asyncio.sleep(wait_for)

    async def _send_subscribe(self) -> None:
        if not self._ws:
            return
        payload = {"op": "subscribe", "args": self._subscriptions}
        await self._ws.send(json.dumps(payload))

    async def _send_unsubscribe(self) -> None:
        if not self._ws:
            return
        payload = {"op": "unsubscribe", "args": self._subscriptions}
        await self._ws.send(json.dumps(payload))

    async def _handle_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8", "ignore")
        try:
            payload: MarketPayload = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("OKX WS otrzymał nieprawidłowy JSON: %s", message)
            return
        try:
            await self._callback(payload)
        except Exception as exc:  # pragma: no cover - callback użytkownika
            logger.exception("Błąd callbacka OKX WS: %s", exc)


def _build_ws_args(
    subscriptions: Iterable[MarketSubscription],
    *,
    default_channel: str,
) -> list[Dict[str, Any]]:
    payloads: list[Dict[str, Any]] = []
    for subscription in subscriptions:
        channel = subscription.params.get("channel") if subscription.params else None
        channel = channel or subscription.channel or default_channel
        base_args = {k: v for k, v in (subscription.params or {}).items() if k != "channel"}
        symbols = list(subscription.symbols) or [None]
        for symbol in symbols:
            payload = {"channel": channel, **base_args}
            if symbol:
                payload.setdefault("instId", symbol)
            payloads.append(payload)
    return payloads


class _OKXBaseAdapter(RESTWebSocketAdapter):
    """Bazowa implementacja adapterów OKX."""

    def __init__(
        self,
        *,
        name: str,
        inst_type: str,
        default_td_mode: str,
        demo_mode: bool,
        http_client=None,
        ws_factory: Optional[
            Callable[[Iterable[MarketSubscription], Callable[[MarketPayload], Awaitable[None]]], WebSocketSubscription]
        ],
        compliance_ack: bool,
    ) -> None:
        super().__init__(
            name=name,
            base_url="https://www.okx.com",
            demo_mode=demo_mode,
            http_client=http_client,
            compliance_ack=compliance_ack,
        )
        self._ws_factory = ws_factory or self._default_ws_factory
        self._inst_type = inst_type
        self._default_td_mode = default_td_mode

    async def authenticate(self, credentials: ExchangeCredentials) -> None:
        if not credentials.api_key or not credentials.api_secret or not credentials.passphrase:
            raise ValueError("OKX wymaga klucza, sekretu oraz passphrase")
        await super().authenticate(credentials)

    async def fetch_market_data(self, symbol: str) -> MarketPayload:
        params = {"instId": symbol, "instType": self._inst_type}
        response: Dict[str, Any] = await self._request("GET", "/api/v5/market/ticker", params=params)
        if str(response.get("code")) not in ("0", "None", ""):
            raise RuntimeError(f"OKX API error {response.get('code')}: {response.get('msg')}")
        data = response.get("data") or []
        ticker = data[0] if data else {}
        bid = float(ticker.get("bidPx", 0.0))
        ask = float(ticker.get("askPx", 0.0))
        last = float(ticker.get("last", 0.0))
        payload: MarketPayload = {
            "symbol": ticker.get("instId", symbol),
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
            "instId": order.symbol,
            "tdMode": order.extra_params.get("tdMode", self._default_td_mode),
            "side": order.side.lower(),
            "ordType": order.order_type.lower(),
            "sz": str(order.quantity),
        }
        if order.price is not None:
            body["px"] = str(order.price)
        if order.client_order_id:
            body["clOrdId"] = order.client_order_id
        body.update({k: v for k, v in order.extra_params.items() if k not in {"tdMode"}})
        response = await self._private_request("POST", "/api/v5/trade/order", data=body)
        order_payload = (response.get("data") or [{}])[0]
        return self._parse_order(order_payload, default_status="NEW", raw=response)

    async def fetch_order_status(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        params = {"ordId": order_id}
        if symbol:
            params["instId"] = symbol
        response = await self._private_request("GET", "/api/v5/trade/order", params=params)
        order_payload = (response.get("data") or [{}])[0]
        if not order_payload:
            order_payload = {"ordId": order_id, "state": "unknown"}
        return self._parse_order(order_payload, default_status="UNKNOWN", raw=response)

    async def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        body: Dict[str, Any] = {"ordId": order_id}
        if symbol:
            body["instId"] = symbol
        response = await self._private_request("POST", "/api/v5/trade/cancel-order", data=body)
        order_payload = (response.get("data") or [{}])[0]
        if not order_payload:
            order_payload = {"ordId": order_id, "state": "canceled"}
        return self._parse_order(order_payload, default_status="CANCELED", raw=response)

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
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        body = ""
        if data and method.upper() != "GET":
            body = json.dumps(data, separators=(",", ":"))
        query = ""
        if params:
            query = "?" + urllib.parse.urlencode(params, doseq=True)
        prehash = f"{timestamp}{method.upper()}{path}{query}{body}"
        signature = base64.b64encode(
            hmac.new(
                self._credentials.api_secret.encode(), prehash.encode(), hashlib.sha256
            ).digest()
        ).decode()
        headers = {
            "OK-ACCESS-KEY": self._credentials.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self._credentials.passphrase or "",
            "Content-Type": "application/json",
        }
        response: Dict[str, Any] = await self._request(
            method,
            path + (query if method.upper() == "GET" and query else ""),
            data=body if body else None,
            headers=headers,
        )
        if str(response.get("code")) not in ("0", "None", ""):
            raise RuntimeError(f"OKX API error {response.get('code')}: {response.get('msg')}")
        return response

    def _parse_order(
        self,
        payload: Dict[str, Any],
        *,
        default_status: str,
        raw: Dict[str, Any],
    ) -> OrderStatus:
        order_id = str(payload.get("ordId") or payload.get("orderId") or "")
        status = str(payload.get("state") or default_status).upper()
        filled = float(payload.get("accFillSz", 0.0))
        qty = float(payload.get("sz", filled))
        avg_price = payload.get("avgPx") or payload.get("px")
        return OrderStatus(
            order_id=order_id,
            status=status,
            filled_quantity=filled,
            remaining_quantity=max(qty - filled, 0.0),
            average_price=float(avg_price) if avg_price not in (None, "") else None,
            raw=raw,
        )

    def _default_ws_factory(
        self,
        subscriptions: Iterable[MarketSubscription],
        callback: Callable[[MarketPayload], Awaitable[None]],
    ) -> WebSocketSubscription:
        args = _build_ws_args(subscriptions, default_channel="tickers")
        return _OKXWebSocketSubscription(
            endpoint=_OKX_PUBLIC_WS,
            subscriptions=args,
            callback=callback,
            initial_backoff=_OKX_WS_INITIAL_BACKOFF,
            max_backoff=_OKX_WS_MAX_BACKOFF,
        )


class OKXMarginAdapter(_OKXBaseAdapter):
    """Adapter obsługujący rynek margin OKX."""

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
            name="okx-margin",
            inst_type="MARGIN",
            default_td_mode="cross",
            demo_mode=demo_mode,
            http_client=http_client,
            ws_factory=ws_factory,
            compliance_ack=compliance_ack,
        )


class OKXDerivativesAdapter(_OKXBaseAdapter):
    """Adapter obsługujący instrumenty pochodne (perpetual/futures) OKX."""

    def __init__(
        self,
        *,
        demo_mode: bool = True,
        http_client=None,
        ws_factory: Optional[
            Callable[[Iterable[MarketSubscription], Callable[[MarketPayload], Awaitable[None]]], WebSocketSubscription]
        ] = None,
        compliance_ack: bool = False,
        inst_type: str = "SWAP",
    ) -> None:
        super().__init__(
            name="okx-derivatives",
            inst_type=inst_type,
            default_td_mode="cross",
            demo_mode=demo_mode,
            http_client=http_client,
            ws_factory=ws_factory,
            compliance_ack=compliance_ack,
        )


__all__ = ["OKXMarginAdapter", "OKXDerivativesAdapter"]

