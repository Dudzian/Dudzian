"""Abstrakcje adapterów giełdowych dla REST + WebSocket."""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence

try:  # pragma: no cover - biblioteka opcjonalna w runtime
    import httpx
except Exception:  # pragma: no cover - środowisko testowe
    httpx = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExchangeCredentials:
    """Parametry autoryzacji konta giełdowego."""

    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    is_read_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RateLimitRule:
    """Konfiguracja limitu: ``rate`` żądań na ``per`` sekund."""

    rate: int
    per: float
    weight: int = 1

    def __post_init__(self) -> None:
        if self.rate <= 0 or self.per <= 0:
            raise ValueError("RateLimitRule musi mieć dodatnie wartości")
        if self.weight <= 0:
            raise ValueError("Waga limitu musi być dodatnia")


class RateLimiter:
    """Asynchroniczny licznik tokenów obsługujący wagę zapytań."""

    def __init__(self, rule: RateLimitRule) -> None:
        self._rule = rule
        self._tokens = float(rule.rate)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, weight: int | None = None) -> None:
        weight = weight or self._rule.weight
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._updated
                if elapsed > 0:
                    refill = elapsed * self._rule.rate / self._rule.per
                    if refill > 0:
                        self._tokens = min(self._rule.rate, self._tokens + refill)
                        self._updated = now
                if self._tokens >= weight:
                    self._tokens -= weight
                    return
                missing = weight - self._tokens
                wait_for = max(missing * self._rule.per / self._rule.rate, 0.01)
                logger.debug("RateLimiter waiting %.3fs (missing %.2f tokens)", wait_for, missing)
                await asyncio.sleep(wait_for)


@dataclass(slots=True)
class MarketSubscription:
    channel: str
    symbols: Sequence[str]
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: str
    quantity: float
    order_type: str = "LIMIT"
    price: Optional[float] = None
    time_in_force: Optional[str] = None
    client_order_id: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderStatus:
    order_id: str
    status: str
    filled_quantity: float
    remaining_quantity: float
    average_price: Optional[float]
    raw: Dict[str, Any] = field(default_factory=dict)


class WebSocketSubscription(Protocol):
    async def __aenter__(self) -> "WebSocketSubscription":
        ...

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:  # type: ignore[override]
        ...


MarketPayload = Dict[str, Any]
CallbackT = Callable[[MarketPayload], Awaitable[None]]
OrderCallbackT = Callable[[OrderStatus], Awaitable[None]]


class ExchangeAdapter(Protocol):
    """Wspólny kontrakt wykorzystywany przez ExecutionService."""

    name: str
    demo_mode: bool

    async def connect(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def authenticate(self, credentials: ExchangeCredentials) -> None:
        ...

    async def fetch_market_data(self, symbol: str) -> MarketPayload:
        ...

    async def stream_market_data(
        self, subscriptions: Iterable[MarketSubscription], callback: CallbackT
    ) -> WebSocketSubscription:
        ...

    async def submit_order(self, order: OrderRequest) -> OrderStatus:
        ...

    async def fetch_order_status(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        ...

    async def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        ...

    async def monitor_order(
        self,
        order_id: str,
        *,
        poll_interval: float = 1.0,
        symbol: Optional[str] = None,
        timeout: float = 30.0,
    ) -> OrderStatus:
        ...


class RESTClient(Protocol):
    async def request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        ...


class RESTWebSocketAdapter(ABC):
    """Bazowa implementacja adaptera REST + WebSocket z retry/backoff."""

    http_client: RESTClient

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        demo_mode: bool = True,
        rate_limit_rule: Optional[RateLimitRule] = None,
        http_client: Optional[RESTClient] = None,
        compliance_ack: bool = False,
    ) -> None:
        if not demo_mode and not compliance_ack:
            raise ValueError("Live trading wymaga potwierdzenia compliance")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.demo_mode = demo_mode
        self._rate_limiter = RateLimiter(rate_limit_rule or RateLimitRule(rate=1200, per=60.0))
        if http_client is not None:
            self.http_client = http_client
        else:
            if httpx is None:
                raise RuntimeError("httpx nie jest dostępny – wstrzyknij klienta REST")
            self.http_client = httpx.AsyncClient(base_url=self.base_url)  # type: ignore[assignment]
        self._credentials: Optional[ExchangeCredentials] = None
        self._closed = False

    async def connect(self) -> None:
        self._closed = False

    async def close(self) -> None:
        self._closed = True
        client_close = getattr(self.http_client, "aclose", None)
        if callable(client_close):
            try:
                result = client_close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:  # pragma: no cover - logujemy
                logger.debug("Błąd podczas zamykania klienta HTTP %s: %s", self.name, exc)

    async def authenticate(self, credentials: ExchangeCredentials) -> None:
        if credentials.is_read_only:
            logger.info("%s: uwierzytelnianie w trybie read-only", self.name)
        self._credentials = credentials

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        weight: int = 1,
        retries: int = 3,
        backoff: float = 0.5,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("Adapter jest zamknięty")
        await self._rate_limiter.acquire(weight)
        url = f"{self.base_url}{path}"
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= retries:
            try:
                response = await self.http_client.request(
                    method,
                    url,
                    params=params,
                    data=data,
                    headers=headers,
                    timeout=timeout,
                )
                status = getattr(response, "status_code", None)
                if status and status >= 400:
                    if status in {429, 418}:
                        await asyncio.sleep(backoff * (attempt + 1))
                        attempt += 1
                        continue
                    raise RuntimeError(f"{self.name} HTTP {status}: {getattr(response, 'text', '')}")
                if hasattr(response, "json"):
                    return await response.json() if asyncio.iscoroutinefunction(response.json) else response.json()
                if hasattr(response, "data"):
                    return response.data  # type: ignore[return-value]
                raise RuntimeError("Nieznany typ odpowiedzi HTTP")
            except Exception as exc:  # pragma: no cover - fallback retry
                last_exc = exc
                logger.debug("%s request attempt %s failed: %s", self.name, attempt, exc)
                await asyncio.sleep(backoff * (attempt + 1))
                attempt += 1
        raise RuntimeError(f"{self.name} request failed: {last_exc}")

    @abstractmethod
    async def fetch_market_data(self, symbol: str) -> MarketPayload:
        raise NotImplementedError

    @abstractmethod
    async def stream_market_data(
        self, subscriptions: Iterable[MarketSubscription], callback: CallbackT
    ) -> WebSocketSubscription:
        raise NotImplementedError

    @abstractmethod
    async def submit_order(self, order: OrderRequest) -> OrderStatus:
        raise NotImplementedError

    @abstractmethod
    async def fetch_order_status(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        raise NotImplementedError

    @abstractmethod
    async def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> OrderStatus:
        raise NotImplementedError

    async def monitor_order(
        self,
        order_id: str,
        *,
        poll_interval: float = 1.0,
        symbol: Optional[str] = None,
        timeout: float = 30.0,
    ) -> OrderStatus:
        deadline = time.monotonic() + timeout
        while True:
            status = await self.fetch_order_status(order_id, symbol=symbol)
            if status.status.upper() in {"FILLED", "CANCELED", "REJECTED"}:
                return status
            if time.monotonic() >= deadline:
                return status
            await asyncio.sleep(poll_interval)


__all__ = [
    "ExchangeCredentials",
    "RateLimitRule",
    "RateLimiter",
    "MarketSubscription",
    "OrderRequest",
    "OrderStatus",
    "ExchangeAdapter",
    "RESTWebSocketAdapter",
    "WebSocketSubscription",
]
