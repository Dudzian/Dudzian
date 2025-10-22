"""Minimalistyczny adapter REST dla rynku spot nowa_gielda."""
from __future__ import annotations

import hmac
import logging
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

import requests
from requests import Response, Session
from requests.exceptions import RequestException

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.nowa_gielda import symbols
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RateLimitRule:
    """Opis pojedynczego limitu wagowego endpointu REST."""

    method: str
    path: str
    weight: int
    window_seconds: float
    max_requests: int

    @property
    def key(self) -> str:
        return f"{self.method.upper()} {self.path}".strip()


_RATE_LIMITS: Mapping[str, RateLimitRule] = {
    rule.key: rule
    for rule in (
        RateLimitRule("GET", "/public/ticker", weight=1, window_seconds=1.0, max_requests=20),
        RateLimitRule("GET", "/public/orderbook", weight=2, window_seconds=1.0, max_requests=10),
        RateLimitRule("POST", "/private/orders", weight=5, window_seconds=1.0, max_requests=5),
        RateLimitRule("DELETE", "/private/orders", weight=1, window_seconds=1.0, max_requests=10),
    )
}


def _strip_none(data: Mapping[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {}
    return {key: value for key, value in data.items() if value is not None}


def _canonical_payload(data: Mapping[str, Any] | None) -> str:
    if not data:
        return ""
    items: list[tuple[str, str]] = []
    for key, value in data.items():
        items.append((str(key), str(value)))
    items.sort()
    return "&".join(f"{key}={value}" for key, value in items)


class _RateLimiter:
    """Prosty licznik zużycia limitów w oknie czasowym."""

    __slots__ = ("_rules", "_state")

    def __init__(self, rules: Mapping[str, RateLimitRule]) -> None:
        self._rules = rules
        self._state: dict[str, tuple[float, int]] = {}

    def consume(self, method: str, path: str) -> None:
        key = f"{method.upper()} {path}".strip()
        rule = self._rules.get(key)
        if rule is None:
            return

        now = time.monotonic()
        window_start, used = self._state.get(key, (now, 0))
        if now - window_start >= rule.window_seconds:
            window_start = now
            used = 0

        projected = used + rule.weight
        if projected > rule.max_requests:
            raise ExchangeThrottlingError(
                message="Limit zapytań dla endpointu został przekroczony",
                status_code=429,
                payload={"endpoint": key, "used": used, "limit": rule.max_requests},
            )

        self._state[key] = (window_start, projected)


_ERROR_CODE_MAPPING: Mapping[str, type[ExchangeAPIError]] = {
    "INVALID_SIGNATURE": ExchangeAuthError,
    "AUTHENTICATION_REQUIRED": ExchangeAuthError,
    "RATE_LIMIT_EXCEEDED": ExchangeThrottlingError,
    "ORDER_NOT_FOUND": ExchangeAPIError,
    "INVALID_SYMBOL": ExchangeAPIError,
}


class NowaGieldaHTTPClient:
    """Klient HTTP odpowiadający za komunikację z REST API nowa_gielda."""

    __slots__ = ("_base_url", "_session", "_timeout", "_rate_limiter")

    def __init__(
        self,
        base_url: str,
        *,
        session: Session | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout
        self._rate_limiter = _RateLimiter(_RATE_LIMITS)

    @property
    def rate_limiter(self) -> _RateLimiter:
        return self._rate_limiter

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        self._rate_limiter.consume(method, path)
        url = f"{self._base_url}{path}"
        try:
            response = self._session.request(
                method,
                url,
                params=params,
                json=json_body,
                headers=headers,
                timeout=self._timeout,
            )
        except RequestException as exc:  # pragma: no cover - zabezpieczenie
            raise ExchangeNetworkError("Błąd połączenia z API nowa_gielda", reason=exc) from exc

        return self._parse_response(method, path, response)

    def _parse_response(self, method: str, path: str, response: Response) -> Mapping[str, Any]:
        status = response.status_code
        if 200 <= status < 300:
            try:
                return response.json()
            except ValueError as exc:  # pragma: no cover - defensywnie
                raise ExchangeAPIError(
                    message="Niepoprawny format JSON odpowiedzi",
                    status_code=status,
                    payload=response.text,
                ) from exc

        payload: Any
        try:
            payload = response.json()
        except ValueError:  # pragma: no cover - fallback
            payload = response.text

        message = ""
        code: str | None = None
        if isinstance(payload, Mapping):
            message = str(payload.get("message", ""))
            raw_code = payload.get("code")
            code = str(raw_code) if raw_code is not None else None

        exc_cls: type[ExchangeAPIError] | None = None
        if code:
            exc_cls = _ERROR_CODE_MAPPING.get(code)

        if exc_cls is None:
            if status in {401, 403}:
                exc_cls = ExchangeAuthError
            elif status == 429:
                exc_cls = ExchangeThrottlingError
            else:
                exc_cls = ExchangeAPIError

        raise exc_cls(
            message=message or f"Błąd API ({status}) przy {method.upper()} {path}",
            status_code=status,
            payload=payload,
        )

    # --- Public helpers -------------------------------------------------
    def fetch_ticker(self, symbol: str) -> Mapping[str, Any]:
        return self._request("GET", "/public/ticker", params={"symbol": symbol})

    def fetch_orderbook(self, symbol: str, depth: int = 50) -> Mapping[str, Any]:
        return self._request(
            "GET",
            "/public/orderbook",
            params={"symbol": symbol, "depth": depth},
        )

    def create_order(
        self,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str],
    ) -> Mapping[str, Any]:
        return self._request("POST", "/private/orders", json_body=payload, headers=headers)

    def cancel_order(
        self,
        order_id: str,
        *,
        headers: Mapping[str, str],
        symbol: Optional[str] = None,
    ) -> Mapping[str, Any]:
        params = {"orderId": order_id}
        if symbol is not None:
            params["symbol"] = symbol
        return self._request("DELETE", "/private/orders", params=params, headers=headers)


class NowaGieldaSpotAdapter(ExchangeAdapter):
    """Adapter implementujący podstawowe operacje dla nowa_gielda."""

    __slots__ = ("_environment", "_ip_allowlist", "_metrics", "_http_client")

    name: str = "nowa_gielda_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
    ) -> None:
        super().__init__(credentials)
        self._environment = environment or credentials.environment
        self._ip_allowlist: tuple[str, ...] = ()
        self._metrics: Mapping[str, Any] | None = None
        self._http_client = NowaGieldaHTTPClient(self._determine_base_url(self._environment))

    # --- Utilities ---------------------------------------------------------
    @staticmethod
    def _timestamp() -> int:
        return int(time.time() * 1000)

    def _secret(self) -> bytes:
        secret = self.credentials.secret or ""
        return secret.encode("utf-8")

    # --- Configuration -----------------------------------------------------
    def configure_network(self, *, ip_allowlist: Optional[Sequence[str]] = None) -> None:
        self._ip_allowlist = tuple(ip_allowlist or ())
        if self._ip_allowlist:
            _LOGGER.debug("Skonfigurowano allowlistę IP: %s", self._ip_allowlist)

    # --- ExchangeAdapter API -----------------------------------------------
    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 0.0},
            total_equity=0.0,
            available_margin=0.0,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self) -> Iterable[str]:
        return symbols.supported_internal_symbols()

    # --- Market data -----------------------------------------------------
    def fetch_ticker(self, symbol: str) -> Mapping[str, Any]:
        exchange_symbol = symbols.to_exchange_symbol(symbol)
        payload = self._http_client.fetch_ticker(exchange_symbol)
        response_symbol = payload.get("symbol")
        if response_symbol and symbols.to_internal_symbol(response_symbol) != symbol:
            raise ExchangeAPIError(
                message="Symbol w odpowiedzi API nie zgadza się z zapytaniem",
                status_code=200,
                payload=payload,
            )
        return {
            "symbol": symbol,
            "best_bid": float(payload["bestBid"]),
            "best_ask": float(payload["bestAsk"]),
            "last_price": float(payload["lastPrice"]),
            "timestamp": float(payload.get("timestamp", self._timestamp())),
        }

    def fetch_orderbook(self, symbol: str, depth: int = 50) -> Mapping[str, Any]:
        exchange_symbol = symbols.to_exchange_symbol(symbol)
        payload = self._http_client.fetch_orderbook(exchange_symbol, depth=depth)
        response_symbol = payload.get("symbol")
        if response_symbol and symbols.to_internal_symbol(response_symbol) != symbol:
            raise ExchangeAPIError(
                message="Symbol w orderbooku nie zgadza się z zapytaniem",
                status_code=200,
                payload=payload,
            )
        return payload

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        return ()

    def place_order(self, request: OrderRequest) -> OrderResult:
        payload = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(request.symbol),
                "side": request.side,
                "type": request.order_type,
                "quantity": request.quantity,
                "price": request.price,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(timestamp, "POST", "/private/orders", body=payload)
        headers = self.build_auth_headers(timestamp, signature)
        _LOGGER.debug("Składanie zlecenia %s z nagłówkami %s", payload, headers)
        response = self._http_client.create_order(payload, headers=headers)
        order_id = str(response["orderId"])
        status = str(response.get("status", "accepted"))
        filled_qty = float(response.get("filledQuantity", 0.0))
        avg_price_raw = response.get("avgPrice")
        avg_price = float(avg_price_raw) if avg_price_raw is not None else None
        return OrderResult(
            order_id=order_id,
            status=status,
            filled_quantity=filled_qty,
            avg_price=avg_price,
            raw_response=response,
        )

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:
        _LOGGER.debug("Anulowanie zlecenia %s dla symbolu %s", order_id, symbol)
        timestamp = self._timestamp()
        exchange_symbol = symbols.to_exchange_symbol(symbol) if symbol else None
        params = _strip_none({"orderId": order_id, "symbol": exchange_symbol})
        signature = self.sign_request(
            timestamp,
            "DELETE",
            "/private/orders",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        self._http_client.cancel_order(order_id, headers=headers, symbol=exchange_symbol)

    def stream_public_data(self, *, channels: Sequence[str]) -> Protocol:
        raise NotImplementedError("Strumień publiczny nie jest wspierany w testowym adapterze")

    def stream_private_data(self, *, channels: Sequence[str]) -> Protocol:
        raise NotImplementedError("Strumień prywatny nie jest wspierany w testowym adapterze")

    # --- Custom helpers ----------------------------------------------------
    def sign_request(
        self,
        timestamp: int,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> str:
        suffix_parts: list[str] = []
        canonical_params = _canonical_payload(_strip_none(params))
        canonical_body = _canonical_payload(_strip_none(body))
        if canonical_params:
            suffix_parts.append(f"P:{canonical_params}")
        if canonical_body:
            suffix_parts.append(f"B:{canonical_body}")
        suffix = "|".join(suffix_parts)
        message = f"{timestamp}{method.upper()}{path}{suffix}".encode("utf-8")
        return hmac.new(self._secret(), message, sha256).hexdigest()

    def build_auth_headers(self, timestamp: int, signature: str) -> Mapping[str, str]:
        return {
            "X-API-KEY": self.credentials.key_id,
            "X-API-SIGN": signature,
            "X-API-TIMESTAMP": str(timestamp),
        }

    def _determine_base_url(self, environment: Environment) -> str:
        if environment is Environment.LIVE:
            return "https://api.nowa-gielda.example"
        if environment is Environment.PAPER:
            return "https://paper.nowa-gielda.example"
        return "https://testnet.nowa-gielda.example"

    def rate_limit_rule(self, method: str, path: str) -> RateLimitRule | None:
        key = f"{method.upper()} {path}".strip()
        return _RATE_LIMITS.get(key)

    def request_weight(self, method: str, path: str) -> int:
        rule = self.rate_limit_rule(method, path)
        return rule.weight if rule else 1


__all__ = ["NowaGieldaSpotAdapter", "RateLimitRule"]
