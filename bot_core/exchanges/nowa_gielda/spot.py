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
        RateLimitRule("GET", "/public/ohlcv", weight=2, window_seconds=1.0, max_requests=10),
        RateLimitRule("GET", "/private/account", weight=2, window_seconds=1.0, max_requests=10),
        RateLimitRule("GET", "/private/orders", weight=3, window_seconds=1.0, max_requests=5),
        RateLimitRule("GET", "/private/orders/history", weight=3, window_seconds=1.0, max_requests=5),
        RateLimitRule("GET", "/private/trades", weight=3, window_seconds=1.0, max_requests=5),
        RateLimitRule("GET", "/private/deposits", weight=2, window_seconds=1.0, max_requests=5),
        RateLimitRule("GET", "/private/withdrawals", weight=3, window_seconds=1.0, max_requests=5),
        RateLimitRule("GET", "/private/fees", weight=2, window_seconds=1.0, max_requests=5),
        RateLimitRule("GET", "/private/rebates", weight=2, window_seconds=1.0, max_requests=5),
        RateLimitRule("GET", "/private/interest", weight=2, window_seconds=1.0, max_requests=5),
        RateLimitRule("GET", "/private/transfers", weight=2, window_seconds=1.0, max_requests=5),
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

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        *,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Mapping[str, Any]:
        params = _strip_none(
            {
                "symbol": symbol,
                "interval": interval,
                "start": start,
                "end": end,
                "limit": limit,
            }
        )
        return self._request("GET", "/public/ohlcv", params=params)

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

    def fetch_account(
        self,
        *,
        headers: Mapping[str, str],
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/account", headers=headers)

    def fetch_trades(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/trades", params=params, headers=headers)

    def fetch_open_orders(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/orders", params=params, headers=headers)

    def fetch_order_history(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/orders/history", params=params, headers=headers)

    def fetch_deposits(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/deposits", params=params, headers=headers)

    def fetch_withdrawals(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/withdrawals", params=params, headers=headers)

    def fetch_fees(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/fees", params=params, headers=headers)

    def fetch_transfers(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/transfers", params=params, headers=headers)

    def fetch_rebates(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/rebates", params=params, headers=headers)

    def fetch_interest(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._request("GET", "/private/interest", params=params, headers=headers)


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
        timestamp = self._timestamp()
        signature = self.sign_request(timestamp, "GET", "/private/account")
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_account(headers=headers)

        raw_balances = payload.get("balances", [])
        if not isinstance(raw_balances, Sequence):
            raise ExchangeAPIError(
                message="Niepoprawny format listy sald konta",
                status_code=200,
                payload=payload,
            )

        balances: dict[str, float] = {}
        for entry in raw_balances:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja salda ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            asset = entry.get("asset")
            total = entry.get("total")
            if not asset:
                raise ExchangeAPIError(
                    message="Brak identyfikatora waluty w saldzie",
                    status_code=200,
                    payload=payload,
                )
            try:
                balances[str(asset)] = float(total)
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Niepoprawna wartość salda",
                    status_code=200,
                    payload=payload,
                ) from exc

        def _float_field(name: str, default: float = 0.0) -> float:
            value = payload.get(name, default)
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message=f"Niepoprawna wartość pola {name}",
                    status_code=200,
                    payload=payload,
                ) from exc

        return AccountSnapshot(
            balances=balances,
            total_equity=_float_field("totalEquity"),
            available_margin=_float_field("availableMargin"),
            maintenance_margin=_float_field("maintenanceMargin"),
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
        exchange_symbol = symbols.to_exchange_symbol(symbol)
        payload = self._http_client.fetch_ohlcv(
            exchange_symbol,
            interval,
            start=start,
            end=end,
            limit=limit,
        )

        response_symbol = payload.get("symbol")
        if response_symbol and symbols.to_internal_symbol(response_symbol) != symbol:
            raise ExchangeAPIError(
                message="Symbol w odpowiedzi OHLCV nie zgadza się z zapytaniem",
                status_code=200,
                payload=payload,
            )

        raw_candles = payload.get("candles", [])
        if not isinstance(raw_candles, Sequence):
            raise ExchangeAPIError(
                message="Lista świec ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        candles: list[list[float]] = []
        for candle in raw_candles:
            if not isinstance(candle, Sequence) or len(candle) < 6:
                raise ExchangeAPIError(
                    message="Świeca ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            open_time, open_price, high_price, low_price, close_price, volume = candle[:6]
            try:
                candles.append(
                    [
                        float(open_time),
                        float(open_price),
                        float(high_price),
                        float(low_price),
                        float(close_price),
                        float(volume),
                    ]
                )
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Niepoprawne wartości liczby w świecy",
                    status_code=200,
                    payload=payload,
                ) from exc

        return candles

    def fetch_trades_history(
        self,
        *,
        symbol: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Mapping[str, Any]]:
        params = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(symbol) if symbol else None,
                "start": start,
                "end": end,
                "limit": limit,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(
            timestamp,
            "GET",
            "/private/trades",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_trades(headers=headers, params=params)

        raw_trades = payload.get("trades", [])
        if not isinstance(raw_trades, Sequence):
            raise ExchangeAPIError(
                message="Lista transakcji ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        trades: list[Mapping[str, Any]] = []
        for entry in raw_trades:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja historii transakcji ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            entry_symbol = entry.get("symbol")
            if not entry_symbol:
                raise ExchangeAPIError(
                    message="Transakcja nie zawiera symbolu",
                    status_code=200,
                    payload=payload,
                )

            internal_symbol = symbols.to_internal_symbol(str(entry_symbol))
            if symbol and internal_symbol != symbol:
                raise ExchangeAPIError(
                    message="Symbol transakcji nie zgadza się z filtrem",
                    status_code=200,
                    payload=payload,
                )

            trade_id_raw = entry.get("tradeId") or entry.get("id")
            if trade_id_raw is None:
                raise ExchangeAPIError(
                    message="Transakcja nie posiada identyfikatora",
                    status_code=200,
                    payload=payload,
                )

            order_id_raw = entry.get("orderId")
            side_raw = entry.get("side")
            price_raw = entry.get("price")
            qty_raw = entry.get("quantity") or entry.get("qty")
            fee_raw = entry.get("fee")
            timestamp_raw = entry.get("timestamp") or entry.get("time")

            try:
                trade = {
                    "trade_id": str(trade_id_raw),
                    "order_id": str(order_id_raw) if order_id_raw is not None else None,
                    "symbol": internal_symbol,
                    "side": str(side_raw) if side_raw is not None else "",
                    "price": float(price_raw),
                    "quantity": float(qty_raw),
                    "fee": float(fee_raw) if fee_raw is not None else None,
                    "timestamp": float(timestamp_raw),
                    "raw": dict(entry),
                }
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Transakcja zawiera niepoprawne wartości liczbowe",
                    status_code=200,
                    payload=payload,
                ) from exc

            trades.append(trade)

        return trades

    def fetch_open_orders(
        self,
        *,
        symbol: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Mapping[str, Any]]:
        params = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(symbol) if symbol else None,
                "limit": limit,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(
            timestamp,
            "GET",
            "/private/orders",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_open_orders(headers=headers, params=params)

        raw_orders = payload.get("orders", [])
        if not isinstance(raw_orders, Sequence):
            raise ExchangeAPIError(
                message="Lista zleceń ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        orders: list[Mapping[str, Any]] = []
        for entry in raw_orders:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja zlecenia ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            entry_symbol = entry.get("symbol")
            if not entry_symbol:
                raise ExchangeAPIError(
                    message="Zlecenie nie zawiera symbolu",
                    status_code=200,
                    payload=payload,
                )

            internal_symbol = symbols.to_internal_symbol(str(entry_symbol))
            if symbol and internal_symbol != symbol:
                raise ExchangeAPIError(
                    message="Symbol zlecenia nie zgadza się z filtrem",
                    status_code=200,
                    payload=payload,
                )

            order_id_raw = entry.get("orderId") or entry.get("id")
            status_raw = entry.get("status")
            side_raw = entry.get("side")
            type_raw = entry.get("type")
            price_raw = entry.get("price")
            avg_price_raw = entry.get("avgPrice")
            quantity_raw = entry.get("quantity") or entry.get("qty")
            filled_raw = entry.get("filledQuantity") or entry.get("filled")
            timestamp_raw = entry.get("timestamp") or entry.get("createdAt")

            if order_id_raw is None:
                raise ExchangeAPIError(
                    message="Zlecenie nie posiada identyfikatora",
                    status_code=200,
                    payload=payload,
                )

            try:
                order = {
                    "order_id": str(order_id_raw),
                    "symbol": internal_symbol,
                    "status": str(status_raw) if status_raw is not None else "",
                    "side": str(side_raw) if side_raw is not None else "",
                    "type": str(type_raw) if type_raw is not None else "",
                    "price": float(price_raw) if price_raw is not None else None,
                    "avg_price": float(avg_price_raw) if avg_price_raw is not None else None,
                    "quantity": float(quantity_raw),
                    "filled_quantity": float(filled_raw) if filled_raw is not None else 0.0,
                    "timestamp": float(timestamp_raw)
                    if timestamp_raw is not None
                    else float(timestamp),
                    "raw": dict(entry),
                }
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Zlecenie zawiera niepoprawne wartości liczbowe",
                    status_code=200,
                    payload=payload,
                ) from exc

            orders.append(order)

        return orders

    def fetch_closed_orders(
        self,
        *,
        symbol: Optional[str] = None,
        limit: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Sequence[Mapping[str, Any]]:
        params = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(symbol) if symbol else None,
                "limit": limit,
                "start": start,
                "end": end,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(
            timestamp,
            "GET",
            "/private/orders/history",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_order_history(headers=headers, params=params)

        raw_orders = payload.get("orders", [])
        if not isinstance(raw_orders, Sequence):
            raise ExchangeAPIError(
                message="Lista zamkniętych zleceń ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        orders: list[Mapping[str, Any]] = []
        for entry in raw_orders:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja zamkniętego zlecenia ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            entry_symbol = entry.get("symbol")
            if not entry_symbol:
                raise ExchangeAPIError(
                    message="Zamknięte zlecenie nie zawiera symbolu",
                    status_code=200,
                    payload=payload,
                )

            internal_symbol = symbols.to_internal_symbol(str(entry_symbol))
            if symbol and internal_symbol != symbol:
                raise ExchangeAPIError(
                    message="Symbol zamkniętego zlecenia nie zgadza się z filtrem",
                    status_code=200,
                    payload=payload,
                )

            order_id_raw = entry.get("orderId") or entry.get("id")
            status_raw = entry.get("status")
            side_raw = entry.get("side")
            type_raw = entry.get("type")
            price_raw = entry.get("price")
            avg_price_raw = entry.get("avgPrice")
            quantity_raw = entry.get("quantity") or entry.get("qty")
            filled_raw = (
                entry.get("executedQuantity")
                or entry.get("filledQuantity")
                or entry.get("filled")
            )
            created_raw = entry.get("timestamp") or entry.get("createdAt")
            closed_raw = entry.get("closedAt") or entry.get("updatedAt")

            if order_id_raw is None:
                raise ExchangeAPIError(
                    message="Zamknięte zlecenie nie posiada identyfikatora",
                    status_code=200,
                    payload=payload,
                )

            try:
                order = {
                    "order_id": str(order_id_raw),
                    "symbol": internal_symbol,
                    "status": str(status_raw) if status_raw is not None else "",
                    "side": str(side_raw) if side_raw is not None else "",
                    "type": str(type_raw) if type_raw is not None else "",
                    "price": float(price_raw) if price_raw is not None else None,
                    "avg_price": float(avg_price_raw) if avg_price_raw is not None else None,
                    "quantity": float(quantity_raw),
                    "filled_quantity": float(filled_raw) if filled_raw is not None else 0.0,
                    "timestamp": float(created_raw)
                    if created_raw is not None
                    else float(timestamp),
                    "closed_timestamp": float(closed_raw)
                    if closed_raw is not None
                    else None,
                    "raw": dict(entry),
                }
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Zamknięte zlecenie zawiera niepoprawne wartości liczbowe",
                    status_code=200,
                    payload=payload,
                ) from exc

            orders.append(order)

        return orders

    def fetch_deposits_history(
        self,
        *,
        symbol: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
        status: Optional[str] = None,
    ) -> Sequence[Mapping[str, Any]]:
        params = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(symbol) if symbol else None,
                "start": start,
                "end": end,
                "limit": limit,
                "status": status,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(
            timestamp,
            "GET",
            "/private/deposits",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_deposits(headers=headers, params=params)

        raw_deposits = payload.get("deposits", [])
        if not isinstance(raw_deposits, Sequence):
            raise ExchangeAPIError(
                message="Lista depozytów ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        deposits: list[Mapping[str, Any]] = []
        for entry in raw_deposits:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja depozytu ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            entry_symbol = entry.get("symbol")
            if not entry_symbol:
                raise ExchangeAPIError(
                    message="Depozyt nie zawiera symbolu",
                    status_code=200,
                    payload=payload,
                )

            internal_symbol = symbols.to_internal_symbol(str(entry_symbol))
            if symbol and internal_symbol != symbol:
                raise ExchangeAPIError(
                    message="Symbol depozytu nie zgadza się z filtrem",
                    status_code=200,
                    payload=payload,
                )

            amount_raw = entry.get("amount")
            timestamp_raw = entry.get("timestamp") or entry.get("createdAt")
            if amount_raw is None:
                raise ExchangeAPIError(
                    message="Depozyt nie zawiera kwoty",
                    status_code=200,
                    payload=payload,
                )

            if timestamp_raw is None:
                timestamp_raw = timestamp

            fee_raw = entry.get("fee")
            completed_raw = entry.get("completedAt") or entry.get("updatedAt")
            network_raw = entry.get("network")
            tx_id_raw = entry.get("txId") or entry.get("transactionId")

            try:
                deposit = {
                    "transfer_id": str(entry.get("depositId") or entry.get("id") or entry.get("reference")),
                    "symbol": internal_symbol,
                    "status": str(entry.get("status", "")),
                    "amount": float(amount_raw),
                    "fee": float(fee_raw) if fee_raw is not None else None,
                    "network": str(network_raw) if network_raw is not None else "",
                    "tx_id": str(tx_id_raw) if tx_id_raw is not None else "",
                    "timestamp": float(timestamp_raw),
                    "completed_timestamp": float(completed_raw)
                    if completed_raw is not None
                    else None,
                    "raw": dict(entry),
                }
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Depozyt zawiera niepoprawne wartości liczbowe",
                    status_code=200,
                    payload=payload,
                ) from exc

            deposits.append(deposit)

        return deposits

    def fetch_withdrawals_history(
        self,
        *,
        symbol: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
        status: Optional[str] = None,
    ) -> Sequence[Mapping[str, Any]]:
        params = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(symbol) if symbol else None,
                "start": start,
                "end": end,
                "limit": limit,
                "status": status,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(
            timestamp,
            "GET",
            "/private/withdrawals",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_withdrawals(headers=headers, params=params)

        raw_withdrawals = payload.get("withdrawals", [])
        if not isinstance(raw_withdrawals, Sequence):
            raise ExchangeAPIError(
                message="Lista wypłat ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        withdrawals: list[Mapping[str, Any]] = []
        for entry in raw_withdrawals:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja wypłaty ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            entry_symbol = entry.get("symbol")
            if not entry_symbol:
                raise ExchangeAPIError(
                    message="Wypłata nie zawiera symbolu",
                    status_code=200,
                    payload=payload,
                )

            internal_symbol = symbols.to_internal_symbol(str(entry_symbol))
            if symbol and internal_symbol != symbol:
                raise ExchangeAPIError(
                    message="Symbol wypłaty nie zgadza się z filtrem",
                    status_code=200,
                    payload=payload,
                )

            amount_raw = entry.get("amount")
            fee_raw = entry.get("fee")
            timestamp_raw = entry.get("timestamp") or entry.get("createdAt")
            completed_raw = entry.get("completedAt") or entry.get("updatedAt")
            if amount_raw is None:
                raise ExchangeAPIError(
                    message="Wypłata nie zawiera kwoty",
                    status_code=200,
                    payload=payload,
                )

            if timestamp_raw is None:
                timestamp_raw = timestamp

            network_raw = entry.get("network")
            tx_id_raw = entry.get("txId") or entry.get("transactionId")
            address_raw = entry.get("address") or entry.get("destination")
            tag_raw = entry.get("tag") or entry.get("memo")

            try:
                withdrawal = {
                    "transfer_id": str(entry.get("withdrawalId") or entry.get("id") or entry.get("reference")),
                    "symbol": internal_symbol,
                    "status": str(entry.get("status", "")),
                    "amount": float(amount_raw),
                    "fee": float(fee_raw) if fee_raw is not None else None,
                    "network": str(network_raw) if network_raw is not None else "",
                    "address": str(address_raw) if address_raw is not None else "",
                    "tag": str(tag_raw) if tag_raw is not None else "",
                    "tx_id": str(tx_id_raw) if tx_id_raw is not None else "",
                    "timestamp": float(timestamp_raw),
                    "completed_timestamp": float(completed_raw)
                    if completed_raw is not None
                    else None,
                    "raw": dict(entry),
                }
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Wypłata zawiera niepoprawne wartości liczbowe",
                    status_code=200,
                    payload=payload,
                ) from exc

            withdrawals.append(withdrawal)

        return withdrawals

    def fetch_transfers_history(
        self,
        *,
        symbol: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
        status: Optional[str] = None,
        direction: Optional[str] = None,
        from_account: Optional[str] = None,
        to_account: Optional[str] = None,
    ) -> Sequence[Mapping[str, Any]]:
        params = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(symbol) if symbol else None,
                "start": start,
                "end": end,
                "limit": limit,
                "status": status,
                "direction": direction,
                "from": from_account,
                "to": to_account,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(
            timestamp,
            "GET",
            "/private/transfers",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_transfers(headers=headers, params=params)

        raw_transfers = payload.get("transfers", [])
        if not isinstance(raw_transfers, Sequence):
            raise ExchangeAPIError(
                message="Lista transferów ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        transfers: list[Mapping[str, Any]] = []
        for entry in raw_transfers:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja transferu ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            entry_symbol = entry.get("symbol")
            if not entry_symbol:
                raise ExchangeAPIError(
                    message="Transfer nie zawiera symbolu",
                    status_code=200,
                    payload=payload,
                )

            internal_symbol = symbols.to_internal_symbol(str(entry_symbol))
            if symbol and internal_symbol != symbol:
                raise ExchangeAPIError(
                    message="Symbol transferu nie zgadza się z filtrem",
                    status_code=200,
                    payload=payload,
                )

            transfer_id_raw = entry.get("transferId") or entry.get("id")
            amount_raw = entry.get("amount")
            status_raw = entry.get("status")
            from_account_raw = entry.get("fromAccount") or entry.get("from")
            to_account_raw = entry.get("toAccount") or entry.get("to")
            timestamp_raw = entry.get("timestamp") or entry.get("createdAt")
            completed_raw = entry.get("completedAt") or entry.get("updatedAt")

            if transfer_id_raw is None:
                raise ExchangeAPIError(
                    message="Transfer nie posiada identyfikatora",
                    status_code=200,
                    payload=payload,
                )

            try:
                transfer = {
                    "transfer_id": str(transfer_id_raw),
                    "symbol": internal_symbol,
                    "amount": float(amount_raw),
                    "status": str(status_raw) if status_raw is not None else "",
                    "from_account": str(from_account_raw) if from_account_raw is not None else "",
                    "to_account": str(to_account_raw) if to_account_raw is not None else "",
                    "timestamp": float(timestamp_raw)
                    if timestamp_raw is not None
                    else float(timestamp),
                    "completed_timestamp": float(completed_raw)
                    if completed_raw is not None
                    else None,
                    "raw": dict(entry),
                }
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Transfer zawiera niepoprawne wartości liczbowe",
                    status_code=200,
                    payload=payload,
                ) from exc

            transfers.append(transfer)

        return transfers

    def fetch_fee_rates(
        self,
        *,
        symbol: Optional[str] = None,
    ) -> Sequence[Mapping[str, Any]]:
        params = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(symbol) if symbol else None,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(
            timestamp,
            "GET",
            "/private/fees",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_fees(headers=headers, params=params)

        raw_fees = payload.get("fees", [])
        if not isinstance(raw_fees, Sequence):
            raise ExchangeAPIError(
                message="Lista stawek prowizyjnych ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        fees: list[Mapping[str, Any]] = []
        for entry in raw_fees:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja stawek prowizyjnych ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            entry_symbol = entry.get("symbol")
            if not entry_symbol:
                raise ExchangeAPIError(
                    message="Stawka prowizyjna nie zawiera symbolu",
                    status_code=200,
                    payload=payload,
                )

            internal_symbol = symbols.to_internal_symbol(str(entry_symbol))
            if symbol and internal_symbol != symbol:
                raise ExchangeAPIError(
                    message="Symbol stawek prowizyjnych nie zgadza się z filtrem",
                    status_code=200,
                    payload=payload,
                )

            maker_raw = entry.get("maker") or entry.get("makerFee")
            taker_raw = entry.get("taker") or entry.get("takerFee")
            volume_raw = (
                entry.get("thirtyDayVolume")
                or entry.get("volume30d")
                or entry.get("thirtyDayTurnover")
            )

            if maker_raw is None or taker_raw is None:
                raise ExchangeAPIError(
                    message="Stawka prowizyjna nie zawiera wartości maker/taker",
                    status_code=200,
                    payload=payload,
                )

            try:
                fee_entry = {
                    "symbol": internal_symbol,
                    "maker_fee": float(maker_raw),
                    "taker_fee": float(taker_raw),
                    "thirty_day_volume": float(volume_raw) if volume_raw is not None else None,
                    "raw": dict(entry),
                }
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Stawka prowizyjna zawiera niepoprawne wartości liczbowe",
                    status_code=200,
                    payload=payload,
                ) from exc

            fees.append(fee_entry)

        return fees

    def fetch_rebates_history(
        self,
        *,
        symbol: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
        rebate_type: Optional[str] = None,
    ) -> Sequence[Mapping[str, Any]]:
        params = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(symbol) if symbol else None,
                "start": start,
                "end": end,
                "limit": limit,
                "type": rebate_type,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(
            timestamp,
            "GET",
            "/private/rebates",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_rebates(headers=headers, params=params)

        raw_rebates = payload.get("rebates", [])
        if not isinstance(raw_rebates, Sequence):
            raise ExchangeAPIError(
                message="Lista zwrotów prowizyjnych ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        rebates: list[Mapping[str, Any]] = []
        for entry in raw_rebates:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja zwrotu prowizyjnego ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            entry_symbol = entry.get("symbol")
            if not entry_symbol:
                raise ExchangeAPIError(
                    message="Zwrot prowizyjny nie zawiera symbolu",
                    status_code=200,
                    payload=payload,
                )

            internal_symbol = symbols.to_internal_symbol(str(entry_symbol))
            if symbol and internal_symbol != symbol:
                raise ExchangeAPIError(
                    message="Symbol zwrotu prowizyjnego nie zgadza się z filtrem",
                    status_code=200,
                    payload=payload,
                )

            rebate_id_raw = entry.get("rebateId") or entry.get("id") or entry.get("reference")
            amount_raw = entry.get("amount")
            rate_raw = entry.get("rate") or entry.get("feeRate")
            type_raw = entry.get("type") or entry.get("feeType")
            order_id_raw = entry.get("orderId")
            timestamp_raw = entry.get("timestamp") or entry.get("createdAt")
            settled_raw = entry.get("settledAt") or entry.get("updatedAt")

            if rebate_id_raw is None:
                raise ExchangeAPIError(
                    message="Zwrot prowizyjny nie posiada identyfikatora",
                    status_code=200,
                    payload=payload,
                )

            if amount_raw is None:
                raise ExchangeAPIError(
                    message="Zwrot prowizyjny nie zawiera kwoty",
                    status_code=200,
                    payload=payload,
                )

            if timestamp_raw is None:
                timestamp_raw = timestamp

            try:
                rebate = {
                    "rebate_id": str(rebate_id_raw),
                    "symbol": internal_symbol,
                    "amount": float(amount_raw),
                    "rate": float(rate_raw) if rate_raw is not None else None,
                    "type": str(type_raw) if type_raw is not None else "",
                    "order_id": str(order_id_raw) if order_id_raw is not None else "",
                    "timestamp": float(timestamp_raw),
                    "settled_timestamp": float(settled_raw) if settled_raw is not None else None,
                    "raw": dict(entry),
                }
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Zwrot prowizyjny zawiera niepoprawne wartości liczbowe",
                    status_code=200,
                    payload=payload,
                ) from exc

            rebates.append(rebate)

        return rebates

    def fetch_interest_history(
        self,
        *,
        symbol: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Mapping[str, Any]]:
        params = _strip_none(
            {
                "symbol": symbols.to_exchange_symbol(symbol) if symbol else None,
                "start": start,
                "end": end,
                "limit": limit,
            }
        )
        timestamp = self._timestamp()
        signature = self.sign_request(
            timestamp,
            "GET",
            "/private/interest",
            params=params,
        )
        headers = self.build_auth_headers(timestamp, signature)
        payload = self._http_client.fetch_interest(headers=headers, params=params)

        raw_interest = payload.get("interest", [])
        if not isinstance(raw_interest, Sequence):
            raise ExchangeAPIError(
                message="Lista naliczonych odsetek ma niepoprawny format",
                status_code=200,
                payload=payload,
            )

        records: list[Mapping[str, Any]] = []
        for entry in raw_interest:
            if not isinstance(entry, Mapping):
                raise ExchangeAPIError(
                    message="Pozycja odsetek ma niepoprawny format",
                    status_code=200,
                    payload=payload,
                )

            entry_symbol = entry.get("symbol")
            if not entry_symbol:
                raise ExchangeAPIError(
                    message="Rekord odsetek nie zawiera symbolu",
                    status_code=200,
                    payload=payload,
                )

            internal_symbol = symbols.to_internal_symbol(str(entry_symbol))
            if symbol and internal_symbol != symbol:
                raise ExchangeAPIError(
                    message="Symbol odsetek nie zgadza się z filtrem",
                    status_code=200,
                    payload=payload,
                )

            interest_id_raw = entry.get("interestId") or entry.get("id")
            if interest_id_raw is None:
                raise ExchangeAPIError(
                    message="Rekord odsetek nie posiada identyfikatora",
                    status_code=200,
                    payload=payload,
                )

            amount_raw = entry.get("amount")
            rate_raw = entry.get("rate")
            type_raw = entry.get("type")
            order_id_raw = entry.get("orderId")
            timestamp_raw = entry.get("timestamp")
            accrual_raw = entry.get("accrualTimestamp")
            if timestamp_raw is None:
                timestamp_raw = timestamp

            try:
                record = {
                    "interest_id": str(interest_id_raw),
                    "symbol": internal_symbol,
                    "amount": float(amount_raw),
                    "rate": float(rate_raw) if rate_raw is not None else None,
                    "type": str(type_raw) if type_raw is not None else "",
                    "order_id": str(order_id_raw) if order_id_raw is not None else "",
                    "timestamp": float(timestamp_raw),
                    "accrual_timestamp": float(accrual_raw) if accrual_raw is not None else None,
                    "raw": dict(entry),
                }
            except (TypeError, ValueError) as exc:
                raise ExchangeAPIError(
                    message="Rekord odsetek zawiera niepoprawne wartości liczbowe",
                    status_code=200,
                    payload=payload,
                ) from exc

            records.append(record)

        return records

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
