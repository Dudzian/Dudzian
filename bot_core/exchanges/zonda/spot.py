"""Adapter REST dla rynku spot giełdy Zonda."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from hashlib import sha512
import hmac
from typing import Any, Mapping, MutableMapping, Optional, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)

_LOGGER = logging.getLogger(__name__)


_BASE_URLS: Mapping[Environment, str] = {
    Environment.LIVE: "https://api.zonda.exchange/rest",
    Environment.PAPER: "https://api.zonda.exchange/rest",
    Environment.TESTNET: "https://api-sandbox.zonda.exchange/rest",
}


_CANDLE_INTERVALS: Mapping[str, str] = {
    "1m": "60",
    "5m": "300",
    "15m": "900",
    "30m": "1800",
    "1h": "3600",
    "4h": "14400",
    "1d": "86400",
}


_DEFAULT_HEADERS = {"User-Agent": "bot-core/1.0 (+https://github.com/)"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class _SignedRequest:
    path: str
    method: str
    params: Mapping[str, Any]


class ZondaSpotAdapter(ExchangeAdapter):
    """Implementacja interfejsu `ExchangeAdapter` dla Zonda Spot."""

    name = "zonda_spot"

    def __init__(self, credentials: ExchangeCredentials, *, environment: Environment) -> None:
        super().__init__(credentials)
        try:
            self._base_url = _BASE_URLS[environment]
        except KeyError as exc:  # pragma: no cover - brak konfiguracji
            raise ValueError(f"Nieobsługiwane środowisko Zonda: {environment}") from exc
        self._environment = environment
        self._permission_set = frozenset(perm.lower() for perm in credentials.permissions)
        self._http_timeout = 15
        self._ip_allowlist: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Konfiguracja sieciowa
    # ------------------------------------------------------------------
    def configure_network(self, *, ip_allowlist: Optional[Sequence[str]] = None) -> None:  # type: ignore[override]
        self._ip_allowlist = tuple(ip_allowlist) if ip_allowlist else ()
        if self._ip_allowlist:
            _LOGGER.info("Skonfigurowano allowlistę IP dla Zonda Spot: %s", self._ip_allowlist)

    # ------------------------------------------------------------------
    # Publiczne API
    # ------------------------------------------------------------------
    def fetch_symbols(self) -> Sequence[str]:  # type: ignore[override]
        payload = self._public_request("/trading/ticker")
        if not isinstance(payload, Mapping):
            return []
        items = payload.get("items")
        symbols: list[str] = []
        if isinstance(items, Mapping):
            for symbol in items.keys():
                if isinstance(symbol, str):
                    symbols.append(symbol)
        return sorted(set(symbols))

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:  # type: ignore[override]
        resolution = _CANDLE_INTERVALS.get(interval)
        if resolution is None:
            raise ValueError(f"Nieobsługiwany interwał {interval!r} dla Zonda Spot")

        params: MutableMapping[str, Any] = {}
        if start is not None:
            params["from"] = int(start)
        if end is not None:
            params["to"] = int(end)
        if limit is not None:
            params["limit"] = int(limit)

        payload = self._public_request(f"/trading/candle/history/{symbol}/{resolution}", params=params)
        items = []
        if isinstance(payload, Mapping):
            items = payload.get("items", [])  # type: ignore[assignment]

        candles: list[list[float]] = []
        if isinstance(items, Sequence):
            for entry in items:
                if isinstance(entry, Mapping):
                    timestamp = _as_float(entry.get("time"))
                    open_price = _as_float(entry.get("open")) or _as_float(entry.get("o"))
                    high_price = _as_float(entry.get("high")) or _as_float(entry.get("h"))
                    low_price = _as_float(entry.get("low")) or _as_float(entry.get("l"))
                    close_price = _as_float(entry.get("close")) or _as_float(entry.get("c"))
                    volume = _as_float(entry.get("volume")) or _as_float(entry.get("v"))
                    candles.append(
                        [
                            timestamp,
                            open_price,
                            high_price,
                            low_price,
                            close_price,
                            volume,
                        ]
                    )
        if limit is not None:
            return candles[:limit]
        return candles

    # ------------------------------------------------------------------
    # Operacje prywatne
    # ------------------------------------------------------------------
    def fetch_account_snapshot(self) -> AccountSnapshot:  # type: ignore[override]
        if "read" not in self._permission_set:
            raise PermissionError("Poświadczenia Zonda nie mają uprawnień do odczytu.")

        payload = self._signed_request(
            _SignedRequest(path="/trading/account/balances", method="GET", params={})
        )
        balances_section = payload.get("balances") if isinstance(payload, Mapping) else None
        balances: MutableMapping[str, float] = {}
        if isinstance(balances_section, Sequence):
            for entry in balances_section:
                if isinstance(entry, Mapping):
                    currency = entry.get("currency") or entry.get("symbol")
                    total = _as_float(entry.get("available")) + _as_float(entry.get("locked"))
                    if isinstance(currency, str):
                        balances[currency] = total

        summary = payload.get("summary") if isinstance(payload, Mapping) else {}
        total_equity = _as_float(summary.get("total")) if isinstance(summary, Mapping) else 0.0
        available_margin = _as_float(summary.get("available")) if isinstance(summary, Mapping) else 0.0
        maintenance_margin = _as_float(summary.get("margin")) if isinstance(summary, Mapping) else 0.0

        return AccountSnapshot(
            balances=dict(balances),
            total_equity=total_equity,
            available_margin=available_margin,
            maintenance_margin=maintenance_margin,
        )

    def place_order(self, request: OrderRequest) -> OrderResult:  # type: ignore[override]
        if "trade" not in self._permission_set:
            raise PermissionError("Poświadczenia Zonda nie mają uprawnień tradingowych.")

        payload: MutableMapping[str, Any] = {
            "symbol": request.symbol,
            "side": request.side.lower(),
            "type": request.order_type.lower(),
            "amount": f"{request.quantity:.10f}",
        }
        if request.price is not None:
            payload["price"] = f"{request.price:.10f}"
        if request.time_in_force:
            payload["timeInForce"] = request.time_in_force.upper()
        if request.client_order_id:
            payload["clientOrderId"] = request.client_order_id

        response = self._signed_request(
            _SignedRequest(path="/trading/order/new", method="POST", params=payload)
        )
        order_data = response.get("order", {}) if isinstance(response, Mapping) else {}
        return OrderResult(
            order_id=str(order_data.get("id", "")) if isinstance(order_data, Mapping) else "",
            status=str(order_data.get("status", "NEW")) if isinstance(order_data, Mapping) else "NEW",
            filled_quantity=_as_float(order_data.get("filled", 0.0)) if isinstance(order_data, Mapping) else 0.0,
            avg_price=_as_float(order_data.get("avgPrice")) if isinstance(order_data, Mapping) else None,
            raw_response=order_data if isinstance(order_data, Mapping) else {},
        )

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:  # type: ignore[override]
        if "trade" not in self._permission_set:
            raise PermissionError("Poświadczenia Zonda nie mają uprawnień tradingowych.")
        params: MutableMapping[str, Any] = {"id": order_id}
        if symbol:
            params["symbol"] = symbol
        response = self._signed_request(
            _SignedRequest(path="/trading/order/cancel", method="POST", params=params)
        )
        if not isinstance(response, Mapping) or response.get("status") not in {"Ok", "ok", "success"}:
            raise RuntimeError(f"Zonda nie potwierdziła anulowania zlecenia: {response}")

    def stream_public_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        raise NotImplementedError("Streaming danych publicznych dla Zonda nie jest jeszcze zaimplementowany")

    def stream_private_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        raise NotImplementedError("Streaming danych prywatnych dla Zonda nie jest jeszcze zaimplementowany")

    # ------------------------------------------------------------------
    # Prywatne metody pomocnicze
    # ------------------------------------------------------------------
    def _public_request(
        self,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        method: str = "GET",
    ) -> Mapping[str, Any] | list[Any]:
        query = f"?{urlencode(params)}" if params else ""
        url = f"{self._base_url}{path}{query}"
        request = Request(url, method=method, headers=dict(_DEFAULT_HEADERS))
        return self._execute_request(request)

    def _signed_request(self, ctx: _SignedRequest) -> Mapping[str, Any]:
        if not self._credentials.secret:
            raise RuntimeError("Do podpisanych endpointów Zonda wymagany jest secret klucza API.")

        url = f"{self._base_url}{ctx.path}"
        nonce = str(int(time.time() * 1000))
        body = b""
        headers = dict(_DEFAULT_HEADERS)
        headers["API-Key"] = self._credentials.key_id
        headers["API-Nonce"] = nonce

        if ctx.method.upper() == "GET":
            if ctx.params:
                query = urlencode(ctx.params)
                separator = "?" if "?" not in url else "&"
                url = f"{url}{separator}{query}"
            payload_string = ""
        else:
            payload_string = json.dumps(ctx.params)
            body = payload_string.encode("utf-8")
            headers["Content-Type"] = "application/json"

        signature_payload = f"{ctx.path}{nonce}{payload_string}".encode("utf-8")
        signature = hmac.new(
            self._credentials.secret.encode("utf-8"), signature_payload, sha512
        ).hexdigest()
        headers["API-Signature"] = signature

        request = Request(url, method=ctx.method.upper(), headers=headers, data=body or None)
        response = self._execute_request(request)
        if isinstance(response, Mapping):
            return response
        raise RuntimeError("Niepoprawna odpowiedź z prywatnego endpointu Zonda")

    @staticmethod
    def _execute_request(request: Request) -> Mapping[str, Any] | list[Any]:
        try:
            with urlopen(request, timeout=15) as response:  # nosec: B310 - zaufany endpoint
                raw = response.read()
        except HTTPError as exc:  # pragma: no cover - zależy od środowiska zewnętrznego
            _LOGGER.error("Błąd HTTP podczas komunikacji z Zonda: %s", exc)
            raise RuntimeError(f"Zonda API zwróciło błąd HTTP: {exc}") from exc
        except URLError as exc:  # pragma: no cover - zależy od sieci
            _LOGGER.error("Błąd sieci podczas komunikacji z Zonda: %s", exc)
            raise RuntimeError("Nie udało się połączyć z API Zonda") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - błędy po stronie API
            _LOGGER.error("Niepoprawna odpowiedź JSON z API Zonda: %s", exc)
            raise RuntimeError("Niepoprawna odpowiedź JSON z API Zonda") from exc
        return data
