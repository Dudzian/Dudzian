"""Adapter REST dla rynku spot Zonda (dawniej BitBay)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from hashlib import sha512
import hmac
from typing import Iterable, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)

_LOGGER = logging.getLogger(__name__)

_API_BASE_URL = "https://api.zonda.exchange/rest"
_DEFAULT_HEADERS = {"User-Agent": "bot-core/1.0 (+https://github.com/)"}


def _normalize_interval(interval: str) -> int:
    """Konwertuje interwał tekstowy na sekundy wymagane przez API świec."""

    mapping = {
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1_800,
        "1h": 3_600,
        "4h": 14_400,
        "6h": 21_600,
        "12h": 43_200,
        "1d": 86_400,
        "1w": 604_800,
    }
    key = interval.strip().lower()
    if key.endswith("min"):
        key = key[:-3] + "m"
    if key.endswith("h") and key[:-1].isdigit():
        key = key
    if key.endswith("d") and key[:-1].isdigit():
        key = key
    if key.endswith("w") and key[:-1].isdigit():
        key = key
    if key not in mapping and key.endswith("m") and key[:-1].isdigit():
        mapping[key] = int(key[:-1]) * 60
    try:
        return mapping[key]
    except KeyError as exc:  # pragma: no cover - walidacja zostanie wychwycona w testach integracyjnych
        raise ValueError(f"Nieobsługiwany interwał Zonda: {interval}") from exc


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _json_body(payload: Mapping[str, object] | None) -> str:
    if not payload:
        return ""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


@dataclass(slots=True)
class _OrderPayload:
    order_id: str
    status: str
    filled_quantity: float
    avg_price: Optional[float]
    raw: Mapping[str, object]


class ZondaSpotAdapter(ExchangeAdapter):
    """Adapter REST obsługujący podstawowe operacje tradingowe Zonda."""

    __slots__ = (
        "_environment",
        "_base_url",
        "_ip_allowlist",
        "_permission_set",
    )

    name: str = "zonda_spot"

    def __init__(self, credentials: ExchangeCredentials, *, environment: Environment | None = None) -> None:
        super().__init__(credentials)
        self._environment = environment or credentials.environment
        self._base_url = _API_BASE_URL
        self._ip_allowlist: tuple[str, ...] = ()
        self._permission_set = frozenset(perm.lower() for perm in credentials.permissions)

    # ------------------------------------------------------------------
    # Pomocnicze metody HTTP
    def _build_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self._base_url}{path}"

    def _execute_request(self, request: Request) -> dict[str, object] | list[object]:
        try:
            with urlopen(request, timeout=15) as response:  # nosec: B310 - kontrolowany endpoint
                payload = response.read()
        except HTTPError as exc:  # pragma: no cover - zależne od sieci
            _LOGGER.error("Zonda zwróciła błąd HTTP: %s", exc)
            raise RuntimeError(f"Zonda API zwróciła błąd HTTP: {exc}") from exc
        except URLError as exc:  # pragma: no cover - zależne od sieci
            _LOGGER.error("Błąd sieci podczas komunikacji z Zonda: %s", exc)
            raise RuntimeError("Nie udało się połączyć z API Zonda") from exc

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:  # pragma: no cover
            _LOGGER.error("Niepoprawna odpowiedź JSON od Zonda: %s", exc)
            raise RuntimeError("Niepoprawna odpowiedź API Zonda") from exc
        return data

    def _public_request(
        self,
        path: str,
        *,
        params: Optional[Mapping[str, object]] = None,
        method: str = "GET",
    ) -> dict[str, object] | list[object]:
        query = f"?{urlencode(params or {})}" if params else ""
        request = Request(self._build_url(path) + query, headers=dict(_DEFAULT_HEADERS), method=method)
        return self._execute_request(request)

    def _signed_request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, object]] = None,
        data: Optional[Mapping[str, object]] = None,
    ) -> dict[str, object] | list[object]:
        if not self._credentials.secret:
            raise RuntimeError("Poświadczenia Zonda wymagają secret do podpisywania żądań prywatnych.")

        body = _json_body(data)
        timestamp = str(int(time.time() * 1000))
        payload = f"{timestamp}{method.upper()}{path}{body}"
        signature = hmac.new(
            self._credentials.secret.encode("utf-8"),
            payload.encode("utf-8"),
            sha512,
        ).hexdigest()

        headers = dict(_DEFAULT_HEADERS)
        headers.update(
            {
                "API-Key": self._credentials.key_id,
                "API-Hash": signature,
                "Request-Timestamp": timestamp,
            }
        )

        if params:
            query = f"?{urlencode(params)}"
        else:
            query = ""

        data_bytes: Optional[bytes] = None
        if body:
            data_bytes = body.encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            self._build_url(path) + query,
            headers=headers,
            data=data_bytes,
            method=method,
        )
        return self._execute_request(request)

    # ------------------------------------------------------------------
    # Interfejs ExchangeAdapter
    def configure_network(self, *, ip_allowlist: Optional[Sequence[str]] = None) -> None:
        self._ip_allowlist = tuple(ip_allowlist or ())
        if self._ip_allowlist:
            _LOGGER.info("Zonda allowlist IP ustawiony na: %s", self._ip_allowlist)

    def fetch_account_snapshot(self) -> AccountSnapshot:
        if "read" not in self._permission_set and "trade" not in self._permission_set:
            raise PermissionError("Poświadczenia Zonda nie mają uprawnień do odczytu sald.")

        response = self._signed_request("POST", "/trading/balance")
        if not isinstance(response, Mapping):
            raise RuntimeError("Niepoprawna odpowiedź balansu z Zonda")

        balances_section = response.get("balances", [])
        balances: dict[str, float] = {}
        available_total = 0.0
        total_equity = 0.0
        if isinstance(balances_section, list):
            for entry in balances_section:
                if not isinstance(entry, Mapping):
                    continue
                currency = entry.get("currency") or entry.get("code")
                available = _to_float(entry.get("available"))
                locked = _to_float(entry.get("locked") or entry.get("reserved"))
                if isinstance(currency, str):
                    balances[currency] = available + locked
                    available_total += available
                    total_equity += available + locked

        return AccountSnapshot(
            balances=balances,
            total_equity=total_equity,
            available_margin=available_total,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self) -> Iterable[str]:
        response = self._public_request("/trading/ticker")
        if not isinstance(response, Mapping):
            raise RuntimeError("Niepoprawna odpowiedź ticker z Zonda")
        items = response.get("items")
        if isinstance(items, Mapping):
            return [str(symbol) for symbol in items.keys()]
        raise RuntimeError("Brak sekcji 'items' w odpowiedzi ticker Zonda")

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        resolution = _normalize_interval(interval)
        params: dict[str, object] = {}
        if start is not None:
            params["from"] = int(start // 1000)
        if end is not None:
            params["to"] = int(end // 1000)
        if limit is not None:
            params["limit"] = int(limit)

        path = f"/trading/candle/history/{symbol}/{resolution}"
        response = self._public_request(path, params=params)
        if not isinstance(response, Mapping):
            raise RuntimeError("Niepoprawna odpowiedź świec z Zonda")
        items = response.get("items")
        if not isinstance(items, list):
            raise RuntimeError("Sekcja 'items' odpowiedzi świec Zonda ma niepoprawny format")

        candles: list[Sequence[float]] = []
        for entry in items:
            if not isinstance(entry, Mapping):
                continue
            timestamp = int(_to_float(entry.get("time"))) * 1000
            open_price = _to_float(entry.get("open"))
            close_price = _to_float(entry.get("close"))
            high_price = _to_float(entry.get("high"))
            low_price = _to_float(entry.get("low"))
            volume = _to_float(entry.get("volume"))
            candles.append([float(timestamp), open_price, high_price, low_price, close_price, volume])
        return candles

    def _parse_order_payload(self, response: Mapping[str, object]) -> _OrderPayload:
        order_payload: Mapping[str, object]
        if "order" in response and isinstance(response["order"], Mapping):
            order_payload = response["order"]  # type: ignore[assignment]
        elif "offer" in response and isinstance(response["offer"], Mapping):
            order_payload = response["offer"]  # type: ignore[assignment]
        else:
            order_payload = response

        order_id = str(order_payload.get("id") or order_payload.get("orderId") or order_payload.get("offerId") or "")
        status = str(order_payload.get("status", "UNKNOWN"))
        filled = _to_float(
            order_payload.get("filled")
            or order_payload.get("filledAmount")
            or order_payload.get("executed")
            or order_payload.get("amountFilled")
        )
        avg_price_raw = (
            order_payload.get("avgPrice")
            or order_payload.get("averagePrice")
            or order_payload.get("price")
        )
        avg_price = _to_float(avg_price_raw) if avg_price_raw is not None else None
        return _OrderPayload(
            order_id=order_id,
            status=status.upper(),
            filled_quantity=filled,
            avg_price=avg_price,
            raw=order_payload,
        )

    def place_order(self, request: OrderRequest) -> OrderResult:
        if "trade" not in self._permission_set:
            raise PermissionError("Poświadczenia Zonda nie mają uprawnień tradingowych.")

        payload: dict[str, object] = {
            "market": request.symbol,
            "side": request.side.lower(),
            "type": request.order_type.lower(),
            "amount": str(request.quantity),
        }
        if request.price is not None:
            payload["price"] = str(request.price)
        if request.time_in_force:
            payload["timeInForce"] = request.time_in_force
        if request.client_order_id:
            payload["clientOrderId"] = request.client_order_id

        response = self._signed_request("POST", "/trading/offer", data=payload)
        if not isinstance(response, Mapping):
            raise RuntimeError("Niepoprawna odpowiedź zlecenia z Zonda")
        order = self._parse_order_payload(response)
        return OrderResult(
            order_id=order.order_id,
            status=order.status,
            filled_quantity=order.filled_quantity,
            avg_price=order.avg_price,
            raw_response=dict(response),
        )

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:
        del symbol
        response = self._signed_request("DELETE", f"/trading/order/{order_id}")
        if isinstance(response, Mapping):
            order = self._parse_order_payload(response)
            if order.status in {"CANCELLED", "CANCELED", "REJECTED"}:
                return
        raise RuntimeError(f"Nieoczekiwana odpowiedź anulowania Zonda: {response}")

    def stream_public_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        raise NotImplementedError("Streaming Zonda zostanie dodany w przyszłym etapie.")

    def stream_private_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        raise NotImplementedError("Streaming prywatny Zonda zostanie dodany w przyszłym etapie.")


__all__ = ["ZondaSpotAdapter"]
