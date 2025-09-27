"""Adapter REST dla kontraktów terminowych Binance (USD-M)."""
from __future__ import annotations

import hmac
import json
import logging
import time
from hashlib import sha256
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

_DEFAULT_HEADERS = {"User-Agent": "bot-core/1.0 (+https://github.com/)"}


def _determine_public_base(environment: Environment) -> str:
    """Zwraca bazowy adres REST dla danych publicznych kontraktów USD-M."""

    if environment is Environment.TESTNET or environment is Environment.PAPER:
        return "https://testnet.binancefuture.com"
    return "https://fapi.binance.com"


def _determine_trading_base(environment: Environment) -> str:
    """Zwraca bazowy adres REST dla wywołań podpisanych kontraktów USD-M."""

    if environment is Environment.TESTNET or environment is Environment.PAPER:
        return "https://testnet.binancefuture.com"
    return "https://fapi.binance.com"


def _stringify_params(params: Mapping[str, object]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for key, value in params.items():
        if isinstance(value, bool):
            normalized.append((key, "true" if value else "false"))
        elif value is None:
            continue
        else:
            normalized.append((key, str(value)))
    return normalized


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class BinanceFuturesAdapter(ExchangeAdapter):
    """Adapter REST dla rynku terminowego Binance (USD-M)."""

    name: str = "binance_futures"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
    ) -> None:
        super().__init__(credentials)
        self._environment = environment or credentials.environment
        self._public_base = _determine_public_base(self._environment)
        self._trading_base = _determine_trading_base(self._environment)
        self._permission_set = frozenset(perm.lower() for perm in self._credentials.permissions)
        self._ip_allowlist: tuple[str, ...] = ()

    def configure_network(self, *, ip_allowlist: Optional[Sequence[str]] = None) -> None:
        if ip_allowlist is None:
            self._ip_allowlist = ()
        else:
            self._ip_allowlist = tuple(ip_allowlist)
        _LOGGER.info("Ustawiono allowlistę IP dla Binance Futures: %s", self._ip_allowlist)

    def _public_request(
        self,
        path: str,
        params: Optional[Mapping[str, object]] = None,
        *,
        method: str = "GET",
    ) -> dict[str, object] | list[object]:
        query = f"?{urlencode(_stringify_params(params or {}))}" if params else ""
        url = f"{self._public_base}{path}{query}"
        data = None
        headers = dict(_DEFAULT_HEADERS)
        if method in {"POST", "PUT"} and params:
            data = urlencode(_stringify_params(params)).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = Request(url, headers=headers, data=data, method=method)
        return self._execute_request(request)

    def _signed_request(
        self,
        path: str,
        *,
        method: str = "GET",
        params: Optional[Mapping[str, object]] = None,
    ) -> dict[str, object] | list[object]:
        if not self._credentials.secret:
            raise RuntimeError("Do podpisanych endpointów wymagany jest secret klucza API Binance.")

        timestamp_ms = int(time.time() * 1000)
        base_params = dict(params or {})
        base_params.setdefault("timestamp", timestamp_ms)
        payload_items = _stringify_params(base_params)
        query_string = urlencode(payload_items)
        signature = hmac.new(
            self._credentials.secret.encode("utf-8"),
            query_string.encode("utf-8"),
            sha256,
        ).hexdigest()
        signed_query = f"{query_string}&signature={signature}"

        url = f"{self._trading_base}{path}"
        headers = dict(_DEFAULT_HEADERS)
        headers["X-MBX-APIKEY"] = self._credentials.key_id

        if method in {"POST", "PUT"}:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            data = signed_query.encode("utf-8")
            request = Request(url, headers=headers, data=data, method=method)
        else:
            separator = "?" if "?" not in url else "&"
            request = Request(f"{url}{separator}{signed_query}", headers=headers, method=method)
        return self._execute_request(request)

    @staticmethod
    def _execute_request(request: Request) -> dict[str, object] | list[object]:
        try:
            with urlopen(request, timeout=15) as response:  # nosec: B310 - zaufany endpoint
                payload = response.read()
        except HTTPError as exc:  # pragma: no cover - zależne od API
            _LOGGER.error("Błąd HTTP podczas komunikacji z Binance Futures: %s", exc)
            raise RuntimeError(f"Binance Futures API zwróciło błąd HTTP: {exc}") from exc
        except URLError as exc:  # pragma: no cover - zależne od sieci
            _LOGGER.error("Błąd sieci podczas komunikacji z Binance Futures: %s", exc)
            raise RuntimeError("Nie udało się połączyć z API Binance Futures") from exc

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:  # pragma: no cover - błąd po stronie API
            _LOGGER.error("Niepoprawna odpowiedź JSON z Binance Futures: %s", exc)
            raise RuntimeError("Niepoprawna odpowiedź JSON od API Binance Futures") from exc
        return data

    def fetch_account_snapshot(self) -> AccountSnapshot:
        if not ({"read", "trade"} & self._permission_set):
            raise PermissionError("Poświadczenia nie pozwalają na odczyt danych konta Binance Futures.")

        payload = self._signed_request("/fapi/v2/account")
        if not isinstance(payload, Mapping):
            raise RuntimeError("Niepoprawna odpowiedź konta z Binance Futures")

        assets = payload.get("assets", [])
        balances: dict[str, float] = {}
        available_margin = _to_float(payload.get("totalAvailableBalance"), 0.0)
        maintenance_margin = _to_float(payload.get("totalMaintMargin"), 0.0)

        if isinstance(assets, list):
            for entry in assets:
                if not isinstance(entry, Mapping):
                    continue
                asset = entry.get("asset")
                wallet_balance = _to_float(entry.get("walletBalance"), 0.0)
                if isinstance(asset, str):
                    balances[asset] = wallet_balance

        total_equity = _to_float(payload.get("totalMarginBalance"), 0.0)
        if total_equity == 0.0:
            wallet = _to_float(payload.get("totalWalletBalance"), 0.0)
            unrealized = _to_float(payload.get("totalUnrealizedProfit"), 0.0)
            total_equity = wallet + unrealized

        return AccountSnapshot(
            balances=balances,
            total_equity=total_equity,
            available_margin=available_margin,
            maintenance_margin=maintenance_margin,
        )

    def fetch_symbols(self) -> Iterable[str]:
        payload = self._public_request("/fapi/v1/exchangeInfo")
        if not isinstance(payload, Mapping) or "symbols" not in payload:
            raise RuntimeError("Niepoprawna odpowiedź exchangeInfo z Binance Futures")

        symbols_raw = payload.get("symbols")
        if not isinstance(symbols_raw, list):
            raise RuntimeError("Pole 'symbols' w odpowiedzi Binance Futures ma niepoprawny format")

        active: list[str] = []
        for entry in symbols_raw:
            if not isinstance(entry, Mapping):
                continue
            status = entry.get("status")
            symbol = entry.get("symbol")
            if status != "TRADING" or not isinstance(symbol, str):
                continue
            active.append(symbol)
        return active

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        params: dict[str, object] = {"symbol": symbol, "interval": interval}
        if start is not None:
            params["startTime"] = int(start)
        if end is not None:
            params["endTime"] = int(end)
        if limit is not None:
            params["limit"] = int(limit)

        payload = self._public_request("/fapi/v1/klines", params=params)
        if not isinstance(payload, list):
            raise RuntimeError("Odpowiedź klines z Binance Futures ma nieoczekiwany format")

        candles: list[Sequence[float]] = []
        for entry in payload:
            if not isinstance(entry, list) or len(entry) < 6:
                continue
            open_time = float(entry[0])
            open_price = float(entry[1])
            high = float(entry[2])
            low = float(entry[3])
            close = float(entry[4])
            volume = float(entry[5])
            candles.append([open_time, open_price, high, low, close, volume])
        return candles

    def place_order(self, request: OrderRequest) -> OrderResult:
        if "trade" not in self._permission_set:
            raise PermissionError("Aktualne poświadczenia nie mają uprawnień tradingowych.")

        params: dict[str, object] = {
            "symbol": request.symbol,
            "side": request.side.upper(),
            "type": request.order_type.upper(),
            "quantity": request.quantity,
        }
        if request.price is not None:
            params["price"] = request.price
        if request.time_in_force is not None:
            params["timeInForce"] = request.time_in_force
        if request.client_order_id is not None:
            params["newClientOrderId"] = request.client_order_id

        payload = self._signed_request("/fapi/v1/order", method="POST", params=params)
        if not isinstance(payload, Mapping):
            raise RuntimeError("Odpowiedź z endpointu futures order ma niepoprawny format")

        payload_dict = dict(payload)
        order_id = str(payload_dict.get("orderId"))
        status = str(payload_dict.get("status", "UNKNOWN"))
        filled_qty = _to_float(payload_dict.get("executedQty", 0.0))
        avg_price_field = payload_dict.get("avgPrice", payload_dict.get("price"))
        avg_price = _to_float(avg_price_field) if avg_price_field not in (None, "0", 0, 0.0) else None

        return OrderResult(
            order_id=order_id,
            status=status,
            filled_quantity=filled_qty,
            avg_price=avg_price,
            raw_response=payload_dict,
        )

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:
        if "trade" not in self._permission_set:
            raise PermissionError("Aktualne poświadczenia nie mają uprawnień tradingowych.")
        if not symbol:
            raise ValueError("Anulowanie na Binance Futures wymaga podania symbolu.")

        params: dict[str, object] = {"orderId": order_id, "symbol": symbol}
        response = self._signed_request("/fapi/v1/order", method="DELETE", params=params)
        if isinstance(response, Mapping):
            response_map = dict(response)
            status = response_map.get("status")
            if status in {"CANCELED", "PENDING_CANCEL", "NEW"}:
                return
            raise RuntimeError(f"Nieoczekiwana odpowiedź anulowania z Binance Futures: {response_map}")
        raise RuntimeError("Niepoprawna odpowiedź anulowania z Binance Futures")

    def stream_public_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        raise NotImplementedError("Stream publiczny futures zostanie dodany w kolejnej iteracji.")

    def stream_private_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        raise NotImplementedError("Stream prywatny futures zostanie dodany w kolejnej iteracji.")


__all__ = ["BinanceFuturesAdapter"]
