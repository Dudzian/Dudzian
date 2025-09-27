"""Adapter REST dla rynku spot Binance do obsługi danych publicznych i prywatnych."""
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


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _determine_public_base(environment: Environment) -> str:
    """Zwraca właściwy punkt końcowy REST dla danych publicznych."""

    # Binance nie udostępnia pełnych danych historycznych na testnecie,
    # dlatego zarówno środowisko PAPER, jak i TESTNET wykorzystują publiczny endpoint produkcyjny.
    return "https://api.binance.com"


def _determine_trading_base(environment: Environment) -> str:
    """Zwraca punkt końcowy dla operacji wymagających podpisu."""

    if environment is Environment.TESTNET or environment is Environment.PAPER:
        # Oficjalny testnet Binance udostępnia podpisane endpointy pod domeną testnet.binance.vision.
        return "https://testnet.binance.vision"
    return "https://api.binance.com"


def _stringify_params(params: Mapping[str, object]) -> list[tuple[str, str]]:
    """Konwertuje wartości parametrów na tekst wymagany przez API Binance."""

    normalized: list[tuple[str, str]] = []
    for key, value in params.items():
        if isinstance(value, bool):
            normalized.append((key, "true" if value else "false"))
        elif value is None:
            continue
        else:
            normalized.append((key, str(value)))
    return normalized


class BinanceSpotAdapter(ExchangeAdapter):
    """Adapter dla rynku spot Binance z obsługą danych publicznych i podpisanych."""

    __slots__ = (
        "_environment",
        "_public_base",
        "_trading_base",
        "_ip_allowlist",
        "_permission_set",
    )

    name: str = "binance_spot"

    def __init__(self, credentials: ExchangeCredentials, *, environment: Environment | None = None) -> None:
        super().__init__(credentials)
        self._environment = environment or credentials.environment
        self._public_base = _determine_public_base(self._environment)
        self._trading_base = _determine_trading_base(self._environment)
        self._ip_allowlist: tuple[str, ...] = ()
        self._permission_set = frozenset(perm.lower() for perm in self._credentials.permissions)

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

        data: Optional[bytes]
        if method in {"POST", "PUT"}:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            data = signed_query.encode("utf-8")
            request = Request(url, headers=headers, data=data, method=method)
        else:
            separator = "?" if "?" not in url else "&"
            request = Request(f"{url}{separator}{signed_query}", headers=headers, method=method)
            data = None

        return self._execute_request(request)

    @staticmethod
    def _execute_request(request: Request) -> dict[str, object] | list[object]:
        try:
            with urlopen(request, timeout=15) as response:  # nosec: B310 - endpoint zaufany
                payload = response.read()
        except HTTPError as exc:  # pragma: no cover - zależne od API i środowiska sieciowego
            _LOGGER.error("Błąd HTTP podczas komunikacji z Binance: %s", exc)
            raise RuntimeError(f"Binance API zwróciło błąd HTTP: {exc}") from exc
        except URLError as exc:  # pragma: no cover - zależne od sieci
            _LOGGER.error("Błąd sieci podczas komunikacji z Binance: %s", exc)
            raise RuntimeError("Nie udało się połączyć z API Binance") from exc

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:  # pragma: no cover - niepoprawne JSON to błąd API
            _LOGGER.error("Niepoprawna odpowiedź JSON od Binance: %s", exc)
            raise RuntimeError("Niepoprawna odpowiedź JSON od API Binance") from exc
        return data

    def configure_network(self, *, ip_allowlist: Optional[Sequence[str]] = None) -> None:
        """Zachowuje konfigurację allowlisty, aby risk engine mógł ją audytować."""

        if ip_allowlist is None:
            self._ip_allowlist = ()
        else:
            self._ip_allowlist = tuple(ip_allowlist)
        _LOGGER.info("Ustawiono allowlistę IP dla Binance: %s", self._ip_allowlist)

    def fetch_account_snapshot(self) -> AccountSnapshot:
        """Pobiera podstawowe dane o stanie rachunku do oceny limitów ryzyka."""

        if not ({"read", "trade"} & self._permission_set):
            raise PermissionError("Poświadczenia nie pozwalają na odczyt danych konta Binance.")

        payload = self._signed_request("/api/v3/account")
        if not isinstance(payload, dict):
            raise RuntimeError("Niepoprawna odpowiedź konta z Binance")

        balances_section = payload.get("balances", [])
        balances: dict[str, float] = {}
        available_margin = 0.0
        if isinstance(balances_section, list):
            for entry in balances_section:
                if not isinstance(entry, Mapping):
                    continue
                asset = entry.get("asset")
                free = _to_float(entry.get("free", 0.0))
                locked = _to_float(entry.get("locked", 0.0))
                if not isinstance(asset, str):
                    continue
                balances[asset] = free + locked
                available_margin += free

        total_equity = sum(balances.values())
        maintenance_margin = _to_float(
            payload.get("maintMarginBalance", payload.get("totalMarginBalance", 0.0))
        )

        return AccountSnapshot(
            balances=balances,
            total_equity=total_equity,
            available_margin=available_margin,
            maintenance_margin=maintenance_margin,
        )

    def fetch_symbols(self) -> Iterable[str]:
        """Pobiera listę aktywnych symboli spot z Binance."""

        payload = self._public_request("/api/v3/exchangeInfo")
        if not isinstance(payload, dict) or "symbols" not in payload:
            raise RuntimeError("Niepoprawna odpowiedź exchangeInfo z Binance")

        symbols_section = payload.get("symbols")
        if not isinstance(symbols_section, list):
            raise RuntimeError("Pole 'symbols' w odpowiedzi Binance ma niepoprawny format")

        active_symbols: list[str] = []
        for entry in symbols_section:
            if not isinstance(entry, dict):
                continue
            status = entry.get("status")
            symbol = entry.get("symbol")
            if status != "TRADING" or not isinstance(symbol, str):
                continue
            active_symbols.append(symbol)
        return active_symbols

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        """Pobiera świece OHLCV w formacie zgodnym z modułem danych."""

        params: dict[str, object] = {"symbol": symbol, "interval": interval}
        if start is not None:
            params["startTime"] = int(start)
        if end is not None:
            params["endTime"] = int(end)
        if limit is not None:
            params["limit"] = int(limit)

        payload = self._public_request("/api/v3/klines", params=params)
        if not isinstance(payload, list):
            raise RuntimeError("Odpowiedź klines z Binance ma nieoczekiwany format")

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
        """Składa podpisane zlecenie typu limit/market na rynku spot."""

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

        payload = self._signed_request("/api/v3/order", method="POST", params=params)
        if not isinstance(payload, Mapping):
            raise RuntimeError("Odpowiedź z endpointu order ma niepoprawny format")

        payload_dict = dict(payload)

        order_id = str(payload_dict.get("orderId"))
        status = str(payload_dict.get("status", "UNKNOWN"))
        filled_qty = _to_float(payload_dict.get("executedQty", 0.0))
        raw_price = payload_dict.get("price")
        avg_price = _to_float(raw_price) if raw_price not in (None, "0", 0, 0.0) else None

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
        params: dict[str, object] = {"orderId": order_id}
        if symbol:
            params["symbol"] = symbol
        response = self._signed_request("/api/v3/order", method="DELETE", params=params)
        if isinstance(response, Mapping):
            response_map = dict(response)
            if response_map.get("status") in {"CANCELED", "PENDING_CANCEL"}:
                return
        # Jeśli API zwróciło błąd, `urlopen` podniesie wyjątek HTTPError; jeśli odpowiedź jest dziwna,
        # sygnalizujemy to wyjątkiem, aby egzekucja mogła przejść w tryb bezpieczny.
            raise RuntimeError(f"Nieoczekiwana odpowiedź anulowania z Binance: {response_map}")
        raise RuntimeError("Niepoprawna odpowiedź anulowania z Binance")

    def stream_public_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        raise NotImplementedError("Stream danych publicznych zostanie dodany w przyszłym etapie.")

    def stream_private_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        raise NotImplementedError("Stream danych prywatnych zostanie dodany w przyszłym etapie.")


__all__ = ["BinanceSpotAdapter"]
