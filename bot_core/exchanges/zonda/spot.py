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

# --- Konfiguracja endpointów -------------------------------------------------

_BASE_URLS: Mapping[Environment, str] = {
    Environment.LIVE: "https://api.zonda.exchange/rest",
    Environment.PAPER: "https://api.zonda.exchange/rest",          # brak osobnego paper – użyj produkcyjnego REST
    Environment.TESTNET: "https://api-sandbox.zonda.exchange/rest",
}

_DEFAULT_HEADERS = {"User-Agent": "bot-core/1.0 (+https://github.com/)"}


def _extract_pair(symbol: str, entry: Mapping[str, object]) -> tuple[str, str] | None:
    """Zwraca bazę i kwotowanie dla pary handlowej."""

    if "-" in symbol:
        base, quote = symbol.split("-", 1)
        if base and quote:
            return base.upper(), quote.upper()

    market_info = entry.get("market")
    if isinstance(market_info, Mapping):
        base_value = market_info.get("first") or market_info.get("base")
        quote_value = market_info.get("second") or market_info.get("quote")
        if (
            isinstance(base_value, str)
            and isinstance(quote_value, str)
            and base_value
            and quote_value
        ):
            return base_value.upper(), quote_value.upper()
    return None


def _direct_rate(
    base: str,
    quote: str,
    prices: Mapping[tuple[str, str], float],
) -> float | None:
    """Zwraca bezpośredni kurs dla pary base/quote lub jego odwrotność."""

    base = base.upper()
    quote = quote.upper()
    if base == quote:
        return 1.0

    direct = prices.get((base, quote))
    if direct is not None and direct > 0:
        return direct

    reverse = prices.get((quote, base))
    if reverse is not None and reverse > 0:
        return 1.0 / reverse
    return None


def _convert_with_intermediaries(
    asset: str,
    target: str,
    prices: Mapping[tuple[str, str], float],
    intermediaries: Sequence[str],
) -> float | None:
    """Próbuje przeliczyć aktywo na walutę docelową przy użyciu kursów pośrednich."""

    asset = asset.upper()
    target = target.upper()
    rate = _direct_rate(asset, target, prices)
    if rate is not None:
        return rate

    for intermediary in intermediaries:
        intermediate = intermediary.upper()
        if intermediate in {asset, target}:
            continue
        first_leg = _direct_rate(asset, intermediate, prices)
        if first_leg is None:
            continue
        second_leg = _direct_rate(intermediate, target, prices)
        if second_leg is None:
            continue
        return first_leg * second_leg
    return None


# --- Funkcje pomocnicze ------------------------------------------------------

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
    if key not in mapping and key.endswith("m") and key[:-1].isdigit():
        mapping[key] = int(key[:-1]) * 60
    try:
        return mapping[key]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Nieobsługiwany interwał Zonda: {interval}") from exc


def _json_body(payload: Mapping[str, object] | None) -> str:
    if not payload:
        return ""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class _OrderPayload:
    order_id: str
    status: str
    filled_quantity: float
    avg_price: Optional[float]
    raw: Mapping[str, object]


# --- Adapter -----------------------------------------------------------------

class ZondaSpotAdapter(ExchangeAdapter):
    """Adapter REST obsługujący podstawowe operacje tradingowe Zonda."""

    __slots__ = (
        "_environment",
        "_base_url",
        "_ip_allowlist",
        "_permission_set",
        "_settings",
        "_valuation_asset",
        "_secondary_valuation_assets",
    )
    name: str = "zonda_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
        settings: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(credentials)
        self._environment = environment or credentials.environment
        try:
            self._base_url = _BASE_URLS[self._environment]
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Nieobsługiwane środowisko Zonda: {self._environment}") from exc
        self._ip_allowlist: tuple[str, ...] = ()
        self._permission_set = frozenset(perm.lower() for perm in credentials.permissions)
        self._settings = dict(settings or {})
        self._valuation_asset = self._extract_valuation_asset()
        self._secondary_valuation_assets = self._extract_secondary_assets()

    # --- Konfiguracja wyceny -------------------------------------------------

    def _extract_valuation_asset(self) -> str:
        raw = self._settings.get("valuation_asset", "PLN")
        if isinstance(raw, str):
            asset = raw.strip().upper()
            return asset or "PLN"
        return "PLN"

    def _extract_secondary_assets(self) -> tuple[str, ...]:
        raw = self._settings.get("secondary_valuation_assets")
        defaults = ("PLN", "USDT", "USD", "EUR")

        def _append(container: list[str], value: object) -> None:
            text = str(value).strip().upper()
            if not text or text == self._valuation_asset:
                return
            if text not in container:
                container.append(text)

        assets: list[str] = []
        if raw is None:
            for entry in defaults:
                _append(assets, entry)
        elif isinstance(raw, str):
            for token in raw.split(","):
                _append(assets, token)
        elif isinstance(raw, Sequence):
            for entry in raw:
                _append(assets, entry)
        else:  # pragma: no cover - zabezpieczenie nietypowych konfiguracji
            for entry in defaults:
                _append(assets, entry)
        return tuple(assets)

    # --- HTTP ----------------------------------------------------------------

    def _build_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self._base_url}{path}"

    def _execute_request(self, request: Request) -> dict[str, object] | list[object]:
        try:
            with urlopen(request, timeout=15) as response:  # nosec: B310 – kontrolowany endpoint
                payload = response.read()
        except HTTPError as exc:  # pragma: no cover
            _LOGGER.error("Zonda zwróciła błąd HTTP: %s", exc)
            raise RuntimeError(f"Zonda API zwróciła błąd HTTP: {exc}") from exc
        except URLError as exc:  # pragma: no cover
            _LOGGER.error("Błąd sieci podczas komunikacji z Zonda: %s", exc)
            raise RuntimeError("Nie udało się połączyć z API Zonda") from exc

        try:
            parsed: object = json.loads(payload)
        except json.JSONDecodeError as exc:  # pragma: no cover
            _LOGGER.error("Niepoprawna odpowiedź JSON od Zonda: %s", exc)
            raise RuntimeError("Niepoprawna odpowiedź API Zonda") from exc

        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return parsed

        _LOGGER.error("Nieoczekiwany format odpowiedzi API Zonda: %r", parsed)
        raise RuntimeError("Nieoczekiwany format odpowiedzi API Zonda")

    def _public_request(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
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
        params: Mapping[str, object] | None = None,
        data: Mapping[str, object] | None = None,
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

        query = f"?{urlencode(params)}" if params else ""
        data_bytes: bytes | None = None
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

    def _fetch_price_map(self) -> Mapping[tuple[str, str], float]:
        response = self._public_request("/trading/ticker")
        prices: dict[tuple[str, str], float] = {}
        if isinstance(response, Mapping):
            items = response.get("items")
            if isinstance(items, Mapping):
                for symbol, raw in items.items():
                    if not isinstance(symbol, str) or not isinstance(raw, Mapping):
                        continue
                    pair = _extract_pair(symbol, raw)
                    if pair is None:
                        continue
                    rate = _to_float(
                        raw.get("rate")
                        or raw.get("last")
                        or raw.get("average")
                        or raw.get("averagePrice")
                    )
                    if rate <= 0 and isinstance(raw.get("ticker"), Mapping):
                        rate = _to_float(raw["ticker"].get("rate"))
                    if rate <= 0:
                        continue
                    prices[pair] = rate
        return prices

    # --- ExchangeAdapter API --------------------------------------------------

    def configure_network(self, *, ip_allowlist: Optional[Sequence[str]] = None) -> None:  # type: ignore[override]
        self._ip_allowlist = tuple(ip_allowlist or ())
        if self._ip_allowlist:
            _LOGGER.info("Zonda allowlist IP ustawiony na: %s", self._ip_allowlist)

    def fetch_symbols(self) -> Iterable[str]:  # type: ignore[override]
        response = self._public_request("/trading/ticker")
        if not isinstance(response, Mapping):
            return []
        items = response.get("items")
        if isinstance(items, Mapping):
            return sorted(str(symbol) for symbol in items.keys())
        return []

    def fetch_ohlcv(  # type: ignore[override]
        self,
        symbol: str,
        interval: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        resolution = _normalize_interval(interval)
        params: dict[str, object] = {}
        # API Zondy przyjmuje znacznik czasu w sekundach
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
            return []

        candles: list[Sequence[float]] = []
        for entry in items:
            if not isinstance(entry, Mapping):
                continue
            # Zwracamy ms (spójnie z resztą systemu)
            timestamp = int(_to_float(entry.get("time"))) * 1000
            open_price = _to_float(entry.get("open") or entry.get("o"))
            high_price = _to_float(entry.get("high") or entry.get("h"))
            low_price = _to_float(entry.get("low") or entry.get("l"))
            close_price = _to_float(entry.get("close") or entry.get("c"))
            volume = _to_float(entry.get("volume") or entry.get("v"))
            candles.append([float(timestamp), open_price, high_price, low_price, close_price, volume])
        return candles

    def fetch_account_snapshot(self) -> AccountSnapshot:  # type: ignore[override]
        if "read" not in self._permission_set and "trade" not in self._permission_set:
            raise PermissionError("Poświadczenia Zonda nie mają uprawnień do odczytu sald.")

        response = self._signed_request("POST", "/trading/balance")
        if not isinstance(response, Mapping):
            raise RuntimeError("Niepoprawna odpowiedź balansu z Zonda")

        balances_section = response.get("balances", [])
        balances: dict[str, float] = {}
        free_balances: dict[str, float] = {}
        if isinstance(balances_section, list):
            for entry in balances_section:
                if not isinstance(entry, Mapping):
                    continue
                currency = entry.get("currency") or entry.get("code")
                available = _to_float(entry.get("available"))
                locked = _to_float(entry.get("locked") or entry.get("reserved"))
                if not isinstance(currency, str):
                    continue
                asset = currency.strip().upper()
                if not asset:
                    continue
                total_balance = available + locked
                balances[asset] = total_balance
                free_balances[asset] = available

        prices = self._fetch_price_map()
        valuation_currency = self._valuation_asset
        intermediaries = self._secondary_valuation_assets
        total_equity = 0.0
        available_margin = 0.0
        for asset, total_balance in balances.items():
            conversion = _convert_with_intermediaries(asset, valuation_currency, prices, intermediaries)
            if conversion is None:
                _LOGGER.debug(
                    "Pomijam aktywo %s – brak kursu do %s w danych ticker.",
                    asset,
                    valuation_currency,
                )
                continue
            total_equity += total_balance * conversion
            available_margin += free_balances.get(asset, 0.0) * conversion

        return AccountSnapshot(
            balances=balances,
            total_equity=total_equity,
            available_margin=available_margin,
            maintenance_margin=0.0,
        )

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

    def place_order(self, request: OrderRequest) -> OrderResult:  # type: ignore[override]
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

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:  # type: ignore[override]
        del symbol  # Zonda nie wymaga symbolu do anulowania
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
