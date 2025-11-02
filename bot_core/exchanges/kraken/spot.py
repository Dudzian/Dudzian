"""Adapter REST dla Kraken Spot zgodny z interfejsem `ExchangeAdapter`."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import random
import time
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Callable, Mapping, MutableMapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.error_mapping import raise_for_kraken_error
from bot_core.exchanges.health import Watchdog
from bot_core.exchanges.rate_limiter import (
    RateLimitRule,
    get_global_rate_limiter_registry,
    normalize_rate_limit_rules,
)
from bot_core.exchanges.streaming import LocalLongPollStream
from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry
from bot_core.exchanges.http_client import urlopen


_LOGGER = logging.getLogger(__name__)

_BASE_URLS: Mapping[Environment, str] = {
    Environment.LIVE: "https://api.kraken.com",
    Environment.PAPER: "https://api.kraken.com",
    Environment.TESTNET: "https://api.kraken.com",
}

_DEFAULT_HEADERS = {
    "User-Agent": "bot-core/kraken-spot",
    "Accept": "application/json",
}

_RETRYABLE_STATUS = {429, 500, 502, 503, 504, 520, 521, 522, 524}
_MAX_RETRIES = 3
_BASE_BACKOFF = 0.5
_BACKOFF_CAP = 4.0

_RATE_LIMIT_DEFAULTS: tuple[RateLimitRule, ...] = (
    RateLimitRule(rate=20, per=1.0),
    RateLimitRule(rate=180, per=60.0),
)


_INTERVAL_MAPPING: Mapping[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class _RequestContext:
    path: str
    params: Mapping[str, Any]


@dataclass(slots=True)
class KrakenOpenOrder:
    """Znormalizowana reprezentacja otwartego zlecenia na Kraken Spot."""

    order_id: str
    symbol: str
    side: str
    order_type: str
    price: float | None
    volume: float
    volume_executed: float
    timestamp: float
    flags: tuple[str, ...]


@dataclass(slots=True)
class KrakenTrade:
    """Znormalizowana reprezentacja transakcji z historii konta."""

    trade_id: str
    order_id: str | None
    symbol: str
    side: str
    order_type: str
    price: float
    volume: float
    cost: float
    fee: float
    timestamp: float


@dataclass(slots=True)
class KrakenTicker:
    """Znormalizowany ticker 24h z publicznego API Kraken Spot."""

    symbol: str
    best_ask: float
    best_bid: float
    last_price: float
    volume_24h: float
    vwap_24h: float
    high_24h: float
    low_24h: float
    open_price: float
    timestamp: float


@dataclass(slots=True)
class KrakenOrderBookEntry:
    """Pojedynczy wiersz orderbooka Kraken Spot."""

    price: float
    volume: float
    timestamp: float


@dataclass(slots=True)
class KrakenOrderBook:
    """Znormalizowany orderbook (bids/asks) z publicznego API Kraken Spot."""

    symbol: str
    bids: tuple[KrakenOrderBookEntry, ...]
    asks: tuple[KrakenOrderBookEntry, ...]
    depth: int
    timestamp: float


class KrakenSpotAdapter(ExchangeAdapter):
    """Implementacja publicznych i prywatnych endpointów Kraken Spot."""

    name = "kraken_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment,
        settings: Mapping[str, object] | None = None,
        metrics_registry: MetricsRegistry | None = None,
        watchdog: Watchdog | None = None,
    ) -> None:
        super().__init__(credentials)
        self._environment = environment
        try:
            self._base_url = _BASE_URLS[environment]
        except KeyError as exc:  # pragma: no cover - brak konfiguracji
            raise ValueError(f"Nieobsługiwane środowisko Kraken: {environment}") from exc
        self._http_timeout = 15
        self._permission_set = set(credentials.permissions)
        self._ip_allowlist: Sequence[str] | None = None
        self._last_nonce = 0
        self._settings = dict(settings or {})
        asset = str(self._settings.get("valuation_asset", "ZUSD") or "ZUSD").strip().upper()
        if asset and not asset.startswith("Z") and len(asset) <= 4:
            asset = f"Z{asset}"
        self._valuation_asset = asset or "ZUSD"
        self._metrics = metrics_registry or get_global_metrics_registry()
        self._metric_base_labels = {
            "exchange": self.name,
            "environment": self._environment.value,
        }
        self._metric_http_latency = self._metrics.histogram(
            "kraken_spot_http_latency_seconds",
            "Czas odpowiedzi zapytań HTTP do API Kraken Spot.",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
        )
        self._metric_retries = self._metrics.counter(
            "kraken_spot_retries_total",
            "Liczba ponowień zapytań do API Kraken Spot (powód=throttled/server_error/network).",
        )
        self._metric_signed_requests = self._metrics.counter(
            "kraken_spot_signed_requests_total",
            "Liczba podpisanych zapytań HTTP wysłanych do API Kraken Spot.",
        )
        self._metric_api_errors = self._metrics.counter(
            "kraken_spot_api_errors_total",
            "Błędy API Kraken Spot (powód=auth/throttled/api_error/http_error/network/json).",
        )
        self._metric_open_orders = self._metrics.gauge(
            "kraken_spot_open_orders",
            "Liczba otwartych zleceń raportowanych przez Kraken Spot.",
        )
        self._metric_trades = self._metrics.counter(
            "kraken_spot_trades_fetched_total",
            "Łączna liczba transakcji pobranych z historii Kraken Spot.",
        )
        self._metric_ticker_last_price = self._metrics.gauge(
            "kraken_spot_ticker_last_price",
            "Ostatnia cena transakcyjna raportowana przez Kraken Spot.",
        )
        self._metric_ticker_spread = self._metrics.gauge(
            "kraken_spot_ticker_spread",
            "Spread między najlepszą ofertą kupna i sprzedaży na Kraken Spot.",
        )
        self._metric_orderbook_levels = self._metrics.gauge(
            "kraken_spot_orderbook_levels",
            "Łączna liczba poziomów orderbooka (bids+asks) zwracanych przez Kraken Spot.",
        )
        self._watchdog = watchdog or Watchdog()
        self._rate_limiter = get_global_rate_limiter_registry().configure(
            f"{self.name}:{self._environment.value}",
            normalize_rate_limit_rules(
                self._settings.get("rate_limit_rules"),
                _RATE_LIMIT_DEFAULTS,
            ),
            metric_labels={"exchange": self.name, "environment": self._environment.value},
        )

    # ------------------------------------------------------------------
    # Konfiguracja streamingu long-pollowego
    # ------------------------------------------------------------------
    def _stream_settings(self) -> Mapping[str, object]:
        raw = self._settings.get("stream")
        if isinstance(raw, Mapping):
            return raw
        return {}

    def _build_stream(self, scope: str, channels: Sequence[str]) -> LocalLongPollStream:
        stream_settings = dict(self._stream_settings())
        base_url = str(
            stream_settings.get("base_url", self._settings.get("stream_base_url", "http://127.0.0.1:8765"))
        )
        default_path = f"/stream/{self.name}/{scope}"
        path = str(
            stream_settings.get(
                f"{scope}_path",
                self._settings.get(f"stream_{scope}_path", default_path),
            )
            or default_path
        )
        poll_interval = float(
            stream_settings.get(
                "poll_interval",
                self._settings.get("stream_poll_interval", 0.5),
            )
        )
        timeout = float(stream_settings.get("timeout", self._settings.get("stream_timeout", 10.0)))
        max_retries = int(stream_settings.get("max_retries", self._settings.get("stream_max_retries", 3)))
        backoff_base = float(
            stream_settings.get("backoff_base", self._settings.get("stream_backoff_base", 0.25))
        )
        backoff_cap = float(
            stream_settings.get("backoff_cap", self._settings.get("stream_backoff_cap", 2.0))
        )
        jitter = stream_settings.get("jitter", self._settings.get("stream_jitter", (0.05, 0.30)))
        channel_param = stream_settings.get(f"{scope}_channel_param")
        if channel_param is None:
            channel_param = stream_settings.get(
                "channel_param", self._settings.get("stream_channel_param", "channels")
            )
        cursor_param = stream_settings.get(f"{scope}_cursor_param")
        if cursor_param is None:
            cursor_param = stream_settings.get(
                "cursor_param", self._settings.get("stream_cursor_param", "cursor")
            )
        initial_cursor = stream_settings.get(f"{scope}_initial_cursor")
        if initial_cursor is None:
            initial_cursor = stream_settings.get("initial_cursor")
        channel_serializer = None
        serializer_candidate = stream_settings.get(f"{scope}_channel_serializer")
        if not callable(serializer_candidate):
            serializer_candidate = stream_settings.get("channel_serializer")
        if callable(serializer_candidate):
            channel_serializer = serializer_candidate
        else:
            separator = stream_settings.get(f"{scope}_channel_separator")
            if separator is None:
                separator = stream_settings.get(
                    "channel_separator", self._settings.get("stream_channel_separator", ",")
                )
            if isinstance(separator, str):
                channel_serializer = lambda values, sep=separator: sep.join(values)  # noqa: E731
        headers_raw = stream_settings.get("headers")
        header_map = dict(headers_raw) if isinstance(headers_raw, Mapping) else None
        params: dict[str, object] = {}
        base_params = stream_settings.get("params")
        if isinstance(base_params, Mapping):
            params.update(base_params)
        scope_params = stream_settings.get(f"{scope}_params")
        if isinstance(scope_params, Mapping):
            params.update(scope_params)
        token_key = f"{scope}_token"
        if isinstance(stream_settings.get(token_key), str):
            params.setdefault("token", stream_settings[token_key])
        elif isinstance(stream_settings.get("auth_token"), str):
            params.setdefault("token", stream_settings["auth_token"])
        http_method = stream_settings.get(f"{scope}_method")
        if http_method is None:
            http_method = stream_settings.get("method", "GET")
        params_in_body = stream_settings.get(f"{scope}_params_in_body")
        if params_in_body is None:
            params_in_body = stream_settings.get("params_in_body", False)
        channels_in_body = stream_settings.get(f"{scope}_channels_in_body")
        if channels_in_body is None:
            channels_in_body = stream_settings.get("channels_in_body", False)
        cursor_in_body = stream_settings.get(f"{scope}_cursor_in_body")
        if cursor_in_body is None:
            cursor_in_body = stream_settings.get("cursor_in_body", False)
        body_params: dict[str, object] = {}
        base_body = stream_settings.get("body_params")
        if isinstance(base_body, Mapping):
            body_params.update(base_body)
        scope_body = stream_settings.get(f"{scope}_body_params")
        if isinstance(scope_body, Mapping):
            body_params.update(scope_body)
        body_encoder = stream_settings.get(f"{scope}_body_encoder")
        if body_encoder is None:
            body_encoder = stream_settings.get("body_encoder")

        buffer_size_raw = stream_settings.get(f"{scope}_buffer_size")
        if buffer_size_raw is None:
            buffer_size_raw = stream_settings.get("buffer_size", 64)
        try:
            buffer_size = int(buffer_size_raw)
        except (TypeError, ValueError):
            buffer_size = 64
        if buffer_size < 1:
            buffer_size = 1

        return LocalLongPollStream(
            base_url=base_url,
            path=path,
            channels=channels,
            adapter=self.name,
            scope=scope,
            environment=self._environment.value,
            params=params,
            headers=header_map,
            poll_interval=poll_interval,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_cap=backoff_cap,
            jitter=jitter if isinstance(jitter, Sequence) else (0.05, 0.30),
            channel_param=str(channel_param).strip() if channel_param not in (None, "") else "",
            cursor_param=str(cursor_param).strip() if cursor_param not in (None, "") else "",
            initial_cursor=initial_cursor,
            channel_serializer=channel_serializer,
            http_method=str(http_method or "GET"),
            params_in_body=bool(params_in_body),
            channels_in_body=bool(channels_in_body),
            cursor_in_body=bool(cursor_in_body),
            body_params=body_params or None,
            body_encoder=body_encoder,
            buffer_size=buffer_size,
            metrics_registry=self._metrics,
        ).start()

    # ------------------------------------------------------------------
    # Konfiguracja sieciowa
    # ------------------------------------------------------------------
    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # type: ignore[override]
        self._ip_allowlist = tuple(ip_allowlist) if ip_allowlist else None

    # ------------------------------------------------------------------
    # Dane konta i publiczne API
    # ------------------------------------------------------------------
    def fetch_account_snapshot(self) -> AccountSnapshot:  # type: ignore[override]
        if "read" not in self._permission_set:
            raise PermissionError("Poświadczenia Kraken nie mają uprawnień do odczytu.")
        def _call() -> AccountSnapshot:
            balances_payload = self._private_request(_RequestContext(path="/0/private/Balance", params={}))
            trade_balance_payload = self._private_request(
                _RequestContext(path="/0/private/TradeBalance", params={"asset": self._valuation_asset})
            )

            balances_data = balances_payload.get("result", {}) if isinstance(balances_payload, Mapping) else {}
            balances: MutableMapping[str, float] = {}
            for asset, amount in balances_data.items():
                try:
                    balances[asset] = float(amount)
                except (TypeError, ValueError):
                    continue

            trade_data = trade_balance_payload.get("result", {}) if isinstance(trade_balance_payload, Mapping) else {}
            total_equity = float(trade_data.get("eb", trade_data.get("e", 0.0)) or 0.0)
            available_margin = float(trade_data.get("mf", 0.0) or 0.0)
            maintenance_margin = float(trade_data.get("m", 0.0) or 0.0)

            return AccountSnapshot(
                balances=dict(balances),
                total_equity=total_equity,
                available_margin=available_margin,
                maintenance_margin=maintenance_margin,
            )

        return self._watchdog.execute("kraken_spot_fetch_account", _call)

    def fetch_symbols(self) -> Sequence[str]:  # type: ignore[override]
        payload = self._public_request("/0/public/AssetPairs", params={})
        result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
        symbols: list[str] = []
        for value in result.values():
            if isinstance(value, Mapping):
                altname = value.get("altname") or value.get("wsname")
                if isinstance(altname, str):
                    symbols.append(altname)
        return sorted(set(symbols))

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:  # type: ignore[override]
        minutes = _INTERVAL_MAPPING.get(interval)
        if minutes is None:
            raise ValueError(f"Nieobsługiwany interwał {interval!r} dla Kraken Spot")

        params: MutableMapping[str, Any] = {"pair": symbol, "interval": minutes}
        if start:
            params["since"] = int(start)
        payload = self._public_request("/0/public/OHLC", params=params)
        result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
        candles = []
        for values in result.values():
            if isinstance(values, Sequence):
                for candle in values:
                    if isinstance(candle, Sequence) and len(candle) >= 6:
                        ts, o, h, l, c, v = candle[:6]
                        candles.append([float(ts), float(o), float(h), float(l), float(c), float(v)])
                break
        if limit is not None:
            candles = candles[:limit]
        return candles

    def fetch_ticker(self, symbol: str) -> KrakenTicker:
        """Pobiera ticker 24h i aktualizuje metryki top-of-book."""

        payload = self._public_request("/0/public/Ticker", params={"pair": symbol})
        result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(result, Mapping) or not result:
            raise ExchangeAPIError(
                "Kraken nie zwrócił danych tickera dla wskazanego symbolu.",
                400,
                payload=payload,
            )

        # Kraken może zwrócić wiele par (np. aliasy); wybieramy pierwszą.
        first_entry = next(iter(result.values()))
        if not isinstance(first_entry, Mapping):
            raise ExchangeAPIError(
                "Kraken zwrócił niepoprawną strukturę tickera.",
                400,
                payload=payload,
            )

        def _first_value(field: object) -> object:
            if isinstance(field, Sequence) and not isinstance(field, (str, bytes, bytearray)):
                return field[0] if field else None
            return field

        def _last_value(field: object) -> object:
            if isinstance(field, Sequence) and not isinstance(field, (str, bytes, bytearray)):
                return field[-1] if field else None
            return field

        best_ask = _to_float(_first_value(first_entry.get("a")))
        best_bid = _to_float(_first_value(first_entry.get("b")))
        last_price = _to_float(_first_value(first_entry.get("c")))
        volume_field = first_entry.get("v")
        volume_24h = 0.0
        tail_value = _last_value(volume_field)
        if tail_value is not None:
            volume_24h = _to_float(tail_value)
        vwap_field = first_entry.get("p")
        vwap_24h = 0.0
        tail_value = _last_value(vwap_field)
        if tail_value is not None:
            vwap_24h = _to_float(tail_value)
        high_field = first_entry.get("h")
        high_24h = 0.0
        tail_value = _last_value(high_field)
        if tail_value is not None:
            high_24h = _to_float(tail_value)
        low_field = first_entry.get("l")
        low_24h = 0.0
        tail_value = _last_value(low_field)
        if tail_value is not None:
            low_24h = _to_float(tail_value)
        open_field = first_entry.get("o")
        open_price = 0.0
        tail_value = _last_value(open_field)
        if tail_value is not None:
            open_price = _to_float(tail_value)
        timestamp = time.time()

        ticker = KrakenTicker(
            symbol=symbol,
            best_ask=best_ask,
            best_bid=best_bid,
            last_price=last_price,
            volume_24h=volume_24h,
            vwap_24h=vwap_24h,
            high_24h=high_24h,
            low_24h=low_24h,
            open_price=open_price,
            timestamp=timestamp,
        )

        labels = self._labels(endpoint="/0/public/Ticker", signed="false", symbol=symbol)
        self._metric_ticker_last_price.set(ticker.last_price, labels=labels)
        spread = max(ticker.best_ask - ticker.best_bid, 0.0)
        self._metric_ticker_spread.set(spread, labels=labels)
        return ticker

    def fetch_order_book(self, symbol: str, *, depth: int = 50) -> KrakenOrderBook:
        """Pobiera orderbook (bids/asks) do analizy płynności i audytu ryzyka."""

        if depth <= 0:
            raise ValueError("Parametr depth musi być dodatni.")

        payload = self._public_request("/0/public/Depth", params={"pair": symbol, "count": depth})
        result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(result, Mapping) or not result:
            raise ExchangeAPIError(
                "Kraken nie zwrócił danych orderbooka dla wskazanego symbolu.",
                400,
                payload=payload,
            )

        first_entry = next(iter(result.values()))
        if not isinstance(first_entry, Mapping):
            raise ExchangeAPIError(
                "Kraken zwrócił niepoprawną strukturę orderbooka.",
                400,
                payload=payload,
            )

        bids_raw = first_entry.get("bids")
        if not isinstance(bids_raw, Sequence) or isinstance(bids_raw, (str, bytes, bytearray)):
            bids_raw = []
        asks_raw = first_entry.get("asks")
        if not isinstance(asks_raw, Sequence) or isinstance(asks_raw, (str, bytes, bytearray)):
            asks_raw = []

        bids: list[KrakenOrderBookEntry] = []
        for row in bids_raw:
            if isinstance(row, Sequence) and len(row) >= 2:
                price = _to_float(row[0])
                volume = _to_float(row[1])
                ts = _to_float(row[2]) if len(row) > 2 else 0.0
                bids.append(KrakenOrderBookEntry(price=price, volume=volume, timestamp=ts))

        asks: list[KrakenOrderBookEntry] = []
        for row in asks_raw:
            if isinstance(row, Sequence) and len(row) >= 2:
                price = _to_float(row[0])
                volume = _to_float(row[1])
                ts = _to_float(row[2]) if len(row) > 2 else 0.0
                asks.append(KrakenOrderBookEntry(price=price, volume=volume, timestamp=ts))

        bids.sort(key=lambda item: item.price, reverse=True)
        asks.sort(key=lambda item: item.price)

        snapshot_time = time.time()
        order_book = KrakenOrderBook(
            symbol=symbol,
            bids=tuple(bids[:depth]),
            asks=tuple(asks[:depth]),
            depth=min(depth, len(bids) + len(asks)),
            timestamp=snapshot_time,
        )

        labels = self._labels(endpoint="/0/public/Depth", signed="false", symbol=symbol)
        self._metric_orderbook_levels.set(float(len(order_book.bids) + len(order_book.asks)), labels=labels)
        return order_book

    # ------------------------------------------------------------------
    # Operacje tradingowe
    # ------------------------------------------------------------------
    def place_order(self, request: OrderRequest) -> OrderResult:  # type: ignore[override]
        if "trade" not in self._permission_set:
            raise PermissionError("Poświadczenia Kraken nie mają uprawnień tradingowych.")

        params: MutableMapping[str, Any] = {
            "pair": request.symbol,
            "type": request.side.lower(),
            "volume": f"{request.quantity:.10f}",
        }

        order_type = request.order_type.lower()
        if order_type == "market":
            params["ordertype"] = "market"
        elif order_type == "limit":
            if request.price is None:
                raise ValueError("Zlecenie limit na Kraken wymaga ceny.")
            params["ordertype"] = "limit"
            params["price"] = f"{request.price:.10f}"
        else:
            raise ValueError(f"Nieobsługiwany typ zlecenia dla Kraken Spot: {request.order_type}")

        if request.time_in_force:
            tif = request.time_in_force.upper()
            if tif not in {"GTC", "IOC", "GTD"}:
                raise ValueError(f"Nieobsługiwane time in force '{tif}' dla Kraken Spot")
            params["timeinforce"] = tif
        if request.client_order_id:
            params["userref"] = request.client_order_id

        def _call() -> OrderResult:
            payload = self._private_request(_RequestContext(path="/0/private/AddOrder", params=params))
            result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
            txid_seq = result.get("txid") if isinstance(result, Mapping) else None
            txid: str | None = None
            if isinstance(txid_seq, Sequence) and txid_seq:
                first = txid_seq[0]
                if isinstance(first, str):
                    txid = first
            return OrderResult(
                order_id=txid or "",
                status="NEW",
                filled_quantity=0.0,
                avg_price=None,
                raw_response=result if isinstance(result, Mapping) else {},
            )

        return self._watchdog.execute("kraken_spot_place_order", _call)

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # type: ignore[override]
        if "trade" not in self._permission_set:
            raise PermissionError("Poświadczenia Kraken nie mają uprawnień tradingowych.")
        def _call() -> None:
            params: Mapping[str, Any] = {"txid": order_id}
            payload = self._private_request(_RequestContext(path="/0/private/CancelOrder", params=params))
            result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
            if not isinstance(result, Mapping) or int(result.get("count", 0)) < 1:
                raise ExchangeAPIError(
                    "Kraken nie potwierdził anulowania zlecenia.",
                    400,
                    payload=payload,
                )

        self._watchdog.execute("kraken_spot_cancel_order", _call)

    # ------------------------------------------------------------------
    # Raportowanie stanu konta
    # ------------------------------------------------------------------
    def fetch_open_orders(self) -> Sequence[KrakenOpenOrder]:
        """Zwraca aktualne otwarte zlecenia wraz z metadanymi."""

        if "read" not in self._permission_set:
            raise PermissionError("Poświadczenia Kraken nie mają uprawnień do odczytu zleceń.")

        def _call() -> Sequence[KrakenOpenOrder]:
            context = _RequestContext(path="/0/private/OpenOrders", params={"trades": True})
            payload = self._private_request(context)
            result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
            open_orders_payload = result.get("open", {}) if isinstance(result, Mapping) else {}

            orders: list[KrakenOpenOrder] = []
            if isinstance(open_orders_payload, Mapping):
                for order_id, entry in open_orders_payload.items():
                    if not isinstance(order_id, str) or not isinstance(entry, Mapping):
                        continue
                    descr = entry.get("descr") if isinstance(entry.get("descr"), Mapping) else {}
                    pair = descr.get("pair") if isinstance(descr, Mapping) else None
                    order_type = descr.get("ordertype") if isinstance(descr, Mapping) else None
                    side = descr.get("type") if isinstance(descr, Mapping) else None
                    price_value: float | None = None
                    if isinstance(descr, Mapping):
                        raw_price = descr.get("price") or descr.get("price2")
                        price_value = _to_float(raw_price, default=float("nan"))
                        if price_value != price_value:  # NaN -> brak ceny
                            price_value = None

                    flags_field = entry.get("oflags")
                    flags: tuple[str, ...]
                    if isinstance(flags_field, str):
                        flags = tuple(flag for flag in flags_field.split(",") if flag)
                    elif isinstance(flags_field, Sequence):
                        flags = tuple(str(flag) for flag in flags_field if isinstance(flag, str))
                    else:
                        flags = ()

                    order = KrakenOpenOrder(
                        order_id=order_id,
                        symbol=str(pair) if isinstance(pair, str) else "",
                        side=str(side) if isinstance(side, str) else "",
                        order_type=str(order_type) if isinstance(order_type, str) else "",
                        price=price_value,
                        volume=_to_float(entry.get("vol")),
                        volume_executed=_to_float(entry.get("vol_exec")),
                        timestamp=_to_float(entry.get("opentm")),
                        flags=flags,
                    )
                    orders.append(order)

            orders.sort(key=lambda item: item.timestamp)
            labels = self._labels(endpoint=context.path, signed="true")
            self._metric_open_orders.set(float(len(orders)), labels=labels)
            return orders

        return self._watchdog.execute("kraken_spot_fetch_open_orders", _call)

    def fetch_trades_history(
        self,
        *,
        start: int | None = None,
        end: int | None = None,
        symbol: str | None = None,
        max_pages: int = 6,
    ) -> Sequence[KrakenTrade]:
        """Pobiera historię transakcji z opcjonalną filtracją po symbolu."""

        if "read" not in self._permission_set:
            raise PermissionError("Poświadczenia Kraken nie mają uprawnień do odczytu historii transakcji.")

        if max_pages <= 0:
            raise ValueError("Parametr max_pages musi być dodatni.")

        base_params: MutableMapping[str, Any] = {"trades": True, "type": "all", "ofs": 0}
        if start is not None:
            base_params["start"] = int(start)
        if end is not None:
            base_params["end"] = int(end)

        trades: list[KrakenTrade] = []
        fetched_raw = 0
        labels = self._labels(endpoint="/0/private/TradesHistory", signed="true")

        for _ in range(max_pages):
            context = _RequestContext(path="/0/private/TradesHistory", params=dict(base_params))
            payload = self._private_request(context)
            result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
            trades_payload = result.get("trades", {}) if isinstance(result, Mapping) else {}

            page_items = []
            if isinstance(trades_payload, Mapping):
                for trade_id, entry in trades_payload.items():
                    if not isinstance(trade_id, str) or not isinstance(entry, Mapping):
                        continue
                    pair = entry.get("pair")
                    if symbol and str(pair) != symbol:
                        continue
                    trade = KrakenTrade(
                        trade_id=trade_id,
                        order_id=str(entry.get("ordertxid")) if entry.get("ordertxid") else None,
                        symbol=str(pair) if isinstance(pair, str) else "",
                        side=str(entry.get("type")) if isinstance(entry.get("type"), str) else "",
                        order_type=str(entry.get("ordertype"))
                        if isinstance(entry.get("ordertype"), str)
                        else "",
                        price=_to_float(entry.get("price")),
                        volume=_to_float(entry.get("vol")),
                        cost=_to_float(entry.get("cost")),
                        fee=_to_float(entry.get("fee")),
                        timestamp=_to_float(entry.get("time")),
                    )
                    page_items.append(trade)

            trades.extend(page_items)
            self._metric_trades.inc(amount=float(len(page_items)), labels=labels)

            raw_count = len(trades_payload) if isinstance(trades_payload, Mapping) else 0
            fetched_raw += raw_count
            count_value = result.get("count") if isinstance(result, Mapping) else None
            total_expected = int(count_value) if isinstance(count_value, (int, float)) else raw_count
            if raw_count == 0 or fetched_raw >= total_expected:
                break
            base_params["ofs"] = fetched_raw

        trades.sort(key=lambda item: item.timestamp)
        return trades

    # ------------------------------------------------------------------
    # Streaming (do implementacji w dalszych etapach)
    # ------------------------------------------------------------------
    def stream_public_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        return self._build_stream("public", channels)

    def stream_private_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        if not ({"read", "trade"} & self._permission_set):
            raise PermissionError("Poświadczenia Kraken nie pozwalają na prywatny stream danych.")
        return self._build_stream("private", channels)

    # ------------------------------------------------------------------
    # Wewnętrzne narzędzia HTTP/podpisy
    # ------------------------------------------------------------------
    def _public_request(self, path: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        url = f"{self._base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"

        def build_request() -> Request:
            return Request(url, headers=dict(_DEFAULT_HEADERS))

        return self._perform_request(build_request, endpoint=path, signed=False)

    def _private_request(self, context: _RequestContext) -> Mapping[str, Any]:
        if not self.credentials.secret:
            raise PermissionError("Poświadczenia Kraken wymagają sekretu do wywołań prywatnych.")

        sorted_items = [(key, context.params[key]) for key in sorted(context.params.keys())]

        def build_request() -> Request:
            nonce = self._generate_nonce()
            post_items = [("nonce", nonce)] + [(k, v) for k, v in sorted_items]
            encoded = urlencode(post_items)
            data = encoded.encode("utf-8")

            encoded_params = urlencode(sorted_items)
            message = (nonce + encoded_params).encode("utf-8")
            sha_digest = hashlib.sha256(message).digest()
            decoded_secret = base64.b64decode(self.credentials.secret)
            mac = hmac.new(decoded_secret, (context.path.encode("utf-8") + sha_digest), hashlib.sha512)
            signature = base64.b64encode(mac.digest()).decode("utf-8")

            headers = dict(_DEFAULT_HEADERS)
            headers.update(
                {
                    "API-Key": self.credentials.key_id,
                    "API-Sign": signature,
                    "Content-Type": "application/x-www-form-urlencoded",
                }
            )
            return Request(f"{self._base_url}{context.path}", data=data, headers=headers)

        payload = self._perform_request(build_request, endpoint=context.path, signed=True)
        return payload

    def _perform_request(
        self,
        request_factory: Callable[[], Request],
        *,
        endpoint: str,
        signed: bool,
    ) -> Mapping[str, Any]:
        attempt = 0
        backoff = _BASE_BACKOFF
        while True:
            weight = 2.0 if signed else 1.0
            self._rate_limiter.acquire(weight=weight)
            request = request_factory()
            labels = self._labels(endpoint=endpoint, signed="true" if signed else "false")
            start = time.monotonic()
            try:
                if signed:
                    self._metric_signed_requests.inc(labels=self._metric_base_labels)
                with urlopen(request, timeout=self._http_timeout) as response:
                    payload = self._parse_response(response.read(), labels)
            except HTTPError as exc:
                duration = max(time.monotonic() - start, 0.0)
                self._metric_http_latency.observe(duration, labels=labels)
                reason = self._classify_http_error(exc.code)
                self._metric_api_errors.inc(labels={**labels, "reason": reason})
                if exc.code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                    attempt += 1
                    self._metric_retries.inc(labels={**labels, "reason": reason})
                    sleep_for = min(backoff * (2 ** (attempt - 1)), _BACKOFF_CAP)
                    jitter = random.uniform(0.0, 0.15 * sleep_for)
                    _LOGGER.debug(
                        "Retrying Kraken Spot request %s after HTTPError %s (attempt %s)",
                        endpoint,
                        exc.code,
                        attempt,
                    )
                    time.sleep(sleep_for + jitter)
                    continue
                if exc.code in {401, 403}:
                    raise ExchangeAuthError(
                        f"Kraken odrzucił uwierzytelnienie ({exc.code}).",
                        exc.code,
                        payload=None,
                    ) from exc
                if exc.code == 429:
                    raise ExchangeThrottlingError(
                        "Kraken zgłosił limit zapytań (HTTP 429).",
                        exc.code,
                        payload=None,
                    ) from exc
                raise ExchangeAPIError(
                    f"Kraken API zgłosiło błąd HTTP {exc.code}.",
                    exc.code,
                    payload=None,
                ) from exc
            except URLError as exc:
                duration = max(time.monotonic() - start, 0.0)
                self._metric_http_latency.observe(duration, labels=labels)
                self._metric_api_errors.inc(labels={**labels, "reason": "network"})
                if attempt < _MAX_RETRIES - 1:
                    attempt += 1
                    self._metric_retries.inc(labels={**labels, "reason": "network"})
                    sleep_for = min(backoff * (2 ** (attempt - 1)), _BACKOFF_CAP)
                    _LOGGER.debug(
                        "Retrying Kraken Spot request %s after network error (attempt %s)",
                        endpoint,
                        attempt,
                    )
                    time.sleep(sleep_for)
                    continue
                raise ExchangeNetworkError("Błąd sieciowy podczas komunikacji z Kraken Spot.", reason=exc) from exc
            duration = max(time.monotonic() - start, 0.0)
            self._metric_http_latency.observe(duration, labels=labels)
            self._ensure_no_error(payload, endpoint=endpoint, signed=signed)
            return payload

    def _parse_response(self, body: bytes, labels: Mapping[str, str]) -> Mapping[str, Any]:
        try:
            text = body.decode("utf-8")
            payload = json.loads(text)
        except (UnicodeDecodeError, JSONDecodeError) as exc:
            self._metric_api_errors.inc(labels={**labels, "reason": "json"})
            raise ExchangeAPIError(
                "Kraken zwrócił niepoprawny JSON.",
                500,
                payload=None,
            ) from exc
        if not isinstance(payload, Mapping):
            self._metric_api_errors.inc(labels={**labels, "reason": "json"})
            raise ExchangeAPIError(
                "Kraken zwrócił nieoczekiwaną strukturę odpowiedzi.",
                500,
                payload=None,
            )
        return payload

    def _classify_http_error(self, status_code: int) -> str:
        if status_code in {401, 403}:
            return "auth"
        if status_code == 429:
            return "throttled"
        if 500 <= status_code < 600:
            return "server_error"
        return "http_error"

    def _ensure_no_error(self, payload: Mapping[str, Any], *, endpoint: str, signed: bool) -> None:
        errors = payload.get("error") if isinstance(payload, Mapping) else None
        if not errors:
            return
        labels = self._labels(endpoint=endpoint, signed="true" if signed else "false")
        try:
            raise_for_kraken_error(
                payload=payload,
                default_message=f"Kraken API zwróciło błąd ({endpoint})",
            )
        except ExchangeAuthError:
            self._metric_api_errors.inc(labels={**labels, "reason": "auth"})
            raise
        except ExchangeThrottlingError:
            self._metric_api_errors.inc(labels={**labels, "reason": "throttled"})
            raise
        except ExchangeAPIError:
            self._metric_api_errors.inc(labels={**labels, "reason": "api_error"})
            raise

    def _generate_nonce(self) -> str:
        candidate = int(time.time() * 1000)
        if candidate <= self._last_nonce:
            candidate = self._last_nonce + 1
        self._last_nonce = candidate
        return str(candidate)

    def _labels(self, **extra: str) -> dict[str, str]:
        labels = dict(self._metric_base_labels)
        labels.update((key, str(value)) for key, value in extra.items())
        return labels


__all__ = [
    "KrakenSpotAdapter",
    "KrakenOpenOrder",
    "KrakenTrade",
    "KrakenTicker",
    "KrakenOrderBookEntry",
    "KrakenOrderBook",
]
