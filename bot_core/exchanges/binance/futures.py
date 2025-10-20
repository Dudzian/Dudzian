"""Adapter REST dla kontraktów terminowych Binance (USD-M)."""
from __future__ import annotations

from dataclasses import dataclass
import hmac
import json
import logging
import random
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
from bot_core.exchanges.binance._utils import (
    _normalize_depth,
    _stringify_params,
    _timestamp_ms_to_seconds,
    _to_float,
)
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.error_mapping import raise_for_binance_error
from bot_core.exchanges.streaming import LocalLongPollStream
from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry

_LOGGER = logging.getLogger(__name__)

_DEFAULT_HEADERS = {"User-Agent": "bot-core/1.0 (+https://github.com/)"}
_RETRYABLE_STATUS = {418, 429}
_MAX_RETRIES = 3
_BASE_BACKOFF = 0.4
_BACKOFF_CAP = 4.0
_JITTER_RANGE = (0.05, 0.35)
_WEIGHT_HEADERS = (
    "x-mbx-used-weight",
    "x-mbx-used-weight-1m",
    "x-mbx-order-count-10s",
    "x-mbx-order-count-1m",
)


@dataclass(slots=True)
class FuturesPosition:
    """Model pojedynczej pozycji futures wykorzystywany w raportach hedgingowych."""

    symbol: str
    side: str
    quantity: float
    entry_price: float
    mark_price: float
    notional: float
    unrealized_pnl: float
    leverage: float
    isolated: bool
    liquidation_price: float | None = None

    def to_mapping(self) -> dict[str, object]:
        data: dict[str, object] = {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "mark_price": self.mark_price,
            "notional": self.notional,
            "unrealized_pnl": self.unrealized_pnl,
            "leverage": self.leverage,
            "isolated": self.isolated,
        }
        if self.liquidation_price is not None:
            data["liquidation_price"] = self.liquidation_price
        return data


@dataclass(slots=True)
class FundingRateEvent:
    """Pojedyncze zdarzenie stopy finansowania z rynku futures."""

    symbol: str
    funding_rate: float
    funding_time: int
    mark_price: float | None = None
    next_funding_time: int | None = None
    interest_rate: float | None = None

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


def _to_int(value: object, default: int | None = 0) -> int | None:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _extract_weight(headers: Mapping[str, str]) -> float | None:
    for header in _WEIGHT_HEADERS:
        if header in headers:
            try:
                return float(headers[header])
            except (TypeError, ValueError):  # pragma: no cover - metryka pomocnicza
                return None
    return None


class BinanceFuturesAdapter(ExchangeAdapter):
    """Adapter REST dla rynku terminowego Binance (USD-M)."""

    __slots__ = (
        "_environment",
        "_public_base",
        "_trading_base",
        "_permission_set",
        "_ip_allowlist",
        "_settings",
        "_metrics",
        "_metric_base_labels",
        "_metric_http_latency",
        "_metric_retries",
        "_metric_signed_requests",
        "_metric_weight",
        "_metric_position_notional",
        "_metric_position_active",
        "_metric_position_long",
        "_metric_position_short",
        "_metric_position_gross",
        "_metric_position_net",
        "_metric_funding_rate",
        "_tracked_position_labels",
    )

    name: str = "binance_futures"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
        settings: Mapping[str, object] | None = None,
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        super().__init__(credentials)
        self._environment = environment or credentials.environment
        self._public_base = _determine_public_base(self._environment)
        self._trading_base = _determine_trading_base(self._environment)
        self._permission_set = frozenset(perm.lower() for perm in self._credentials.permissions)
        self._ip_allowlist: tuple[str, ...] = ()
        self._settings = dict(settings or {})
        self._metrics = metrics_registry or get_global_metrics_registry()
        self._metric_base_labels = {
            "exchange": self.name,
            "environment": self._environment.value,
        }
        self._metric_http_latency = self._metrics.histogram(
            "binance_futures_http_latency_seconds",
            "Czas trwania zapytań HTTP do API Binance Futures.",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
        )
        self._metric_retries = self._metrics.counter(
            "binance_futures_retries_total",
            "Liczba ponowień zapytań do API Binance Futures (powód=throttled/network/server_error).",
        )
        self._metric_signed_requests = self._metrics.counter(
            "binance_futures_signed_requests_total",
            "Łączna liczba podpisanych zapytań HTTP wysłanych do API Binance Futures.",
        )
        self._metric_weight = self._metrics.gauge(
            "binance_futures_used_weight",
            "Ostatnie wartości nagłówków X-MBX-USED-WEIGHT z API Binance Futures.",
        )
        self._metric_position_notional = self._metrics.gauge(
            "binance_futures_position_notional",
            "Wielkość notional otwartych pozycji Binance Futures w USDT.",
        )
        self._metric_position_active = self._metrics.gauge(
            "binance_futures_open_positions",
            "Liczba otwartych pozycji na Binance Futures.",
        )
        self._metric_position_long = self._metrics.gauge(
            "binance_futures_long_notional_total",
            "Suma notional pozycji długich Binance Futures w USDT.",
        )
        self._metric_position_short = self._metrics.gauge(
            "binance_futures_short_notional_total",
            "Suma notional pozycji krótkich Binance Futures w USDT.",
        )
        self._metric_position_gross = self._metrics.gauge(
            "binance_futures_gross_notional",
            "Łączny notional pozycji (long+short) na Binance Futures w USDT.",
        )
        self._metric_position_net = self._metrics.gauge(
            "binance_futures_net_notional",
            "Wartość netto ekspozycji (long-short) na Binance Futures w USDT.",
        )
        self._metric_funding_rate = self._metrics.gauge(
            "binance_futures_funding_rate",
            "Ostatnio zarejestrowane stopy finansowania na Binance Futures.",
        )
        self._tracked_position_labels: set[tuple[str, str]] = set()

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
        )

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
        return self._execute_request(request, signed=False, endpoint=path)

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
        return self._execute_request(request, signed=True, endpoint=path)

    def _execute_request(
        self,
        request: Request,
        *,
        signed: bool,
        endpoint: str,
    ) -> dict[str, object] | list[object]:
        attempt = 0
        while True:
            start = time.perf_counter()
            try:
                with urlopen(request, timeout=15) as response:  # nosec: B310 - zaufany endpoint
                    status_code = getattr(response, "status", getattr(response, "code", 200))
                    payload = response.read()
                    headers = {k.lower(): v for k, v in response.headers.items()}
            except HTTPError as exc:  # pragma: no cover - zachowanie walidowane w testach jednostkowych
                error = self._translate_http_error(exc)
                if self._should_retry(error.status_code) and attempt < _MAX_RETRIES:
                    self._record_retry(reason=self._retry_reason(error.status_code))
                    self._sleep(self._backoff_delay(attempt))
                    attempt += 1
                    continue
                raise error from exc
            except URLError as exc:  # pragma: no cover - zachowanie walidowane w testach jednostkowych
                if attempt < _MAX_RETRIES:
                    self._record_retry(reason="network")
                    self._sleep(self._backoff_delay(attempt))
                    attempt += 1
                    continue
                raise ExchangeNetworkError("Nie udało się połączyć z API Binance Futures", reason=exc) from exc
            else:
                elapsed = max(time.perf_counter() - start, 0.0)
                labels = dict(self._metric_base_labels)
                labels["method"] = request.method or "GET"
                self._metric_http_latency.observe(elapsed, labels=labels)
                weight = _extract_weight(headers)
                if weight is not None:
                    self._metric_weight.set(weight, labels=self._metric_base_labels)
                if signed:
                    self._metric_signed_requests.inc(labels=self._metric_base_labels)
                return self._parse_payload(
                    payload,
                    status_code=int(status_code or 200),
                    endpoint=endpoint,
                )

    def _parse_payload(
        self,
        payload: bytes,
        *,
        status_code: int,
        endpoint: str,
    ) -> dict[str, object] | list[object]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:  # pragma: no cover - błąd po stronie API
            _LOGGER.error("Niepoprawna odpowiedź JSON z Binance Futures: %s", exc)
            raise ExchangeAPIError(
                message="Niepoprawna odpowiedź JSON od API Binance Futures",
                status_code=500,
                payload=payload.decode("utf-8", "replace"),
            ) from exc
        self._raise_for_api_error(
            data,
            status_code=status_code,
            default_message=f"Binance Futures API zwróciło błąd ({endpoint})",
        )
        return data

    def _translate_http_error(self, exc: HTTPError) -> ExchangeAPIError:
        status = getattr(exc, "code", 500)
        raw_body = exc.read() or b""
        parsed_payload: object | None = None
        message = exc.reason if hasattr(exc, "reason") else ""
        if raw_body:
            try:
                parsed_payload = json.loads(raw_body)
            except json.JSONDecodeError:
                parsed_payload = raw_body.decode("utf-8", "replace")
            else:
                if isinstance(parsed_payload, Mapping):
                    message = str(
                        parsed_payload.get("msg")
                        or parsed_payload.get("message")
                        or parsed_payload.get("code")
                        or message
                        or "Binance Futures API zwróciło błąd"
                    )
                else:
                    message = str(parsed_payload)
        if not message:
            message = "Binance Futures API zwróciło błąd"

        error_cls: type[ExchangeAPIError]
        if status in {401, 403}:
            error_cls = ExchangeAuthError
        elif status in _RETRYABLE_STATUS:
            error_cls = ExchangeThrottlingError
        else:
            error_cls = ExchangeAPIError

        return error_cls(message=message, status_code=status, payload=parsed_payload)

    def _should_retry(self, status_code: int) -> bool:
        return status_code in _RETRYABLE_STATUS or status_code >= 500

    def _retry_reason(self, status_code: int) -> str:
        if status_code in _RETRYABLE_STATUS:
            return "throttled"
        if status_code >= 500:
            return "server_error"
        return "unknown"

    def _backoff_delay(self, attempt: int) -> float:
        delay = min(_BASE_BACKOFF * (2**attempt), _BACKOFF_CAP)
        jitter = random.uniform(*_JITTER_RANGE)
        return delay + jitter

    def _record_retry(self, *, reason: str) -> None:
        labels = dict(self._metric_base_labels)
        labels["reason"] = reason
        self._metric_retries.inc(labels=labels)

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def _raise_for_api_error(
        self,
        payload: object,
        *,
        status_code: int,
        default_message: str,
    ) -> None:
        if not isinstance(payload, Mapping):
            return

        code_value = payload.get("code")
        try:
            numeric_code = int(code_value) if code_value is not None else None
        except (TypeError, ValueError):
            numeric_code = None

        if numeric_code not in (None, 0):
            raise_for_binance_error(
                status_code=status_code,
                payload=payload,
                default_message=default_message,
            )

        success = payload.get("success")
        if success in {False, "false", "False"}:
            raise_for_binance_error(
                status_code=status_code,
                payload=payload,
                default_message=default_message,
            )

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

    @staticmethod
    def _normalize_contract_symbol(symbol: str) -> str:
        """Zwraca symbol kontraktu w notacji wymaganej przez API (np. BTCUSDT)."""

        cleaned = str(symbol or "").strip().upper().replace("/", "")
        if not cleaned:
            raise ValueError("Symbol kontraktu Binance Futures nie może być pusty.")
        return cleaned

    def fetch_ticker(self, symbol: str) -> BinanceFuturesTicker:
        """Pobiera statystyki 24h dla kontraktu USD-M."""

        exchange_symbol = self._normalize_contract_symbol(symbol)

        payload = self._public_request("/fapi/v1/ticker/24hr", params={"symbol": exchange_symbol})
        if not isinstance(payload, Mapping):
            raise ExchangeAPIError(
                "Binance Futures zwrócił niepoprawną strukturę tickera.",
                400,
                payload=payload,
            )

        best_bid = _to_float(payload.get("bidPrice"))
        best_ask = _to_float(payload.get("askPrice"))
        last_price = _to_float(payload.get("lastPrice"))
        price_change_percent = _to_float(payload.get("priceChangePercent"))
        open_price = _to_float(payload.get("openPrice"))
        high_24h = _to_float(payload.get("highPrice"))
        low_24h = _to_float(payload.get("lowPrice"))
        volume_base = _to_float(payload.get("volume"))
        volume_quote = _to_float(payload.get("quoteVolume"))
        open_interest = _to_float(payload.get("openInterest"))
        timestamp = _timestamp_ms_to_seconds(payload.get("closeTime"), fallback=time.time())

        return BinanceFuturesTicker(
            symbol=exchange_symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            last_price=last_price,
            price_change_percent=price_change_percent,
            open_price=open_price,
            high_24h=high_24h,
            low_24h=low_24h,
            volume_24h_base=volume_base,
            volume_24h_quote=volume_quote,
            open_interest=open_interest,
            timestamp=timestamp,
        )

    def fetch_order_book(self, symbol: str, *, depth: int = 50) -> BinanceFuturesOrderBook:
        """Pobiera orderbook futures ograniczony do wskazanej głębokości."""

        exchange_symbol = self._normalize_contract_symbol(symbol)

        normalized_depth = _normalize_depth(depth)
        params = {"symbol": exchange_symbol, "limit": normalized_depth}
        payload = self._public_request("/fapi/v1/depth", params=params)
        if not isinstance(payload, Mapping):
            raise ExchangeAPIError(
                "Binance Futures zwrócił niepoprawną strukturę orderbooka.",
                400,
                payload=payload,
            )

        bids_raw = payload.get("bids")
        asks_raw = payload.get("asks")
        bids: list[BinanceFuturesOrderBookLevel] = []
        if isinstance(bids_raw, Sequence):
            for entry in bids_raw:
                if not isinstance(entry, Sequence) or len(entry) < 2:
                    continue
                price = _to_float(entry[0])
                quantity = _to_float(entry[1])
                if price <= 0 or quantity <= 0:
                    continue
                bids.append(BinanceFuturesOrderBookLevel(price=price, quantity=quantity))

        asks: list[BinanceFuturesOrderBookLevel] = []
        if isinstance(asks_raw, Sequence):
            for entry in asks_raw:
                if not isinstance(entry, Sequence) or len(entry) < 2:
                    continue
                price = _to_float(entry[0])
                quantity = _to_float(entry[1])
                if price <= 0 or quantity <= 0:
                    continue
                asks.append(BinanceFuturesOrderBookLevel(price=price, quantity=quantity))

        try:
            last_update_id = int(payload.get("lastUpdateId", 0))
        except (TypeError, ValueError):
            last_update_id = 0
        timestamp = _timestamp_ms_to_seconds(payload.get("T") or payload.get("E"), fallback=time.time())

        return BinanceFuturesOrderBook(
            symbol=exchange_symbol,
            bids=tuple(bids),
            asks=tuple(asks),
            depth=normalized_depth,
            last_update_id=last_update_id,
            timestamp=timestamp,
        )

    def fetch_open_orders(self) -> Sequence[BinanceFuturesOpenOrder]:
        """Zwraca otwarte zlecenia futures poprzez podpisane API."""

        if not ({"read", "trade"} & self._permission_set):
            raise PermissionError("Poświadczenia nie pozwalają na odczyt zleceń Binance Futures.")

        payload = self._signed_request("/fapi/v1/openOrders")
        if not isinstance(payload, list):
            raise ExchangeAPIError(
                "Binance Futures zwrócił niepoprawną strukturę listy zleceń.",
                400,
                payload=payload,
            )

        orders: list[BinanceFuturesOpenOrder] = []
        for entry in payload:
            if not isinstance(entry, Mapping):
                continue
            raw_symbol = entry.get("symbol")
            if isinstance(raw_symbol, str):
                try:
                    exchange_symbol = self._normalize_contract_symbol(raw_symbol)
                except ValueError:
                    exchange_symbol = raw_symbol.strip()
            else:
                exchange_symbol = ""
            price_value = _to_float(entry.get("price"))
            price = price_value if price_value > 0 else None
            stop_value = _to_float(entry.get("stopPrice"))
            stop_price = stop_value if stop_value > 0 else None
            timestamp = _timestamp_ms_to_seconds(entry.get("updateTime") or entry.get("time"), fallback=time.time())
            order = BinanceFuturesOpenOrder(
                order_id=str(entry.get("orderId", "")),
                symbol=exchange_symbol,
                status=str(entry.get("status", "")),
                side=str(entry.get("side", "")),
                order_type=str(entry.get("type", "")),
                price=price,
                orig_quantity=_to_float(entry.get("origQty")),
                executed_quantity=_to_float(entry.get("executedQty")),
                time_in_force=(str(entry.get("timeInForce")) if entry.get("timeInForce") else None),
                client_order_id=(
                    str(entry.get("clientOrderId")) if entry.get("clientOrderId") not in (None, "") else None
                ),
                stop_price=stop_price,
                reduce_only=_to_bool(entry.get("reduceOnly")),
                close_position=_to_bool(entry.get("closePosition")),
                working_type=(str(entry.get("workingType")) if entry.get("workingType") else None),
                price_protect=_to_bool(entry.get("priceProtect")),
                position_side=(str(entry.get("positionSide")) if entry.get("positionSide") else None),
                update_time=timestamp,
            )
            orders.append(order)

        orders.sort(key=lambda item: item.update_time)
        return orders

    def fetch_funding_rates(
        self,
        *,
        symbol: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> Sequence[FundingRateEvent]:
        """Pobiera zdarzenia finansowania i aktualizuje metryki telemetryczne."""

        params: dict[str, object] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        if limit is not None:
            params["limit"] = int(limit)

        payload = self._public_request("/fapi/v1/fundingRate", params=params or None)
        if not isinstance(payload, list):
            raise RuntimeError("Odpowiedź fundingRate z Binance Futures ma niepoprawny format")

        events: list[FundingRateEvent] = []
        for entry in payload:
            if not isinstance(entry, Mapping):
                continue
            event_symbol = entry.get("symbol")
            funding_rate = entry.get("fundingRate")
            funding_time = entry.get("fundingTime")
            if not isinstance(event_symbol, str):
                continue
            rate_value = _to_float(funding_rate, default=0.0)
            funding_time_int = _to_int(funding_time, default=0)
            mark_price = entry.get("markPrice")
            next_funding_time = entry.get("nextFundingTime")
            interest_rate = entry.get("interestRate")

            event = FundingRateEvent(
                symbol=event_symbol,
                funding_rate=rate_value,
                funding_time=funding_time_int,
                mark_price=_to_float(mark_price) if mark_price is not None else None,
                next_funding_time=_to_int(next_funding_time, default=None),
                interest_rate=_to_float(interest_rate) if interest_rate is not None else None,
            )
            events.append(event)

            metric_labels = {
                "exchange": self.name,
                "environment": self._environment.value,
                "symbol": event_symbol,
            }
            self._metric_funding_rate.set(rate_value, labels=metric_labels)

        return events

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

    def fetch_positions(self) -> list[FuturesPosition]:
        """Pobiera aktywne pozycje i aktualizuje metryki hedgingowe."""

        payload = self._signed_request("/fapi/v2/positionRisk")
        if not isinstance(payload, list):
            raise RuntimeError("Odpowiedź positionRisk z Binance Futures ma niepoprawny format")

        positions: list[FuturesPosition] = []
        current_labels: set[tuple[str, str]] = set()
        long_notional = 0.0
        short_notional = 0.0

        for entry in payload:
            if not isinstance(entry, Mapping):
                continue
            symbol = str(entry.get("symbol", "")).strip()
            if not symbol:
                continue
            quantity = _to_float(entry.get("positionAmt"), 0.0)
            if quantity == 0.0:
                continue
            entry_price = _to_float(entry.get("entryPrice"), 0.0)
            mark_price = _to_float(entry.get("markPrice"), 0.0) or entry_price
            notional = abs(quantity * mark_price)
            side = "long" if quantity > 0 else "short"
            unrealized = _to_float(entry.get("unRealizedProfit"), 0.0)
            leverage = _to_float(entry.get("leverage"), 0.0)
            isolated_field = str(entry.get("isolated", entry.get("marginType", ""))).lower()
            isolated = isolated_field in {"true", "1", "isolated"}
            liquidation_price = _to_float(entry.get("liquidationPrice"), 0.0)
            if liquidation_price <= 0.0:
                liquidation_price = None

            position = FuturesPosition(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                mark_price=mark_price,
                notional=notional,
                unrealized_pnl=unrealized,
                leverage=leverage,
                isolated=isolated,
                liquidation_price=liquidation_price,
            )
            positions.append(position)

            metric_labels = {
                "exchange": self.name,
                "environment": self._environment.value,
                "symbol": symbol,
                "side": side,
            }
            self._metric_position_notional.set(notional, labels=metric_labels)
            current_labels.add((symbol, side))
            if side == "long":
                long_notional += notional
            else:
                short_notional += notional

        # Zerujemy metryki pozycji, które zostały zamknięte od ostatniego odczytu.
        for symbol, side in self._tracked_position_labels - current_labels:
            metric_labels = {
                "exchange": self.name,
                "environment": self._environment.value,
                "symbol": symbol,
                "side": side,
            }
            self._metric_position_notional.set(0.0, labels=metric_labels)
        self._tracked_position_labels = current_labels

        base_labels = dict(self._metric_base_labels)
        self._metric_position_active.set(float(len(positions)), labels=base_labels)
        self._metric_position_long.set(long_notional, labels=base_labels)
        self._metric_position_short.set(short_notional, labels=base_labels)
        gross_notional = long_notional + short_notional
        net_notional = long_notional - short_notional
        self._metric_position_gross.set(gross_notional, labels=base_labels)
        self._metric_position_net.set(net_notional, labels=base_labels)

        return positions

    def build_hedging_report(self) -> Mapping[str, object]:
        """Generuje raport ekspozycji dla modułu hedgingowego i audytu ryzyka."""

        positions = self.fetch_positions()
        long_notional = sum(position.notional for position in positions if position.side == "long")
        short_notional = sum(position.notional for position in positions if position.side == "short")
        gross_notional = long_notional + short_notional
        net_notional = long_notional - short_notional

        return {
            "timestamp": int(time.time() * 1000),
            "exchange": self.name,
            "environment": self._environment.value,
            "valuation_asset": "USDT",
            "summary": {
                "gross_notional": gross_notional,
                "long_notional": long_notional,
                "short_notional": short_notional,
                "net_notional": net_notional,
                "open_positions": len(positions),
            },
            "positions": [position.to_mapping() for position in positions],
        }

    # Zarządzanie kluczem nasłuchu (listenKey) jest wymagane przed uruchomieniem strumieni prywatnych.
    # Chociaż WebSockety są zabronione w aktualnej architekturze, utrzymujemy logikę REST,
    # aby daemon gRPC mógł w przyszłości tworzyć własny mostek IPC.

    def create_listen_key(self) -> str:
        if "trade" not in self._permission_set and "read" not in self._permission_set:
            raise PermissionError("Poświadczenia nie zezwalają na operacje na listenKey Binance Futures.")

        payload = self._signed_request("/fapi/v1/listenKey", method="POST")
        if not isinstance(payload, Mapping) or "listenKey" not in payload:
            raise RuntimeError("Odpowiedź utworzenia listenKey Binance Futures ma niepoprawny format")

        listen_key = payload.get("listenKey")
        if not isinstance(listen_key, str) or not listen_key:
            raise RuntimeError("API Binance Futures zwróciło pusty listenKey")
        return listen_key

    def keepalive_listen_key(self, listen_key: str) -> None:
        if not listen_key:
            raise ValueError("listenKey nie może być pusty")
        if "trade" not in self._permission_set and "read" not in self._permission_set:
            raise PermissionError("Poświadczenia nie zezwalają na operacje na listenKey Binance Futures.")

        params = {"listenKey": listen_key}
        response = self._signed_request("/fapi/v1/listenKey", method="PUT", params=params)
        if isinstance(response, Mapping) and response.get("code") in (None, 0):
            return
        if response not in ({}, None):
            raise RuntimeError(f"Podtrzymanie listenKey Binance Futures zwróciło nieoczekiwane dane: {response}")

    def close_listen_key(self, listen_key: str) -> None:
        if not listen_key:
            raise ValueError("listenKey nie może być pusty")
        if "trade" not in self._permission_set and "read" not in self._permission_set:
            raise PermissionError("Poświadczenia nie zezwalają na operacje na listenKey Binance Futures.")

        params = {"listenKey": listen_key}
        response = self._signed_request("/fapi/v1/listenKey", method="DELETE", params=params)
        if isinstance(response, Mapping):
            code = response.get("code")
            if code in (None, 0):
                return
        if response in ({}, None):
            return
        raise RuntimeError(f"Zamknięcie listenKey Binance Futures zwróciło nieoczekiwane dane: {response}")

    def stream_public_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        return self._build_stream("public", channels)

    def stream_private_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        if "trade" not in self._permission_set and "read" not in self._permission_set:
            raise PermissionError("Poświadczenia nie pozwalają na strumień prywatny Binance Futures.")
        return self._build_stream("private", channels)


__all__ = [
    "BinanceFuturesAdapter",
    "BinanceFuturesTicker",
    "BinanceFuturesOrderBookLevel",
    "BinanceFuturesOrderBook",
    "BinanceFuturesOpenOrder",
    "FuturesPosition",
    "FundingRateEvent",
]


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"true", "1", "yes", "y"}
    return False


@dataclass(slots=True)
class BinanceFuturesTicker:
    """Znormalizowany ticker 24h z rynku futures Binance USD-M."""

    symbol: str
    best_bid: float
    best_ask: float
    last_price: float
    price_change_percent: float
    open_price: float
    high_24h: float
    low_24h: float
    volume_24h_base: float
    volume_24h_quote: float
    open_interest: float
    timestamp: float


@dataclass(slots=True)
class BinanceFuturesOrderBookLevel:
    price: float
    quantity: float


@dataclass(slots=True)
class BinanceFuturesOrderBook:
    symbol: str
    bids: tuple[BinanceFuturesOrderBookLevel, ...]
    asks: tuple[BinanceFuturesOrderBookLevel, ...]
    depth: int
    last_update_id: int
    timestamp: float


@dataclass(slots=True)
class BinanceFuturesOpenOrder:
    order_id: str
    symbol: str
    status: str
    side: str
    order_type: str
    price: float | None
    orig_quantity: float
    executed_quantity: float
    time_in_force: str | None
    client_order_id: str | None
    stop_price: float | None
    reduce_only: bool
    close_position: bool
    working_type: str | None
    price_protect: bool
    position_side: str | None
    update_time: float
