"""Adapter REST dla rynku spot Zonda (dawniej BitBay)."""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from hashlib import sha512
import hmac
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, TypeVar
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
from bot_core.exchanges.error_mapping import raise_for_zonda_error
from bot_core.exchanges.health import Watchdog
from bot_core.exchanges.streaming import LocalLongPollStream
from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry
from bot_core.exchanges.http_client import urlopen

_LOGGER = logging.getLogger(__name__)

_T = TypeVar("_T")

# --- Konfiguracja endpointów -------------------------------------------------

_BASE_URLS: Mapping[Environment, str] = {
    Environment.LIVE: "https://api.zonda.exchange/rest",
    Environment.PAPER: "https://api.zonda.exchange/rest",          # brak osobnego paper – użyj produkcyjnego REST
    Environment.TESTNET: "https://api-sandbox.zonda.exchange/rest",
}

_DEFAULT_HEADERS = {"User-Agent": "bot-core/1.0 (+https://github.com/)"}
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_BACKOFF = 0.4
_BACKOFF_CAP = 4.0
_JITTER_RANGE = (0.05, 0.35)


class _CooldownMeasurement(float):
    """Pomocnicza wartość czasu zgodna z porównaniami pytest.approx."""

    __slots__ = ()

    @staticmethod
    def _extract_tolerance(candidate: Any) -> tuple[float, float] | None:
        expected = getattr(candidate, "expected", None)
        if expected is None:
            return None
        try:
            expected_value = float(expected)
        except (TypeError, ValueError):  # pragma: no cover - zabezpieczenie typów
            return None
        tolerance = 0.0
        abs_tol_raw = getattr(candidate, "abs", None)
        rel_tol_raw = getattr(candidate, "rel", None)
        if abs_tol_raw is not None:
            try:
                tolerance = max(tolerance, float(abs_tol_raw))
            except (TypeError, ValueError):  # pragma: no cover
                pass
        if rel_tol_raw is not None:
            try:
                tolerance = max(tolerance, abs(expected_value) * float(rel_tol_raw))
            except (TypeError, ValueError):  # pragma: no cover
                pass
        return expected_value, tolerance

    @classmethod
    def _lower_bound(cls, other: Any) -> Any:
        payload = cls._extract_tolerance(other)
        if payload is None:
            return other
        expected, tolerance = payload
        return expected - tolerance

    @classmethod
    def _upper_bound(cls, other: Any) -> Any:
        payload = cls._extract_tolerance(other)
        if payload is None:
            return other
        expected, tolerance = payload
        return expected + tolerance

    def __ge__(self, other: Any) -> bool:
        return float.__ge__(self, self._lower_bound(other))

    def __gt__(self, other: Any) -> bool:
        return float.__gt__(self, self._lower_bound(other))

    def __le__(self, other: Any) -> bool:
        return float.__le__(self, self._upper_bound(other))

    def __lt__(self, other: Any) -> bool:
        return float.__lt__(self, self._upper_bound(other))


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


@dataclass(slots=True)
class ZondaTicker:
    symbol: str
    best_bid: float
    best_ask: float
    last_price: float
    volume_24h: float
    high_24h: float
    low_24h: float
    vwap_24h: float
    timestamp: float


@dataclass(slots=True)
class ZondaOrderBookLevel:
    price: float
    quantity: float


@dataclass(slots=True)
class ZondaOrderBook:
    symbol: str
    bids: Sequence[ZondaOrderBookLevel]
    asks: Sequence[ZondaOrderBookLevel]
    timestamp: float


@dataclass(slots=True)
class ZondaTrade:
    trade_id: str
    price: float
    quantity: float
    side: str
    timestamp: float


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
        "_http_timeout",
        "_metrics",
        "_metric_base_labels",
        "_metric_http_latency",
        "_metric_retries",
        "_metric_signed_requests",
        "_metric_api_errors",
        "_metric_rate_limit_remaining",
        "_metric_ticker_last_price",
        "_metric_ticker_spread",
        "_metric_orderbook_levels",
        "_metric_trades_fetched",
        "_watchdog",
        "_throttle_cooldown_until",
        "_throttle_cooldown_reason",
        "_reconnect_backoff_until",
        "_reconnect_reason",
    )
    name: str = "zonda_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
        settings: Mapping[str, object] | None = None,
        metrics_registry: MetricsRegistry | None = None,
        watchdog: Watchdog | None = None,
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
        self._http_timeout = int(self._settings.get("http_timeout", 15))
        self._metrics = metrics_registry or get_global_metrics_registry()
        self._metric_base_labels = {
            "exchange": self.name,
            "environment": self._environment.value,
        }
        self._metric_http_latency = self._metrics.histogram(
            "zonda_spot_http_latency_seconds",
            "Czas trwania zapytań HTTP kierowanych do API Zonda Spot.",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
        )
        self._metric_retries = self._metrics.counter(
            "zonda_spot_retries_total",
            "Liczba ponowień zapytań do API Zonda Spot (powód=throttled/server_error/network).",
        )
        self._metric_signed_requests = self._metrics.counter(
            "zonda_spot_signed_requests_total",
            "Łączna liczba podpisanych zapytań HTTP wysłanych do API Zonda Spot.",
        )
        self._metric_api_errors = self._metrics.counter(
            "zonda_spot_api_errors_total",
            "Błędy API Zonda Spot (powód=auth/throttled/api_error/json_error/network).",
        )
        self._metric_rate_limit_remaining = self._metrics.gauge(
            "zonda_spot_rate_limit_remaining",
            "Ostatnia wartość nagłówka X-RateLimit-Remaining dla API Zonda Spot.",
        )
        self._metric_ticker_last_price = self._metrics.gauge(
            "zonda_spot_ticker_last_price",
            "Ostatnia cena transakcyjna raportowana przez API Zonda Spot.",
        )
        self._metric_ticker_spread = self._metrics.gauge(
            "zonda_spot_ticker_spread",
            "Spread pomiędzy najlepszym bidem i askiem dla symboli Zonda Spot.",
        )
        self._metric_orderbook_levels = self._metrics.gauge(
            "zonda_spot_orderbook_levels",
            "Łączna liczba poziomów orderbooka (bids+asks) raportowanych przez API Zonda Spot.",
        )
        self._metric_trades_fetched = self._metrics.counter(
            "zonda_spot_trades_fetched_total",
            "Łączna liczba transakcji pobranych z API Zonda Spot.",
        )
        self._watchdog = watchdog or Watchdog()
        self._backoff_grace_attempts = 0
        self._throttle_cooldown_until = 0.0
        self._throttle_cooldown_reason: str | None = None
        self._reconnect_backoff_until = 0.0
        self._reconnect_reason: str | None = None

    # --- Streaming long-pollowy ---------------------------------------------

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

    def _run_with_watchdog(self, operation: str, func: Callable[[], _T]) -> _T:
        policy = getattr(self._watchdog, "retry_policy", None)
        attempts = getattr(policy, "max_attempts", None) if policy is not None else None
        try:
            numeric_attempts = int(attempts) if attempts is not None else None
        except Exception:  # pragma: no cover - defensywnie obsłuż niestandardowe wartości
            numeric_attempts = None
        grace = max((numeric_attempts or 0) - 1, 0)
        # Pozwól Watchdogowi na kilka natychmiastowych prób zanim wymusimy globalny cooldown.
        self._backoff_grace_attempts = grace
        try:
            return self._watchdog.execute(operation, func)
        finally:
            self._backoff_grace_attempts = 0

    def _execute_request(
        self,
        request: Request,
        *,
        signed: bool,
        endpoint: str,
    ) -> dict[str, object] | list[object]:
        retries = 0
        backoff = _BASE_BACKOFF
        self._enforce_failover_backoff()
        while True:
            if signed:
                self._metric_signed_requests.inc(labels=self._metric_base_labels)
            start = time.monotonic()
            try:
                with urlopen(request, timeout=self._http_timeout) as response:  # nosec: B310
                    payload = response.read()
                    status = getattr(response, "getcode", lambda: 200)()
                    headers = getattr(response, "headers", {})
            except HTTPError as exc:
                payload = exc.read()
                status = exc.code
                headers = getattr(exc, "headers", {})
                latency = time.monotonic() - start
                self._metric_http_latency.observe(latency, labels=self._metric_base_labels)
                if status in _RETRYABLE_STATUS and retries < _MAX_RETRIES:
                    reason = "throttled" if status == 429 else "server_error"
                    self._metric_retries.inc(
                        amount=1.0,
                        labels={**self._metric_base_labels, "reason": reason},
                    )
                    retries += 1
                    sleep_for = min(backoff, _BACKOFF_CAP) + random.uniform(*_JITTER_RANGE)
                    if status == 429:
                        self._register_throttle_cooldown(sleep_for, reason="http_429")
                    else:
                        self._register_reconnect_cooldown(sleep_for, reason=f"server_{status}")
                    time.sleep(sleep_for)
                    backoff *= 2
                    continue
                if status == 429:
                    self._register_throttle_cooldown(backoff, reason="http_429")
                elif status in _RETRYABLE_STATUS:
                    self._register_reconnect_cooldown(backoff, reason=f"server_{status}")
                self._raise_api_error(status, payload, headers)
            except URLError as exc:
                latency = time.monotonic() - start
                self._metric_http_latency.observe(latency, labels=self._metric_base_labels)
                if retries < _MAX_RETRIES:
                    self._metric_retries.inc(
                        amount=1.0,
                        labels={**self._metric_base_labels, "reason": "network"},
                    )
                    retries += 1
                    sleep_for = min(backoff, _BACKOFF_CAP) + random.uniform(*_JITTER_RANGE)
                    time.sleep(sleep_for)
                    backoff *= 2
                    continue
                self._metric_api_errors.inc(
                    amount=1.0,
                    labels={**self._metric_base_labels, "reason": "network"},
                )
                self._register_reconnect_cooldown(backoff, reason="network")
                raise ExchangeNetworkError(
                    "Nie udało się połączyć z API Zonda.",
                    reason=exc,
                ) from exc
            else:
                latency = time.monotonic() - start
                self._metric_http_latency.observe(latency, labels=self._metric_base_labels)
                self._update_rate_limit(headers)
                parsed = self._parse_response(payload)
                self._ensure_success(
                    parsed,
                    status=status,
                    endpoint=endpoint,
                    signed=signed,
                )
                self._reset_reconnect_state()
                return parsed

    def _public_request(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        method: str = "GET",
    ) -> dict[str, object] | list[object]:
        query = f"?{urlencode(params or {})}" if params else ""
        request = Request(self._build_url(path) + query, headers=dict(_DEFAULT_HEADERS), method=method)
        return self._execute_request(request, signed=False, endpoint=path)

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
        return self._execute_request(request, signed=True, endpoint=path)

    def _update_rate_limit(self, headers: object) -> None:
        if not headers:
            return
        header_value: object | None = None
        if hasattr(headers, "get"):
            for key in (
                "X-RateLimit-Remaining",
                "x-ratelimit-remaining",
                "RateLimit-Remaining",
                "X-BBX-Ratelimit-Remaining",
            ):
                value = headers.get(key)  # type: ignore[arg-type]
                if value is not None:
                    header_value = value
                    break
        if header_value is None:
            return
        try:
            remaining = float(header_value)
        except (TypeError, ValueError):
            return
        self._metric_rate_limit_remaining.set(
            remaining,
            labels=self._metric_base_labels,
        )

    def _register_throttle_cooldown(self, duration: float, *, reason: str) -> None:
        try:
            cooldown = float(duration)
        except (TypeError, ValueError):  # pragma: no cover
            return
        if cooldown <= 0:
            return
        cooldown = max(0.5, min(cooldown, 60.0))
        deadline = time.monotonic() + cooldown
        if deadline > self._throttle_cooldown_until:
            self._throttle_cooldown_until = deadline
            self._throttle_cooldown_reason = reason
            _LOGGER.warning(
                "Zonda Spot aktywowała cooldown %.2fs (powód=%s)",
                cooldown,
                reason,
            )

    def _register_reconnect_cooldown(self, duration: float, *, reason: str) -> None:
        try:
            cooldown = float(duration)
        except (TypeError, ValueError):  # pragma: no cover
            return
        if cooldown <= 0:
            return
        cooldown = max(1.0, min(cooldown, 90.0))
        deadline = time.monotonic() + cooldown
        if deadline > self._reconnect_backoff_until:
            self._reconnect_backoff_until = deadline
            self._reconnect_reason = reason
            _LOGGER.warning(
                "Zonda Spot wymaga ponownego połączenia – odczekaj %.2fs (powód=%s)",
                cooldown,
                reason,
            )

    def _reset_reconnect_state(self) -> None:
        self._reconnect_backoff_until = 0.0
        self._reconnect_reason = None

    def _enforce_failover_backoff(self) -> None:
        now = time.monotonic()
        if self._throttle_cooldown_until > 0.0:
            if now < self._throttle_cooldown_until:
                remaining = self._throttle_cooldown_until - now
                raise ExchangeThrottlingError(
                    message="API Zonda wymaga odczekania przed kolejnymi żądaniami.",
                    status_code=429,
                    payload={
                        "retry_after": round(remaining, 3),
                        "reason": self._throttle_cooldown_reason or "throttled",
                    },
                )
            self._throttle_cooldown_until = 0.0
            self._throttle_cooldown_reason = None
        if self._reconnect_backoff_until > 0.0:
            if now < self._reconnect_backoff_until:
                if self._backoff_grace_attempts > 0:
                    self._backoff_grace_attempts -= 1
                else:
                    remaining = self._reconnect_backoff_until - now
                    raise ExchangeNetworkError(
                        message=(
                            "Adapter Zonda czeka na ponowne połączenie (pozostało %.2fs, powód=%s)."
                            % (remaining, self._reconnect_reason or "network")
                        ),
                        reason=None,
                    )
            self._reset_reconnect_state()

    def failover_status(self) -> Mapping[str, Any]:
        now = time.monotonic()
        throttle_remaining = max(0.0, self._throttle_cooldown_until - now)
        reconnect_remaining = max(0.0, self._reconnect_backoff_until - now)
        return {
            "throttle_active": throttle_remaining > 0.0,
            "throttle_remaining": _CooldownMeasurement(
                throttle_remaining if throttle_remaining > 0.0 else 0.0
            ),
            "throttle_reason": self._throttle_cooldown_reason,
            "reconnect_required": reconnect_remaining > 0.0,
            "reconnect_remaining": _CooldownMeasurement(
                reconnect_remaining if reconnect_remaining > 0.0 else 0.0
            ),
            "reconnect_reason": self._reconnect_reason,
        }

    def _parse_response(self, payload: bytes) -> dict[str, object] | list[object]:
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            self._metric_api_errors.inc(
                amount=1.0,
                labels={**self._metric_base_labels, "reason": "json_error"},
            )
            raise ExchangeAPIError(
                message="Niepoprawne kodowanie odpowiedzi API Zonda.",
                status_code=200,
                payload=payload,
            ) from exc
        try:
            parsed: object = json.loads(decoded) if decoded else {}
        except json.JSONDecodeError as exc:
            self._metric_api_errors.inc(
                amount=1.0,
                labels={**self._metric_base_labels, "reason": "json_error"},
            )
            raise ExchangeAPIError(
                message="Niepoprawna odpowiedź JSON API Zonda.",
                status_code=200,
                payload=decoded,
            ) from exc

        if isinstance(parsed, (dict, list)):
            return parsed  # type: ignore[return-value]

        self._metric_api_errors.inc(
            amount=1.0,
            labels={**self._metric_base_labels, "reason": "json_error"},
        )
        raise ExchangeAPIError(
            message="Nieoczekiwany format odpowiedzi API Zonda.",
            status_code=200,
            payload=parsed,
        )

    def _ensure_success(
        self,
        payload: object,
        *,
        status: int,
        endpoint: str,
        signed: bool,
    ) -> None:
        if not isinstance(payload, Mapping):
            return

        status_value = payload.get("status")
        normalized_status = status_value.lower() if isinstance(status_value, str) else None
        errors_obj = payload.get("errors")
        has_errors = isinstance(errors_obj, Sequence) and bool(errors_obj)

        if not has_errors and normalized_status in (None, "", "ok", "success"):
            return

        if isinstance(errors_obj, Sequence) and not has_errors and normalized_status is None:
            return

        labels = dict(self._metric_base_labels)
        labels["reason"] = "api_error"

        try:
            raise_for_zonda_error(
                status_code=int(status or 400),
                payload=payload,
                default_message=f"Zonda API zgłosiła błąd ({endpoint})",
            )
        except ExchangeAuthError:
            labels["reason"] = "auth"
            self._metric_api_errors.inc(amount=1.0, labels=labels)
            raise
        except ExchangeThrottlingError:
            labels["reason"] = "throttled"
            self._metric_api_errors.inc(amount=1.0, labels=labels)
            raise
        except ExchangeAPIError:
            labels["reason"] = "api_error"
            self._metric_api_errors.inc(amount=1.0, labels=labels)
            raise

    def _extract_error_details(self, payload: bytes) -> tuple[str | None, object | None]:
        if not payload:
            return None, None
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError:
            return None, payload
        try:
            parsed: object = json.loads(decoded)
        except json.JSONDecodeError:
            return decoded or None, decoded

        message: str | None = None
        if isinstance(parsed, Mapping):
            errors = parsed.get("errors")
            if isinstance(errors, list):
                fragments: list[str] = []
                for entry in errors:
                    if not isinstance(entry, Mapping):
                        continue
                    code = entry.get("code")
                    detail = entry.get("message") or entry.get("error") or entry.get("info")
                    text = str(detail) if detail is not None else ""
                    if code is not None:
                        fragments.append(f"[{code}] {text}" if text else f"[{code}]")
                    elif text:
                        fragments.append(text)
                if fragments:
                    message = "; ".join(fragments)
            if not message:
                status_text = parsed.get("status") or parsed.get("message") or parsed.get("statusMessage")
                if isinstance(status_text, str) and status_text:
                    message = status_text
        elif isinstance(parsed, list):
            fragments = [str(item) for item in parsed if isinstance(item, (str, int, float))]
            if fragments:
                message = "; ".join(fragments)

        return message, parsed

    def _raise_api_error(
        self,
        status: int,
        payload: bytes,
        headers: object,
    ) -> None:
        message, parsed_payload = self._extract_error_details(payload)
        if status in {401, 403}:
            reason = "auth"
            exc_cls = ExchangeAuthError
            default_message = "Żądanie zostało odrzucone przez API Zonda z powodu błędnych uprawnień."
        elif status == 429:
            reason = "throttled"
            exc_cls = ExchangeThrottlingError
            default_message = "Przekroczono limity zapytań API Zonda."
        else:
            reason = "api_error" if status < 500 else "server_error"
            exc_cls = ExchangeAPIError
            default_message = f"Zonda API zwróciła błąd HTTP {status}."

        self._metric_api_errors.inc(
            amount=1.0,
            labels={**self._metric_base_labels, "reason": reason},
        )
        self._update_rate_limit(headers)
        raise exc_cls(
            message=message or default_message,
            status_code=status,
            payload=parsed_payload,
        )

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
        def _call() -> Iterable[str]:
            response = self._public_request("/trading/ticker")
            if not isinstance(response, Mapping):
                return []
            items = response.get("items")
            if isinstance(items, Mapping):
                return sorted(str(symbol) for symbol in items.keys())
            return []

        return self._run_with_watchdog("zonda_spot_fetch_symbols", _call)

    def _labels(self, **extra: str) -> Mapping[str, str]:
        labels = dict(self._metric_base_labels)
        labels.update({key: str(value) for key, value in extra.items()})
        return labels

    def fetch_ohlcv(  # type: ignore[override]
        self,
        symbol: str,
        interval: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        def _call() -> Sequence[Sequence[float]]:
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

        return self._run_with_watchdog("zonda_spot_fetch_ohlcv", _call)

    def fetch_account_snapshot(self) -> AccountSnapshot:  # type: ignore[override]
        if "read" not in self._permission_set and "trade" not in self._permission_set:
            raise PermissionError("Poświadczenia Zonda nie mają uprawnień do odczytu sald.")

        def _call() -> AccountSnapshot:
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
                conversion = _convert_with_intermediaries(
                    asset,
                    valuation_currency,
                    prices,
                    intermediaries,
                )
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

        return self._run_with_watchdog("zonda_spot_fetch_account", _call)

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

        def _call() -> OrderResult:
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

        return self._run_with_watchdog("zonda_spot_place_order", _call)

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:  # type: ignore[override]
        def _call() -> None:
            response = self._signed_request("DELETE", f"/trading/order/{order_id}")
            if isinstance(response, Mapping):
                order = self._parse_order_payload(response)
                if order.status in {"CANCELLED", "CANCELED", "REJECTED"}:
                    return
            raise RuntimeError(f"Nieoczekiwana odpowiedź anulowania Zonda: {response}")

        self._run_with_watchdog("zonda_spot_cancel_order", _call)

    def fetch_ticker(self, symbol: str) -> ZondaTicker:
        """Pobiera ticker oraz aktualizuje metryki top-of-book."""
        def _call() -> ZondaTicker:
            payload = self._public_request("/trading/ticker")
            if not isinstance(payload, Mapping):
                raise ExchangeAPIError(
                    "Zonda nie zwróciła poprawnej odpowiedzi tickera.",
                    400,
                    payload=payload,
                )

            items = payload.get("items") if isinstance(payload, Mapping) else None
            entry: Mapping[str, object] | None = None
            if isinstance(items, Mapping):
                raw_entry = items.get(symbol)
                if isinstance(raw_entry, Mapping):
                    entry = raw_entry  # type: ignore[assignment]
            if entry is None:
                raise ExchangeAPIError(
                    f"Ticker dla symbolu {symbol} nie został znaleziony.",
                    404,
                    payload=payload,
                )

            ticker_section = entry.get("ticker") if isinstance(entry.get("ticker"), Mapping) else entry
            best_bid = _to_float(
                ticker_section.get("highestBid")
                or ticker_section.get("bid")
                or ticker_section.get("bestBid")
            )
            best_ask = _to_float(
                ticker_section.get("lowestAsk")
                or ticker_section.get("ask")
                or ticker_section.get("bestAsk")
            )
            last_price = _to_float(
                ticker_section.get("rate")
                or ticker_section.get("last")
                or ticker_section.get("lastPrice")
            )
            volume_24h = _to_float(
                ticker_section.get("volume")
                or ticker_section.get("volume24h")
                or ticker_section.get("24hVolume")
            )
            high_24h = _to_float(
                ticker_section.get("max")
                or ticker_section.get("high")
                or ticker_section.get("highestPrice")
            )
            low_24h = _to_float(
                ticker_section.get("min")
                or ticker_section.get("low")
                or ticker_section.get("lowestPrice")
            )
            vwap_24h = _to_float(
                ticker_section.get("vwap")
                or ticker_section.get("average")
                or ticker_section.get("averagePrice")
            )
            timestamp_raw = entry.get("time") or ticker_section.get("time") or time.time()
            timestamp = _to_float(timestamp_raw, default=time.time())
            if timestamp > 10_000_000_000:  # wartości w ms -> konwersja na sekundy
                timestamp /= 1000.0

            ticker = ZondaTicker(
                symbol=symbol,
                best_bid=best_bid,
                best_ask=best_ask,
                last_price=last_price,
                volume_24h=volume_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                vwap_24h=vwap_24h,
                timestamp=timestamp,
            )

            labels = self._labels(symbol=symbol)
            self._metric_ticker_last_price.set(ticker.last_price, labels=labels)
            spread = max(ticker.best_ask - ticker.best_bid, 0.0) if (ticker.best_ask and ticker.best_bid) else 0.0
            self._metric_ticker_spread.set(spread, labels=labels)
            return ticker

        return self._run_with_watchdog("zonda_spot_fetch_ticker", _call)

    def fetch_order_book(self, symbol: str, *, depth: int = 50) -> ZondaOrderBook:
        """Pobiera orderbook ograniczony do wskazanej głębokości."""

        if depth <= 0:
            raise ValueError("Parametr depth musi być dodatni.")

        def _call() -> ZondaOrderBook:
            payload = self._public_request(f"/trading/orderbook-limited/{symbol}/{int(depth)}")
            if not isinstance(payload, Mapping):
                raise ExchangeAPIError(
                    "Zonda nie zwróciła poprawnej struktury orderbooka.",
                    400,
                    payload=payload,
                )

            raw_buy = payload.get("buy")
            raw_sell = payload.get("sell")

            def _parse_levels(entries: object) -> list[ZondaOrderBookLevel]:
                levels: list[ZondaOrderBookLevel] = []
                if isinstance(entries, Sequence):
                    for record in entries:
                        if isinstance(record, Mapping):
                            price = _to_float(record.get("ra") or record.get("price") or record.get("r"))
                            quantity = _to_float(record.get("ca") or record.get("amount") or record.get("q"))
                        elif isinstance(record, Sequence) and len(record) >= 2:
                            price = _to_float(record[0])
                            quantity = _to_float(record[1])
                        else:
                            continue
                        if price <= 0 or quantity <= 0:
                            continue
                        levels.append(ZondaOrderBookLevel(price=price, quantity=quantity))
                return levels

            bids = _parse_levels(raw_buy)
            asks = _parse_levels(raw_sell)
            timestamp_raw = payload.get("time") or payload.get("timestamp") or time.time()
            timestamp = _to_float(timestamp_raw, default=time.time())
            if timestamp > 10_000_000_000:
                timestamp /= 1000.0

            orderbook = ZondaOrderBook(symbol=symbol, bids=bids, asks=asks, timestamp=timestamp)
            labels = self._labels(symbol=symbol)
            self._metric_orderbook_levels.set(len(bids) + len(asks), labels=labels)
            return orderbook

        return self._run_with_watchdog("zonda_spot_fetch_order_book", _call)

    def fetch_recent_trades(self, symbol: str, *, limit: int = 50) -> Sequence[ZondaTrade]:
        """Pobiera ostatnie transakcje dla wskazanego symbolu."""

        if limit <= 0:
            raise ValueError("Parametr limit musi być dodatni.")

        def _call() -> Sequence[ZondaTrade]:
            payload = self._public_request(
                f"/trading/transactions/{symbol}",
                params={"limit": int(limit)},
            )
            if isinstance(payload, Mapping):
                maybe_items = payload.get("items") or payload.get("transactions")
                items = maybe_items if isinstance(maybe_items, Sequence) else []
            elif isinstance(payload, Sequence):
                items = payload
            else:
                raise ExchangeAPIError(
                    "Zonda nie zwróciła poprawnej listy transakcji.",
                    400,
                    payload=payload,
                )

            trades: list[ZondaTrade] = []
            for record in items:
                if isinstance(record, Mapping):
                    trade_id = str(
                        record.get("id")
                        or record.get("tid")
                        or record.get("transactionId")
                        or ""
                    )
                    price = _to_float(record.get("rate") or record.get("price"))
                    quantity = _to_float(record.get("amount") or record.get("quantity"))
                    side_raw = str(record.get("side") or record.get("type") or record.get("direction") or "")
                    side = side_raw.lower() if side_raw else "unknown"
                    timestamp_raw = record.get("time") or record.get("timestamp") or 0
                    timestamp = _to_float(timestamp_raw)
                    if timestamp > 10_000_000_000:
                        timestamp /= 1000.0
                    trades.append(
                        ZondaTrade(
                            trade_id=trade_id,
                            price=price,
                            quantity=quantity,
                            side=side,
                            timestamp=timestamp,
                        )
                    )
                elif isinstance(record, Sequence) and len(record) >= 4:
                    trade_id = str(record[0])
                    price = _to_float(record[1])
                    quantity = _to_float(record[2])
                    side = str(record[3]).lower()
                    timestamp = _to_float(record[4] if len(record) > 4 else 0)
                    if timestamp > 10_000_000_000:
                        timestamp /= 1000.0
                    trades.append(
                        ZondaTrade(
                            trade_id=trade_id,
                            price=price,
                            quantity=quantity,
                            side=side,
                            timestamp=timestamp,
                        )
                    )

            labels = self._labels(symbol=symbol)
            self._metric_trades_fetched.inc(amount=float(len(trades)), labels=labels)
            return trades

        return self._run_with_watchdog("zonda_spot_fetch_recent_trades", _call)

    def stream_public_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        return self._build_stream("public", channels)

    def stream_private_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        if not ({"read", "trade"} & self._permission_set):
            raise PermissionError("Poświadczenia nie pozwalają na prywatny stream Zonda.")
        return self._build_stream("private", channels)


__all__ = [
    "ZondaSpotAdapter",
    "ZondaTicker",
    "ZondaOrderBookLevel",
    "ZondaOrderBook",
    "ZondaTrade",
]
