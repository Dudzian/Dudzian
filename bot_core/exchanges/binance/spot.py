"""Adapter REST dla rynku spot Binance do obsługi danych publicznych i prywatnych."""
from __future__ import annotations

from dataclasses import dataclass
import hmac
import json
import logging
import random
import time
from hashlib import sha256
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request

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
from bot_core.exchanges.binance.symbols import (
    filter_supported_exchange_symbols,
    normalize_symbol,
    to_exchange_symbol,
)
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.error_mapping import raise_for_binance_error
from bot_core.exchanges.health import Watchdog
from bot_core.exchanges.rate_limiter import (
    RateLimitRule,
    get_global_rate_limiter_registry,
    normalize_rate_limit_rules,
)
from bot_core.exchanges.streaming import LocalLongPollStream
from bot_core.exchanges.http_client import urlopen
from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry

_LOGGER = logging.getLogger(__name__)

_DEFAULT_HEADERS = {"User-Agent": "bot-core/1.0 (+https://github.com/)"}
_RETRYABLE_STATUS = {418, 429}
_MAX_RETRIES = 3
_BASE_BACKOFF = 0.4
_BACKOFF_CAP = 4.0
_JITTER_RANGE = (0.05, 0.35)

_RATE_LIMIT_DEFAULTS: tuple[RateLimitRule, ...] = (
    RateLimitRule(rate=50, per=1.0),
    RateLimitRule(rate=1200, per=60.0),
)


class _CooldownMeasurement(float):
    """Pomocnicza wartość czasu z tolerancją zgodną z pytest.approx."""

    __slots__ = ()

    @staticmethod
    def _extract_tolerance(candidate: Any) -> tuple[float, float] | None:
        expected = getattr(candidate, "expected", None)
        if expected is None:
            return None
        try:
            expected_value = float(expected)
        except (TypeError, ValueError):  # pragma: no cover - zgodność typów
            return None
        abs_tol_raw = getattr(candidate, "abs", None)
        rel_tol_raw = getattr(candidate, "rel", None)
        tolerance = 0.0
        if abs_tol_raw is not None:
            try:
                tolerance = max(tolerance, float(abs_tol_raw))
            except (TypeError, ValueError):  # pragma: no cover - zabezpieczenie
                pass
        if rel_tol_raw is not None:
            try:
                tolerance = max(tolerance, abs(expected_value) * float(rel_tol_raw))
            except (TypeError, ValueError):  # pragma: no cover - zabezpieczenie
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


def _direct_conversion_rate(
    base: str, quote: str, prices: Mapping[str, float]
) -> Optional[float]:
    """Zwraca kurs wymiany dla pary base/quote na podstawie tickerów."""
    if base == quote:
        return 1.0

    direct_symbol = f"{base}{quote}"
    if direct_symbol in prices:
        return prices[direct_symbol]

    reverse_symbol = f"{quote}{base}"
    reverse_price = prices.get(reverse_symbol)
    if reverse_price:
        if reverse_price == 0:
            return None
        return 1.0 / reverse_price
    return None


def _determine_intermediaries(target: str, prices: Mapping[str, float]) -> set[str]:
    """Wyszukuje aktywa, które mają notowania z daną walutą docelową."""
    intermediaries: set[str] = set()
    target_len = len(target)
    for symbol in prices.keys():
        if symbol.endswith(target) and len(symbol) > target_len:
            intermediaries.add(symbol[: -target_len])
        elif symbol.startswith(target) and len(symbol) > target_len:
            intermediaries.add(symbol[target_len:])
    return intermediaries


def _convert_to_target(
    asset: str, target: str, prices: Mapping[str, float]
) -> Optional[float]:
    """Próbuje przeliczyć aktywo na walutę docelową przy użyciu triangulacji."""
    direct_rate = _direct_conversion_rate(asset, target, prices)
    if direct_rate is not None:
        return direct_rate

    for intermediary in _determine_intermediaries(target, prices):
        if intermediary == asset:
            continue
        first_leg = _direct_conversion_rate(asset, intermediary, prices)
        if first_leg is None:
            continue
        second_leg = _direct_conversion_rate(intermediary, target, prices)
        if second_leg is None:
            continue
        return first_leg * second_leg
    return None


@dataclass(slots=True)
class BinanceTicker:
    """Znormalizowany ticker 24h z rynku spot Binance."""

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
    timestamp: float


@dataclass(slots=True)
class BinanceOrderBookLevel:
    """Pojedynczy poziom orderbooka Binance Spot."""

    price: float
    quantity: float


@dataclass(slots=True)
class BinanceOrderBook:
    """Orderbook Binance Spot w formacie kompatybilnym ze StreamGateway."""

    symbol: str
    bids: tuple[BinanceOrderBookLevel, ...]
    asks: tuple[BinanceOrderBookLevel, ...]
    depth: int
    last_update_id: int
    timestamp: float


@dataclass(slots=True)
class BinanceOpenOrder:
    """Znormalizowana reprezentacja otwartego zlecenia Binance Spot."""

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
    iceberg_quantity: float | None
    is_working: bool
    update_time: float


class BinanceSpotAdapter(ExchangeAdapter):
    """Adapter dla rynku spot Binance z obsługą danych publicznych i podpisanych."""

    __slots__ = (
        "_environment",
        "_public_base",
        "_trading_base",
        "_ip_allowlist",
        "_permission_set",
        "_settings",
        "_valuation_asset",
        "_secondary_valuation_assets",
        "_metrics",
        "_metric_base_labels",
        "_metric_http_latency",
        "_metric_retries",
        "_metric_signed_requests",
        "_metric_weight",
        "_watchdog",
        "_throttle_cooldown_until",
        "_throttle_cooldown_reason",
        "_reconnect_backoff_until",
        "_reconnect_reason",
    )

    name: str = "binance_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
        settings: Mapping[str, object] | None = None,
        metrics_registry: MetricsRegistry | None = None,
        watchdog: Watchdog | None = None,
        network_error_handler: Callable[[str, Exception], None] | None = None,
    ) -> None:
        super().__init__(credentials)
        self._environment = environment or credentials.environment
        self._public_base = _determine_public_base(self._environment)
        self._trading_base = _determine_trading_base(self._environment)
        self._ip_allowlist: tuple[str, ...] = ()
        self._permission_set = frozenset(perm.lower() for perm in self._credentials.permissions)
        self._settings = dict(settings or {})
        self._valuation_asset = self._extract_valuation_asset()
        self._secondary_valuation_assets = self._extract_secondary_assets()
        self._metrics = metrics_registry or get_global_metrics_registry()
        self._metric_base_labels = {
            "exchange": self.name,
            "environment": self._environment.value,
        }
        self._metric_http_latency = self._metrics.histogram(
            "binance_spot_http_latency_seconds",
            "Czas trwania zapytań HTTP kierowanych do API Binance Spot.",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
        )
        self._metric_retries = self._metrics.counter(
            "binance_spot_retries_total",
            "Liczba ponowień zapytań do API Binance Spot (powód=throttled/network/server_error).",
        )
        self._metric_signed_requests = self._metrics.counter(
            "binance_spot_signed_requests_total",
            "Łączna liczba podpisanych zapytań HTTP wysłanych do API Binance Spot.",
        )
        self._metric_weight = self._metrics.gauge(
            "binance_spot_used_weight",
            "Ostatnie wartości nagłówków X-MBX-USED-WEIGHT od Binance Spot.",
        )
        self._watchdog = watchdog or Watchdog()
        self._network_error_handler = network_error_handler
        self._throttle_cooldown_until = 0.0
        self._throttle_cooldown_reason: str | None = None
        self._reconnect_backoff_until = 0.0
        self._reconnect_reason: str | None = None
        self._rate_limiter = get_global_rate_limiter_registry().configure(
            f"{self.name}:{self._environment.value}",
            normalize_rate_limit_rules(
                self._settings.get("rate_limit_rules"),
                _RATE_LIMIT_DEFAULTS,
            ),
            metric_labels={"exchange": self.name, "environment": self._environment.value},
        )

    # ----------------------------------------------------------------------------------
    # Konfiguracja streamingu long-pollowego
    # ----------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------
    # Konfiguracja wyceny sald
    # ----------------------------------------------------------------------------------
    def _extract_valuation_asset(self) -> str:
        raw = self._settings.get("valuation_asset", "USDT")
        if isinstance(raw, str):
            asset = raw.strip().upper()
            return asset or "USDT"
        return "USDT"

    def _extract_secondary_assets(self) -> tuple[str, ...]:
        raw = self._settings.get("secondary_valuation_assets")
        defaults = ("USDX", "BUSD", "USDC")

        def _append(container: list[str], value: object) -> None:
            asset = str(value).strip().upper()
            if not asset:
                return
            if asset == self._valuation_asset:
                return
            if asset not in container:
                container.append(asset)

        assets: list[str] = []
        if raw is None:
            for entry in defaults:
                _append(assets, entry)
        elif isinstance(raw, str):
            _append(assets, raw)
        elif isinstance(raw, Iterable):
            for entry in raw:
                _append(assets, entry)
        else:  # pragma: no cover - ochrona przed nietypową konfiguracją
            for entry in defaults:
                _append(assets, entry)
        return tuple(assets)

    # ----------------------------------------------------------------------------------
    # HTTP helpers
    # ----------------------------------------------------------------------------------
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
        return self._execute_request(request, endpoint=path, signed=False)

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

        return self._execute_request(request, endpoint=path, signed=True)

    @staticmethod
    def _calculate_backoff(attempt: int) -> float:
        base_delay = min(_BASE_BACKOFF * (2 ** (attempt - 1)), _BACKOFF_CAP)
        jitter = random.uniform(*_JITTER_RANGE)
        return base_delay + jitter

    def _record_weight_headers(self, headers: Any) -> None:
        if not headers:
            return

        def _get(name: str) -> Any:
            getter = getattr(headers, "get", None)
            if callable(getter):
                try:
                    return getter(name)
                except Exception:  # pragma: no cover - defensywnie przed niestandardowym nagłówkiem
                    return None
            getter = getattr(headers, "getheader", None)
            if callable(getter):
                return getter(name)
            if isinstance(headers, Mapping):  # pragma: no cover - fallback na mapowanie
                return headers.get(name)
            return None

        for window, header_name in (
            ("1m", "X-MBX-USED-WEIGHT-1M"),
            ("1s", "X-MBX-USED-WEIGHT-1S"),
        ):
            raw_value = _get(header_name)
            if raw_value in (None, ""):
                continue
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError):
                continue
            labels = dict(self._metric_base_labels)
            labels["window"] = window
            self._metric_weight.set(numeric, labels=labels)

    def _extract_retry_after(self, headers: Any) -> float | None:
        if not headers:
            return None

        def _get(name: str) -> Any:
            getter = getattr(headers, "get", None)
            if callable(getter):
                try:
                    return getter(name)
                except Exception:  # pragma: no cover
                    return None
            if isinstance(headers, Mapping):
                return headers.get(name)
            return None

        raw_value = _get("Retry-After") or _get("retry-after")
        if raw_value in (None, ""):
            return None
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return None

    def _register_throttle_cooldown(self, duration: float, *, reason: str) -> None:
        try:
            cooldown = float(duration)
        except (TypeError, ValueError):  # pragma: no cover - zabezpieczenie
            return
        if cooldown <= 0:
            return
        cooldown = max(0.5, min(cooldown, 60.0))
        deadline = time.monotonic() + cooldown
        if deadline > self._throttle_cooldown_until:
            self._throttle_cooldown_until = deadline
            self._throttle_cooldown_reason = reason
            _LOGGER.warning(
                "Binance Spot aktywował globalny cooldown %.2fs (powód=%s)",
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
                "Binance Spot wymaga ponownego połączenia – odczekaj %.2fs (powód=%s)",
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
                    message="Binance API pozostaje w globalnym cooldownie.",
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
                remaining = self._reconnect_backoff_until - now
                raise ExchangeNetworkError(
                    message=(
                        "Adapter Binance oczekuje na ponowne połączenie (pozostało %.2fs, powód=%s)."
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

    def _execute_request(
        self,
        request: Request,
        *,
        endpoint: str | None = None,
        signed: bool = False,
    ) -> dict[str, object] | list[object]:
        path = endpoint or urlsplit(request.full_url).path or "unknown"
        method = request.get_method() or "GET"
        metric_labels = {
            **self._metric_base_labels,
            "endpoint": path,
            "method": method,
        }
        self._enforce_failover_backoff()
        for attempt in range(1, _MAX_RETRIES + 1):
            weight = 5.0 if signed else 1.0
            self._rate_limiter.acquire(weight=weight)
            start = time.monotonic()
            if signed:
                self._metric_signed_requests.inc(labels={**self._metric_base_labels, "method": method})
            try:
                with urlopen(request, timeout=15) as response:  # nosec: B310 - endpoint zaufany
                    status_code = getattr(response, "status", getattr(response, "code", 200))
                    payload = response.read()
                    headers = getattr(response, "headers", None)
                latency = time.monotonic() - start
                self._metric_http_latency.observe(latency, labels=metric_labels)
                if headers is not None:
                    self._record_weight_headers(headers)
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError as exc:
                    _LOGGER.error(
                        "Niepoprawna odpowiedź JSON od Binance (endpoint=%s, environment=%s): %s",
                        path,
                        self._environment.value,
                        exc,
                    )
                    raise ExchangeAPIError(
                        message="Niepoprawna odpowiedź JSON od API Binance.",
                        status_code=0,
                        payload=None,
                    ) from exc
                self._raise_for_api_error(
                    data,
                    status_code=int(status_code or 200),
                    default_message=f"Binance API zwróciło błąd ({path})",
                )
                self._reset_reconnect_state()
                return data
            except HTTPError as exc:
                latency = time.monotonic() - start
                self._metric_http_latency.observe(latency, labels=metric_labels)
                status_code = exc.code
                raw_payload = b""
                try:
                    raw_payload = exc.read() or b""
                except Exception:  # pragma: no cover - zabezpieczenie przed nietypowym obiektem
                    raw_payload = b""
                parsed_payload: object | None = None
                if raw_payload:
                    try:
                        parsed_payload = json.loads(raw_payload)
                    except json.JSONDecodeError:
                        parsed_payload = raw_payload.decode("utf-8", errors="replace")

                if status_code in {401, 403}:
                    _LOGGER.error(
                        "Binance odrzuciło uwierzytelnienie (endpoint=%s, status=%s).",
                        path,
                        status_code,
                    )
                    raise ExchangeAuthError(
                        message="Binance API odrzuciło uwierzytelnienie.",
                        status_code=status_code,
                        payload=parsed_payload,
                    ) from exc

                if status_code in _RETRYABLE_STATUS:
                    self._metric_retries.inc(
                        labels={**metric_labels, "reason": "throttled", "status": str(status_code)}
                    )
                    retry_after = self._extract_retry_after(getattr(exc, "headers", None))
                    delay = self._calculate_backoff(attempt)
                    if retry_after is not None:
                        delay = max(delay, retry_after)
                    self._register_throttle_cooldown(delay, reason=f"http_{status_code}")
                    _LOGGER.warning(
                        "Binance rate limit (endpoint=%s, attempt=%s/%s, delay=%.2fs).",
                        path,
                        attempt,
                        _MAX_RETRIES,
                        delay,
                    )
                    if attempt == _MAX_RETRIES:
                        raise ExchangeThrottlingError(
                            message="Binance API odrzuciło zapytanie z powodu przekroczenia limitu.",
                            status_code=status_code,
                            payload=parsed_payload,
                        ) from exc
                    time.sleep(delay)
                    continue

                if 500 <= status_code < 600:
                    self._metric_retries.inc(
                        labels={**metric_labels, "reason": "server_error", "status": str(status_code)}
                    )
                    delay = self._calculate_backoff(attempt)
                    self._register_reconnect_cooldown(delay, reason=f"server_{status_code}")
                    _LOGGER.warning(
                        "Błąd serwera Binance (status=%s, endpoint=%s, attempt=%s/%s); retry za %.2fs.",
                        status_code,
                        path,
                        attempt,
                        _MAX_RETRIES,
                        delay,
                    )
                    if attempt == _MAX_RETRIES:
                        raise ExchangeAPIError(
                            message="Binance API zwróciło błąd serwera.",
                            status_code=status_code,
                            payload=parsed_payload,
                        ) from exc
                    time.sleep(delay)
                    continue

                _LOGGER.error(
                    "Błąd HTTP podczas komunikacji z Binance (status=%s, endpoint=%s).",
                    status_code,
                    path,
                )
                raise ExchangeAPIError(
                    message="Binance API zwróciło błąd.",
                    status_code=status_code,
                    payload=parsed_payload,
                ) from exc
            except URLError as exc:
                latency = time.monotonic() - start
                self._metric_http_latency.observe(latency, labels=metric_labels)
                self._metric_retries.inc(labels={**metric_labels, "reason": "network"})
                delay = self._calculate_backoff(attempt)
                self._notify_network_error(path, exc)
                _LOGGER.warning(
                    "Błąd sieci podczas komunikacji z Binance (endpoint=%s, attempt=%s/%s); retry za %.2fs: %s",
                    path,
                    attempt,
                    _MAX_RETRIES,
                    delay,
                    exc,
                )
                if attempt == _MAX_RETRIES:
                    self._register_reconnect_cooldown(delay, reason="network")
                    raise ExchangeNetworkError(
                        message="Nie udało się połączyć z API Binance.",
                        reason=exc,
                    ) from exc
                time.sleep(delay)
                continue

                raise ExchangeNetworkError(
                    message="Nie udało się uzyskać odpowiedzi od API Binance po wielokrotnych próbach.",
                    reason=None,
        )

    def _notify_network_error(self, endpoint: str, exc: Exception) -> None:
        if not self._network_error_handler:
            return
        try:
            self._network_error_handler(endpoint, exc)
        except Exception:  # pragma: no cover - defensywne logowanie
            _LOGGER.debug("Network error handler raised", exc_info=True)

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
        numeric_code: int | None
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

        status_field = payload.get("status")
        if isinstance(status_field, str) and status_field.lower() == "error":
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

    # ----------------------------------------------------------------------------------
    # ExchangeAdapter API
    # ----------------------------------------------------------------------------------
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

        def _call() -> AccountSnapshot:
            payload = self._signed_request("/api/v3/account")
            if not isinstance(payload, dict):
                raise RuntimeError("Niepoprawna odpowiedź konta z Binance")

            balances_section = payload.get("balances", [])
            balances: dict[str, float] = {}
            free_balances: dict[str, float] = {}
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
                    free_balances[asset] = free

            ticker_payload = self._public_request("/api/v3/ticker/price")
            prices: dict[str, float] = {}
            if isinstance(ticker_payload, list):
                for entry in ticker_payload:
                    if not isinstance(entry, Mapping):
                        continue
                    symbol = entry.get("symbol")
                    price = _to_float(entry.get("price", 0.0))
                    if isinstance(symbol, str):
                        prices[symbol] = price

            valuation_currency = self._valuation_asset
            secondary_currencies = self._secondary_valuation_assets or ("USDX",)
            total_equity = 0.0
            available_margin = 0.0
            for asset, total_balance in balances.items():
                conversion = _convert_to_target(asset, valuation_currency, prices)
                if conversion is None:
                    for secondary in secondary_currencies:
                        first_leg = _convert_to_target(asset, secondary, prices)
                        if first_leg is None:
                            continue
                        second_leg = _convert_to_target(secondary, valuation_currency, prices)
                        if second_leg is None:
                            continue
                        conversion = first_leg * second_leg
                        break
                if conversion is None:
                    continue
                total_equity += total_balance * conversion
                available_margin += free_balances.get(asset, 0.0) * conversion

            maintenance_margin = _to_float(
                payload.get("maintMarginBalance", payload.get("totalMarginBalance", 0.0))
            )

            return AccountSnapshot(
                balances=balances,
                total_equity=total_equity,
                available_margin=available_margin,
                maintenance_margin=maintenance_margin,
            )

        return self._watchdog.execute("binance_spot_fetch_account", _call)

    def fetch_symbols(self) -> Iterable[str]:
        """Pobiera listę aktywnych symboli spot z Binance."""
        def _call() -> Iterable[str]:
            payload = self._public_request("/api/v3/exchangeInfo")
            if not isinstance(payload, dict) or "symbols" not in payload:
                raise RuntimeError("Niepoprawna odpowiedź exchangeInfo z Binance")

            symbols_section = payload.get("symbols")
            if not isinstance(symbols_section, list):
                raise RuntimeError("Pole 'symbols' w odpowiedzi Binance ma niepoprawny format")

            raw_symbols: list[str] = []
            for entry in symbols_section:
                if not isinstance(entry, dict):
                    continue
                status = entry.get("status")
                symbol = entry.get("symbol")
                if status != "TRADING" or not isinstance(symbol, str):
                    continue
                raw_symbols.append(symbol)
            return filter_supported_exchange_symbols(raw_symbols)

        return self._watchdog.execute("binance_spot_fetch_symbols", _call)

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        """Pobiera świece OHLCV w formacie zgodnym z modułem danych."""
        exchange_symbol = to_exchange_symbol(symbol)
        if exchange_symbol is None:
            raise ValueError(f"Symbol {symbol!r} nie jest wspierany przez adapter Binance Spot.")

        params: dict[str, object] = {"symbol": exchange_symbol, "interval": interval}
        if start is not None:
            params["startTime"] = int(start)
        if end is not None:
            params["endTime"] = int(end)
        if limit is not None:
            params["limit"] = int(limit)

        def _call() -> Sequence[Sequence[float]]:
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

        return self._watchdog.execute("binance_spot_fetch_ohlcv", _call)

    def _resolve_symbol(self, symbol: str) -> tuple[str, str]:
        """Zwraca parę (symbol_binance, symbol_kanoniczny)."""

        exchange_symbol = to_exchange_symbol(symbol)
        if exchange_symbol is None:
            raise ValueError(f"Symbol {symbol!r} nie jest wspierany przez Binance Spot.")

        canonical_symbol = (
            normalize_symbol(symbol)
            or normalize_symbol(exchange_symbol)
            or exchange_symbol
        )
        return exchange_symbol, canonical_symbol

    def fetch_ticker(self, symbol: str) -> BinanceTicker:
        """Pobiera statystyki 24h dla symbolu w formacie zgodnym z StreamGateway."""

        exchange_symbol, canonical_symbol = self._resolve_symbol(symbol)

        def _call() -> BinanceTicker:
            payload = self._public_request("/api/v3/ticker/24hr", params={"symbol": exchange_symbol})
            if not isinstance(payload, Mapping):
                raise ExchangeAPIError(
                    "Binance Spot zwrócił niepoprawną strukturę tickera.",
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
            timestamp = _timestamp_ms_to_seconds(payload.get("closeTime"), fallback=time.time())

            return BinanceTicker(
                symbol=canonical_symbol,
                best_bid=best_bid,
                best_ask=best_ask,
                last_price=last_price,
                price_change_percent=price_change_percent,
                open_price=open_price,
                high_24h=high_24h,
                low_24h=low_24h,
                volume_24h_base=volume_base,
                volume_24h_quote=volume_quote,
                timestamp=timestamp,
            )

        return self._watchdog.execute("binance_spot_fetch_ticker", _call)

    def fetch_order_book(self, symbol: str, *, depth: int = 50) -> BinanceOrderBook:
        """Pobiera orderbook (bids/asks) ograniczony do wskazanej głębokości."""

        exchange_symbol, canonical_symbol = self._resolve_symbol(symbol)

        normalized_depth = _normalize_depth(depth)
        params = {"symbol": exchange_symbol, "limit": normalized_depth}
        def _call() -> BinanceOrderBook:
            payload = self._public_request("/api/v3/depth", params=params)
            if not isinstance(payload, Mapping):
                raise ExchangeAPIError(
                    "Binance Spot zwrócił niepoprawną strukturę orderbooka.",
                    400,
                    payload=payload,
                )

            bids_raw = payload.get("bids")
            asks_raw = payload.get("asks")
            bids: list[BinanceOrderBookLevel] = []
            if isinstance(bids_raw, Sequence):
                for entry in bids_raw:
                    if not isinstance(entry, Sequence) or len(entry) < 2:
                        continue
                    price = _to_float(entry[0])
                    quantity = _to_float(entry[1])
                    if price <= 0 or quantity <= 0:
                        continue
                    bids.append(BinanceOrderBookLevel(price=price, quantity=quantity))

            asks: list[BinanceOrderBookLevel] = []
            if isinstance(asks_raw, Sequence):
                for entry in asks_raw:
                    if not isinstance(entry, Sequence) or len(entry) < 2:
                        continue
                    price = _to_float(entry[0])
                    quantity = _to_float(entry[1])
                    if price <= 0 or quantity <= 0:
                        continue
                    asks.append(BinanceOrderBookLevel(price=price, quantity=quantity))

            try:
                last_update_id = int(payload.get("lastUpdateId", 0))
            except (TypeError, ValueError):
                last_update_id = 0
            timestamp = _timestamp_ms_to_seconds(payload.get("E"), fallback=time.time())

            return BinanceOrderBook(
                symbol=canonical_symbol,
                bids=tuple(bids),
                asks=tuple(asks),
                depth=normalized_depth,
                last_update_id=last_update_id,
                timestamp=timestamp,
            )

        return self._watchdog.execute("binance_spot_fetch_order_book", _call)

    def fetch_open_orders(self) -> Sequence[BinanceOpenOrder]:
        """Zwraca listę otwartych zleceń wykorzystując podpisane API Binance."""

        if not ({"read", "trade"} & self._permission_set):
            raise PermissionError("Poświadczenia nie pozwalają na odczyt zleceń Binance Spot.")

        def _call() -> Sequence[BinanceOpenOrder]:
            payload = self._signed_request("/api/v3/openOrders")
            if not isinstance(payload, list):
                raise ExchangeAPIError(
                    "Binance Spot zwrócił niepoprawną strukturę listy zleceń.",
                    400,
                    payload=payload,
                )

            orders: list[BinanceOpenOrder] = []
            for entry in payload:
                if not isinstance(entry, Mapping):
                    continue
                raw_symbol = entry.get("symbol")
                exchange_symbol = str(raw_symbol) if isinstance(raw_symbol, str) else ""
                canonical_symbol = normalize_symbol(exchange_symbol) or exchange_symbol
                price_value = _to_float(entry.get("price"))
                price = price_value if price_value > 0 else None
                stop_price_value = _to_float(entry.get("stopPrice"))
                stop_price = stop_price_value if stop_price_value > 0 else None
                iceberg_value = _to_float(entry.get("icebergQty"))
                iceberg = iceberg_value if iceberg_value > 0 else None
                timestamp = _timestamp_ms_to_seconds(entry.get("updateTime") or entry.get("time"))
                order = BinanceOpenOrder(
                    order_id=str(entry.get("orderId", "")),
                    symbol=canonical_symbol,
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
                    iceberg_quantity=iceberg,
                    is_working=bool(entry.get("isWorking", True)),
                    update_time=timestamp,
                )
                orders.append(order)

            orders.sort(key=lambda item: item.update_time)
            return orders

        return self._watchdog.execute("binance_spot_fetch_open_orders", _call)

    def place_order(self, request: OrderRequest) -> OrderResult:
        """Składa podpisane zlecenie typu limit/market na rynku spot."""
        if "trade" not in self._permission_set:
            raise PermissionError("Aktualne poświadczenia nie mają uprawnień tradingowych.")

        exchange_symbol = to_exchange_symbol(request.symbol)
        if exchange_symbol is None:
            raise ValueError(
                "Symbol zlecenia nie jest wspierany lub posiada niepoprawny format."
            )

        params: dict[str, object] = {
            "symbol": exchange_symbol,
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

        def _call() -> OrderResult:
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

        return self._watchdog.execute("binance_spot_place_order", _call)

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:
        if "trade" not in self._permission_set:
            raise PermissionError("Aktualne poświadczenia nie mają uprawnień tradingowych.")
        params: dict[str, object] = {"orderId": order_id}
        if symbol:
            exchange_symbol = to_exchange_symbol(symbol)
            if exchange_symbol is None:
                raise ValueError("Symbol anulowanego zlecenia ma niepoprawny format.")
            params["symbol"] = exchange_symbol
        def _call() -> None:
            response = self._signed_request("/api/v3/order", method="DELETE", params=params)
            if isinstance(response, Mapping):
                response_map = dict(response)
                if response_map.get("status") in {"CANCELED", "PENDING_CANCEL"}:
                    return
                raise RuntimeError(f"Nieoczekiwana odpowiedź anulowania z Binance: {response_map}")
            raise RuntimeError("Niepoprawna odpowiedź anulowania z Binance")

        self._watchdog.execute("binance_spot_cancel_order", _call)

    def stream_public_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        return self._build_stream("public", channels)

    def stream_private_data(self, *, channels: Sequence[str]):  # type: ignore[override]
        if not ({"read", "trade"} & self._permission_set):
            raise PermissionError("Poświadczenia nie pozwalają na strumień prywatny Binance Spot.")
        return self._build_stream("private", channels)


__all__ = [
    "BinanceSpotAdapter",
    "BinanceTicker",
    "BinanceOrderBookLevel",
    "BinanceOrderBook",
    "BinanceOpenOrder",
]
