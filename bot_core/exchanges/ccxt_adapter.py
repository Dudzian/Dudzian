"""Wspólna logika adapterów CCXT dla rynków spot."""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from bot_core.alerts.base import AlertRouter
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
from bot_core.exchanges.health import RetryPolicy, Watchdog
from bot_core.exchanges.network_guard import NetworkAccessGuard, NetworkAccessViolation
from bot_core.exchanges.rate_limiter import (
    RateLimitRule,
    get_global_rate_limiter_registry,
    normalize_rate_limit_rules,
)
from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry
from bot_core.exchanges.streaming import LocalLongPollStream

try:  # pragma: no cover - opcjonalny import CCXT
    import ccxt  # type: ignore
    from ccxt.base.exchange import Exchange as CCXTExchange  # type: ignore
    from ccxt.base.errors import (  # type: ignore
        AuthenticationError as _CCXTAuthenticationError,
        ExchangeError as _CCXTExchangeError,
        NetworkError as _CCXTNetworkError,
        PermissionDenied as _CCXTPermissionDenied,
        RateLimitExceeded as _CCXTRateLimitExceeded,
    )
except Exception:  # pragma: no cover - środowiska bez CCXT
    ccxt = None  # type: ignore

    class CCXTExchange:  # type: ignore[override]
        """Minimalny interfejs wykorzystywany w testach jednostkowych."""

        symbols: list[str]

    class _CCXTExchangeError(Exception):
        ...

    class _CCXTAuthenticationError(_CCXTExchangeError):
        ...

    class _CCXTPermissionDenied(_CCXTAuthenticationError):
        ...

    class _CCXTRateLimitExceeded(_CCXTExchangeError):
        ...

    class _CCXTNetworkError(_CCXTExchangeError):
        ...


_CCXT_INSTALL_HINT = (
    "Biblioteka 'ccxt' nie jest dostępna – zainstaluj zależność 'ccxt>=4.0.0' w swoim "
    "środowisku (np. `pip install ccxt`)."
)


_LOGGER = logging.getLogger(__name__)

_DEFAULT_LATENCY_BUCKETS: tuple[float, ...] = (
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
)


def merge_adapter_settings(
    defaults: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Łączy słowniki konfiguracji adaptera zachowując zagnieżdżone wartości."""

    merged = deepcopy(defaults)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_adapter_settings(existing, value)
        else:
            merged[key] = value
    return merged


class CCXTSpotAdapter(ExchangeAdapter):
    """Bazowa implementacja adaptera CCXT dla rynków spot."""

    __slots__ = (
        "_environment",
        "_client",
        "_settings",
        "_metrics",
        "_metric_labels",
        "_metric_requests",
        "_metric_failures",
        "_metric_latency",
        "_ip_allowlist",
        "_exchange_id",
        "_network_errors",
        "_auth_errors",
        "_rate_limit_errors",
        "_base_errors",
        "_network_guard",
        "_rate_limiter",
        "_retry_policy",
        "_sleep",
    )

    name = "ccxt_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        exchange_id: str,
        environment: Environment | None = None,
        settings: Mapping[str, Any] | None = None,
        client: CCXTExchange | None = None,
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        super().__init__(credentials)
        self._exchange_id = exchange_id
        self._environment = environment or credentials.environment or Environment.LIVE
        self._settings = dict(settings or {})
        self._ip_allowlist: tuple[str, ...] = ()
        self._metrics = metrics_registry or get_global_metrics_registry()
        self._metric_labels = {"exchange": self.name}
        self._metric_requests = self._metrics.counter(
            "exchange_ccxt_requests_total",
            "Liczba wywołań CCXT wykonanych przez adapter.",
        )
        self._metric_failures = self._metrics.counter(
            "exchange_ccxt_failures_total",
            "Liczba błędów CCXT obsłużonych przez adapter.",
        )
        self._metric_latency = self._metrics.histogram(
            "exchange_ccxt_http_latency_seconds",
            "Czas odpowiedzi operacji CCXT.",
            _DEFAULT_LATENCY_BUCKETS,
        )
        self._network_errors = self._resolve_error_types(
            "network_error_types", (_CCXTNetworkError,) if _CCXTNetworkError else ()
        )
        self._auth_errors = self._resolve_error_types(
            "auth_error_types",
            tuple(
                error
                for error in (_CCXTAuthenticationError, _CCXTPermissionDenied)
                if error
            ),
        )
        self._rate_limit_errors = self._resolve_error_types(
            "rate_limit_error_types",
            (_CCXTRateLimitExceeded,) if _CCXTRateLimitExceeded else (),
        )
        self._base_errors = self._resolve_error_types(
            "base_error_types", (_CCXTExchangeError,) if _CCXTExchangeError else ()
        )
        self._network_guard = NetworkAccessGuard(logger=_LOGGER)
        self._client = client or self._build_client()
        default_rules = (
            RateLimitRule(rate=60, per=60.0),
        )
        configured_rules = self._settings.get("rate_limit_rules")
        self._rate_limiter = get_global_rate_limiter_registry().configure(
            f"{self.name}:{self._exchange_id}:{self._environment.value}",
            normalize_rate_limit_rules(configured_rules, default_rules),
            metric_labels={"exchange": self.name, "mode": self._environment.value},
        )
        retry_settings = self._settings.get("retry_policy")
        if isinstance(retry_settings, Mapping):
            self._retry_policy = RetryPolicy(**retry_settings)  # type: ignore[arg-type]
        else:
            self._retry_policy = RetryPolicy()
        self._sleep: Callable[[float], None] = self._settings.get("sleep_callable") or time.sleep

    # --- Metody pomocnicze -------------------------------------------------

    def _resolve_error_types(
        self, setting_key: str, defaults: tuple[type[Exception], ...]
    ) -> tuple[type[Exception], ...]:
        configured = self._settings.get(setting_key)
        if configured:
            return tuple(type_ for type_ in configured if type_)
        return tuple(type_ for type_ in defaults if type_)

    def _ensure_network_access(self) -> None:
        try:
            self._network_guard.ensure_configured()
        except NetworkAccessViolation as exc:
            raise ExchangeNetworkError(
                message=f"Naruszenie konfiguracji sieci CCXT: {exc.reason}",
                reason=exc,
            ) from exc

    def _build_client(self) -> CCXTExchange:
        if ccxt is None:  # pragma: no cover - środowiska offline
            raise RuntimeError(
                f"{_CCXT_INSTALL_HINT} Jeżeli uruchamiasz testy bez dostępu do ccxt, "
                "przekaż gotowy klient poprzez argument 'client'."
            )
        try:
            constructor: Callable[..., CCXTExchange] = getattr(ccxt, self._exchange_id)
        except AttributeError as exc:  # pragma: no cover - konfiguracja
            raise ValueError(f"Nieznany adapter CCXT: {self._exchange_id}") from exc

        config = dict(self._settings.get("ccxt_config", {}))
        if self.credentials.key_id and "apiKey" not in config:
            config["apiKey"] = self.credentials.key_id
        if self.credentials.secret and "secret" not in config:
            config["secret"] = self.credentials.secret
        if self.credentials.passphrase and "password" not in config:
            config["password"] = self.credentials.passphrase
        config.setdefault("enableRateLimit", True)

        client = constructor(config)
        sandbox_enabled = self._settings.get("sandbox_mode")
        if sandbox_enabled is None:
            sandbox_enabled = self._environment in (Environment.PAPER, Environment.TESTNET)
        if sandbox_enabled and hasattr(client, "set_sandbox_mode"):
            try:
                client.set_sandbox_mode(True)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensywnie
                _LOGGER.debug("Nie udało się włączyć trybu sandbox dla %s", self.name)
        return client

    def _wrap_call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        last_exception: tuple[Exception, BaseException | None, bool] | None = None
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            self._ensure_network_access()
            self._rate_limiter.acquire()
            start = time.perf_counter()
            self._metric_requests.inc(labels=self._metric_labels)
            try:
                result = func(*args, **kwargs)
            except tuple(self._network_errors) as exc:
                self._metric_failures.inc(labels=self._metric_labels)
                wrapped = ExchangeNetworkError("Błąd sieci CCXT", reason=exc)
                last_exception = (wrapped, exc, True)
            except tuple(self._auth_errors) as exc:
                self._metric_failures.inc(labels=self._metric_labels)
                wrapped = ExchangeAuthError(
                    "Błąd autoryzacji CCXT", status_code=401, payload=str(exc)
                )
                last_exception = (wrapped, exc, False)
            except tuple(self._rate_limit_errors) as exc:
                self._metric_failures.inc(labels=self._metric_labels)
                wrapped = ExchangeThrottlingError(
                    "Przekroczono limity CCXT", status_code=429, payload=str(exc)
                )
                last_exception = (wrapped, exc, True)
            except tuple(self._base_errors) as exc:
                self._metric_failures.inc(labels=self._metric_labels)
                wrapped = ExchangeAPIError("Błąd CCXT", status_code=500, payload=str(exc))
                last_exception = (wrapped, exc, False)
            else:
                return result
            finally:
                elapsed = time.perf_counter() - start
                self._metric_latency.observe(elapsed, labels=self._metric_labels)

            if last_exception is None:
                continue
            wrapped_exc, original_exc, should_retry = last_exception
            if should_retry and attempt < self._retry_policy.max_attempts:
                delay = self._retry_policy.compute_delay(attempt)
                self._sleep(delay)
                last_exception = None
                continue
            if original_exc is not None:
                raise wrapped_exc from original_exc
            raise wrapped_exc

    def _call_client(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._client, method_name)
        return self._wrap_call(method, *args, **kwargs)

    @staticmethod
    def _to_float(value: object) -> float | None:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return numeric

    @staticmethod
    def _to_timestamp(value: object) -> float | None:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if numeric > 10_000_000_000:  # timestamp w ms
            numeric /= 1000.0
        if numeric <= 0:
            return None
        return numeric

    # --- Implementacja interfejsu ExchangeAdapter -------------------------

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:
        self._ip_allowlist = tuple(ip_allowlist or ())
        self._network_guard.configure(ip_allowlist=self._ip_allowlist)

    def set_alert_router(self, alert_router: "AlertRouter | None") -> None:
        self._network_guard.attach_alert_router(alert_router)

    def fetch_account_snapshot(self) -> AccountSnapshot:
        balance: Mapping[str, Any] = self._call_client("fetch_balance") or {}
        free = balance.get("free") or {}
        total = balance.get("total") or {}
        used = balance.get("used") or {}

        normalized_balances = {
            str(asset): float(amount)
            for asset, amount in free.items()
            if amount is not None
        }
        total_equity = sum(float(amount or 0.0) for amount in total.values())
        available_margin = sum(float(amount or 0.0) for amount in free.values())
        maintenance_margin = sum(float(amount or 0.0) for amount in used.values())

        return AccountSnapshot(
            balances=normalized_balances,
            total_equity=total_equity,
            available_margin=available_margin,
            maintenance_margin=maintenance_margin,
        )

    def fetch_symbols(self) -> Sequence[str]:
        markets = self._call_client("load_markets")
        if markets is None:
            return tuple(getattr(self._client, "symbols", []) or [])
        if isinstance(markets, Mapping):
            return tuple(sorted(str(symbol) for symbol in markets.keys()))
        return tuple(getattr(self._client, "symbols", []) or [])

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:
        params = dict(self._settings.get("fetch_ohlcv_params", {}))
        if end is not None:
            params.setdefault("until", end)
        result = self._call_client(
            "fetch_ohlcv",
            symbol,
            interval,
            since=start,
            limit=limit,
            params=params or None,
        )
        return result or []

    def place_order(self, request: OrderRequest) -> OrderResult:
        params = dict(self._settings.get("create_order_params", {}))
        payload = {
            "symbol": request.symbol,
            "type": request.order_type,
            "side": request.side,
            "amount": request.quantity,
            "price": request.price,
            "params": params or None,
        }
        response: Mapping[str, Any] = self._call_client(
            "create_order",
            payload["symbol"],
            payload["type"],
            payload["side"],
            payload["amount"],
            payload["price"],
            params=payload["params"],
        )

        filled_quantity = float(response.get("filled") or response.get("amount_filled") or response.get("amount") or 0.0)
        remaining = response.get("remaining")
        if remaining is not None:
            try:
                filled_quantity = float(response.get("amount", 0.0)) - float(remaining)
            except (TypeError, ValueError):
                filled_quantity = float(response.get("filled") or filled_quantity)

        avg_price = response.get("average")
        if avg_price is None:
            avg_price = response.get("price")
        try:
            avg_price_value = float(avg_price) if avg_price is not None else None
        except (TypeError, ValueError):
            avg_price_value = None

        return OrderResult(
            order_id=str(response.get("id") or response.get("order", "")),
            status=str(response.get("status", "unknown")),
            filled_quantity=filled_quantity,
            avg_price=avg_price_value,
            raw_response=dict(response),
        )

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        params = dict(self._settings.get("cancel_order_params", {}))
        self._call_client("cancel_order", order_id, symbol, params=params or None)

    def _stream_settings(self) -> Mapping[str, object]:
        raw = self._settings.get("stream")
        if isinstance(raw, Mapping):
            return raw
        return {}

    def _build_stream(self, scope: str, channels: Sequence[str]) -> LocalLongPollStream:
        stream_settings = dict(self._stream_settings())
        base_url = str(stream_settings.get("base_url", "http://127.0.0.1:8765"))
        default_path = f"/stream/{self.name}/{scope}"
        path = str(stream_settings.get(f"{scope}_path", stream_settings.get("path", default_path)) or default_path)
        poll_interval = float(stream_settings.get("poll_interval", 0.5))
        timeout = float(stream_settings.get("timeout", 10.0))
        max_retries = int(stream_settings.get("max_retries", 3))
        backoff_base = float(stream_settings.get("backoff_base", 0.25))
        backoff_cap = float(stream_settings.get("backoff_cap", 2.0))
        jitter = stream_settings.get("jitter", (0.05, 0.30))
        channel_param = stream_settings.get(f"{scope}_channel_param", stream_settings.get("channel_param", "channels"))
        cursor_param = stream_settings.get(f"{scope}_cursor_param", stream_settings.get("cursor_param", "cursor"))
        initial_cursor = stream_settings.get(f"{scope}_initial_cursor", stream_settings.get("initial_cursor"))
        channel_serializer = stream_settings.get(f"{scope}_channel_serializer") or stream_settings.get("channel_serializer")
        if not callable(channel_serializer):
            separator = stream_settings.get(f"{scope}_channel_separator", stream_settings.get("channel_separator", ","))
            if isinstance(separator, str):
                channel_serializer = lambda values, sep=separator: sep.join(values)  # noqa: E731
            else:
                channel_serializer = None
        headers_raw = stream_settings.get("headers")
        headers = dict(headers_raw) if isinstance(headers_raw, Mapping) else None
        params: dict[str, object] = {}
        base_params = stream_settings.get("params")
        if isinstance(base_params, Mapping):
            params.update(base_params)
        scope_params = stream_settings.get(f"{scope}_params")
        if isinstance(scope_params, Mapping):
            params.update(scope_params)
        token_key = f"{scope}_token"
        token_value = stream_settings.get(token_key, stream_settings.get("auth_token"))
        if isinstance(token_value, str):
            params.setdefault("token", token_value)
        http_method = stream_settings.get(f"{scope}_method", stream_settings.get("method", "GET"))
        params_in_body = bool(stream_settings.get(f"{scope}_params_in_body", stream_settings.get("params_in_body", False)))
        channels_in_body = bool(stream_settings.get(f"{scope}_channels_in_body", stream_settings.get("channels_in_body", False)))
        cursor_in_body = bool(stream_settings.get(f"{scope}_cursor_in_body", stream_settings.get("cursor_in_body", False)))
        body_params: dict[str, object] = {}
        base_body = stream_settings.get("body_params")
        if isinstance(base_body, Mapping):
            body_params.update(base_body)
        scope_body = stream_settings.get(f"{scope}_body_params")
        if isinstance(scope_body, Mapping):
            body_params.update(scope_body)
        body_encoder = stream_settings.get(f"{scope}_body_encoder", stream_settings.get("body_encoder"))
        buffer_size = int(stream_settings.get("buffer_size", 64))

        self._ensure_network_access()

        return LocalLongPollStream(
            base_url=base_url,
            path=path,
            channels=channels,
            adapter=self.name,
            scope=scope,
            environment=self._environment.value,
            params=params,
            headers=headers,
            poll_interval=poll_interval,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_cap=backoff_cap,
            jitter=jitter if isinstance(jitter, Sequence) else (0.05, 0.30),
            channel_param=str(channel_param) if channel_param not in (None, "") else "",
            cursor_param=str(cursor_param) if cursor_param not in (None, "") else "",
            initial_cursor=str(initial_cursor) if initial_cursor not in (None, "") else None,
            channel_serializer=channel_serializer if callable(channel_serializer) else None,
            http_method=str(http_method or "GET"),
            params_in_body=params_in_body,
            channels_in_body=channels_in_body,
            cursor_in_body=cursor_in_body,
            body_params=body_params or None,
            body_encoder=body_encoder,
            buffer_size=max(1, buffer_size),
            metrics_registry=self._metrics,
        )

    def stream_public_data(self, *, channels: Sequence[str]):
        return self._build_stream("public", channels)

    def stream_private_data(self, *, channels: Sequence[str]):
        permissions = {perm.lower() for perm in self.credentials.permissions}
        if not ({"trade", "read"} & permissions):
            raise PermissionError("Poświadczenia CCXT nie pozwalają na strumienie prywatne.")
        return self._build_stream("private", channels)

    def fetch_recent_fills(
        self,
        symbol: str | None = None,
        *,
        limit: int = 50,
        since: int | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        permissions = {perm.lower() for perm in self.credentials.permissions}
        if not ({"trade", "read"} & permissions):
            raise PermissionError("Poświadczenia CCXT nie zezwalają na odczyt transakcji prywatnych.")

        params = dict(self._settings.get("fetch_my_trades_params", {}))
        result = self._call_client(
            "fetch_my_trades",
            symbol,
            since=since,
            limit=limit,
            params=params or None,
        )

        fills: list[dict[str, Any]] = []
        if isinstance(result, Sequence):
            for entry in result:
                if not isinstance(entry, Mapping):
                    continue
                trade_id = entry.get("id")
                order_id = entry.get("order") or entry.get("orderId")
                symbol_value = entry.get("symbol") or symbol or ""
                price = self._to_float(entry.get("price"))
                amount = self._to_float(entry.get("amount")) or self._to_float(entry.get("quantity"))
                cost = self._to_float(entry.get("cost"))
                fee_info = entry.get("fee") if isinstance(entry.get("fee"), Mapping) else None
                fee_cost = self._to_float(entry.get("fee")) if fee_info is None else self._to_float(fee_info.get("cost"))
                fee_currency = None
                if isinstance(fee_info, Mapping):
                    currency = fee_info.get("currency")
                    if currency not in (None, ""):
                        fee_currency = str(currency)
                maker_flag = entry.get("takerOrMaker")
                if isinstance(maker_flag, str):
                    maker = maker_flag.lower() == "maker"
                else:
                    maker = bool(entry.get("maker"))
                timestamp = self._to_timestamp(entry.get("timestamp"))
                normalized = {
                    "trade_id": str(trade_id) if trade_id not in (None, "") else "",
                    "order_id": str(order_id) if order_id not in (None, "") else None,
                    "symbol": str(symbol_value),
                    "side": str(entry.get("side") or ""),
                    "price": price,
                    "quantity": amount,
                    "cost": cost,
                    "fee": fee_cost,
                    "fee_asset": fee_currency,
                    "maker": maker,
                    "timestamp": timestamp,
                }
                fills.append(normalized)

        fills.sort(key=lambda item: item.get("timestamp") or 0.0)
        return tuple(fills)



class WatchdogCCXTAdapter(CCXTSpotAdapter):
    """Rozszerzenie adaptera CCXT o integrację z watchdogiem."""

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        exchange_id: str,
        environment: Environment | None = None,
        settings: Mapping[str, Any] | None = None,
        client: CCXTExchange | None = None,
        metrics_registry: MetricsRegistry | None = None,
        watchdog: Watchdog | None = None,
    ) -> None:
        self._watchdog = watchdog or Watchdog()
        super().__init__(
            credentials,
            exchange_id=exchange_id,
            environment=environment,
            settings=settings,
            client=client,
            metrics_registry=metrics_registry,
        )

    def _call_client(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        operation = f"{self.name}.{method_name}"
        return self._watchdog.execute(
            operation,
            lambda: super()._call_client(method_name, *args, **kwargs),
        )


__all__ = ["CCXTSpotAdapter", "WatchdogCCXTAdapter", "merge_adapter_settings"]

