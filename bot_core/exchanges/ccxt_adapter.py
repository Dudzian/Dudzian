"""Wspólna logika adapterów CCXT dla rynków spot."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Callable, Sequence

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
from bot_core.exchanges.network_guard import (
    ExchangeNetworkGuard,
    NetworkAccessGuard,
    NetworkAccessViolation,
    build_exchange_network_guard,
)
from bot_core.exchanges.rate_limiter import RateLimitRule, normalize_rate_limit_rules
from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry

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

    class _CCXTExchangeError(Exception): ...

    class _CCXTAuthenticationError(_CCXTExchangeError): ...

    class _CCXTPermissionDenied(_CCXTAuthenticationError): ...

    class _CCXTRateLimitExceeded(_CCXTExchangeError): ...

    class _CCXTNetworkError(_CCXTExchangeError): ...


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
        "_network_guard_bundle",
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
        network_guard: ExchangeNetworkGuard | None = None,
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
            tuple(error for error in (_CCXTAuthenticationError, _CCXTPermissionDenied) if error),
        )
        self._rate_limit_errors = self._resolve_error_types(
            "rate_limit_error_types",
            (_CCXTRateLimitExceeded,) if _CCXTRateLimitExceeded else (),
        )
        self._base_errors = self._resolve_error_types(
            "base_error_types", (_CCXTExchangeError,) if _CCXTExchangeError else ()
        )
        default_rules = (RateLimitRule(rate=60, per=60.0),)
        configured_rules = self._settings.get("rate_limit_rules")
        guard_labels = {"exchange": self.name, "mode": self._environment.value}
        guard = network_guard or build_exchange_network_guard(
            adapter_name=f"{self.name}:{self._exchange_id}",
            environment=self._environment.value,
            rate_limit_rules=normalize_rate_limit_rules(configured_rules, default_rules),
            default_rules=default_rules,
            metric_labels=guard_labels,
        )
        self._network_guard_bundle = guard
        self._network_guard = guard.network_access
        self._rate_limiter = guard.rate_limiter
        self._client = client or self._build_client()
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

    def _wrap_call(
        self,
        func: Callable[..., Any],
        *args: Any,
        retry: bool = True,
        **kwargs: Any,
    ) -> Any:
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
            if retry and should_retry and attempt < self._retry_policy.max_attempts:
                delay = self._retry_policy.compute_delay(attempt)
                self._sleep(delay)
                last_exception = None
                continue
            if original_exc is not None:
                raise wrapped_exc from original_exc
            raise wrapped_exc

    def _call_client(
        self,
        method_name: str,
        *args: Any,
        retry: bool = True,
        **kwargs: Any,
    ) -> Any:
        method = getattr(self._client, method_name)
        return self._wrap_call(method, *args, retry=retry, **kwargs)

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
            str(asset): float(amount) for asset, amount in free.items() if amount is not None
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
        client_order_id_param = self._settings.get("client_order_id_param")
        if (
            request.client_order_id
            and isinstance(client_order_id_param, str)
            and client_order_id_param
        ):
            params.setdefault(client_order_id_param, request.client_order_id)
        payload = {
            "symbol": request.symbol,
            "type": request.order_type,
            "side": request.side,
            "amount": request.quantity,
            "price": request.price,
            "params": params or None,
        }
        try:
            response: Mapping[str, Any] = self._call_client(
                "create_order",
                payload["symbol"],
                payload["type"],
                payload["side"],
                payload["amount"],
                payload["price"],
                retry=False,
                params=payload["params"],
            )
        except (ExchangeNetworkError, ExchangeThrottlingError, ExchangeAPIError) as exc:
            if request.client_order_id and self._is_ambiguous_place_order_error(exc):
                try:
                    reconciled = self.fetch_order_by_client_id(
                        request.client_order_id, symbol=request.symbol
                    )
                except Exception:
                    reconciled = None
                if reconciled is not None:
                    return reconciled
            raise

        return self._order_result_from_mapping(response)

    @staticmethod
    def _is_ambiguous_place_order_error(exc: Exception) -> bool:
        if isinstance(exc, (ExchangeNetworkError, ExchangeThrottlingError)):
            return True
        if isinstance(exc, ExchangeAPIError):
            return (exc.status_code or 0) >= 500
        return False

    @staticmethod
    def _order_result_from_mapping(response: Mapping[str, Any]) -> OrderResult:
        filled_quantity = float(
            response.get("filled") or response.get("amount_filled") or response.get("amount") or 0.0
        )
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

    def fetch_order_by_client_id(
        self,
        client_order_id: str,
        *,
        symbol: str | None = None,
    ) -> OrderResult | None:
        params = dict(self._settings.get("fetch_open_orders_params", {}))
        try:
            orders = self._call_client(
                "fetch_open_orders",
                symbol,
                retry=False,
                params=params or None,
            )
        except (AttributeError, ExchangeAPIError, ExchangeNetworkError, NotImplementedError):
            return None
        if not isinstance(orders, Sequence):
            return None
        for entry in orders:
            if not isinstance(entry, Mapping):
                continue
            candidate_ids = {
                entry.get("clientOrderId"),
                entry.get("client_order_id"),
            }
            info = entry.get("info")
            if isinstance(info, Mapping):
                candidate_ids.update({info.get("clientOrderId"), info.get("client_order_id")})
            if client_order_id in {str(value) for value in candidate_ids if value}:
                return self._order_result_from_mapping(entry)
        return None

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        params = dict(self._settings.get("cancel_order_params", {}))
        # Dla mutacji preferujemy safety > availability: brak automatycznego retry
        # ogranicza ryzyko zduplikowanego skutku ubocznego przy błędzie niejednoznacznym.
        try:
            self._call_client("cancel_order", order_id, symbol, retry=False, params=params or None)
        except ExchangeAPIError as exc:
            # Minimalna normalizacja idempotentna:
            # jeżeli giełda raportuje, że zlecenie już nie istnieje / jest finalne,
            # traktujemy cancel jako bezpieczny sukces.
            if self._is_idempotent_cancel_error(exc):
                return
            raise

    @staticmethod
    def _is_idempotent_cancel_error(exc: ExchangeAPIError) -> bool:
        if exc.status_code not in {404, 409, 410, 422}:
            return False
        text = " ".join(
            part.lower()
            for part in (str(exc.message or ""), str(exc.payload or ""))
            if part and part != "None"
        )
        if not text:
            return False
        markers = (
            "order not found",
            "unknown order",
            "not open order",
            "already canceled",
            "already cancelled",
            "already closed",
            "already filled",
            "does not exist",
            "no such order",
        )
        return any(marker in text for marker in markers)

    def stream_public_data(self, *, channels: Sequence[str]):
        raise NotImplementedError(
            "Adapter CCXT wspiera wyłącznie snapshoty REST – brak kanałów streamingowych."
        )

    def stream_private_data(self, *, channels: Sequence[str]):
        raise NotImplementedError(
            "Adapter CCXT wspiera wyłącznie snapshoty REST – brak kanałów streamingowych."
        )


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
        network_guard: ExchangeNetworkGuard | None = None,
    ) -> None:
        guard = network_guard
        if guard is None:
            guard = build_exchange_network_guard(
                adapter_name=f"{self.name}:{exchange_id}",
                environment=(environment or credentials.environment or Environment.LIVE).value,
                rate_limit_rules=normalize_rate_limit_rules(
                    (settings or {}).get("rate_limit_rules"),
                    (),
                ),
                default_rules=(RateLimitRule(rate=60, per=60.0),),
                watchdog=watchdog,
                metric_labels={"exchange": self.name},
            )
        self._watchdog = guard.watchdog
        super().__init__(
            credentials,
            exchange_id=exchange_id,
            environment=environment,
            settings=settings,
            client=client,
            metrics_registry=metrics_registry,
            network_guard=guard,
        )

    def _call_client(
        self,
        method_name: str,
        *args: Any,
        retry: bool = True,
        **kwargs: Any,
    ) -> Any:
        operation = f"{self.name}.{method_name}"
        if not retry:
            return super()._call_client(method_name, *args, retry=False, **kwargs)
        return self._watchdog.execute(
            operation,
            lambda: super()._call_client(method_name, *args, retry=True, **kwargs),
        )


__all__ = ["CCXTSpotAdapter", "WatchdogCCXTAdapter", "merge_adapter_settings"]
