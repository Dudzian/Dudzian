# -*- coding: utf-8 -*-
"""Lekki wrapper zapewniający kompatybilność z historycznym API."""
from __future__ import annotations

import asyncio
import inspect
import logging
import math
import sys
import time
import types
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

# ---- ccxt async importy odporne na różne wersje / brak biblioteki ----
try:  # pragma: no cover - importowany tylko jeśli ccxt jest dostępne
    import ccxt.async_support as ccxt_async  # type: ignore
except Exception:  # pragma: no cover - starsze wersje ccxt
    try:
        import ccxt.asyncio as ccxt_async  # type: ignore
    except Exception:  # pragma: no cover - brak ccxt
        # Utwórz atrapę modułu, aby testy mogły się odwoływać do ccxt.asyncio
        ccxt_async = types.ModuleType("ccxt.asyncio")  # type: ignore
        ccxt_module = sys.modules.setdefault("ccxt", types.ModuleType("ccxt"))
        setattr(ccxt_module, "asyncio", ccxt_async)
        sys.modules["ccxt.asyncio"] = ccxt_async

# ---- odporny import Mode ----
try:  # pragma: no cover
    from KryptoLowca.managers.exchange_core import Mode  # type: ignore
except Exception:  # pragma: no cover
    from managers.exchange_core import Mode  # type: ignore

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

__all__ = [
    "ExchangeManager",
    "ExchangeError",
    "AuthenticationError",
    "OrderResult",
]


@dataclass(slots=True)
class _EndpointMetrics:
    total_calls: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_latency_ms: float = 0.0


@dataclass(slots=True)
class _APIMetrics:
    total_calls: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    consecutive_errors: int = 0
    window_calls: int = 0
    window_errors: int = 0
    last_endpoint: Optional[str] = None


class ExchangeError(RuntimeError):
    """Podstawowy wyjątek warstwy wymiany."""


class AuthenticationError(ExchangeError):
    """Błąd uwierzytelniania API giełdy."""


@dataclass(slots=True)
class OrderResult:
    """Minimalna struktura opisująca wynik zlecenia."""

    id: Any
    symbol: str
    side: str
    qty: float
    price: Optional[float]
    status: str


class ExchangeManager:
    """Asynchroniczny wrapper wykorzystywany w testach jednostkowych."""

    def __init__(self, exchange, *, user_id: Optional[int] = None) -> None:
        self.exchange = exchange
        self._user_id = user_id
        self._retry_attempts = 1
        self._retry_delay = 0.05
        self.mode = Mode.PAPER
        self._markets: Dict[str, Dict[str, Any]] = {}

        # Telemetria / alerty / throttling
        self._db_manager: Optional[Any] = None
        self._alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

        self._metrics = _APIMetrics()
        self._endpoint_metrics: Dict[str, _EndpointMetrics] = defaultdict(lambda: _EndpointMetrics())
        self._throttle_lock = asyncio.Lock()
        self._min_interval = 0.0
        self._last_request_ts = 0.0
        self._window_start = time.monotonic()
        self._window_count = 0
        self._rate_limit_window = 60.0
        self._rate_limit_per_minute: Optional[int] = None
        self._max_calls_per_window: Optional[int] = None
        self._alert_usage_threshold = 0.85
        self._rate_alert_active = False
        self._alert_cooldown_seconds = 5.0
        self._alert_last: Dict[str, float] = {}
        self._error_alert_threshold = 3

    @classmethod
    async def create(
        cls,
        config,
        db_manager: Optional[Any] = None,
        security_manager: Optional[Any] = None,
    ) -> "ExchangeManager":
        if ccxt_async is None:
            raise ExchangeError("Biblioteka ccxt nie jest dostępna w trybie asynchronicznym.")

        exchange_id = getattr(config, "exchange_name", "binance")
        try:
            exchange_cls = getattr(ccxt_async, exchange_id)
        except AttributeError as exc:  # pragma: no cover - nieznana giełda
            raise ExchangeError(f"Nieobsługiwana giełda: {exchange_id}") from exc

        kwargs = {
            "apiKey": getattr(config, "api_key", ""),
            "secret": getattr(config, "api_secret", ""),
            "enableRateLimit": True,
        }
        try:
            exchange = exchange_cls(kwargs)
        except Exception as exc:
            try:
                from ccxt.base.errors import AuthenticationError as CCXTAuthError  # type: ignore
            except Exception:  # pragma: no cover
                CCXTAuthError = type("AuthenticationError", (Exception,), {})  # type: ignore
            if isinstance(exc, CCXTAuthError):
                raise AuthenticationError(str(exc)) from exc
            raise ExchangeError(str(exc)) from exc

        user_id = None
        if db_manager is not None:
            try:
                user_id = await db_manager.ensure_user("system@bot")
            except Exception:
                logger.warning("Nie udało się utworzyć użytkownika w bazie – kontynuuję bez ID.")

        manager = cls(exchange, user_id=user_id)
        manager._db_manager = db_manager
        manager.mode = getattr(config, "mode", Mode.PAPER)
        if isinstance(manager.mode, str):
            try:
                manager.mode = Mode(manager.mode)
            except ValueError:
                pass

        # Konfiguracja limitów/alertów z configu
        per_minute = max(0, int(getattr(config, "rate_limit_per_minute", 0) or 0))
        window_seconds = float(getattr(config, "rate_limit_window_seconds", 60.0) or 60.0)
        manager._rate_limit_window = max(0.1, window_seconds)
        manager._rate_limit_per_minute = per_minute or None
        if per_minute > 0:
            calls_per_window = per_minute * (manager._rate_limit_window / 60.0)
            manager._max_calls_per_window = max(1, int(math.floor(calls_per_window)))
        manager._alert_usage_threshold = max(
            0.1,
            min(1.0, float(getattr(config, "rate_limit_alert_threshold", 0.85) or 0.85)),
        )
        manager._error_alert_threshold = max(1, int(getattr(config, "error_alert_threshold", 3) or 3))

        rate_limit_ms = getattr(exchange, "rateLimit", None)
        try:
            manager._min_interval = max(0.0, float(rate_limit_ms) / 1000.0) if rate_limit_ms else 0.0
        except (TypeError, ValueError):  # pragma: no cover
            manager._min_interval = 0.0
        manager._window_start = time.monotonic()
        manager._window_count = 0
        return manager

    # --------------------------- operacje pomocnicze ---------------------------
    async def _before_call(self, endpoint: str) -> None:
        wait_time = 0.0
        should_alert = False
        usage_pct = 0.0
        async with self._throttle_lock:
            now = time.monotonic()
            if now - self._window_start >= self._rate_limit_window:
                self._window_start = now
                self._window_count = 0
                self._metrics.window_calls = 0
                self._metrics.window_errors = 0
                self._rate_alert_active = False
            if self._max_calls_per_window:
                usage_pct = (self._window_count + 1) / self._max_calls_per_window
                if self._window_count >= self._max_calls_per_window:
                    wait_time = max(wait_time, self._rate_limit_window - (now - self._window_start))
                if usage_pct >= self._alert_usage_threshold and not self._rate_alert_active:
                    should_alert = True
                    self._rate_alert_active = True
            if self._min_interval > 0:
                delta = now - self._last_request_ts
                if delta < self._min_interval:
                    wait_time = max(wait_time, self._min_interval - delta)
            self._window_count += 1
            self._metrics.window_calls = self._window_count
            self._last_request_ts = now + wait_time if wait_time > 0 else now
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        if should_alert and self._max_calls_per_window:
            remaining = max(0.0, self._rate_limit_window - (time.monotonic() - self._window_start))
            context = {
                "endpoint": endpoint,
                "usage": usage_pct,
                "calls_in_window": self._window_count,
                "max_calls": self._max_calls_per_window,
                "window_seconds": self._rate_limit_window,
                "reset_in_seconds": remaining,
            }
            self._raise_alert(
                f"Zużyto {usage_pct * 100:.1f}% dostępnego limitu API w bieżącym oknie.",
                context=context,
                key="rate_limit",
            )

    def _record_metrics(self, endpoint: str, latency_ms: float, *, success: bool) -> None:
        metrics = self._metrics
        metrics.total_calls += 1
        metrics.avg_latency_ms = (
            ((metrics.avg_latency_ms * (metrics.total_calls - 1)) + latency_ms) / metrics.total_calls
        )
        metrics.max_latency_ms = max(metrics.max_latency_ms, latency_ms)
        metrics.last_latency_ms = latency_ms
        metrics.last_endpoint = endpoint

        endpoint_metrics = self._endpoint_metrics[endpoint]
        endpoint_metrics.total_calls += 1
        endpoint_metrics.avg_latency_ms = (
            (endpoint_metrics.avg_latency_ms * (endpoint_metrics.total_calls - 1) + latency_ms)
            / endpoint_metrics.total_calls
        )
        endpoint_metrics.max_latency_ms = max(endpoint_metrics.max_latency_ms, latency_ms)
        endpoint_metrics.last_latency_ms = latency_ms

        if success:
            metrics.consecutive_errors = 0
        else:
            metrics.total_errors += 1
            metrics.window_errors += 1
            metrics.consecutive_errors += 1
            endpoint_metrics.total_errors += 1
            if (
                self._error_alert_threshold
                and metrics.consecutive_errors == self._error_alert_threshold
            ):
                context = {
                    "endpoint": endpoint,
                    "consecutive_errors": metrics.consecutive_errors,
                }
                self._raise_alert(
                    f"Przekroczono próg błędów API ({metrics.consecutive_errors}) dla {endpoint}",
                    context=context,
                    key="error",
                )

    def _register_error(self, endpoint: str, exc: Exception, *, final: bool) -> None:
        level = logging.ERROR if final else logging.WARNING
        logger.log(level, "Wywołanie %s nie powiodło się: %s", endpoint, exc)
        if final and self._db_manager and self._user_id:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover - brak aktywnej pętli
                pass
            else:
                loop.create_task(
                    self._db_manager.log(
                        self._user_id,
                        "ERROR",
                        f"Wywołanie {endpoint} zakończyło się błędem: {exc}",
                        category="exchange",
                    )
                )

    def _raise_alert(self, message: str, context: Optional[Dict[str, Any]] = None, *, key: str) -> None:
        now = time.monotonic()
        last = self._alert_last.get(key, 0.0)
        if now - last < self._alert_cooldown_seconds:
            return
        self._alert_last[key] = now
        logger.critical("[ALERT] %s | context=%s", message, context or {})
        if self._alert_callback:
            try:
                self._alert_callback(message, context or {})
            except Exception:  # pragma: no cover - nie przerywamy głównego przepływu
                logger.exception("Alert callback zgłosił wyjątek")
        if self._db_manager and self._user_id:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover
                return
            loop.create_task(
                self._db_manager.log(
                    self._user_id,
                    "CRITICAL",
                    message,
                    category="exchange",
                    context=context,
                )
            )

    async def _run_with_retry(self, coro_factory: Callable[[], Any], *, endpoint: str) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self._retry_attempts + 1):
            await self._before_call(endpoint)
            start = time.perf_counter()
            try:
                result = await coro_factory()
            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000.0
                self._record_metrics(endpoint, latency_ms, success=False)
                final = attempt == self._retry_attempts
                self._register_error(endpoint, exc, final=final)
                last_exc = exc
                if final:
                    raise
                await asyncio.sleep(self._retry_delay)
            else:
                latency_ms = (time.perf_counter() - start) * 1000.0
                self._record_metrics(endpoint, latency_ms, success=True)
                return result
        if last_exc:
            raise last_exc
        return None

    async def _call_with_metrics(self, endpoint: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        async def _factory() -> Any:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        return await self._run_with_retry(_factory, endpoint=endpoint)

    def _market(self, symbol: str) -> Dict[str, Any]:
        if symbol in self._markets:
            return self._markets[symbol]
        if symbol.upper() in self._markets:
            return self._markets[symbol.upper()]
        return {}

    # --------------------------------- API ------------------------------------
    async def load_markets(self) -> Dict[str, Any]:
        markets = await self._call_with_metrics("load_markets", self.exchange.load_markets)
        self._markets = markets or {}
        return markets

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Iterable[Any]:
        if not symbol:
            raise ValueError("Symbol nie może być pusty.")
        if limit <= 0:
            raise ValueError("Limit musi być dodatni.")
        return await self._call_with_metrics(
            "fetch_ohlcv", self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit
        )

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        if side.lower() not in {"buy", "sell"}:
            raise ValueError("Dozwolone strony zlecenia: buy/sell.")
        if quantity <= 0:
            raise ValueError("Ilość musi być dodatnia.")
        raw = await self._call_with_metrics(
            "create_market_order", self.exchange.create_market_order, symbol, side, quantity
        )
        return OrderResult(
            id=raw.get("id"),
            symbol=raw.get("symbol", symbol),
            side=raw.get("side", side),
            qty=float(raw.get("amount", quantity)),
            price=raw.get("price"),
            status=str(raw.get("status", "filled")).lower(),
        )

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> OrderResult:
        if side.lower() not in {"buy", "sell"}:
            raise ValueError("Dozwolone strony zlecenia: buy/sell.")
        if quantity <= 0:
            raise ValueError("Ilość musi być dodatnia.")

        qty = self.quantize_amount(symbol, quantity)
        if qty <= 0:
            raise ValueError("Skwantowana ilość wynosi 0.")

        order_type = order_type.lower()
        px: Optional[float] = None
        if order_type != "market":
            if price is None:
                raise ValueError("Cena wymagana dla zleceń LIMIT/STOP.")
            px = self.quantize_price(symbol, float(price))
        else:
            px = float(price) if price is not None else None

        params = dict(params or {})
        if client_order_id:
            params.setdefault("newClientOrderId", client_order_id)
            params.setdefault("clientOrderId", client_order_id)

        notional_price = px if px is not None else price
        if order_type == "market" and notional_price is None:
            last_price = await self._last_price(symbol)
            notional_price = last_price
        min_notional = self.min_notional(symbol)
        if min_notional and notional_price:
            if qty * float(notional_price) < min_notional - 1e-9:
                raise ValueError("Wartość zlecenia poniżej min_notional")

        response = await self._submit_order(
            symbol, side.lower(), order_type, qty, px, params or {}, client_order_id
        )
        amount = float(response.get("amount") or response.get("quantity") or qty)
        filled_price = response.get("price") or px or notional_price or 0.0
        status = str(response.get("status") or "open").lower()
        order_id = response.get("id") or response.get("orderId") or response.get("order_id")
        return OrderResult(
            id=order_id,
            symbol=response.get("symbol", symbol),
            side=response.get("side", side),
            qty=amount,
            price=float(filled_price) if filled_price is not None else None,
            status=status,
        )

    async def fetch_balance(self) -> Dict[str, Any]:
        data = await self._call_with_metrics("fetch_balance", self.exchange.fetch_balance)
        if isinstance(data, dict) and "total" in data:
            return data["total"]
        return data

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        if symbol:
            rows = await self._call_with_metrics(
                "fetch_open_orders", self.exchange.fetch_open_orders, symbol
            )
        else:
            rows = await self._call_with_metrics("fetch_open_orders", self.exchange.fetch_open_orders)
        results: List[OrderResult] = []
        for row in rows or []:
            results.append(
                OrderResult(
                    id=row.get("id"),
                    symbol=row.get("symbol", symbol or ""),
                    side=row.get("side", ""),
                    qty=float(row.get("amount", 0.0)),
                    price=row.get("price"),
                    status=str(row.get("status", "open")).lower(),
                )
            )
        return results

    async def cancel_order(self, order_id: Any, symbol: str) -> bool:
        if not order_id or not symbol:
            raise ValueError("Identyfikator zlecenia i symbol są wymagane.")
        result = await self._call_with_metrics(
            "cancel_order", self.exchange.cancel_order, order_id, symbol
        )
        return bool(result) if result is not None else True

    async def close(self) -> None:
        close = getattr(self.exchange, "close", None)
        if close:
            maybe = close()
            if inspect.isawaitable(maybe):
                await maybe

    # ------------------------------- helpers ---------------------------------
    def quantize_amount(self, symbol: str, amount: float) -> float:
        market = self._market(symbol)
        value = float(amount)
        limits = market.get("limits") or {}
        precision = market.get("precision") or {}
        step = (limits.get("amount") or {}).get("step")
        if step:
            step_val = float(step)
            if step_val > 0:
                value = math.floor(value / step_val) * step_val
        elif precision.get("amount") is not None:
            value = round(value, int(precision["amount"]))
        return max(value, 0.0)

    def quantize_price(self, symbol: str, price: float) -> float:
        market = self._market(symbol)
        value = float(price)
        limits = market.get("limits") or {}
        precision = market.get("precision") or {}
        step = (limits.get("price") or {}).get("step")
        if step:
            step_val = float(step)
            if step_val > 0:
                value = math.floor(value / step_val) * step_val
        elif precision.get("price") is not None:
            value = round(value, int(precision["price"]))
        return max(value, 0.0)

    def min_notional(self, symbol: str) -> float:
        market = self._market(symbol)
        limits = market.get("limits") or {}
        cost = limits.get("cost") or {}
        value = cost.get("min")
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):  # pragma: no cover
            return 0.0

    async def _last_price(self, symbol: str) -> Optional[float]:
        ticker_fn = getattr(self.exchange, "fetch_ticker", None)
        if not callable(ticker_fn):
            return None
        try:
            ticker = await self._call_with_metrics("fetch_ticker", ticker_fn, symbol)
        except Exception:
            return None
        if not isinstance(ticker, dict):
            return None
        for key in ("last", "close", "bid", "ask"):
            value = ticker.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    async def _submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        params: Dict[str, Any],
        client_order_id: Optional[str],
    ) -> Dict[str, Any]:
        if order_type == "market":
            method = getattr(self.exchange, "create_order", None)
            if callable(method):
                try:
                    return await self._call_with_metrics(
                        "create_order",
                        method,
                        symbol,
                        "market",
                        side,
                        quantity,
                        None,
                        params,
                    )
                except TypeError:
                    return await self._call_with_metrics(
                        "create_order",
                        method,
                        symbol,
                        "market",
                        side,
                        quantity,
                        None,
                    )

            market_method = getattr(self.exchange, "create_market_order", None)
            if market_method is None:
                raise ExchangeError("Exchange does not provide create_order API")
            try:
                return await self._call_with_metrics(
                    "create_market_order", market_method, symbol, side, quantity, params
                )
            except TypeError:
                return await self._call_with_metrics(
                    "create_market_order", market_method, symbol, side, quantity
                )

        method = getattr(self.exchange, "create_order", None)
        if callable(method):
            try:
                return await self._call_with_metrics(
                    "create_order",
                    method,
                    symbol,
                    order_type,
                    side,
                    quantity,
                    price,
                    params,
                )
            except TypeError:
                return await self._call_with_metrics(
                    "create_order",
                    method,
                    symbol,
                    order_type,
                    side,
                    quantity,
                    price,
                )

        limit_method = getattr(self.exchange, "create_limit_order", None)
        if limit_method is None:
            raise ExchangeError("Exchange does not provide limit order API")
        try:
            return await self._call_with_metrics(
                "create_limit_order", limit_method, symbol, side, quantity, price, params
            )
        except TypeError:
            return await self._call_with_metrics(
                "create_limit_order", limit_method, symbol, side, quantity, price
            )

    # ------------------------------ telemetry ---------------------------------
    def register_alert_handler(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Zarejestruj zewnętrzny handler alertów (np. GUI / moduł powiadomień)."""
        self._alert_callback = callback

    def get_api_metrics(self) -> Dict[str, Any]:
        """Zwróć metryki zużycia API (łącznie i per-endpoint)."""
        usage = None
        if self._max_calls_per_window:
            usage = self._window_count / self._max_calls_per_window if self._max_calls_per_window else None
        return {
            "total_calls": self._metrics.total_calls,
            "total_errors": self._metrics.total_errors,
            "avg_latency_ms": self._metrics.avg_latency_ms,
            "max_latency_ms": self._metrics.max_latency_ms,
            "last_latency_ms": self._metrics.last_latency_ms,
            "consecutive_errors": self._metrics.consecutive_errors,
            "window_calls": self._metrics.window_calls,
            "window_errors": self._metrics.window_errors,
            "last_endpoint": self._metrics.last_endpoint,
            "rate_limit_per_minute": self._rate_limit_per_minute,
            "rate_limit_window_seconds": self._rate_limit_window,
            "current_window_usage": usage,
            "endpoints": {name: asdict(data) for name, data in self._endpoint_metrics.items()},
        }

    def reset_api_metrics(self) -> None:
        """Wyzeruj liczniki metryk (np. po raporcie)."""
        self._metrics = _APIMetrics()
        self._endpoint_metrics = defaultdict(lambda: _EndpointMetrics())
        self._window_count = 0
        self._metrics.window_calls = 0
        self._metrics.window_errors = 0
        self._metrics.consecutive_errors = 0
        self._window_start = time.monotonic()
        self._rate_alert_active = False
        self._alert_last.clear()
