# -*- coding: utf-8 -*-
"""Lekki wrapper zapewniający kompatybilność z historycznym API."""
from __future__ import annotations

import asyncio
import inspect
import logging
import math
import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - importowany tylko jeśli ccxt jest dostępne
    import ccxt.async_support as ccxt_async  # type: ignore
except Exception:  # pragma: no cover - starsze wersje ccxt
    try:
        import ccxt.asyncio as ccxt_async  # type: ignore
    except Exception:  # pragma: no cover - brak ccxt
        ccxt_async = types.ModuleType("ccxt.asyncio")  # type: ignore
        ccxt_module = sys.modules.setdefault("ccxt", types.ModuleType("ccxt"))
        setattr(ccxt_module, "asyncio", ccxt_async)
        sys.modules["ccxt.asyncio"] = ccxt_async

from managers.exchange_core import Mode

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
            from ccxt.base.errors import AuthenticationError as CCXTAuthError  # type: ignore

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
        manager.mode = getattr(config, "mode", Mode.PAPER)
        if isinstance(manager.mode, str):
            try:
                manager.mode = Mode(manager.mode)
            except ValueError:
                pass
        return manager

    # --------------------------- operacje pomocnicze ---------------------------
    async def _run_with_retry(self, coro_factory) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self._retry_attempts + 1):
            try:
                return await coro_factory()
            except Exception as exc:
                last_exc = exc
                if attempt == self._retry_attempts:
                    raise
                await asyncio.sleep(self._retry_delay)
        if last_exc:
            raise last_exc
        return None

    async def _maybe_await(self, result: Any) -> Any:
        if inspect.isawaitable(result):
            return await result
        return result

    def _market(self, symbol: str) -> Dict[str, Any]:
        if symbol in self._markets:
            return self._markets[symbol]
        if symbol.upper() in self._markets:
            return self._markets[symbol.upper()]
        return {}

    # --------------------------------- API ------------------------------------
    async def load_markets(self) -> Dict[str, Any]:
        markets = await self.exchange.load_markets()
        self._markets = markets or {}
        return markets

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Iterable[Any]:
        if not symbol:
            raise ValueError("Symbol nie może być pusty.")
        if limit <= 0:
            raise ValueError("Limit musi być dodatni.")
        async def _call() -> Any:
            return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        return await self._run_with_retry(_call)

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        if side.lower() not in {"buy", "sell"}:
            raise ValueError("Dozwolone strony zlecenia: buy/sell.")
        if quantity <= 0:
            raise ValueError("Ilość musi być dodatnia.")
        raw = await self.exchange.create_market_order(symbol, side, quantity)
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
        async def _call() -> Any:
            return await self.exchange.fetch_balance()

        data = await self._run_with_retry(_call)
        if isinstance(data, dict) and "total" in data:
            return data["total"]
        return data

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        async def _call():
            if symbol:
                return await self.exchange.fetch_open_orders(symbol)
            return await self.exchange.fetch_open_orders()

        rows = await self._run_with_retry(_call)
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
        result = await self.exchange.cancel_order(order_id, symbol)
        return bool(result) if result is not None else True

    async def close(self) -> None:
        close = getattr(self.exchange, "close", None)
        if close:
            await close()

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
        ticker = await self._maybe_await(ticker_fn(symbol))
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
                    return await self._maybe_await(
                        method(symbol, "market", side, quantity, None, params)
                    )
                except TypeError:
                    return await self._maybe_await(
                        method(symbol, "market", side, quantity, None)
                    )

            market_method = getattr(self.exchange, "create_market_order", None)
            if market_method is None:
                raise ExchangeError("Exchange does not provide create_order API")
            try:
                return await self._maybe_await(
                    market_method(symbol, side, quantity, params)
                )
            except TypeError:
                return await self._maybe_await(market_method(symbol, side, quantity))

        method = getattr(self.exchange, "create_order", None)
        if callable(method):
            try:
                return await self._maybe_await(
                    method(symbol, order_type, side, quantity, price, params)
                )
            except TypeError:
                return await self._maybe_await(
                    method(symbol, order_type, side, quantity, price)
                )

        limit_method = getattr(self.exchange, "create_limit_order", None)
        if limit_method is None:
            raise ExchangeError("Exchange does not provide limit order API")
        try:
            return await self._maybe_await(
                limit_method(symbol, side, quantity, price, params)
            )
        except TypeError:
            return await self._maybe_await(limit_method(symbol, side, quantity, price))
