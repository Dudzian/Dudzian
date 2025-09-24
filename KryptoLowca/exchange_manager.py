# -*- coding: utf-8 -*-
"""Lekki wrapper zapewniający kompatybilność z historycznym API."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - importowany tylko jeśli ccxt jest dostępne
    import ccxt.async_support as ccxt_async  # type: ignore
except Exception:  # pragma: no cover - starsze wersje ccxt
    try:
        import ccxt.asyncio as ccxt_async  # type: ignore
    except Exception:  # pragma: no cover - brak ccxt
        ccxt_async = None  # type: ignore

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

        return cls(exchange, user_id=user_id)

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

    # --------------------------------- API ------------------------------------
    async def load_markets(self) -> Dict[str, Any]:
        return await self.exchange.load_markets()

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Iterable[Any]:
        if not symbol:
            raise ValueError("Symbol nie może być pusty.")
        if limit <= 0:
            raise ValueError("Limit musi być dodatni.")
        return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

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

    async def fetch_balance(self) -> Dict[str, Any]:
        data = await self.exchange.fetch_balance()
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
