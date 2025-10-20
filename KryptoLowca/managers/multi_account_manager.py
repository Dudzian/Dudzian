"""Zarządzanie wieloma kontami przy użyciu LiveExecutionRouter z bot_core."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Deque, Dict, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - zależność opcjonalna w CI
    from bot_core.execution import ExecutionContext, LiveExecutionRouter
    from bot_core.exchanges.base import OrderRequest, OrderResult
except Exception:  # pragma: no cover - fallback gdy moduły nie są dostępne
    ExecutionContext = None  # type: ignore[assignment]
    LiveExecutionRouter = None  # type: ignore[assignment]
    OrderRequest = None  # type: ignore[assignment]
    OrderResult = None  # type: ignore[assignment]

try:  # pragma: no cover - zależność opcjonalna
    from bot_core.market_intel import MarketIntelAggregator, MarketIntelQuery
except Exception:  # pragma: no cover - fallback
    MarketIntelAggregator = None  # type: ignore[assignment]
    MarketIntelQuery = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ManagedAccount:
    """Definicja konta wraz z kontekstem egzekucji i telemetrią."""

    exchange: str
    account: str
    execution_route: str
    context: ExecutionContext
    telemetry_tags: Mapping[str, str] = field(default_factory=dict)


class MultiExchangeAccountManager:
    """Kieruje zlecenia przez :class:`LiveExecutionRouter` w układzie round-robin."""

    def __init__(
        self,
        router: "LiveExecutionRouter",
        *,
        base_context: "ExecutionContext",
        market_intel: Optional["MarketIntelAggregator"] = None,
        telemetry_emitter: Optional[Callable[[str, Mapping[str, object]], None]] = None,
    ) -> None:
        if LiveExecutionRouter is None or ExecutionContext is None or OrderRequest is None:
            raise RuntimeError("bot_core nie jest dostępny – MultiExchangeAccountManager wymaga nowych modułów")

        self._router: LiveExecutionRouter = router
        self._base_context: ExecutionContext = base_context
        self._market_intel = market_intel
        self._telemetry_emitter = telemetry_emitter

        self._accounts: Dict[str, ManagedAccount] = {}
        self._round_robin: Deque[str] = deque()
        self._order_bindings: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ utils
    @property
    def supported_exchanges(self) -> Sequence[str]:
        """Zwraca listę giełd obsługiwanych przez podpięty router."""

        try:
            return tuple(self._router.list_adapters())  # type: ignore[attr-defined]
        except AttributeError:
            # starsze wersje routera nie mają metody list_adapters – sięgamy do prywatnych pól
            adapters = getattr(self._router, "_adapters", {})
            return tuple(sorted(adapters.keys()))

    # ----------------------------------------------------------------- account
    def register_account(
        self,
        *,
        exchange: str,
        account: str,
        execution_route: Optional[str] = None,
        portfolio_id: Optional[str] = None,
        risk_profile: Optional[str] = None,
        environment: Optional[str] = None,
        metadata: Optional[Mapping[str, str]] = None,
        telemetry_tags: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Dodaje konto wraz z kontekstem egzekucji."""

        route = (execution_route or exchange).strip()
        if not route:
            raise ValueError("execution_route nie może być pusty")

        base_meta: MutableMapping[str, str] = dict(self._base_context.metadata or {})
        base_meta.update({"exchange": exchange, "account": account, "execution_route": route})
        if metadata:
            base_meta.update({str(k): str(v) for k, v in metadata.items()})

        context = replace(
            self._base_context,
            portfolio_id=portfolio_id or self._base_context.portfolio_id,
            risk_profile=risk_profile or self._base_context.risk_profile,
            environment=environment or self._base_context.environment,
            metadata=dict(base_meta),
        )

        key = f"{exchange}:{account}"
        managed = ManagedAccount(
            exchange=exchange,
            account=account,
            execution_route=route,
            context=context,
            telemetry_tags=dict(telemetry_tags or {}),
        )

        self._accounts[key] = managed
        if key not in self._round_robin:
            self._round_robin.append(key)

    # ---------------------------------------------------------------- dispatch
    async def dispatch_order(self, order: "OrderRequest") -> "OrderResult":
        """Wysyła zlecenie korzystając z kolejnego konta w kolejce."""

        key = await self._choose_account()
        managed = self._accounts[key]
        context = self._context_with_route(managed)
        result = await asyncio.to_thread(self._router.execute, order, context)
        self._order_bindings[result.order_id] = key
        self._emit_telemetry(managed, result)
        return result

    async def cancel_order(self, order_id: str) -> None:
        """Anuluje zlecenie przy użyciu zapamiętanego kontekstu."""

        key = self._order_bindings.get(order_id)
        context = self._context_with_route(self._accounts[key]) if key else self._base_context
        await asyncio.to_thread(self._router.cancel, order_id, context)
        if key:
            self._order_bindings.pop(order_id, None)

    # ------------------------------------------------------------ market intel
    def collect_market_intel(
        self,
        symbol: str,
        *,
        interval: str = "1h",
        lookback_bars: int = 24,
    ) -> Optional[object]:
        """Buduje snapshot market intel dla podanego instrumentu."""

        if self._market_intel is None or MarketIntelQuery is None:
            return None
        try:
            query = MarketIntelQuery(symbol=symbol, interval=interval, lookback_bars=lookback_bars)
            return self._market_intel.build_snapshot(query)
        except Exception:
            logger.debug("Nie udało się zbudować market intel dla %s", symbol, exc_info=True)
            return None

    # ----------------------------------------------------------------- helpers
    async def _choose_account(self) -> str:
        async with self._lock:
            if not self._round_robin:
                raise RuntimeError("Brak zarejestrowanych kont w MultiExchangeAccountManager")
            key = self._round_robin[0]
            self._round_robin.rotate(-1)
            return key

    @staticmethod
    def _context_with_route(managed: ManagedAccount) -> ExecutionContext:
        meta = dict(managed.context.metadata or {})
        meta.setdefault("execution_route", managed.execution_route)
        return replace(managed.context, metadata=meta)

    def _emit_telemetry(self, managed: ManagedAccount, result: "OrderResult") -> None:
        if not callable(self._telemetry_emitter):
            return
        payload = {
            "exchange": managed.exchange,
            "account": managed.account,
            "order_status": result.status,
            "filled_qty": float(result.filled_quantity),
        }
        payload.update({str(k): str(v) for k, v in managed.telemetry_tags.items()})
        try:
            self._telemetry_emitter("multi_account.execution", payload)
        except Exception:
            logger.debug("Telemetry emitter zgłosił wyjątek", exc_info=True)


__all__ = ["MultiExchangeAccountManager", "ManagedAccount"]

