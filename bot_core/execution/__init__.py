"""Moduł egzekucji zleceń."""

from bot_core.execution.base import ExecutionContext, ExecutionService, RetryPolicy

# Zgodność wstecz/naprzód: obsłuż zarówno RoutingPlan (main), jak i RouteDefinition (stara nazwa)
try:
    # gałąź main
    from bot_core.execution.live_router import LiveExecutionRouter, RoutingPlan
    RouteDefinition = RoutingPlan  # alias dla kompatybilności
except ImportError:
    # gałąź z wcześniejszą nazwą
    from bot_core.execution.live_router import LiveExecutionRouter, RouteDefinition  # type: ignore
    RoutingPlan = RouteDefinition  # alias do ujednolicenia API

from bot_core.execution.paper import (  # noqa: F401 - eksport publiczny
    InsufficientBalanceError,
    LedgerEntry,
    MarketMetadata,
    PaperTradingExecutionService,
    ShortPosition,
)

__all__ = [
    "ExecutionContext",
    "ExecutionService",
    "RetryPolicy",
    "LiveExecutionRouter",
    "RoutingPlan",
    "RouteDefinition",
    "PaperTradingExecutionService",
    "MarketMetadata",
    "LedgerEntry",
    "InsufficientBalanceError",
    "ShortPosition",
]
