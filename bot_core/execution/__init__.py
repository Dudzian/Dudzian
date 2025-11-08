"""Moduł egzekucji zleceń."""

from bot_core.execution.base import ExecutionContext, ExecutionService, PriceResolver, RetryPolicy

# Zgodność wstecz/naprzód: obsłuż zarówno RoutingPlan (nowa nazwa),
# jak i RouteDefinition (starsza nazwa) z live_router.
try:
    # nowsza gałąź
    from bot_core.execution.live_router import (  # noqa: F401
        LiveExecutionRouter,
        QoSConfig,
        RouterRuntimeStats,
        RoutingPlan,
    )
    RouteDefinition = RoutingPlan  # alias dla kompatybilności
except ImportError:
    # starsza gałąź
    from bot_core.execution.live_router import LiveExecutionRouter, RouteDefinition  # type: ignore  # noqa: F401
    QoSConfig = object  # type: ignore  # noqa: N816 - brak w starszych gałęziach
    RoutingPlan = RouteDefinition  # alias ujednolicający API
    RouterRuntimeStats = object  # type: ignore

from bot_core.execution.bridge import (  # noqa: F401 - eksport publiczny
    ExchangeAdapterExecutionService,
    decision_to_order_request,
)
from bot_core.execution.paper import (  # noqa: F401 - eksport publiczny
    InsufficientBalanceError,
    LedgerEntry,
    MarketMetadata,
    PaperTradingExecutionService,
    ShortPosition,
)
from bot_core.execution.execution_service import (  # noqa: F401 - eksport publiczny
    build_live_execution_service,
    resolve_execution_mode,
)

__all__ = [
    "ExecutionContext",
    "ExecutionService",
    "PriceResolver",
    "RetryPolicy",
    "LiveExecutionRouter",
    "RoutingPlan",
    "RouteDefinition",
    "QoSConfig",
    "RouterRuntimeStats",
    "PaperTradingExecutionService",
    "MarketMetadata",
    "LedgerEntry",
    "InsufficientBalanceError",
    "ShortPosition",
    "ExchangeAdapterExecutionService",
    "decision_to_order_request",
    "resolve_execution_mode",
    "build_live_execution_service",
]
