"""Pakiet adapterów giełdowych."""

from __future__ import annotations

import os

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
    ExchangeError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)

__all__ = [
    "AccountSnapshot",
    "Environment",
    "ExchangeAdapter",
    "ExchangeCredentials",
    "OrderRequest",
    "OrderResult",
    "ExchangeError",
    "ExchangeAPIError",
    "ExchangeAuthError",
    "ExchangeThrottlingError",
    "ExchangeNetworkError",
]

if os.environ.get("BOT_CORE_MINIMAL_EXCHANGES") != "1":  # pragma: no cover - pełny import dla runtime
    from bot_core.exchanges.core import (  # noqa: WPS433
        BaseBackend,
        Event,
        EventBus,
        MarketRules,
        Mode,
        OrderDTO,
        OrderSide,
        OrderStatus,
        OrderType,
        PaperBackend,
        PositionDTO,
    )
    from bot_core.exchanges.manager import ExchangeManager  # noqa: WPS433
    from bot_core.exchanges.binance.futures import BinanceFuturesAdapter  # noqa: WPS433
    from bot_core.exchanges.binance.margin import BinanceMarginAdapter  # noqa: WPS433
    from bot_core.exchanges.binance.spot import BinanceSpotAdapter  # noqa: WPS433
    from bot_core.exchanges.bitfinex.spot import BitfinexSpotAdapter  # noqa: WPS433
    from bot_core.exchanges.bybit.spot import BybitSpotAdapter  # noqa: WPS433
    from bot_core.exchanges.coinbase.spot import CoinbaseSpotAdapter  # noqa: WPS433
    from bot_core.exchanges.kucoin.spot import KuCoinSpotAdapter  # noqa: WPS433
    from bot_core.exchanges.kraken.futures import KrakenFuturesAdapter  # noqa: WPS433
    from bot_core.exchanges.kraken.margin import KrakenMarginAdapter  # noqa: WPS433
    from bot_core.exchanges.kraken.spot import KrakenSpotAdapter  # noqa: WPS433
    from bot_core.exchanges.nowa_gielda.spot import NowaGieldaSpotAdapter  # noqa: WPS433
    from bot_core.exchanges.okx.spot import OKXSpotAdapter  # noqa: WPS433
    from bot_core.exchanges.zonda.margin import ZondaMarginAdapter  # noqa: WPS433
    from bot_core.exchanges.zonda.spot import ZondaSpotAdapter  # noqa: WPS433
    from bot_core.exchanges.health import (  # noqa: WPS433
        CircuitBreaker,
        CircuitOpenError,
        HealthCheck,
        HealthCheckResult,
        HealthMonitor,
        HealthStatus,
        RetryPolicy,
        Watchdog,
    )

    __all__ += [
        "BaseBackend",
        "Event",
        "EventBus",
        "MarketRules",
        "Mode",
        "OrderDTO",
        "OrderSide",
        "OrderStatus",
        "OrderType",
        "PaperBackend",
        "PositionDTO",
        "ExchangeManager",
        "BinanceFuturesAdapter",
        "BinanceSpotAdapter",
        "BinanceMarginAdapter",
        "BitfinexSpotAdapter",
        "BybitSpotAdapter",
        "CoinbaseSpotAdapter",
        "KuCoinSpotAdapter",
        "KrakenFuturesAdapter",
        "KrakenSpotAdapter",
        "KrakenMarginAdapter",
        "NowaGieldaSpotAdapter",
        "OKXSpotAdapter",
        "ZondaSpotAdapter",
        "ZondaMarginAdapter",
        "CircuitBreaker",
        "CircuitOpenError",
        "HealthCheck",
        "HealthCheckResult",
        "HealthMonitor",
        "HealthStatus",
        "RetryPolicy",
        "Watchdog",
    ]
