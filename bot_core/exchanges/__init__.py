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
from bot_core.exchanges.core import (
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
try:  # pragma: no cover - środowisko testowe może nie zawierać pełnej konfiguracji managera
    from bot_core.exchanges.manager import (
        ExchangeManager,
        NativeAdapterInfo,
        iter_registered_native_adapters,
        register_native_adapter,
        reload_native_adapters,
    )
except Exception:  # pragma: no cover - zapewniamy import pakietu
    ExchangeManager = None  # type: ignore[assignment]
    register_native_adapter = None  # type: ignore[assignment]
    iter_registered_native_adapters = None  # type: ignore[assignment]
    reload_native_adapters = None  # type: ignore[assignment]
    NativeAdapterInfo = None  # type: ignore[assignment]
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.margin import BinanceMarginAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter
from bot_core.exchanges.bitfinex.spot import BitfinexSpotAdapter
from bot_core.exchanges.bybit import BybitFuturesAdapter, BybitMarginAdapter, BybitSpotAdapter
from bot_core.exchanges.coinbase import (
    CoinbaseFuturesAdapter,
    CoinbaseMarginAdapter,
    CoinbaseSpotAdapter,
)
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.kucoin.spot import KuCoinSpotAdapter
from bot_core.exchanges.kraken.futures import KrakenFuturesAdapter
from bot_core.exchanges.kraken.margin import KrakenMarginAdapter
from bot_core.exchanges.kraken.spot import KrakenSpotAdapter
from bot_core.exchanges.nowa_gielda.spot import NowaGieldaSpotAdapter
from bot_core.exchanges.okx import OKXFuturesAdapter, OKXMarginAdapter, OKXSpotAdapter
from bot_core.exchanges.zonda.margin import ZondaMarginAdapter
from bot_core.exchanges.zonda.spot import ZondaSpotAdapter
from bot_core.exchanges.health import (
    CircuitBreaker,
    CircuitOpenError,
    HealthCheck,
    HealthCheckResult,
    HealthMonitor,
    HealthStatus,
    RetryPolicy,
    Watchdog,
)
from . import streaming

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
    "KrakenFuturesAdapter",
    "KrakenSpotAdapter",
    "KrakenMarginAdapter",
    "BybitSpotAdapter",
    "BybitMarginAdapter",
    "BybitFuturesAdapter",
    "CoinbaseSpotAdapter",
    "CoinbaseMarginAdapter",
    "CoinbaseFuturesAdapter",
    "BitfinexSpotAdapter",
    "KuCoinSpotAdapter",
    "OKXSpotAdapter",
    "OKXMarginAdapter",
    "OKXFuturesAdapter",
    "NowaGieldaSpotAdapter",
    "MarketRules",
    "Mode",
    "OrderDTO",
    "OrderRequest",
    "OrderResult",
    "OrderSide",
    "OrderStatus",
    "OrderType",
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
    "PaperBackend",
    "PositionDTO",
    "streaming",
    "register_native_adapter",
    "reload_native_adapters",
    "iter_registered_native_adapters",
    "NativeAdapterInfo",
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
