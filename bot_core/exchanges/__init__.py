"""Pakiet adapterów giełdowych."""

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
    from bot_core.exchanges.manager import ExchangeManager
except Exception:  # pragma: no cover - zapewniamy import pakietu
    ExchangeManager = None  # type: ignore[assignment]
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.margin import BinanceMarginAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter
from bot_core.exchanges.bitfinex.spot import BitfinexSpotAdapter
from bot_core.exchanges.bybit.spot import BybitSpotAdapter
from bot_core.exchanges.coinbase.spot import CoinbaseSpotAdapter
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
from bot_core.exchanges.okx.spot import OKXSpotAdapter
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

__all__ = [
    "AccountSnapshot",
    "BaseBackend",
    "BinanceFuturesAdapter",
    "BinanceSpotAdapter",
    "BinanceMarginAdapter",
    "Event",
    "EventBus",
    "Environment",
    "ExchangeManager",
    "ExchangeAdapter",
    "ExchangeCredentials",
    "ExchangeError",
    "ExchangeAPIError",
    "ExchangeAuthError",
    "ExchangeThrottlingError",
    "ExchangeNetworkError",
    "KrakenFuturesAdapter",
    "KrakenSpotAdapter",
    "KrakenMarginAdapter",
    "BybitSpotAdapter",
    "CoinbaseSpotAdapter",
    "BitfinexSpotAdapter",
    "KuCoinSpotAdapter",
    "OKXSpotAdapter",
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
]
