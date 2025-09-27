"""Fabryki i adaptery giełd wykorzystywane przez ExchangeManager."""
from __future__ import annotations

from .adapters import (
    AdapterError,
    BaseExchangeAdapter,
    CCXTExchangeAdapter,
    ExchangeAdapterFactory,
    create_exchange_adapter,
)
from .binance import BinanceTestnetAdapter
from .interfaces import (
    ExchangeAdapter,
    ExchangeCredentials,
    MarketSubscription,
    OrderRequest,
    OrderStatus,
    RateLimitRule,
    RESTWebSocketAdapter,
    WebSocketSubscription,
)
from .kraken import KrakenDemoAdapter
from .zonda import ZondaAdapter

__all__ = [
    "AdapterError",
    "BaseExchangeAdapter",
    "CCXTExchangeAdapter",
    "ExchangeAdapterFactory",
    "create_exchange_adapter",
    "BinanceTestnetAdapter",
    "KrakenDemoAdapter",
    "ZondaAdapter",
    "ExchangeAdapter",
    "ExchangeCredentials",
    "MarketSubscription",
    "OrderRequest",
    "OrderStatus",
    "RateLimitRule",
    "RESTWebSocketAdapter",
    "WebSocketSubscription",
]
