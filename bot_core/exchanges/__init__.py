"""Pakiet adapterów giełdowych."""

from __future__ import annotations

import importlib

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeBackend,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.factory import ExchangeAdapterConfig, build_exchange_adapter
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
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.interfaces import (
    MarketStreamHandle,
    MarketSubscription,
    PrivateStreamSubscription,
    PublicStreamSubscription,
    StreamSubscription,
)
from bot_core.optional import missing_module_proxy
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
_LAZY_EXCHANGE_PACKAGES = {
    "binance",
    "bitfinex",
    "bitget",
    "bitmex",
    "bitstamp",
    "bybit",
    "coinbase",
    "deribit",
    "gateio",
    "gemini",
    "huobi",
    "kraken",
    "kucoin",
    "mexc",
    "network_guard",
    "nowa_gielda",
    "okx",
    "zonda",
    "http_client",
}


_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "BaseBackend": ("bot_core.exchanges.core", "BaseBackend"),
    "Event": ("bot_core.exchanges.core", "Event"),
    "EventBus": ("bot_core.exchanges.core", "EventBus"),
    "MarketRules": ("bot_core.exchanges.core", "MarketRules"),
    "Mode": ("bot_core.exchanges.core", "Mode"),
    "OrderDTO": ("bot_core.exchanges.core", "OrderDTO"),
    "OrderSide": ("bot_core.exchanges.core", "OrderSide"),
    "OrderStatus": ("bot_core.exchanges.core", "OrderStatus"),
    "OrderType": ("bot_core.exchanges.core", "OrderType"),
    "PaperBackend": ("bot_core.exchanges.core", "PaperBackend"),
    "PositionDTO": ("bot_core.exchanges.core", "PositionDTO"),
    "ExchangeIOLayer": ("bot_core.exchanges.io", "ExchangeIOLayer"),
    "streaming": ("bot_core.exchanges.streaming", None),
}


def _optional_adapter(module: str, symbol: str):
    try:
        mod = importlib.import_module(module)
        return getattr(mod, symbol)
    except (ModuleNotFoundError, ImportError) as exc:
        dep = getattr(exc, "name", None) or "dependency"
        message = f"Brak opcjonalnej zależności '{dep}' wymaganej dla {module}:{symbol}."
        return missing_module_proxy(message, cause=exc)


def __getattr__(name: str):
    if name in _LAZY_EXCHANGE_PACKAGES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    export = _LAZY_EXPORTS.get(name)
    if export is not None:
        module, symbol = export
        try:
            mod = importlib.import_module(module)
            value = mod if symbol is None else getattr(mod, symbol)
        except (ModuleNotFoundError, ImportError) as exc:
            dep = getattr(exc, "name", None) or "dependency"
            message = f"Brak opcjonalnej zależności '{dep}' wymaganej dla {module}:{name}."
            value = missing_module_proxy(message, cause=exc)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(
        set(globals().keys()) | set(_LAZY_EXCHANGE_PACKAGES) | set(_LAZY_EXPORTS.keys())
    )


BinanceFuturesAdapter = _optional_adapter(
    "bot_core.exchanges.binance.futures", "BinanceFuturesAdapter"
)
BinanceMarginAdapter = _optional_adapter(
    "bot_core.exchanges.binance.margin", "BinanceMarginAdapter"
)
BinanceSpotAdapter = _optional_adapter(
    "bot_core.exchanges.binance.spot", "BinanceSpotAdapter"
)
BitfinexSpotAdapter = _optional_adapter(
    "bot_core.exchanges.bitfinex.spot", "BitfinexSpotAdapter"
)
BitmexFuturesAdapter = _optional_adapter(
    "bot_core.exchanges.bitmex.futures", "BitmexFuturesAdapter"
)
BitmexSpotAdapter = _optional_adapter(
    "bot_core.exchanges.bitmex.spot", "BitmexSpotAdapter"
)
BitgetSpotAdapter = _optional_adapter(
    "bot_core.exchanges.bitget.spot", "BitgetSpotAdapter"
)
BitstampSpotAdapter = _optional_adapter(
    "bot_core.exchanges.bitstamp.spot", "BitstampSpotAdapter"
)
DeribitFuturesAdapter = _optional_adapter(
    "bot_core.exchanges.deribit.futures", "DeribitFuturesAdapter"
)
DeribitSpotAdapter = _optional_adapter(
    "bot_core.exchanges.deribit.spot", "DeribitSpotAdapter"
)
BybitFuturesAdapter = _optional_adapter(
    "bot_core.exchanges.bybit.futures", "BybitFuturesAdapter"
)
BybitMarginAdapter = _optional_adapter(
    "bot_core.exchanges.bybit.margin", "BybitMarginAdapter"
)
BybitSpotAdapter = _optional_adapter(
    "bot_core.exchanges.bybit.spot", "BybitSpotAdapter"
)
CoinbaseFuturesAdapter = _optional_adapter(
    "bot_core.exchanges.coinbase.futures", "CoinbaseFuturesAdapter"
)
CoinbaseMarginAdapter = _optional_adapter(
    "bot_core.exchanges.coinbase.margin", "CoinbaseMarginAdapter"
)
CoinbaseSpotAdapter = _optional_adapter(
    "bot_core.exchanges.coinbase.spot", "CoinbaseSpotAdapter"
)
GateIOSpotAdapter = _optional_adapter(
    "bot_core.exchanges.gateio.spot", "GateIOSpotAdapter"
)
GeminiSpotAdapter = _optional_adapter(
    "bot_core.exchanges.gemini.spot", "GeminiSpotAdapter"
)
HuobiSpotAdapter = _optional_adapter("bot_core.exchanges.huobi.spot", "HuobiSpotAdapter")
MexcSpotAdapter = _optional_adapter("bot_core.exchanges.mexc.spot", "MexcSpotAdapter")
KuCoinSpotAdapter = _optional_adapter(
    "bot_core.exchanges.kucoin.spot", "KuCoinSpotAdapter"
)
KrakenFuturesAdapter = _optional_adapter(
    "bot_core.exchanges.kraken.futures", "KrakenFuturesAdapter"
)
KrakenMarginAdapter = _optional_adapter(
    "bot_core.exchanges.kraken.margin", "KrakenMarginAdapter"
)
KrakenSpotAdapter = _optional_adapter(
    "bot_core.exchanges.kraken.spot", "KrakenSpotAdapter"
)
NowaGieldaSpotAdapter = _optional_adapter(
    "bot_core.exchanges.nowa_gielda.spot", "NowaGieldaSpotAdapter"
)
OKXFuturesAdapter = _optional_adapter("bot_core.exchanges.okx.futures", "OKXFuturesAdapter")
OKXMarginAdapter = _optional_adapter("bot_core.exchanges.okx.margin", "OKXMarginAdapter")
OKXSpotAdapter = _optional_adapter("bot_core.exchanges.okx.spot", "OKXSpotAdapter")
ZondaMarginAdapter = _optional_adapter("bot_core.exchanges.zonda.margin", "ZondaMarginAdapter")
ZondaSpotAdapter = _optional_adapter("bot_core.exchanges.zonda.spot", "ZondaSpotAdapter")
__all__ = [
    "AccountSnapshot",
    "Environment",
    "ExchangeAdapter", 
    "ExchangeBackend", 
    "ExchangeAdapterConfig", 
    "build_exchange_adapter", 
    "ExchangeIOLayer", 
    "ExchangeCredentials", 
    "OrderRequest",
    "OrderResult",
    "ExchangeError",
    "ExchangeAPIError",
    "ExchangeAuthError",
    "ExchangeThrottlingError",
    "ExchangeNetworkError",
    "MarketStreamHandle",
    "MarketSubscription",
    "StreamSubscription",
    "PublicStreamSubscription",
    "PrivateStreamSubscription",
    "KrakenFuturesAdapter",
    "KrakenSpotAdapter",
    "KrakenMarginAdapter",
    "BybitSpotAdapter",
    "BybitMarginAdapter",
    "BybitFuturesAdapter",
    "BitmexSpotAdapter",
    "BitmexFuturesAdapter",
    "CoinbaseSpotAdapter",
    "CoinbaseMarginAdapter",
    "CoinbaseFuturesAdapter",
    "BitfinexSpotAdapter",
    "BitgetSpotAdapter",
    "BitstampSpotAdapter",
    "DeribitSpotAdapter",
    "DeribitFuturesAdapter",
    "GateIOSpotAdapter",
    "GeminiSpotAdapter",
    "HuobiSpotAdapter",
    "MexcSpotAdapter",
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
