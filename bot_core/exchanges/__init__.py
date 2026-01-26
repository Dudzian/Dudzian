"""Pakiet adapterów giełdowych."""

from __future__ import annotations

import os
import importlib
from types import ModuleType
from typing import Any

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
_PYDANTIC_ERROR: Exception | None = None
try:  # pragma: no cover - zależne od środowiska
    importlib.import_module("pydantic")
except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover - brak lub uszkodzone pydantic
    _HAS_PYDANTIC = False
    _PYDANTIC_ERROR = exc
else:  # pragma: no cover - pydantic dostępny
    _HAS_PYDANTIC = True


def _missing_pydantic_class(symbol: str):
    message = (
        "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć bot_core.exchanges.core."
    )

    class _MissingPydantic:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(message) from _PYDANTIC_ERROR

        def __getattr__(self, name: str) -> object:
            raise RuntimeError(message) from _PYDANTIC_ERROR

    _MissingPydantic.__name__ = symbol
    return _MissingPydantic


class _MissingOptionalDependency:
    def __init__(
        self,
        dependency: str,
        *,
        module_path: str,
        symbol: str,
        cause: Exception | None = None,
    ) -> None:
        self._message = (
            f"Brak opcjonalnej zależności '{dependency}' wymaganej dla "
            f"{module_path}:{symbol}. Zainstaluj ją, aby używać tego adaptera."
        )
        self._cause = cause

    def __getattr__(self, name: str) -> object:
        raise RuntimeError(self._message) from self._cause

    def __call__(self, *args, **kwargs):
        raise RuntimeError(self._message) from self._cause


def _import_adapter(module_path: str, symbol: str):
    try:  # pragma: no cover - zależne od dostępnych backendów
        module = importlib.import_module(module_path)
        return getattr(module, symbol)
    except ModuleNotFoundError as exc:  # pragma: no cover - light env
        dependency = exc.name or module_path
        return _MissingOptionalDependency(
            dependency,
            module_path=module_path,
            symbol=symbol,
            cause=exc,
        )
    except ImportError as exc:  # pragma: no cover - light env
        dependency = exc.name or "import"
        return _MissingOptionalDependency(
            dependency,
            module_path=module_path,
            symbol=symbol,
            cause=exc,
        )


if _HAS_PYDANTIC:  # pragma: no cover - zależne od środowiska
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
else:  # pragma: no cover - brak pydantic w light env
    BaseBackend = _missing_pydantic_class("BaseBackend")  # type: ignore[assignment]
    Event = _missing_pydantic_class("Event")  # type: ignore[assignment]
    EventBus = _missing_pydantic_class("EventBus")  # type: ignore[assignment]
    MarketRules = _missing_pydantic_class("MarketRules")  # type: ignore[assignment]
    Mode = _missing_pydantic_class("Mode")  # type: ignore[assignment]
    OrderDTO = _missing_pydantic_class("OrderDTO")  # type: ignore[assignment]
    OrderSide = _missing_pydantic_class("OrderSide")  # type: ignore[assignment]
    OrderStatus = _missing_pydantic_class("OrderStatus")  # type: ignore[assignment]
    OrderType = _missing_pydantic_class("OrderType")  # type: ignore[assignment]
    PaperBackend = _missing_pydantic_class("PaperBackend")  # type: ignore[assignment]
    PositionDTO = _missing_pydantic_class("PositionDTO")  # type: ignore[assignment]
if _HAS_PYDANTIC:  # pragma: no cover - zależne od środowiska
    from bot_core.exchanges.io import ExchangeIOLayer
    try:  # pragma: no cover - środowisko testowe może nie zawierać pełnej konfiguracji managera
        from bot_core.exchanges.manager import (
            ExchangeManager,
            NativeAdapterInfo,
            iter_registered_native_adapters,
            register_native_adapter,
            reload_native_adapters,
        )
    except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover - zapewniamy import pakietu
        dependency = exc.name or "import"
        ExchangeManager = _MissingOptionalDependency(  # type: ignore[assignment]
            dependency,
            module_path="bot_core.exchanges.manager",
            symbol="ExchangeManager",
            cause=exc,
        )
        register_native_adapter = _MissingOptionalDependency(  # type: ignore[assignment]
            dependency,
            module_path="bot_core.exchanges.manager",
            symbol="register_native_adapter",
            cause=exc,
        )
        iter_registered_native_adapters = _MissingOptionalDependency(  # type: ignore[assignment]
            dependency,
            module_path="bot_core.exchanges.manager",
            symbol="iter_registered_native_adapters",
            cause=exc,
        )
        reload_native_adapters = _MissingOptionalDependency(  # type: ignore[assignment]
            dependency,
            module_path="bot_core.exchanges.manager",
            symbol="reload_native_adapters",
            cause=exc,
        )
        NativeAdapterInfo = _MissingOptionalDependency(  # type: ignore[assignment]
            dependency,
            module_path="bot_core.exchanges.manager",
            symbol="NativeAdapterInfo",
            cause=exc,
        )
else:  # pragma: no cover - brak pydantic w light env
    ExchangeIOLayer = _missing_pydantic_class("ExchangeIOLayer")  # type: ignore[assignment]
    ExchangeManager = None  # type: ignore[assignment]
    register_native_adapter = None  # type: ignore[assignment]
    iter_registered_native_adapters = None  # type: ignore[assignment]
    reload_native_adapters = None  # type: ignore[assignment]
    NativeAdapterInfo = None  # type: ignore[assignment]
BinanceFuturesAdapter = _import_adapter("bot_core.exchanges.binance.futures", "BinanceFuturesAdapter")
BinanceMarginAdapter = _import_adapter("bot_core.exchanges.binance.margin", "BinanceMarginAdapter")
BinanceSpotAdapter = _import_adapter("bot_core.exchanges.binance.spot", "BinanceSpotAdapter")
BitfinexSpotAdapter = _import_adapter("bot_core.exchanges.bitfinex.spot", "BitfinexSpotAdapter")
BitmexFuturesAdapter = _import_adapter("bot_core.exchanges.bitmex", "BitmexFuturesAdapter")
BitmexSpotAdapter = _import_adapter("bot_core.exchanges.bitmex", "BitmexSpotAdapter")
BitgetSpotAdapter = _import_adapter("bot_core.exchanges.bitget.spot", "BitgetSpotAdapter")
BitstampSpotAdapter = _import_adapter("bot_core.exchanges.bitstamp.spot", "BitstampSpotAdapter")
DeribitFuturesAdapter = _import_adapter("bot_core.exchanges.deribit", "DeribitFuturesAdapter")
DeribitSpotAdapter = _import_adapter("bot_core.exchanges.deribit", "DeribitSpotAdapter")
BybitFuturesAdapter = _import_adapter("bot_core.exchanges.bybit", "BybitFuturesAdapter")
BybitMarginAdapter = _import_adapter("bot_core.exchanges.bybit", "BybitMarginAdapter")
BybitSpotAdapter = _import_adapter("bot_core.exchanges.bybit", "BybitSpotAdapter")
CoinbaseFuturesAdapter = _import_adapter("bot_core.exchanges.coinbase", "CoinbaseFuturesAdapter")
CoinbaseMarginAdapter = _import_adapter("bot_core.exchanges.coinbase", "CoinbaseMarginAdapter")
CoinbaseSpotAdapter = _import_adapter("bot_core.exchanges.coinbase", "CoinbaseSpotAdapter")
GateIOSpotAdapter = _import_adapter("bot_core.exchanges.gateio.spot", "GateIOSpotAdapter")
GeminiSpotAdapter = _import_adapter("bot_core.exchanges.gemini.spot", "GeminiSpotAdapter")
HuobiSpotAdapter = _import_adapter("bot_core.exchanges.huobi.spot", "HuobiSpotAdapter")
MexcSpotAdapter = _import_adapter("bot_core.exchanges.mexc.spot", "MexcSpotAdapter")
LoopbackExchangeAdapter = _import_adapter(
    "bot_core.exchanges.testing.loopback", "LoopbackExchangeAdapter"
)
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
KuCoinSpotAdapter = _import_adapter("bot_core.exchanges.kucoin.spot", "KuCoinSpotAdapter")
KrakenFuturesAdapter = _import_adapter("bot_core.exchanges.kraken.futures", "KrakenFuturesAdapter")
KrakenMarginAdapter = _import_adapter("bot_core.exchanges.kraken.margin", "KrakenMarginAdapter")
KrakenSpotAdapter = _import_adapter("bot_core.exchanges.kraken.spot", "KrakenSpotAdapter")
NowaGieldaSpotAdapter = _import_adapter("bot_core.exchanges.nowa_gielda.spot", "NowaGieldaSpotAdapter")
OKXFuturesAdapter = _import_adapter("bot_core.exchanges.okx", "OKXFuturesAdapter")
OKXMarginAdapter = _import_adapter("bot_core.exchanges.okx", "OKXMarginAdapter")
OKXSpotAdapter = _import_adapter("bot_core.exchanges.okx", "OKXSpotAdapter")
ZondaMarginAdapter = _import_adapter("bot_core.exchanges.zonda.margin", "ZondaMarginAdapter")
ZondaSpotAdapter = _import_adapter("bot_core.exchanges.zonda.spot", "ZondaSpotAdapter")
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
try:  # pragma: no cover - opcjonalne zależności strumieni
    from . import streaming
except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover - light env
    streaming = _MissingOptionalDependency(
        exc.name or "dependency",
        module_path="bot_core.exchanges.streaming",
        symbol="streaming",
        cause=exc,
    )  # type: ignore[assignment]

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
    "LoopbackExchangeAdapter",
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

# Subpakiety giełd ładowane leniwie (dla kompatybilności ścieżek typu
# bot_core.exchanges.binance.futures.* używanych m.in. w testach/monkeypatch).
_LAZY_EXCHANGE_PACKAGES = {
    "binance",
    "bitfinex",
    "bitget",
    "bitmex",
    "bitstamp",
    "coinbase",
    "bybit",
    "deribit",
    "gateio",
    "gemini",
    "huobi",
    "kraken",
    "kucoin",
    "mexc",
    "nowa_gielda",
    "okx",
    "testing",
    "zonda",
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _LAZY_EXCHANGE_PACKAGES:
        mod: ModuleType = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | _LAZY_EXCHANGE_PACKAGES)
