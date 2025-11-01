"""Fabryki i adaptery giełd wykorzystywane przez ExchangeManager."""
from __future__ import annotations

from typing import Any, Dict, Type

from .adapters import (
    AdapterError,
    BaseExchangeAdapter,
    CCXTExchangeAdapter,
    ExchangeAdapterFactory,
    create_exchange_adapter,
)
from .binance import BinanceTestnetAdapter
from .bitstamp import BitstampAdapter
from .bybit import BybitSpotAdapter
from .interfaces import (
    ExchangeAdapter,
    ExchangeCredentials,
    MarketStreamHandle,
    MarketSubscription,
    OrderRequest,
    OrderStatus,
    RateLimitRule,
    RESTStreamingAdapter,
)
from .kraken import KrakenDemoAdapter
from .okx import OKXDerivativesAdapter, OKXMarginAdapter
from .zonda import ZondaAdapter


def _register_default_adapters() -> None:
    """Rejestruje wbudowane adaptery REST w fabryce."""

    def _factory_for(
        adapter_cls: Type[RESTStreamingAdapter],
    ) -> "Callable[[Dict[str, Any]], BaseExchangeAdapter]":
        def _build(options: Dict[str, Any]) -> RESTStreamingAdapter:
            opts = dict(options)
            http_client = opts.pop("http_client", None)
            demo_mode = bool(opts.pop("demo_mode", True))
            compliance_ack = bool(opts.pop("compliance_ack", False))
            adapter_kwargs = opts.pop("adapter_kwargs", None) or {}
            if not isinstance(adapter_kwargs, dict):
                raise AdapterError(
                    "adapter_kwargs musi być słownikiem przekazywanym do konstruktora adaptera"
                )
            extra_kwargs: Dict[str, Any] = {**opts, **adapter_kwargs}
            return adapter_cls(
                demo_mode=demo_mode,
                http_client=http_client,
                compliance_ack=compliance_ack,
                **extra_kwargs,
            )

        return _build

    registrations: Dict[str, Type[RESTStreamingAdapter]] = {
        "binance-testnet": BinanceTestnetAdapter,
        "bitstamp": BitstampAdapter,
        "bybit-spot": BybitSpotAdapter,
        "bybit": BybitSpotAdapter,
        "kraken-demo": KrakenDemoAdapter,
        "okx-margin": OKXMarginAdapter,
        "okx-derivatives": OKXDerivativesAdapter,
        "zonda": ZondaAdapter,
        "bitbay": ZondaAdapter,
    }

    for name, adapter_cls in registrations.items():
        ExchangeAdapterFactory.register(name, _factory_for(adapter_cls))


_register_default_adapters()

__all__ = [
    "AdapterError",
    "BaseExchangeAdapter",
    "CCXTExchangeAdapter",
    "ExchangeAdapterFactory",
    "create_exchange_adapter",
    "BinanceTestnetAdapter",
    "BitstampAdapter",
    "BybitSpotAdapter",
    "KrakenDemoAdapter",
    "OKXDerivativesAdapter",
    "OKXMarginAdapter",
    "ZondaAdapter",
    "ExchangeAdapter",
    "ExchangeCredentials",
    "MarketStreamHandle",
    "MarketSubscription",
    "OrderRequest",
    "OrderStatus",
    "RateLimitRule",
    "RESTStreamingAdapter",
]
