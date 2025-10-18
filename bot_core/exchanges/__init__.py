"""Pakiet adapterów giełdowych."""

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter
from bot_core.exchanges.bitfinex.spot import BitfinexSpotAdapter
from bot_core.exchanges.coinbase.spot import CoinbaseSpotAdapter
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeAuthError,
    ExchangeError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.kraken.futures import KrakenFuturesAdapter
from bot_core.exchanges.kraken.spot import KrakenSpotAdapter
from bot_core.exchanges.nowa_gielda.spot import NowaGieldaSpotAdapter
from bot_core.exchanges.okx.spot import OKXSpotAdapter
from bot_core.exchanges.zonda.spot import ZondaSpotAdapter

__all__ = [
    "AccountSnapshot",
    "BinanceFuturesAdapter",
    "BinanceSpotAdapter",
    "Environment",
    "ExchangeAdapter",
    "ExchangeCredentials",
    "ExchangeError",
    "ExchangeAPIError",
    "ExchangeAuthError",
    "ExchangeThrottlingError",
    "ExchangeNetworkError",
    "KrakenFuturesAdapter",
    "KrakenSpotAdapter",
    "CoinbaseSpotAdapter",
    "BitfinexSpotAdapter",
    "OKXSpotAdapter",
    "NowaGieldaSpotAdapter",
    "OrderRequest",
    "OrderResult",
    "ZondaSpotAdapter",
]
