"""Adaptery Binance."""

from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.margin import BinanceMarginAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter
from bot_core.exchanges.binance import symbols as symbols

__all__ = [
    "BinanceFuturesAdapter",
    "BinanceMarginAdapter",
    "BinanceSpotAdapter",
    "symbols",
]
