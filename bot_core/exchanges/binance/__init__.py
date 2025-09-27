"""Adaptery Binance."""

from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter

__all__ = ["BinanceFuturesAdapter", "BinanceSpotAdapter"]
