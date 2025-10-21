"""Adapter Coinbase Spot."""

from bot_core.exchanges.coinbase.futures import CoinbaseFuturesAdapter
from bot_core.exchanges.coinbase.margin import CoinbaseMarginAdapter
from bot_core.exchanges.coinbase.spot import CoinbaseSpotAdapter

__all__ = ["CoinbaseSpotAdapter", "CoinbaseMarginAdapter", "CoinbaseFuturesAdapter"]

