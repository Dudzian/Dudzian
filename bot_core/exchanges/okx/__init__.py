"""Adapter OKX Spot."""

from bot_core.exchanges.okx.futures import OKXFuturesAdapter
from bot_core.exchanges.okx.margin import OKXMarginAdapter
from bot_core.exchanges.okx.spot import OKXSpotAdapter

__all__ = ["OKXSpotAdapter", "OKXMarginAdapter", "OKXFuturesAdapter"]

