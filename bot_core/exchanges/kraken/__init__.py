"""Adaptery giełdowe dla Kraken Spot/Futures."""

from bot_core.exchanges.kraken.futures import KrakenFuturesAdapter
from bot_core.exchanges.kraken.spot import KrakenSpotAdapter

__all__ = ["KrakenSpotAdapter", "KrakenFuturesAdapter"]
