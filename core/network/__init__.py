"""Pomocnicze komponenty sieciowe współdzielone przez moduły bota."""

from .async_http import RateLimitedAsyncClient, get_rate_limited_client

__all__ = ["RateLimitedAsyncClient", "get_rate_limited_client"]
