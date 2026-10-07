"""Compatibility re-export for HTTP transport URL validation."""

from bot_core.http_url import UnsafeHttpUrl, require_http_request, require_http_url

__all__ = ["UnsafeHttpUrl", "require_http_request", "require_http_url"]
