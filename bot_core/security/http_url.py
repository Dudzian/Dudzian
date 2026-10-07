"""Fail-closed validation for URLs passed to stdlib HTTP transports."""

from __future__ import annotations

from urllib.parse import urlsplit
from urllib.request import Request


class UnsafeHttpUrl(ValueError):
    """Raised when a URL is not safe for the project's HTTP-only transports."""


def require_http_url(value: str) -> str:
    """Allow only explicit HTTP(S) URLs with a host and no embedded credentials."""

    if type(value) is not str or not value or value != value.strip():
        raise UnsafeHttpUrl("HTTP URL must be a non-empty canonical string")
    if any(character in value for character in ("\r", "\n", "\t")):
        raise UnsafeHttpUrl("HTTP URL contains control characters")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError as exc:
        raise UnsafeHttpUrl("HTTP URL is malformed") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise UnsafeHttpUrl("only http:// and https:// URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeHttpUrl("embedded URL credentials are forbidden")
    return value


def require_http_request(value: Request) -> Request:
    """Validate a urllib Request before transport use."""

    require_http_url(value.full_url)
    return value


__all__ = ["UnsafeHttpUrl", "require_http_request", "require_http_url"]
