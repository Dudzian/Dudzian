from __future__ import annotations

from urllib.request import Request

import pytest

from bot_core.security.http_url import UnsafeHttpUrl, require_http_request, require_http_url


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com/path?x=1",
        "http://127.0.0.1:8080/health",
        "https://[::1]:8443/status",
    ],
)
def test_require_http_url_accepts_explicit_http_transports(value: str) -> None:
    assert require_http_url(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        " https://example.com",
        "https://example.com\n",
        "file:///etc/passwd",
        "ftp://example.com/file",
        "https:///missing-host",
        "https://user@example.com/",
        "https://user:secret@example.com/",
    ],
)
def test_require_http_url_rejects_unsafe_or_ambiguous_urls(value: str) -> None:
    with pytest.raises(UnsafeHttpUrl):
        require_http_url(value)


def test_require_http_request_preserves_validated_request() -> None:
    request = Request("https://example.com/api", data=b"{}")
    assert require_http_request(request) is request


def test_require_http_request_rejects_non_http_scheme() -> None:
    with pytest.raises(UnsafeHttpUrl):
        require_http_request(Request("file:///etc/passwd"))
