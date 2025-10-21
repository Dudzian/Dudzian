"""Wspólne pomocniki HTTP wykorzystywane przez kanały alertowe."""
from __future__ import annotations

from typing import Protocol
from urllib import request
from urllib.request import addinfourl


class HttpOpener(Protocol):
    """Interfejs dla funkcji otwierającej żądanie HTTP."""

    def __call__(self, req: request.Request, *, timeout: float) -> addinfourl:
        ...


def default_opener(req: request.Request, *, timeout: float) -> addinfourl:
    """Domyślne wywołanie `urllib.request.urlopen` z kontrolą czasu."""

    return request.urlopen(req, timeout=timeout)  # noqa: S310 - wywołujemy zaufane API kanałów alertowych


__all__ = ["HttpOpener", "default_opener"]
