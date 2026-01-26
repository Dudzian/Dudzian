"""Helpers for optional runtime dependencies."""
from __future__ import annotations

from typing import Any


class MissingModuleProxy:
    """Proxy raising a clear RuntimeError when accessed."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        self._message = message
        self._cause = cause

    def _raise(self) -> None:
        raise RuntimeError(self._message) from self._cause

    def __getattr__(self, name: str) -> Any:
        self._raise()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._raise()


def missing_module_proxy(message: str, *, cause: Exception | None = None) -> MissingModuleProxy:
    return MissingModuleProxy(message, cause=cause)
