"""Legacy wrapper mirroring :mod:`bot_core.exchanges.manager`."""

from __future__ import annotations

from typing import Any

from bot_core.exchanges import manager as _impl
from bot_core.exchanges.manager import *  # noqa: F401,F403 - public API passthrough

__all__ = list(getattr(_impl, "__all__", ()))
__doc__ = __doc__ + ("\n\n" + (_impl.__doc__ or ""))


def __getattr__(name: str) -> Any:  # pragma: no cover
    return getattr(_impl, name)


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(__all__) | set(dir(_impl)))
