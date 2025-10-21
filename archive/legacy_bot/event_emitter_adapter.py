"""Compatibility bridge to the canonical event emitter implementation."""

from __future__ import annotations

from typing import Any

from bot_core.events import emitter as _impl
from bot_core.events.emitter import *  # noqa: F401,F403 - legacy re-export

__all__ = list(getattr(_impl, "__all__", ()))
__doc__ = __doc__ + ("\n\n" + (_impl.__doc__ or ""))


def __getattr__(name: str) -> Any:  # pragma: no cover - delegation helper
    return getattr(_impl, name)


def __dir__() -> list[str]:  # pragma: no cover - cosmetic helper for REPLs
    return sorted(set(__all__) | set(dir(_impl)))
