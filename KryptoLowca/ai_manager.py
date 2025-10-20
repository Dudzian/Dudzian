"""Legacy compatibility wrapper for :mod:`bot_core.ai.manager`.

The modern implementation of the asynchronous AI manager lives in
``bot_core.ai.manager``.  The legacy ``KryptoLowca`` namespace continues to be
used by a number of historical scripts and tests.  To avoid duplicating the
implementation (and the maintenance burden that comes with it) this module
simply re-exports the canonical symbols from the new package.
"""

from __future__ import annotations

from typing import Any

from bot_core.ai import manager as _impl
from bot_core.ai.manager import *  # noqa: F401,F403 - re-export public API

__all__ = list(getattr(_impl, "__all__", ()))
__doc__ = __doc__ + ("\n\n" + (_impl.__doc__ or ""))


def __getattr__(name: str) -> Any:  # pragma: no cover - delegation helper
    return getattr(_impl, name)


def __dir__() -> list[str]:  # pragma: no cover - cosmetic helper for REPLs
    return sorted(set(__all__) | set(dir(_impl)))
