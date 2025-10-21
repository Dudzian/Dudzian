"""Legacy entrypoint delegating to ``KryptoLowca.run_trading_gui_paper_emitter``."""

from __future__ import annotations

from typing import Any

from KryptoLowca import run_trading_gui_paper_emitter as _impl
from KryptoLowca.run_trading_gui_paper_emitter import *  # noqa: F401,F403

__all__ = [name for name in dir(_impl) if not name.startswith("_")]
__doc__ = __doc__ + ("\n\n" + (_impl.__doc__ or ""))


def __getattr__(name: str) -> Any:  # pragma: no cover
    return getattr(_impl, name)


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(__all__) | set(dir(_impl)))
