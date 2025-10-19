"""Warstwa zgodności delegująca do natywnego modułu ``bot_core.ai.manager``.

Moduł ``bot_core.ai.manager`` zawiera właściwą implementację asynchronicznego
menedżera modeli AI.  Pakiet ``KryptoLowca`` nadal eksportuje te same symbole,
aby zachować kompatybilność wsteczną ze starszymi skryptami i testami.
"""
from __future__ import annotations

from typing import Any, Iterable

from bot_core.ai import manager as _impl
from bot_core.ai.manager import *  # noqa: F401,F403

__doc__ = _impl.__doc__
__all__ = list(getattr(_impl, "__all__", ()))


def __getattr__(name: str) -> Any:  # pragma: no cover - delegacja dla atrybutów pomocniczych
    return getattr(_impl, name)


def __dir__() -> Iterable[str]:  # pragma: no cover - utrzymanie wygodnej introspekcji
    return sorted(set(__all__) | set(dir(_impl)))
