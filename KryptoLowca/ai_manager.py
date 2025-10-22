"""Legacy compatibility wrapper for :mod:`bot_core.ai.manager`.

The modern implementation of the asynchronous AI manager lives in
``bot_core.ai.manager``.  The legacy ``KryptoLowca`` namespace continues to be
used by a number of historical scripts and tests.  To avoid duplicating the
implementation (and the maintenance burden that comes with it) this module
simply re-exports the canonical symbols from the new package.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Iterator

from bot_core.ai import manager as _impl
from bot_core.ai.manager import *  # noqa: F401,F403 - re-export public API


def _exported_names() -> list[str]:
    return list(getattr(_impl, "__all__", ()))


class _AllProxy(Sequence[str]):
    __slots__ = ()

    def _data(self) -> list[str]:
        return _exported_names()

    def __iter__(self) -> Iterator[str]:
        return iter(self._data())

    def __len__(self) -> int:
        return len(self._data())

    def __getitem__(self, index: int) -> str:
        return self._data()[index]

    def __repr__(self) -> str:  # pragma: no cover - diagnostyka
        return repr(self._data())

    def __eq__(self, other: object) -> bool:  # pragma: no cover - dla testów
        if isinstance(other, Sequence):
            return list(self) == list(other)
        if isinstance(other, set):
            return set(self) == set(other)
        return list(self) == list(other) if hasattr(other, "__iter__") else False


__all__ = _AllProxy()
__doc__ = __doc__ + ("\n\n" + (_impl.__doc__ or ""))


def __getattr__(name: str) -> Any:  # pragma: no cover - delegation helper
    if name == "__all__":
        return _AllProxy()
    return getattr(_impl, name)


def __dir__() -> list[str]:  # pragma: no cover - cosmetic helper for REPLs
    return sorted(set(_exported_names()) | set(dir(_impl)))
