"""Alias do archiwalnego modułu Stage6 HyperCare.

Bieżący runtime Stage6 nie powinien wykonywać kodu HyperCare; moduł jest
pozostawiony wyłącznie dla kompatybilności z historycznymi narzędziami i testami.
"""

from bot_core.runtime.archive import stage6_hypercare as _archived
from bot_core.runtime.archive.stage6_hypercare import *  # noqa: F401,F403

__all__ = getattr(_archived, "__all__", [])


def __getattr__(name: str):
    return getattr(_archived, name)


def __setattr__(name: str, value: object) -> None:
    setattr(_archived, name, value)
