"""Pakiet z lekkim event-busem wykorzystywanym w rdzeniu bota."""
from __future__ import annotations

from . import emitter as _emitter
from .emitter import *  # noqa: F401,F403

__all__ = list(_emitter.__all__)
del _emitter
