# exchange_manager.py
# -*- coding: utf-8 -*-
"""Shim dla starszych importów ExchangeManager."""
from __future__ import annotations

from managers import exchange_manager as _core
from managers.exchange_manager import *  # noqa: F401,F403

__all__ = getattr(_core, "__all__", None)
if not __all__:
    __all__ = [name for name in dir(_core) if not name.startswith("_")]
