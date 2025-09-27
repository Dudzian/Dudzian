"""Read-only compatibility wrappers bridging legacy entry points.

The modules in this package only re-export symbols from the modern
``KryptoLowca`` namespace so that historical imports keep working.
Do not modify them – new code must target ``bot_core`` instead.
"""

from __future__ import annotations

__all__ = []
