"""Zgodność wsteczna dla migratora Stage6 – deleguje do bot_core.runtime.stage6_preset_cli."""

from __future__ import annotations

import warnings

from bot_core.runtime.stage6_preset_cli import main

warnings.warn(
    "KryptoLowca.scripts.preset_editor_cli jest przestarzałe – użyj bot_core.runtime.stage6_preset_cli",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
