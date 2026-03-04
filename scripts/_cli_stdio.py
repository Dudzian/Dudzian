from __future__ import annotations

import sys


def configure_cli_stdio() -> None:
    """Wymusza UTF-8 na stdout/stderr i zachowuje niezawodny fallback backslashreplace."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:  # pragma: no cover - depends on stream implementation
            try:
                reconfigure(errors="backslashreplace")
            except Exception:  # pragma: no cover - depends on stream implementation
                pass
