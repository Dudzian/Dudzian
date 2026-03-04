from __future__ import annotations

import sys


def configure_cli_stdio() -> None:
    """Zapobiega błędom kodowania CLI na stdout/stderr w środowiskach bez UTF-8."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(errors="backslashreplace")
        except Exception:  # pragma: no cover - depends on stream implementation
            pass
