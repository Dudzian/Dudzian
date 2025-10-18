"""Zgodnościowy wrapper delegujący do modułowego launchera AutoTradera."""

from __future__ import annotations

from pathlib import Path
import sys


def _ensure_repo_root() -> None:
    current_dir = Path(__file__).resolve().parent
    for candidate in (current_dir, *current_dir.parents):
        package_init = candidate / "KryptoLowca" / "__init__.py"
        if package_init.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break


if __package__ in (None, ""):
    _ensure_repo_root()


from KryptoLowca.auto_trader.paper import (  # noqa: E402
    HeadlessTradingStub,
    PaperAutoTradeApp,
    main,
)

__all__ = ["HeadlessTradingStub", "PaperAutoTradeApp", "main"]


if __name__ == "__main__":  # pragma: no cover - manualne uruchomienie
    main(sys.argv[1:])
