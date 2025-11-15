"""Punkt wejścia modułu `python -m ui.pyside_app`."""
from .app import main


if __name__ == "__main__":  # pragma: no cover - przekierowanie do funkcji main
    raise SystemExit(main())
