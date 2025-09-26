"""Szkielet API (np. FastAPI) do zarządzania botem."""
from __future__ import annotations

from typing import Any, Dict

from KryptoLowca.logging_utils import get_logger

logger = get_logger(__name__)


class TradingAPI:
    """Minimalny kontroler API – do rozbudowy w kolejnych iteracjach."""

    def __init__(self) -> None:
        self._routes: Dict[str, Any] = {}

    def add_route(self, path: str, handler: Any) -> None:
        logger.debug("Rejestruję endpoint %s", path)
        self._routes[path] = handler

    def routes(self) -> Dict[str, Any]:
        return dict(self._routes)


__all__ = ["TradingAPI"]
