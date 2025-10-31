"""Konfiguracja loggerów specyficznych dla warstwy UI."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

_ONBOARDING_LOG_DIRECTORY: Final[Path] = Path("logs/ui/onboarding")
_UPDATE_LOG_DIRECTORY: Final[Path] = Path("logs/ui/updates")
_ONBOARDING_LOGGER_NAMESPACE: Final[str] = "ui.onboarding"
_UPDATE_LOGGER_NAMESPACE: Final[str] = "ui.offline_update"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def _get_or_create_file_logger(directory: Path, namespace: str, filename: str) -> logging.Logger:
    """Tworzy (lub ponownie wykorzystuje) logger zapisujący do wskazanego pliku."""

    directory.mkdir(parents=True, exist_ok=True)
    logger_name = f"{namespace}[{directory}]"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.FileHandler(directory / filename, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def get_onboarding_logger(log_directory: str | Path = _ONBOARDING_LOG_DIRECTORY) -> logging.Logger:
    """Zwraca logger zapisujący zdarzenia kreatora onboardingowego."""

    return _get_or_create_file_logger(Path(log_directory), _ONBOARDING_LOGGER_NAMESPACE, "onboarding.log")


def get_update_logger(log_directory: str | Path = _UPDATE_LOG_DIRECTORY) -> logging.Logger:
    """Zwraca logger zapisujący import pakietów aktualizacji offline."""

    return _get_or_create_file_logger(Path(log_directory), _UPDATE_LOGGER_NAMESPACE, "offline_update.log")


__all__ = ["get_onboarding_logger", "get_update_logger"]
