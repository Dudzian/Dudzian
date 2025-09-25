"""Utilities for consistent logging configuration across the bot.

This module centralises setup of rotating file handlers so that every
component (GUI, background workers, integration tests) writes to the same
log files without growing indefinitely.  Importing :func:`get_logger`
ensures the handler exists and can be reused safely.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

# Repository root (folder that contains this module)
_APP_ROOT = Path(__file__).resolve().parent
LOGS_DIR = _APP_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = LOGS_DIR / "trading.log"


def _ensure_rotating_handler(
    logger: logging.Logger,
    log_file: Union[str, Path] = DEFAULT_LOG_FILE,
    max_bytes: int = 2_000_000,
    backup_count: int = 5,
) -> None:
    """Attach a RotatingFileHandler to ``logger`` if missing."""
    log_path = Path(log_file)
    identifier = "_krypto_rotating_handler"

    for handler in logger.handlers:
        if getattr(handler, identifier, False):
            return

    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )
    setattr(handler, identifier, True)
    logger.addHandler(handler)


def setup_app_logging(
    log_file: Union[str, Path] = DEFAULT_LOG_FILE,
    level: int = logging.INFO,
    max_bytes: int = 2_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure and return the shared application logger."""
    root = logging.getLogger("KryptoLowca")
    if getattr(root, "_krypto_logging_configured", False):
        return root

    _ensure_rotating_handler(root, log_file=log_file, max_bytes=max_bytes, backup_count=backup_count)
    root.setLevel(level)
    root.propagate = True
    setattr(root, "_krypto_logging_configured", True)
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger that uses the shared rotating handler."""
    setup_app_logging()
    return logging.getLogger(name if name else "KryptoLowca")


__all__ = [
    "LOGS_DIR",
    "DEFAULT_LOG_FILE",
    "get_logger",
    "setup_app_logging",
]
