"""Bounded, append-only-across-restarts Windows service logging boundary."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILE_NAME = "backend.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 4
LOG_FORMAT = "%(asctime)sZ level=%(levelname)s pid=%(process)d %(message)s"
ALLOWED_LOG_FILES = frozenset({LOG_FILE_NAME, *(f"{LOG_FILE_NAME}.{i}" for i in range(1, 5))})


class UtcFormatter(logging.Formatter):
    converter = __import__("time").gmtime


def create_handler(logs: Path, *, max_bytes: int = MAX_BYTES,
                   backup_count: int = BACKUP_COUNT) -> RotatingFileHandler:
    """Create the sole production sink; ``a`` preserves earlier service runs."""
    handler = RotatingFileHandler(
        logs / LOG_FILE_NAME, mode="a", maxBytes=max_bytes,
        backupCount=backup_count, encoding="utf-8", delay=False,
    )
    handler.setFormatter(UtcFormatter(LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S"))
    return handler


def configure_service_logger(logs: Path) -> logging.Logger:
    logger = logging.getLogger("cryptohunter.windows.acceptance")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for old in list(logger.handlers):
        old.close()
        logger.removeHandler(old)
    logger.addHandler(create_handler(logs))
    return logger


def close_service_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
