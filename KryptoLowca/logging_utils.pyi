from typing import Any
from pathlib import Path
import logging

LOGS_DIR: Path
DEFAULT_LOG_FILE: Path

def get_logger(name: str) -> logging.Logger: ...
def setup_logging(*args: Any, **kwargs: Any) -> None: ...
def setup_app_logging(*args: Any, **kwargs: Any) -> None: ...

_LISTENER: Any
_QUEUE: Any
