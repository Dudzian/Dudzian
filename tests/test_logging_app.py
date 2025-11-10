"""Tests for application logging configuration environment handling."""

from __future__ import annotations

import importlib
import logging
import sys
from contextlib import suppress
from logging.handlers import QueueHandler
from pathlib import Path

import pytest


def _reload_logging_app() -> None:
    module_name = "bot_core.logging.app"
    module = sys.modules.pop(module_name, None)
    if module is not None and hasattr(module, "_stop_queue_listener"):
        with suppress(Exception):
            module._stop_queue_listener()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    if hasattr(root_logger, "_bot_core_logging_configured"):
        delattr(root_logger, "_bot_core_logging_configured")

    for logger in list(logging.Logger.manager.loggerDict.values()):  # type: ignore[attr-defined]
        if isinstance(logger, logging.Logger):
            logger.handlers.clear()
            if hasattr(logger, "_bot_core_logging_configured"):
                delattr(logger, "_bot_core_logging_configured")

    importlib.import_module(module_name)


def test_logging_env_ignores_archival_variables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BOT_CORE_LOG_DIR", raising=False)
    monkeypatch.delenv("BOT_CORE_LOG_FILE", raising=False)
    monkeypatch.delenv("BOT_CORE_LOGGER_NAME", raising=False)
    monkeypatch.delenv("BOT_CORE_LOG_LEVEL", raising=False)
    monkeypatch.delenv("BOT_CORE_LOG_FORMAT", raising=False)
    monkeypatch.delenv("BOT_CORE_LOG_SHIP_VECTOR", raising=False)

    monkeypatch.setenv("KRYPT_LOWCA_LOG_DIR", str(tmp_path / "sunset_dir"))
    monkeypatch.setenv("KRYPT_LOWCA_LOG_FILE", str(tmp_path / "sunset.log"))
    monkeypatch.setenv("KRYPT_LOWCA_LOGGER_NAME", "sunset.logger")
    monkeypatch.setenv("KRYPT_LOWCA_LOG_LEVEL", "debug")
    monkeypatch.setenv("KRYPT_LOWCA_LOG_FORMAT", "text")
    monkeypatch.setenv("KRYPT_LOWCA_LOG_SHIP_VECTOR", "https://sunset.invalid")

    _reload_logging_app()
    module = sys.modules["bot_core.logging.app"]

    assert module.LOGS_DIR == module._PACKAGE_ROOT / "logs"
    assert module.DEFAULT_LOG_FILE == module.LOGS_DIR / "trading.log"

    logger = module.setup_app_logging()
    assert logger.name == "bot_core"
    assert logger.level == logging.INFO

    listener = getattr(module, "_LISTENER")
    assert listener is None or not any(
        isinstance(handler, module.VectorHttpHandler) for handler in listener.handlers
    )


def test_logging_env_uses_new_prefix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOT_CORE_LOG_DIR", str(tmp_path))

    _reload_logging_app()
    module = sys.modules["bot_core.logging.app"]

    logs_dir = getattr(module, "LOGS_DIR")
    assert isinstance(logs_dir, Path)
    assert logs_dir == tmp_path


def test_logging_env_allows_custom_logger_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_CORE_LOGGER_NAME", "custom.bot")

    _reload_logging_app()
    module = sys.modules["bot_core.logging.app"]

    logger = module.setup_app_logging()
    assert logger.name == "custom.bot"


def test_logging_env_uses_env_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_CORE_LOG_LEVEL", "debug")

    _reload_logging_app()
    module = sys.modules["bot_core.logging.app"]

    logger = module.setup_app_logging()
    assert logger.level == logging.DEBUG


def test_logging_env_allows_custom_log_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "alt.log"
    monkeypatch.setenv("BOT_CORE_LOG_FILE", str(target))

    _reload_logging_app()
    module = sys.modules["bot_core.logging.app"]

    default_path = getattr(module, "DEFAULT_LOG_FILE")
    assert default_path == target


def test_setup_app_logging_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOT_CORE_LOG_DIR", str(tmp_path))

    _reload_logging_app()
    module = sys.modules["bot_core.logging.app"]

    first = module.setup_app_logging()
    second = module.setup_app_logging()

    assert first is second
    assert len(first.handlers) == 1
    assert isinstance(first.handlers[0], QueueHandler)


def test_setup_app_logging_attaches_vector_handler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOT_CORE_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("BOT_CORE_LOG_SHIP_VECTOR", "http://localhost:8686/logs")

    _reload_logging_app()
    module = sys.modules["bot_core.logging.app"]

    logger = module.setup_app_logging()
    listener = getattr(module, "_LISTENER")

    assert len(logger.handlers) == 1
    assert any(isinstance(handler, module.VectorHttpHandler) for handler in listener.handlers)


def test_get_logger_returns_named_child(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOT_CORE_LOG_DIR", str(tmp_path))

    _reload_logging_app()
    module = sys.modules["bot_core.logging.app"]

    configured = module.setup_app_logging()
    child = module.get_logger("bot_core.subsystem")

    assert child.name == "bot_core.subsystem"
    assert module.get_logger() is logging.getLogger(configured.name)


def test_setup_app_logging_registers_atexit_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BOT_CORE_LOG_DIR", str(tmp_path))

    _reload_logging_app()
    module = sys.modules["bot_core.logging.app"]

    registered: list[object] = []
    monkeypatch.setattr(module.atexit, "register", lambda callback: registered.append(callback))

    module.setup_app_logging()
    module.setup_app_logging()

    assert registered == [module._stop_queue_listener]
    module._stop_queue_listener()


def test_stop_queue_listener_unregisters_from_atexit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BOT_CORE_LOG_DIR", str(tmp_path))

    _reload_logging_app()
    module = sys.modules["bot_core.logging.app"]

    monkeypatch.setattr(module.atexit, "register", lambda callback: None)
    unregistered: list[object] = []
    monkeypatch.setattr(module.atexit, "unregister", lambda callback: unregistered.append(callback))

    module.setup_app_logging()
    assert module._LISTENER is not None

    module._stop_queue_listener()

    assert module._LISTENER is None
    assert unregistered == [module._stop_queue_listener]

