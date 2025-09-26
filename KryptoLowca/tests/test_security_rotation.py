from __future__ import annotations

import importlib
import logging
from datetime import datetime
from logging.handlers import QueueHandler, RotatingFileHandler

import pytest

from KryptoLowca.managers.security_manager import SecurityManager
from KryptoLowca.security import KeyRotationManager, SecretBackend, SecretManager


@pytest.mark.parametrize("fmt", ["json", "text"])
def test_logging_setup_supports_json(fmt: str, tmp_path, monkeypatch):
    monkeypatch.setenv("KRYPT_LOWCA_LOG_FORMAT", fmt)
    monkeypatch.setenv("KRYPT_LOWCA_LOG_DIR", str(tmp_path))
    import KryptoLowca.logging_utils as logging_utils

    importlib.reload(logging_utils)
    log_file = tmp_path / "test.log"
    root_logger = logging.getLogger("KryptoLowca")
    root_logger.handlers.clear()
    if hasattr(root_logger, "_krypto_logging_configured"):
        delattr(root_logger, "_krypto_logging_configured")
    logging_utils.setup_app_logging(log_file=log_file)
    logger = logging_utils.get_logger("KryptoLowca.test")
    assert root_logger.handlers, "queue handler should be attached"
    handler = root_logger.handlers[0]
    assert isinstance(handler, QueueHandler)
    listener = getattr(logging_utils, "_LISTENER", None)
    assert listener is not None
    rotating = None
    for h in listener.handlers:
        if isinstance(h, RotatingFileHandler):
            rotating = h
            break
    assert rotating is not None, "rotating file handler required"
    formatter = rotating.formatter
    if fmt == "json" and getattr(logging_utils, "jsonlogger", None) is not None:
        assert formatter.__class__.__module__.startswith("pythonjsonlogger"), "expected JSON formatter"
    else:
        assert isinstance(formatter, logging.Formatter)
    listener.stop()
    logging_utils._LISTENER = None
    logging_utils._QUEUE = None


def test_key_rotation_manager(tmp_path):
    key_file = tmp_path / "api.enc"
    salt_file = tmp_path / "salt.bin"
    mgr = SecurityManager(key_file=key_file, salt_file=salt_file)
    mgr.save_encrypted_keys({"binance": "A" * 64}, "passphrase")
    rot = KeyRotationManager(mgr, rotation_days=1)

    status_before = rot.status()
    assert status_before.rotation_required is True  # brak metadanych => wymusza rotację

    status_after = rot.ensure_rotation("passphrase")
    assert status_after.last_rotation is not None
    assert status_after.rotation_required is False

    loaded = mgr.load_encrypted_keys("passphrase")
    assert loaded["binance"].startswith("A")


def test_secret_manager_file_backend(tmp_path):
    secrets_file = tmp_path / "secrets.json"
    manager = SecretManager(backend=SecretBackend.FILE, file_path=secrets_file)
    manager.set_secret("BINANCE_KEY", "demo-value")

    assert manager.get_secret("BINANCE_KEY") == "demo-value"
    last = manager.last_rotation("BINANCE_KEY")
    assert isinstance(last, datetime)
    assert last.tzinfo is not None


def test_secret_manager_env_backend(monkeypatch):
    monkeypatch.setenv("TEST_TOKEN", "seed")
    manager = SecretManager(backend=SecretBackend.ENV, prefix="TEST_")
    manager.set_secret("TOKEN", "123")
    assert manager.get_secret("TOKEN") == "123"
    assert manager.last_rotation("TOKEN") is not None
