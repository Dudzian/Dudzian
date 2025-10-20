from __future__ import annotations

import importlib
import logging
from datetime import datetime, timedelta, timezone
from logging.handlers import QueueHandler, RotatingFileHandler
from pathlib import Path
from typing import Dict

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.security import (
    EncryptedFileSecretStorage,
    RotationRegistry,
    RotationStatus,
    SecretManager,
    SecretStorage,
    SecretStorageError,
)
from KryptoLowca.security import KeyRotationManager


class _MemorySecretStorage(SecretStorage):
    """Lekki magazyn sekretów wykorzystywany w testach."""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    def get_secret(self, key: str) -> str | None:  # pragma: no cover - prosta logika
        return self._store.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self._store[key] = value

    def delete_secret(self, key: str) -> None:
        self._store.pop(key, None)


@pytest.mark.parametrize("fmt", ["json", "text"])
def test_logging_setup_supports_json(fmt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_rotation_registry_marks_and_reports(tmp_path: Path) -> None:
    registry = RotationRegistry(tmp_path / "rotation.json")
    status = registry.status("binance", "trading", interval_days=1)
    assert isinstance(status, RotationStatus)
    assert status.last_rotated is None
    assert status.is_due is True
    assert status.is_overdue is True

    reference_time = datetime.now(timezone.utc)
    registry.mark_rotated("binance", "trading", timestamp=reference_time)
    updated = registry.status("binance", "trading", interval_days=1, now=reference_time + timedelta(hours=12))
    assert updated.last_rotated is not None
    assert updated.is_due is False
    assert updated.is_overdue is False
    assert 0.4 < updated.days_since_rotation < 0.6


def test_secret_manager_encrypted_storage_roundtrip(tmp_path: Path) -> None:
    storage_path = tmp_path / "secrets.enc"
    try:
        storage = EncryptedFileSecretStorage(storage_path, passphrase="unit-pass")
    except SecretStorageError as exc:  # pragma: no cover - brak cryptography
        pytest.skip(f"Encrypted storage unavailable: {exc}")

    manager = SecretManager(storage, namespace="tests.demo")
    credentials = ExchangeCredentials(
        key_id="paper-key",
        secret="A" * 64,
        passphrase=None,
        environment=Environment.PAPER,
        permissions=("trade", "read"),
    )
    manager.store_exchange_credentials("binance", credentials)

    loaded = manager.load_exchange_credentials(
        "binance",
        expected_environment=Environment.PAPER,
        required_permissions=("read",),
    )
    assert loaded.key_id == "paper-key"
    assert loaded.secret == "A" * 64


def test_secret_manager_memory_storage_handles_missing() -> None:
    storage = _MemorySecretStorage()
    manager = SecretManager(storage, namespace="tests.memory")

    manager.store_secret_value("alert", "token-123")
    assert manager.load_secret_value("alert") == "token-123"

    manager.delete_secret_value("alert")
    with pytest.raises(SecretStorageError):
        manager.load_secret_value("alert")


def test_key_rotation_manager_rotates_exchange_credentials(tmp_path: Path) -> None:
    storage = _MemorySecretStorage()
    manager = SecretManager(storage, namespace="tests.rotation")
    rotation = KeyRotationManager(manager, registry_path=tmp_path / "rotation.json", default_interval_days=30)

    manager.store_exchange_credentials(
        "binance_demo",
        ExchangeCredentials(
            key_id="demo-key",
            secret="s" * 32,
            passphrase="hunter2",
            environment=Environment.PAPER,
            permissions=("trade", "read"),
        ),
    )

    status_before = rotation.status("binance_demo")
    assert status_before.is_due is True

    def rotate(payload: ExchangeCredentials) -> ExchangeCredentials:
        return ExchangeCredentials(
            key_id=payload.key_id,
            secret="n" * 32,
            passphrase=payload.passphrase,
            environment=payload.environment,
            permissions=payload.permissions,
        )

    updated = rotation.rotate_exchange_credentials(
        "binance_demo",
        expected_environment=Environment.PAPER,
        rotation_callback=rotate,
    )

    assert updated.secret == "n" * 32
    status_after = rotation.status("binance_demo")
    assert status_after.is_due is False
    assert status_after.last_rotated is not None


def test_key_rotation_manager_ensure_rotation_uses_callback(tmp_path: Path) -> None:
    storage = _MemorySecretStorage()
    manager = SecretManager(storage, namespace="tests.rotation.ensure")
    rotation = KeyRotationManager(manager, registry_path=tmp_path / "ensure.json", default_interval_days=1)

    manager.store_exchange_credentials(
        "binance_demo",
        ExchangeCredentials(
            key_id="demo-key",
            secret="s" * 32,
            passphrase=None,
            environment=Environment.PAPER,
            permissions=("trade",),
        ),
    )

    invoked: list[str] = []

    def rotate(payload: ExchangeCredentials) -> ExchangeCredentials:
        invoked.append(payload.secret or "")
        return ExchangeCredentials(
            key_id=payload.key_id,
            secret="updated" * 4,
            passphrase=payload.passphrase,
            environment=payload.environment,
            permissions=payload.permissions,
        )

    state = rotation.ensure_exchange_rotation(
        "binance_demo",
        expected_environment=Environment.PAPER,
        rotation_callback=rotate,
    )

    assert state.was_rotated is True
    assert invoked == ["s" * 32]
    status = rotation.status("binance_demo")
    assert status.last_rotated is not None

    state_again = rotation.ensure_exchange_rotation(
        "binance_demo",
        expected_environment=Environment.PAPER,
        rotation_callback=rotate,
    )

    assert state_again.was_rotated is False
    assert invoked == ["s" * 32]
