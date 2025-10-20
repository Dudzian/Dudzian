"""Testy magazynu sekretów `bot_core.security` wykorzystywanego przez aplikację."""

from __future__ import annotations

from typing import Dict, Optional

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.security import SecretManager, SecretStorage, SecretStorageError


class _MemorySecretStorage(SecretStorage):
    """Niewielki magazyn w pamięci pomagający izolować testy."""

    def __init__(self) -> None:
        self.store: Dict[str, str] = {}

    def get_secret(self, key: str) -> Optional[str]:
        return self.store.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self.store[key] = value

    def delete_secret(self, key: str) -> None:
        self.store.pop(key, None)


@pytest.fixture()
def secret_manager() -> SecretManager:
    """Zwraca SecretManager z magazynem in-memory dla prostych testów."""

    storage = _MemorySecretStorage()
    return SecretManager(storage, namespace="tests.security")


def _sample_credentials(environment: Environment = Environment.PAPER) -> ExchangeCredentials:
    return ExchangeCredentials(
        key_id="paper-key",
        secret="S" * 32,
        passphrase="hunter2",
        environment=environment,
        permissions=("trade", "read"),
    )


def test_store_and_load_exchange_credentials(secret_manager: SecretManager) -> None:
    secret_manager.store_exchange_credentials("binance_demo", _sample_credentials())

    loaded = secret_manager.load_exchange_credentials(
        "binance_demo",
        expected_environment=Environment.PAPER,
        required_permissions=("trade",),
    )

    assert loaded.key_id == "paper-key"
    assert loaded.secret == "S" * 32
    assert loaded.environment is Environment.PAPER


def test_environment_mismatch_raises(secret_manager: SecretManager) -> None:
    secret_manager.store_exchange_credentials("binance_demo", _sample_credentials(Environment.LIVE))

    with pytest.raises(SecretStorageError) as excinfo:
        secret_manager.load_exchange_credentials(
            "binance_demo",
            expected_environment=Environment.PAPER,
        )

    assert "nie pasuje" in str(excinfo.value)


def test_permission_checks(secret_manager: SecretManager) -> None:
    secret_manager.store_exchange_credentials("paper", _sample_credentials())

    with pytest.raises(SecretStorageError):
        secret_manager.load_exchange_credentials(
            "paper",
            expected_environment=Environment.PAPER,
            required_permissions=("withdraw",),
        )

    secret_manager.store_exchange_credentials(
        "paper",
        _sample_credentials(),
    )

    with pytest.raises(SecretStorageError):
        secret_manager.load_exchange_credentials(
            "paper",
            expected_environment=Environment.PAPER,
            forbidden_permissions=("trade",),
        )


def test_generic_secret_roundtrip(secret_manager: SecretManager) -> None:
    secret_manager.store_secret_value("smtp", "hunter2", purpose="email")

    loaded = secret_manager.load_secret_value("smtp", purpose="email")
    assert loaded == "hunter2"

    secret_manager.delete_secret_value("smtp", purpose="email")
    with pytest.raises(SecretStorageError):
        secret_manager.load_secret_value("smtp", purpose="email")


def test_missing_exchange_secret_is_reported(secret_manager: SecretManager) -> None:
    with pytest.raises(SecretStorageError) as excinfo:
        secret_manager.load_exchange_credentials(
            "unknown",
            expected_environment=Environment.PAPER,
        )

    assert "Brak sekretu" in str(excinfo.value)

