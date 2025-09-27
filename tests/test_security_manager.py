"""Testy warstwy zarządzania sekretami wykorzystującej natywne keychainy."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.security import KeyringSecretStorage, SecretManager, SecretStorageError


class _InMemoryKeyring:
    """Minimalna implementacja back-endu keyring na potrzeby testów."""

    class errors:
        class PasswordDeleteError(Exception):
            """Wyjątek naśladujący oryginalny interfejs biblioteki keyring."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        try:
            del self._store[(service, username)]
        except KeyError as exc:
            raise self.errors.PasswordDeleteError(str(exc)) from exc


@pytest.fixture(autouse=True)
def fake_keyring() -> types.ModuleType:
    """Podmienia moduł ``keyring`` na wariant in-memory, aby testy były deterministyczne."""

    module = types.ModuleType("keyring")
    backend = _InMemoryKeyring()
    module.get_password = backend.get_password
    module.set_password = backend.set_password
    module.delete_password = backend.delete_password
    module.errors = backend.errors
    sys.modules["keyring"] = module
    yield module
    sys.modules.pop("keyring", None)


def test_roundtrip_store_and_load_credentials() -> None:
    storage = KeyringSecretStorage(service_name="unit.test")
    manager = SecretManager(storage, namespace="tests")

    credentials = ExchangeCredentials(
        key_id="abc",
        secret="topsecret",
        passphrase="phrase",
        environment=Environment.PAPER,
        permissions=("read", "trade"),
    )

    manager.store_exchange_credentials("binance_paper_trading", credentials)
    loaded = manager.load_exchange_credentials(
        "binance_paper_trading", expected_environment=Environment.PAPER
    )

    assert loaded.key_id == credentials.key_id
    assert loaded.secret == credentials.secret
    assert loaded.passphrase == credentials.passphrase
    assert loaded.permissions == credentials.permissions
    assert loaded.environment == Environment.PAPER


def test_load_missing_secret_raises_error() -> None:
    storage = KeyringSecretStorage(service_name="unit.test")
    manager = SecretManager(storage)

    with pytest.raises(SecretStorageError):
        manager.load_exchange_credentials("missing", expected_environment=Environment.LIVE)


def test_environment_mismatch_detected() -> None:
    storage = KeyringSecretStorage(service_name="unit.test")
    manager = SecretManager(storage)

    credentials = ExchangeCredentials(
        key_id="abc",
        secret="topsecret",
        passphrase=None,
        environment=Environment.PAPER,
        permissions=("read",),
    )

    manager.store_exchange_credentials("binance_paper", credentials)

    with pytest.raises(SecretStorageError) as excinfo:
        manager.load_exchange_credentials("binance_paper", expected_environment=Environment.LIVE)

    assert "nie pasuje" in str(excinfo.value)


def test_store_and_load_generic_secret() -> None:
    storage = KeyringSecretStorage(service_name="unit.test")
    manager = SecretManager(storage, namespace="tests")

    manager.store_secret_value("telegram_bot", "token123", purpose="alerts")
    loaded = manager.load_secret_value("telegram_bot", purpose="alerts")

    assert loaded == "token123"

    manager.delete_secret_value("telegram_bot", purpose="alerts")

    with pytest.raises(SecretStorageError):
        manager.load_secret_value("telegram_bot", purpose="alerts")
