"""Testy warstwy zarządzania sekretami wykorzystującej natywne keychainy."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.security import (
    EncryptedFileSecretStorage,
    KeyringSecretStorage,
    SecretManager,
    SecretStorageError,
    create_default_secret_storage,
)


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
        "binance_paper_trading",
        expected_environment=Environment.PAPER,
        required_permissions=("read", "trade"),
        forbidden_permissions=("withdraw",),
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


def test_required_permissions_are_enforced() -> None:
    storage = KeyringSecretStorage(service_name="unit.test")
    manager = SecretManager(storage)

    credentials = ExchangeCredentials(
        key_id="abc",
        secret="sekret",
        passphrase=None,
        environment=Environment.PAPER,
        permissions=("read",),
    )

    manager.store_exchange_credentials("binance_paper", credentials)

    with pytest.raises(SecretStorageError) as excinfo:
        manager.load_exchange_credentials(
            "binance_paper",
            expected_environment=Environment.PAPER,
            required_permissions=("read", "trade"),
        )

    assert "wymaganych uprawnień" in str(excinfo.value)


def test_forbidden_permissions_are_detected() -> None:
    storage = KeyringSecretStorage(service_name="unit.test")
    manager = SecretManager(storage)

    credentials = ExchangeCredentials(
        key_id="abc",
        secret="sekret",
        passphrase=None,
        environment=Environment.PAPER,
        permissions=("read", "trade", "withdraw"),
    )

    manager.store_exchange_credentials("binance_live", credentials)

    with pytest.raises(SecretStorageError) as excinfo:
        manager.load_exchange_credentials(
            "binance_live",
            expected_environment=Environment.PAPER,
            forbidden_permissions=("withdraw",),
        )

    assert "zabronione uprawnienia" in str(excinfo.value)


def test_store_and_load_generic_secret() -> None:
    storage = KeyringSecretStorage(service_name="unit.test")
    manager = SecretManager(storage, namespace="tests")

    manager.store_secret_value("telegram_bot", "token123", purpose="alerts")
    loaded = manager.load_secret_value("telegram_bot", purpose="alerts")

    assert loaded == "token123"

    manager.delete_secret_value("telegram_bot", purpose="alerts")

    with pytest.raises(SecretStorageError):
        manager.load_secret_value("telegram_bot", purpose="alerts")


def test_encrypted_file_secret_storage_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("cryptography.fernet")

    storage_path = tmp_path / "secrets.age"
    storage = EncryptedFileSecretStorage(storage_path, passphrase="haslo")

    assert storage.get_secret("api") is None
    storage.set_secret("api", "sekret")
    assert storage.get_secret("api") == "sekret"

    storage.delete_secret("api")
    assert storage.get_secret("api") is None

    storage_again = EncryptedFileSecretStorage(storage_path, passphrase="haslo")
    assert storage_again.get_secret("api") is None


def test_encrypted_file_secret_storage_persists_between_instances(tmp_path: Path) -> None:
    pytest.importorskip("cryptography.fernet")

    storage_path = tmp_path / "secrets.age"
    first = EncryptedFileSecretStorage(storage_path, passphrase="tajne")
    first.set_secret("binance", "key123")

    second = EncryptedFileSecretStorage(storage_path, passphrase="tajne")
    assert second.get_secret("binance") == "key123"


def test_encrypted_file_secret_storage_rotation_and_backup(tmp_path: Path) -> None:
    pytest.importorskip("cryptography.fernet")

    storage_path = tmp_path / "vault.age"
    storage = EncryptedFileSecretStorage(storage_path, passphrase="pierwsze")
    storage.set_secret("alpha", "secret-alpha")

    storage.rotate_passphrase("drugie", iterations=100_000)
    assert storage.get_secret("alpha") == "secret-alpha"
    rotated = EncryptedFileSecretStorage(storage_path, passphrase="drugie")
    assert rotated.get_secret("alpha") == "secret-alpha"

    snapshot = rotated.export_backup()
    restored_path = tmp_path / "restored.age"
    restored = EncryptedFileSecretStorage.recover_from_backup(restored_path, "drugie", snapshot)
    assert restored.get_secret("alpha") == "secret-alpha"


def test_create_default_secret_storage_linux_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setenv("WAYLAND_DISPLAY", "")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "address")
    monkeypatch.setattr("platform.system", lambda: "Linux")

    storage = create_default_secret_storage(namespace="unit.gui")
    assert isinstance(storage, KeyringSecretStorage)


def test_create_default_secret_storage_linux_headless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    pytest.importorskip("cryptography.fernet")

    storage = create_default_secret_storage(
        namespace="unit.headless",
        headless_passphrase="bardzotajne",
        headless_path=tmp_path / "vault.age",
    )

    assert isinstance(storage, EncryptedFileSecretStorage)
    storage.set_secret("kraken", "sekret")

    reloaded = EncryptedFileSecretStorage(tmp_path / "vault.age", passphrase="bardzotajne")
    assert reloaded.get_secret("kraken") == "sekret"


def test_create_default_secret_storage_headless_requires_passphrase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    with pytest.raises(SecretStorageError):
        create_default_secret_storage(headless_path=tmp_path / "vault.age")
