from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Callable, Tuple

import pytest

from core.security.secret_store import ExchangeCredentials, SecretStore, SecretStoreError
from bot_core.security.base import SecretStorageError
from bot_core.security.keyring_storage import KeyringSecretStorage
from bot_core.security.hwid import HwIdProvider


class _InMemoryKeyring:
    class errors:
        class PasswordDeleteError(Exception):
            pass

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


class _StaticHwIdProvider(HwIdProvider):
    def __init__(self, value: str = "unit-hwid") -> None:
        super().__init__(fingerprint_reader=lambda: value)


@pytest.fixture
def secret_store_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[dict | None], Tuple[SecretStore, _InMemoryKeyring, Path]]:
    def factory(legacy_payload: dict | None = None) -> Tuple[SecretStore, _InMemoryKeyring, Path]:
        backend = _InMemoryKeyring()
        module = types.ModuleType("keyring")
        module.get_password = backend.get_password  # type: ignore[assignment]
        module.set_password = backend.set_password  # type: ignore[assignment]
        module.delete_password = backend.delete_password  # type: ignore[assignment]
        module.errors = backend.errors  # type: ignore[assignment]
        module.get_keyring = lambda: backend
        module.set_keyring = lambda _: None
        monkeypatch.setitem(sys.modules, "keyring", module)

        monkeypatch.setattr(
            KeyringSecretStorage,
            "_ensure_native_backend",
            lambda self, keyring_module: backend,
            raising=False,
        )

        index_path = tmp_path / "secret_index.json"
        legacy_path = tmp_path / "api_credentials.json"
        if legacy_payload is not None:
            legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

        storage = KeyringSecretStorage(
            service_name="tests.desktop",
            hwid_provider=_StaticHwIdProvider(),
            index_path=index_path,
        )

        store = SecretStore(storage=storage, legacy_path=legacy_path)
        return store, backend, legacy_path

    return factory


def test_secret_store_saves_and_loads_credentials(secret_store_factory: Callable[[dict | None], Tuple[SecretStore, _InMemoryKeyring, Path]]) -> None:
    store, backend, _legacy = secret_store_factory()

    credentials = ExchangeCredentials(
        exchange="Binance",
        api_key="APIKEY",
        api_secret="SECRET",
        api_passphrase="phrase",
    )

    store.save_exchange_credentials(credentials)

    listed = store.list_exchanges()
    assert listed == ("binance",)

    loaded = store.load_exchange_credentials("BINANCE")
    assert loaded.api_key == "APIKEY"
    assert loaded.api_secret == "SECRET"
    assert loaded.api_passphrase == "phrase"

    stored_keys = {key for (service, key) in backend._store.keys() if service == "tests.desktop"}
    assert "desktop.exchange:binance" in stored_keys


def test_secret_store_migrates_plaintext_file(secret_store_factory: Callable[[dict | None], Tuple[SecretStore, _InMemoryKeyring, Path]]) -> None:
    legacy_payload = {
        "binance": {"api_key": "LEGACY", "api_secret": "SECRET", "api_passphrase": ""},
        "  ": {"api_key": "", "api_secret": ""},
    }
    store, backend, legacy_path = secret_store_factory(legacy_payload)

    exchanges = store.list_exchanges()
    assert exchanges == ("binance",)

    loaded = store.load_exchange_credentials("binance")
    assert loaded.api_key == "LEGACY"
    assert loaded.api_secret == "SECRET"
    assert loaded.api_passphrase is None

    assert not legacy_path.exists()
    assert legacy_path.with_suffix(".legacy").exists()

    stored_keys = {key for (service, key) in backend._store.keys() if service == "tests.desktop"}
    assert "desktop.exchange:binance" in stored_keys


def test_secret_store_rotate_master_key_delegates(secret_store_factory: Callable[[dict | None], Tuple[SecretStore, _InMemoryKeyring, Path]]) -> None:
    store, _, _ = secret_store_factory()

    called = {"value": False}

    def _patched_rotate(self: KeyringSecretStorage) -> None:
        called["value"] = True

    original = store._storage.rotate_master_key
    store._storage.rotate_master_key = _patched_rotate.__get__(store._storage, KeyringSecretStorage)  # type: ignore[assignment]
    try:
        store.rotate_master_key()
    finally:
        store._storage.rotate_master_key = original  # type: ignore[assignment]

    assert called["value"] is True


def test_secret_store_errors_are_wrapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FailingStorage(KeyringSecretStorage):
        def __init__(self) -> None:
            pass

        def set_secret(self, key: str, value: str) -> None:  # type: ignore[override]
            raise SecretStorageError("błąd zapisu")

        def get_secret(self, key: str) -> str | None:  # type: ignore[override]
            raise SecretStorageError("błąd odczytu")

        def list_registered_keys(self) -> tuple[str, ...]:  # type: ignore[override]
            raise SecretStorageError("błąd listy")

        def rotate_master_key(self) -> None:  # type: ignore[override]
            raise SecretStorageError("błąd rotacji")

    monkeypatch.setenv("HOME", str(tmp_path))
    store = SecretStore(storage=_FailingStorage(), legacy_path=tmp_path / "api_credentials.json")

    with pytest.raises(SecretStoreError) as excinfo:
        store.save_exchange_credentials(ExchangeCredentials(exchange="binance", api_key="k", api_secret="s"))
    assert "błąd zapisu" in str(excinfo.value)

    with pytest.raises(SecretStoreError) as excinfo:
        store.list_exchanges()
    assert "błąd listy" in str(excinfo.value)

    with pytest.raises(SecretStoreError) as excinfo:
        store.rotate_master_key()
    assert "błąd rotacji" in str(excinfo.value)

    with pytest.raises(SecretStoreError) as excinfo:
        store.load_exchange_credentials("binance")
    assert "błąd odczytu" in str(excinfo.value)


def test_security_details_token_is_exposed(secret_store_factory: Callable[[dict | None], Tuple[SecretStore, _InMemoryKeyring, Path]]) -> None:
    store, _, _ = secret_store_factory()
    assert store.security_details_token() == "onboarding.strategy.credentials.secured"
