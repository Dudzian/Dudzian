from __future__ import annotations

import sys
import types

import pytest

from bot_core.security.keyring_storage import KeyringSecretStorage
from bot_core.security.base import SecretStorageError


class DummyBackend:
    def __init__(self) -> None:
        self._secrets: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, key: str) -> str | None:
        return self._secrets.get((service, key))

    def set_password(self, service: str, key: str, value: str) -> None:
        self._secrets[(service, key)] = value

    def delete_password(self, service: str, key: str) -> None:
        self._secrets.pop((service, key), None)


class DummyHwId:
    def read(self) -> str:
        return "stub-fingerprint"


def _install_keyring_stub(monkeypatch: pytest.MonkeyPatch, backend: DummyBackend | None = None) -> types.ModuleType:
    module = types.ModuleType("keyring")
    errors = types.SimpleNamespace(PasswordDeleteError=Exception)
    current_backend = backend or DummyBackend()

    def get_password(service: str, key: str) -> str | None:
        return current_backend.get_password(service, key)

    def set_password(service: str, key: str, value: str) -> None:
        result = current_backend.set_password(service, key, value)
        return result

    def delete_password(service: str, key: str) -> None:
        current_backend.delete_password(service, key)

    def get_keyring() -> DummyBackend:
        return current_backend

    def set_keyring(new_backend: DummyBackend) -> None:
        nonlocal current_backend
        current_backend = new_backend

    module.get_password = get_password  # type: ignore[assignment]
    module.set_password = set_password  # type: ignore[assignment]
    module.delete_password = delete_password  # type: ignore[assignment]
    module.get_keyring = get_keyring  # type: ignore[assignment]
    module.set_keyring = set_keyring  # type: ignore[assignment]
    module.errors = errors  # type: ignore[assignment]

    monkeypatch.setitem(sys.modules, "keyring", module)
    return module


def _install_backend_module(monkeypatch: pytest.MonkeyPatch, module_name: str, class_name: str) -> type[DummyBackend]:
    package, _, leaf = module_name.rpartition(".")
    if package:
        package_module = sys.modules.get(package)
        if package_module is None:
            package_module = types.ModuleType(package)
            package_module.__path__ = []  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, package, package_module)
    backend_module = types.ModuleType(module_name)

    class NativeBackend(DummyBackend):
        pass

    setattr(backend_module, class_name, NativeBackend)
    monkeypatch.setitem(sys.modules, module_name, backend_module)
    if package:
        setattr(sys.modules[package], leaf, backend_module)
    return NativeBackend


def test_linux_backend_promotes_secret_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    keyring_module = _install_keyring_stub(monkeypatch)
    native_cls = _install_backend_module(
        monkeypatch, "keyring.backends.SecretService", "SecretServiceKeyring"
    )

    storage = KeyringSecretStorage(hwid_provider=DummyHwId())

    assert isinstance(storage._backend, native_cls)  # type: ignore[attr-defined]
    storage.set_secret("api", "value")
    assert keyring_module.get_password("dudzian.trading", "api") is not None


def test_windows_backend_promotes_win_vault(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32", raising=False)
    _install_keyring_stub(monkeypatch)
    native_cls = _install_backend_module(
        monkeypatch, "keyring.backends.Windows", "WinVaultKeyring"
    )

    storage = KeyringSecretStorage(hwid_provider=DummyHwId())
    assert isinstance(storage._backend, native_cls)  # type: ignore[attr-defined]


def test_macos_backend_promotes_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)
    _install_keyring_stub(monkeypatch)
    native_cls = _install_backend_module(
        monkeypatch, "keyring.backends.macOS", "Keyring"
    )

    storage = KeyringSecretStorage(hwid_provider=DummyHwId())
    assert isinstance(storage._backend, native_cls)  # type: ignore[attr-defined]


def test_missing_native_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    _install_keyring_stub(monkeypatch)
    # brak modułu SecretService – import zakończy się błędem

    with pytest.raises(SecretStorageError):
        KeyringSecretStorage(hwid_provider=DummyHwId())
