from __future__ import annotations

import base64
import json
import sys
import types
from pathlib import Path

import pytest

from bot_core.security import fingerprint as fp
from bot_core.security.fingerprint import FingerprintError


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
        self._store.pop((service, username), None)


@pytest.fixture(autouse=True)
def _reset_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOT_CORE_UI_PYTHON", raising=False)


def test_license_secret_uses_keyring_and_encrypts_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backend = _InMemoryKeyring()
    module = types.ModuleType("keyring")
    module.get_password = backend.get_password  # type: ignore[assignment]
    module.set_password = backend.set_password  # type: ignore[assignment]
    module.delete_password = backend.delete_password  # type: ignore[assignment]
    module.errors = backend.errors  # type: ignore[assignment]
    module.get_keyring = lambda: backend
    module.set_keyring = lambda _: None
    monkeypatch.setitem(sys.modules, "keyring", module)

    monkeypatch.setattr(fp, "get_local_fingerprint", lambda: "DEVICE-001")

    secret_path = tmp_path / "license_secret.key"
    secret = fp.load_license_secret(secret_path)
    assert len(secret) == 48

    stored = backend._store.get((fp.LICENSE_SECRET_KEYRING_SERVICE, fp.LICENSE_SECRET_KEYRING_ENTRY))
    assert stored is not None
    document = json.loads(secret_path.read_text(encoding="utf-8"))
    assert document["version"] == fp.LICENSE_SECRET_FILE_VERSION
    assert document["hwid_digest"]


def test_license_secret_migrates_plaintext_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fp, "_load_secret_from_keyring", lambda: None, raising=False)
    monkeypatch.setattr(fp, "_store_secret_in_keyring", lambda _secret: False, raising=False)
    monkeypatch.setattr(fp, "get_local_fingerprint", lambda: "DEVICE-ABC")

    secret_path = tmp_path / "license_secret.key"
    secret_bytes = b"S" * 48
    secret_path.write_text(base64.b64encode(secret_bytes).decode("ascii"), encoding="utf-8")

    loaded = fp.load_license_secret(secret_path)
    assert loaded == secret_bytes

    payload = json.loads(secret_path.read_text(encoding="utf-8"))
    assert payload["version"] == fp.LICENSE_SECRET_FILE_VERSION
    assert payload["hwid_digest"]


def test_license_secret_rejects_mismatched_fingerprint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fp, "_load_secret_from_keyring", lambda: None, raising=False)
    monkeypatch.setattr(fp, "_store_secret_in_keyring", lambda _secret: False, raising=False)

    secret_path = tmp_path / "license_secret.key"
    monkeypatch.setattr(fp, "get_local_fingerprint", lambda: "LOCAL-123")
    fp.load_license_secret(secret_path)

    monkeypatch.setattr(fp, "get_local_fingerprint", lambda: "OTHER-456")
    with pytest.raises(FingerprintError):
        fp.load_license_secret(secret_path, create=False)
