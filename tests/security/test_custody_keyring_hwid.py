"""Synthetic custody envelopes; no native credential collection is accessed."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from bot_core import local_signing_custody as custody
from bot_core.root_proof_issuer_substrate import (
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)
from bot_core.security.base import SecretStorageError
from bot_core.security.custody_hwid import CUSTODY_HWID_PREFIX, CUSTODY_HWID_SCHEME
from bot_core.security.hwid import HwIdProvider, HwIdProviderError
from bot_core.security import keyring_storage
from bot_core.security.keyring_storage import KeyringSecretStorage


SYNTHETIC_HWID = CUSTODY_HWID_PREFIX + "a" * 96


@pytest.fixture
def synthetic_native(monkeypatch):
    records: dict[tuple[str, str], str] = {}

    class Keyring:
        def __init__(self):
            self.get_password = Mock(side_effect=lambda service, key: records.get((service, key)))
            self.set_password = Mock(
                side_effect=lambda service, key, value: records.__setitem__((service, key), value)
            )
            self.delete_password = Mock(
                side_effect=lambda service, key: records.pop((service, key), None)
            )

    Keyring.__module__ = "keyring.backends.SecretService"
    backend = Keyring()
    module = ModuleType("keyring")
    module.get_keyring = lambda: backend
    module.get_password = backend.get_password
    module.set_password = backend.set_password
    module.delete_password = backend.delete_password
    module.errors = SimpleNamespace(PasswordDeleteError=KeyError)
    backends = ModuleType("keyring.backends")
    native = ModuleType("keyring.backends.SecretService")
    native.Keyring = Keyring
    monkeypatch.setitem(sys.modules, "keyring", module)
    monkeypatch.setitem(sys.modules, "keyring.backends", backends)
    monkeypatch.setitem(sys.modules, "keyring.backends.SecretService", native)
    monkeypatch.setattr(KeyringSecretStorage, "_platform_identifier", lambda self: "linux")
    monkeypatch.setattr(keyring_storage.CustodyHwIdProvider, "read", lambda self: SYNTHETIC_HWID)
    return records, backend, module


def _storage(index_path: Path, *, custody_hwid: bool = True) -> KeyringSecretStorage:
    return KeyringSecretStorage(
        service_name="synthetic-custody", index_path=index_path, custody_hwid=custody_hwid
    )


def test_versioned_custody_envelopes_survive_new_storage_instance(
    synthetic_native, tmp_path, monkeypatch
):
    records, backend, module = synthetic_native
    path = tmp_path / "index.json"
    writer = _storage(path)
    writer.set_secret("seed", "synthetic seed")
    for key in (writer.MASTER_KEY_SLOT, "seed"):
        document = json.loads(records[(writer._service_name, key)])
        assert document["version"] == 2
        assert document["hwid_scheme"] == CUSTODY_HWID_SCHEME
        assert document["hwid_digest"] == writer._read_hwid_digest()
    before = dict(records)
    index_before = path.read_bytes()
    writes = backend.set_password.call_count
    reader = _storage(path)
    monkeypatch.setattr(module, "get_password", Mock(side_effect=AssertionError("global read")))
    monkeypatch.setattr(keyring_storage.os, "urandom", Mock(side_effect=AssertionError("random")))
    assert reader.get_existing_native_secret("seed") == "synthetic seed"
    assert records == before
    assert path.read_bytes() == index_before
    assert backend.set_password.call_count == writes


@pytest.mark.parametrize("record_key", [KeyringSecretStorage.MASTER_KEY_SLOT, "seed"])
@pytest.mark.parametrize(
    "field,value,match",
    [
        ("hwid_scheme", None, "controlled migration"),
        ("hwid_scheme", "unknown-v2", "controlled migration"),
        ("version", None, "version"),
        ("version", True, "version"),
        ("version", "2", "version"),
        ("version", 3, "version"),
        ("hwid_digest", None, "HWID digest"),
        ("hwid_digest", "", "HWID digest"),
        ("hwid_digest", False, "HWID digest"),
        ("hwid_digest", "a" * 63, "HWID digest"),
        ("hwid_digest", "A" * 64, "HWID digest"),
        ("hwid_digest", "0" * 64, "HWID digest"),
    ],
)
def test_invalid_binding_refuses_without_mutation(
    synthetic_native, tmp_path, record_key, field, value, match
):
    records, backend, _ = synthetic_native
    path = tmp_path / "index.json"
    storage = _storage(path)
    storage.set_secret("seed", "synthetic seed")
    slot = (storage._service_name, record_key)
    document = json.loads(records[slot])
    if value is None:
        document.pop(field)
    else:
        document[field] = value
    records[slot] = json.dumps(document)
    before = dict(records)
    writes = backend.set_password.call_count
    index_before = path.read_bytes()
    for action in (
        lambda: storage.get_existing_native_secret("seed"),
        lambda: storage.set_secret("seed", "incompatible overwrite forbidden"),
    ):
        with pytest.raises(SecretStorageError, match=match):
            action()
    assert records == before
    assert path.read_bytes() == index_before
    assert backend.set_password.call_count == writes


@pytest.mark.parametrize("record_key", [KeyringSecretStorage.MASTER_KEY_SLOT, "seed"])
@pytest.mark.parametrize("raw", ["null", "[]", "true", "42", '"text"', "{"])
def test_malformed_custody_record_refuses(synthetic_native, tmp_path, record_key, raw):
    records, backend, _ = synthetic_native
    storage = _storage(tmp_path / "index.json")
    storage.set_secret("seed", "synthetic seed")
    records[(storage._service_name, record_key)] = raw
    before = dict(records)
    writes = backend.set_password.call_count
    for action in (
        lambda: storage.get_existing_native_secret("seed"),
        lambda: storage.set_secret("seed", "malformed overwrite forbidden"),
    ):
        with pytest.raises(SecretStorageError):
            action()
    assert records == before
    assert backend.set_password.call_count == writes


@pytest.mark.parametrize("record_key", [KeyringSecretStorage.MASTER_KEY_SLOT, "seed"])
@pytest.mark.parametrize("field", ["hwid_digest", "hwid_scheme", "version"])
def test_duplicate_custody_binding_fields_refuse_without_writes(
    synthetic_native, tmp_path, record_key, field
):
    records, backend, _ = synthetic_native
    path = tmp_path / "index.json"
    storage = _storage(path)
    storage.set_secret("seed", "synthetic seed")
    slot = (storage._service_name, record_key)
    raw = records[slot]
    # Duplicate even the identical valid value: choosing either occurrence is forbidden.
    extra = json.dumps(field) + ":" + json.dumps(json.loads(raw)[field])
    records[slot] = raw[:-1] + "," + extra + "}"
    before = dict(records)
    index_before = path.read_bytes()
    writes = backend.set_password.call_count
    for action in (
        lambda: storage.get_existing_native_secret("seed"),
        lambda: storage.get_secret("seed"),
        lambda: storage.set_secret("seed", "duplicate overwrite forbidden"),
    ):
        with pytest.raises(SecretStorageError):
            action()
    assert records == before
    assert path.read_bytes() == index_before
    assert backend.set_password.call_count == writes


@pytest.mark.parametrize("record_key", [KeyringSecretStorage.MASTER_KEY_SLOT, "seed"])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_custody_json_refuses_without_writes(
    synthetic_native, tmp_path, record_key, value
):
    records, backend, _ = synthetic_native
    path = tmp_path / "index.json"
    storage = _storage(path)
    storage.set_secret("seed", "synthetic seed")
    slot = (storage._service_name, record_key)
    records[slot] = records[slot][:-1] + ',"extra":' + value + "}"
    before = dict(records)
    index_before = path.read_bytes()
    writes = backend.set_password.call_count
    for action in (
        lambda: storage.get_existing_native_secret("seed"),
        lambda: storage.get_secret("seed"),
        lambda: storage.set_secret("seed", "nonfinite overwrite forbidden"),
    ):
        with pytest.raises(SecretStorageError):
            action()
    assert records == before
    assert path.read_bytes() == index_before
    assert backend.set_password.call_count == writes


def test_legacy_json_parsing_is_unchanged(synthetic_native, tmp_path):
    storage = _storage(tmp_path / "index.json", custody_hwid=False)
    parsed = storage._parse_credential_document('{"version":1,"version":2,"extra":NaN}')
    assert parsed["version"] == 2
    assert parsed["extra"] != parsed["extra"]


def test_changed_custody_device_refuses(synthetic_native, tmp_path, monkeypatch):
    records, backend, _ = synthetic_native
    storage = _storage(tmp_path / "index.json")
    storage.set_secret("seed", "synthetic seed")
    before = dict(records)
    writes = backend.set_password.call_count
    monkeypatch.setattr(
        keyring_storage.CustodyHwIdProvider,
        "read",
        lambda self: CUSTODY_HWID_PREFIX + "b" * 96,
    )
    reader = _storage(tmp_path / "index.json")
    with pytest.raises(SecretStorageError, match="fingerprint mismatch"):
        reader.get_existing_native_secret("seed")
    assert records == before
    assert backend.set_password.call_count == writes


@pytest.mark.parametrize(
    "value",
    [
        None,
        False,
        42,
        "",
        "legacy",
        "a" * 96,
        CUSTODY_HWID_PREFIX + "A" * 96,
        CUSTODY_HWID_PREFIX + "a" * 95,
        SYNTHETIC_HWID + "\n",
        " " + SYNTHETIC_HWID,
    ],
)
def test_malformed_provider_hwid_cannot_create_master(
    synthetic_native, tmp_path, monkeypatch, value
):
    records, backend, _ = synthetic_native
    monkeypatch.setattr(keyring_storage.CustodyHwIdProvider, "read", lambda self: value)
    path = tmp_path / "index.json"
    storage = _storage(path)
    with pytest.raises(SecretStorageError, match="Invalid versioned custody HWID"):
        storage.set_secret("seed", "synthetic seed")
    assert records == {}
    backend.set_password.assert_not_called()
    assert not path.exists()


def test_unavailable_custody_hwid_cannot_create_master(synthetic_native, tmp_path, monkeypatch):
    records, backend, _ = synthetic_native
    monkeypatch.setattr(
        keyring_storage.CustodyHwIdProvider,
        "read",
        Mock(side_effect=HwIdProviderError("protected identity unavailable")),
    )
    storage = _storage(tmp_path / "index.json")
    with pytest.raises(SecretStorageError):
        storage.set_secret("seed", "synthetic seed")
    assert records == {}
    backend.set_password.assert_not_called()


def test_legacy_custody_requires_controlled_migration_without_rewrite(synthetic_native, tmp_path):
    records, backend, _ = synthetic_native
    path = tmp_path / "index.json"
    legacy = KeyringSecretStorage(
        service_name="synthetic-custody",
        index_path=path,
        hwid_provider=HwIdProvider(fingerprint_reader=lambda: "existing-legacy-device"),
    )
    legacy.set_secret("seed", "existing synthetic seed")
    assert legacy.get_existing_native_secret("seed") == "existing synthetic seed"
    assert all("hwid_scheme" not in json.loads(raw) for raw in records.values())
    before = dict(records)
    writes = backend.set_password.call_count
    index_before = path.read_bytes()
    versioned = _storage(path)
    for action in (
        lambda: versioned.get_existing_native_secret("seed"),
        lambda: versioned.set_secret("seed", "replacement forbidden"),
    ):
        with pytest.raises(SecretStorageError, match="controlled migration required"):
            action()
    assert records == before
    assert path.read_bytes() == index_before
    assert backend.set_password.call_count == writes
    assert legacy.get_existing_native_secret("seed") == "existing synthetic seed"


def test_readonly_custody_never_creates_missing_master(synthetic_native, tmp_path, monkeypatch):
    records, backend, _ = synthetic_native
    path = tmp_path / "index.json"
    storage = _storage(path)
    records[(storage._service_name, "seed")] = "existing synthetic ciphertext"
    monkeypatch.setattr(keyring_storage.os, "urandom", Mock(side_effect=AssertionError("random")))
    with pytest.raises(SecretStorageError, match="Existing native master"):
        storage.get_existing_native_secret("seed")
    with pytest.raises(SecretStorageError, match="Existing custody master"):
        storage.set_secret("seed", "orphan overwrite forbidden")
    with pytest.raises(SecretStorageError, match="Existing custody master"):
        storage.get_secret("seed")
    assert storage.get_existing_native_secret("absent") is None
    backend.set_password.assert_not_called()
    backend.delete_password.assert_not_called()
    assert not path.exists()


def test_custody_generic_master_rotation_requires_controlled_ceremony(synthetic_native, tmp_path):
    records, backend, _ = synthetic_native
    path = tmp_path / "index.json"
    storage = _storage(path)
    storage.set_secret("seed", "synthetic seed")
    before = dict(records)
    index_before = path.read_bytes()
    writes = backend.set_password.call_count
    with pytest.raises(SecretStorageError, match="reviewed controlled ceremony"):
        storage.rotate_master_key()
    assert records == before
    assert path.read_bytes() == index_before
    assert backend.set_password.call_count == writes


def test_valid_versioned_custody_secret_can_be_updated(synthetic_native, tmp_path):
    storage = _storage(tmp_path / "index.json")
    storage.set_secret("seed", "first synthetic secret")
    storage.set_secret("seed", "second synthetic secret")
    assert storage.get_existing_native_secret("seed") == "second synthetic secret"


def test_readonly_custody_rejects_chained_backend(synthetic_native, tmp_path):
    _, backend, _ = synthetic_native
    storage = _storage(tmp_path / "index.json")
    storage._backend = SimpleNamespace(backends=[backend])
    with pytest.raises(SecretStorageError, match="Exact native"):
        storage.get_existing_native_secret("seed")
    backend.get_password.assert_not_called()
    backend.set_password.assert_not_called()


@pytest.mark.parametrize("mode", [None, 0, 1, "true", [], object()])
def test_custody_mode_requires_exact_bool(mode):
    with pytest.raises(SecretStorageError, match="internal device identity"):
        KeyringSecretStorage(custody_hwid=mode)


def test_custody_mode_rejects_injected_hwid_provider():
    with pytest.raises(SecretStorageError, match="internal device identity"):
        KeyringSecretStorage(
            custody_hwid=True,
            hwid_provider=HwIdProvider(fingerprint_reader=lambda: SYNTHETIC_HWID),
        )


@pytest.mark.parametrize(
    "unit",
    ["", "other.service", "../other.service", "/run/credentials/other", True, 42, [], object()],
)
def test_custody_mode_rejects_arbitrary_credential_unit(unit):
    with pytest.raises(SecretStorageError, match="Exact installed custody signer unit"):
        KeyringSecretStorage(custody_hwid=True, custody_systemd_unit=unit)


@pytest.mark.parametrize(
    "unit",
    ["cryptohunter-root-proof-signer.service", "cryptohunter-history-attestation-signer.service"],
)
def test_credential_unit_cannot_enable_custody_without_explicit_mode(unit):
    with pytest.raises(SecretStorageError, match="Exact installed custody signer unit"):
        KeyringSecretStorage(custody_systemd_unit=unit)


@pytest.mark.parametrize(
    "unit",
    [
        None,
        "cryptohunter-root-proof-signer.service",
        "cryptohunter-history-attestation-signer.service",
    ],
)
def test_custody_credential_unit_selects_internal_provider(
    synthetic_native, tmp_path, monkeypatch, unit
):
    provider = Mock()
    monkeypatch.setattr(keyring_storage, "CustodyHwIdProvider", provider)
    KeyringSecretStorage(
        custody_hwid=True, custody_systemd_unit=unit, index_path=tmp_path / "index.json"
    )
    provider.assert_called_once_with(systemd_unit=unit)


@pytest.mark.parametrize(
    "platform,role,expected",
    [
        ("linux", ProviderRole.ROOT_PROOF_SIGNING, True),
        ("linux", ProviderRole.HISTORY_ATTESTATION_SIGNING, True),
        ("linux", ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING, False),
        ("linux", ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING, False),
        ("win32", ProviderRole.ROOT_PROOF_SIGNING, False),
        ("darwin", ProviderRole.ROOT_PROOF_SIGNING, False),
    ],
)
def test_only_linux_stage9_authorities_select_versioned_hwid(
    synthetic_native, tmp_path, monkeypatch, platform, role, expected
):
    monkeypatch.setattr(custody.sys, "platform", platform)
    monkeypatch.setattr(
        KeyringSecretStorage, "_ensure_native_backend", lambda self, module: object()
    )
    provider = Mock()
    monkeypatch.setattr(keyring_storage, "CustodyHwIdProvider", provider)
    security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "synthetic-domain")
    for adapter, attribute in (
        (custody.NativeKeyringSigningSecretReader, "_NativeKeyringSigningSecretReader__storage"),
        (
            custody.NativeKeyringSigningSecretAdministrator,
            "_NativeKeyringSigningSecretAdministrator__storage",
        ),
    ):
        instance = adapter(index_path=tmp_path / "index.json", security=security, role=role)
        assert getattr(instance, attribute)._custody_hwid is expected
        if expected:
            unit = None
            if adapter is custody.NativeKeyringSigningSecretReader:
                unit = {
                    ProviderRole.ROOT_PROOF_SIGNING: "cryptohunter-root-proof-signer.service",
                    ProviderRole.HISTORY_ATTESTATION_SIGNING: (
                        "cryptohunter-history-attestation-signer.service"
                    ),
                }[role]
            provider.assert_called_with(systemd_unit=unit)
        else:
            provider.assert_not_called()
