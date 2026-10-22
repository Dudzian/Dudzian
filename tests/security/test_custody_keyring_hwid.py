"""Synthetic custody envelopes; no native credential collection is accessed."""

from __future__ import annotations

import base64
import hashlib
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


def _seed_synthetic_master(storage: KeyringSecretStorage, records) -> None:
    """Model a previously authorized master directly in the synthetic backend."""
    records[(storage._service_name, storage.MASTER_KEY_SLOT)] = json.dumps(
        {
            "version": storage.SECRET_VERSION,
            "key_b64": base64.b64encode(bytes(range(storage.MASTER_KEY_BYTES))).decode("ascii"),
            "hwid_scheme": CUSTODY_HWID_SCHEME,
            "hwid_digest": hashlib.sha256(SYNTHETIC_HWID.encode("utf-8")).hexdigest(),
        }
    )


def test_versioned_custody_envelopes_survive_new_storage_instance(
    synthetic_native, tmp_path, monkeypatch
):
    records, backend, module = synthetic_native
    path = tmp_path / "index.json"
    writer = _storage(path)
    _seed_synthetic_master(writer, records)
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
    _seed_synthetic_master(storage, records)
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
    actions = (
        lambda: storage.get_existing_native_secret("seed"),
        lambda: storage.set_secret("seed", "incompatible overwrite forbidden"),
    )
    if record_key == storage.MASTER_KEY_SLOT:
        actions += (lambda: storage.set_secret("new-seed", "invalid master forbidden"),)
    for action in actions:
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
    _seed_synthetic_master(storage, records)
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
    _seed_synthetic_master(storage, records)
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
    _seed_synthetic_master(storage, records)
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
    path = tmp_path / "index.json"
    storage = _storage(path)
    _seed_synthetic_master(storage, records)
    storage.set_secret("seed", "synthetic seed")
    before = dict(records)
    index_before = path.read_bytes()
    writes = backend.set_password.call_count
    monkeypatch.setattr(
        keyring_storage.CustodyHwIdProvider,
        "read",
        lambda self: CUSTODY_HWID_PREFIX + "b" * 96,
    )
    monkeypatch.setattr(keyring_storage.os, "urandom", Mock(side_effect=AssertionError("random")))
    reader = _storage(path)
    for action in (
        lambda: reader.get_existing_native_secret("seed"),
        lambda: reader.get_secret("seed"),
        lambda: reader.set_secret("seed", "changed device overwrite forbidden"),
        lambda: reader.set_secret("new-seed", "changed device new secret forbidden"),
    ):
        with pytest.raises(SecretStorageError, match="fingerprint mismatch"):
            action()
    assert records == before
    assert path.read_bytes() == index_before
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
        lambda: versioned.set_secret("new-seed", "implicit migration forbidden"),
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


@pytest.mark.parametrize("orphaned_secret", [False, True], ids=["empty-store", "orphaned-seed"])
@pytest.mark.parametrize(
    "index_contents",
    [None, b"", b'{"version":1,"keys":{}}', b'{"keys":{"old-seed":{}}}'],
    ids=["absent-index", "empty-file", "empty-index", "existing-index"],
)
def test_custody_write_new_name_without_master_refuses_before_any_mutation(
    synthetic_native, tmp_path, monkeypatch, orphaned_secret, index_contents
):
    records, backend, _ = synthetic_native
    path = tmp_path / "index.json"
    storage = _storage(path)
    if orphaned_secret:
        _seed_synthetic_master(storage, records)
        storage.set_secret("old-seed", "existing synthetic secret")
        del records[(storage._service_name, storage.MASTER_KEY_SLOT)]
    if index_contents is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(index_contents)
    before = dict(records)
    files_before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*")}
    backend.set_password.reset_mock()
    backend.delete_password.reset_mock()
    random = Mock(side_effect=AssertionError("no key or nonce generation permitted"))
    encrypt = Mock(side_effect=AssertionError("no ciphertext generation permitted"))
    store_master = Mock(side_effect=AssertionError("no master mutation permitted"))
    register = Mock(side_effect=AssertionError("no index mutation permitted"))
    monkeypatch.setattr(keyring_storage.os, "urandom", random)
    monkeypatch.setattr(storage, "_encrypt_value", encrypt)
    monkeypatch.setattr(storage, "_store_master_key", store_master)
    monkeypatch.setattr(storage, "_register_key", register)

    with pytest.raises(SecretStorageError, match="Existing custody master key required"):
        storage.set_secret("new-seed", "new synthetic secret forbidden")

    assert records == before
    assert (storage._service_name, storage.MASTER_KEY_SLOT) not in records
    assert (storage._service_name, "new-seed") not in records
    assert {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*")} == (
        files_before
    )
    backend.set_password.assert_not_called()
    backend.delete_password.assert_not_called()
    for forbidden in (random, encrypt, store_master, register):
        forbidden.assert_not_called()


@pytest.mark.parametrize("options", [{}, {"allow_create": True}, {"allow_create": False}])
def test_custody_master_loader_cannot_enable_automatic_initialization(
    synthetic_native, tmp_path, monkeypatch, options
):
    records, backend, _ = synthetic_native
    path = tmp_path / "index.json"
    storage = _storage(path)
    random = Mock(side_effect=AssertionError("automatic custody master generation forbidden"))
    monkeypatch.setattr(keyring_storage.os, "urandom", random)

    with pytest.raises(SecretStorageError, match="Existing custody master key required"):
        storage._load_master_key(**options)

    assert records == {}
    random.assert_not_called()
    backend.set_password.assert_not_called()
    assert not path.exists()


def test_existing_valid_custody_master_permits_new_name_without_master_generation(
    synthetic_native, tmp_path, monkeypatch
):
    records, backend, _ = synthetic_native
    path = tmp_path / "index.json"
    storage = _storage(path)
    _seed_synthetic_master(storage, records)
    master_slot = (storage._service_name, storage.MASTER_KEY_SLOT)
    original_master = records[master_slot]
    real_random = keyring_storage.os.urandom

    def only_cipher_nonce(size):
        assert size == 12, "only a ciphertext nonce may be generated"
        return real_random(size)

    random = Mock(side_effect=only_cipher_nonce)
    monkeypatch.setattr(keyring_storage.os, "urandom", random)
    storage.set_secret("new-seed", "new synthetic secret")

    assert records[master_slot] == original_master
    assert storage.get_secret("new-seed") == "new synthetic secret"
    assert storage.get_existing_native_secret("new-seed") == "new synthetic secret"
    random.assert_called_once_with(12)
    assert [(call.args[0], call.args[1]) for call in backend.set_password.call_args_list] == [
        (storage._service_name, "new-seed")
    ]
    assert json.loads(path.read_text())["keys"].keys() == {"new-seed"}


def test_legacy_empty_store_retains_automatic_master_creation(synthetic_native, tmp_path):
    records, _, _ = synthetic_native
    storage = KeyringSecretStorage(
        service_name="synthetic-legacy",
        index_path=tmp_path / "legacy-index.json",
        hwid_provider=HwIdProvider(fingerprint_reader=lambda: "legacy-device"),
    )

    storage.set_secret("seed", "legacy synthetic secret")

    master = json.loads(records[(storage._service_name, storage.MASTER_KEY_SLOT)])
    assert len(base64.b64decode(master["key_b64"])) == storage.MASTER_KEY_BYTES
    assert "hwid_scheme" not in master
    assert storage.get_secret("seed") == "legacy synthetic secret"


def test_custody_generic_master_rotation_requires_controlled_ceremony(synthetic_native, tmp_path):
    records, backend, _ = synthetic_native
    path = tmp_path / "index.json"
    storage = _storage(path)
    _seed_synthetic_master(storage, records)
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
    records, _, _ = synthetic_native
    storage = _storage(tmp_path / "index.json")
    _seed_synthetic_master(storage, records)
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
