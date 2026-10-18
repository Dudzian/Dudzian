"""Synthetic custody diagnostics; these tests establish no physical deployment proof."""

from __future__ import annotations

import base64
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bot_core import production_signer_custody as custody
from bot_core.local_signing_custody import (
    LocalSigningMetadata,
    NativeKeyringSigningSecretReader,
    SigningKeyLifecycle,
    _canonical_signing_identity,
    _encode_record,
    _SEMANTIC_ROLE,
)
from bot_core.root_proof_issuer_substrate import (
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)
from bot_core.security import keyring_storage
from bot_core.security.base import SecretStorageError
from bot_core.security.keyring_storage import KeyringSecretStorage


@pytest.fixture
def diagnostic(monkeypatch):
    if sys.platform != "linux":
        pytest.skip("Installed native signer custody observations require Linux principals")
    security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "production.example")
    role = ProviderRole.ROOT_PROOF_SIGNING
    canonical = _canonical_signing_identity(security, role, "a" * 32)
    seed = bytes(range(32))
    public = (
        Ed25519PrivateKey.from_private_bytes(seed)
        .public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    metadata = LocalSigningMetadata(
        security.profile,
        security.trust_domain,
        role,
        _SEMANTIC_ROLE[role],
        canonical.credential_id,
        canonical.provider_namespace,
        canonical.key_handle,
        canonical.key_version,
        canonical.lifecycle_namespace,
        1,
        SigningKeyLifecycle.ACTIVE,
        public,
        canonical.protected_private_material_reference,
    )
    config = {
        "schema_version": 1,
        "profile": "PRODUCTION_LOCAL",
        "trust_domain": security.trust_domain,
        "service_role": "root_signer",
        "custody_directory": "/var/lib/cryptohunter/root-proof-signer",
        "keyring_index_path": "/var/lib/cryptohunter/root-proof-signer/index.json",
    }
    installed = SimpleNamespace(
        principal=lambda role: ("ch_root_signer", 1001, 2001, "pg_root_signer"),
        pin=SimpleNamespace(provisioning=SimpleNamespace(trust_domain=security.trust_domain)),
        source_sha256="b" * 64,
    )
    mutable = {"metadata": _encode_record(metadata), "seed": base64.b64encode(seed).decode()}
    monkeypatch.setattr(custody, "_installed_boundary", lambda: installed)
    monkeypatch.setattr(custody, "_protected_bytes", lambda path: json.dumps(config).encode())
    monkeypatch.setattr(
        custody,
        "_private_bytes",
        lambda path, **kwargs: json.dumps(config).encode()
        if path.name == "custody.json"
        else mutable["metadata"],
    )
    monkeypatch.setattr(os, "getuid", lambda: 1001)
    monkeypatch.setattr(os, "geteuid", lambda: 1001)
    monkeypatch.setattr(os, "getgid", lambda: 2001)
    monkeypatch.setattr(os, "getegid", lambda: 2001)
    monkeypatch.setattr(os, "getgroups", lambda: [2001])
    monkeypatch.setattr(NativeKeyringSigningSecretReader, "__init__", lambda self, **kw: None)
    read = Mock(side_effect=lambda self, reference: mutable["seed"])
    monkeypatch.setattr(
        NativeKeyringSigningSecretReader,
        "read_existing_authority_secret",
        lambda self, reference: read(self, reference),
    )
    monkeypatch.setattr(
        NativeKeyringSigningSecretReader,
        "read_authority_secret",
        Mock(side_effect=AssertionError("legacy write-capable read forbidden")),
    )
    return config, mutable, read


def test_local_seed_match_remains_explicitly_blocked_and_immutable(diagnostic):
    observation = custody.qualify_installed_signer_custody("root_signer")
    facts = observation.public_facts()
    assert observation.local_private_seed_matches_public is True
    assert observation.private_control_status == "BLOCKED"
    assert "INDEPENDENT_PRIVATE_CONTROL_ATTESTATION_UNAVAILABLE" in observation.blockers
    assert "CROSS_ROLE_NATIVE_KEYRING_ACCESS_DENIAL_UNPROVEN" in observation.blockers
    assert custody.validate_signer_custody_facts("root_signer", facts) == facts
    assert facts["semantic_role"] == "ROOT_PROOF_SIGNING"
    assert facts["credential_semantic_role"] == "ROOT_PROOF_ISSUER_SIGNING"
    assert "protected_private_material_reference" not in facts
    assert diagnostic[1]["seed"] not in json.dumps(facts)
    with pytest.raises(FrozenInstanceError):
        observation.private_control_status = "PASS"
    facts["private_control_status"] = "PASS"
    assert observation.public_facts()["private_control_status"] == "BLOCKED"


@pytest.mark.parametrize(
    "change",
    [
        "profile",
        "trust_domain",
        "service_role",
        "extra_field",
        "bool_version",
        "relative",
        "parent_traversal",
        "foreign_index",
    ],
)
def test_untrusted_installation_never_loads_native_seed(diagnostic, change):
    config, _, read = diagnostic
    if change in {"profile", "trust_domain", "service_role"}:
        config[change] = "substituted"
    elif change == "extra_field":
        config["caller_reader"] = "injected"
    elif change == "bool_version":
        config["schema_version"] = True
    elif change == "relative":
        config["custody_directory"] = "relative"
    elif change == "parent_traversal":
        config["custody_directory"] = "/var/lib/../unsafe"
    else:
        config["keyring_index_path"] = "/var/lib/cryptohunter/history-attestation-signer/index.json"
    with pytest.raises(custody.ProductionSignerCustodyError):
        custody.qualify_installed_signer_custody("root_signer")
    read.assert_not_called()


@pytest.mark.parametrize("role", ["issuer", "administrator", "cha", "ROOT_PROOF_SIGNING", None])
def test_uninstalled_roles_have_no_custody_port(diagnostic, role):
    with pytest.raises(custody.ProductionSignerCustodyError):
        custody.qualify_installed_signer_custody(role)
    diagnostic[2].assert_not_called()


@pytest.mark.parametrize("change", ["wrong_uid", "supplementary_group"])
def test_process_identity_fails_closed_before_secret_read(diagnostic, monkeypatch, change):
    if change == "wrong_uid":
        monkeypatch.setattr(os, "getuid", lambda: 1002)
    else:
        monkeypatch.setattr(os, "getgroups", lambda: [2001, 2002])
    with pytest.raises(custody.ProductionSignerCustodyError):
        custody.qualify_installed_signer_custody("root_signer")
    diagnostic[2].assert_not_called()


@pytest.mark.parametrize("change", ["malformed", "duplicate", "wrong_role", "bool_generation"])
def test_metadata_cannot_assert_a_substituted_credential(diagnostic, change):
    _, mutable, read = diagnostic
    value = json.loads(mutable["metadata"])
    if change == "malformed":
        mutable["metadata"] = b"null"
    elif change == "duplicate":
        mutable["metadata"] = mutable["metadata"].rstrip()[:-1] + b',"key_version":1}'
    else:
        value["provider_role" if change == "wrong_role" else "lifecycle_generation"] = (
            "HISTORY_ATTESTATION_SIGNING" if change == "wrong_role" else True
        )
        mutable["metadata"] = json.dumps(value).encode()
    with pytest.raises(custody.ProductionSignerCustodyError):
        custody.qualify_installed_signer_custody("root_signer")
    read.assert_not_called()


@pytest.mark.parametrize("encoded", [None, "", "NaN", base64.b64encode(b"z" * 32).decode()])
def test_absent_malformed_and_wrong_native_seed_cannot_match(diagnostic, encoded):
    diagnostic[1]["seed"] = encoded
    with pytest.raises(custody.ProductionSignerCustodyError):
        custody.qualify_installed_signer_custody("root_signer")


def test_metadata_change_after_native_read_is_rejected(diagnostic):
    _, mutable, read = diagnostic
    original = mutable["seed"]

    def changing(self, reference):
        mutable["metadata"] += b" "
        return original

    read.side_effect = changing
    with pytest.raises(custody.ProductionSignerCustodyError, match="CHANGED"):
        custody.qualify_installed_signer_custody("root_signer")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("private_control_status", "PASS"),
        ("local_private_seed_matches_public", 1),
        ("key_version", True),
        ("lifecycle_generation", float("nan")),
        ("lifecycle_generation", float("inf")),
        ("state", "ACTIVE "),
        ("authority_identity", "other.authority"),
        ("identity", "other-credential"),
        ("public_material_identity", "sha256:" + "a" * 64),
        ("semantic_role", "HISTORY_ATTESTATION_SIGNING"),
        ("credential_semantic_role", "HISTORY_ATTESTATION_SIGNING"),
        ("metadata_sha256", "a" * 65),
        ("public_key_b64", "?" * 44),
        ("caller_claim", "PASS"),
        ("trust_domain", None),
    ],
)
def test_untrusted_wire_facts_fail_closed(diagnostic, field, value):
    facts = custody.qualify_installed_signer_custody("root_signer").public_facts()
    facts[field] = value
    with pytest.raises(custody.ProductionSignerCustodyError, match="UNTRUSTED"):
        custody.validate_signer_custody_facts("root_signer", facts)


def test_wire_validation_cannot_change_role(diagnostic):
    facts = custody.qualify_installed_signer_custody("root_signer").public_facts()
    with pytest.raises(custody.ProductionSignerCustodyError):
        custody.validate_signer_custody_facts("history_signer", facts)


@pytest.mark.skipif(
    sys.platform != "linux", reason="Protected custody descriptor traversal is POSIX-only"
)
def test_actual_unprotected_filesystem_is_rejected_without_creating_files(tmp_path):
    path = tmp_path / "root-proof-signing.json"
    path.write_bytes(b"public metadata")
    before = tuple(tmp_path.iterdir())
    with pytest.raises(custody.ProductionSignerCustodyError):
        custody._private_bytes(
            path, uid=1001, gid=2001, directory_owner=1001, directory_mode=0o700, file_mode=0o600
        )
    assert tuple(tmp_path.iterdir()) == before


@pytest.mark.skipif(
    sys.platform != "linux", reason="Protected custody file descriptor checks are POSIX-only"
)
@pytest.mark.parametrize(
    "change",
    ["file_symlink", "directory_symlink", "hardlink", "oversized", "acl", "mode", "race"],
)
def test_descriptor_boundary_rejects_unsafe_files_using_real_io(tmp_path, monkeypatch, change):
    directory = tmp_path / "custody"
    directory.mkdir(mode=0o700)
    path = directory / "root-proof-signing.json"
    path.write_bytes(b"bounded-public-metadata")
    path.chmod(0o600)
    directory_inode = directory.stat().st_ino
    file_inode = path.stat().st_ino
    actual_fstat = os.fstat
    observations = []

    def installed_metadata(descriptor):
        actual = actual_fstat(descriptor)
        values = {name: getattr(actual, name) for name in dir(actual) if name.startswith("st_")}
        # Only ownership/mode of this isolated fixture is simulated. Real open,
        # no-follow, size, link count, descriptors and I/O remain in use.
        values["st_uid"] = 1001 if actual.st_ino in {directory_inode, file_inode} else 0
        values["st_gid"] = 2001 if actual.st_ino in {directory_inode, file_inode} else actual.st_gid
        values["st_mode"] = actual.st_mode & ~0o022
        if stat.S_ISREG(actual.st_mode):
            observations.append(descriptor)
            if change == "mode":
                values["st_mode"] |= 0o040
            if change == "race" and len(observations) > 1:
                values["st_mtime_ns"] += 1
        return SimpleNamespace(**values)

    if change == "file_symlink":
        path.rename(directory / "actual.json")
        path.symlink_to(directory / "actual.json")
    elif change == "directory_symlink":
        directory.rename(tmp_path / "actual-custody")
        directory.symlink_to(tmp_path / "actual-custody", target_is_directory=True)
    elif change == "hardlink":
        os.link(path, directory / "alias.json")
    elif change == "oversized":
        path.write_bytes(b"x" * 65537)
    monkeypatch.setattr(os, "fstat", installed_metadata)
    monkeypatch.setattr(
        os, "listxattr", lambda fd: ["system.posix_acl_access"] if change == "acl" else []
    )
    with pytest.raises(custody.ProductionSignerCustodyError):
        custody._private_bytes(
            path, uid=1001, gid=2001, directory_owner=1001, directory_mode=0o700, file_mode=0o600
        )


@pytest.fixture
def existing_native_storage(monkeypatch):
    class NativeBackend:
        def __init__(self):
            self.records = {}
            self.get_password = Mock(side_effect=lambda service, key: self.records.get(key))
            self.set_password = Mock(side_effect=AssertionError("credential write forbidden"))
            self.delete_password = Mock(side_effect=AssertionError("credential deletion forbidden"))

    backend = NativeBackend()
    storage = object.__new__(KeyringSecretStorage)
    storage._backend = backend
    storage._service_name = "synthetic-native-service"
    storage._keyring = SimpleNamespace(
        get_password=Mock(side_effect=AssertionError("global backend"))
    )
    monkeypatch.setattr(
        keyring_storage.importlib,
        "import_module",
        lambda name: SimpleNamespace(SecretServiceKeyring=NativeBackend),
    )
    monkeypatch.setattr(storage, "_platform_identifier", lambda: "linux")
    monkeypatch.setattr(
        storage, "_read_hwid_digest", lambda: hashlib.sha256(b"synthetic").hexdigest()
    )
    monkeypatch.setattr(
        storage, "_store_master_key", Mock(side_effect=AssertionError("master write"))
    )
    monkeypatch.setattr(
        storage, "_load_master_key", Mock(side_effect=AssertionError("legacy read"))
    )
    return storage, backend


def test_missing_native_master_never_generates_or_writes(existing_native_storage, monkeypatch):
    storage, backend = existing_native_storage
    backend.records["secret"] = "existing ciphertext"
    monkeypatch.setattr(
        os, "urandom", Mock(side_effect=AssertionError("random generation forbidden"))
    )
    with pytest.raises(SecretStorageError, match="Existing native master"):
        storage.get_existing_native_secret("secret")
    backend.set_password.assert_not_called()
    storage._load_master_key.assert_not_called()
    storage._store_master_key.assert_not_called()


def test_absent_native_secret_has_no_provisioning_side_effect(existing_native_storage):
    storage, backend = existing_native_storage
    assert storage.get_existing_native_secret("absent") is None
    assert backend.get_password.call_count == 1
    backend.set_password.assert_not_called()


def test_existing_native_envelope_decrypts_without_mutation(existing_native_storage, monkeypatch):
    storage, backend = existing_native_storage
    digest = hashlib.sha256(b"synthetic").hexdigest()
    master = b"m" * 32
    backend.records[storage.MASTER_KEY_SLOT] = json.dumps(
        {
            "version": 2,
            "key_b64": base64.b64encode(master).decode(),
            "hwid_digest": digest,
        }
    )
    backend.records["secret"] = storage._encrypt_value(
        "existing seed", storage_key="secret", master=master, hwid_digest=digest
    )
    monkeypatch.setattr(
        os, "urandom", Mock(side_effect=AssertionError("random generation forbidden"))
    )
    assert storage.get_existing_native_secret("secret") == "existing seed"
    backend.set_password.assert_not_called()
    storage._load_master_key.assert_not_called()
    storage._store_master_key.assert_not_called()


def test_chained_or_replaced_backend_is_never_a_readonly_native_source(existing_native_storage):
    storage, backend = existing_native_storage
    storage._backend = SimpleNamespace(backends=[backend])
    with pytest.raises(SecretStorageError, match="Exact native"):
        storage.get_existing_native_secret("secret")
    backend.get_password.assert_not_called()
