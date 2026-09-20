from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
from pathlib import Path
import threading
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
import pytest
import bot_core.local_signing_custody as custody_module

from bot_core.local_signing_custody import (
    LocalHistoryAttestationSigningProvider,
    LocalRootProofSigningProvider,
    LocalSigningCustodyError,
    LocalSigningLifecycleError,
    LocalSigningProvisioningConflict,
    NativeKeyringSigningSecretAdministrator,
    NativeKeyringSigningSecretReader,
    SigningKeyLifecycle,
    provision_local_signing_authority,
    transition_local_signing_lifecycle,
)
from bot_core.security.keyring_storage import KeyringSecretStorage
from bot_core.root_proof_issuer_substrate import (
    CheckpointCapabilities,
    CredentialRoleIdentity,
    CredentialSemanticRole,
    ProviderCapabilities,
    ProviderIdentity,
    ProviderRole,
    RootProofIssuerCompositionGate,
    SecurityProfile,
    SecurityProfileIdentity,
)


class PlaintextMemoryStore:
    """Deliberately unqualified backend used for laundering rejection tests."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.lock = threading.Lock()

    def get_secret(self, key: str) -> str | None:
        with self.lock:
            return self.values.get(key)

    def set_secret(self, key: str, value: str) -> None:
        with self.lock:
            self.values[key] = value

    def delete_secret(self, key: str) -> None:
        with self.lock:
            self.values.pop(key, None)


@pytest.fixture
def security() -> SecurityProfileIdentity:
    return SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "custody-test")


@pytest.fixture
def native_keyring(monkeypatch: pytest.MonkeyPatch):
    """Persistent mocked OS layer beneath the real production adapter class."""

    import keyring

    values: dict[tuple[str, str], str] = {}
    lock = threading.Lock()

    def get_password(service: str, key: str) -> str | None:
        with lock:
            return values.get((service, key))

    def set_password(service: str, key: str, value: str) -> None:
        with lock:
            values[(service, key)] = value

    def delete_password(service: str, key: str) -> None:
        with lock:
            values.pop((service, key), None)

    monkeypatch.setattr(keyring, "get_password", get_password)
    monkeypatch.setattr(keyring, "set_password", set_password)
    monkeypatch.setattr(keyring, "delete_password", delete_password)
    monkeypatch.setattr(KeyringSecretStorage, "_ensure_native_backend", lambda self, module: object())
    return values


@pytest.fixture
def store(
    tmp_path: Path, native_keyring, security: SecurityProfileIdentity
) -> NativeKeyringSigningSecretAdministrator:
    return _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING)


def _admin(
    path: Path, security: SecurityProfileIdentity, role: ProviderRole
) -> NativeKeyringSigningSecretAdministrator:
    return NativeKeyringSigningSecretAdministrator(
        index_path=path / "keyring-index.json", security=security, role=role
    )


def _provision_pair(path: Path, store: NativeKeyringSigningSecretAdministrator, security: SecurityProfileIdentity):
    root = provision_local_signing_authority(
        path, store, security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    history = provision_local_signing_authority(
        path,
        _admin(path, security, ProviderRole.HISTORY_ATTESTATION_SIGNING),
        security=security,
        role=ProviderRole.HISTORY_ATTESTATION_SIGNING,
    )
    return root, history


def test_offline_provisioning_is_distinct_and_restart_stable(
    tmp_path: Path, store: NativeKeyringSigningSecretAdministrator, security: SecurityProfileIdentity
) -> None:
    root, history = _provision_pair(tmp_path, store, security)
    root_provider = LocalRootProofSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)
    restarted = LocalRootProofSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)
    history_provider = LocalHistoryAttestationSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)

    assert root.public_key != history.public_key
    assert root.key_material_identity != history.key_material_identity
    assert root.credential_id != history.credential_id
    assert root.key_handle != history.key_handle
    assert root.lifecycle_namespace != history.lifecycle_namespace
    assert root.provider_namespace != history.provider_namespace
    assert root_provider.credential_identities() == restarted.credential_identities()
    assert root_provider.public_key(root.credential_id) == restarted.public_key(root.credential_id)
    assert root_provider.lifecycle_generation() == restarted.lifecycle_generation() == 1
    assert history_provider.public_key(history.credential_id) == history.public_key
    assert len(root.public_key) == len(history.public_key) == 32


def test_sign_verify_and_exclusive_role_apis(
    tmp_path: Path, store: NativeKeyringSigningSecretAdministrator, security: SecurityProfileIdentity
) -> None:
    root, history = _provision_pair(tmp_path, store, security)
    root_provider = LocalRootProofSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)
    history_provider = LocalHistoryAttestationSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)
    root_signature = root_provider.sign_root_proof(b"canonical-root")
    history_signature = history_provider.sign_history_head(b"canonical-history")
    Ed25519PublicKey.from_public_bytes(root.public_key).verify(root_signature, b"canonical-root")
    Ed25519PublicKey.from_public_bytes(history.public_key).verify(
        history_signature, b"canonical-history"
    )
    assert len(root_signature) == len(history_signature) == 64
    assert callable(root_provider.sign_root_proof)
    assert not callable(getattr(root_provider, "sign_history_head", None))
    assert callable(history_provider.sign_history_head)
    assert not callable(getattr(history_provider, "sign_root_proof", None))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (SigningKeyLifecycle.ACTIVE, SigningKeyLifecycle.VERIFY_ONLY),
        (SigningKeyLifecycle.ACTIVE, SigningKeyLifecycle.REVOKED),
    ],
)
def test_active_lifecycle_transitions_block_signing(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
    first: SigningKeyLifecycle,
    second: SigningKeyLifecycle,
) -> None:
    provision_local_signing_authority(
        tmp_path, _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING), security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    provider = LocalRootProofSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)
    assert provider.lifecycle_state() is first
    changed = transition_local_signing_lifecycle(
        tmp_path, security=security, role=ProviderRole.ROOT_PROOF_SIGNING, target=second
    )
    assert changed.lifecycle_generation == 2
    assert provider.lifecycle_state() is second
    with pytest.raises(LocalSigningLifecycleError, match="not ACTIVE"):
        provider.sign_root_proof(b"new output")
    with pytest.raises(LocalSigningLifecycleError, match="not ACTIVE"):
        provider.active_credential_identity()


def test_verify_only_may_be_revoked_and_revoked_is_terminal(
    tmp_path: Path, store: NativeKeyringSigningSecretAdministrator, security: SecurityProfileIdentity
) -> None:
    provision_local_signing_authority(
        tmp_path, _admin(tmp_path, security, ProviderRole.HISTORY_ATTESTATION_SIGNING), security=security, role=ProviderRole.HISTORY_ATTESTATION_SIGNING
    )
    transition_local_signing_lifecycle(
        tmp_path,
        security=security,
        role=ProviderRole.HISTORY_ATTESTATION_SIGNING,
        target=SigningKeyLifecycle.VERIFY_ONLY,
    )
    final = transition_local_signing_lifecycle(
        tmp_path,
        security=security,
        role=ProviderRole.HISTORY_ATTESTATION_SIGNING,
        target=SigningKeyLifecycle.REVOKED,
    )
    assert final.lifecycle_generation == 3
    for target in (SigningKeyLifecycle.ACTIVE, SigningKeyLifecycle.VERIFY_ONLY):
        with pytest.raises(LocalSigningLifecycleError):
            transition_local_signing_lifecycle(
                tmp_path,
                security=security,
                role=ProviderRole.HISTORY_ATTESTATION_SIGNING,
                target=target,
            )


def test_missing_bad_secret_and_no_silent_rekey_fail_closed(
    tmp_path: Path, store: NativeKeyringSigningSecretAdministrator, security: SecurityProfileIdentity
) -> None:
    with pytest.raises(LocalSigningCustodyError):
        LocalRootProofSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)
    metadata = provision_local_signing_authority(
        tmp_path, _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING), security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    identity = metadata.credential_identity()
    store.create_authority_secret(
        metadata.protected_private_material_reference, base64.b64encode(b"x" * 32).decode()
    )
    with pytest.raises(LocalSigningCustodyError, match="unavailable/corrupt") as error:
        LocalRootProofSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)
    assert base64.b64encode(b"x" * 32).decode() not in str(error.value)
    with pytest.raises(LocalSigningCustodyError):
        provision_local_signing_authority(
            tmp_path, _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING), security=security, role=ProviderRole.ROOT_PROOF_SIGNING
        )
    assert metadata.credential_identity() == identity


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("storage_schema_version", 99),
        ("security_profile", "TEST"),
        ("trust_domain", "wrong"),
        ("provider_role", "HISTORY_ATTESTATION_SIGNING"),
        ("credential_semantic_role", "HISTORY_ATTESTATION_SIGNING"),
        ("public_key_b64", base64.b64encode(b"short").decode()),
        ("lifecycle_state", "UNKNOWN"),
        ("key_version", 0),
        ("lifecycle_generation", -1),
    ],
)
def test_corrupt_or_cross_role_metadata_fails_closed(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
    field: str,
    value: object,
) -> None:
    provision_local_signing_authority(
        tmp_path, _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING), security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    path = tmp_path / "root-proof-signing.json"
    document = json.loads(path.read_text())
    document[field] = value
    path.write_text(json.dumps(document))
    path.chmod(0o600)
    with pytest.raises(LocalSigningCustodyError, match="unavailable/corrupt"):
        LocalRootProofSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)


@pytest.mark.parametrize("raw", [b"{", b"[]", b""])
def test_truncated_or_malformed_record_fails_closed(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
    raw: bytes,
) -> None:
    provision_local_signing_authority(
        tmp_path, _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING), security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    path = tmp_path / "root-proof-signing.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    with pytest.raises(LocalSigningCustodyError):
        LocalRootProofSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)


def test_public_private_mismatch_and_wrong_security_fail_closed(
    tmp_path: Path, store: NativeKeyringSigningSecretAdministrator, security: SecurityProfileIdentity
) -> None:
    metadata = provision_local_signing_authority(
        tmp_path, _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING), security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    store.create_authority_secret(
        metadata.protected_private_material_reference, base64.b64encode(b"z" * 32).decode()
    )
    with pytest.raises(LocalSigningCustodyError):
        LocalRootProofSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)
    with pytest.raises(LocalSigningCustodyError):
        LocalRootProofSigningProvider(
            tmp_path,
            keyring_index_path=tmp_path / "keyring-index.json",
            security=SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "wrong"),
        )
    with pytest.raises(LocalSigningCustodyError):
        LocalRootProofSigningProvider(
            tmp_path,
            keyring_index_path=tmp_path / "keyring-index.json",
            security=SecurityProfileIdentity(SecurityProfile.TEST, security.trust_domain),
        )


def test_concurrent_provisioning_has_one_identity(
    tmp_path: Path, store: NativeKeyringSigningSecretAdministrator, security: SecurityProfileIdentity
) -> None:
    def provision():
        return provision_local_signing_authority(
            tmp_path, _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING), security=security, role=ProviderRole.ROOT_PROOF_SIGNING
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: provision(), range(16)))
    assert len({item.credential_id for item in results}) == 1
    assert len({item.public_key for item in results}) == 1
    assert len({item.protected_private_material_reference for item in results}) == 1


def test_same_physical_material_is_rejected_even_under_other_role_names(
    tmp_path: Path, store: NativeKeyringSigningSecretAdministrator, security: SecurityProfileIdentity
) -> None:
    seed = b"\x42" * 32
    provision_local_signing_authority(
        tmp_path,
        store,
        security=security,
        role=ProviderRole.ROOT_PROOF_SIGNING,
        private_seed=seed,
    )
    with pytest.raises(LocalSigningProvisioningConflict, match="aliases"):
        provision_local_signing_authority(
            tmp_path,
            _admin(tmp_path, security, ProviderRole.HISTORY_ATTESTATION_SIGNING),
            security=security,
            role=ProviderRole.HISTORY_ATTESTATION_SIGNING,
            private_seed=seed,
        )


def test_private_material_absent_from_files_and_public_surfaces(
    tmp_path: Path, store: NativeKeyringSigningSecretAdministrator, security: SecurityProfileIdentity
) -> None:
    seed = b"\x43" * 32
    metadata = provision_local_signing_authority(
        tmp_path,
        store,
        security=security,
        role=ProviderRole.ROOT_PROOF_SIGNING,
        private_seed=seed,
    )
    provider = LocalRootProofSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)
    encoded = base64.b64encode(seed).decode()
    public_values = (
        repr(provider),
        repr(metadata),
        (tmp_path / "root-proof-signing.json").read_text(),
        repr(provider.credential_identities()),
        repr(provider.capabilities),
        repr(provider.public_key(metadata.credential_id)),
    )
    assert all(encoded not in value and seed.hex() not in value for value in public_values)
    assert not hasattr(provider, "private_key")
    assert not hasattr(provider, "export_private_key")


@dataclass(frozen=True)
class _OtherProvider:
    identity: ProviderIdentity
    capabilities: ProviderCapabilities
    credentials: tuple[CredentialRoleIdentity, ...] = ()

    def credential_identities(self): return self.credentials
    def active_bundle(self): return b"x"
    def verify_signed_successor(self, candidate): return True
    def authoritative_state(self, subject): return None
    def compare_and_swap_bind(self, request): return request
    def state_at_revision(self, subject, authoritative_state_revision): return None
    def retained_history(self, subject): return None
    def resolve_claimant(self, claimant_id): return None
    def historical_claimant(self, claimant_id, generation): return None
    def active_requester_credential(self, requester_id): return None
    def historical_requester_credential(self, credential_id): return None
    def current_head(self): return None
    def append_exact_successor(self, expected_head, record): return record
    def record_at(self, sequence): return None
    def current_checkpoint(self): return None
    def advance_exact_successor(self, expected, successor): return True
    def authenticated_checkpoint_at(self, sequence): return None
    def evidence_for(self, subject_id, history_head): return None
    def verify_evidence(self, evidence): return True
    def reserve_or_resolve_attempt_id(self, authorization): return None
    def finalize_attempt(self, identity, *, expected_fence): return None
    def replace_after_authoritative_unbound(self, authorization, evidence, *, expected_fence): return None
    def record_recovery_resolution(self, operation_id, resolution, *, expected_fence): return None
    def attempt(self, operation_id): return None


def _composition_others(security: SecurityProfileIdentity) -> list[_OtherProvider]:
    signing_roles = {ProviderRole.ROOT_PROOF_SIGNING, ProviderRole.HISTORY_ATTESTATION_SIGNING}
    credential_roles = {
        ProviderRole.CLAIMANT_IDENTITY_REGISTRY: CredentialSemanticRole.ROOT_PROOF_CLAIMANT,
        ProviderRole.REQUESTER_CREDENTIAL_REGISTRY: CredentialSemanticRole.ROOT_PROOF_REQUESTER,
    }
    result = []
    for role in ProviderRole:
        if role in signing_roles:
            continue
        namespace = f"{security.trust_domain}.{role.value.lower()}"
        semantic = credential_roles.get(role)
        credentials = () if semantic is None else (
            CredentialRoleIdentity(semantic, f"cred-{role.value}", namespace, f"key-{role.value}", f"life-{role.value}", None),
        )
        capabilities = ProviderCapabilities(
            True,
            authoritative_reads=role is not ProviderRole.CHECKPOINT_AUTHORITY,
            durable_state=role not in {ProviderRole.RECONCILIATION_EVIDENCE},
            compare_and_swap=role in {ProviderRole.ENTITLEMENT_REGISTRY, ProviderRole.CHA_ATTEMPT_STORE},
            checkpoint=CheckpointCapabilities(True, True, True, False, False, True, True, True, True)
            if role is ProviderRole.CHECKPOINT_AUTHORITY else None,
        )
        result.append(_OtherProvider(ProviderIdentity(role, security, namespace), capabilities, credentials))
    return result


def test_real_local_providers_qualify_local_and_fail_server_ready(
    tmp_path: Path, store: NativeKeyringSigningSecretAdministrator, security: SecurityProfileIdentity
) -> None:
    _provision_pair(tmp_path, store, security)
    root = LocalRootProofSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)
    history = LocalHistoryAttestationSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)
    providers = [*_composition_others(security), root, history]
    local = RootProofIssuerCompositionGate().qualify(security, providers)
    assert local.qualified, local.failures

    server = SecurityProfileIdentity(SecurityProfile.PRODUCTION_SERVER_READY, security.trust_domain)
    server_providers = [
        replace(item, identity=replace(item.identity, security=server)) for item in _composition_others(security)
    ]
    # Genuine SERVER_READY qualification against the same local adapters fails
    # both profile binding and hardware/non-exportability requirements.
    result = RootProofIssuerCompositionGate().qualify(server, [*server_providers, root, history])
    assert not result.qualified
    assert any(failure.role in {
        ProviderRole.ROOT_PROOF_SIGNING,
        ProviderRole.HISTORY_ATTESTATION_SIGNING,
    } for failure in result.failures)


@pytest.mark.parametrize("backend", [PlaintextMemoryStore(), object()])
def test_arbitrary_or_plaintext_secret_backend_is_not_production_custody(
    tmp_path: Path, security: SecurityProfileIdentity, backend: object
) -> None:
    with pytest.raises(LocalSigningCustodyError, match="unqualified"):
        provision_local_signing_authority(
            tmp_path,
            backend,  # type: ignore[arg-type]
            security=security,
            role=ProviderRole.ROOT_PROOF_SIGNING,
        )
    with pytest.raises(TypeError):
        LocalRootProofSigningProvider(
            tmp_path, secret_reader=backend, security=security  # type: ignore[call-arg]
        )


def test_authority_scoped_readers_reject_opposite_role_references(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
) -> None:
    root, history = _provision_pair(tmp_path, store, security)
    root_reader = NativeKeyringSigningSecretReader(
        index_path=tmp_path / "keyring-index.json",
        security=security,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    history_reader = NativeKeyringSigningSecretReader(
        index_path=tmp_path / "keyring-index.json",
        security=security,
        role=ProviderRole.HISTORY_ATTESTATION_SIGNING,
    )

    assert root_reader.service_namespace != history_reader.service_namespace
    with pytest.raises(LocalSigningCustodyError, match="outside authority scope"):
        root_reader.read_authority_secret(history.protected_private_material_reference)
    with pytest.raises(LocalSigningCustodyError, match="outside authority scope"):
        history_reader.read_authority_secret(root.protected_private_material_reference)


def test_authority_scoped_reader_rejects_other_trust_domain(
    tmp_path: Path, native_keyring
) -> None:
    domain_a = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "domain-a")
    domain_b = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "domain-b")
    directory_a = tmp_path / "domain-a"
    directory_b = tmp_path / "domain-b"
    index = tmp_path / "keyring-index.json"
    material_a = provision_local_signing_authority(
        directory_a,
        NativeKeyringSigningSecretAdministrator(
            index_path=index,
            security=domain_a,
            role=ProviderRole.ROOT_PROOF_SIGNING,
        ),
        security=domain_a,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    material_b = provision_local_signing_authority(
        directory_b,
        NativeKeyringSigningSecretAdministrator(
            index_path=index,
            security=domain_b,
            role=ProviderRole.ROOT_PROOF_SIGNING,
        ),
        security=domain_b,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    reader_a = NativeKeyringSigningSecretReader(
        index_path=index,
        security=domain_a,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    reader_b = NativeKeyringSigningSecretReader(
        index_path=index,
        security=domain_b,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )

    assert reader_a.service_namespace != reader_b.service_namespace
    assert reader_a.read_authority_secret(material_a.protected_private_material_reference)
    with pytest.raises(LocalSigningCustodyError, match="outside authority scope"):
        reader_a.read_authority_secret(material_b.protected_private_material_reference)


def test_runtime_reader_is_immutable_and_provider_exposes_no_secret_capability(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
) -> None:
    provision_local_signing_authority(
        tmp_path, store, security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    reader = NativeKeyringSigningSecretReader(
        index_path=tmp_path / "keyring-index.json",
        security=security,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    with pytest.raises(AttributeError, match="immutable"):
        reader._NativeKeyringSigningSecretReader__storage = PlaintextMemoryStore()
    provider = LocalRootProofSigningProvider(
        tmp_path,
        keyring_index_path=tmp_path / "keyring-index.json",
        security=security,
    )
    assert not hasattr(provider, "__dict__")
    assert not hasattr(provider, "secret_reader")
    assert not hasattr(provider, "get_secret")
    assert not hasattr(provider, "read_authority_secret")
    assert not hasattr(provider, "set_secret")
    assert not hasattr(provider, "delete_secret")
    with pytest.raises(AttributeError, match="immutable"):
        provider._LocalSigningProviderBase__secret_reader = PlaintextMemoryStore()


def test_malicious_path_subclass_is_rejected_before_semantic_methods(
    tmp_path: Path, security: SecurityProfileIdentity
) -> None:
    calls: list[str] = []

    class MaliciousPath(type(Path())):
        def is_absolute(self):
            calls.append("is_absolute")
            return True

        def expanduser(self):
            calls.append("expanduser")
            return self

        def is_symlink(self):
            calls.append("is_symlink")
            return False

        def stat(self, *args, **kwargs):
            calls.append("stat")
            return super().stat(*args, **kwargs)

        def __fspath__(self):
            calls.append("fspath")
            return super().__fspath__()

    malicious = MaliciousPath(tmp_path)
    with pytest.raises(LocalSigningCustodyError, match="exact platform Path"):
        NativeKeyringSigningSecretReader(
            index_path=malicious,
            security=security,
            role=ProviderRole.ROOT_PROOF_SIGNING,
        )
    with pytest.raises(LocalSigningCustodyError, match="exact platform Path"):
        provision_local_signing_authority(
            malicious,
            object(),  # type: ignore[arg-type]
            security=security,
            role=ProviderRole.ROOT_PROOF_SIGNING,
        )
    with pytest.raises(LocalSigningCustodyError, match="exact platform Path"):
        LocalRootProofSigningProvider(
            malicious,
            keyring_index_path=tmp_path / "keyring-index.json",
            security=security,
        )
    assert calls == []


def test_restart_uses_fresh_native_reader_and_preserves_exact_identity(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
) -> None:
    metadata = provision_local_signing_authority(
        tmp_path, _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING), security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    first = LocalRootProofSigningProvider(
        tmp_path,
        keyring_index_path=tmp_path / "keyring-index.json",
        security=security,
    )
    expected = (
        first.credential_identities(),
        first.public_key(metadata.credential_id),
        first.lifecycle_generation(),
        first.lifecycle_state(),
    )
    del first, store
    restarted = LocalRootProofSigningProvider(
        tmp_path,
        keyring_index_path=tmp_path / "keyring-index.json",
        security=security,
    )
    assert (
        restarted.credential_identities(),
        restarted.public_key(metadata.credential_id),
        restarted.lifecycle_generation(),
        restarted.lifecycle_state(),
    ) == expected


def test_concurrent_cross_role_same_seed_cannot_create_alias(
    tmp_path: Path,
    native_keyring,
    security: SecurityProfileIdentity,
) -> None:
    seed = b"\x55" * 32

    def provision(role: ProviderRole):
        administrator = NativeKeyringSigningSecretAdministrator(
            index_path=tmp_path / "keyring-index.json", security=security, role=role
        )
        return provision_local_signing_authority(
            tmp_path, administrator, security=security, role=role, private_seed=seed
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(provision, ProviderRole.ROOT_PROOF_SIGNING),
            executor.submit(provision, ProviderRole.HISTORY_ATTESTATION_SIGNING),
        ]
    successes = [future.result() for future in futures if future.exception() is None]
    failures = [future.exception() for future in futures if future.exception() is not None]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], LocalSigningProvisioningConflict)
    records = list(tmp_path.glob("*-signing.json"))
    assert len(records) == 1


def test_concurrent_cross_role_random_material_remains_distinct(
    tmp_path: Path,
    native_keyring,
    security: SecurityProfileIdentity,
) -> None:
    def provision(role: ProviderRole):
        administrator = NativeKeyringSigningSecretAdministrator(
            index_path=tmp_path / "keyring-index.json", security=security, role=role
        )
        return provision_local_signing_authority(
            tmp_path, administrator, security=security, role=role
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(provision, role)
            for role in (
                ProviderRole.ROOT_PROOF_SIGNING,
                ProviderRole.HISTORY_ATTESTATION_SIGNING,
            )
        ]
    records = [future.result() for future in futures]
    assert len({record.public_key for record in records}) == 2
    assert len({record.key_material_identity for record in records}) == 2


def test_concurrent_lifecycle_transitions_are_linearizable_and_monotonic(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
) -> None:
    metadata = provision_local_signing_authority(
        tmp_path, _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING), security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    barrier = threading.Barrier(2)

    def transition(target: SigningKeyLifecycle):
        barrier.wait()
        return transition_local_signing_lifecycle(
            tmp_path,
            security=security,
            role=ProviderRole.ROOT_PROOF_SIGNING,
            target=target,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(transition, SigningKeyLifecycle.VERIFY_ONLY),
            executor.submit(transition, SigningKeyLifecycle.REVOKED),
        ]
    committed = [future.result() for future in futures if future.exception() is None]
    errors = [future.exception() for future in futures if future.exception() is not None]
    provider = LocalRootProofSigningProvider(tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security)
    assert provider.lifecycle_state() is SigningKeyLifecycle.REVOKED
    assert provider.lifecycle_generation() == metadata.lifecycle_generation + len(committed)
    assert all(isinstance(error, LocalSigningLifecycleError) for error in errors)
    with pytest.raises(LocalSigningLifecycleError):
        transition_local_signing_lifecycle(
            tmp_path,
            security=security,
            role=ProviderRole.ROOT_PROOF_SIGNING,
            target=SigningKeyLifecycle.VERIFY_ONLY,
        )


def test_sign_and_revoke_have_an_explicit_linearization_order(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provision_local_signing_authority(
        tmp_path, _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING), security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    provider = LocalRootProofSigningProvider(
        tmp_path,
        keyring_index_path=tmp_path / "keyring-index.json",
        security=security,
    )
    entered = threading.Event()
    release = threading.Event()
    original = NativeKeyringSigningSecretReader.read_authority_secret

    def blocking_get(self, key):
        if threading.current_thread().name == "signer":
            entered.set()
            assert release.wait(5)
        return original(self, key)

    monkeypatch.setattr(
        NativeKeyringSigningSecretReader, "read_authority_secret", blocking_get
    )
    order: list[str] = []

    def sign():
        provider.sign_root_proof(b"linearized")
        order.append("signed")

    def revoke():
        transition_local_signing_lifecycle(
            tmp_path,
            security=security,
            role=ProviderRole.ROOT_PROOF_SIGNING,
            target=SigningKeyLifecycle.REVOKED,
        )
        order.append("revoked")

    signing = threading.Thread(target=sign, name="signer")
    revoking = threading.Thread(target=revoke, name="revoker")
    signing.start()
    assert entered.wait(5)
    revoking.start()
    time.sleep(0.05)
    assert order == []
    release.set()
    signing.join(5)
    revoking.join(5)
    assert order == ["signed", "revoked"]
    with pytest.raises(LocalSigningLifecycleError):
        provider.active_credential_identity()


def test_failed_metadata_commit_cleans_only_new_protected_secret(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references: list[str] = []
    original_set = NativeKeyringSigningSecretAdministrator.create_authority_secret

    def recording_set(self, key, value):
        references.append(key)
        return original_set(self, key, value)

    monkeypatch.setattr(
        NativeKeyringSigningSecretAdministrator,
        "create_authority_secret",
        recording_set,
    )
    monkeypatch.setattr(
        custody_module,
        "_write_new_record",
        lambda path, payload: (_ for _ in ()).throw(OSError("injected commit failure")),
    )
    with pytest.raises(OSError, match="injected"):
        provision_local_signing_authority(
            tmp_path, _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING), security=security, role=ProviderRole.ROOT_PROOF_SIGNING
        )
    assert len(references) == 1
    assert store.read_authority_secret(references[0]) is None
    assert not (tmp_path / "root-proof-signing.json").exists()


def test_exact_security_role_and_lifecycle_boundaries(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
) -> None:
    class SecuritySubclass(SecurityProfileIdentity):
        pass

    malformed = object.__new__(SecuritySubclass)
    object.__setattr__(malformed, "profile", SecurityProfile.PRODUCTION_LOCAL)
    object.__setattr__(malformed, "trust_domain", security.trust_domain)
    with pytest.raises(LocalSigningCustodyError):
        provision_local_signing_authority(
            tmp_path,
            store,
            security=malformed,
            role=ProviderRole.ROOT_PROOF_SIGNING,
        )
    with pytest.raises(LocalSigningProvisioningConflict):
        provision_local_signing_authority(
            tmp_path, store, security=security, role="ROOT_PROOF_SIGNING"  # type: ignore[arg-type]
        )


def _rewrite_root_metadata(tmp_path: Path, **changes: object) -> None:
    path = tmp_path / "root-proof-signing.json"
    document = json.loads(path.read_text())
    document.update(changes)
    path.write_text(json.dumps(document))
    path.chmod(0o600)


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("provider_namespace", "forged.namespace"),
        ("credential_id", "root-proof-signing-11111111111111111111111111111111"),
        ("credential_id", "history-attestation-signing-11111111111111111111111111111111"),
        ("credential_id", "root-proof-signing-ABCDEF11111111111111111111111111"),
        ("credential_id", "root-proof-signing-short"),
        ("key_handle", "root-proof-signing-key-22222222222222222222222222222222"),
        ("key_handle", "forged-key-handle"),
        ("key_version", 2),
        ("lifecycle_namespace", "forged.lifecycle"),
    ],
)
def test_each_forged_persisted_identity_field_is_rejected(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
    field: str,
    forged: object,
) -> None:
    provision_local_signing_authority(
        tmp_path, store, security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    _rewrite_root_metadata(tmp_path, **{field: forged})
    with pytest.raises(LocalSigningCustodyError, match="unavailable/corrupt"):
        LocalRootProofSigningProvider(
            tmp_path,
            keyring_index_path=tmp_path / "keyring-index.json",
            security=security,
        )


@pytest.mark.parametrize(
    ("state", "generation"),
    [
        ("ACTIVE", 2),
        ("ACTIVE", 999),
        ("VERIFY_ONLY", 1),
        ("REVOKED", 1),
        ("REVOKED", 4),
        ("REVOKED", 999),
    ],
)
def test_impossible_lifecycle_state_generation_is_rejected(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
    state: str,
    generation: int,
) -> None:
    provision_local_signing_authority(
        tmp_path, store, security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    _rewrite_root_metadata(
        tmp_path, lifecycle_state=state, lifecycle_generation=generation
    )
    with pytest.raises(LocalSigningCustodyError, match="unavailable/corrupt"):
        LocalRootProofSigningProvider(
            tmp_path,
            keyring_index_path=tmp_path / "keyring-index.json",
            security=security,
        )


def test_same_scope_alternate_reference_cannot_rebind_physical_identity(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
) -> None:
    provision_local_signing_authority(
        tmp_path, store, security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    alternate_credential = "root-proof-signing-33333333333333333333333333333333"
    alternate_reference = store.new_secret_reference(alternate_credential)
    alternate_seed = b"\x77" * 32
    alternate_public = (
        Ed25519PrivateKey.from_private_bytes(alternate_seed)
        .public_key()
        .public_bytes_raw()
    )
    store.create_authority_secret(
        alternate_reference, base64.b64encode(alternate_seed).decode("ascii")
    )
    _rewrite_root_metadata(
        tmp_path,
        protected_private_material_reference=alternate_reference,
        public_key_b64=base64.b64encode(alternate_public).decode("ascii"),
    )

    with pytest.raises(LocalSigningCustodyError, match="unavailable/corrupt"):
        LocalRootProofSigningProvider(
            tmp_path,
            keyring_index_path=tmp_path / "keyring-index.json",
            security=security,
        )


def test_complete_identity_laundering_with_same_physical_key_is_rejected(
    tmp_path: Path,
    store: NativeKeyringSigningSecretAdministrator,
    security: SecurityProfileIdentity,
) -> None:
    metadata = provision_local_signing_authority(
        tmp_path, store, security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    forged_unique = "44444444444444444444444444444444"
    _rewrite_root_metadata(
        tmp_path,
        credential_id=f"root-proof-signing-{forged_unique}",
        provider_namespace="forged.production-local.root-proof-signing",
        key_handle=f"root-proof-signing-key-{forged_unique}",
        key_version=2,
        lifecycle_namespace="forged.production-local.root-proof-signing.lifecycle",
    )
    with pytest.raises(LocalSigningCustodyError, match="unavailable/corrupt"):
        provider = LocalRootProofSigningProvider(
            tmp_path,
            keyring_index_path=tmp_path / "keyring-index.json",
            security=security,
        )
        provider.public_key(metadata.credential_id)
