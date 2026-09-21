from __future__ import annotations

import threading
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import pytest

from bot_core.local_signing_custody import (
    LocalCHAFreshnessProposerSigningProvider,
    LocalFreshnessAuthorityFinalizationSigningProvider,
    LocalSigningLifecycleError,
    LocalSigningProvisioningConflict,
    NativeKeyringSigningSecretAdministrator,
    NativeKeyringSigningSecretReader,
    SigningKeyLifecycle,
    provision_local_signing_authority,
    rotate_local_freshness_signing_authority,
    transition_local_signing_lifecycle,
)
from bot_core.root_proof_issuer_substrate import (
    CredentialSemanticRole,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)
from bot_core.security.keyring_storage import KeyringSecretStorage


@pytest.fixture
def security() -> SecurityProfileIdentity:
    return SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "freshness-custody")


@pytest.fixture(autouse=True)
def native_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring

    values: dict[tuple[str, str], str] = {}
    lock = threading.Lock()
    monkeypatch.setattr(
        keyring, "get_password", lambda service, key: values.get((service, key))
    )
    monkeypatch.setattr(
        keyring,
        "set_password",
        lambda service, key, value: values.__setitem__((service, key), value),
    )
    monkeypatch.setattr(
        keyring, "delete_password", lambda service, key: values.pop((service, key), None)
    )
    monkeypatch.setattr(
        KeyringSecretStorage, "_ensure_native_backend", lambda self, module: object()
    )
    return values


def admin(
    path: Path, security: SecurityProfileIdentity, role: ProviderRole
) -> NativeKeyringSigningSecretAdministrator:
    return NativeKeyringSigningSecretAdministrator(
        index_path=path / "keyring.json", security=security, role=role
    )


def provision(
    path: Path,
    security: SecurityProfileIdentity,
    role: ProviderRole,
    seed: bytes | None = None,
):
    return provision_local_signing_authority(
        path,
        admin(path, security, role),
        security=security,
        role=role,
        private_seed=seed,
    )


def test_distinct_provisioning_restart_snapshot_and_active_signing(
    tmp_path: Path, security: SecurityProfileIdentity
) -> None:
    authority = provision(
        tmp_path, security, ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING
    )
    proposer = provision(
        tmp_path, security, ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING
    )
    authority_provider = LocalFreshnessAuthorityFinalizationSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    restarted = LocalFreshnessAuthorityFinalizationSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    proposer_provider = LocalCHAFreshnessProposerSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )

    assert authority_provider.credential_snapshot() == restarted.credential_snapshot()
    assert authority_provider.credential_snapshot().key_version == 1
    assert authority_provider.credential_snapshot().lifecycle_generation == 1
    assert authority.key_version != authority.lifecycle_generation + 1
    assert authority.credential_semantic_role is (
        CredentialSemanticRole.ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_FINALIZATION_SIGNING_V1
    )
    assert proposer.credential_semantic_role is (
        CredentialSemanticRole.ACCOUNT_GENESIS_FRESHNESS_PROPOSER_SIGNING_V1
    )
    assert authority_provider.identity != proposer_provider.identity
    assert authority_provider.credential_snapshot().key_material_identity != (
        proposer_provider.credential_snapshot().key_material_identity
    )
    assert admin(tmp_path, security, authority.provider_role).service_namespace != admin(
        tmp_path, security, proposer.provider_role
    ).service_namespace

    signed = authority_provider.sign_finalization(b"receipt")
    proposal = proposer_provider.sign_freshness_proposal(b"proposal")
    Ed25519PublicKey.from_public_bytes(signed.snapshot.public_key).verify(
        signed.signature, b"receipt"
    )
    Ed25519PublicKey.from_public_bytes(proposal.snapshot.public_key).verify(
        proposal.signature, b"proposal"
    )


@pytest.mark.parametrize(
    "role",
    [
        ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING,
        ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING,
    ],
)
@pytest.mark.parametrize(
    "target", [SigningKeyLifecycle.VERIFY_ONLY, SigningKeyLifecycle.REVOKED]
)
def test_non_active_denies_signing_and_revoked_is_terminal(
    tmp_path: Path,
    security: SecurityProfileIdentity,
    role: ProviderRole,
    target: SigningKeyLifecycle,
) -> None:
    provision(tmp_path, security, role)
    provider = (
        LocalFreshnessAuthorityFinalizationSigningProvider(
            tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
        )
        if role is ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING
        else LocalCHAFreshnessProposerSigningProvider(
            tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
        )
    )
    transition_local_signing_lifecycle(
        tmp_path, security=security, role=role, target=target
    )
    operation = (
        provider.sign_finalization
        if role is ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING
        else provider.sign_freshness_proposal
    )
    with pytest.raises(LocalSigningLifecycleError):
        operation(b"new")
    if target is SigningKeyLifecycle.VERIFY_ONLY:
        transition_local_signing_lifecycle(
            tmp_path,
            security=security,
            role=role,
            target=SigningKeyLifecycle.REVOKED,
        )
    for revival in (SigningKeyLifecycle.ACTIVE, SigningKeyLifecycle.VERIFY_ONLY):
        with pytest.raises(LocalSigningLifecycleError):
            transition_local_signing_lifecycle(
                tmp_path, security=security, role=role, target=revival
            )


def test_planned_rotation_separates_version_generation_and_retains_verification(
    tmp_path: Path, security: SecurityProfileIdentity
) -> None:
    role = ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING
    provision(tmp_path, security, role)
    provider = LocalCHAFreshnessProposerSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    old = provider.sign_freshness_proposal(b"candidate")
    successor = rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
    )
    assert successor.key_version == 2
    assert successor.lifecycle_generation == 1
    assert provider.verify_historical(b"candidate", old.signature, old.snapshot)
    current = provider.sign_freshness_proposal(b"new")
    assert current.snapshot.key_version == 2
    assert current.snapshot.lifecycle_generation == 1


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING,
            ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING,
        ),
        (
            ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING,
            ProviderRole.ROOT_PROOF_SIGNING,
        ),
        (
            ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING,
            ProviderRole.HISTORY_ATTESTATION_SIGNING,
        ),
        (ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING, ProviderRole.ROOT_PROOF_SIGNING),
        (
            ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING,
            ProviderRole.HISTORY_ATTESTATION_SIGNING,
        ),
    ],
)
def test_raw_key_material_aliases_are_rejected_across_roles(
    tmp_path: Path,
    security: SecurityProfileIdentity,
    left: ProviderRole,
    right: ProviderRole,
) -> None:
    seed = b"a" * 32
    provision(tmp_path, security, left, seed)
    with pytest.raises(LocalSigningProvisioningConflict, match="aliases"):
        provision(tmp_path, security, right, seed)


@pytest.mark.parametrize(
    "target", [SigningKeyLifecycle.VERIFY_ONLY, SigningKeyLifecycle.REVOKED]
)
def test_sign_transition_race_has_one_lock_linearization(
    tmp_path: Path,
    security: SecurityProfileIdentity,
    monkeypatch: pytest.MonkeyPatch,
    target: SigningKeyLifecycle,
) -> None:
    role = ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING
    provision(tmp_path, security, role)
    provider = LocalFreshnessAuthorityFinalizationSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    entered = threading.Event()
    release = threading.Event()
    original = NativeKeyringSigningSecretReader.read_authority_secret

    def blocked(self, reference):
        if threading.current_thread().name == "signer":
            entered.set()
            assert release.wait(5)
        return original(self, reference)

    monkeypatch.setattr(NativeKeyringSigningSecretReader, "read_authority_secret", blocked)
    result: list[object] = []
    signer = threading.Thread(
        target=lambda: result.append(provider.sign_finalization(b"linearized")),
        name="signer",
    )
    signer.start()
    assert entered.wait(5)
    transition_done = threading.Event()
    transitioner = threading.Thread(
        target=lambda: (
            transition_local_signing_lifecycle(
                tmp_path, security=security, role=role, target=target
            ),
            transition_done.set(),
        )
    )
    transitioner.start()
    assert not transition_done.wait(0.05)
    release.set()
    signer.join(5)
    transitioner.join(5)
    signed = result[0]
    assert signed.snapshot.lifecycle_state is SigningKeyLifecycle.ACTIVE
    assert signed.snapshot.lifecycle_generation == 1
    assert provider.lifecycle_state() is target


def test_test_profile_cannot_be_provisioned_as_production(
    tmp_path: Path, security: SecurityProfileIdentity
) -> None:
    test_security = SecurityProfileIdentity(SecurityProfile.TEST, security.trust_domain)
    with pytest.raises(LocalSigningProvisioningConflict):
        provision(
            tmp_path,
            test_security,
            ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING,
        )
