from __future__ import annotations

import threading
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import pytest
import bot_core.local_signing_custody as custody

from bot_core.local_signing_custody import (
    LocalCHAFreshnessProposerSigningProvider,
    LocalFreshnessAuthorityFinalizationSigningProvider,
    LocalHistoryAttestationSigningProvider,
    LocalRootProofSigningProvider,
    LocalSigningLifecycleError,
    LocalSigningProvisioningConflict,
    NativeKeyringSigningSecretAdministrator,
    NativeKeyringSigningSecretReader,
    SigningKeyLifecycle,
    provision_local_signing_authority,
    rotate_local_freshness_signing_authority,
    transition_local_signing_credential_lifecycle,
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
    monkeypatch.setattr(keyring, "get_password", lambda service, key: values.get((service, key)))
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
    authority = provision(tmp_path, security, ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING)
    proposer = provision(tmp_path, security, ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING)
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
    assert (
        admin(tmp_path, security, authority.provider_role).service_namespace
        != admin(tmp_path, security, proposer.provider_role).service_namespace
    )

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
@pytest.mark.parametrize("target", [SigningKeyLifecycle.VERIFY_ONLY, SigningKeyLifecycle.REVOKED])
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
    transition_local_signing_lifecycle(tmp_path, security=security, role=role, target=target)
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
        rotation_operation_id="planned-rotation",
    )
    assert successor.key_version == 2
    assert successor.lifecycle_generation == 1
    assert provider.verify_historical(b"candidate", old.signature, old.snapshot)
    current = provider.sign_freshness_proposal(b"new")
    assert current.snapshot.key_version == 2
    assert current.snapshot.lifecycle_generation == 1


def test_rotated_credential_can_be_revoked_without_changing_current(
    tmp_path: Path, security: SecurityProfileIdentity
) -> None:
    role = ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING
    provision(tmp_path, security, role)
    provider = LocalCHAFreshnessProposerSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    old = provider.sign_freshness_proposal(b"v1")
    rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
        rotation_operation_id="revoke-after-rotation",
    )
    assert provider.verify_historical(b"v1", old.signature, old.snapshot)
    revoked = transition_local_signing_credential_lifecycle(
        tmp_path,
        security=security,
        role=role,
        credential_identity=old.snapshot.credential_identity,
        expected_lifecycle_generation=2,
        target=SigningKeyLifecycle.REVOKED,
    )
    assert revoked.lifecycle_generation == 3
    assert provider.sign_freshness_proposal(b"v2").snapshot.key_version == 2
    with pytest.raises(LocalSigningLifecycleError, match="historical trust"):
        provider.verify_historical(b"v1", old.signature, old.snapshot)
    for target in (SigningKeyLifecycle.ACTIVE, SigningKeyLifecycle.VERIFY_ONLY):
        with pytest.raises(LocalSigningLifecycleError):
            transition_local_signing_credential_lifecycle(
                tmp_path,
                security=security,
                role=role,
                credential_identity=old.snapshot.credential_identity,
                expected_lifecycle_generation=3,
                target=target,
            )


def test_three_versions_have_independent_lifecycle_generations(
    tmp_path: Path, security: SecurityProfileIdentity
) -> None:
    role = ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING
    provision(tmp_path, security, role)
    provider = LocalFreshnessAuthorityFinalizationSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    v1 = provider.credential_snapshot()
    rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
        rotation_operation_id="multi-X",
    )
    v2 = provider.credential_snapshot()
    rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
        rotation_operation_id="multi-Y",
    )
    v3 = provider.credential_snapshot()
    assert [v1.key_version, v2.key_version, v3.key_version] == [1, 2, 3]
    for snapshot in (v1, v2):
        changed = transition_local_signing_credential_lifecycle(
            tmp_path,
            security=security,
            role=role,
            credential_identity=snapshot.credential_identity,
            expected_lifecycle_generation=2,
            target=SigningKeyLifecycle.REVOKED,
        )
        assert changed.lifecycle_generation == 3
        assert provider.credential_snapshot() == v3


@pytest.mark.parametrize("cut", [f"R{number}" for number in range(8)])
def test_every_rotation_crash_cut_recovers_without_v3(
    tmp_path: Path,
    security: SecurityProfileIdentity,
    cut: str,
) -> None:
    role = ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING
    provision(tmp_path, security, role)
    seed = bytes([int(cut[1]) + 1]) * 32

    def fail_at(stage: str) -> None:
        if stage == cut:
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match=cut):
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id=f"crash-{cut}",
            private_seed=seed,
            fault_injector=fail_at,
        )
    recovered = rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
        rotation_operation_id=f"crash-{cut}",
        private_seed=seed,
    )
    assert recovered.key_version == 2
    provider = LocalCHAFreshnessProposerSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    assert provider.credential_snapshot().key_version == 2


@pytest.mark.parametrize("cut", [f"R{number}" for number in range(8)])
def test_generated_successor_crash_matrix_replays_by_operation_identity(
    tmp_path: Path,
    security: SecurityProfileIdentity,
    cut: str,
) -> None:
    role = ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING
    provision(tmp_path, security, role)
    operation_id = f"generated-{cut}"

    def fail_at(stage: str) -> None:
        if stage == cut:
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match=cut):
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id=operation_id,
            fault_injector=fail_at,
        )
    recovered = rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
        rotation_operation_id=operation_id,
    )
    replayed = rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
        rotation_operation_id=operation_id,
    )
    assert recovered == replayed
    assert recovered.key_version == 2


@pytest.mark.parametrize("cut", ["R3", "R4"])
@pytest.mark.parametrize("target", [SigningKeyLifecycle.VERIFY_ONLY, SigningKeyLifecycle.REVOKED])
def test_prelinearization_predecessor_descendant_is_authoritative(
    tmp_path: Path,
    security: SecurityProfileIdentity,
    cut: str,
    target: SigningKeyLifecycle,
) -> None:
    role = ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING
    provision(tmp_path, security, role)
    provider = LocalCHAFreshnessProposerSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    old = provider.sign_freshness_proposal(b"prelinearization")
    operation_id = f"prelinearization-{cut}-{target.value}"

    def fail_at(stage: str) -> None:
        if stage == cut:
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match=cut):
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id=operation_id,
            fault_injector=fail_at,
        )
    with pytest.raises(custody.LocalSigningCustodyError, match="recovery required"):
        LocalCHAFreshnessProposerSigningProvider(
            tmp_path,
            keyring_index_path=tmp_path / "keyring.json",
            security=security,
        )
    changed = transition_local_signing_credential_lifecycle(
        tmp_path,
        security=security,
        role=role,
        credential_identity=old.snapshot.credential_identity,
        expected_lifecycle_generation=1,
        target=target,
    )
    assert changed.lifecycle_generation == 2
    successor = rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
        rotation_operation_id=operation_id,
    )
    assert successor.key_version == 2
    assert successor.lifecycle_state is SigningKeyLifecycle.ACTIVE
    retained_path = custody._retained_path(tmp_path, role, old.snapshot.credential_identity)
    retained = custody._read_record(retained_path, security, role)
    assert retained.lifecycle_state is target
    assert retained.lifecycle_generation == 2
    completed = custody._read_rotation_intent(
        custody._completed_rotation_path(tmp_path, security, role, operation_id),
        security,
        role,
        completed=True,
    )
    assert completed.final_retained == retained
    restarted = LocalCHAFreshnessProposerSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    if target is SigningKeyLifecycle.VERIFY_ONLY:
        assert restarted.verify_historical(b"prelinearization", old.signature, old.snapshot)
    else:
        with pytest.raises(LocalSigningLifecycleError, match="historical trust"):
            restarted.verify_historical(b"prelinearization", old.signature, old.snapshot)
        for revival in (SigningKeyLifecycle.ACTIVE, SigningKeyLifecycle.VERIFY_ONLY):
            with pytest.raises(LocalSigningLifecycleError):
                transition_local_signing_credential_lifecycle(
                    tmp_path,
                    security=security,
                    role=role,
                    credential_identity=old.snapshot.credential_identity,
                    expected_lifecycle_generation=2,
                    target=revival,
                )
        assert (
            rotate_local_freshness_signing_authority(
                tmp_path,
                admin(tmp_path, security, role),
                security=security,
                role=role,
                rotation_operation_id=operation_id,
            )
            == successor
        )
        assert custody._read_record(retained_path, security, role) == retained


def test_presecret_revocation_aborts_tentative_rotation_without_orphan(
    tmp_path: Path, security: SecurityProfileIdentity
) -> None:
    role = ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING
    provision(tmp_path, security, role)
    provider = LocalCHAFreshnessProposerSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    old = provider.credential_snapshot()

    def fail_at_r2(stage: str) -> None:
        if stage == "R2":
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match="R2"):
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id="presecret-revoked",
            fault_injector=fail_at_r2,
        )
    intent, _, _ = custody._rotation_paths(tmp_path, role)
    tentative = custody._read_rotation_intent(intent, security, role)
    administrator = admin(tmp_path, security, role)
    assert (
        administrator.read_authority_secret(
            tentative.successor.protected_private_material_reference
        )
        is None
    )
    transition_local_signing_credential_lifecycle(
        tmp_path,
        security=security,
        role=role,
        credential_identity=old.credential_identity,
        expected_lifecycle_generation=1,
        target=SigningKeyLifecycle.REVOKED,
    )
    with pytest.raises(LocalSigningLifecycleError, match="only ACTIVE"):
        rotate_local_freshness_signing_authority(
            tmp_path,
            administrator,
            security=security,
            role=role,
            rotation_operation_id="presecret-revoked",
        )
    assert not intent.exists()
    assert (
        administrator.read_authority_secret(
            tentative.successor.protected_private_material_reference
        )
        is None
    )
    assert provider.lifecycle_state() is SigningKeyLifecycle.REVOKED
    with pytest.raises(LocalSigningLifecycleError, match="only ACTIVE"):
        rotate_local_freshness_signing_authority(
            tmp_path,
            administrator,
            security=security,
            role=role,
            rotation_operation_id="operation-Y",
        )


def test_r6_revoke_survives_stale_intent_recovery_and_restart(
    tmp_path: Path, security: SecurityProfileIdentity
) -> None:
    role = ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING
    provision(tmp_path, security, role)
    provider = LocalCHAFreshnessProposerSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    old = provider.sign_freshness_proposal(b"old")

    def fail_at_r6(stage: str) -> None:
        if stage == "R6":
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match="R6"):
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id="stale-intent",
            fault_injector=fail_at_r6,
        )
    revoked = transition_local_signing_credential_lifecycle(
        tmp_path,
        security=security,
        role=role,
        credential_identity=old.snapshot.credential_identity,
        expected_lifecycle_generation=2,
        target=SigningKeyLifecycle.REVOKED,
    )
    assert revoked.lifecycle_generation == 3
    recovered = rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
        rotation_operation_id="stale-intent",
    )
    restarted = LocalCHAFreshnessProposerSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    assert recovered.key_version == 2
    assert restarted.credential_snapshot().lifecycle_generation == 1
    with pytest.raises(LocalSigningLifecycleError, match="historical trust"):
        restarted.verify_historical(b"old", old.signature, old.snapshot)


def test_r7_generated_lost_response_conflict_and_new_operation(
    tmp_path: Path, security: SecurityProfileIdentity
) -> None:
    role = ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING
    provision(tmp_path, security, role)

    def fail_at_r7(stage: str) -> None:
        if stage == "R7":
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match="R7"):
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id="operation-X",
            fault_injector=fail_at_r7,
        )
    restarted_admin = admin(tmp_path, security, role)
    v2 = rotate_local_freshness_signing_authority(
        tmp_path,
        restarted_admin,
        security=security,
        role=role,
        rotation_operation_id="operation-X",
    )
    restarted = LocalFreshnessAuthorityFinalizationSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    assert restarted.credential_snapshot().credential_identity == v2.credential_identity()
    with pytest.raises(LocalSigningProvisioningConflict, match="different request"):
        rotate_local_freshness_signing_authority(
            tmp_path,
            restarted_admin,
            security=security,
            role=role,
            rotation_operation_id="operation-X",
            private_seed=b"z" * 32,
        )
    v3 = rotate_local_freshness_signing_authority(
        tmp_path,
        restarted_admin,
        security=security,
        role=role,
        rotation_operation_id="operation-Y",
    )
    assert v3.key_version == 3
    assert (
        rotate_local_freshness_signing_authority(
            tmp_path,
            restarted_admin,
            security=security,
            role=role,
            rotation_operation_id="operation-Y",
        )
        == v3
    )


def test_r5_runtime_fails_closed_until_admin_recovery(
    tmp_path: Path, security: SecurityProfileIdentity
) -> None:
    role = ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING
    provision(tmp_path, security, role)
    provider = LocalCHAFreshnessProposerSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    old = provider.sign_freshness_proposal(b"old")

    def fail_at_r5(stage: str) -> None:
        if stage == "R5":
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match="R5"):
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id="runtime-R5",
            fault_injector=fail_at_r5,
        )
    with pytest.raises(LocalSigningProvisioningConflict):
        # A differently bound operation cannot take over pending recovery.
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id="other-operation",
        )
    with pytest.raises(custody.LocalSigningCustodyError, match="recovery required"):
        provider.verify_historical(b"old", old.signature, old.snapshot)
    rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
        rotation_operation_id="runtime-R5",
    )
    assert provider.verify_historical(b"old", old.signature, old.snapshot)


def test_completed_pending_retry_cleans_and_restores_runtime(
    tmp_path: Path, security: SecurityProfileIdentity
) -> None:
    role = ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING
    provision(tmp_path, security, role)
    provider = LocalCHAFreshnessProposerSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    old = provider.sign_freshness_proposal(b"old")

    def crash_after_completed(stage: str) -> None:
        if stage == "R6_COMPLETED":
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match="R6_COMPLETED"):
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id="completed-pending",
            fault_injector=crash_after_completed,
        )
    intent, retained_prepared, successor_prepared = custody._rotation_paths(tmp_path, role)
    assert all(path.exists() for path in (intent, retained_prepared, successor_prepared))
    with pytest.raises(custody.LocalSigningCustodyError, match="recovery required"):
        LocalCHAFreshnessProposerSigningProvider(
            tmp_path,
            keyring_index_path=tmp_path / "keyring.json",
            security=security,
        )
    with pytest.raises(LocalSigningProvisioningConflict):
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id="operation-Y",
        )
    v2 = rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
        rotation_operation_id="completed-pending",
    )
    assert not any(path.exists() for path in (intent, retained_prepared, successor_prepared))
    restarted = LocalCHAFreshnessProposerSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    assert restarted.sign_freshness_proposal(b"new").snapshot.key_version == 2
    assert restarted.verify_historical(b"old", old.signature, old.snapshot)
    assert (
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id="completed-pending",
        )
        == v2
    )


@pytest.mark.parametrize(
    "cleanup_cut",
    [
        "CLEANUP_AFTER_RETAINED_PREPARED",
        "CLEANUP_AFTER_SUCCESSOR_PREPARED",
        "CLEANUP_AFTER_INTENT_UNLINK",
        "CLEANUP_AFTER_FSYNC",
    ],
)
def test_repeated_cleanup_crashes_converge(
    tmp_path: Path,
    security: SecurityProfileIdentity,
    cleanup_cut: str,
) -> None:
    role = ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING
    provision(tmp_path, security, role)

    def initial_crash(stage: str) -> None:
        if stage == "R6_COMPLETED":
            raise RuntimeError(stage)

    operation_id = f"cleanup-{cleanup_cut}"
    with pytest.raises(RuntimeError, match="R6_COMPLETED"):
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id=operation_id,
            fault_injector=initial_crash,
        )

    def cleanup_crash(stage: str) -> None:
        if stage == cleanup_cut:
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match=cleanup_cut):
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id=operation_id,
            fault_injector=cleanup_crash,
        )
    recovered = rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
        rotation_operation_id=operation_id,
    )
    assert recovered.key_version == 2
    assert not custody._pending_rotation(tmp_path, role)


def test_completed_cleanup_preserves_retained_and_successor_descendants(
    tmp_path: Path, security: SecurityProfileIdentity
) -> None:
    role = ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING
    provision(tmp_path, security, role)
    provider = LocalFreshnessAuthorityFinalizationSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    old = provider.sign_finalization(b"old")

    def crash_after_completed(stage: str) -> None:
        if stage == "R6_COMPLETED":
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match="R6_COMPLETED"):
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, role),
            security=security,
            role=role,
            rotation_operation_id="descendants",
            fault_injector=crash_after_completed,
        )
    current = custody._read_record(custody._record_path(tmp_path, role), security, role)
    transition_local_signing_credential_lifecycle(
        tmp_path,
        security=security,
        role=role,
        credential_identity=old.snapshot.credential_identity,
        expected_lifecycle_generation=2,
        target=SigningKeyLifecycle.REVOKED,
    )
    transition_local_signing_credential_lifecycle(
        tmp_path,
        security=security,
        role=role,
        credential_identity=current.credential_identity(),
        expected_lifecycle_generation=1,
        target=SigningKeyLifecycle.REVOKED,
    )
    replay = rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
        rotation_operation_id="descendants",
    )
    assert replay.lifecycle_state is SigningKeyLifecycle.ACTIVE
    restarted = LocalFreshnessAuthorityFinalizationSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    assert restarted.lifecycle_state() is SigningKeyLifecycle.REVOKED
    with pytest.raises(LocalSigningLifecycleError, match="historical trust"):
        restarted.verify_historical(b"old", old.signature, old.snapshot)


def test_operation_id_is_namespaced_by_exact_authority_scope(
    tmp_path: Path, security: SecurityProfileIdentity
) -> None:
    operation_id = "same-textual-operation"
    roles = (
        ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING,
        ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING,
    )
    results = []
    for role in roles:
        provision(tmp_path, security, role)
        results.append(
            rotate_local_freshness_signing_authority(
                tmp_path,
                admin(tmp_path, security, role),
                security=security,
                role=role,
                rotation_operation_id=operation_id,
            )
        )
    assert [result.key_version for result in results] == [2, 2]
    assert results[0].credential_identity() != results[1].credential_identity()


def test_pending_rotation_is_role_local(tmp_path: Path, security: SecurityProfileIdentity) -> None:
    proposer_role = ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING
    other_roles = (
        ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING,
        ProviderRole.ROOT_PROOF_SIGNING,
        ProviderRole.HISTORY_ATTESTATION_SIGNING,
    )
    provision(tmp_path, security, proposer_role)
    for role in other_roles:
        provision(tmp_path, security, role)

    def crash_at_r5(stage: str) -> None:
        if stage == "R5":
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match="R5"):
        rotate_local_freshness_signing_authority(
            tmp_path,
            admin(tmp_path, security, proposer_role),
            security=security,
            role=proposer_role,
            rotation_operation_id="role-local",
            fault_injector=crash_at_r5,
        )
    finalizer = LocalFreshnessAuthorityFinalizationSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    root = LocalRootProofSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    history = LocalHistoryAttestationSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    assert finalizer.sign_finalization(b"final").signature
    assert root.sign_root_proof(b"root")
    assert history.sign_history_head(b"history")
    with pytest.raises(custody.LocalSigningCustodyError, match="recovery required"):
        LocalCHAFreshnessProposerSigningProvider(
            tmp_path,
            keyring_index_path=tmp_path / "keyring.json",
            security=security,
        )


def test_retained_verify_and_revoke_share_one_linearization_lock(
    tmp_path: Path,
    security: SecurityProfileIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    role = ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING
    provision(tmp_path, security, role)
    provider = LocalCHAFreshnessProposerSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring.json", security=security
    )
    old = provider.sign_freshness_proposal(b"old")
    rotate_local_freshness_signing_authority(
        tmp_path,
        admin(tmp_path, security, role),
        security=security,
        role=role,
        rotation_operation_id="verify-revoke",
    )
    entered = threading.Event()
    release = threading.Event()
    original = custody._resolve_credential_record

    def blocked_resolve(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(custody, "_resolve_credential_record", blocked_resolve)
    verified: list[bool] = []
    verifier = threading.Thread(
        target=lambda: verified.append(
            provider.verify_historical(b"old", old.signature, old.snapshot)
        )
    )
    verifier.start()
    assert entered.wait(5)
    revoked = threading.Event()
    transitioner = threading.Thread(
        target=lambda: (
            transition_local_signing_credential_lifecycle(
                tmp_path,
                security=security,
                role=role,
                credential_identity=old.snapshot.credential_identity,
                expected_lifecycle_generation=2,
                target=SigningKeyLifecycle.REVOKED,
            ),
            revoked.set(),
        )
    )
    transitioner.start()
    assert not revoked.wait(0.05)
    release.set()
    verifier.join(5)
    transitioner.join(5)
    assert verified == [True]
    with pytest.raises(LocalSigningLifecycleError, match="historical trust"):
        provider.verify_historical(b"old", old.signature, old.snapshot)


def test_rotate_and_sign_share_one_linearization_lock(
    tmp_path: Path,
    security: SecurityProfileIdentity,
    monkeypatch: pytest.MonkeyPatch,
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
        if threading.current_thread().name == "old-signer":
            entered.set()
            assert release.wait(5)
        return original(self, reference)

    monkeypatch.setattr(NativeKeyringSigningSecretReader, "read_authority_secret", blocked)
    signed = []
    signer = threading.Thread(
        target=lambda: signed.append(provider.sign_finalization(b"old")),
        name="old-signer",
    )
    signer.start()
    assert entered.wait(5)
    rotated = threading.Event()
    rotator = threading.Thread(
        target=lambda: (
            rotate_local_freshness_signing_authority(
                tmp_path,
                admin(tmp_path, security, role),
                security=security,
                role=role,
                rotation_operation_id="rotate-sign",
            ),
            rotated.set(),
        )
    )
    rotator.start()
    assert not rotated.wait(0.05)
    release.set()
    signer.join(5)
    rotator.join(5)
    assert signed[0].snapshot.key_version == 1
    assert provider.sign_finalization(b"new").snapshot.key_version == 2


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


@pytest.mark.parametrize("target", [SigningKeyLifecycle.VERIFY_ONLY, SigningKeyLifecycle.REVOKED])
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
