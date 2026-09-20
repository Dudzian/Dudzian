from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field, fields, replace
import hashlib
import inspect

import pytest

from bot_core.root_proof_issuer_substrate import (
    CheckpointCapabilities,
    CredentialRoleIdentity,
    CredentialSemanticRole,
    ProviderCapabilities,
    ProviderIdentity,
    ProviderQualificationPolicy,
    ProviderRole,
    QualificationFailureCode,
    RootProofIssuerCompositionGate,
    RootProofIssuerSubstrateConfig,
    RootProofSigningProvider,
    SecurityProfile,
    SecurityProfileIdentity,
    HistoryAttestationSigningProvider,
    SigningCapabilities,
    public_key_material_identity,
    _ROLE_CAPABILITIES,
    _ROLE_METHODS,
    _ROLE_PORTS,
)


LOCAL_SIGNING = SigningCapabilities(True, True, False, False, True, True, True, True)
SERVER_SIGNING = replace(
    LOCAL_SIGNING,
    hardware_or_equivalent_secure_custody=True,
    plaintext_private_key_export_forbidden=True,
)
LOCAL_CHECKPOINT = CheckpointCapabilities(True, True, True, False, False, True, True, True, True)
SERVER_CHECKPOINT = replace(
    LOCAL_CHECKPOINT,
    independent_rollback_domain=True,
    independent_admin_or_security_domain=True,
)


def _test_public_key_bytes(credential_identity: str) -> bytes:
    """Return deterministic 32-byte test material, not generated Ed25519 crypto."""

    return hashlib.sha256(
        b"test-ed25519-public-key\x00" + credential_identity.encode()
    ).digest()


@dataclass(frozen=True)
class GenericProvider:
    identity: ProviderIdentity
    capabilities: ProviderCapabilities
    credentials: tuple[CredentialRoleIdentity, ...] = ()
    _credential_reads: int = field(default=0, compare=False)
    _active_reads: int = field(default=0, compare=False)
    _public_key_reads: int = field(default=0, compare=False)

    def credential_identities(self) -> tuple[CredentialRoleIdentity, ...]:
        object.__setattr__(self, "_credential_reads", self._credential_reads + 1)
        return self.credentials


class TrustRootFake(GenericProvider):
    def active_bundle(self) -> bytes: return b"bundle"
    def verify_signed_successor(self, candidate: bytes) -> bool: return bool(candidate)


class EntitlementFake(GenericProvider):
    def authoritative_state(self, subject): return subject
    def compare_and_swap_bind(self, request): return request
    def state_at_revision(self, subject, authoritative_state_revision): return subject, authoritative_state_revision
    def retained_history(self, subject): return subject


class ClaimantFake(GenericProvider):
    def resolve_claimant(self, claimant_id: str) -> object: return claimant_id
    def historical_claimant(self, claimant_id: str, generation: int) -> object: return claimant_id, generation


class RequesterFake(GenericProvider):
    def active_requester_credential(self, requester_id: str) -> object: return requester_id
    def historical_requester_credential(self, credential_id: str) -> object: return credential_id


class SigningIdentityFakeBase(GenericProvider):
    def active_credential_identity(self) -> CredentialRoleIdentity:
        object.__setattr__(self, "_active_reads", self._active_reads + 1)
        return self.credentials[0]

    def public_key(self, credential_identity: str) -> bytes:
        object.__setattr__(self, "_public_key_reads", self._public_key_reads + 1)
        return _test_public_key_bytes(credential_identity)
    def lifecycle_generation(self) -> int: return 1


class RootSigningFake(SigningIdentityFakeBase):
    def sign_root_proof(self, canonical_payload: bytes) -> bytes: return canonical_payload


class HistorySigningFake(SigningIdentityFakeBase):
    def sign_history_head(self, canonical_head: bytes) -> bytes: return canonical_head


class DualSigningFake(SigningIdentityFakeBase):
    def sign_root_proof(self, canonical_payload: bytes) -> bytes: return canonical_payload
    def sign_history_head(self, canonical_head: bytes) -> bytes: return canonical_head


class HistoryFake(GenericProvider):
    def current_head(self) -> object: return 1
    def append_exact_successor(self, expected_head: object, record: object) -> object: return record
    def record_at(self, sequence: int) -> object: return sequence


class CheckpointFake(GenericProvider):
    def current_checkpoint(self) -> object: return 1
    def advance_exact_successor(self, expected: object, successor: object) -> bool: return expected != successor
    def authenticated_checkpoint_at(self, sequence: int) -> object: return sequence


class ReconciliationFake(GenericProvider):
    def evidence_for(self, subject_id: str, history_head: bytes) -> object: return subject_id, history_head
    def verify_evidence(self, evidence: object) -> bool: return evidence is not None


class AttemptStoreFake(GenericProvider):
    def reserve_or_resolve_attempt_id(self, authorization: object) -> object: return authorization
    def finalize_attempt(self, identity: object, *, expected_fence: int) -> object: return identity, expected_fence
    def replace_after_authoritative_unbound(self, authorization: object, evidence: object, *, expected_fence: int) -> object: return authorization, evidence, expected_fence
    def record_recovery_resolution(self, operation_id: str, resolution: object, *, expected_fence: int) -> object: return operation_id, resolution, expected_fence
    def attempt(self, operation_id: str) -> object: return operation_id


ROLE_FAKES = {
    ProviderRole.DEPLOYMENT_TRUST_ROOT: TrustRootFake,
    ProviderRole.ENTITLEMENT_REGISTRY: EntitlementFake,
    ProviderRole.CLAIMANT_IDENTITY_REGISTRY: ClaimantFake,
    ProviderRole.REQUESTER_CREDENTIAL_REGISTRY: RequesterFake,
    ProviderRole.ROOT_PROOF_SIGNING: RootSigningFake,
    ProviderRole.HISTORY_ATTESTATION_SIGNING: HistorySigningFake,
    ProviderRole.ISSUER_AUTHENTICATED_HISTORY: HistoryFake,
    ProviderRole.CHECKPOINT_AUTHORITY: CheckpointFake,
    ProviderRole.RECONCILIATION_EVIDENCE: ReconciliationFake,
    ProviderRole.CHA_ATTEMPT_STORE: AttemptStoreFake,
}

CREDENTIAL_ROLES = {
    ProviderRole.CLAIMANT_IDENTITY_REGISTRY: CredentialSemanticRole.ROOT_PROOF_CLAIMANT,
    ProviderRole.REQUESTER_CREDENTIAL_REGISTRY: CredentialSemanticRole.ROOT_PROOF_REQUESTER,
    ProviderRole.ROOT_PROOF_SIGNING: CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING,
    ProviderRole.HISTORY_ATTESTATION_SIGNING: CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING,
}


def credential(role: CredentialSemanticRole, namespace: str, suffix: str) -> CredentialRoleIdentity:
    credential_id = f"credential-{suffix}"
    material = None
    if role in {
        CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING,
        CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING,
    }:
        material = public_key_material_identity(_test_public_key_bytes(credential_id))
    return CredentialRoleIdentity(
        role,
        credential_id,
        namespace,
        f"key-{suffix}",
        f"life-{suffix}",
        material,
    )


def capabilities_for(role: ProviderRole, signing, checkpoint) -> ProviderCapabilities:
    base = ProviderCapabilities(implemented=True)
    if role in {ProviderRole.ROOT_PROOF_SIGNING, ProviderRole.HISTORY_ATTESTATION_SIGNING}:
        return replace(base, signing=signing)
    if role is ProviderRole.CHECKPOINT_AUTHORITY:
        return replace(base, durable_state=True, checkpoint=checkpoint)
    authoritative = role is not ProviderRole.CHECKPOINT_AUTHORITY
    durable = role not in {ProviderRole.RECONCILIATION_EVIDENCE}
    cas = role in {ProviderRole.ENTITLEMENT_REGISTRY, ProviderRole.CHA_ATTEMPT_STORE}
    return replace(base, authoritative_reads=authoritative, durable_state=durable, compare_and_swap=cas)


def providers(
    profile: SecurityProfile,
    *,
    trust_domain: str | None = None,
    signing: SigningCapabilities = LOCAL_SIGNING,
    checkpoint: CheckpointCapabilities = LOCAL_CHECKPOINT,
) -> list[GenericProvider]:
    domain = trust_domain or f"{profile.value.lower()}-domain"
    security = SecurityProfileIdentity(profile, domain)
    result = []
    for role in ProviderRole:
        namespace = f"{domain}.{role.value.lower()}"
        semantic_role = CREDENTIAL_ROLES.get(role)
        identities = (credential(semantic_role, namespace, role.value.lower()),) if semantic_role else ()
        result.append(
            ROLE_FAKES[role](
                ProviderIdentity(role, security, namespace),
                capabilities_for(role, signing, checkpoint),
                identities,
            )
        )
    return result


def qualify(items, security: SecurityProfileIdentity | None = None):
    return RootProofIssuerCompositionGate().qualify(security or items[0].identity.security, items)


def replace_provider(items, role: ProviderRole, **changes):
    index = next(i for i, item in enumerate(items) if item.identity.role is role)
    items[index] = replace(items[index], **changes)
    return items[index]


@pytest.mark.parametrize(
    ("profile", "signing", "checkpoint"),
    [
        (SecurityProfile.DEVELOPMENT, LOCAL_SIGNING, LOCAL_CHECKPOINT),
        (SecurityProfile.TEST, LOCAL_SIGNING, LOCAL_CHECKPOINT),
        (SecurityProfile.PRODUCTION_LOCAL, LOCAL_SIGNING, LOCAL_CHECKPOINT),
        (SecurityProfile.PRODUCTION_SERVER_READY, SERVER_SIGNING, SERVER_CHECKPOINT),
    ],
)
def test_role_specific_full_compositions_qualify(profile, signing, checkpoint):
    assert qualify(providers(profile, signing=signing, checkpoint=checkpoint)).qualified


@pytest.mark.parametrize("role", list(ProviderRole))
def test_generic_provider_claiming_each_role_is_interface_mismatch(role):
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    original = next(item for item in items if item.identity.role is role)
    replace_provider(
        items,
        role,
        identity=original.identity,
        capabilities=original.capabilities,
        credentials=original.credentials,
    )
    index = next(i for i, item in enumerate(items) if item.identity.role is role)
    items[index] = GenericProvider(original.identity, original.capabilities, original.credentials)
    assert QualificationFailureCode.PROVIDER_INTERFACE_MISMATCH in {f.code for f in qualify(items).failures}


def test_root_and_history_signing_ports_are_structurally_independent():
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    root = next(item for item in items if item.identity.role is ProviderRole.ROOT_PROOF_SIGNING)
    history = next(item for item in items if item.identity.role is ProviderRole.HISTORY_ATTESTATION_SIGNING)
    assert isinstance(root, RootProofSigningProvider)
    assert not isinstance(root, HistoryAttestationSigningProvider)
    assert isinstance(history, HistoryAttestationSigningProvider)
    assert not isinstance(history, RootProofSigningProvider)


@pytest.mark.parametrize(
    ("role", "wrong_fake"),
    [
        (ProviderRole.ROOT_PROOF_SIGNING, HistorySigningFake),
        (ProviderRole.HISTORY_ATTESTATION_SIGNING, RootSigningFake),
    ],
)
def test_opposite_signing_authority_port_is_rejected(role, wrong_fake):
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    item = next(item for item in items if item.identity.role is role)
    items[items.index(item)] = wrong_fake(item.identity, item.capabilities, item.credentials)
    assert QualificationFailureCode.PROVIDER_INTERFACE_MISMATCH in {
        failure.code for failure in qualify(items).failures
    }


def test_provider_evidence_is_read_once_per_qualification_pass():
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    result = qualify(items)
    assert result.qualified
    assert all(item._credential_reads == 1 for item in items)
    signing = [item for item in items if item.identity.role in {ProviderRole.ROOT_PROOF_SIGNING, ProviderRole.HISTORY_ATTESTATION_SIGNING}]
    assert all(item._active_reads == 1 for item in signing)
    assert all(item._public_key_reads == 1 for item in signing)


def test_security_capability_defaults_are_all_false():
    evidence = ProviderCapabilities(implemented=True)
    assert not evidence.authoritative_reads
    assert not evidence.durable_state
    assert not evidence.compare_and_swap


@pytest.mark.parametrize(
    ("role", "field"),
    [
        (ProviderRole.DEPLOYMENT_TRUST_ROOT, "authoritative_reads"),
        (ProviderRole.DEPLOYMENT_TRUST_ROOT, "durable_state"),
        (ProviderRole.CLAIMANT_IDENTITY_REGISTRY, "durable_state"),
        (ProviderRole.REQUESTER_CREDENTIAL_REGISTRY, "authoritative_reads"),
        (ProviderRole.ISSUER_AUTHENTICATED_HISTORY, "durable_state"),
        (ProviderRole.RECONCILIATION_EVIDENCE, "authoritative_reads"),
    ],
)
def test_role_specific_minimum_capabilities_fail_closed(role, field):
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    item = next(item for item in items if item.identity.role is role)
    replace_provider(items, role, capabilities=replace(item.capabilities, **{field: False}))
    assert QualificationFailureCode.INSUFFICIENT_CAPABILITIES in {f.code for f in qualify(items).failures}


@pytest.mark.parametrize(
    "role",
    list(CREDENTIAL_ROLES),
)
def test_credential_bearing_provider_requires_identity_evidence(role):
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    replace_provider(items, role, credentials=())
    assert QualificationFailureCode.MISSING_CREDENTIAL_IDENTITY_EVIDENCE in {f.code for f in qualify(items).failures}


@pytest.mark.parametrize(
    ("role", "wrong_role"),
    [
        (ProviderRole.CLAIMANT_IDENTITY_REGISTRY, CredentialSemanticRole.ROOT_PROOF_REQUESTER),
        (ProviderRole.REQUESTER_CREDENTIAL_REGISTRY, CredentialSemanticRole.ROOT_PROOF_CLAIMANT),
        (ProviderRole.ROOT_PROOF_SIGNING, CredentialSemanticRole.ROOT_PROOF_REQUESTER),
        (ProviderRole.HISTORY_ATTESTATION_SIGNING, CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING),
    ],
)
def test_signing_provider_rejects_wrong_semantic_role(role, wrong_role):
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    item = next(item for item in items if item.identity.role is role)
    replace_provider(items, role, credentials=(replace(item.credentials[0], semantic_role=wrong_role),))
    assert QualificationFailureCode.CREDENTIAL_ROLE_MISMATCH in {f.code for f in qualify(items).failures}


def test_provider_and_credential_namespace_must_match():
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    item = next(item for item in items if item.identity.role is ProviderRole.ROOT_PROOF_SIGNING)
    replace_provider(items, item.identity.role, credentials=(replace(item.credentials[0], provider_namespace="other"),))
    assert QualificationFailureCode.CREDENTIAL_NAMESPACE_MISMATCH in {f.code for f in qualify(items).failures}


@pytest.mark.parametrize(
    "role",
    [ProviderRole.ROOT_PROOF_SIGNING, ProviderRole.HISTORY_ATTESTATION_SIGNING],
)
def test_stable_signing_handle_claim_requires_handle_evidence(role):
    items = providers(SecurityProfile.PRODUCTION_SERVER_READY, signing=SERVER_SIGNING, checkpoint=SERVER_CHECKPOINT)
    item = next(item for item in items if item.identity.role is role)
    replace_provider(items, role, credentials=(replace(item.credentials[0], key_handle_or_version=None),))
    assert QualificationFailureCode.MISSING_KEY_HANDLE_IDENTITY in {f.code for f in qualify(items).failures}


@pytest.mark.parametrize(
    ("role", "fake_type"),
    [
        (ProviderRole.ROOT_PROOF_SIGNING, RootSigningFake),
        (ProviderRole.HISTORY_ATTESTATION_SIGNING, HistorySigningFake),
    ],
)
def test_signing_metadata_laundering_is_rejected(role, fake_type):
    class LaunderingFake(fake_type):
        def active_credential_identity(self):
            return replace(self.credentials[0], credential_identity="active-other")

    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    item = next(item for item in items if item.identity.role is role)
    index = items.index(item)
    items[index] = LaunderingFake(item.identity, item.capabilities, item.credentials)
    assert QualificationFailureCode.ACTIVE_CREDENTIAL_IDENTITY_MISMATCH in {
        failure.code for failure in qualify(items).failures
    }


@pytest.mark.parametrize(
    "role",
    [ProviderRole.ROOT_PROOF_SIGNING, ProviderRole.HISTORY_ATTESTATION_SIGNING],
)
def test_signing_composition_identity_cardinality_is_exact(role):
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    item = next(item for item in items if item.identity.role is role)
    second = replace(item.credentials[0], credential_identity="second")
    replace_provider(items, role, credentials=(item.credentials[0], second))
    assert QualificationFailureCode.AMBIGUOUS_CREDENTIAL_IDENTITY_EVIDENCE in {
        failure.code for failure in qualify(items).failures
    }


def test_active_signing_identity_namespace_mismatch_is_rejected():
    class NamespaceLaunderingFake(RootSigningFake):
        def active_credential_identity(self):
            return replace(self.credentials[0], provider_namespace="other.namespace")

    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    item = next(item for item in items if item.identity.role is ProviderRole.ROOT_PROOF_SIGNING)
    items[items.index(item)] = NamespaceLaunderingFake(item.identity, item.capabilities, item.credentials)
    assert QualificationFailureCode.ACTIVE_CREDENTIAL_IDENTITY_MISMATCH in {
        failure.code for failure in qualify(items).failures
    }


@pytest.mark.parametrize("failure_mode", ["exception", "invalid_type"])
def test_active_signing_identity_failure_is_structured_and_fail_closed(failure_mode):
    class FailingActiveFake(RootSigningFake):
        def active_credential_identity(self):
            if failure_mode == "exception":
                raise RuntimeError("unavailable")
            return object()

    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    item = next(item for item in items if item.identity.role is ProviderRole.ROOT_PROOF_SIGNING)
    items[items.index(item)] = FailingActiveFake(item.identity, item.capabilities, item.credentials)
    expected = (
        QualificationFailureCode.PROVIDER_EVIDENCE_UNAVAILABLE
        if failure_mode == "exception"
        else QualificationFailureCode.ACTIVE_CREDENTIAL_IDENTITY_MISMATCH
    )
    assert expected in {failure.code for failure in qualify(items).failures}


def test_public_key_fingerprint_must_match_declared_material_identity():
    class DifferentPublicKeyFake(RootSigningFake):
        def public_key(self, credential_identity: str) -> bytes:
            return _test_public_key_bytes("different-key")

    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    item = next(item for item in items if item.identity.role is ProviderRole.ROOT_PROOF_SIGNING)
    items[items.index(item)] = DifferentPublicKeyFake(item.identity, item.capabilities, item.credentials)
    assert QualificationFailureCode.KEY_MATERIAL_IDENTITY_MISMATCH in {
        failure.code for failure in qualify(items).failures
    }


def test_missing_signing_key_material_identity_is_rejected():
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    item = next(item for item in items if item.identity.role is ProviderRole.ROOT_PROOF_SIGNING)
    replace_provider(
        items,
        item.identity.role,
        credentials=(replace(item.credentials[0], key_material_identity=None),),
    )
    codes = {failure.code for failure in qualify(items).failures}
    assert QualificationFailureCode.MISSING_KEY_MATERIAL_IDENTITY in codes


def test_same_signing_material_in_different_namespaces_is_rejected():
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    root = next(item for item in items if item.identity.role is ProviderRole.ROOT_PROOF_SIGNING)
    history = next(item for item in items if item.identity.role is ProviderRole.HISTORY_ATTESTATION_SIGNING)
    shared_credential_id = root.credentials[0].credential_identity
    shared_material = root.credentials[0].key_material_identity
    replace_provider(
        items,
        history.identity.role,
        credentials=(
            replace(
                history.credentials[0],
                credential_identity=shared_credential_id,
                key_material_identity=shared_material,
            ),
        ),
    )
    assert QualificationFailureCode.KEY_MATERIAL_ALIAS in {
        failure.code for failure in qualify(items).failures
    }


def test_dynamic_credential_evidence_cannot_bypass_material_alias_check():
    class RotatingEvidenceRoot(RootSigningFake):
        def credential_identities(self):
            object.__setattr__(self, "_credential_reads", self._credential_reads + 1)
            if self._credential_reads == 1:
                return self.credentials
            rotated_id = "rotated-after-snapshot"
            return (
                replace(
                    self.credentials[0],
                    credential_identity=rotated_id,
                    key_material_identity=public_key_material_identity(
                        _test_public_key_bytes(rotated_id)
                    ),
                ),
            )

    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    root = next(item for item in items if item.identity.role is ProviderRole.ROOT_PROOF_SIGNING)
    history = next(item for item in items if item.identity.role is ProviderRole.HISTORY_ATTESTATION_SIGNING)
    rotating = RotatingEvidenceRoot(root.identity, root.capabilities, root.credentials)
    items[items.index(root)] = rotating
    replace_provider(
        items,
        history.identity.role,
        credentials=(
            replace(
                history.credentials[0],
                credential_identity=root.credentials[0].credential_identity,
                key_material_identity=root.credentials[0].key_material_identity,
            ),
        ),
    )
    result = qualify(items)
    assert rotating._credential_reads == 1
    assert QualificationFailureCode.KEY_MATERIAL_ALIAS in {
        failure.code for failure in result.failures
    }


def test_identity_and_capabilities_are_captured_once():
    class DynamicEvidenceRoot:
        def __init__(self, delegate):
            self.delegate = delegate
            self.identity_reads = 0
            self.capability_reads = 0

        @property
        def identity(self):
            self.identity_reads += 1
            if self.identity_reads == 1:
                return self.delegate.identity
            return replace(
                self.delegate.identity,
                security=SecurityProfileIdentity(SecurityProfile.TEST, "changed"),
            )

        @property
        def capabilities(self):
            self.capability_reads += 1
            if self.capability_reads == 1:
                return self.delegate.capabilities
            return replace(self.delegate.capabilities, implemented=False)

        def credential_identities(self): return self.delegate.credential_identities()
        def active_credential_identity(self): return self.delegate.active_credential_identity()
        def public_key(self, credential_identity): return self.delegate.public_key(credential_identity)
        def lifecycle_generation(self): return 1
        def sign_root_proof(self, canonical_payload): return canonical_payload

    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    root = next(item for item in items if item.identity.role is ProviderRole.ROOT_PROOF_SIGNING)
    dynamic = DynamicEvidenceRoot(root)
    items[items.index(root)] = dynamic
    assert qualify(items).qualified
    assert dynamic.identity_reads == 1
    assert dynamic.capability_reads == 1


def test_same_signing_handle_in_different_namespaces_with_distinct_material_is_accepted():
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    root = next(item for item in items if item.identity.role is ProviderRole.ROOT_PROOF_SIGNING)
    history = next(item for item in items if item.identity.role is ProviderRole.HISTORY_ATTESTATION_SIGNING)
    replace_provider(
        items,
        history.identity.role,
        credentials=(
            replace(
                history.credentials[0],
                key_handle_or_version=root.credentials[0].key_handle_or_version,
            ),
        ),
    )
    assert qualify(items).qualified


def test_structural_port_check_does_not_claim_exact_signature_validation():
    class WrongSignatureTrustRoot(TrustRootFake):
        def active_bundle(self, unexpected): return unexpected
        def verify_signed_successor(self): return True

    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    item = next(item for item in items if item.identity.role is ProviderRole.DEPLOYMENT_TRUST_ROOT)
    items[items.index(item)] = WrongSignatureTrustRoot(item.identity, item.capabilities, item.credentials)
    assert qualify(items).qualified


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("credential_identity", QualificationFailureCode.CREDENTIAL_IDENTITY_ALIAS),
        ("key_handle_or_version", QualificationFailureCode.KEY_HANDLE_ALIAS),
    ],
)
def test_same_namespace_alias_for_forbidden_roles_is_rejected(field, code):
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    claimant = next(item for item in items if item.identity.role is ProviderRole.CLAIMANT_IDENTITY_REGISTRY)
    requester = next(item for item in items if item.identity.role is ProviderRole.REQUESTER_CREDENTIAL_REGISTRY)
    shared_namespace = "shared-custody"
    replace_provider(items, claimant.identity.role, identity=replace(claimant.identity, provider_namespace=shared_namespace), credentials=(replace(claimant.credentials[0], provider_namespace=shared_namespace),))
    value = getattr(claimant.credentials[0], field)
    replace_provider(items, requester.identity.role, identity=replace(requester.identity, provider_namespace=shared_namespace), credentials=(replace(requester.credentials[0], provider_namespace=shared_namespace, **{field: value}),))
    assert code in {f.code for f in qualify(items).failures}


@pytest.mark.parametrize("field", ["credential_identity", "key_handle_or_version"])
def test_same_local_identity_in_different_namespaces_is_accepted(field):
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    claimant = next(item for item in items if item.identity.role is ProviderRole.CLAIMANT_IDENTITY_REGISTRY)
    requester = next(item for item in items if item.identity.role is ProviderRole.REQUESTER_CREDENTIAL_REGISTRY)
    replace_provider(items, requester.identity.role, credentials=(replace(requester.credentials[0], **{field: getattr(claimant.credentials[0], field)}),))
    assert qualify(items).qualified


@pytest.mark.parametrize(
    ("signing", "checkpoint"),
    [(LOCAL_SIGNING, SERVER_CHECKPOINT), (SERVER_SIGNING, LOCAL_CHECKPOINT)],
)
def test_server_ready_rejects_local_security_provider(signing, checkpoint):
    assert not qualify(providers(SecurityProfile.PRODUCTION_SERVER_READY, signing=signing, checkpoint=checkpoint)).qualified


def test_missing_duplicate_and_profile_mismatch_fail_closed():
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    assert QualificationFailureCode.MISSING_PROVIDER in {f.code for f in qualify(items[:-1]).failures}
    assert QualificationFailureCode.DUPLICATE_PROVIDER_ROLE in {f.code for f in qualify(items + [items[0]]).failures}
    server = SecurityProfileIdentity(SecurityProfile.PRODUCTION_SERVER_READY, "server-domain")
    assert QualificationFailureCode.PROFILE_IDENTITY_MISMATCH in {f.code for f in qualify(items, server).failures}


@pytest.mark.parametrize(
    "field",
    [
        "hardware_or_equivalent_secure_custody",
        "plaintext_private_key_export_forbidden",
        "role_isolation",
    ],
)
def test_server_signing_capability_mutations_fail_closed(field):
    signing = replace(SERVER_SIGNING, **{field: False})
    items = providers(SecurityProfile.PRODUCTION_SERVER_READY, signing=signing, checkpoint=SERVER_CHECKPOINT)
    assert not qualify(items).qualified


@pytest.mark.parametrize("field", ["independent_rollback_domain", "independent_admin_or_security_domain"])
def test_server_checkpoint_capability_mutations_fail_closed(field):
    checkpoint = replace(SERVER_CHECKPOINT, **{field: False})
    items = providers(SecurityProfile.PRODUCTION_SERVER_READY, signing=SERVER_SIGNING, checkpoint=checkpoint)
    assert not qualify(items).qualified


def test_class_name_and_caller_flag_do_not_affect_qualification():
    class ProductionHSM(RootSigningFake): pass

    items = providers(SecurityProfile.PRODUCTION_SERVER_READY, checkpoint=SERVER_CHECKPOINT)
    item = next(item for item in items if item.identity.role is ProviderRole.ROOT_PROOF_SIGNING)
    index = items.index(item)
    named = ProductionHSM(item.identity, item.capabilities, item.credentials)
    object.__setattr__(named, "server_ready", True)
    items[index] = named
    assert not qualify(items).qualified


def test_profile_identity_and_capabilities_are_immutable_and_exact():
    identity = SecurityProfileIdentity(SecurityProfile.TEST, "different-domain")
    with pytest.raises(FrozenInstanceError):
        identity.trust_domain = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        LOCAL_SIGNING.ed25519 = False  # type: ignore[misc]
    assert not qualify(providers(SecurityProfile.TEST), identity).qualified


def test_no_insecure_override_parameter_exists():
    assert set(inspect.signature(RootProofIssuerCompositionGate.qualify).parameters) == {"self", "security", "providers"}
    assert set(inspect.signature(RootProofIssuerCompositionGate.__init__).parameters) == {"self"}
    assert "force" not in inspect.signature(ProviderQualificationPolicy.failures_for).parameters


@pytest.mark.parametrize(
    "profile",
    ["PRODUCTION_SERVER_READY", "TEST"],
)
def test_security_profile_identity_rejects_string_enum(profile):
    with pytest.raises(TypeError):
        SecurityProfileIdentity(profile, "domain")  # type: ignore[arg-type]


@pytest.mark.parametrize("trust_domain", [object(), 1, None, "", "   "])
def test_security_profile_identity_requires_exact_nonempty_string(trust_domain):
    with pytest.raises((TypeError, ValueError)):
        SecurityProfileIdentity(SecurityProfile.TEST, trust_domain)  # type: ignore[arg-type]


def test_malformed_server_ready_string_cannot_bypass_gate():
    malformed = object.__new__(SecurityProfileIdentity)
    object.__setattr__(malformed, "profile", "PRODUCTION_SERVER_READY")
    object.__setattr__(malformed, "trust_domain", "prod")
    result = RootProofIssuerCompositionGate().qualify(
        malformed,  # type: ignore[arg-type]
        providers(SecurityProfile.PRODUCTION_LOCAL),
    )
    assert not result.qualified
    assert result.failures[0].code is QualificationFailureCode.INVALID_COMPOSITION_SECURITY_IDENTITY


def test_identity_value_objects_reject_enum_and_nested_type_confusion():
    security = SecurityProfileIdentity(SecurityProfile.TEST, "test")
    with pytest.raises(TypeError):
        ProviderIdentity("ROOT_PROOF_SIGNING", security, "namespace")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ProviderIdentity(ProviderRole.ROOT_PROOF_SIGNING, {}, "namespace")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ProviderIdentity(ProviderRole.ROOT_PROOF_SIGNING, security, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CredentialRoleIdentity(
            "ROOT_PROOF_REQUESTER",  # type: ignore[arg-type]
            "credential",
            "namespace",
            "handle",
            "lifecycle",
            None,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "hardware_or_equivalent_secure_custody",
        "plaintext_private_key_export_forbidden",
    ],
)
def test_signing_capabilities_reject_truthy_security_strings(field_name):
    with pytest.raises(TypeError):
        replace(SERVER_SIGNING, **{field_name: "false"})


@pytest.mark.parametrize(
    "field_name",
    ["independent_rollback_domain", "independent_admin_or_security_domain"],
)
def test_checkpoint_capabilities_reject_truthy_security_strings(field_name):
    with pytest.raises(TypeError):
        replace(SERVER_CHECKPOINT, **{field_name: "false"})


@pytest.mark.parametrize(
    ("evidence", "field_name"),
    [
        (LOCAL_SIGNING, "ed25519"),
        (LOCAL_CHECKPOINT, "authenticated"),
        (ProviderCapabilities(implemented=True), "implemented"),
    ],
)
@pytest.mark.parametrize("integer", [0, 1])
def test_capabilities_reject_integer_bool_confusion(evidence, field_name, integer):
    with pytest.raises(TypeError):
        replace(evidence, **{field_name: integer})


@pytest.mark.parametrize(
    "nested",
    [{"signing": object()}, {"checkpoint": {}}],
)
def test_provider_capabilities_reject_nested_type_confusion(nested):
    with pytest.raises(TypeError):
        ProviderCapabilities(implemented=True, **nested)


def test_all_valid_capability_boolean_fields_are_exact_bool():
    for evidence in (
        LOCAL_SIGNING,
        SERVER_SIGNING,
        LOCAL_CHECKPOINT,
        SERVER_CHECKPOINT,
        ProviderCapabilities(implemented=True),
    ):
        boolean_values = [
            getattr(evidence, item.name)
            for item in fields(evidence)
            if item.type is bool or isinstance(getattr(evidence, item.name), bool)
        ]
        assert boolean_values
        assert all(type(value) is bool for value in boolean_values)


def test_fabricated_invalid_provider_role_is_structured_failure():
    security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "domain")
    malformed = object.__new__(ProviderIdentity)
    object.__setattr__(malformed, "role", "ROOT_PROOF_SIGNING")
    object.__setattr__(malformed, "security", security)
    object.__setattr__(malformed, "provider_namespace", "namespace")
    provider = GenericProvider(malformed, ProviderCapabilities(implemented=True))  # type: ignore[arg-type]
    result = RootProofIssuerCompositionGate().qualify(security, [provider])
    assert not result.qualified
    assert QualificationFailureCode.PROVIDER_EVIDENCE_UNAVAILABLE in {
        failure.code for failure in result.failures
    }


def test_arbitrary_provider_and_security_objects_fail_closed_without_exception():
    security = SecurityProfileIdentity(SecurityProfile.TEST, "test")
    provider_result = RootProofIssuerCompositionGate().qualify(security, [object()])  # type: ignore[list-item]
    security_result = RootProofIssuerCompositionGate().qualify(object(), [])  # type: ignore[arg-type]
    assert not provider_result.qualified
    assert QualificationFailureCode.PROVIDER_EVIDENCE_UNAVAILABLE in {
        failure.code for failure in provider_result.failures
    }
    assert security_result.failures[0].code is QualificationFailureCode.INVALID_COMPOSITION_SECURITY_IDENTITY


def test_policy_rejects_invalid_profile_type_defense_in_depth():
    identity = ProviderIdentity(
        ProviderRole.DEPLOYMENT_TRUST_ROOT,
        SecurityProfileIdentity(SecurityProfile.TEST, "test"),
        "trust-root",
    )
    failures = ProviderQualificationPolicy().failures_for(
        "PRODUCTION_SERVER_READY",  # type: ignore[arg-type]
        identity,
        ProviderCapabilities(implemented=True),
    )
    assert failures == ("security profile evidence has invalid type",)


def test_valid_exact_substrate_config_is_accepted():
    security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "prod-local")
    config = RootProofIssuerSubstrateConfig(
        security,
        ((ProviderRole.ROOT_PROOF_SIGNING, "local-ed25519"),),
    )
    assert config.security is security


@pytest.mark.parametrize("security", ["PRODUCTION_SERVER_READY", {}, object()])
def test_substrate_config_rejects_malformed_security(security):
    with pytest.raises(TypeError):
        RootProofIssuerSubstrateConfig(security, ())  # type: ignore[arg-type]


def test_substrate_config_rejects_fabricated_nested_security():
    malformed = object.__new__(SecurityProfileIdentity)
    object.__setattr__(malformed, "profile", "PRODUCTION_LOCAL")
    object.__setattr__(malformed, "trust_domain", "prod")
    with pytest.raises(TypeError):
        RootProofIssuerSubstrateConfig(malformed, ())


@pytest.mark.parametrize(
    "entries",
    [
        [],
        ([ProviderRole.ROOT_PROOF_SIGNING, "adapter"],),
        ((ProviderRole.ROOT_PROOF_SIGNING,),),
        ((ProviderRole.ROOT_PROOF_SIGNING, "adapter", "extra"),),
        (("ROOT_PROOF_SIGNING", "adapter"),),
        ((ProviderRole.ROOT_PROOF_SIGNING, object()),),
    ],
)
def test_substrate_config_rejects_malformed_provider_selections(entries):
    security = SecurityProfileIdentity(SecurityProfile.TEST, "test")
    with pytest.raises(TypeError):
        RootProofIssuerSubstrateConfig(security, entries)  # type: ignore[arg-type]


@pytest.mark.parametrize("selection", ["", "   "])
def test_substrate_config_rejects_blank_selection(selection):
    security = SecurityProfileIdentity(SecurityProfile.TEST, "test")
    with pytest.raises(ValueError):
        RootProofIssuerSubstrateConfig(
            security,
            ((ProviderRole.ROOT_PROOF_SIGNING, selection),),
        )


def test_substrate_config_rejects_duplicate_exact_role():
    security = SecurityProfileIdentity(SecurityProfile.TEST, "test")
    with pytest.raises(ValueError):
        RootProofIssuerSubstrateConfig(
            security,
            (
                (ProviderRole.ROOT_PROOF_SIGNING, "first"),
                (ProviderRole.ROOT_PROOF_SIGNING, "second"),
            ),
        )


def test_role_mappings_cover_exact_provider_role_set():
    expected = set(ProviderRole)
    assert set(_ROLE_PORTS) == expected
    assert set(_ROLE_METHODS) == expected
    assert set(_ROLE_CAPABILITIES) == expected


@pytest.mark.parametrize(
    "role",
    [ProviderRole.ROOT_PROOF_SIGNING, ProviderRole.HISTORY_ATTESTATION_SIGNING],
)
def test_dual_signing_authority_api_is_rejected(role):
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    item = next(item for item in items if item.identity.role is role)
    items[items.index(item)] = DualSigningFake(
        item.identity,
        item.capabilities,
        item.credentials,
    )
    result = qualify(items)
    assert QualificationFailureCode.SIGNING_AUTHORITY_INTERFACE_ALIAS in {
        failure.code for failure in result.failures
    }


def _provider_with_active_override(items, role, active):
    base_type = (
        RootSigningFake
        if role is ProviderRole.ROOT_PROOF_SIGNING
        else HistorySigningFake
    )

    class ActiveOverride(base_type):
        def active_credential_identity(self):
            return active

    item = next(item for item in items if item.identity.role is role)
    items[items.index(item)] = ActiveOverride(
        item.identity,
        item.capabilities,
        item.credentials,
    )


@pytest.mark.parametrize(
    "role",
    [ProviderRole.ROOT_PROOF_SIGNING, ProviderRole.HISTORY_ATTESTATION_SIGNING],
)
def test_active_credential_subclass_and_custom_equality_are_rejected(role):
    class EqualityBypassCredential(CredentialRoleIdentity):
        def __eq__(self, other):
            return True

    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    declared = next(item for item in items if item.identity.role is role).credentials[0]
    malicious = object.__new__(EqualityBypassCredential)
    for item in fields(declared):
        object.__setattr__(malicious, item.name, getattr(declared, item.name))
    _provider_with_active_override(items, role, malicious)
    result = qualify(items)
    assert QualificationFailureCode.ACTIVE_CREDENTIAL_IDENTITY_MISMATCH in {
        failure.code for failure in result.failures
    }


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("semantic_role", "ROOT_PROOF_ISSUER_SIGNING"),
        ("provider_namespace", object()),
        ("custody_lifecycle_namespace", 1),
        ("key_handle_or_version", object()),
        ("key_material_identity", {}),
    ],
)
def test_fabricated_malformed_active_credential_fails_closed(field_name, invalid_value):
    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    item = next(item for item in items if item.identity.role is ProviderRole.ROOT_PROOF_SIGNING)
    declared = item.credentials[0]
    malformed = object.__new__(CredentialRoleIdentity)
    for name in (
        "semantic_role",
        "credential_identity",
        "provider_namespace",
        "key_handle_or_version",
        "custody_lifecycle_namespace",
        "key_material_identity",
    ):
        object.__setattr__(malformed, name, getattr(declared, name))
    object.__setattr__(malformed, field_name, invalid_value)
    _provider_with_active_override(items, item.identity.role, malformed)
    result = qualify(items)
    assert QualificationFailureCode.ACTIVE_CREDENTIAL_IDENTITY_MISMATCH in {
        failure.code for failure in result.failures
    }


def test_policy_rejects_malformed_identity_and_capability_types():
    failures = ProviderQualificationPolicy().failures_for(
        SecurityProfile.TEST,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
    )
    assert failures == ("provider identity or capability evidence has invalid type",)


def _fabricated_signing_capabilities(field_name, invalid_value):
    malformed = object.__new__(SigningCapabilities)
    for item in fields(SERVER_SIGNING):
        object.__setattr__(malformed, item.name, getattr(SERVER_SIGNING, item.name))
    object.__setattr__(malformed, field_name, invalid_value)
    return malformed


def _fabricated_checkpoint_capabilities(field_name, invalid_value):
    malformed = object.__new__(CheckpointCapabilities)
    for item in fields(SERVER_CHECKPOINT):
        object.__setattr__(malformed, item.name, getattr(SERVER_CHECKPOINT, item.name))
    object.__setattr__(malformed, field_name, invalid_value)
    return malformed


def _fabricated_provider_capabilities(*, signing=None, checkpoint=None):
    malformed = object.__new__(ProviderCapabilities)
    object.__setattr__(malformed, "implemented", True)
    object.__setattr__(malformed, "authoritative_reads", True)
    object.__setattr__(malformed, "durable_state", True)
    object.__setattr__(malformed, "compare_and_swap", True)
    object.__setattr__(malformed, "signing", signing)
    object.__setattr__(malformed, "checkpoint", checkpoint)
    return malformed


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("hardware_or_equivalent_secure_custody", "false"),
        ("ed25519", "false"),
        ("role_isolation", 1),
    ],
)
def test_policy_deeply_rejects_fabricated_signing_capabilities(field_name, invalid_value):
    malformed = _fabricated_signing_capabilities(field_name, invalid_value)
    security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_SERVER_READY, "server")
    identity = ProviderIdentity(ProviderRole.ROOT_PROOF_SIGNING, security, "root-signing")
    evidence = _fabricated_provider_capabilities(signing=malformed)
    failures = ProviderQualificationPolicy().failures_for(
        SecurityProfile.PRODUCTION_SERVER_READY,
        identity,
        evidence,
    )
    assert failures == ("provider identity or capability evidence has invalid type",)
    with pytest.raises(TypeError):
        ProviderCapabilities(implemented=True, signing=malformed)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("independent_rollback_domain", "false"),
        ("independent_admin_or_security_domain", "false"),
        ("authenticated", 1),
    ],
)
def test_policy_deeply_rejects_fabricated_checkpoint_capabilities(field_name, invalid_value):
    malformed = _fabricated_checkpoint_capabilities(field_name, invalid_value)
    security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_SERVER_READY, "server")
    identity = ProviderIdentity(ProviderRole.CHECKPOINT_AUTHORITY, security, "checkpoint")
    evidence = _fabricated_provider_capabilities(checkpoint=malformed)
    failures = ProviderQualificationPolicy().failures_for(
        SecurityProfile.PRODUCTION_SERVER_READY,
        identity,
        evidence,
    )
    assert failures == ("provider identity or capability evidence has invalid type",)
    with pytest.raises(TypeError):
        ProviderCapabilities(implemented=True, checkpoint=malformed)


def test_provider_identity_rejects_fabricated_nested_security_identity():
    malformed = object.__new__(SecurityProfileIdentity)
    object.__setattr__(malformed, "profile", "TEST")
    object.__setattr__(malformed, "trust_domain", "test")
    with pytest.raises(TypeError):
        ProviderIdentity(ProviderRole.ROOT_PROOF_SIGNING, malformed, "root")


@pytest.mark.parametrize("container_kind", ["list", "tuple_subclass"])
def test_credential_container_requires_exact_builtin_tuple(container_kind):
    class RotatingCredentialTuple(tuple):
        def __iter__(self):
            return super().__iter__()

    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    root = next(item for item in items if item.identity.role is ProviderRole.ROOT_PROOF_SIGNING)
    container = (
        [root.credentials[0]]
        if container_kind == "list"
        else RotatingCredentialTuple(root.credentials)
    )
    malformed_provider = RootSigningFake(
        root.identity,
        root.capabilities,
        container,  # type: ignore[arg-type]
    )
    items[items.index(root)] = malformed_provider
    result = qualify(items)
    assert malformed_provider._credential_reads == 1
    assert malformed_provider._active_reads == 0
    assert QualificationFailureCode.PROVIDER_EVIDENCE_UNAVAILABLE in {
        failure.code for failure in result.failures
    }


def test_plain_tuple_is_stored_as_exact_snapshot_credentials():
    provider = providers(SecurityProfile.PRODUCTION_LOCAL)[0]
    snapshot, failure = RootProofIssuerCompositionGate._capture_snapshot(provider)
    assert failure is None
    assert snapshot is not None
    assert type(snapshot.credentials) is tuple


@pytest.mark.parametrize(
    ("kind", "invalid_value"),
    [
        ("signing", "false"),
        ("signing", 1),
        ("checkpoint", "false"),
        ("checkpoint", 1),
    ],
)
def test_policy_and_gate_agree_on_fabricated_nested_capability_evidence(kind, invalid_value):
    if kind == "signing":
        nested = _fabricated_signing_capabilities("ed25519", invalid_value)
        role = ProviderRole.ROOT_PROOF_SIGNING
        evidence = _fabricated_provider_capabilities(signing=nested)
    else:
        nested = _fabricated_checkpoint_capabilities("authenticated", invalid_value)
        role = ProviderRole.CHECKPOINT_AUTHORITY
        evidence = _fabricated_provider_capabilities(checkpoint=nested)
    items = providers(SecurityProfile.PRODUCTION_SERVER_READY, signing=SERVER_SIGNING, checkpoint=SERVER_CHECKPOINT)
    item = next(item for item in items if item.identity.role is role)
    items[items.index(item)] = replace(item, capabilities=evidence)
    policy_failures = ProviderQualificationPolicy().failures_for(
        SecurityProfile.PRODUCTION_SERVER_READY,
        item.identity,
        evidence,
    )
    gate_result = qualify(items)
    assert policy_failures
    assert not gate_result.qualified
    assert QualificationFailureCode.PROVIDER_EVIDENCE_UNAVAILABLE in {
        failure.code for failure in gate_result.failures
    }


def test_policy_and_gate_reject_fabricated_nested_identity_and_provider_flags():
    valid_security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "prod")
    malformed_security = object.__new__(SecurityProfileIdentity)
    object.__setattr__(malformed_security, "profile", "PRODUCTION_LOCAL")
    object.__setattr__(malformed_security, "trust_domain", "prod")
    malformed_identity = object.__new__(ProviderIdentity)
    object.__setattr__(malformed_identity, "role", ProviderRole.DEPLOYMENT_TRUST_ROOT)
    object.__setattr__(malformed_identity, "security", malformed_security)
    object.__setattr__(malformed_identity, "provider_namespace", "trust-root")
    malformed_evidence = _fabricated_provider_capabilities()
    object.__setattr__(malformed_evidence, "implemented", "true")
    policy = ProviderQualificationPolicy()
    assert policy.failures_for(
        SecurityProfile.PRODUCTION_LOCAL,
        malformed_identity,
        ProviderCapabilities(implemented=True),
    )
    valid_identity = ProviderIdentity(
        ProviderRole.DEPLOYMENT_TRUST_ROOT,
        valid_security,
        "trust-root",
    )
    assert policy.failures_for(
        SecurityProfile.PRODUCTION_LOCAL,
        valid_identity,
        malformed_evidence,
    )
    provider = TrustRootFake(valid_identity, malformed_evidence)
    result = RootProofIssuerCompositionGate().qualify(valid_security, [provider])
    assert QualificationFailureCode.PROVIDER_EVIDENCE_UNAVAILABLE in {
        failure.code for failure in result.failures
    }


def test_allow_all_policy_cannot_be_injected_into_gate():
    class AllowAllPolicy:
        def failures_for(self, *args):
            return ()

    with pytest.raises(TypeError):
        RootProofIssuerCompositionGate(AllowAllPolicy())  # type: ignore[call-arg]


def test_gate_has_no_post_construction_policy_override_seam():
    class AllowAllPolicy:
        def failures_for(self, *args):
            return ()

    gate = RootProofIssuerCompositionGate()
    assert not hasattr(gate, "_policy")
    assert not hasattr(gate, "policy")
    with pytest.raises(AttributeError):
        gate._policy = AllowAllPolicy()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        gate.policy = AllowAllPolicy()  # type: ignore[attr-defined]
    items = providers(
        SecurityProfile.PRODUCTION_SERVER_READY,
        signing=LOCAL_SIGNING,
        checkpoint=LOCAL_CHECKPOINT,
    )
    assert not gate.qualify(items[0].identity.security, items).qualified


def test_policy_helper_instance_method_cannot_be_shadowed():
    policy = ProviderQualificationPolicy()
    with pytest.raises(AttributeError):
        policy.failures_for = lambda *args: ()  # type: ignore[method-assign]


@pytest.mark.parametrize(
    ("signing", "checkpoint"),
    [
        (LOCAL_SIGNING, SERVER_CHECKPOINT),
        (SERVER_SIGNING, LOCAL_CHECKPOINT),
        (LOCAL_SIGNING, LOCAL_CHECKPOINT),
    ],
)
def test_server_ready_requirements_cannot_be_bypassed(signing, checkpoint):
    items = providers(
        SecurityProfile.PRODUCTION_SERVER_READY,
        signing=signing,
        checkpoint=checkpoint,
    )
    assert not qualify(items).qualified


def test_public_policy_binds_target_profile_to_provider_identity():
    identity = ProviderIdentity(
        ProviderRole.ROOT_PROOF_SIGNING,
        SecurityProfileIdentity(SecurityProfile.PRODUCTION_SERVER_READY, "server"),
        "root-signing",
    )
    evidence = ProviderCapabilities(implemented=True, signing=LOCAL_SIGNING)
    failures = ProviderQualificationPolicy().failures_for(
        SecurityProfile.PRODUCTION_LOCAL,
        identity,
        evidence,
    )
    assert failures == ("target security profile does not match provider identity",)


def test_public_key_material_identity_accepts_only_canonical_raw_ed25519_bytes():
    canonical = bytes(range(32))
    assert public_key_material_identity(canonical).startswith("sha256:")


@pytest.mark.parametrize(
    "candidate",
    [
        b"",
        b"a" * 31,
        b"a" * 33,
        b"-----BEGIN PUBLIC KEY-----\nnot-canonical\n-----END PUBLIC KEY-----",
        b"\x30\x2a\x30\x05\x06\x03\x2b\x65\x70\x03\x21\x00" + b"a" * 32,
    ],
)
def test_public_key_material_identity_rejects_noncanonical_encodings(candidate):
    with pytest.raises(ValueError):
        public_key_material_identity(candidate)


def test_alternate_public_key_serialization_cannot_qualify():
    class WrappedHistoryKeyFake(HistorySigningFake):
        def public_key(self, credential_identity: str) -> bytes:
            object.__setattr__(self, "_public_key_reads", self._public_key_reads + 1)
            raw = _test_public_key_bytes(credential_identity)
            return b"DER-SPKI-WRAPPER:" + raw

    items = providers(SecurityProfile.PRODUCTION_LOCAL)
    history = next(
        item
        for item in items
        if item.identity.role is ProviderRole.HISTORY_ATTESTATION_SIGNING
    )
    wrapped = WrappedHistoryKeyFake(
        history.identity,
        history.capabilities,
        history.credentials,
    )
    items[items.index(history)] = wrapped
    result = qualify(items)
    assert not result.qualified
    assert wrapped._public_key_reads == 1
    assert QualificationFailureCode.PROVIDER_EVIDENCE_UNAVAILABLE in {
        failure.code for failure in result.failures
    }


def test_signing_ports_do_not_expose_private_key_material():
    assert "private_key" not in inspect.signature(RootSigningFake.public_key).parameters
    assert not hasattr(RootSigningFake, "private_key")
    assert not hasattr(HistorySigningFake, "private_key")
