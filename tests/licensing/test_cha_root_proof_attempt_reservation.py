"""TEST_ONLY provider output through the real guarded INITIAL_BINDING lineage.

The trusted adapter registration exists only in this test harness. It does not
claim a production provider exists. Upstream live requalification, canonical
binding derivation, SQLite transactions and capability verifiers remain real.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import sqlite3
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from bot_core import cha_attempt_store as attempt_store
from bot_core.entitlement_registry_contract import (
    AuthoritativeEntitlementState,
    EntitlementIdentity,
    EntitlementLifecycle,
    EntitlementProvenance,
    RegistryReadOutcome,
    RegistryReadResult,
    RegistrySubject,
    UnboundBinding,
)
from bot_core.licensing import (
    cha_account_reservation as binding_capability,
    cha_root_proof_attempt_reservation as capability,
)
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    CredentialSemanticRole,
    ProviderCapabilities,
    ProviderIdentity,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)
from deployment import windows_production_cha_account_reservation as installed
from tests.licensing import test_cha_account_reservation as upstream_tests

wide = upstream_tests.wide
challenge_harness = upstream_tests.challenge_harness
integration = upstream_tests.integration
package = upstream_tests.package
lppi = upstream_tests.lppi
active = upstream_tests.active
committed = upstream_tests.committed
cha = upstream_tests.cha
reserved = upstream_tests.reserved


def forbidden(*args, **kwargs):
    pytest.fail("reservation boundary attempted mint/sign/issuer/bind")


class TEST_ONLYPort:
    def __init__(self, role, trust, credentials=()):
        self.identity = ProviderIdentity(
            role,
            SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, trust),
            "TEST_ONLY-" + role.value,
        )
        self.capabilities = ProviderCapabilities(
            True, authoritative_reads=True, durable_state=True, compare_and_swap=True
        )
        self.credentials = credentials

    def credential_identities(self):
        return self.credentials


class TEST_ONLYEntitlementPort(TEST_ONLYPort):
    def authoritative_state(self, subject):
        assert subject == self.state.subject
        return RegistryReadResult(
            self.outcome, self.state if self.outcome is RegistryReadOutcome.FOUND else None
        )

    compare_and_swap_bind = forbidden
    state_at_revision = forbidden
    retained_history = forbidden


class TEST_ONLYRequesterPort(TEST_ONLYPort):
    def active_requester_credential(self, principal):
        assert principal == self.record.requester_principal_id
        return self.record

    historical_requester_credential = forbidden


class TEST_ONLYClaimantPort(TEST_ONLYPort):
    def resolve_claimant(self, principal):
        assert principal == self.record.provisioning_principal_id
        return self.record

    historical_claimant = forbidden


class TEST_ONLYIssuanceAuthority:
    """Only tests may simulate the as-yet unavailable trusted adapter output."""

    def __init__(self, trust):
        self.entitlement_registry = TEST_ONLYEntitlementPort(
            ProviderRole.ENTITLEMENT_REGISTRY, trust
        )
        self.requester_registry = TEST_ONLYRequesterPort(
            ProviderRole.REQUESTER_CREDENTIAL_REGISTRY, trust
        )
        self.claimant_registry = TEST_ONLYClaimantPort(
            ProviderRole.CLAIMANT_IDENTITY_REGISTRY, trust
        )
        self.requester_registry.record = capability._RequesterCredentialV1(
            "TEST_ONLY-requester-principal",
            "ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUESTER_V1",
            "TEST_ONLY-requester-key-Ż",
            1,
            "ACTIVE",
            7,
        )
        self.claimant_registry.record = capability._ClaimantIdentityV1(
            "TEST_ONLY-provisioning-principal", "TEST_ONLY-claimant-key-雪", 1, "ACTIVE", 9
        )
        for port, role, key in (
            (
                self.requester_registry,
                CredentialSemanticRole.ROOT_PROOF_REQUESTER,
                self.requester_registry.record.requester_key_id,
            ),
            (
                self.claimant_registry,
                CredentialSemanticRole.ROOT_PROOF_CLAIMANT,
                self.claimant_registry.record.claimant_key_id,
            ),
        ):
            port.credentials = (
                CredentialRoleIdentity(
                    role,
                    key,
                    port.identity.provider_namespace,
                    "TEST_ONLY-key-handle:" + key,
                    "TEST_ONLY-custody:" + key,
                    "sha256:"
                    + ("1" if role is CredentialSemanticRole.ROOT_PROOF_REQUESTER else "2") * 64,
                ),
            )
        self.subject = RegistrySubject("TEST_ONLY-lookup", "PRODUCTION", trust)
        self.entitlement_registry.state = AuthoritativeEntitlementState(
            self.subject,
            EntitlementIdentity(
                "ent_018f3e70-7b5c-7c21-8b9a-0123456789ab", 1, "PRODUCTION", trust, "CryptoHunter"
            ),
            EntitlementProvenance(
                self.claimant_registry.record.provisioning_principal_id,
                self.claimant_registry.record.claimant_key_id,
                self.claimant_registry.record.claimant_key_version,
                "TEST_ONLY-creation-authority",
                "TEST_ONLY-authenticated-creation",
                "3" * 64,
            ),
            EntitlementLifecycle.ACTIVE,
            UnboundBinding(),
            11,
            None,
        )
        self.entitlement_registry.outcome = RegistryReadOutcome.FOUND
        self.available = True
        self.requalifications = 0
        self.contexts = []

    def requalify(self):
        self.requalifications += 1
        if not self.available:
            raise capability.RootProofAttemptReservationError("TEST_ONLY-provider-unavailable")
        return None

    def resolve_initial_binding(self, context_raw):
        self.contexts.append(context_raw)
        return capability._ProviderResolution(
            self.subject,
            self.requester_registry.record.requester_principal_id,
            self.claimant_registry.record.provisioning_principal_id,
        )


@pytest.fixture
def provider(reserved, monkeypatch, tmp_path):
    _, binding, _, state = reserved
    authority = TEST_ONLYIssuanceAuthority(state["pdsa_trust_domain"])
    monkeypatch.setattr(capability, "_TRUSTED_PROVIDER_TYPES", (TEST_ONLYIssuanceAuthority,))
    monkeypatch.setattr(capability, "_issuance_authority_provider", lambda: authority)
    monkeypatch.setattr(
        capability, "_attempt_store_path", lambda: (tmp_path / "attempts.sqlite3").resolve()
    )
    return binding, authority


@pytest.fixture
def authorized(provider):
    binding, authority = provider
    return binding, authority, capability.resolve_root_proof_issuance_authorization(binding)


@pytest.fixture
def attempt(authorized):
    binding, authority, authorization = authorized
    reservation = capability.reserve_root_proof_issuance_attempt(binding, authorization)
    return binding, authority, authorization, reservation


def test_production_missing_provider_and_exact_binding_provenance(reserved, monkeypatch):
    _, binding, _, state = reserved
    with pytest.raises(capability.RootProofAttemptReservationError, match="MISSING_PRODUCTION"):
        capability.resolve_root_proof_issuance_authorization(binding)
    for raw in (
        state,
        canonical_json_bytes(state),
        binding.account_id,
        object.__new__(binding_capability.VerifiedAccountGenesisInitialBinding),
        {},
    ):
        with pytest.raises(binding_capability.AccountInitialBindingError):
            capability.resolve_root_proof_issuance_authorization(raw)
    authority = TEST_ONLYIssuanceAuthority(state["pdsa_trust_domain"])
    monkeypatch.setattr(capability, "_issuance_authority_provider", lambda: authority)
    with pytest.raises(
        capability.RootProofAttemptReservationError, match="TRUSTED_PROVIDER_PROVENANCE"
    ):
        capability.resolve_root_proof_issuance_authorization(binding)
    for copier in (copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            copier(binding)
    assert not capability._attempt_store_path().exists()


def test_provider_security_profile_is_independent_of_protocol_environment(provider):
    binding, authority = provider
    port = authority.requester_registry
    production_identity = port.identity
    for security in (
        SecurityProfileIdentity(SecurityProfile.TEST, production_identity.security.trust_domain),
        SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "TEST_ONLY-other-trust"),
    ):
        port.identity = replace(production_identity, security=security)
        with pytest.raises(capability.RootProofAttemptReservationError, match="UNQUALIFIED"):
            capability.resolve_root_proof_issuance_authorization(binding)
    port.identity = production_identity
    subject = authority.subject
    for changed in (
        replace(subject, environment="TEST"),
        replace(subject, trust_domain="TEST_ONLY-other-trust"),
    ):
        authority.subject = changed
        with pytest.raises(
            capability.RootProofAttemptReservationError, match="AUTHORIZATION_SCOPE"
        ):
            capability.resolve_root_proof_issuance_authorization(binding)
    authority.subject = subject
    authorization = capability.resolve_root_proof_issuance_authorization(binding).authorization
    assert authorization.environment == "PRODUCTION"
    assert port.identity.security.profile is SecurityProfile.PRODUCTION_LOCAL
    assert not capability._attempt_store_path().exists()


def test_exact_deterministic_relation_and_separate_initial_binding_digest(reserved):
    _, binding, _, state = reserved
    raw = canonical_json_bytes(state)
    relation = {
        "schema_version": "InitialBindingReservationRelationV1",
        "environment": "PRODUCTION",
        "pdsa_trust_domain": state["pdsa_trust_domain"],
        "logical_operation_id": state["logical_operation_id"],
        "account_id": state["account_id"],
        "reservation_relation": "EXACT_OPERATION_ACCOUNT",
        "canonical_genesis_request_fingerprint_sha256": state["canonical_request_sha256"],
        "initial_binding_sha256": state["initial_binding_sha256"],
        "retained_initial_binding_sha256": hashlib.sha256(raw).hexdigest(),
    }
    identity = (
        "ibr_"
        + hashlib.sha256(
            b"CRYPTOHUNTER_STAGE9_INITIAL_BINDING_RESERVATION_RELATION_V1\0"
            + canonical_json_bytes(relation)
        ).hexdigest()
    )
    reference = {
        **relation,
        "schema_version": "RootProofInitialBindingReferenceV1",
        "reservation_identity": identity,
    }
    digest = hashlib.sha256(
        b"CRYPTOHUNTER_STAGE9_ROOT_PROOF_INITIAL_BINDING_REFERENCE_V1\0"
        + canonical_json_bytes(reference)
    ).hexdigest()
    expected = {
        **reference,
        "product_scope": "CryptoHunter",
        "initial_binding_reference": "initial-binding-v1:" + digest,
        "initial_binding_digest_sha256": digest,
    }
    assert capability._binding_context(binding) == (raw, expected)
    assert capability._binding_context(binding) == (raw, expected)
    assert digest != state["initial_binding_sha256"]
    assert not identity.startswith("res_")
    for field in (
        "logical_operation_id",
        "account_id",
        "canonical_request_sha256",
        "initial_binding_sha256",
        "pdsa_trust_domain",
        "assigned_at_utc",
    ):
        # Pure identity computation accepts transport solely to demonstrate binding
        # sensitivity; these changed records never pass the capability verifier.
        changed = {**state, field: "TEST_ONLY-changed-" + state[field]}
        context = capability._context_from_retained_binding(canonical_json_bytes(changed))
        assert context["reservation_identity"] != identity
        assert context["initial_binding_digest_sha256"] != digest


def test_exact_authority_owned_reservation_and_jcs_payload(attempt):
    binding, authority, authorization, reservation = attempt
    dto = authorization.authorization
    assert (
        dto.account_id == binding.account_id
        and dto.logical_operation_id == binding.logical_operation_id
    )
    assert dto.environment == "PRODUCTION"
    assert reservation.state is attempt_store.AttemptState.RESERVED_AWAITING_SIGNATURES
    assert reservation.fence == 1 and reservation.reservation_identity == dto.reservation_identity
    identity = uuid.UUID(reservation.issuance_attempt_id.removeprefix("rpa_"))
    assert reservation.issuance_attempt_id == "rpa_" + str(identity)
    assert identity.version == 7 and identity.variant == uuid.RFC_4122
    with sqlite3.connect(capability._attempt_store_path()) as database:
        key, payload = database.execute(
            "SELECT idempotency_key,authorization_json FROM reservations"
        ).fetchone()
        assert payload == canonical_json_bytes(asdict(dto))
        assert (
            key
            == hashlib.sha256(
                b"CRYPTOHUNTER_CHA_ATTEMPT_RESERVATION_IDEMPOTENCY_V1\0" + payload
            ).hexdigest()
        )
        assert "雪".encode() in payload and "Ż".encode() in payload
        assert database.execute("SELECT count(*) FROM current_attempts").fetchone() == (1,)
        assert database.execute("SELECT count(*) FROM immutable_attempts").fetchone() == (0,)
        assert database.execute(
            "SELECT state,digest_status,digest FROM current_attempts"
        ).fetchone() == (
            "RESERVED_AWAITING_SIGNATURES",
            "NOT_YET_DEFINED",
            None,
        )
    assert authority.requalifications > 1
    assert all(parse_canonical(raw)["account_id"] == dto.account_id for raw in authority.contexts)


def test_opaque_capabilities_and_raw_authorization_never_grant_authority(attempt):
    binding, _, authorization, reservation = attempt
    dto = authorization.authorization
    with sqlite3.connect(capability._attempt_store_path()) as database:
        raw_row = database.execute("SELECT * FROM reservations").fetchone()
    for cls, value, consume in (
        (
            capability.VerifiedRootProofIssuanceAuthorization,
            authorization,
            capability.require_verified_root_proof_issuance_authorization,
        ),
        (
            capability.VerifiedRootProofIssuanceAttemptReservation,
            reservation,
            capability.require_verified_root_proof_issuance_attempt_reservation,
        ),
    ):
        for raw in (
            {},
            asdict(dto),
            dto,
            raw_row,
            canonical_json_bytes(asdict(dto)),
            reservation.issuance_attempt_id,
            object.__new__(cls),
        ):
            with pytest.raises(capability.RootProofAttemptReservationError):
                consume(raw)
        with pytest.raises(TypeError):
            cls()
        with pytest.raises(TypeError):
            type("TEST_ONLYForged", (cls,), {})
        for copier in (copy.copy, copy.deepcopy):
            with pytest.raises(TypeError):
                copier(value)
        with pytest.raises(TypeError):
            value.forged = True
        with pytest.raises(AttributeError):
            object.__setattr__(value, "forged", True)
    for function in (
        capability.reserve_root_proof_issuance_attempt,
        capability.load_root_proof_issuance_attempt,
    ):
        with pytest.raises(capability.RootProofAttemptReservationError, match="VERIFIED_ISSUANCE"):
            function(binding, dto)
        for keyword in (
            "issuance_attempt_id",
            "assigned_at_utc",
            "bootstrap_entitlement_id",
            "requester_key_id",
            "claimant_key_id",
            "path",
        ):
            with pytest.raises(TypeError):
                function(binding, authorization, **{keyword: "TEST_ONLY-caller"})
    # Returned DTOs are transport copies, never aliases of retained authority.
    object.__setattr__(dto, "requester_key_id", "TEST_ONLY-caller-tampering")
    assert authorization.authorization.requester_key_id != dto.requester_key_id


def test_loader_empty_never_mints(authorized, monkeypatch):
    binding, _, authorization = authorized
    monkeypatch.setattr(attempt_store, "_new_rpa_id", forbidden)
    with pytest.raises(attempt_store.AttemptNotFoundError):
        capability.load_root_proof_issuance_attempt(binding, authorization)
    with sqlite3.connect(capability._attempt_store_path()) as database:
        assert database.execute("SELECT count(*) FROM reservations").fetchone() == (0,)
        assert database.execute("SELECT count(*) FROM current_attempts").fetchone() == (0,)


def test_retry_lost_response_and_restart_retain_same_attempt(authorized, reserved, monkeypatch):
    binding, authority, authorization = authorized
    upstream, _, _, state = reserved
    issue = capability._issue_reservation

    def lost_response(*args):
        raise RuntimeError("TEST_ONLY-lost-capability-response")

    monkeypatch.setattr(capability, "_issue_reservation", lost_response)
    with pytest.raises(RuntimeError, match="lost-capability-response"):
        capability.reserve_root_proof_issuance_attempt(binding, authorization)
    with sqlite3.connect(capability._attempt_store_path()) as database:
        retained = database.execute("SELECT attempt_id FROM reservations").fetchone()[0]
    monkeypatch.setattr(capability, "_issue_reservation", issue)
    monkeypatch.setattr(attempt_store, "_new_rpa_id", forbidden)
    monkeypatch.setattr(installed, "mint_uuid7", forbidden)
    monkeypatch.setattr(installed, "_utc_now", forbidden)
    assert (
        capability.reserve_root_proof_issuance_attempt(binding, authorization).issuance_attempt_id
        == retained
    )
    assert (
        capability.load_root_proof_issuance_attempt(binding, authorization).issuance_attempt_id
        == retained
    )
    capability._AUTHORIZATIONS.clear()
    capability._RESERVATIONS.clear()
    binding_capability._ISSUED.clear()
    with pytest.raises(capability.RootProofAttemptReservationError):
        capability.require_verified_root_proof_issuance_authorization(authorization)
    reconstructed_binding = installed.load_installed_account_initial_binding(upstream)
    assert reconstructed_binding.account_id == state["account_id"]
    restarted_provider = TEST_ONLYIssuanceAuthority(state["pdsa_trust_domain"])
    monkeypatch.setattr(capability, "_issuance_authority_provider", lambda: restarted_provider)
    reconstructed_authorization = capability.resolve_root_proof_issuance_authorization(
        reconstructed_binding
    )
    loaded = capability.load_root_proof_issuance_attempt(
        reconstructed_binding, reconstructed_authorization
    )
    assert loaded.issuance_attempt_id == retained
    assert restarted_provider is not authority
    assert loaded.state is attempt_store.AttemptState.RESERVED_AWAITING_SIGNATURES
    # A restarted provider may retain all IDs/key versions while its live
    # registry evidence changes. That new context cannot adopt the old winner.
    restarted_provider.requester_registry.record = replace(
        restarted_provider.requester_registry.record, registry_revision=8
    )
    changed_authorization = capability.resolve_root_proof_issuance_authorization(
        reconstructed_binding
    )
    with pytest.raises(capability.RootProofAttemptReservationError, match="EXACT_CURRENT"):
        capability.load_root_proof_issuance_attempt(reconstructed_binding, changed_authorization)
    with pytest.raises(attempt_store.AttemptConflictError):
        capability.reserve_root_proof_issuance_attempt(reconstructed_binding, changed_authorization)
    with sqlite3.connect(capability._attempt_store_path()) as database:
        assert database.execute("SELECT attempt_id FROM reservations").fetchall() == [(retained,)]


def test_live_provider_changes_invalidate_authorization_and_reservation(attempt):
    _, authority, authorization, reservation = attempt
    port = authority.requester_registry
    mutations = (
        (authority, "available", False),
        (authority, "requalify", lambda: False),
        (port, "capabilities", replace(port.capabilities, authoritative_reads=False)),
        (port, "record", replace(port.record, registry_revision=port.record.registry_revision + 1)),
        (authority.entitlement_registry, "outcome", RegistryReadOutcome.UNAVAILABLE),
        (
            authority.entitlement_registry,
            "state",
            replace(authority.entitlement_registry.state, lifecycle=EntitlementLifecycle.REVOKED),
        ),
        (
            authority.claimant_registry,
            "record",
            replace(authority.claimant_registry.record, claimant_key_version=2),
        ),
    )
    for target, field, changed in mutations:
        original = getattr(target, field)
        setattr(target, field, changed)
        try:
            for consume in (
                lambda: authorization.authorization,
                lambda: capability.require_verified_root_proof_issuance_attempt_reservation(
                    reservation
                ),
            ):
                with pytest.raises((ValueError, RuntimeError)):
                    consume()
        finally:
            setattr(target, field, original)
    assert (
        capability.require_verified_root_proof_issuance_attempt_reservation(reservation)
        is reservation
    )


def test_wrong_role_namespace_and_key_material_alias_fail_closed(provider):
    binding, authority = provider
    requester = authority.requester_registry
    claimant = authority.claimant_registry
    original = requester.credentials
    wrong_credentials = (
        (),
        (replace(original[0], semantic_role=CredentialSemanticRole.ROOT_PROOF_CLAIMANT),),
        (replace(original[0], provider_namespace="TEST_ONLY-other-provider"),),
        (
            replace(
                original[0], key_material_identity=claimant.credentials[0].key_material_identity
            ),
        ),
    )
    for credentials in wrong_credentials:
        requester.credentials = credentials
        with pytest.raises(capability.RootProofAttemptReservationError):
            capability.resolve_root_proof_issuance_authorization(binding)
    requester.credentials = original
    assert not capability._attempt_store_path().exists()


def test_same_operation_changed_provider_tuple_conflicts_without_replacement(attempt):
    binding, authority, authorization, reservation = attempt
    retained = reservation.issuance_attempt_id
    authority.requester_registry.record = replace(
        authority.requester_registry.record, requester_key_version=2, registry_revision=8
    )
    with pytest.raises(capability.RootProofAttemptReservationError, match="EVIDENCE_CHANGED"):
        capability.reserve_root_proof_issuance_attempt(binding, authorization)
    current_authorization = capability.resolve_root_proof_issuance_authorization(binding)
    with pytest.raises(attempt_store.AttemptConflictError, match="incompatible"):
        capability.reserve_root_proof_issuance_attempt(binding, current_authorization)
    with sqlite3.connect(capability._attempt_store_path()) as database:
        assert database.execute("SELECT attempt_id,fence FROM current_attempts").fetchall() == [
            (retained, 1)
        ]
        assert database.execute("SELECT count(*) FROM reservations").fetchone() == (1,)


def test_changed_retained_binding_invalidates_every_consequential_use(attempt, reserved):
    binding, _, authorization, reservation = attempt
    _, _, path, state = reserved
    original = path.read_bytes()
    for field, changed in (
        ("initial_binding_sha256", "0" * 64),
        ("canonical_request_sha256", "0" * 64),
        ("account_id", "acct_018f3e70-7b5c-7c21-8b9a-0123456789ab"),
    ):
        path.write_bytes(canonical_json_bytes({**state, field: changed}))
        try:
            for consume in (
                lambda: authorization.authorization,
                lambda: reservation.issuance_attempt_id,
                lambda: capability.load_root_proof_issuance_attempt(binding, authorization),
            ):
                with pytest.raises((ValueError, RuntimeError)):
                    consume()
        finally:
            path.write_bytes(original)


def test_current_fence_and_state_change_invalidates_retained_capability(attempt):
    binding, _, authorization, reservation = attempt
    path = capability._attempt_store_path()
    for sql, parameters in (
        ("UPDATE current_attempts SET fence=?", (2,)),
        ("UPDATE current_attempts SET state=?", ("SIGNED_IMMUTABLE_DURABLE_NOT_SENT",)),
    ):
        with sqlite3.connect(path) as database:
            database.execute(sql, parameters)
        try:
            for consume in (
                lambda: reservation.fence,
                lambda: capability.load_root_proof_issuance_attempt(binding, authorization),
                lambda: capability.reserve_root_proof_issuance_attempt(binding, authorization),
            ):
                with pytest.raises(
                    (attempt_store.AttemptStoreError, capability.RootProofAttemptReservationError)
                ):
                    consume()
        finally:
            with sqlite3.connect(path) as database:
                database.execute(
                    "UPDATE current_attempts SET fence=?,state=?",
                    (1, "RESERVED_AWAITING_SIGNATURES"),
                )


def test_provider_replacement_and_store_owner_change_fail_closed(attempt, monkeypatch, tmp_path):
    _, authority, authorization, reservation = attempt
    original_path = capability._attempt_store_path()
    monkeypatch.setattr(
        capability, "_attempt_store_path", lambda: (tmp_path / "other.sqlite3").resolve()
    )
    with pytest.raises(capability.RootProofAttemptReservationError, match="OWNER_CHANGED"):
        capability.require_verified_root_proof_issuance_attempt_reservation(reservation)
    monkeypatch.setattr(capability, "_attempt_store_path", lambda: original_path)
    replacement = TEST_ONLYIssuanceAuthority(authority.subject.trust_domain)
    monkeypatch.setattr(capability, "_issuance_authority_provider", lambda: replacement)
    with pytest.raises(capability.RootProofAttemptReservationError, match="PROVIDER_CHANGED"):
        capability.require_verified_root_proof_issuance_authorization(authorization)
    with pytest.raises(capability.RootProofAttemptReservationError, match="PROVIDER_CHANGED"):
        capability.require_verified_root_proof_issuance_attempt_reservation(reservation)


def test_public_api_has_no_caller_selectable_identity_time_provider_or_path():
    assert list(
        inspect.signature(capability.resolve_root_proof_issuance_authorization).parameters
    ) == ["binding"]
    for function in (
        capability.reserve_root_proof_issuance_attempt,
        capability.load_root_proof_issuance_attempt,
    ):
        assert list(inspect.signature(function).parameters) == ["binding", "authorization"]


@pytest.mark.parametrize("invalid", [None, True, 0, -1, 1.5, float("nan"), float("inf"), 1 << 53])
def test_provider_record_version_and_revision_fail_closed(invalid):
    authority = TEST_ONLYIssuanceAuthority("TEST_ONLY-trust")
    for record, field in (
        (authority.requester_registry.record, "requester_key_version"),
        (authority.requester_registry.record, "registry_revision"),
        (authority.claimant_registry.record, "claimant_key_version"),
        (authority.claimant_registry.record, "registry_revision"),
    ):
        forged = copy.copy(record)
        object.__setattr__(forged, field, invalid)
        with pytest.raises(ValueError):
            capability._exact_record(forged, type(record))


def test_raw_inactive_and_wrong_role_provider_records_fail_closed():
    authority = TEST_ONLYIssuanceAuthority("TEST_ONLY-trust")
    for record in (authority.requester_registry.record, authority.claimant_registry.record):
        for raw in ({}, asdict(record), "TEST_ONLY-identity"):
            with pytest.raises(TypeError):
                capability._exact_record(raw, type(record))
        forged = copy.copy(record)
        object.__setattr__(forged, "lifecycle", "REVOKED")
        with pytest.raises(ValueError):
            capability._exact_record(forged, type(record))
    forged_requester = copy.copy(authority.requester_registry.record)
    object.__setattr__(forged_requester, "requester_credential_role", "TEST_ONLY-wrong-role")
    with pytest.raises(ValueError):
        capability._exact_record(forged_requester, capability._RequesterCredentialV1)


def test_stage9_scope_ast_stops_at_reserved_without_signing_or_issuer():
    module_path = Path(inspect.getfile(capability))
    tree = ast.parse(module_path.read_text())
    forbidden_calls = {
        "finalize_attempt",
        "replace_after_authoritative_unbound",
        "record_recovery_resolution",
        "compare_and_swap_bind",
        "sign",
        "sign_root_proof",
        "sign_history_head",
        "sign_finalization",
        "sign_freshness_proposal",
        "send",
        "request",
        "urlopen",
        "RootProofAdmissionEvidence",
    }
    observed_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert not forbidden_calls & observed_calls
    forbidden_imports = (
        "requests",
        "httpx",
        "urllib",
        "cryptography",
        "stage10",
        "membership",
        "freshness",
        "device",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports = [node.module or ""]
        else:
            continue
        assert all(not any(part in name.lower() for part in forbidden_imports) for name in imports)
    states = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "AttemptState"
    }
    assert states == {"RESERVED_AWAITING_SIGNATURES"}
