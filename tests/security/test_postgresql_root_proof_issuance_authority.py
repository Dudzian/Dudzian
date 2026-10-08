"""Genuine PostgreSQL authority through the existing guarded Stage 9 boundary."""

from __future__ import annotations

import ast
import inspect
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

import bot_core.postgresql_preaccount_credentials as credentials
import bot_core.postgresql_root_proof_issuance_authority as composition
from bot_core.cha_attempt_store import AttemptState
from bot_core.entitlement_registry_contract import (
    AdminOutcome,
    EntitlementIdentity,
    EntitlementLifecycle,
    EntitlementProvenance,
    ProvisionEntitlementRequest,
    RegistryReadOutcome,
    RegistrySubject,
    RevokeEntitlementRequest,
    SupersedeEntitlementRequest,
    UnboundBinding,
    admin_predecessor_for,
)
from bot_core.licensing import cha_root_proof_attempt_reservation as boundary
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.postgresql_entitlement_registry import (
    PostgreSQLConnectionConfig,
    PostgreSQLEntitlementProvisioningAdminProvider,
    PostgreSQLEntitlementRegistryProvider,
    PostgreSQLRegistryProvisioning,
    provision_postgresql_entitlement_registry,
)
from tests.licensing import test_cha_account_reservation as upstream
from tests.security import test_postgresql_preaccount_credentials_integration as registry_tests

wide = upstream.wide
challenge_harness = upstream.challenge_harness
integration = upstream.integration
package = upstream.package
lppi = upstream.lppi
active = upstream.active
committed = upstream.committed
cha = upstream.cha
reserved = upstream.reserved


@pytest.mark.parametrize("missing", composition._CONFIG_ENVIRONMENT)
def test_runtime_missing_configuration_fails_before_composition(monkeypatch, missing):
    for name in composition._CONFIG_ENVIRONMENT:
        monkeypatch.setenv(name, "deployment-selector")
    monkeypatch.delenv(missing)
    with pytest.raises(
        composition.ProductionLocalIssuanceAuthorityError, match="MISSING_PRODUCTION"
    ):
        composition._configured_root_proof_issuance_authority()


def test_aggregate_rejects_fake_and_subclass_provider_types_before_database_access():
    subject = RegistrySubject("reviewed-lookup", "PRODUCTION", "td_authority")
    with pytest.raises(composition.ProductionLocalIssuanceAuthorityError, match="EXACT_GENUINE"):
        composition.PostgreSQLRootProofIssuanceAuthority(object(), object(), object(), subject)

    class RequesterSubclass(credentials.PostgreSQLRequesterCredentialRegistryProvider):
        pass

    with pytest.raises(composition.ProductionLocalIssuanceAuthorityError, match="EXACT_GENUINE"):
        composition.PostgreSQLRootProofIssuanceAuthority(
            object.__new__(PostgreSQLEntitlementRegistryProvider),
            object.__new__(RequesterSubclass),
            object.__new__(credentials.PostgreSQLClaimantIdentityRegistryProvider),
            subject,
        )
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("UnreviewedAggregate", (composition.PostgreSQLRootProofIssuanceAuthority,), {})


def test_internal_configuration_redacts_connection_secrets():
    secret = "host=database password=test-secret-value"
    config = composition._RuntimeConfiguration(
        PostgreSQLConnectionConfig(secret),
        PostgreSQLConnectionConfig(secret),
        PostgreSQLConnectionConfig(secret),
        "td_authority",
        "entitlements",
        "lookup",
    )
    assert "test-secret-value" not in repr(config)


@pytest.mark.parametrize("subject", [object(), RegistrySubject("lookup", "TEST", "td_authority")])
def test_exact_production_subject_required_before_database_access(subject):
    with pytest.raises(composition.ProductionLocalIssuanceAuthorityError):
        composition.PostgreSQLRootProofIssuanceAuthority(
            object.__new__(PostgreSQLEntitlementRegistryProvider),
            object.__new__(credentials.PostgreSQLRequesterCredentialRegistryProvider),
            object.__new__(credentials.PostgreSQLClaimantIdentityRegistryProvider),
            subject,
        )


def test_composition_stops_at_read_only_public_authority():
    tree = ast.parse(inspect.getsource(composition))
    forbidden = {
        "sign_root_proof",
        "sign_history_head",
        "finalize_attempt",
        "AttemptIdentity",
        "compare_and_swap_bind",
        "sign_freshness_proposal",
        "sign_finalization",
        "provision_credential",
        "rotate_credential",
        "transition_lifecycle",
    }
    assert not {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} & forbidden
    assert boundary._TRUSTED_PROVIDER_TYPES == (composition.PostgreSQLRootProofIssuanceAuthority,)


def _connection(role):
    return PostgreSQLConnectionConfig(make_conninfo(registry_tests.BASE_DSN, user=role))


def _assert_operation_invalidated(binding, authorization):
    for check, args in (
        (boundary.require_verified_root_proof_issuance_authorization, (authorization,)),
        (boundary.load_root_proof_issuance_attempt, (binding, authorization)),
        (boundary.reserve_root_proof_issuance_attempt, (binding, authorization)),
    ):
        with pytest.raises(
            (
                boundary.RootProofAttemptReservationError,
                credentials.CredentialResolutionError,
                composition.ProductionLocalIssuanceAuthorityError,
            )
        ):
            check(*args)


@contextmanager
def _installed_authorities(
    monkeypatch, trust_domain, provisioning_principal="deployment-provisioning-principal"
):
    """Install exact reviewed production names into the temporary real cluster."""

    requester = credentials.PostgreSQLCredentialRegistryProvisioning(
        credentials.REQUESTER_SCHEMA,
        credentials.REQUESTER_OWNER_ROLE,
        credentials.REQUESTER_RUNTIME_ROLE,
        credentials.REQUESTER_ADMIN_ROLE,
        "PRODUCTION",
        trust_domain,
    )
    claimant = credentials.PostgreSQLCredentialRegistryProvisioning(
        credentials.CLAIMANT_SCHEMA,
        credentials.CLAIMANT_OWNER_ROLE,
        credentials.CLAIMANT_RUNTIME_ROLE,
        credentials.CLAIMANT_ADMIN_ROLE,
        "PRODUCTION",
        trust_domain,
    )
    paired = credentials.PostgreSQLPreaccountRegistryProvisioning(requester, claimant)
    suffix = uuid.uuid4().hex[:10]
    entitlement = PostgreSQLRegistryProvisioning(
        f"rpe_{suffix}",
        f"rpeo_{suffix}",
        f"rper_{suffix}",
        f"rpea_{suffix}",
        trust_domain,
    )
    composition._compose_runtime_authority.cache_clear()
    try:
        credentials.provision_postgresql_preaccount_registries(
            PostgreSQLConnectionConfig(registry_tests.BASE_DSN),
            paired,
        )
        provision_postgresql_entitlement_registry(
            PostgreSQLConnectionConfig(registry_tests.BASE_DSN),
            entitlement,
        )
        pair = registry_tests._providers(paired)
        pair.requester_admin.provision_credential(
            principal_id=composition.REQUESTER_PRINCIPAL,
            credential_id="requester-credential",
            key_id="requester-key",
            key_version=1,
            public_key=b"R" * 32,
        )
        pair.claimant_admin.provision_credential(
            principal_id=provisioning_principal,
            credential_id="claimant-credential",
            key_id="claimant-key",
            key_version=1,
            public_key=b"C" * 32,
        )
        subject = RegistrySubject("deployment-entitlement-lookup", "PRODUCTION", trust_domain)
        admin = PostgreSQLEntitlementProvisioningAdminProvider(
            _connection(entitlement.admin_role),
            schema=entitlement.schema,
            environment="PRODUCTION",
            trust_domain=trust_domain,
        )
        result = admin.provision_entitlement(
            ProvisionEntitlementRequest(
                subject,
                EntitlementIdentity(
                    "ent_018f3e70-7b5a-7c21-8b9a-0123456789ab",
                    1,
                    "PRODUCTION",
                    trust_domain,
                    "CryptoHunter",
                ),
                EntitlementProvenance(
                    provisioning_principal,
                    "claimant-key",
                    1,
                    "deployment-security-authority",
                    "provisioning:exact:1",
                    "a" * 64,
                ),
            )
        )
        assert result.outcome is AdminOutcome.COMMITTED
        settings = (
            _connection(entitlement.runtime_role).dsn,
            _connection(requester.runtime_role).dsn,
            _connection(claimant.runtime_role).dsn,
            trust_domain,
            entitlement.schema,
            subject.lookup_handle,
        )
        for name, value in zip(composition._CONFIG_ENVIRONMENT, settings, strict=True):
            monkeypatch.setenv(name, value)
        yield pair, subject, admin
    finally:
        composition._compose_runtime_authority.cache_clear()
        with psycopg.connect(registry_tests.BASE_DSN, autocommit=True) as conn:
            for authority in (entitlement, requester, claimant):
                conn.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(authority.schema)
                    )
                )
            for authority in (entitlement, requester, claimant):
                for role in (
                    authority.runtime_role,
                    authority.admin_role,
                    authority.schema_owner_role,
                ):
                    conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


@pytest.mark.external_postgresql
def test_genuine_initial_binding_authorization_and_durable_exact_reservation(
    reserved,
    monkeypatch,
    tmp_path: Path,
):
    _, binding, _, retained = reserved
    with _installed_authorities(monkeypatch, retained["pdsa_trust_domain"]) as (pair, subject, _):
        monkeypatch.setattr(boundary, "_attempt_store_path", lambda: tmp_path / "attempts.sqlite3")
        authority = boundary._issuance_authority_provider()
        assert type(authority) is composition.PostgreSQLRootProofIssuanceAuthority
        assert boundary._issuance_authority_provider() is authority
        authorization = boundary.resolve_root_proof_issuance_authorization(binding)
        original_authorization = authorization.authorization
        assert original_authorization.requester_principal_id == "CryptoHunterAccountAuthority"
        assert (
            original_authorization.provisioning_principal_id == "deployment-provisioning-principal"
        )
        reservation = boundary.reserve_root_proof_issuance_attempt(binding, authorization)
        attempt_id = reservation.issuance_attempt_id
        evidence_hash = original_authorization.authorization_evidence_sha256
        assert reservation.state is AttemptState.RESERVED_AWAITING_SIGNATURES
        assert (
            boundary.reserve_root_proof_issuance_attempt(binding, authorization).issuance_attempt_id
            == attempt_id
        )
        assert (
            boundary.load_root_proof_issuance_attempt(binding, authorization).issuance_attempt_id
            == attempt_id
        )
        assert (
            type(authority.entitlement_registry.authoritative_state(subject).state.binding)
            is UnboundBinding
        )
        assert (
            authority.entitlement_registry.authoritative_state(subject).state.lifecycle
            is EntitlementLifecycle.ACTIVE
        )
        # Reconnect each port and read the retained credential/key history.
        restored = registry_tests._providers(pair.setup)
        assert restored.requester.public_key("requester-key") == b"R" * 32
        assert restored.claimant.public_key("claimant-key") == b"C" * 32
        original_record = pair.requester.active_requester_credential(
            composition.REQUESTER_PRINCIPAL
        )
        requester_evidence = pair.requester.credential_identities()
        claimant_record = pair.claimant.resolve_claimant("deployment-provisioning-principal")
        claimant_evidence = pair.claimant.credential_identities()
        for change in (
            {"requester_key_version": 2},
            {"registry_revision": 2},
        ):
            with pytest.raises(
                composition.ProductionLocalIssuanceAuthorityError, match="EVIDENCE_CHANGED"
            ):
                authority.validate_resolved_credentials(
                    replace(original_record, **change),
                    claimant_record,
                    requester_evidence,
                    claimant_evidence,
                )

        # Every unrelated principal mutation remains globally qualified while
        # the exact operation A evidence and its retained attempt stay stable.
        def assert_operation_a_unchanged():
            assert (
                boundary.require_verified_root_proof_issuance_authorization(authorization)
                is authorization
            )
            fresh_authorization = boundary.resolve_root_proof_issuance_authorization(binding)
            fresh_record = fresh_authorization.authorization
            assert fresh_record == original_authorization
            assert fresh_record.authorization_evidence_sha256 == evidence_hash
            for candidate in (
                boundary.load_root_proof_issuance_attempt(binding, authorization),
                boundary.reserve_root_proof_issuance_attempt(binding, authorization),
            ):
                assert candidate.issuance_attempt_id == attempt_id
                assert candidate.state is AttemptState.RESERVED_AWAITING_SIGNATURES
            current = authority.entitlement_registry.authoritative_state(subject).state
            assert current.authoritative_state_revision == 1
            assert current.lifecycle is EntitlementLifecycle.ACTIVE
            assert type(current.binding) is UnboundBinding

        claimant_b = "unrelated-deployment-principal-B"
        first_b = pair.claimant_admin.provision_credential(
            principal_id=claimant_b,
            credential_id="claimant-B-credential",
            key_id="claimant-B-key",
            key_version=1,
            public_key=b"B" * 32,
        )
        assert_operation_a_unchanged()
        rotated_b = pair.claimant_admin.rotate_credential(
            principal_id=claimant_b,
            credential_id="claimant-B-credential-next",
            key_id="claimant-B-key-next",
            key_version=2,
            public_key=b"D" * 32,
            expected_revision=first_b.registry_revision,
        )
        assert_operation_a_unchanged()
        pair.claimant_admin.transition_lifecycle(
            principal_id=claimant_b,
            lifecycle=credentials.CredentialLifecycle.VERIFY_ONLY,
            expected_revision=rotated_b.registry_revision,
        )
        assert_operation_a_unchanged()
        pair.claimant_admin.transition_lifecycle(
            principal_id=claimant_b,
            credential_id=first_b.credential_id,
            lifecycle=credentials.CredentialLifecycle.REVOKED,
            expected_revision=2,
        )
        assert_operation_a_unchanged()
        pair.claimant_admin.provision_credential(
            principal_id="unrelated-deployment-principal-C",
            credential_id="claimant-C-credential",
            key_id="claimant-C-key",
            key_version=1,
            public_key=b"E" * 32,
        )
        assert_operation_a_unchanged()
        # Canonical operation evidence contains only A's exact credential
        # identities; the complete registry population stays outside its hash.
        scoped = boundary.parse_canonical(
            boundary._authorization_snapshot(authorization).evidence_raw
        )
        assert scoped["scope"] == "EXACT_OPERATION_SCOPED"
        assert scoped["requester_credential"]["identity"]["credential_identity"] == "requester-key"
        assert scoped["claimant_credential"]["identity"]["credential_identity"] == "claimant-key"
        assert all("credentials" not in port for port in scoped["providers"].values())
        requester_raw = pair.requester.public_key("requester-key")
        with pytest.raises(credentials.CredentialConflictError):
            pair.claimant_admin.provision_credential(
                principal_id="unrelated-alias-principal",
                credential_id="unrelated-alias-credential",
                key_id="unrelated-alias-key",
                key_version=1,
                public_key=requester_raw,
            )
        assert_operation_a_unchanged()
        # A privileged canonical rewrite of unrelated B still cannot bypass
        # the global requester/claimant material non-alias invariant.
        with psycopg.connect(registry_tests.BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL(
                    "UPDATE {}.credentials SET public_key=%s,key_material_identity=%s WHERE credential_id=%s"
                ).format(sql.Identifier(pair.setup.claimant.schema)),
                (
                    requester_raw,
                    credentials.public_key_material_identity(requester_raw),
                    rotated_b.credential_id,
                ),
            )
        with pytest.raises(
            composition.ProductionLocalIssuanceAuthorityError, match="FORBIDDEN_CREDENTIAL_ALIAS"
        ):
            authority.requalify()
        with pytest.raises(
            (
                boundary.RootProofAttemptReservationError,
                composition.ProductionLocalIssuanceAuthorityError,
            )
        ):
            boundary.require_verified_root_proof_issuance_authorization(authorization)
        with psycopg.connect(registry_tests.BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL(
                    "UPDATE {}.credentials SET public_key=%s,key_material_identity=%s WHERE credential_id=%s"
                ).format(sql.Identifier(pair.setup.claimant.schema)),
                (
                    b"D" * 32,
                    credentials.public_key_material_identity(b"D" * 32),
                    rotated_b.credential_id,
                ),
            )
        assert_operation_a_unchanged()
        pair.claimant_admin.transition_lifecycle(
            principal_id="deployment-provisioning-principal",
            lifecycle=credentials.CredentialLifecycle.VERIFY_ONLY,
            expected_revision=1,
        )
        _assert_operation_invalidated(binding, authorization)
        assert restored.claimant.public_key("claimant-key") == b"C" * 32


@pytest.mark.external_postgresql
def test_relevant_credential_and_entitlement_mutations_invalidate_exact_operation(
    reserved,
    monkeypatch,
    tmp_path: Path,
):
    _, binding, _, retained = reserved
    # Reuse the genuine upstream INITIAL_BINDING across separate database
    # installations, so every negative control begins with valid retained A.
    for mutation in (
        "claimant_rotation",
        "requester_rotation",
        "entitlement_revocation",
        "entitlement_supersession",
        "entitlement_revision_corruption",
        "entitlement_binding_corruption",
    ):
        with _installed_authorities(monkeypatch, retained["pdsa_trust_domain"]) as (
            pair,
            subject,
            admin,
        ):
            monkeypatch.setattr(
                boundary,
                "_attempt_store_path",
                lambda name=mutation: tmp_path / f"{name}.sqlite3",
            )
            authority = boundary._issuance_authority_provider()
            authorization = boundary.resolve_root_proof_issuance_authorization(binding)
            reservation = boundary.reserve_root_proof_issuance_attempt(binding, authorization)
            assert reservation.state is AttemptState.RESERVED_AWAITING_SIGNATURES
            original_hash = authorization.authorization.authorization_evidence_sha256
            original_state = authority.entitlement_registry.authoritative_state(subject).state
            if mutation == "claimant_rotation":
                replacement = pair.claimant_admin.rotate_credential(
                    principal_id="deployment-provisioning-principal",
                    credential_id="claimant-credential-next",
                    key_id="claimant-key-next",
                    key_version=2,
                    public_key=b"N" * 32,
                    expected_revision=1,
                )
                assert replacement.key_version == 2
                assert replacement.lifecycle is credentials.CredentialLifecycle.ACTIVE
            elif mutation == "requester_rotation":
                replacement = pair.requester_admin.rotate_credential(
                    principal_id=composition.REQUESTER_PRINCIPAL,
                    credential_id="requester-credential-next",
                    key_id="requester-key-next",
                    key_version=2,
                    public_key=b"S" * 32,
                    expected_revision=1,
                )
                assert replacement.key_version == 2
                assert replacement.lifecycle is credentials.CredentialLifecycle.ACTIVE
            elif mutation == "entitlement_revocation":
                result = admin.revoke_entitlement(
                    RevokeEntitlementRequest(admin_predecessor_for(original_state))
                )
                assert result.outcome is AdminOutcome.COMMITTED
                assert result.state.lifecycle is EntitlementLifecycle.REVOKED
                assert result.state.authoritative_state_revision == 2
            elif mutation == "entitlement_supersession":
                result = admin.supersede_entitlement(
                    SupersedeEntitlementRequest(
                        admin_predecessor_for(original_state),
                        replace(original_state.identity, entitlement_generation=2),
                        original_state.provenance,
                    )
                )
                assert result.outcome is AdminOutcome.COMMITTED
                assert result.state.lifecycle is EntitlementLifecycle.ACTIVE
                assert result.state.identity.entitlement_generation == 2
                assert result.state.authoritative_state_revision == 3
            else:
                # Test damaged authoritative evidence without invoking BIND or
                # manufacturing an issuer-authorized bound state.
                field, value = (
                    ("authoritative_state_revision", 2)
                    if mutation == "entitlement_revision_corruption"
                    else ("binding", {"kind": "BOUND"})
                )
                with psycopg.connect(registry_tests.BASE_DSN, autocommit=True) as conn:
                    conn.execute(
                        sql.SQL(
                            "UPDATE {}.history SET state=jsonb_set(state,%s,%s), "
                            "state_integrity=md5(jsonb_set(state,%s,%s)::text) "
                            "WHERE lookup_handle=%s"
                        ).format(sql.Identifier(admin._schema)),
                        (
                            [field],
                            psycopg.types.json.Jsonb(value),
                            [field],
                            psycopg.types.json.Jsonb(value),
                            subject.lookup_handle,
                        ),
                    )
                assert (
                    authority.entitlement_registry.authoritative_state(subject).outcome
                    is RegistryReadOutcome.CORRUPT
                )
            _assert_operation_invalidated(binding, authorization)
            if mutation in ("requester_rotation", "entitlement_supersession"):
                fresh = boundary.resolve_root_proof_issuance_authorization(binding)
                assert fresh.authorization.authorization_evidence_sha256 != original_hash


@pytest.mark.external_postgresql
def test_genuine_aggregate_rejects_context_scope_and_live_material_tampering(monkeypatch):
    with _installed_authorities(monkeypatch, "td_authority") as (pair, subject, _):
        authority = composition._configured_root_proof_issuance_authority()
        context = {
            "environment": "PRODUCTION",
            "pdsa_trust_domain": subject.trust_domain,
            "product_scope": "CryptoHunter",
            "reservation_relation": "EXACT_OPERATION_ACCOUNT",
        }
        assert (
            authority.resolve_initial_binding(
                canonical_json_bytes(context)
            ).provisioning_principal_id
            == "deployment-provisioning-principal"
        )
        for change in (
            {"environment": "TEST"},
            {"pdsa_trust_domain": "different-domain"},
            {"product_scope": "different-product"},
            {"reservation_relation": "other"},
        ):
            with pytest.raises(composition.ProductionLocalIssuanceAuthorityError):
                authority.resolve_initial_binding(canonical_json_bytes({**context, **change}))
        with psycopg.connect(registry_tests.BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("UPDATE {}.credentials SET key_material_identity=%s").format(
                    sql.Identifier(pair.setup.requester.schema)
                ),
                ("sha256:" + "f" * 64,),
            )
        with pytest.raises(composition.ProductionLocalIssuanceAuthorityError, match="UNAVAILABLE"):
            composition._configured_root_proof_issuance_authority()


@pytest.mark.external_postgresql
@pytest.mark.parametrize("candidate_field", ["account_id", "logical_operation_id"])
def test_genuine_claimant_principal_cannot_be_candidate_account_or_operation(
    monkeypatch, candidate_field
):
    context = {
        "environment": "PRODUCTION",
        "pdsa_trust_domain": "td_authority",
        "product_scope": "CryptoHunter",
        "reservation_relation": "EXACT_OPERATION_ACCOUNT",
        "account_id": "acct_018f3e70-7b5c-7c21-8b9a-0123456789ab",
        "logical_operation_id": "ago_018f3e70-7b5c-7c21-8b9a-0123456789ab",
    }
    with _installed_authorities(monkeypatch, "td_authority", context[candidate_field]) as (
        pair,
        _,
        _,
    ):
        authority = composition._configured_root_proof_issuance_authority()
        assert (
            pair.claimant.resolve_claimant(context[candidate_field]).provisioning_principal_id
            == context[candidate_field]
        )
        with pytest.raises(
            composition.ProductionLocalIssuanceAuthorityError,
            match="PREACCOUNT_CLAIMANT_PRINCIPAL_REQUIRED",
        ):
            authority.resolve_initial_binding(canonical_json_bytes(context))


@pytest.mark.external_postgresql
def test_runtime_seam_rejects_admin_role_in_place_of_requester_runtime(monkeypatch):
    with _installed_authorities(monkeypatch, "td_authority") as (pair, _, _):
        monkeypatch.setenv(
            "CH_ROOT_PROOF_REQUESTER_RUNTIME_DSN", _connection(pair.setup.requester.admin_role).dsn
        )
        with pytest.raises(composition.ProductionLocalIssuanceAuthorityError, match="UNAVAILABLE"):
            composition._configured_root_proof_issuance_authority()
