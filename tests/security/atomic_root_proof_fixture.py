"""A second genuine candidate sharing the expensive immutable LPPI lineage.

Only downstream installed test-state selectors change inside the scoped fixture
namespace. Each namespace still owns one initial LPPI operation, one CHA
mapping and one INITIAL_BINDING. The production establish/verify/signing APIs
create all capabilities; no authority maps, signed records or snapshots are
manufactured. Both candidates use the same real PostgreSQL authority schemas.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from bot_core.entitlement_registry_contract import (
    AdminOutcome,
    EntitlementIdentity,
    ProvisionEntitlementRequest,
    RegistrySubject,
)
from bot_core.licensing import (
    cha_root_proof_attempt_reservation as attempt_boundary,
    cha_root_proof_signed_attempt as signed_boundary,
)
from bot_core.postgresql_atomic_root_proof_commit import prepare_atomic_root_proof_commit
from bot_core.postgresql_root_proof_issuance_authority import (
    PostgreSQLRootProofIssuanceAuthority,
    _configured_root_proof_issuance_authority,
)
from bot_core.postgresql_root_proof_retention import ROOT_PROOF_SIGNING_DOMAIN
from bot_core.postgresql_root_proof_signing_intent import PostgreSQLRootProofSigningIntent
from bot_core.uuid7 import mint_uuid7, reservation_epoch_milliseconds
from deployment import (
    windows_production_cha_account_reservation as installed_account,
    windows_production_cha_operation as installed_cha,
    windows_production_lppi_operation as installed_lppi,
)


def prepare_second_atomic_candidate(value, active_authority, namespace: Path, monkeypatch):
    """Prepare a distinct genuine operation/account/entitlement before migration.

    The immutable ACTIVE enrollment and custody remain pinned to the original
    fixture. Native operation signing, durable child state, CHA signing, exact
    RESERVED/STAGED restoration and issuer verification all run unchanged.
    """
    namespace.mkdir(mode=0o700)
    first = value.preparation
    first_authority = value.genuine.authority
    first_state = first.bound_state
    subject = RegistrySubject(
        "atomic-second-entitlement-" + uuid.uuid4().hex,
        first_state.subject.environment,
        first_state.subject.trust_domain,
    )
    entitlement_id = mint_uuid7("ent_", reservation_epoch_milliseconds(datetime.now(timezone.utc)))
    provisioned = value.genuine.entitlement_admin.provision_entitlement(
        ProvisionEntitlementRequest(
            subject,
            EntitlementIdentity(
                entitlement_id,
                1,
                first_state.identity.environment,
                first_state.identity.trust_domain,
                first_state.identity.product_scope,
            ),
            first_state.provenance,
        )
    )
    assert provisioned.outcome is AdminOutcome.COMMITTED
    authority = PostgreSQLRootProofIssuanceAuthority(
        first_authority.entitlement_registry,
        first_authority.requester_registry,
        first_authority.claimant_registry,
        subject,
    )
    original_path = value.genuine.path
    # A sibling SQLite file shares the already-activated private custody paths;
    # switching its filename never replaces or resigns the original attempt.
    second_path = original_path.with_name("atomic-second-attempt.sqlite3")
    with monkeypatch.context() as selected:
        selected.setattr(
            installed_lppi,
            "_state_path",
            lambda: namespace / "initial-authenticated-operation.json",
        )
        selected.setattr(
            installed_cha,
            "_state_path",
            lambda: namespace / "initial-cha-logical-operation.json",
        )
        selected.setattr(
            installed_account,
            "_state_path",
            lambda: namespace / "initial-cha-account-reservation.json",
        )
        selected.setattr(attempt_boundary, "_attempt_store_path", lambda: second_path)
        # Deployment selection changes to another exact, live-qualified genuine
        # aggregate. No public bytes or fake provider supplies authority, and the
        # original maxsize=1 composition cache remains untouched.
        selected.setattr(attempt_boundary, "_issuance_authority_provider", lambda: authority)
        assert authority is not first_authority
        assert (
            authority.entitlement_registry._schema == first_authority.entitlement_registry._schema
        )
        assert authority.requester_registry._schema == first_authority.requester_registry._schema
        assert authority.claimant_registry._schema == first_authority.claimant_registry._schema
        upstream = installed_lppi.establish_installed_lppi_authenticated_operation(active_authority)
        cha = installed_cha.establish_installed_cha_logical_operation(upstream)
        binding = installed_account.establish_installed_account_initial_binding(cha)
        authorization = attempt_boundary.resolve_root_proof_issuance_authorization(binding)
        attempt = attempt_boundary.reserve_root_proof_issuance_attempt(binding, authorization)
        signed = signed_boundary.sign_root_proof_issuance_attempt(attempt)
        current, retained = signed_boundary._signed_snapshot(signed)
        intent = PostgreSQLRootProofSigningIntent(
            value.intent._PostgreSQLRootProofSigningIntent__connection,
            schema=value.intent_setup.schema,
            scope=value.scope,
            verification_authority=value.signer,
            issuer_authority=authority,
        )
        reserved = intent.reserve(binding, signed)
        _, fixture_private_key = value.signer._load_material()
        signature = fixture_private_key.sign(ROOT_PROOF_SIGNING_DOMAIN + reserved.canonical_payload)
        prepared = value.retention.retain_prepared(
            authorization=authorization.authorization,
            retained_request=retained,
            canonical_payload=reserved.canonical_payload,
            signature=signature,
        )
        preparation = prepare_atomic_root_proof_commit(
            binding,
            signed,
            signing_intent=intent,
            retention=value.retention,
            issuer_authority=authority,
            history=value.history,
        )
        # These are results from real guarded operations, not caller identity
        # changes applied to the first candidate or its protected SQL package.
        assert preparation.locator.operation != first.locator.operation
        assert preparation.locator.account != first.locator.account
        assert preparation.locator.entitlement != first.locator.entitlement
        assert preparation.locator.attempt != first.locator.attempt
        assert preparation.locator.root_proof != first.locator.root_proof
        assert preparation.history_record.predecessor_authenticated_digest == (
            first.history_record.predecessor_authenticated_digest
        )
        assert preparation.prepared.credential_snapshot == first.prepared.credential_snapshot
        assert attempt_boundary._issuance_authority_provider() is authority
    assert _configured_root_proof_issuance_authority() is first_authority
    return SimpleNamespace(
        preparation=preparation,
        prepared=prepared,
        reserved=reserved,
        subject=subject,
        authority=authority,
        authorization=authorization,
        current=current,
        retained=retained,
        binding=binding,
        signed=signed,
        namespace=namespace,
        path=second_path,
    )
