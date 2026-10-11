"""Odczytowa rekoncyliacja rzeczywistych autorytetów Stage 9, bez emisji Proof."""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace

import psycopg
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from psycopg import sql

from bot_core import root_proof_issuance_reconciliation as reconciliation
from bot_core.authenticated_issuer_history import (
    NO_PREDECESSOR,
    HistoryEventIdentity,
    HistoryStreamIdentity,
    ReconciliationOutcome,
    attest_head,
)
from bot_core.cha_attempt_store import (
    AttemptNotFoundError,
    AttemptState,
    RecoveryResolution,
    SQLiteCHAAttemptStore,
)
from bot_core.cha_issuance_request import IssuanceSigningRole
from bot_core.entitlement_registry_contract import (
    AdminOutcome,
    BindOutcome,
    BindRequest,
    BoundBinding,
    RegistryReadOutcome,
    RegistryReadResult,
    RetainedHistoryResult,
    RevokeEntitlementRequest,
    SupersedeEntitlementRequest,
    admin_predecessor_for,
    predecessor_for,
)
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.licensing.cha_account_reservation import VerifiedAccountGenesisInitialBinding
from bot_core.licensing.cha_root_proof_signed_attempt import _IssuanceReconciliationSource
from bot_core.local_signing_custody import SigningKeyLifecycle
from bot_core.postgresql_authenticated_issuer_history import (
    HistoryOperationIndeterminate,
    HistoryStorageUnavailable,
    PostgreSQLAuthenticatedIssuerHistory,
)
from bot_core.postgresql_checkpoint_authority import (
    CheckpointOperationIndeterminate,
    CheckpointStorageUnavailable,
    PostgreSQLCheckpointAuthority,
)
from bot_core.postgresql_checkpoint_schema import (
    PostgreSQLCheckpointProvisioning,
    provision_postgresql_checkpoint,
    provision_postgresql_checkpoint_stream,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_issuer_history_schema import (
    PostgreSQLIssuerHistoryProvisioning,
    provision_postgresql_issuer_history,
    provision_postgresql_issuer_history_stream,
)
from bot_core.postgresql_preaccount_credentials import CredentialLifecycle
from tests.licensing import test_cha_root_proof_signed_attempt as signed_store_tests
from tests.licensing.test_cha_attempt_store_stage9 import authorization
from tests.security import test_root_proof_issuer_runtime as preflight_tests
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN, _connection, _Signer

pytestmark = pytest.mark.external_postgresql


@contextmanager
def _history_boundary(value, monkeypatch):
    auth = value.current.reservation.authorization
    suffix = uuid.uuid4().hex[:10]
    history_setup = PostgreSQLIssuerHistoryProvisioning(
        f"rch_{suffix}", f"rcho_{suffix}", f"rchr_{suffix}", f"rcha_{suffix}", auth.trust_domain
    )
    checkpoint_setup = PostgreSQLCheckpointProvisioning(
        f"rcc_{suffix}", f"rcco_{suffix}", f"rccr_{suffix}", f"rcca_{suffix}", auth.trust_domain
    )
    stream = HistoryStreamIdentity(
        f"reconciliation-{suffix}",
        "IndependentAccountGenesisRootProofIssuer",
        "PRODUCTION_LOCAL",
        auth.environment,
        auth.trust_domain,
        auth.product_scope,
        1,
    )
    checkpoint_id = f"reconciliation-checkpoint-{suffix}"
    installed = []
    try:
        provision_postgresql_issuer_history(PostgreSQLConnectionConfig(BASE_DSN), history_setup)
        installed.append(history_setup)
        provision_postgresql_issuer_history_stream(
            _connection(history_setup.admin_role), schema=history_setup.schema, stream=stream
        )
        provision_postgresql_checkpoint(PostgreSQLConnectionConfig(BASE_DSN), checkpoint_setup)
        installed.append(checkpoint_setup)
        provision_postgresql_checkpoint_stream(
            _connection(checkpoint_setup.admin_role),
            schema=checkpoint_setup.schema,
            checkpoint_id=checkpoint_id,
            stream=stream,
        )
        history = PostgreSQLAuthenticatedIssuerHistory(
            _connection(history_setup.runtime_role), schema=history_setup.schema, stream=stream
        )
        signer = _Signer(auth.trust_domain)
        checkpoint = PostgreSQLCheckpointAuthority(
            _connection(checkpoint_setup.runtime_role),
            schema=checkpoint_setup.schema,
            checkpoint_id=checkpoint_id,
            stream=stream,
            history=history,
            verification_authorities=(signer,),
        )
        runtime = reconciliation._compose_reconciliation_runtime(
            value.authority, history, checkpoint
        )
        monkeypatch.setattr(reconciliation, "_configured_reconciliation_runtime", lambda: runtime)
        yield SimpleNamespace(
            value=value,
            auth=auth,
            history=history,
            checkpoint=checkpoint,
            signer=signer,
            runtime=runtime,
            stream=stream,
            history_setup=history_setup,
            checkpoint_setup=checkpoint_setup,
            checkpoint_id=checkpoint_id,
        )
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            for setup in reversed(installed):
                conn.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(setup.schema))
                )
                for role in (setup.runtime_role, setup.admin_role, setup.schema_owner_role):
                    conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


@pytest.fixture
def authorities(monkeypatch, tmp_path):
    """Silnik używa rzeczywistych czterech magazynów i Ed25519.

    Wyłącznie wcześniejsza lineage InitialBinding jest poza tą unit boundary.
    Jej autentyczna, pełna granica jest sprawdzana w osobnym teście public API;
    publiczny guard pozostaje aktywny w testach caller DTO i fabricated opaque.
    """
    keys = (Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate())
    public = tuple(
        key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        for key in keys
    )
    auth = authorization(
        bootstrap_entitlement_id="ent_018f3e70-7b5a-7c21-8b9a-0123456789ab",
        provisioning_principal_id="deployment-provisioning-principal",
        requester_key_id="requester-key",
        claimant_key_id="claimant-key",
    )
    with preflight_tests.genuine._installed_authorities(
        monkeypatch, auth.trust_domain, requester_public=public[0], claimant_public=public[1]
    ) as (pair, subject, entitlement_admin):
        authority = preflight_tests.boundary._issuance_authority_provider()
        path = tmp_path / "attempts.sqlite3"
        identities = tuple(
            signed_store_tests.signer_identity(role, auth, private)
            for role, private in zip(IssuanceSigningRole, keys, strict=True)
        )
        with SQLiteCHAAttemptStore(path, auth.trust_domain) as store:
            current = store.reserve_or_resolve_attempt_id(auth)
            for private, identity in zip(keys, identities, strict=True):
                current = signed_store_tests.checkpoint(store, current, private, identity)
            identity = signed_store_tests.final_identity(store, current)
            current = store.finalize_attempt(identity, expected_fence=current.fence)
            retained = store.signed_request(auth.logical_operation_id)
        selector = object()

        def read_source(binding):
            assert binding is selector
            with SQLiteCHAAttemptStore.open_readonly(path, auth.trust_domain) as reader:
                observed = reader.attempt(auth.logical_operation_id)
                try:
                    request = reader.signed_request(auth.logical_operation_id)
                except AttemptNotFoundError:
                    if observed.state is not AttemptState.RESERVED_AWAITING_SIGNATURES:
                        raise
                    request = None
            return _IssuanceReconciliationSource(
                selector,
                b"{}",
                b"{}",
                observed.reservation.authorization,
                observed,
                request,
                authority,
            )

        monkeypatch.setattr(reconciliation, "_read_issuance_reconciliation_source", read_source)
        value = SimpleNamespace(
            binding=selector,
            authority=authority,
            current=current,
            retained=retained,
            subject=subject,
            entitlement_admin=entitlement_admin,
            pair=pair,
            path=path,
            keys=keys,
            signer_identities=identities,
        )
        with _history_boundary(value, monkeypatch) as boundary:
            yield boundary


def _report(authorities):
    return reconciliation._reconcile_selected_authorities(
        authorities.value.binding, authorities.runtime
    )


def _no_grant(report):
    assert report.authorizes_operations is False
    assert report.atomic_global_snapshot is False
    assert not hasattr(report, "authority_authenticated_evidence_reference")
    assert not hasattr(report, "outcome_authoritatively_unbound")


def _bound_state(authorities, **changes):
    auth = authorities.auth
    initial = authorities.value.authority.entitlement_registry.authoritative_state(
        authorities.value.subject
    ).state
    requested = BoundBinding(
        auth.logical_operation_id,
        auth.account_id,
        auth.canonical_genesis_request_fingerprint_sha256,
        auth.entitlement_generation,
        authorities.value.current.reservation.issuance_attempt_id,
        auth.requester_principal_id,
        auth.requester_key_id,
        auth.requester_key_version,
        auth.provisioning_principal_id,
        auth.claimant_key_id,
        auth.claimant_key_version,
        authorities.value.retained.digest,
        authorities.value.retained.reference,
        authorities.value.current.reservation.issuance_attempt_id.replace("rpa_", "rpf_"),
        "retained-issuer-credential",
        1,
    )
    requested = replace(requested, **changes)
    result = authorities.value.authority.entitlement_registry.compare_and_swap_bind(
        BindRequest(authorities.value.subject, predecessor_for(initial), requested)
    )
    assert result.outcome is BindOutcome.NEW_BIND_COMMITTED
    return initial, result.authoritative_state


def _payload(authorities, predecessor, state):
    # Niezależny fixture projekcji odczytowej; nie implementuje emisji Proof.
    return parse_canonical(
        canonical_json_bytes(
            {
                "schema_version": "1",
                "observation_kind": "ROOT_PROOF_ISSUANCE_BINDING_OBSERVATION",
                "authorization": asdict(authorities.auth),
                "entitlement": asdict(state),
                "predecessor": asdict(predecessor_for(predecessor)),
                "attempt_identity_digest_sha256": authorities.value.current.identity.digest_sha256,
            }
        )
    )


def _append(authorities, *, payload=None, event=None, accept=True):
    previous = authorities.history.current_record()
    if event is None:
        event = HistoryEventIdentity(
            authorities.auth.logical_operation_id,
            authorities.value.current.reservation.issuance_attempt_id,
            authorities.value.current.reservation.issuance_attempt_id.replace("rpa_", "rpf_"),
        )
    if payload is None:
        predecessor, state = _bound_state(authorities)
        payload = _payload(authorities, predecessor, state)
    record, replayed = authorities.history.append(
        expected_digest=NO_PREDECESSOR if previous is None else previous.authenticated_digest,
        event_identity=event,
        payload=payload,
    )
    assert not replayed
    head = attest_head(record, authorities.signer)
    authorities.history.retain_attested_head(head, authorities.signer)
    if accept:
        current = authorities.checkpoint.current_checkpoint()
        authorities.checkpoint.advance(
            expected_revision=0 if current is None else current.checkpoint_revision, head=head
        )
    return head


def test_pre_issuance_is_valid_observation_without_issuance_history_or_capability(authorities):
    before = preflight_tests._database_dump(authorities.value.path)
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.NOT_FOUND
    assert report.disposition == "PRE_ISSUANCE_OBSERVED"
    assert len(report.observations) == 2
    _no_grant(report)
    assert preflight_tests._database_dump(authorities.value.path) == before
    assert authorities.history.retained_history() == ()
    assert authorities.checkpoint.current_checkpoint() is None
    with pytest.raises(FrozenInstanceError):
        report.authorizes_operations = True


def test_exact_retained_binding_history_and_checkpoint_are_aligned(authorities):
    _append(authorities)
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.EXACT_COMMITTED
    assert report.disposition == "ALIGNED_COMMITTED_OBSERVED"
    assert report.observations[0] == report.observations[1]
    _no_grant(report)


def test_bound_registry_without_history_fails_closed(authorities):
    _bound_state(authorities)
    report = _report(authorities)
    assert report.outcome in (ReconciliationOutcome.NOT_FOUND, ReconciliationOutcome.INDETERMINATE)
    assert report.disposition == "REFUSED"
    _no_grant(report)


def test_missing_checkpoint_for_authenticated_history_is_not_unbound(authorities):
    _append(authorities, accept=False)
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.NOT_FOUND
    assert report.disposition == "REFUSED"
    assert report.reasons
    _no_grant(report)


def test_history_ahead_of_retained_checkpoint_is_stale(authorities):
    _append(authorities)
    _append(
        authorities,
        payload={"fixture": "unrelated-operation"},
        event=HistoryEventIdentity("another-operation", "another-attempt", "another-proof"),
        accept=False,
    )
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.STALE
    assert report.disposition == "REFUSED"
    _no_grant(report)


@pytest.mark.parametrize("mutation", ["prefix_rollback", "corrupt_record", "corrupt_checkpoint"])
def test_rollback_and_corruption_fail_closed(authorities, mutation):
    _append(authorities)
    if mutation == "prefix_rollback":
        schema = sql.Identifier(authorities.history_setup.schema)
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(sql.SQL("DELETE FROM {}.attestations").format(schema))
            conn.execute(sql.SQL("DELETE FROM {}.records").format(schema))
            conn.execute(
                sql.SQL("UPDATE {}.streams SET head_sequence=0,head_digest=%s").format(schema),
                (NO_PREDECESSOR,),
            )
        authorities.history.verify()
    elif mutation == "corrupt_record":
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("UPDATE {}.records SET canonical_record=%s WHERE sequence=1").format(
                    sql.Identifier(authorities.history_setup.schema)
                ),
                (b"{}",),
            )
    else:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("UPDATE {}.entries SET checkpoint=%s WHERE revision=1").format(
                    sql.Identifier(authorities.checkpoint_setup.schema)
                ),
                (b"{}",),
            )
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.CORRUPT
    assert report.disposition == "REFUSED"
    _no_grant(report)


@pytest.mark.parametrize("field", ["signed_request_payload_digest_sha256", "logical_operation_id"])
def test_other_binding_digest_or_operation_is_conflict(authorities, field):
    changes = (
        {field: "f" * 64}
        if field.endswith("sha256")
        else {field: "ago_018f3e70-7b5a-7c21-8b9a-111111111111"}
    )
    _bound_state(authorities, **changes)
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.CONFLICT
    assert report.disposition == "REFUSED"
    _no_grant(report)


def test_authenticated_payload_identity_mismatch_is_conflict(authorities):
    predecessor, state = _bound_state(authorities)
    payload = _payload(authorities, predecessor, state)
    payload["authorization"]["account_id"] = "acct_018f3e70-7b5a-7c21-8b9a-111111111111"
    _append(authorities, payload=payload)
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.CONFLICT
    _no_grant(report)


def test_registry_revision_changes_during_reads_are_indeterminate(authorities, monkeypatch):
    called = False

    def mutate(name):
        nonlocal called
        if called or name != "FIRST_OBSERVATION_READ":
            return
        called = True
        state = authorities.value.authority.entitlement_registry.authoritative_state(
            authorities.value.subject
        ).state
        result = authorities.value.entitlement_admin.revoke_entitlement(
            RevokeEntitlementRequest(admin_predecessor_for(state))
        )
        assert result.outcome is AdminOutcome.COMMITTED

    monkeypatch.setattr(reconciliation, "_cut", mutate)
    report = _report(authorities)
    assert called
    assert report.outcome is ReconciliationOutcome.INDETERMINATE
    assert len(report.observations) == 2
    assert tuple(sample.registry_revision for sample in report.observations) == (1, 2)
    assert all(not sample.read_failures for sample in report.observations)
    assert "CHANGED:registry_revision" in report.reasons
    _no_grant(report)


@pytest.mark.parametrize(
    ("authority", "error", "outcome"),
    [
        ("history", HistoryStorageUnavailable, ReconciliationOutcome.UNAVAILABLE),
        ("history", HistoryOperationIndeterminate, ReconciliationOutcome.INDETERMINATE),
        ("checkpoint", CheckpointStorageUnavailable, ReconciliationOutcome.UNAVAILABLE),
        ("checkpoint", CheckpointOperationIndeterminate, ReconciliationOutcome.INDETERMINATE),
    ],
)
def test_typed_unavailable_and_indeterminate_are_preserved(
    authorities, monkeypatch, authority, error, outcome
):
    def fail(*args, **kwargs):
        raise error("do-not-disclose-deployment-secret")

    monkeypatch.setattr(type(getattr(authorities, authority)), "authenticated_read_snapshot", fail)
    report = _report(authorities)
    assert report.outcome is outcome
    assert report.disposition == "REFUSED"
    assert "do-not-disclose-deployment-secret" not in repr(report)
    _no_grant(report)


def test_unknown_cha_outcome_does_not_become_pre_issuance(authorities):
    current = authorities.value.current
    with SQLiteCHAAttemptStore(authorities.value.path, authorities.auth.trust_domain) as store:
        store.record_recovery_resolution(
            authorities.auth.logical_operation_id,
            RecoveryResolution(
                current.reservation.issuance_attempt_id,
                AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
                "fixture:unknown-outcome",
                hashlib.sha256(b"fixture:unknown-outcome").hexdigest(),
            ),
            expected_fence=current.fence,
        )
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.INDETERMINATE
    assert report.disposition == "REFUSED"
    _no_grant(report)


def test_historically_revoked_history_signer_requires_exact_v2_acceptance(authorities):
    _append(authorities)
    authorities.signer.lifecycle = SigningKeyLifecycle.REVOKED
    authorities.signer.generation = 2
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.EXACT_COMMITTED
    _no_grant(report)


def test_revoked_signer_without_v2_acceptance_fails_closed(authorities):
    _append(authorities, accept=False)
    authorities.signer.lifecycle = SigningKeyLifecycle.REVOKED
    authorities.signer.generation = 2
    report = _report(authorities)
    assert report.outcome in (
        ReconciliationOutcome.UNAVAILABLE,
        ReconciliationOutcome.INDETERMINATE,
    )
    assert report.disposition == "REFUSED"
    _no_grant(report)


def test_restart_re_reads_durable_authorities_without_cached_report(authorities, monkeypatch):
    _append(authorities)
    before = _report(authorities)
    history = PostgreSQLAuthenticatedIssuerHistory(
        _connection(authorities.history_setup.runtime_role),
        schema=authorities.history_setup.schema,
        stream=authorities.stream,
    )
    checkpoint = PostgreSQLCheckpointAuthority(
        _connection(authorities.checkpoint_setup.runtime_role),
        schema=authorities.checkpoint_setup.schema,
        checkpoint_id=authorities.checkpoint_id,
        stream=authorities.stream,
        history=history,
        verification_authorities=(authorities.signer,),
    )
    runtime = reconciliation._compose_reconciliation_runtime(
        authorities.value.authority, history, checkpoint
    )
    monkeypatch.setattr(reconciliation, "_configured_reconciliation_runtime", lambda: runtime)
    authorities.runtime = runtime
    after = _report(authorities)
    assert before == after
    _no_grant(after)


@pytest.mark.parametrize("value", [object(), SimpleNamespace(environment="PRODUCTION"), {}])
def test_caller_dto_is_rejected(authorities, value):
    report = reconciliation.reconcile_root_proof_issuance_attempt(value)
    assert report.disposition == "REFUSED"
    assert report.outcome is ReconciliationOutcome.CONFLICT
    _no_grant(report)


def test_fabricated_exact_type_binding_and_report_are_rejected(authorities):
    fabricated = object.__new__(VerifiedAccountGenesisInitialBinding)
    report = reconciliation.reconcile_root_proof_issuance_attempt(fabricated)
    assert report.outcome is ReconciliationOutcome.CONFLICT
    replayed = reconciliation.reconcile_root_proof_issuance_attempt(_report(authorities))
    assert replayed.outcome is ReconciliationOutcome.CONFLICT
    _no_grant(replayed)


def test_foreign_checkpoint_cannot_be_used_as_caller_authority(authorities):
    with pytest.raises(TypeError):
        reconciliation.reconcile_root_proof_issuance_attempt(
            authorities.value.binding, checkpoint=authorities.checkpoint
        )
    with pytest.raises((TypeError, RuntimeError, ValueError)):
        reconciliation._compose_reconciliation_runtime(
            authorities.value.authority, authorities.history, object()
        )


def test_missing_attempt_store_does_not_create_database(authorities):
    path = authorities.value.path
    path.unlink()
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.NOT_FOUND
    assert report.disposition == "REFUSED"
    assert not path.exists()
    _no_grant(report)


def test_sqlite_digest_corruption_cannot_become_missing_history(authorities):
    with sqlite3.connect(authorities.value.path) as conn:
        conn.execute("UPDATE current_attempts SET digest=?", ("f" * 64,))
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.CORRUPT
    _no_grant(report)


@pytest.mark.parametrize("signatures", [0, 1, 2])
def test_valid_partial_attempt_is_indeterminate_with_exact_observed_stage(authorities, signatures):
    authorities.value.path.unlink()
    with SQLiteCHAAttemptStore(authorities.value.path, authorities.auth.trust_domain) as writer:
        current = writer.reserve_or_resolve_attempt_id(authorities.auth)
        for private, signer in list(
            zip(authorities.value.keys, authorities.value.signer_identities, strict=True)
        )[:signatures]:
            current = signed_store_tests.checkpoint(writer, current, private, signer)
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.INDETERMINATE
    assert report.disposition == "REFUSED"
    assert report.observations[0].attempt_state == current.state.value
    assert report.observations[0].attempt_identity_digest_sha256 is None
    _no_grant(report)


def test_registry_not_found_preserves_exact_status_and_never_proves_unbound(
    authorities, monkeypatch
):
    registry = type(authorities.value.authority.entitlement_registry)
    monkeypatch.setattr(
        registry,
        "authoritative_state",
        lambda self, subject: RegistryReadResult(RegistryReadOutcome.NOT_FOUND, None),
    )
    monkeypatch.setattr(
        registry,
        "retained_history",
        lambda self, subject: RetainedHistoryResult(
            RegistryReadOutcome.NOT_FOUND, subject, (), None, None
        ),
    )
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.NOT_FOUND
    assert report.disposition == "REFUSED"
    _no_grant(report)


@pytest.mark.parametrize("missing", ["current", "retained"])
def test_inconsistent_registry_presence_between_reads_is_indeterminate(
    authorities, monkeypatch, missing
):
    registry = type(authorities.value.authority.entitlement_registry)
    if missing == "current":
        monkeypatch.setattr(
            registry,
            "authoritative_state",
            lambda self, subject: RegistryReadResult(RegistryReadOutcome.NOT_FOUND, None),
        )
    else:
        monkeypatch.setattr(
            registry,
            "retained_history",
            lambda self, subject: RetainedHistoryResult(
                RegistryReadOutcome.NOT_FOUND, subject, (), None, None
            ),
        )
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.INDETERMINATE
    assert report.reasons == ("REGISTRY_READS_CHANGED_WITHIN_OBSERVATION",)
    assert report.disposition == "REFUSED"
    _no_grant(report)


def test_stably_revoked_entitlement_is_stale(authorities):
    state = authorities.value.authority.entitlement_registry.authoritative_state(
        authorities.value.subject
    ).state
    authorities.value.entitlement_admin.revoke_entitlement(
        RevokeEntitlementRequest(admin_predecessor_for(state))
    )
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.STALE
    assert report.disposition == "REFUSED"
    _no_grant(report)


def test_retained_requester_signature_uses_historical_public_generation(authorities):
    authorities.value.pair.requester_admin.transition_lifecycle(
        principal_id=authorities.auth.requester_principal_id,
        lifecycle=CredentialLifecycle.REVOKED,
        expected_revision=1,
    )
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.NOT_FOUND
    assert report.disposition == "PRE_ISSUANCE_OBSERVED"
    assert report.observations[0].requester_retained_revisions == (1, 2)
    _no_grant(report)


def test_foreign_history_exact_type_is_rejected_by_composition(authorities):
    foreign = PostgreSQLAuthenticatedIssuerHistory(
        _connection(authorities.history_setup.runtime_role),
        schema=authorities.history_setup.schema,
        stream=authorities.stream,
    )
    with pytest.raises(reconciliation.RootProofIssuanceReconciliationError):
        reconciliation._compose_reconciliation_runtime(
            authorities.value.authority, foreign, authorities.checkpoint
        )


@pytest.mark.parametrize("revoked", [False, True])
def test_recovered_bound_attempt_contradicts_authoritative_unbound_registry(authorities, revoked):
    current = authorities.value.current
    with SQLiteCHAAttemptStore(authorities.value.path, authorities.auth.trust_domain) as store:
        store.record_recovery_resolution(
            authorities.auth.logical_operation_id,
            RecoveryResolution(
                current.reservation.issuance_attempt_id,
                AttemptState.EXACT_BOUND_RECOVERED,
                "fixture:bound-outcome",
                hashlib.sha256(b"fixture:bound-outcome").hexdigest(),
            ),
            expected_fence=current.fence,
        )
    if revoked:
        state = authorities.value.authority.entitlement_registry.authoritative_state(
            authorities.value.subject
        ).state
        result = authorities.value.entitlement_admin.revoke_entitlement(
            RevokeEntitlementRequest(admin_predecessor_for(state))
        )
        assert result.outcome is AdminOutcome.COMMITTED
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.CONFLICT
    assert report.reasons == ("CHA_RECOVERED_BOUND_REGISTRY_CONFLICT",)
    _no_grant(report)


def test_retained_exact_generation_supersession_is_stale(authorities):
    state = authorities.value.authority.entitlement_registry.authoritative_state(
        authorities.value.subject
    ).state
    result = authorities.value.entitlement_admin.supersede_entitlement(
        SupersedeEntitlementRequest(
            admin_predecessor_for(state),
            replace(state.identity, entitlement_generation=2),
            state.provenance,
        )
    )
    assert result.outcome is AdminOutcome.COMMITTED
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.STALE
    assert report.reasons == ("ENTITLEMENT_GENERATION_SUPERSEDED",)
    assert report.observations[0].registry_retained_revisions == (1, 2, 3)
    _no_grant(report)


def test_other_valid_tail_signer_conflicts_with_retained_history_head(authorities, monkeypatch):
    _append(authorities)
    head = _append(
        authorities,
        payload={"fixture": "unrelated-operation"},
        event=HistoryEventIdentity("another-operation", "another-attempt", "another-proof"),
        accept=False,
    )
    other_signer = _Signer(authorities.auth.trust_domain)
    other_signer.namespace = "isolated-other-history-signing"
    other_signer.lifecycle_namespace = "isolated-other-history-lifecycle"
    other_signer.identity = replace(
        other_signer.identity, provider_namespace=other_signer.namespace
    )
    checkpoint = PostgreSQLCheckpointAuthority(
        _connection(authorities.checkpoint_setup.runtime_role),
        schema=authorities.checkpoint_setup.schema,
        checkpoint_id=authorities.checkpoint_id,
        stream=authorities.stream,
        history=authorities.history,
        verification_authorities=(authorities.signer, other_signer),
    )
    other_head = attest_head(authorities.history.current_record(), other_signer)
    checkpoint.advance(expected_revision=1, head=other_head)
    assert head.signing_credential_identity != other_head.signing_credential_identity
    runtime = reconciliation._compose_reconciliation_runtime(
        authorities.value.authority, authorities.history, checkpoint
    )
    authorities.checkpoint = checkpoint
    authorities.runtime = runtime
    monkeypatch.setattr(reconciliation, "_configured_reconciliation_runtime", lambda: runtime)
    report = _report(authorities)
    assert report.outcome is ReconciliationOutcome.CONFLICT
    assert report.reasons == ("CHECKPOINT_CURRENT_SIGNER_MISMATCH",)
    _no_grant(report)
