"""Orkiestracja z rzeczywistymi autorytetami i przerwaniami na ich granicach.

Szybka granica omija wyłącznie wcześniejszą lineage InitialBinding. Rozwiązanie
publicznej autoryzacji, podpisy Ed25519, SQLite i trzy autorytety PostgreSQL
pozostają rzeczywiste. Pełna lineage jest w osobnym teście publicznego API.
"""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql

from bot_core import (
    cha_root_proof_signing_custody as custody,
    postgresql_authenticated_issuer_history as history_storage,
    postgresql_checkpoint_authority as checkpoint_storage,
    root_proof_issuance_reconciliation as reconciliation,
    root_proof_issuer_orchestration as orchestration,
)
from bot_core.authenticated_issuer_history import (
    NO_PREDECESSOR,
    HistoryEventIdentity,
    ReconciliationOutcome,
    attest_head,
)
from bot_core.cha_attempt_store import (
    AttemptState,
    RecoveryResolution,
    SQLiteCHAAttemptStore,
)
from bot_core.cha_issuance_request import IssuanceSigningRole
from bot_core.entitlement_registry_contract import (
    AdminOutcome,
    BindOutcome,
    BindRequest,
    RevokeEntitlementRequest,
    admin_predecessor_for,
    predecessor_for,
)
from bot_core.licensing import (
    cha_root_proof_attempt_reservation as reservation,
    cha_root_proof_signed_attempt as signed,
)
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.cha_account_reservation import VerifiedAccountGenesisInitialBinding
from bot_core.local_signing_custody import LocalRootProofSigningProvider
from bot_core.postgresql_preaccount_credentials import CredentialLifecycle
from bot_core.root_proof_issuer_recovery import RootProofRecoveryPoint
from tests.security import test_root_proof_issuance_reconciliation as retained_tests

authorities = retained_tests.authorities
external = pytest.mark.external_postgresql


@pytest.fixture
def selected_authorities(authorities, monkeypatch):
    """Zachowaj publiczne sprawdzanie autoryzacji, podmień wcześniejszy binding.

    Pierwotny fixture reconciliation ma syntetyczny evidence digest. Tutaj całą
    autoryzację odtwarzamy z rzeczywistego resolution i podpisujemy jej dokładne
    request bytes. To nie jest fixture emitujący Root Proof.
    """
    value = authorities.value
    initial = authorities.auth
    context = {
        name: getattr(initial, name)
        for name in (
            "environment",
            "product_scope",
            "logical_operation_id",
            "account_id",
            "canonical_genesis_request_fingerprint_sha256",
            "initial_binding_reference",
            "initial_binding_digest_sha256",
            "reservation_identity",
            "reservation_relation",
            "initial_binding_sha256",
        )
    }
    context["pdsa_trust_domain"] = initial.trust_domain
    raw = canonical_json_bytes({"fixture": "wcześniejsza-genuine-lineage-poza-granicą"})

    def earlier_binding(binding):
        assert binding is value.binding
        return raw, context

    monkeypatch.setattr(reservation, "_binding_context", earlier_binding)
    monkeypatch.setattr(reservation, "_attempt_store_path", lambda: value.path)
    _, auth, _ = reservation._resolve_from_provider(value.binding, value.authority)
    value.path.unlink()
    identities = tuple(
        retained_tests.signed_store_tests.signer_identity(role, auth, private)
        for role, private in zip(IssuanceSigningRole, value.keys, strict=True)
    )
    with SQLiteCHAAttemptStore(value.path, auth.trust_domain) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        for private, identity in zip(value.keys, identities, strict=True):
            current = retained_tests.signed_store_tests.checkpoint(
                store, current, private, identity
            )
        identity = retained_tests.signed_store_tests.final_identity(store, current)
        current = store.finalize_attempt(identity, expected_fence=current.fence)
        retained = store.signed_request(auth.logical_operation_id)
    authorities.auth = auth
    value.current, value.retained, value.signer_identities = current, retained, identities
    # Prawdziwy retained reader; testowy seam nie zastępuje jego walidacji.
    monkeypatch.setattr(
        reconciliation,
        "_read_issuance_reconciliation_source",
        signed._read_issuance_reconciliation_source,
    )
    return authorities


def _report(authorities):
    return orchestration._orchestrate_selected_authorities(
        authorities.value.binding, authorities.runtime
    )


def _no_grant(report):
    assert type(report) is orchestration.RootProofIssuanceOrchestrationResult
    assert report.authorizes_operations is False
    assert report.root_proof_issuance_available is False
    assert report.atomic_global_snapshot is False
    assert "SEMANTIC_ISSUER_SIGNING_COMPOSITION_UNAVAILABLE" in report.reasons
    assert "ATOMIC_SIGNED_PROOF_BIND_HISTORY_AUTHORITY_UNAVAILABLE" in report.reasons
    assert report.blocked_operations == (
        "ROOT_PROOF_SIGN",
        "REGISTRY_BIND",
        "ISSUER_HISTORY_APPEND",
        "CHECKPOINT_ADVANCE",
        "CHA_FINALIZE",
        "ATTEMPT_REPLACEMENT",
        "ISSUER_TRANSPORT",
    )
    for name in (
        "root_proof",
        "root_proof_bytes",
        "authority_authenticated_evidence_reference",
        "outcome_authoritatively_unbound",
    ):
        assert not hasattr(report, name)


def _prohibit_mutations(authorities, monkeypatch):
    """Rejestrator obejmuje produkcyjne porty i CHA, poza setupem scenariusza."""
    called = []

    def forbidden(*args, **kwargs):
        called.append(True)
        pytest.fail("orkiestracja wywołała mutację autorytetu")

    for target, names in (
        (type(authorities.value.authority.entitlement_registry), ("compare_and_swap_bind",)),
        (type(authorities.history), ("append", "append_exact_successor", "retain_attested_head")),
        (type(authorities.checkpoint), ("advance",)),
        (type(authorities.signer), ("sign_history_head",)),
        (
            SQLiteCHAAttemptStore,
            (
                "reserve_or_resolve_attempt_id",
                "finalize_attempt",
                "prepare_signature",
                "persist_signature",
                "record_recovery_resolution",
                "replace_after_authoritative_unbound",
                "supersede_pre_send",
                "_write",
            ),
        ),
        (signed, ("execute_local_issuance",)),
        (custody._RuntimeCustody, ("_sign",)),
        (LocalRootProofSigningProvider, ("sign_root_proof",)),
    ):
        for name in names:
            monkeypatch.setattr(target, name, forbidden)
    return called


def _durable_snapshot(authorities):
    return (
        retained_tests.preflight_tests._database_dump(authorities.value.path),
        authorities.value.authority.entitlement_registry.retained_history(
            authorities.value.subject
        ),
        authorities.history.retained_history(),
        authorities.checkpoint.authenticated_read_snapshot(),
    )


def _cha_resolution(authorities, state):
    current = authorities.value.current
    with SQLiteCHAAttemptStore(authorities.value.path, authorities.auth.trust_domain) as store:
        store.record_recovery_resolution(
            authorities.auth.logical_operation_id,
            RecoveryResolution(
                current.reservation.issuance_attempt_id,
                state,
                "fixture:zachowany-wynik",
                hashlib.sha256(b"fixture:zachowany-wynik").hexdigest(),
            ),
            expected_fence=current.fence,
        )


@external
def test_live_admission_authenticates_current_tuple_and_blocks_every_write(
    selected_authorities, monkeypatch
):
    authorities = selected_authorities
    before = _durable_snapshot(authorities)
    calls = _prohibit_mutations(authorities, monkeypatch)
    report = _report(authorities)
    assert report.disposition == "EXECUTION_BLOCKED"
    assert (
        report.admission is orchestration.RootProofAdmissionDecision.LIVE_ELIGIBLE_EXECUTION_BLOCKED
    )
    assert report.recovery.recovery_point is RootProofRecoveryPoint.BEFORE_FIRST_MUTATION
    assert report.recovery.outcome is ReconciliationOutcome.NOT_FOUND
    assert len(report.observations) == 3
    assert report.observations[0] == report.observations[1] == report.observations[2]
    assert (
        report.observations[0].attempt_identity_digest_sha256
        == authorities.value.current.identity.digest_sha256
    )
    _no_grant(report)
    assert calls == []
    assert _durable_snapshot(authorities) == before
    with pytest.raises(FrozenInstanceError):
        report.authorizes_operations = True
    with pytest.raises(FrozenInstanceError):
        report.recovery.predecessor_proven = True


@external
def test_interruption_before_first_mutation_leaves_every_store_unchanged(
    selected_authorities, monkeypatch
):
    authorities = selected_authorities
    before = _durable_snapshot(authorities)
    calls = _prohibit_mutations(authorities, monkeypatch)
    cuts = []

    def interrupt(name):
        cuts.append(name)
        if name == "BEFORE_FIRST_MUTATION":
            raise RuntimeError("fixture:utrata-odpowiedzi-przed-pierwszym-zapisem")

    monkeypatch.setattr(orchestration, "_cut", interrupt)
    report = _report(authorities)
    assert cuts == ["RETAINED_STATE_OBSERVED", "LIVE_ADMISSION_RECHECKED", "BEFORE_FIRST_MUTATION"]
    assert report.disposition == "REFUSED"
    assert len(report.observations) == 3
    assert "fixture:utrata-odpowiedzi" not in repr(report)
    _no_grant(report)
    assert calls == []
    assert _durable_snapshot(authorities) == before


@external
@pytest.mark.parametrize(
    "state", [AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN, AttemptState.EXACT_BOUND_RECOVERED]
)
def test_unknown_or_contradictory_prior_outcome_is_observed_before_new_admission(
    selected_authorities, monkeypatch, state
):
    authorities = selected_authorities
    _cha_resolution(authorities, state)
    before = _durable_snapshot(authorities)
    calls = _prohibit_mutations(authorities, monkeypatch)

    def forbidden_admission(*args):
        pytest.fail("zachowany wynik nie może przejść do nowej decyzji admission")

    monkeypatch.setattr(orchestration, "_live_admission", forbidden_admission)
    report = _report(authorities)
    assert report.admission is orchestration.RootProofAdmissionDecision.NOT_EVALUATED
    if state is AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN:
        assert report.recovery.recovery_point is RootProofRecoveryPoint.UNKNOWN_PRIOR_OUTCOME
        assert report.recovery.outcome is ReconciliationOutcome.INDETERMINATE
    else:
        assert report.disposition == "REFUSED"
        assert report.recovery.outcome is ReconciliationOutcome.CONFLICT
    _no_grant(report)
    assert calls == []
    assert _durable_snapshot(authorities) == before


@external
@pytest.mark.parametrize("principal", ["requester", "claimant"])
def test_revoked_live_public_credential_refuses_admission_even_with_historical_signature(
    selected_authorities, monkeypatch, principal
):
    authorities = selected_authorities
    admin = getattr(authorities.value.pair, principal + "_admin")
    principal_id = (
        authorities.auth.requester_principal_id
        if principal == "requester"
        else authorities.auth.provisioning_principal_id
    )
    admin.transition_lifecycle(
        principal_id=principal_id, lifecycle=CredentialLifecycle.REVOKED, expected_revision=1
    )
    before = _durable_snapshot(authorities)
    calls = _prohibit_mutations(authorities, monkeypatch)
    report = _report(authorities)
    assert report.disposition == "REFUSED"
    assert report.admission is orchestration.RootProofAdmissionDecision.REFUSED
    assert report.recovery.recovery_point is RootProofRecoveryPoint.BEFORE_FIRST_MUTATION
    _no_grant(report)
    assert calls == []
    assert _durable_snapshot(authorities) == before


@external
@pytest.mark.parametrize(
    "change", ["registry_revision", "requester_lifecycle", "history_signer_generation"]
)
def test_third_observation_refuses_changes_after_live_admission(
    selected_authorities, monkeypatch, change
):
    authorities = selected_authorities
    calls = _prohibit_mutations(authorities, monkeypatch)
    changed = False

    def mutate(name):
        nonlocal changed
        if name != "LIVE_ADMISSION_RECHECKED":
            return
        changed = True
        if change == "registry_revision":
            state = authorities.value.authority.entitlement_registry.authoritative_state(
                authorities.value.subject
            ).state
            result = authorities.value.entitlement_admin.revoke_entitlement(
                RevokeEntitlementRequest(admin_predecessor_for(state))
            )
            assert result.outcome is AdminOutcome.COMMITTED
        elif change == "requester_lifecycle":
            authorities.value.pair.requester_admin.transition_lifecycle(
                principal_id=authorities.auth.requester_principal_id,
                lifecycle=CredentialLifecycle.REVOKED,
                expected_revision=1,
            )
        else:
            authorities.signer.generation += 1

    monkeypatch.setattr(orchestration, "_cut", mutate)
    report = _report(authorities)
    assert changed
    assert report.disposition == "REFUSED"
    assert report.admission is orchestration.RootProofAdmissionDecision.REFUSED
    assert len(report.observations) == 3
    assert report.observations[0] == report.observations[1]
    assert report.observations[2] != report.observations[1]
    assert "AUTHORITATIVE_PREDECESSOR_CHANGED_BEFORE_MUTATION" in report.reasons
    _no_grant(report)
    assert calls == []


@external
def test_changed_registry_between_retained_reads_blocks_live_admission(
    selected_authorities, monkeypatch
):
    authorities = selected_authorities
    calls = _prohibit_mutations(authorities, monkeypatch)

    def change_predecessor(name):
        if name == "RETAINED_STATE_OBSERVED":
            current = authorities.value.authority.entitlement_registry.authoritative_state(
                authorities.value.subject
            ).state
            result = authorities.value.entitlement_admin.revoke_entitlement(
                RevokeEntitlementRequest(admin_predecessor_for(current))
            )
            assert result.outcome is AdminOutcome.COMMITTED

    monkeypatch.setattr(orchestration, "_cut", change_predecessor)
    report = _report(authorities)
    assert report.disposition == "REFUSED"
    assert report.outcome is ReconciliationOutcome.INDETERMINATE
    assert report.admission is orchestration.RootProofAdmissionDecision.NOT_EVALUATED
    assert tuple(sample.registry_revision for sample in report.observations) == (1, 2)
    assert "CHANGED:registry_revision" in report.reasons
    _no_grant(report)
    assert calls == []


@external
@pytest.mark.parametrize("timing", ["RETAINED_STATE_OBSERVED", "LIVE_ADMISSION_RECHECKED"])
def test_changed_pinned_checkpoint_instance_is_refused(selected_authorities, monkeypatch, timing):
    authorities = selected_authorities
    checkpoint = checkpoint_storage.PostgreSQLCheckpointAuthority(
        retained_tests._connection(authorities.checkpoint_setup.runtime_role),
        schema=authorities.checkpoint_setup.schema,
        checkpoint_id=authorities.checkpoint_id,
        stream=authorities.stream,
        history=authorities.history,
        verification_authorities=(authorities.signer,),
    )
    replacement = reconciliation._compose_reconciliation_runtime(
        authorities.value.authority, authorities.history, checkpoint
    )
    before = _durable_snapshot(authorities)
    calls = _prohibit_mutations(authorities, monkeypatch)

    def replace_composition(name):
        if name == timing:
            monkeypatch.setattr(
                reconciliation, "_configured_reconciliation_runtime", lambda: replacement
            )

    monkeypatch.setattr(orchestration, "_cut", replace_composition)
    report = _report(authorities)
    assert report.disposition == "REFUSED"
    assert report.outcome is ReconciliationOutcome.INDETERMINATE
    assert any(reason.startswith("PINNED_AUTHORITIES_CHANGED") for reason in report.reasons)
    _no_grant(report)
    assert calls == []
    assert _durable_snapshot(authorities) == before


@external
def test_unavailable_history_signing_authority_blocks_before_admission(
    selected_authorities, monkeypatch
):
    authorities = selected_authorities
    before = retained_tests.preflight_tests._database_dump(authorities.value.path)
    calls = _prohibit_mutations(authorities, monkeypatch)

    def unavailable(self):
        raise history_storage.HistorySigningAuthorityUnavailable("fixture:sekretny-selector")

    monkeypatch.setattr(type(authorities.signer), "lifecycle_state", unavailable)
    report = _report(authorities)
    assert report.disposition == "REFUSED"
    assert report.recovery.outcome is ReconciliationOutcome.UNAVAILABLE
    assert report.admission is orchestration.RootProofAdmissionDecision.NOT_EVALUATED
    assert "fixture:sekretny-selector" not in repr(report)
    _no_grant(report)
    assert calls == []
    assert retained_tests.preflight_tests._database_dump(authorities.value.path) == before


@external
def test_real_registry_cas_conflict_cannot_become_recoverable_same_operation(
    selected_authorities, monkeypatch
):
    authorities = selected_authorities
    predecessor, committed = retained_tests._bound_state(
        authorities, account_id="acct_018f3e70-7b5c-7c21-8b9a-111111111111"
    )
    competing = replace(committed.binding, account_id=authorities.auth.account_id)
    conflict = authorities.value.authority.entitlement_registry.compare_and_swap_bind(
        BindRequest(authorities.value.subject, predecessor_for(predecessor), competing)
    )
    assert conflict.outcome is BindOutcome.CONFLICT_BOUND_TO_DIFFERENT_TUPLE
    before = _durable_snapshot(authorities)
    calls = _prohibit_mutations(authorities, monkeypatch)
    report = _report(authorities)
    assert report.disposition == "REFUSED"
    assert report.recovery.outcome is ReconciliationOutcome.CONFLICT
    assert report.recovery.recovery_point is RootProofRecoveryPoint.UNRESOLVED
    _no_grant(report)
    assert calls == []
    assert _durable_snapshot(authorities) == before


@external
@pytest.mark.parametrize(
    "field",
    ["logical_operation_id", "signed_request_payload_digest_sha256", "requester_key_version"],
)
def test_bound_other_tuple_is_refused_without_attempt_replacement(
    selected_authorities, monkeypatch, field
):
    authorities = selected_authorities
    changes = {
        field: "ago_018f3e70-7b5a-7c21-8b9a-111111111111"
        if field == "logical_operation_id"
        else "f" * 64
        if field.endswith("sha256")
        else 2
    }
    retained_tests._bound_state(authorities, **changes)
    before = _durable_snapshot(authorities)
    calls = _prohibit_mutations(authorities, monkeypatch)
    report = _report(authorities)
    assert report.disposition == "REFUSED"
    assert report.recovery.outcome is ReconciliationOutcome.CONFLICT
    _no_grant(report)
    assert calls == []
    assert _durable_snapshot(authorities) == before


@external
@pytest.mark.parametrize("primitive", ["registry", "history", "history_attestation", "checkpoint"])
def test_lost_response_after_real_commit_is_recognized_without_retrying_write(
    selected_authorities, monkeypatch, primitive
):
    authorities = selected_authorities
    if primitive == "registry":
        port_type = type(authorities.value.authority.entitlement_registry)
        original = port_type.compare_and_swap_bind

        def lost(self, request):
            result = original(self, request)
            assert result.outcome is BindOutcome.NEW_BIND_COMMITTED
            raise psycopg.OperationalError("fixture:odpowiedź-po-BIND-utracona")

        with monkeypatch.context() as fault:
            fault.setattr(port_type, "compare_and_swap_bind", lost)
            with pytest.raises(psycopg.OperationalError):
                retained_tests._bound_state(authorities)
        expected = RootProofRecoveryPoint.REGISTRY_BOUND
    else:
        predecessor, state = retained_tests._bound_state(authorities)
        payload = retained_tests._payload(authorities, predecessor, state)
        event = HistoryEventIdentity(
            authorities.auth.logical_operation_id,
            authorities.value.retained.attempt_id,
            state.binding.root_proof_id,
        )

        def lose_after_commit(name):
            if name == "after_commit":
                raise psycopg.OperationalError("fixture:odpowiedź-po-commit-utracona")

        if primitive == "history":
            with monkeypatch.context() as fault:
                fault.setattr(history_storage, "_persistence_cut", lose_after_commit)
                with pytest.raises(history_storage.HistoryOperationIndeterminate):
                    authorities.history.append(
                        expected_digest=NO_PREDECESSOR, event_identity=event, payload=payload
                    )
            assert len(authorities.history.retained_history()) == 1
            expected = RootProofRecoveryPoint.UNRESOLVED
        else:
            record, _ = authorities.history.append(
                expected_digest=NO_PREDECESSOR, event_identity=event, payload=payload
            )
            head = attest_head(record, authorities.signer)
            if primitive == "history_attestation":
                port_type = type(authorities.history)
                original = port_type.retain_attested_head

                def lose_attestation_response(self, retained_head, signer):
                    original(self, retained_head, signer)
                    raise psycopg.OperationalError("fixture:odpowiedź-po-attestation-utracona")

                with monkeypatch.context() as fault:
                    fault.setattr(port_type, "retain_attested_head", lose_attestation_response)
                    with pytest.raises(psycopg.OperationalError):
                        authorities.history.retain_attested_head(head, authorities.signer)
                expected = RootProofRecoveryPoint.HISTORY_RECORDED
            else:
                authorities.history.retain_attested_head(head, authorities.signer)
                with monkeypatch.context() as fault:
                    fault.setattr(checkpoint_storage, "_persistence_cut", lose_after_commit)
                    with pytest.raises(checkpoint_storage.CheckpointOperationIndeterminate):
                        authorities.checkpoint.advance(expected_revision=0, head=head)
                assert authorities.checkpoint.current_checkpoint().checkpoint_revision == 1
                expected = RootProofRecoveryPoint.CHECKPOINT_ACCEPTED_CHA_PENDING
    before = _durable_snapshot(authorities)
    calls = _prohibit_mutations(authorities, monkeypatch)
    report = _report(authorities)
    assert report.disposition == ("REFUSED" if primitive == "history" else "EXECUTION_BLOCKED")
    assert report.recovery.recovery_point is expected
    assert report.admission is orchestration.RootProofAdmissionDecision.NOT_EVALUATED
    if primitive == "history":
        assert report.recovery.outcome is ReconciliationOutcome.INDETERMINATE
        assert report.recovery.reasons == ("EXACT_SIGNED_HISTORY_HEAD_REQUIRED",)
        assert report.observations[0].history_sequence == 1
        assert report.observations[0].history_head_observations == ()
    else:
        assert report.recovery.registry_binding_observed is True
        assert report.recovery.predecessor_proven is True
        assert report.recovery.history_record_observed is (primitive != "registry")
        assert report.recovery.checkpoint_acceptance_observed is (primitive == "checkpoint")
    _no_grant(report)
    assert calls == []
    assert _durable_snapshot(authorities) == before


@external
@pytest.mark.parametrize("cha_recovered", [False, True])
def test_accepted_checkpoint_and_identical_retry_preserve_attempt_and_proof_ids(
    selected_authorities, monkeypatch, cha_recovered
):
    authorities = selected_authorities
    retained_tests._append(authorities)
    if cha_recovered:
        _cha_resolution(authorities, AttemptState.EXACT_BOUND_RECOVERED)
    before = _durable_snapshot(authorities)
    calls = _prohibit_mutations(authorities, monkeypatch)
    first, second = _report(authorities), _report(authorities)
    assert first == second
    assert first.disposition == "EXECUTION_BLOCKED"
    assert first.admission is orchestration.RootProofAdmissionDecision.NOT_EVALUATED
    assert first.recovery.recovery_point is (
        RootProofRecoveryPoint.CHA_RECOVERY_RECORDED
        if cha_recovered
        else RootProofRecoveryPoint.CHECKPOINT_ACCEPTED_CHA_PENDING
    )
    assert first.recovery.outcome is ReconciliationOutcome.EXACT_COMMITTED
    assert first.recovery.root_proof_id == authorities.value.retained.attempt_id.replace(
        "rpa_", "rpf_"
    )
    assert (
        dict(first.observations[0].identifiers)["issuance_attempt_id"]
        == authorities.value.retained.attempt_id
    )
    assert first.recovery.checkpoint_acceptance_observed is True
    _no_grant(first)
    assert calls == []
    assert _durable_snapshot(authorities) == before


@external
@pytest.mark.parametrize("rollback", ["history_prefix", "checkpoint_prefix"])
def test_retained_checkpoint_rollback_refuses_recovery(selected_authorities, monkeypatch, rollback):
    authorities = selected_authorities
    retained_tests._append(authorities)
    # Najpierw odczyt utrwala floor autorytetu, później cofamy fizyczny prefiks.
    assert _report(authorities).recovery.outcome is ReconciliationOutcome.EXACT_COMMITTED
    with psycopg.connect(retained_tests.BASE_DSN, autocommit=True) as conn:
        if rollback == "history_prefix":
            schema = sql.Identifier(authorities.history_setup.schema)
            conn.execute(sql.SQL("DELETE FROM {}.attestations").format(schema))
            conn.execute(sql.SQL("DELETE FROM {}.records").format(schema))
            conn.execute(
                sql.SQL("UPDATE {}.streams SET head_sequence=0,head_digest=%s").format(schema),
                (NO_PREDECESSOR,),
            )
        else:
            schema = sql.Identifier(authorities.checkpoint_setup.schema)
            conn.execute(sql.SQL("DELETE FROM {}.entries").format(schema))
            conn.execute(sql.SQL("UPDATE {}.checkpoints SET revision=0").format(schema))
    before = retained_tests.preflight_tests._database_dump(authorities.value.path)
    calls = _prohibit_mutations(authorities, monkeypatch)
    report = _report(authorities)
    assert report.disposition == "REFUSED"
    assert report.recovery.outcome is ReconciliationOutcome.CORRUPT
    _no_grant(report)
    assert calls == []
    assert retained_tests.preflight_tests._database_dump(authorities.value.path) == before


@pytest.mark.parametrize(
    "untrusted",
    [
        None,
        {},
        b"binding",
        object(),
        SimpleNamespace(environment="PRODUCTION"),
        object.__new__(VerifiedAccountGenesisInitialBinding),
    ],
)
def test_public_caller_dto_or_fabricated_opaque_cannot_select_authority(untrusted, monkeypatch):
    def forbidden():
        pytest.fail("odmowa genuine guard musi poprzedzić wybór autorytetów")

    monkeypatch.setattr(reconciliation, "_configured_reconciliation_runtime", forbidden)
    report = orchestration.orchestrate_root_proof_issuance(untrusted)
    assert report.disposition == "REFUSED"
    assert report.recovery.outcome is ReconciliationOutcome.CONFLICT
    assert report.admission is orchestration.RootProofAdmissionDecision.NOT_EVALUATED
    assert report.observations == ()
    _no_grant(report)


def test_public_api_rejects_replayed_result_and_caller_selected_ports():
    report = orchestration.orchestrate_root_proof_issuance(object())
    replayed = orchestration.orchestrate_root_proof_issuance(report)
    assert replayed.disposition == "REFUSED"
    assert replayed.recovery.outcome is ReconciliationOutcome.CONFLICT
    _no_grant(replayed)
    with pytest.raises(TypeError):
        orchestration.orchestrate_root_proof_issuance(object(), runtime=object())


def test_missing_composition_is_refused_without_disclosing_deployment_secret(monkeypatch):
    # Ten przypadek izoluje awarię kompozycji po wcześniejszym genuine guard.
    monkeypatch.setattr(orchestration, "_initial_binding_snapshot", lambda value: None)

    def unavailable():
        raise RuntimeError("fixture:password=secret deployment-private-selector")

    monkeypatch.setattr(reconciliation, "_configured_reconciliation_runtime", unavailable)
    report = orchestration.orchestrate_root_proof_issuance(object())
    assert report.disposition == "REFUSED"
    assert report.outcome is ReconciliationOutcome.UNAVAILABLE
    assert report.recovery.outcome is ReconciliationOutcome.UNAVAILABLE
    assert report.observations == ()
    assert "secret" not in repr(report)
    assert "private-selector" not in repr(report)
    _no_grant(report)
