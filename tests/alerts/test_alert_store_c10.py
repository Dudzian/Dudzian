"""S9D-C10 atomic authorization provenance and restore-parity tests."""

from dataclasses import replace
from datetime import timedelta

import pytest

from bot_core.alerts.store import (
    AlertStore,
    AlertStoreError,
    InMemoryAlertStoreCarrier,
    InMemoryHistoricalAuthorizationDecisionAuthority,
    PRODUCTION_SOURCE_RESOLUTION_POLICIES,
    _seal_alert,
    _seal_history,
    _seal_obligation,
    _seal_replay,
)
from tests.alerts.test_alert_store import accept, authorized, initial, source_authority
from tests.security.test_authentication import NOW
from tests.security.test_downstream_operation_authorization import arranged


@pytest.mark.parametrize("failure", ("AUDIT", "BEFORE", "DURING"))
def test_failed_operator_transaction_leaves_no_committed_decision(tmp_path, failure):
    decisions = InMemoryHistoricalAuthorizationDecisionAuthority()
    carrier = InMemoryAlertStoreCarrier()
    auth, authority, sources, _, store, item = initial(
        tmp_path,
        carrier=carrier,
        audit_ready=(lambda: failure != "AUDIT"),
        historical_authorizations=decisions,
    )
    request, proof = authorized(auth, authority, item, 1)
    before = carrier.load()
    if failure != "AUDIT":
        carrier.fail_next = failure
    expected = "AUDIT_UNAVAILABLE" if failure == "AUDIT" else "CARRIER_COMMIT_FAILED"
    with pytest.raises(AlertStoreError, match=expected):
        store.mutate(
            alert_id="alert-001",
            proof=proof,
            request=request,
            expected_alert_revision=1,
            now_utc=NOW,
        )
    assert carrier.load() == store.snapshot() == before
    assert decisions.committed_decisions == ()
    assert before.audit_outbox == before.operator_replays == ()


def test_orphan_material_cannot_legitimize_forged_operator_snapshot(tmp_path):
    failed_decisions = InMemoryHistoricalAuthorizationDecisionAuthority()
    auth, authority, sources, carrier, store, item = initial(
        tmp_path,
        audit_ready=lambda: False,
        historical_authorizations=failed_decisions,
    )
    request, proof = authorized(auth, authority, item, 1)
    with pytest.raises(AlertStoreError, match="AUDIT_UNAVAILABLE"):
        store.mutate(
            alert_id="alert-001",
            proof=proof,
            request=request,
            expected_alert_revision=1,
            now_utc=NOW,
        )

    # A coherently sealed committed transaction from the same live request shape
    # is still not restorable through the failed transaction's empty authority.
    committed_decisions = InMemoryHistoricalAuthorizationDecisionAuthority()
    auth2, authority2, sources2, carrier2, store2, item2 = initial(
        tmp_path / "committed", historical_authorizations=committed_decisions
    )
    request2, proof2 = authorized(auth2, authority2, item2, 1)
    store2.mutate(
        alert_id="alert-001",
        proof=proof2,
        request=request2,
        expected_alert_revision=1,
        now_utc=NOW,
    )
    carrier._snapshot = carrier2.load()
    with pytest.raises(AlertStoreError, match="AUDIT_PROVENANCE_INVALID"):
        AlertStore.restore(
            authority,
            sources2,
            carrier,
            historical_authorizations=failed_decisions,
        )
    assert failed_decisions.committed_decisions == ()


def _manual_store(tmp_path):
    auth, authority, item, _, _ = arranged(tmp_path, "M0.12/ALERT_MANUAL_FACT_RESOLUTION")
    sources = source_authority(alert_type="OPERATOR_WORKFLOW_REQUIRED")
    failing = accept(sources, "FAILING", alert_type="OPERATOR_WORKFLOW_REQUIRED")
    carrier = InMemoryAlertStoreCarrier()
    decisions = InMemoryHistoricalAuthorizationDecisionAuthority()
    store = AlertStore(authority, sources, carrier, historical_authorizations=decisions)
    store.observe(alert_id="alert-001", evidence_set=failing, now_utc=NOW)
    request, proof = authorized(auth, authority, item, 1)
    store.mutate(
        alert_id="alert-001",
        proof=proof,
        request=request,
        expected_alert_revision=1,
        now_utc=NOW,
    )
    return authority, sources, carrier, decisions


def _replace_successor(snapshot, **changes):
    revisions = list(snapshot.accepted_revisions)
    revisions[-1] = _seal_alert(replace(revisions[-1], content_fingerprint_sha256="", **changes))
    history = list(snapshot.mutation_history)
    history[-1] = _seal_history(
        replace(
            history[-1],
            after_fingerprint=revisions[-1].content_fingerprint_sha256,
        )
    )
    return replace(
        snapshot,
        accepted_revisions=tuple(revisions),
        mutation_history=tuple(history),
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"lifecycle_state": "RAISED"},
        {"lifecycle_state": "ACKNOWLEDGED"},
        {"resolution_mode": None},
        {"resolution_mode": "AUTOMATIC_FACT"},
        {"resolution_mode": "EVIL"},
    ),
)
def test_manual_resolution_restore_requires_exact_terminal_semantics(tmp_path, changes):
    authority, sources, carrier, decisions = _manual_store(tmp_path)
    carrier._snapshot = _replace_successor(carrier.load(), **changes)
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)


def test_successor_current_time_must_equal_history_time(tmp_path):
    authority, sources, carrier, decisions = _manual_store(tmp_path)
    carrier._snapshot = _replace_successor(
        carrier.load(),
        current_at_utc=(NOW + timedelta(seconds=1)).isoformat().replace("+00:00", "Z"),
    )
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)


def test_operator_provenance_uses_one_exact_transaction_time(tmp_path):
    authority, sources, carrier, decisions = _manual_store(tmp_path)
    snapshot = carrier.load()
    later = (NOW + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    tampered = _replace_successor(snapshot, current_at_utc=later)
    obligation = _seal_obligation(
        replace(tampered.audit_outbox[0], obligation_id="", timestamp_utc=later)
    )
    history = list(tampered.mutation_history)
    history[-1] = _seal_history(
        replace(
            history[-1],
            timestamp_utc=later,
            audit_obligation_id=obligation.obligation_id,
        )
    )
    replay = _seal_replay(
        replace(
            tampered.operator_replays[0],
            replay_id="",
            audit_obligation_id=obligation.obligation_id,
        )
    )
    carrier._snapshot = replace(
        tampered,
        mutation_history=tuple(history),
        audit_outbox=(obligation,),
        operator_replays=(replay,),
    )
    with pytest.raises(AlertStoreError, match="AUDIT_PROVENANCE_INVALID"):
        AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)


def test_obligation_time_must_equal_history_time(tmp_path):
    authority, sources, carrier, decisions = _manual_store(tmp_path)
    snapshot = carrier.load()
    later = (NOW + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    obligation = _seal_obligation(
        replace(snapshot.audit_outbox[0], obligation_id="", timestamp_utc=later)
    )
    history = list(snapshot.mutation_history)
    history[-1] = _seal_history(replace(history[-1], audit_obligation_id=obligation.obligation_id))
    replay = _seal_replay(
        replace(
            snapshot.operator_replays[0],
            replay_id="",
            audit_obligation_id=obligation.obligation_id,
        )
    )
    carrier._snapshot = replace(
        snapshot,
        mutation_history=tuple(history),
        audit_outbox=(obligation,),
        operator_replays=(replay,),
    )
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)


def test_committed_operator_transaction_restores_with_atomic_provenance(tmp_path):
    decisions = InMemoryHistoricalAuthorizationDecisionAuthority()
    auth, authority, sources, carrier, store, item = initial(
        tmp_path, historical_authorizations=decisions
    )
    request, proof = authorized(auth, authority, item, 1)
    committed = store.mutate(
        alert_id="alert-001",
        proof=proof,
        request=request,
        expected_alert_revision=1,
        now_utc=NOW,
    )
    assert len(decisions.committed_decisions) == 1
    restored = AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)
    assert restored.current("alert-001") == committed


def test_production_policy_catalog_is_explicitly_open_until_upstream_integration():
    expected = {
        "MARKET_DATA_CURRENT_CONDITION",
        "EXECUTION_ROUTE_CONDITION",
        "KILL_SWITCH_ACTIVE",
        "RECONCILIATION_DIVERGENCE",
        "PERSISTENCE_RECOVERY_REQUIRED",
        "RISK_DECISION_DENIED",
        "DOMAIN_EXECUTION_FAILURE",
        "SECURITY_PRIVILEGED_FAILURE",
    }
    assert set(PRODUCTION_SOURCE_RESOLUTION_POLICIES) == expected
    assert {status for _, status in PRODUCTION_SOURCE_RESOLUTION_POLICIES.values()} == {
        "OPEN_SOURCE_AUTHORITY"
    }
