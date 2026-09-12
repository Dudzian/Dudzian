"""S9D-C11 tests for a single crash-atomic operator authority publication."""

from dataclasses import fields

import pytest

from bot_core.alerts.store import (
    AlertStore,
    AlertStoreError,
    AlertStoreSnapshot,
    AtomicAlertAuthorityState,
    InMemoryAlertStoreCarrier,
    InMemoryHistoricalAuthorizationDecisionAuthority,
)
from tests.alerts.test_alert_store import authorized, initial
from tests.security.test_authentication import NOW


def _arranged(tmp_path):
    carrier = InMemoryAlertStoreCarrier()
    decisions = InMemoryHistoricalAuthorizationDecisionAuthority()
    auth, authority, sources, _, store, item = initial(
        tmp_path,
        carrier=carrier,
        historical_authorizations=decisions,
    )
    request, proof = authorized(auth, authority, item, 1)
    return authority, sources, carrier, decisions, store, request, proof


@pytest.mark.parametrize("point", ("BEFORE_PUBLICATION", "FORMER_POST_CARRIER_PRE_DECISION"))
def test_combined_boundary_failure_exposes_neither_side(tmp_path, point):
    authority, sources, carrier, decisions, store, request, proof = _arranged(tmp_path)
    before_state = carrier._state
    carrier.fail_combined_next = point
    with pytest.raises(AlertStoreError, match="COMBINED_COMMIT_FAILED"):
        store.mutate(
            alert_id="alert-001",
            proof=proof,
            request=request,
            expected_alert_revision=1,
            now_utc=NOW,
        )
    assert carrier._state is before_state
    assert carrier.load().store_revision == 1
    assert decisions.committed_decisions == ()
    restored = AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)
    assert restored.current("alert-001").alert_revision == 1


def test_historical_transaction_dependency_failure_exposes_neither_side(tmp_path):
    _, _, carrier, decisions, store, request, proof = _arranged(tmp_path)
    before_state = carrier._state
    decisions.fail_next = True
    with pytest.raises(AlertStoreError, match="HISTORICAL_TRANSACTION_COMMIT_FAILED"):
        store.mutate(
            alert_id="alert-001",
            proof=proof,
            request=request,
            expected_alert_revision=1,
            now_utc=NOW,
        )
    assert carrier._state is before_state
    assert decisions.committed_decisions == ()
    assert store.snapshot() == before_state.alert_store_snapshot


def test_successful_operator_commit_has_one_combined_publication_and_restarts(tmp_path):
    authority, sources, carrier, decisions, store, request, proof = _arranged(tmp_path)
    before_state = carrier._state
    committed = store.mutate(
        alert_id="alert-001",
        proof=proof,
        request=request,
        expected_alert_revision=1,
        now_utc=NOW,
    )
    after_state = carrier._state
    assert isinstance(after_state, AtomicAlertAuthorityState)
    assert after_state is not before_state
    assert after_state.alert_store_snapshot.store_revision == 2
    assert tuple(after_state.committed_historical_authorization_decisions.values()) == (
        decisions.committed_decisions[0],
    )
    obligation = after_state.alert_store_snapshot.audit_outbox[0]
    assert decisions.resolve(obligation.historical_authorization_decision_id)

    restarted = AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)
    assert restarted.current("alert-001") == committed
    assert restarted.snapshot().operator_replays[0].result_revision == committed.alert_revision


def test_alert_snapshot_cannot_embed_or_mint_historical_membership(tmp_path):
    authority, sources, carrier, decisions, store, request, proof = _arranged(tmp_path)
    assert "historical_authorization_decisions" not in {
        item.name for item in fields(AlertStoreSnapshot)
    }

    # Copying a coherently sealed operator snapshot into the ordinary snapshot
    # projection cannot copy its trusted combined-carrier membership.
    other = _arranged(tmp_path / "other")
    other[4].mutate(
        alert_id="alert-001",
        proof=other[6],
        request=other[5],
        expected_alert_revision=1,
        now_utc=NOW,
    )
    carrier._snapshot = other[2].load()
    with pytest.raises(AlertStoreError, match="AUDIT_PROVENANCE_INVALID"):
        AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)
    assert decisions.committed_decisions == ()
