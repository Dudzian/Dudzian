"""S9D-C12 combined-state bijection and provenance minting tests."""

from dataclasses import replace
from types import MappingProxyType

import pytest
import bot_core.alerts.store as alerts

from bot_core.alerts.store import (
    AlertStore,
    AlertStoreError,
    AtomicAlertAuthorityState,
    InMemoryAlertStoreCarrier,
    InMemoryHistoricalAuthorizationDecisionAuthority,
    _seal_history,
    _seal_obligation,
    _seal_replay,
)
from bot_core.security.authentication import _seed_trusted_downstream_operation_definition
from tests.alerts.test_alert_store import authorized, initial
from tests.alerts.test_alert_store_c7 import committed_ack, restarted_authorization
from tests.security.test_authentication import NOW
from tests.security.test_downstream_operation_authorization import definition


def _committed(tmp_path):
    decisions = InMemoryHistoricalAuthorizationDecisionAuthority()
    auth, authority, sources, carrier, store, item = initial(
        tmp_path, historical_authorizations=decisions
    )
    before = carrier.load()
    request, proof = authorized(auth, authority, item, 1)
    store.mutate(
        alert_id="alert-001",
        proof=proof,
        request=request,
        expected_alert_revision=1,
        now_utc=NOW,
    )
    return auth, authority, sources, carrier, decisions, before


def test_orphan_decision_rejects_valid_pre_operator_snapshot(tmp_path):
    _, authority, sources, carrier, decisions, before = _committed(tmp_path)
    assert decisions.committed_decisions
    carrier._snapshot = before
    with pytest.raises(AlertStoreError, match="AUDIT_PROVENANCE_INVALID"):
        AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)


def test_missing_decision_rejects_committed_operator_snapshot(tmp_path):
    _, authority, sources, carrier, decisions, _ = _committed(tmp_path)
    carrier._state = replace(
        carrier._state,
        committed_historical_authorization_decisions=MappingProxyType({}),
    )
    with pytest.raises(AlertStoreError, match="AUDIT_PROVENANCE_INVALID"):
        AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)


@pytest.mark.parametrize(
    "change",
    (
        {"operator_id": "operator-tampered"},
        {"environment": "LIVE"},
        {"operation": "M0.12/ALERT_CLEAR_SUPPRESSION"},
        {"authorized_at_utc": "2026-09-12T00:00:00Z"},
    ),
)
def test_decision_identity_is_recomputed_on_restore(tmp_path, change):
    _, authority, sources, carrier, decisions, _ = _committed(tmp_path)
    key, decision = next(iter(carrier.load_historical_decisions().items()))
    tampered = replace(decision, **change)
    carrier._state = replace(
        carrier._state,
        committed_historical_authorization_decisions=MappingProxyType({key: tampered}),
    )
    with pytest.raises(AlertStoreError, match="AUDIT_PROVENANCE_INVALID"):
        AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)


def test_decision_mapping_key_must_equal_sealed_decision_id(tmp_path):
    _, authority, sources, carrier, decisions, _ = _committed(tmp_path)
    decision = decisions.committed_decisions[0]
    carrier._state = replace(
        carrier._state,
        committed_historical_authorization_decisions=MappingProxyType({"different-key": decision}),
    )
    with pytest.raises(AlertStoreError, match="AUDIT_PROVENANCE_INVALID"):
        AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)


def test_one_decision_cannot_back_two_operator_edges(tmp_path):
    auth, authority, sources, carrier, decisions, _ = _committed(tmp_path)
    suppression = definition("M0.12/ALERT_SET_SUPPRESSION")
    _seed_trusted_downstream_operation_definition(auth, suppression)
    request, proof = authorized(auth, authority, suppression, 2)
    store = AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)
    store.mutate(
        alert_id="alert-001",
        proof=proof,
        request=request,
        expected_alert_revision=2,
        now_utc=NOW,
    )
    snapshot = carrier.load()
    first, second = snapshot.audit_outbox
    second = _seal_obligation(
        replace(
            second,
            obligation_id="",
            historical_authorization_decision_id=first.historical_authorization_decision_id,
        )
    )
    history = list(snapshot.mutation_history)
    history[-1] = _seal_history(replace(history[-1], audit_obligation_id=second.obligation_id))
    replays = list(snapshot.operator_replays)
    replays[-1] = _seal_replay(
        replace(replays[-1], replay_id="", audit_obligation_id=second.obligation_id)
    )
    carrier._snapshot = replace(
        snapshot,
        audit_outbox=(first, second),
        mutation_history=tuple(history),
        operator_replays=tuple(replays),
    )
    with pytest.raises(AlertStoreError, match="AUDIT_PROVENANCE_INVALID"):
        AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)


def test_no_importable_capability_or_public_combined_mint_api(tmp_path):
    _, _, _, carrier, _, _ = _committed(tmp_path)
    old_state = carrier._state
    assert not hasattr(alerts, "_HISTORICAL_TRANSACTION_CAPABILITY")
    assert not hasattr(carrier, "commit_combined")
    assert not hasattr(alerts.AlertStoreCarrier, "commit_combined")
    assert carrier._state is old_state


def test_operator_state_cannot_transfer_between_carriers_through_ordinary_api(tmp_path):
    carrier_a = InMemoryAlertStoreCarrier()
    _, _, _, _, store_a, _ = initial(tmp_path / "a", carrier=carrier_a)
    _, _, _, carrier_b, _, _ = _committed(tmp_path / "b")
    before = carrier_a._state
    with pytest.raises(AlertStoreError, match="AUDIT_PROVENANCE_INVALID"):
        carrier_a.commit(
            store_a.snapshot().store_revision,
            carrier_b.load(),
        )
    assert carrier_a._state is before


def test_fresh_historical_view_restores_from_combined_carrier(tmp_path):
    sources, carrier, store = committed_ack(
        tmp_path / "live", InMemoryHistoricalAuthorizationDecisionAuthority()
    )
    fresh_view = InMemoryHistoricalAuthorizationDecisionAuthority()
    restored = AlertStore.restore(
        restarted_authorization(tmp_path / "restart"),
        sources,
        carrier,
        historical_authorizations=fresh_view,
    )
    obligation = restored.snapshot().audit_outbox[0]
    assert restored.current("alert-001") == store.current("alert-001")
    assert fresh_view.resolve(obligation.historical_authorization_decision_id)


def test_valid_atomic_state_has_exact_decision_bijection(tmp_path):
    _, authority, sources, carrier, decisions, _ = _committed(tmp_path)
    assert isinstance(carrier._state, AtomicAlertAuthorityState)
    snapshot_ids = {
        item.historical_authorization_decision_id for item in carrier.load().audit_outbox
    }
    assert snapshot_ids == set(carrier.load_historical_decisions())
    AlertStore.restore(authority, sources, carrier, historical_authorizations=decisions)
