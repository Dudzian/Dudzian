"""S9D-C15 interleaved delivery recovery and immutable-shape regressions."""

from dataclasses import replace
from datetime import timedelta

import pytest

from bot_core.alerts.store import AlertStore, AlertStoreError, AtomicAlertAuthorityState
from tests.alerts.test_alert_store import Adapter, authorized, initial
from tests.alerts.test_alert_store_c8 import escalated
from tests.security.test_authentication import NOW


def test_delivery_recovery_preserves_intervening_operator_edge(tmp_path):
    adapter = Adapter()
    auth, authority, sources, carrier, store, item = initial(tmp_path, adapter=adapter)
    carrier.fail_next = "DURING"
    with pytest.raises(AlertStoreError, match="CARRIER_COMMIT_FAILED"):
        store.request_delivery(
            "alert-001",
            expected_alert_revision=1,
            expected_delivery_revision=0,
            attempt_id="interleaved",
            now_utc=NOW,
        )
    request, proof = authorized(auth, authority, item, 1)
    store.mutate(
        alert_id="alert-001",
        proof=proof,
        request=request,
        expected_alert_revision=1,
        now_utc=NOW + timedelta(seconds=1),
    )
    recovered = store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="interleaved",
        now_utc=NOW + timedelta(seconds=2),
    )
    assert adapter.external_effect_count == 1
    assert [x.mutation_type for x in store.snapshot().mutation_history][-2:] == [
        "ACKNOWLEDGE",
        "DELIVERY_ATTEMPT",
    ]
    assert (
        len([x for x in store.snapshot().delivery_attempts if x.attempt_id == "interleaved"]) == 1
    )
    assert recovered.alert_revision == 3
    restored = AlertStore.restore(authority, sources, carrier, delivery_adapter=adapter)
    assert restored.snapshot() == store.snapshot()
    assert (
        store.request_delivery(
            "alert-001",
            expected_alert_revision=1,
            expected_delivery_revision=0,
            attempt_id="interleaved",
            now_utc=NOW + timedelta(hours=1),
        )
        == recovered
    )


def test_route_recovery_after_sibling_route_commit(tmp_path):
    adapter, _, authority, sources, carrier, store = escalated(tmp_path)
    intents = store.snapshot().escalation_route_intents[:2]
    first, second = intents
    baseline = adapter.external_effect_count
    carrier.fail_next = "DURING"
    with pytest.raises(AlertStoreError, match="CARRIER_COMMIT_FAILED"):
        store.execute_escalation_route(
            intent_id=first.intent_id, now_utc=NOW + timedelta(seconds=301)
        )
    store.execute_escalation_route(intent_id=second.intent_id, now_utc=NOW + timedelta(seconds=302))
    store.execute_escalation_route(intent_id=first.intent_id, now_utc=NOW + timedelta(seconds=303))
    assert adapter.external_effect_count == baseline + 2
    attempts = store.snapshot().delivery_attempts
    assert sum(x.escalation_route_intent_id == first.intent_id for x in attempts) == 1
    assert sum(x.escalation_route_intent_id == second.intent_id for x in attempts) == 1
    assert (
        AlertStore.restore(authority, sources, carrier, delivery_adapter=adapter).snapshot()
        == store.snapshot()
    )


class MutableSequenceCarrier:
    def __init__(self, state):
        self.state = AtomicAlertAuthorityState(
            replace(
                state.alert_store_snapshot,
                accepted_revisions=list(state.alert_store_snapshot.accepted_revisions),
            ),
            dict(state.committed_historical_authorization_decisions),
            dict(state.committed_historical_source_decisions),
        )

    def load_atomic_state(self):
        return self.state


def test_restore_rejects_mutable_record_collection(tmp_path):
    _, authority, sources, carrier, _, _ = initial(tmp_path)
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.restore(authority, sources, MutableSequenceCarrier(carrier.load_atomic_state()))
