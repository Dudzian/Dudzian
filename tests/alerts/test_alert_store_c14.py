"""S9D-C14 delivery crash replay and immutable atomic pinning tests."""

from dataclasses import replace
from datetime import timedelta

import pytest

from bot_core.alerts.store import (
    AlertStore,
    AlertStoreError,
    AtomicAlertAuthorityState,
)
from tests.alerts.test_alert_store import Adapter, initial
from tests.alerts.test_alert_store_c8 import escalated
from tests.security.test_authentication import NOW


@pytest.mark.parametrize("delay", (timedelta(seconds=1), timedelta(hours=1)))
def test_ordinary_delivery_crash_retry_uses_stable_semantic_context(tmp_path, delay):
    adapter = Adapter()
    auth, authority, sources, carrier, store, _ = initial(tmp_path, adapter=adapter)
    del auth
    carrier.fail_next = "DURING"
    with pytest.raises(AlertStoreError, match="CARRIER_COMMIT_FAILED"):
        store.request_delivery(
            "alert-001",
            expected_alert_revision=1,
            expected_delivery_revision=0,
            attempt_id="later-retry",
            now_utc=NOW,
        )
    assert adapter.external_effect_count == 1
    recovered = store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="later-retry",
        now_utc=NOW + delay,
    )
    attempt = store.snapshot().delivery_attempts[0]
    assert adapter.external_effect_count == 1
    assert len(store.snapshot().delivery_attempts) == 1
    assert attempt.attempted_at_utc == NOW.isoformat().replace("+00:00", "Z")
    assert recovered.current_at_utc == (NOW + delay).isoformat().replace("+00:00", "Z")

    restored = AlertStore.restore(authority, sources, carrier, delivery_adapter=adapter)
    assert restored.snapshot().delivery_attempts == (attempt,)
    assert restored.current("alert-001") == recovered


def test_escalation_route_crash_retry_at_later_time_reuses_first_result(tmp_path):
    adapter, _, authority, sources, carrier, store = escalated(tmp_path)
    intent = store.snapshot().escalation_route_intents[0]
    first_time = NOW + timedelta(seconds=301)
    retry_time = first_time + timedelta(hours=1)
    baseline = adapter.external_effect_count
    carrier.fail_next = "DURING"
    with pytest.raises(AlertStoreError, match="CARRIER_COMMIT_FAILED"):
        store.execute_escalation_route(intent_id=intent.intent_id, now_utc=first_time)
    store.execute_escalation_route(intent_id=intent.intent_id, now_utc=retry_time)
    matching = [
        item
        for item in store.snapshot().delivery_attempts
        if item.escalation_route_intent_id == intent.intent_id
    ]
    assert adapter.external_effect_count == baseline + 1
    assert len(matching) == 1
    assert matching[0].attempted_at_utc == first_time.isoformat().replace("+00:00", "Z")
    restored = AlertStore.restore(authority, sources, carrier, delivery_adapter=adapter)
    assert restored.snapshot().delivery_attempts == store.snapshot().delivery_attempts


def test_same_attempt_id_changed_revision_or_route_still_rejects(tmp_path):
    adapter = Adapter()
    _, _, _, _, store, _ = initial(tmp_path, adapter=adapter)
    store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="stable-id",
        now_utc=NOW,
    )
    with pytest.raises(AlertStoreError, match="DELIVERY_ATTEMPT_REPLAY_CONFLICT"):
        store.request_delivery(
            "alert-001",
            expected_alert_revision=2,
            expected_delivery_revision=1,
            attempt_id="stable-id",
            now_utc=NOW + timedelta(seconds=1),
        )
    with pytest.raises(AlertStoreError, match="DELIVERY_ROUTE_INTENT_REQUIRED"):
        store.request_delivery(
            "alert-001",
            expected_alert_revision=1,
            expected_delivery_revision=0,
            attempt_id="stable-id",
            now_utc=NOW + timedelta(seconds=1),
            route="TRAY_PERSISTENT",
        )


class MutableProjectionCarrier:
    def __init__(self, state):
        snapshot = replace(
            state.alert_store_snapshot,
            current_designations=dict(state.alert_store_snapshot.current_designations),
            dedup_index=dict(state.alert_store_snapshot.dedup_index),
        )
        self.current = snapshot.current_designations
        self.dedup = snapshot.dedup_index
        self.decisions = dict(state.committed_historical_authorization_decisions)
        self.state = AtomicAlertAuthorityState(
            snapshot,
            self.decisions,
            dict(state.committed_historical_source_decisions),
        )
        self.loads = 0

    def load_atomic_state(self):
        self.loads += 1
        return self.state

    def load(self):  # pragma: no cover
        raise AssertionError("separate snapshot read")

    def load_historical_decisions(self):  # pragma: no cover
        raise AssertionError("separate decision read")


def test_restore_defensively_pins_mutable_combined_projection(tmp_path):
    from tests.alerts.test_alert_store import authorized

    auth, authority, sources, carrier, store, item = initial(tmp_path)
    request, proof = authorized(auth, authority, item, 1)
    store.mutate(
        alert_id="alert-001",
        proof=proof,
        request=request,
        expected_alert_revision=1,
        now_utc=NOW,
    )
    mutable = MutableProjectionCarrier(carrier.load_atomic_state())
    restored = AlertStore.restore(authority, sources, mutable)
    pinned = restored.snapshot()
    mutable.current["alert-001"] = 999
    mutable.dedup.clear()
    mutable.decisions.clear()
    assert mutable.loads == 1
    assert restored.snapshot() == pinned
    assert restored.current("alert-001").alert_revision == 2
