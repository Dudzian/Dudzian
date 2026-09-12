"""S9D-C13 atomic-read and publication-surface regressions."""

from bot_core.alerts.store import (
    AlertStore,
    InMemoryAlertStoreCarrier,
    InMemoryHistoricalAuthorizationDecisionAuthority,
)
from tests.alerts.test_alert_store import accept, authorized, initial
from tests.security.test_authentication import NOW


class RacingAtomicReadCarrier(InMemoryAlertStoreCarrier):
    """Publishes the next complete state immediately after returning the pinned one."""

    def __init__(self, pinned, successor):
        super().__init__()
        self._state = pinned
        self.successor = successor
        self.returned = None

    def load_atomic_state(self):
        with self._lock:
            self.returned = self._state
            self._state = self.successor
            return self.returned

    def load(self):  # pragma: no cover - authoritative restore must never call it
        raise AssertionError("torn snapshot projection read")

    def load_historical_decisions(self):  # pragma: no cover
        raise AssertionError("torn historical projection read")


def test_restore_pins_one_atomic_state_during_concurrent_non_operator_commit(tmp_path):
    decisions = InMemoryHistoricalAuthorizationDecisionAuthority()
    auth, authority, sources, carrier, store, item = initial(
        tmp_path, historical_authorizations=decisions
    )
    request, proof = authorized(auth, authority, item, 1)
    store.mutate(
        alert_id="alert-001",
        proof=proof,
        request=request,
        expected_alert_revision=1,
        now_utc=NOW,
    )
    state_before_redelivery = carrier.load_atomic_state()

    newer = accept(sources, "FAILING", revision=2)
    store.observe(alert_id="alert-001", evidence_set=newer, now_utc=NOW)
    state_after_redelivery = carrier.load_atomic_state()
    assert (
        state_before_redelivery.committed_historical_authorization_decisions
        == state_after_redelivery.committed_historical_authorization_decisions
    )

    racing = RacingAtomicReadCarrier(state_before_redelivery, state_after_redelivery)
    fresh_view = InMemoryHistoricalAuthorizationDecisionAuthority()
    restored = AlertStore.restore(
        authority,
        sources,
        racing,
        historical_authorizations=fresh_view,
    )
    assert restored.snapshot() == racing.returned.alert_store_snapshot
    assert (
        restored.snapshot().store_revision
        == state_before_redelivery.alert_store_snapshot.store_revision
    )
    assert racing._state is state_after_redelivery


def test_fresh_historical_view_is_read_only_public_surface():
    view = InMemoryHistoricalAuthorizationDecisionAuthority()
    public = {name for name in dir(view) if not name.startswith("_")}
    assert public == {"bind_transaction_carrier", "committed_decisions", "fail_next", "resolve"}
    assert not hasattr(view, "commit_combined")
    assert not hasattr(view, "publish")
