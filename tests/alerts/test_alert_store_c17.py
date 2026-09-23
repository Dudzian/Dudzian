"""S9D-C17 cache-independent historical source restore tests."""

from datetime import timedelta

from bot_core.alerts.store import AlertStore, SourceEvidenceSet
from tests.alerts.test_alert_store import accept, initial, source_authority
from tests.security.test_authentication import NOW


def test_fresh_source_authority_restores_without_live_validation_caches(tmp_path):
    _, authority, _, carrier, store, _ = initial(tmp_path)
    fresh = source_authority()
    accept(fresh, "FAILING", revision=1)
    assert not fresh._validated_references
    assert not fresh._reference_sets
    restored = AlertStore.restore(authority, fresh, carrier)
    assert restored.snapshot() == store.snapshot()


def test_newer_current_evidence_does_not_invalidate_committed_history(tmp_path):
    _, authority, _, carrier, store, _ = initial(tmp_path)
    fresh = source_authority()
    accept(fresh, "FAILING", revision=1)
    accept(fresh, "HEALTHY", revision=2)
    assert not fresh._validated_references
    assert not fresh._reference_sets
    restored = AlertStore.restore(authority, fresh, carrier)
    assert restored.snapshot() == store.snapshot()
    assert restored.current("alert-001").fact_state == "FAILING"


def test_exact_replay_after_freshness_does_not_change_historical_decision(tmp_path):
    _, _, sources, carrier, store, _ = initial(tmp_path)
    before = carrier.load_atomic_state().committed_historical_source_decisions
    decision = next(iter(before.values()))
    evidence_set = SourceEvidenceSet(decision.evidence_ids)
    store.observe(
        alert_id="alert-001",
        evidence_set=evidence_set,
        now_utc=NOW + timedelta(hours=1),
    )
    assert carrier.load_atomic_state().committed_historical_source_decisions == before
