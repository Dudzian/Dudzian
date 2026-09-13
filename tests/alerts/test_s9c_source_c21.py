"""S9D-C21 nonretroactive S9C/AlertStore transaction-time regressions."""

from datetime import datetime, timezone

import pytest

from bot_core.alerts.s9c_source import S9CObservationSourceAuthority
from bot_core.alerts.store import AlertStore, AlertStoreError, SourceEvidenceSet
from tests.alerts.test_s9c_source_c20 import RUN_B, _raw, _system


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
def test_backdated_later_publication_cannot_invalidate_durable_decision(category):
    authority, publisher, adapter, carrier, store = _system()
    first = publisher.publish(_raw(category, condition="BLOCKED", second=10),
                              now_utc="2030-01-01T00:00:10Z")
    store.observe(alert_id="alert", evidence_set=SourceEvidenceSet((first.acceptance_id,)),
                  now_utc=datetime(2030, 1, 1, 0, 0, 10, tzinfo=timezone.utc))
    before = authority._carrier.read()  # noqa: SLF001 - assert authoritative carrier atomicity
    with pytest.raises(ValueError, match="TRANSACTION_TIME_ROLLBACK"):
        publisher.publish(_raw(category, condition="OK", run=RUN_B, second=5),
                          now_utc="2030-01-01T00:00:05Z")
    assert authority._carrier.read() == before  # noqa: SLF001
    restored = AlertStore.restore(object(), S9CObservationSourceAuthority(authority), carrier)
    assert restored.current("alert").source_evidence_reference == first.acceptance_id
    assert adapter.validates_reference(first.acceptance_id)


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
def test_new_source_edge_cannot_precede_acceptance_or_current_alert(category):
    _, publisher, _, carrier, store = _system()
    first = publisher.publish(_raw(category, condition="BLOCKED", second=10),
                              now_utc="2030-01-01T00:00:10Z")
    with pytest.raises(AlertStoreError, match="TIME_ROLLBACK"):
        store.observe(alert_id="alert", evidence_set=SourceEvidenceSet((first.acceptance_id,)),
                      now_utc=datetime(2030, 1, 1, 0, 0, 9, tzinfo=timezone.utc))
    assert carrier.load().store_revision == 0
    assert not carrier.load_atomic_state().committed_historical_source_decisions

    store.observe(alert_id="alert", evidence_set=SourceEvidenceSet((first.acceptance_id,)),
                  now_utc=datetime(2030, 1, 1, 0, 0, 10, tzinfo=timezone.utc))
    successor = publisher.publish(_raw(category, condition="BLOCKED", run=RUN_B, second=11),
                                  now_utc="2030-01-01T00:00:11Z")
    before = carrier.load().store_revision
    with pytest.raises(AlertStoreError, match="TIME_ROLLBACK"):
        store.observe(alert_id="ignored", evidence_set=SourceEvidenceSet((successor.acceptance_id,)),
                      now_utc=datetime(2030, 1, 1, 0, 0, 9, tzinfo=timezone.utc))
    assert carrier.load().store_revision == before
