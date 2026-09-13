"""S9D-C22 same-semantic whole-second nonretroactivity regressions."""

from datetime import datetime, timezone

import pytest

from bot_core.alerts.s9c_source import S9CObservationSourceAuthority
from bot_core.alerts.store import AlertStore, SourceEvidenceSet
from tests.alerts.test_s9c_source_c20 import RUN_B, _raw, _system


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
def test_later_same_second_runtime_cannot_invalidate_durable_source_decision(category):
    authority, publisher, adapter, alert_carrier, store = _system()
    first = publisher.publish(_raw(category, condition="BLOCKED", second=10),
                              now_utc="2030-01-01T00:00:10Z")
    alert = store.observe(
        alert_id="alert", evidence_set=SourceEvidenceSet((first.acceptance_id,)),
        now_utc=datetime(2030, 1, 1, 0, 0, 10, tzinfo=timezone.utc),
    )
    source_before = authority._carrier.read()  # noqa: SLF001
    alert_before = alert_carrier.load_atomic_state()
    with pytest.raises(ValueError, match="SEMANTIC_TRANSACTION_TIME_COLLISION"):
        publisher.publish(_raw(category, condition="OK", run=RUN_B, second=10),
                          now_utc="2030-01-01T00:00:10Z")
    assert authority._carrier.read() == source_before  # noqa: SLF001
    assert alert_carrier.load_atomic_state() == alert_before
    decision = next(iter(alert_before.committed_historical_source_decisions.values()))
    assert adapter.validates_historical_decision(decision)
    restored = AlertStore.restore(
        object(), S9CObservationSourceAuthority(authority), alert_carrier
    )
    assert restored.current(alert.alert_id).source_evidence_reference == first.acceptance_id
