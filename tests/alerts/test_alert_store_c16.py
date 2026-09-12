"""S9D-C16 current delivery gates and atomic source provenance regressions."""
from datetime import timedelta

import pytest

from bot_core.alerts.store import AlertStore, AlertStoreError
from bot_core.security.authentication import _seed_trusted_downstream_operation_definition
from bot_core.security.authorization import _seed_trusted_operation_entitlement
from tests.alerts.test_alert_store import Adapter, accept, entitlement, initial
from tests.alerts.test_alert_store_c8 import escalated
from tests.security.test_authentication import NOW, RAW_PIN
from tests.security.test_downstream_operation_authorization import definition, exact_request


def suppress_at(store, auth, authority, revision, when):
    item = definition("M0.12/ALERT_SET_SUPPRESSION")
    _seed_trusted_downstream_operation_definition(auth, item)
    request = exact_request(item, revision=revision)
    proof = auth.issue_authentication_proof(request, RAW_PIN, when)
    _seed_trusted_operation_entitlement(authority, entitlement(request))
    return store.mutate(alert_id="alert-001", proof=proof, request=request,
        expected_alert_revision=revision, now_utc=when)


def test_unconsumed_error_route_obeys_current_suppression_and_expiry(tmp_path):
    adapter, auth, authority, _, _, store = escalated(tmp_path)
    intent = store.snapshot().escalation_route_intents[1]
    suppressed = suppress_at(store, auth, authority, 3, NOW + timedelta(seconds=301))
    calls = adapter.external_effect_count
    with pytest.raises(AlertStoreError, match="SUPPRESSED_DELIVERY"):
        store.execute_escalation_route(intent_id=intent.intent_id,
            now_utc=NOW + timedelta(seconds=302))
    assert adapter.external_effect_count == calls
    expiry = suppressed.suppression.expires_at_utc
    from datetime import datetime
    at_expiry = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
    with pytest.raises(AlertStoreError, match="SUPPRESSION_EXPIRED_RETRY"):
        store.execute_escalation_route(intent_id=intent.intent_id, now_utc=at_expiry)
    assert adapter.external_effect_count == calls
    assert not store.current("alert-001").suppression.suppressed
    store.execute_escalation_route(intent_id=intent.intent_id,
        now_utc=at_expiry + timedelta(seconds=1))
    assert adapter.external_effect_count == calls + 1


def test_historical_effect_finalizes_despite_later_suppression(tmp_path):
    adapter, auth, authority, sources, carrier, store = escalated(tmp_path)
    intent = store.snapshot().escalation_route_intents[1]
    calls = adapter.external_effect_count
    carrier.fail_next = "DURING"
    with pytest.raises(AlertStoreError, match="CARRIER_COMMIT_FAILED"):
        store.execute_escalation_route(intent_id=intent.intent_id,
            now_utc=NOW + timedelta(seconds=301))
    suppress_at(store, auth, authority, 3, NOW + timedelta(seconds=302))
    store.execute_escalation_route(intent_id=intent.intent_id,
        now_utc=NOW + timedelta(seconds=303))
    assert adapter.external_effect_count == calls + 1
    assert AlertStore.restore(authority, sources, carrier, delivery_adapter=adapter).snapshot() == store.snapshot()


def test_critical_first_execution_enforces_current_mandatory_route_floor(tmp_path):
    adapter = Adapter("FAILED")
    auth, authority, _, _, store, _ = initial(
        tmp_path, adapter=adapter, severity="CRITICAL"
    )
    store.request_delivery(
        "alert-001", expected_alert_revision=1, expected_delivery_revision=0,
        attempt_id="critical-failure", now_utc=NOW,
    )
    store.escalate(
        "alert-001", expected_alert_revision=2, expected_delivery_revision=1,
        expected_escalation_revision=0, now_utc=NOW,
    )
    intents = {item.route: item for item in store.snapshot().escalation_route_intents}
    suppress_at(store, auth, authority, 3, NOW + timedelta(seconds=1))
    calls = adapter.external_effect_count
    store.execute_escalation_route(
        intent_id=intents["IN_APP"].intent_id, now_utc=NOW + timedelta(seconds=2)
    )
    store.execute_escalation_route(
        intent_id=intents["TRAY_PERSISTENT"].intent_id,
        now_utc=NOW + timedelta(seconds=3),
    )
    assert adapter.external_effect_count == calls + 2
    with pytest.raises(AlertStoreError, match="SUPPRESSED_DELIVERY"):
        store.execute_escalation_route(
            intent_id=intents["OPERATOR_ATTENTION_REQUIRED"].intent_id,
            now_utc=NOW + timedelta(seconds=4),
        )
    assert adapter.external_effect_count == calls + 2


def test_source_replay_never_rewrites_atomic_historical_time(tmp_path):
    _, authority, sources, carrier, store, _ = initial(tmp_path)
    evidence = accept(sources, "FAILING", revision=2)
    committed = store.observe(alert_id="alert-001", evidence_set=evidence,
        now_utc=NOW + timedelta(seconds=1))
    before = carrier.load_atomic_state()
    replay = store.observe(alert_id="alert-001", evidence_set=evidence,
        now_utc=NOW + timedelta(hours=1))
    after = carrier.load_atomic_state()
    assert replay == committed
    assert before == after
    assert AlertStore.restore(authority, sources, carrier).snapshot() == store.snapshot()


def test_failed_source_commit_publishes_no_historical_source_decision(tmp_path):
    _, _, sources, carrier, store, _ = initial(tmp_path)
    evidence = accept(sources, "HEALTHY", revision=2)
    before = carrier.load_atomic_state()
    carrier.fail_next = "DURING"
    with pytest.raises(AlertStoreError, match="CARRIER_COMMIT_FAILED"):
        store.observe(alert_id="alert-001", evidence_set=evidence,
            now_utc=NOW + timedelta(seconds=1))
    assert carrier.load_atomic_state() == before
