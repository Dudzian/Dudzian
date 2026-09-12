from dataclasses import replace
from datetime import timedelta

import pytest

from bot_core.alerts.store import (
    AlertStore,
    AlertStoreError,
    InMemoryAlertStoreCarrier,
    InMemoryHistoricalAuthorizationDecisionAuthority,
    InMemorySourceEvidenceAuthority,
    SourceEvidenceSet,
    SourceSelector,
    _source_fingerprint,
    _seed_trusted_source_evidence,
)
from bot_core.security.authentication import _seed_trusted_downstream_operation_definition
from bot_core.security.authorization import _seed_trusted_operation_entitlement
from tests.alerts.test_alert_store import (
    Adapter,
    accept,
    authorized,
    entitlement,
    evidence,
    initial,
    source_authority,
)
from tests.alerts.test_alert_store_c6 import suppress
from tests.security.test_authentication import NOW, RAW_PIN
from tests.security.test_downstream_operation_authorization import (
    arranged,
    definition,
    exact_request,
)


def test_owner_rejects_cross_environment_before_provenance_or_commit(tmp_path):
    historical = InMemoryHistoricalAuthorizationDecisionAuthority()
    auth, authority, _, carrier, store, item = initial(
        tmp_path, historical_authorizations=historical
    )
    wrong = exact_request(
        item,
        revision=1,
        environment="TESTNET",
        intent=item.declared_intent,
        causation="cross-env",
        correlation="cross-env",
    )
    proof = auth.issue_authentication_proof(wrong, RAW_PIN, NOW)
    _seed_trusted_operation_entitlement(authority, entitlement(wrong))
    before = store.snapshot()
    with pytest.raises(AlertStoreError, match="AUTHORIZATION_ENVIRONMENT_MISMATCH"):
        store.mutate(
            alert_id="alert-001", proof=proof, request=wrong, expected_alert_revision=1, now_utc=NOW
        )
    assert store.snapshot() == carrier.load() == before
    assert not historical._decisions
    right, right_proof = authorized(auth, authority, item, 1)
    assert store.mutate(
        alert_id="alert-001",
        proof=right_proof,
        request=right,
        expected_alert_revision=1,
        now_utc=NOW,
    ).acknowledged


def escalated(tmp_path, *, carrier=None):
    adapter = Adapter("FAILED")
    auth, authority, sources, carrier, store, _ = initial(
        tmp_path, adapter=adapter, severity="ERROR", carrier=carrier
    )
    store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="initial",
        now_utc=NOW,
    )
    store.escalate(
        "alert-001",
        expected_alert_revision=2,
        expected_delivery_revision=1,
        expected_escalation_revision=0,
        now_utc=NOW + timedelta(seconds=300),
    )
    return adapter, auth, authority, sources, carrier, store


def test_ordinary_delivery_cannot_bypass_current_route_intents(tmp_path):
    adapter, _, _, _, _, store = escalated(tmp_path)
    calls = adapter.external_effect_count
    with pytest.raises(AlertStoreError, match="DELIVERY_ROUTE_INTENT_REQUIRED"):
        store.request_delivery(
            "alert-001",
            expected_alert_revision=3,
            expected_delivery_revision=1,
            attempt_id="bypass",
            now_utc=NOW + timedelta(seconds=301),
        )
    assert adapter.external_effect_count == calls
    intent = next(x for x in store.snapshot().escalation_route_intents if x.route == "IN_APP")
    store.execute_escalation_route(intent_id=intent.intent_id, now_utc=NOW + timedelta(seconds=301))
    assert (
        sum(
            x.escalation_route_intent_id == intent.intent_id
            for x in store.snapshot().delivery_attempts
        )
        == 1
    )


def test_superseded_route_intent_rejected_before_adapter(tmp_path):
    adapter, _, _, _, _, store = escalated(tmp_path)
    old = next(x for x in store.snapshot().escalation_route_intents if x.route == "TRAY_PERSISTENT")
    first = next(x for x in store.snapshot().escalation_route_intents if x.route == "IN_APP")
    store.execute_escalation_route(intent_id=first.intent_id, now_utc=NOW + timedelta(seconds=301))
    store.escalate(
        "alert-001",
        expected_alert_revision=4,
        expected_delivery_revision=2,
        expected_escalation_revision=1,
        now_utc=NOW + timedelta(seconds=601),
    )
    calls = adapter.external_effect_count
    before = store.snapshot()
    with pytest.raises(AlertStoreError, match="STALE_ESCALATION_ROUTE_INTENT"):
        store.execute_escalation_route(
            intent_id=old.intent_id, now_utc=NOW + timedelta(seconds=602)
        )
    assert adapter.external_effect_count == calls and store.snapshot() == before


def test_idempotent_adapter_closes_carrier_failure_window(tmp_path):
    carrier = InMemoryAlertStoreCarrier()
    adapter = Adapter()
    _, _, _, _, store, _ = initial(tmp_path, adapter=adapter, carrier=carrier)
    carrier.fail_next = "DURING"
    with pytest.raises(AlertStoreError, match="CARRIER_COMMIT_FAILED"):
        store.request_delivery(
            "alert-001",
            expected_alert_revision=1,
            expected_delivery_revision=0,
            attempt_id="crash-safe",
            now_utc=NOW,
        )
    assert adapter.external_effect_count == 1 and not store.snapshot().delivery_attempts
    store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="crash-safe",
        now_utc=NOW,
    )
    assert adapter.external_effect_count == 1 and len(store.snapshot().delivery_attempts) == 1
    with pytest.raises(AlertStoreError, match="REPLAY_CONFLICT"):
        store.request_delivery(
            "alert-001",
            expected_alert_revision=2,
            expected_delivery_revision=1,
            attempt_id="crash-safe",
            now_utc=NOW,
        )


def test_escalation_route_retry_is_crash_safe(tmp_path):
    adapter, _, _, _, carrier, store = escalated(tmp_path)
    intent = store.snapshot().escalation_route_intents[0]
    baseline = adapter.external_effect_count
    carrier.fail_next = "DURING"
    with pytest.raises(AlertStoreError, match="CARRIER_COMMIT_FAILED"):
        store.execute_escalation_route(
            intent_id=intent.intent_id, now_utc=NOW + timedelta(seconds=301)
        )
    assert adapter.external_effect_count == baseline + 1
    store.execute_escalation_route(intent_id=intent.intent_id, now_utc=NOW + timedelta(seconds=301))
    assert adapter.external_effect_count == baseline + 1
    assert (
        sum(
            attempt.escalation_route_intent_id == intent.intent_id
            for attempt in store.snapshot().delivery_attempts
        )
        == 1
    )


def test_delivery_preflight_time_and_expiry_materialization(tmp_path):
    adapter = Adapter()
    auth, authority, _, _, store, _ = initial(tmp_path, adapter=adapter)
    suppress(store, auth, authority)
    calls = adapter.external_effect_count
    with pytest.raises(AlertStoreError, match="TIME_ROLLBACK"):
        store.request_delivery(
            "alert-001",
            expected_alert_revision=2,
            expected_delivery_revision=0,
            attempt_id="past",
            now_utc=NOW - timedelta(seconds=1),
        )
    assert adapter.external_effect_count == calls
    with pytest.raises(AlertStoreError, match="SUPPRESSION_EXPIRED_RETRY"):
        store.request_delivery(
            "alert-001",
            expected_alert_revision=2,
            expected_delivery_revision=0,
            attempt_id="expiry",
            now_utc=NOW + timedelta(hours=1),
        )
    assert store.current("alert-001").alert_revision == 3
    assert store.snapshot().mutation_history[-1].mutation_type == "UNSUPPRESS_EXPIRED"
    store.request_delivery(
        "alert-001",
        expected_alert_revision=3,
        expected_delivery_revision=0,
        attempt_id="expiry",
        now_utc=NOW + timedelta(hours=1),
    )


@pytest.mark.parametrize("operation", ["M0.12/ALERT_ACKNOWLEDGE", "M0.12/ALERT_SET_SUPPRESSION"])
def test_operator_exact_replay_is_atomic_noop(tmp_path, operation):
    auth, authority, _, _, store, item = initial(tmp_path, operation)
    req, proof = authorized(auth, authority, item, 1)
    result = store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
    )
    before = store.snapshot()
    replay = store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
    )
    assert replay == result and store.snapshot() == before
    assert len(before.operator_replays) == len(before.audit_outbox) == 1


def test_conflicting_operator_replay_identity_rejected(tmp_path):
    auth, authority, _, _, store, ack = initial(tmp_path)
    req, proof = authorized(auth, authority, ack, 1)
    store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
    )
    suppression = definition("M0.12/ALERT_SET_SUPPRESSION")
    _seed_trusted_downstream_operation_definition(auth, suppression)
    conflicting = exact_request(
        suppression,
        revision=2,
        intent=suppression.declared_intent,
        causation=req.causation_id,
        correlation=req.correlation_id,
    )
    proof2 = auth.issue_authentication_proof(conflicting, RAW_PIN, NOW)
    _seed_trusted_operation_entitlement(authority, entitlement(conflicting))
    with pytest.raises(AlertStoreError, match="OPERATOR_REPLAY_CONFLICT"):
        store.mutate(
            alert_id="alert-001",
            proof=proof2,
            request=conflicting,
            expected_alert_revision=2,
            now_utc=NOW,
        )


def test_clear_and_manual_resolution_exact_replays_are_noops(tmp_path):
    auth, authority, _, _, store, _ = initial(tmp_path / "clear")
    suppress(store, auth, authority)
    clear = definition("M0.12/ALERT_CLEAR_SUPPRESSION")
    _seed_trusted_downstream_operation_definition(auth, clear)
    req, proof = authorized(auth, authority, clear, 2)
    result = store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=2, now_utc=NOW
    )
    before = store.snapshot()
    assert (
        store.mutate(
            alert_id="alert-001", proof=proof, request=req, expected_alert_revision=2, now_utc=NOW
        )
        == result
    )
    assert store.snapshot() == before

    manual = definition("M0.12/ALERT_MANUAL_FACT_RESOLUTION")
    auth2, authority2, _, _, _ = arranged(tmp_path / "manual", manual.operation)
    sources = source_authority(alert_type="OPERATOR_WORKFLOW_REQUIRED")
    failing = accept(sources, "FAILING", alert_type="OPERATOR_WORKFLOW_REQUIRED")
    manual_store = AlertStore(authority2, sources, InMemoryAlertStoreCarrier())
    manual_store.observe(alert_id="alert-001", evidence_set=failing, now_utc=NOW)
    req2, proof2 = authorized(auth2, authority2, manual, 1)
    resolved = manual_store.mutate(
        alert_id="alert-001", proof=proof2, request=req2, expected_alert_revision=1, now_utc=NOW
    )
    snapshot = manual_store.snapshot()
    assert (
        manual_store.mutate(
            alert_id="alert-001", proof=proof2, request=req2, expected_alert_revision=1, now_utc=NOW
        )
        == resolved
    )
    assert manual_store.snapshot() == snapshot


def test_source_replay_after_freshness_and_redelivery_fields(tmp_path):
    auth, authority, sources, _, store, item = initial(tmp_path)
    initial_ids = tuple(sources._accepted)
    req, proof = authorized(auth, authority, item, 1)
    acknowledged = store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
    )
    before = store.snapshot()
    replay = store.observe(
        alert_id="ignored",
        evidence_set=SourceEvidenceSet(initial_ids),
        now_utc=NOW + timedelta(hours=1),
    )
    assert replay == acknowledged and store.snapshot() == before

    values = []
    for source in ("source-a", "source-b"):
        value = evidence(source, "FAILING", 2)
        value = replace(
            value,
            observed_at_utc=(NOW + timedelta(seconds=10)).isoformat().replace("+00:00", "Z"),
            content_fingerprint_sha256="",
        )
        value = replace(value, content_fingerprint_sha256=_source_fingerprint(value))
        _seed_trusted_source_evidence(sources, value)
        values.append(value)
    updated = store.observe(
        alert_id="ignored",
        evidence_set=SourceEvidenceSet(tuple(x.evidence_id for x in values)),
        now_utc=NOW + timedelta(seconds=10),
    )
    assert updated.occurrence_count == 2
    assert updated.last_seen_at_utc == (NOW + timedelta(seconds=10)).isoformat().replace(
        "+00:00", "Z"
    )
    assert updated.raised_at_utc == acknowledged.raised_at_utc
    again = store.observe(
        alert_id="ignored",
        evidence_set=SourceEvidenceSet(tuple(reversed([x.evidence_id for x in values]))),
        now_utc=NOW + timedelta(seconds=11),
    )
    assert again == updated and again.occurrence_count == 2


def test_selector_duplicate_sources_fail_at_construction():
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        InMemorySourceEvidenceAuthority(
            (
                SourceSelector(
                    "TYPE", "OBSERVATION", "PAPER", "scope", "FACT", ("same", "same"), True, "ERROR"
                ),
            )
        )
