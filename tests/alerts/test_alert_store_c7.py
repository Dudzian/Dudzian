from dataclasses import replace
from datetime import timedelta

import pytest

from bot_core.alerts.store import (
    AlertStore,
    AlertStoreError,
    InMemoryHistoricalAuthorizationDecisionAuthority,
    SourceEvidenceSet,
    _route_intent_id,
    _seal_alert,
    _seal_history,
    _seal_obligation,
)
from bot_core.security.authorization import AuthorizationAuthority
from tests.alerts.test_alert_store import Adapter, accept, authorized, initial
from tests.security.test_authentication import NOW, prepared


def committed_ack(tmp_path, historical):
    auth, authority, sources, carrier, store, item = initial(
        tmp_path, historical_authorizations=historical
    )
    req, proof = authorized(auth, authority, item, 1)
    store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
    )
    return sources, carrier, store


def restarted_authorization(tmp_path):
    _, authentication, _ = prepared(tmp_path)
    assert not authentication.snapshot.accepted_authentication_proofs
    return AuthorizationAuthority(authentication)


def test_historical_audit_restores_after_ephemeral_m010_restart(tmp_path):
    historical = InMemoryHistoricalAuthorizationDecisionAuthority()
    sources, carrier, store = committed_ack(tmp_path / "live", historical)
    restarted = restarted_authorization(tmp_path / "restart")
    restored = AlertStore.restore(restarted, sources, carrier, historical_authorizations=historical)
    assert restored.current("alert-001") == store.current("alert-001")


def test_coherent_audit_tamper_rejected_after_m010_restart(tmp_path):
    historical = InMemoryHistoricalAuthorizationDecisionAuthority()
    sources, carrier, store = committed_ack(tmp_path / "live", historical)
    snap = store.snapshot()
    obligation = _seal_obligation(
        replace(
            snap.audit_outbox[0],
            account_id="acct_evil",
            operator_id="op_evil",
            device_installation_id="dev_evil",
            obligation_id="",
        )
    )
    edge = _seal_history(
        replace(
            snap.mutation_history[-1], audit_obligation_id=obligation.obligation_id, mutation_id=""
        )
    )
    carrier._snapshot = replace(
        snap, audit_outbox=(obligation,), mutation_history=(snap.mutation_history[0], edge)
    )
    with pytest.raises(AlertStoreError, match="AUDIT_PROVENANCE_INVALID|CONTRACT_INCONSISTENT"):
        AlertStore.restore(
            restarted_authorization(tmp_path / "restart"),
            sources,
            carrier,
            historical_authorizations=historical,
        )


@pytest.mark.parametrize(
    "fence",
    [
        (("source-a", 999, 999), ("source-b", 999, 999)),
        (("source-a", 1, 1),),
        (("source-a", 1, 1), ("source-a", 1, 1)),
        (("source-a", 1, 1), ("source-b", 1, 1), ("source-c", 1, 1)),
        (("source-a", 0, 1), ("source-b", 1, 1)),
        (("source-a", 2, -100), ("source-b", 1, 1)),
    ],
)
def test_restore_binds_exact_closed_source_fence(tmp_path, fence):
    _, authority, sources, carrier, store, _ = initial(tmp_path)
    snap = store.snapshot()
    forged = _seal_alert(
        replace(snap.accepted_revisions[0], source_fence=fence, content_fingerprint_sha256="")
    )
    edge = _seal_history(
        replace(
            snap.mutation_history[0],
            after_fingerprint=forged.content_fingerprint_sha256,
            mutation_id="",
        )
    )
    carrier._snapshot = replace(snap, accepted_revisions=(forged,), mutation_history=(edge,))
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_UNACCEPTED|CONTRACT_INCONSISTENT"):
        AlertStore.restore(authority, sources, carrier)


def test_reordered_source_set_and_replay_after_ack_return_current(tmp_path):
    historical = InMemoryHistoricalAuthorizationDecisionAuthority()
    auth, authority, sources, _, store, item = initial(
        tmp_path, historical_authorizations=historical
    )
    ids = tuple(sources._accepted)
    before = store.snapshot()
    assert (
        store.observe(
            alert_id="ignored", evidence_set=SourceEvidenceSet(tuple(reversed(ids))), now_utc=NOW
        ).alert_revision
        == 1
    )
    assert store.snapshot() == before
    req, proof = authorized(auth, authority, item, 1)
    acknowledged = store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
    )
    revision = store.snapshot().store_revision
    replay = store.observe(alert_id="ignored", evidence_set=SourceEvidenceSet(ids), now_utc=NOW)
    assert replay == acknowledged and replay.alert_revision == 2
    assert store.snapshot().store_revision == revision


def escalated(tmp_path):
    adapter = Adapter("FAILED")
    _, authority, sources, carrier, store, _ = initial(tmp_path, adapter=adapter, severity="ERROR")
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
    return adapter, authority, sources, carrier, store


def test_route_intent_is_consumed_once_and_arbitrary_attempt_rejected(tmp_path):
    _, _, _, _, store = escalated(tmp_path)
    intent = next(
        item
        for item in store.snapshot().escalation_route_intents
        if item.route == "TRAY_PERSISTENT"
    )
    with pytest.raises(AlertStoreError, match="DELIVERY_ROUTE_INTENT_REQUIRED"):
        store.request_delivery(
            "alert-001",
            expected_alert_revision=3,
            expected_delivery_revision=1,
            attempt_id="caller-invented",
            now_utc=NOW,
            route="TRAY_PERSISTENT",
            escalation_route_intent_id=intent.intent_id,
        )
    first = store.execute_escalation_route(
        intent_id=intent.intent_id, now_utc=NOW + timedelta(seconds=301)
    )
    revision = store.snapshot().store_revision
    retry = store.execute_escalation_route(
        intent_id=intent.intent_id, now_utc=NOW + timedelta(seconds=301)
    )
    assert retry == first and store.snapshot().store_revision == revision


def test_route_intent_attempt_tamper_and_missing_attempt_link_rejected(tmp_path):
    adapter, authority, sources, carrier, store = escalated(tmp_path)
    snap = store.snapshot()
    original = snap.escalation_route_intents[0]
    unchanged_id = replace(original, attempt_id="evil")
    carrier._snapshot = replace(
        snap, escalation_route_intents=(unchanged_id,) + snap.escalation_route_intents[1:]
    )
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.restore(authority, sources, carrier, delivery_adapter=adapter)
    resealed = replace(unchanged_id, intent_id="")
    resealed = replace(resealed, intent_id=_route_intent_id(resealed))
    carrier._snapshot = replace(
        snap, escalation_route_intents=(resealed,) + snap.escalation_route_intents[1:]
    )
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.restore(authority, sources, carrier, delivery_adapter=adapter)

    carrier._snapshot = snap
    result = store.execute_escalation_route(
        intent_id=original.intent_id, now_utc=NOW + timedelta(seconds=301)
    )
    committed = store.snapshot()
    attempt = committed.delivery_attempts[-1]
    carrier._snapshot = replace(
        committed,
        delivery_attempts=committed.delivery_attempts[:-1]
        + (replace(attempt, escalation_route_intent_id=None),),
    )
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.restore(authority, sources, carrier, delivery_adapter=adapter)
    assert result.delivery.channel == original.route
    assert result.delivery.destination == original.destination


def test_resolved_alert_rejects_unconsumed_pre_resolution_intent(tmp_path):
    _, _, sources, _, store = escalated(tmp_path)
    intent = store.snapshot().escalation_route_intents[-1]
    healthy = accept(sources, "HEALTHY", 2, severity="ERROR")
    store.observe(alert_id="ignored", evidence_set=healthy, now_utc=NOW + timedelta(seconds=300))
    with pytest.raises(AlertStoreError, match="ALERT_RESOLVED"):
        store.execute_escalation_route(
            intent_id=intent.intent_id, now_utc=NOW + timedelta(seconds=302)
        )
