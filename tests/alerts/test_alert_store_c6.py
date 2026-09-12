from dataclasses import replace
from datetime import timedelta

import pytest

from bot_core.alerts.store import (
    AlertStore,
    AlertStoreError,
    InMemoryAlertStoreCarrier,
    SuppressionState,
    _delivery_fingerprint,
    _history_id,
    _seal_alert,
    _seal_delivery,
    _seal_history,
    _seal_obligation,
    _source_fingerprint,
    _seed_trusted_source_evidence,
)
from bot_core.security.authentication import _seed_trusted_downstream_operation_definition
from tests.alerts.test_alert_store import Adapter, accept, authorized, evidence, initial
from tests.security.test_authentication import NOW
from tests.security.test_downstream_operation_authorization import definition


def suppress(store, auth, authority, revision=1):
    item = definition("M0.12/ALERT_SET_SUPPRESSION")
    _seed_trusted_downstream_operation_definition(auth, item)
    req, proof = authorized(auth, authority, item, revision)
    return store.mutate(
        alert_id="alert-001",
        proof=proof,
        request=req,
        expected_alert_revision=revision,
        now_utc=NOW,
    )


@pytest.mark.parametrize("severity", ["WARNING", "ERROR"])
def test_suppression_blocks_ordinary_delivery_without_changing_fact(tmp_path, severity):
    adapter = Adapter()
    auth, authority, _, _, store, _ = initial(tmp_path, adapter=adapter, severity=severity)
    before = store.current("alert-001")
    after = suppress(store, auth, authority)
    assert (after.fact_state, after.source_fence, after.severity) == (
        before.fact_state,
        before.source_fence,
        before.severity,
    )
    with pytest.raises(AlertStoreError, match="SUPPRESSED_DELIVERY"):
        store.request_delivery(
            "alert-001",
            expected_alert_revision=2,
            expected_delivery_revision=0,
            attempt_id="blocked",
            now_utc=NOW,
        )
    assert not adapter.results


def test_clear_and_expiry_restore_delivery(tmp_path):
    adapter = Adapter()
    auth, authority, _, _, store, _ = initial(tmp_path, adapter=adapter)
    suppress(store, auth, authority)
    clear = definition("M0.12/ALERT_CLEAR_SUPPRESSION")
    _seed_trusted_downstream_operation_definition(auth, clear)
    req, proof = authorized(auth, authority, clear, 2)
    store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=2, now_utc=NOW
    )
    assert (
        store.request_delivery(
            "alert-001",
            expected_alert_revision=3,
            expected_delivery_revision=0,
            attempt_id="after-clear",
            now_utc=NOW,
        ).delivery.state
        == "DELIVERED"
    )

    auth2, authority2, _, _, other, _ = initial(tmp_path / "expiry", adapter=Adapter())
    suppress(other, auth2, authority2)
    other.expire_suppression(
        "alert-001", expected_alert_revision=2, now_utc=NOW + timedelta(hours=1)
    )
    assert (
        other.request_delivery(
            "alert-001",
            expected_alert_revision=3,
            expected_delivery_revision=0,
            attempt_id="after-expiry",
            now_utc=NOW + timedelta(hours=1),
        ).delivery.state
        == "DELIVERED"
    )


def test_suppressed_critical_keeps_mandatory_routes_executable(tmp_path):
    adapter = Adapter("FAILED")
    auth, authority, _, _, store, _ = initial(tmp_path, adapter=adapter, severity="CRITICAL")
    store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="failure",
        now_utc=NOW,
    )
    suppress(store, auth, authority, 2)
    escalated = store.escalate(
        "alert-001",
        expected_alert_revision=3,
        expected_delivery_revision=1,
        expected_escalation_revision=0,
        now_utc=NOW,
    )
    assert escalated.delivery.escalation_routes == ("IN_APP", "TRAY_PERSISTENT")
    intents = store.snapshot().escalation_route_intents
    assert tuple(x.route for x in intents) == ("IN_APP", "TRAY_PERSISTENT")
    store.execute_escalation_route(intent_id=intents[0].intent_id, now_utc=NOW)


def test_escalation_requires_every_route_executor(tmp_path):
    adapter = Adapter("FAILED")
    *_, store, _ = initial(tmp_path, adapter=adapter, severity="ERROR")
    store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="failure",
        now_utc=NOW,
    )
    store._route_adapters.pop("TRAY_PERSISTENT")
    with pytest.raises(AlertStoreError, match="ESCALATION_ROUTE_UNAVAILABLE"):
        store.escalate(
            "alert-001",
            expected_alert_revision=2,
            expected_delivery_revision=1,
            expected_escalation_revision=0,
            now_utc=NOW + timedelta(seconds=300),
        )


def test_escalation_routes_are_independently_executable_and_restore_exact(tmp_path):
    adapter = Adapter("FAILED")
    _, authority, sources, carrier, store, _ = initial(tmp_path, adapter=adapter, severity="ERROR")
    store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="failure",
        now_utc=NOW,
    )
    escalated = store.escalate(
        "alert-001",
        expected_alert_revision=2,
        expected_delivery_revision=1,
        expected_escalation_revision=0,
        now_utc=NOW + timedelta(seconds=300),
    )
    routes = ("IN_APP", "LOCAL_OS_NOTIFICATION", "TRAY_PERSISTENT")
    assert tuple(item.route for item in store.snapshot().escalation_route_intents) == routes
    for index, intent in enumerate(store.snapshot().escalation_route_intents):
        store.execute_escalation_route(
            intent_id=intent.intent_id, now_utc=NOW + timedelta(seconds=301 + index)
        )
    AlertStore.restore(authority, sources, carrier, delivery_adapter=adapter)
    snap = store.snapshot()
    for intents in (
        snap.escalation_route_intents[:-1],
        snap.escalation_route_intents + (snap.escalation_route_intents[0],),
    ):
        carrier._snapshot = replace(snap, escalation_route_intents=intents)
        with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
            AlertStore.restore(authority, sources, carrier, delivery_adapter=adapter)


def test_suppressed_error_cannot_advance_escalation(tmp_path):
    adapter = Adapter("FAILED")
    auth, authority, _, _, store, _ = initial(tmp_path, adapter=adapter, severity="ERROR")
    store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="failure",
        now_utc=NOW,
    )
    suppress(store, auth, authority, 2)
    with pytest.raises(AlertStoreError, match="SUPPRESSED_DELIVERY"):
        store.escalate(
            "alert-001",
            expected_alert_revision=3,
            expected_delivery_revision=1,
            expected_escalation_revision=0,
            now_utc=NOW + timedelta(seconds=300),
        )


def test_source_replays_and_vector_advancement_are_semantic(tmp_path):
    _, _, sources, _, store, _ = initial(tmp_path)
    initial_snapshot = store.snapshot()
    first_ids = tuple(x.evidence_id for x in sources._accepted.values())
    replay = store.observe(
        alert_id="different",
        evidence_set=__import__(
            "bot_core.alerts.store", fromlist=["SourceEvidenceSet"]
        ).SourceEvidenceSet(first_ids),
        now_utc=NOW,
    )
    assert replay.alert_revision == 1 and store.snapshot() == initial_snapshot
    # A remains current; B advances. Then both advance with B revision reset under a new generation.
    old_a = next(x for x in sources._accepted.values() if x.source_id == "source-a")
    b2 = evidence("source-b", "FAILING", 2)
    _seed_trusted_source_evidence(sources, b2)
    store.observe(
        alert_id="ignored",
        evidence_set=__import__(
            "bot_core.alerts.store", fromlist=["SourceEvidenceSet"]
        ).SourceEvidenceSet((old_a.evidence_id, b2.evidence_id)),
        now_utc=NOW,
    )
    a2 = evidence("source-a", "FAILING", 2)
    _seed_trusted_source_evidence(sources, a2)
    bgen = replace(
        evidence("source-b", "FAILING", 1),
        evidence_id="b-generation-2",
        source_generation=2,
        content_fingerprint_sha256="",
    )
    bgen = replace(bgen, content_fingerprint_sha256=_source_fingerprint(bgen))
    _seed_trusted_source_evidence(sources, bgen)
    latest = store.observe(
        alert_id="ignored",
        evidence_set=__import__(
            "bot_core.alerts.store", fromlist=["SourceEvidenceSet"]
        ).SourceEvidenceSet((a2.evidence_id, bgen.evidence_id)),
        now_utc=NOW,
    )
    assert latest.source_fence == (("source-a", 1, 2), ("source-b", 2, 1))


def test_delivery_replay_survives_later_ack_and_suppression(tmp_path):
    adapter = Adapter()
    auth, authority, _, _, store, ack = initial(tmp_path, adapter=adapter)
    delivered = store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="stable",
        now_utc=NOW,
    )
    req, proof = authorized(auth, authority, ack, 2)
    store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=2, now_utc=NOW
    )
    suppress(store, auth, authority, 3)
    revision = store.snapshot().store_revision
    replay = store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="stable",
        now_utc=NOW,
    )
    assert replay == delivered and store.snapshot().store_revision == revision


def test_coherent_delivery_tamper_rejected_by_external_provenance(tmp_path):
    adapter = Adapter("FAILED")
    auth, authority, sources, carrier, store, _ = initial(tmp_path, adapter=adapter)
    store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="real",
        now_utc=NOW,
    )
    snap = store.snapshot()
    attempt = replace(snap.delivery_attempts[0], destination="evil", content_fingerprint_sha256="")
    attempt = _seal_delivery(attempt)
    after = snap.accepted_revisions[-1]
    changed = _seal_alert(
        replace(
            after,
            delivery=replace(after.delivery, destination="evil"),
            content_fingerprint_sha256="",
        )
    )
    edge = _seal_history(
        replace(
            snap.mutation_history[-1],
            after_fingerprint=changed.content_fingerprint_sha256,
            mutation_id="",
        )
    )
    carrier._snapshot = replace(
        snap,
        accepted_revisions=(snap.accepted_revisions[0], changed),
        mutation_history=(snap.mutation_history[0], edge),
        delivery_attempts=(attempt,),
    )
    with pytest.raises(AlertStoreError, match="DELIVERY_PROVENANCE_INVALID|CONTRACT_INCONSISTENT"):
        AlertStore.restore(authority, sources, carrier, delivery_adapter=adapter)


def test_coherent_delivery_outcome_tamper_rejected_by_external_provenance(tmp_path):
    adapter = Adapter("FAILED")
    _, authority, sources, carrier, store, _ = initial(tmp_path, adapter=adapter)
    store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="real",
        now_utc=NOW,
    )
    snap = store.snapshot()
    attempt = _seal_delivery(
        replace(snap.delivery_attempts[0], outcome="DELIVERED", content_fingerprint_sha256="")
    )
    after = snap.accepted_revisions[-1]
    changed = _seal_alert(
        replace(
            after,
            delivery=replace(after.delivery, state="DELIVERED", failure_count=0),
            content_fingerprint_sha256="",
        )
    )
    edge = _seal_history(
        replace(
            snap.mutation_history[-1],
            after_fingerprint=changed.content_fingerprint_sha256,
            mutation_id="",
        )
    )
    carrier._snapshot = replace(
        snap,
        accepted_revisions=(snap.accepted_revisions[0], changed),
        mutation_history=(snap.mutation_history[0], edge),
        delivery_attempts=(attempt,),
    )
    with pytest.raises(AlertStoreError, match="DELIVERY_PROVENANCE_INVALID"):
        AlertStore.restore(authority, sources, carrier, delivery_adapter=adapter)


@pytest.mark.parametrize(
    "changes",
    [
        {"account_id": "acct_evil", "operator_id": "op_evil", "device_installation_id": "dev_evil"},
        {"causation_id": "evil-cause", "correlation_id": "evil-correlation"},
    ],
)
def test_coherent_audit_tamper_rejected_by_m010_provenance(tmp_path, changes):
    auth, authority, sources, carrier, store, item = initial(tmp_path)
    req, proof = authorized(auth, authority, item, 1)
    store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
    )
    snap = store.snapshot()
    obligation = _seal_obligation(replace(snap.audit_outbox[0], **changes, obligation_id=""))
    edge_changes = {
        key: value for key, value in changes.items() if key in {"causation_id", "correlation_id"}
    }
    edge = _seal_history(
        replace(
            snap.mutation_history[-1],
            audit_obligation_id=obligation.obligation_id,
            **edge_changes,
            mutation_id="",
        )
    )
    carrier._snapshot = replace(
        snap, audit_outbox=(obligation,), mutation_history=(snap.mutation_history[0], edge)
    )
    with pytest.raises(AlertStoreError, match="AUDIT_PROVENANCE_INVALID|CONTRACT_INCONSISTENT"):
        AlertStore.restore(authority, sources, carrier)
