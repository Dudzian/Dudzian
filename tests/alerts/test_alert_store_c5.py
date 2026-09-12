from dataclasses import replace
from datetime import timedelta
from threading import Barrier, Thread

import pytest

from bot_core.alerts.store import (
    AlertStore,
    AlertStoreError,
    DeliveryState,
    InMemoryAlertStoreCarrier,
    InMemorySourceEvidenceAuthority,
    MutationHistoryEntry,
    SourceEvidenceSet,
    SuppressionState,
    _history_id,
    _seal_alert,
    _seal_history,
    _seed_trusted_source_evidence,
)
from bot_core.security.authentication import _seed_trusted_downstream_operation_definition
from tests.alerts.test_alert_store import (
    Adapter,
    accept,
    authorized,
    evidence,
    initial,
    source_authority,
)
from tests.security.test_downstream_operation_authorization import arranged, definition

from tests.security.test_authentication import NOW


def test_condition_currentness_is_independent_and_mixed_set_rejected(tmp_path):
    sources = source_authority()
    a1 = evidence("source-a", "FAILING", condition="order-123")
    b1 = evidence("source-b", "FAILING", condition="order-123")
    a2 = evidence("source-a", "FAILING", condition="order-456")
    b2 = evidence("source-b", "FAILING", condition="order-456")
    for item in (a1, b1, a2, b2):
        _seed_trusted_source_evidence(sources, item)
    assert sources.validate_current(SourceEvidenceSet((a1.evidence_id, b1.evidence_id)), NOW)
    assert sources.validate_current(SourceEvidenceSet((a2.evidence_id, b2.evidence_id)), NOW)
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_CONDITION_MISMATCH"):
        sources.validate_current(SourceEvidenceSet((a1.evidence_id, b2.evidence_id)), NOW)


def test_source_currentness_lock_is_held_through_carrier_commit(tmp_path):
    entered, release = Barrier(2), Barrier(2)

    class RacingAuthority(InMemorySourceEvidenceAuthority):
        armed = False

        def consume_current(self, evidence_set, now_utc, consumer):
            with self._lock:
                fact = self._validate_current_locked(evidence_set, now_utc)
                if self.armed:
                    entered.wait()
                    release.wait()
                return consumer(fact)

    base = source_authority()
    sources = RacingAuthority(tuple(base._selectors.values()))
    failing = accept(sources, "FAILING", 1)
    auth, authority, _, _, _ = __import__(
        "tests.security.test_downstream_operation_authorization", fromlist=["arranged"]
    ).arranged(tmp_path)
    store = AlertStore(authority, sources, InMemoryAlertStoreCarrier())
    store.observe(alert_id="alert", evidence_set=failing, now_utc=NOW)
    healthy = accept(sources, "HEALTHY", 2)
    sources.armed = True
    result = []
    t1 = Thread(
        target=lambda: result.append(
            store.observe(alert_id="alert", evidence_set=healthy, now_utc=NOW)
        )
    )
    t1.start()
    entered.wait()
    newer = tuple(evidence(s, "FAILING", 3) for s in ("source-a", "source-b"))
    published = []
    t2 = Thread(
        target=lambda: (
            [_seed_trusted_source_evidence(sources, x) for x in newer],
            published.append(True),
        )
    )
    t2.start()
    assert not published  # publisher is fenced behind the resolving transaction
    release.wait()
    t1.join()
    t2.join()
    assert result[0].fact_state == "HEALTHY" and published


@pytest.mark.parametrize("start,next_severity", [("ERROR", "INFO"), ("CRITICAL", "WARNING")])
def test_source_policy_never_allows_severity_downgrade(tmp_path, start, next_severity):
    _, _, sources, _, store, _ = initial(tmp_path, severity=start)
    newer = accept(sources, "FAILING", 2, severity=next_severity)
    with pytest.raises(AlertStoreError, match="SOURCE_SEVERITY_POLICY_MISMATCH|SEVERITY_DOWNGRADE"):
        store.observe(alert_id="alert-001", evidence_set=newer, now_utc=NOW)


@pytest.mark.parametrize(
    "severity,denied,allowed,routes",
    [
        ("WARNING", (299, 300, 899), 900, ("IN_APP", "LOCAL_OS_NOTIFICATION")),
        ("ERROR", (299,), 300, ("IN_APP", "LOCAL_OS_NOTIFICATION", "TRAY_PERSISTENT")),
        ("CRITICAL", (), 0, ("IN_APP", "TRAY_PERSISTENT", "OPERATOR_ATTENTION_REQUIRED")),
    ],
)
def test_exact_per_severity_escalation_policy(tmp_path, severity, denied, allowed, routes):
    *_, store, _ = initial(tmp_path, adapter=Adapter("FAILED"), severity=severity)
    failed = store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="attempt",
        now_utc=NOW,
    )
    for seconds in denied:
        with pytest.raises(AlertStoreError, match="ESCALATION_POLICY_NOT_SATISFIED"):
            store.escalate(
                "alert-001",
                expected_alert_revision=2,
                expected_delivery_revision=1,
                expected_escalation_revision=0,
                now_utc=NOW + timedelta(seconds=seconds),
            )
    result = store.escalate(
        "alert-001",
        expected_alert_revision=2,
        expected_delivery_revision=1,
        expected_escalation_revision=0,
        now_utc=NOW + timedelta(seconds=allowed),
    )
    assert result.delivery.escalation_routes == routes


def test_ack_after_automatic_resolution_remains_terminal(tmp_path):
    auth, authority, sources, _, store, item = initial(tmp_path)
    healthy = accept(sources, "HEALTHY", 2)
    resolved = store.observe(alert_id="alert-001", evidence_set=healthy, now_utc=NOW)
    req, proof = authorized(auth, authority, item, 2)
    after = store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=2, now_utc=NOW
    )
    assert (after.lifecycle_state, after.fact_state, after.resolution_mode) == (
        "RESOLVED",
        "HEALTHY",
        "AUTOMATIC_FACT",
    )


def test_ack_after_manual_resolution_remains_terminal(tmp_path):
    auth, authority, manual, _, _ = arranged(tmp_path, "M0.12/ALERT_MANUAL_FACT_RESOLUTION")
    sources = source_authority(alert_type="OPERATOR_WORKFLOW_REQUIRED")
    failing = accept(sources, "FAILING", alert_type="OPERATOR_WORKFLOW_REQUIRED")
    store = AlertStore(authority, sources, InMemoryAlertStoreCarrier())
    store.observe(alert_id="alert-001", evidence_set=failing, now_utc=NOW)
    req, proof = authorized(auth, authority, manual, 1)
    resolved = store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
    )
    ack = definition("M0.12/ALERT_ACKNOWLEDGE")
    _seed_trusted_downstream_operation_definition(auth, ack)
    req2, proof2 = authorized(auth, authority, ack, 2)
    after = store.mutate(
        alert_id="alert-001", proof=proof2, request=req2, expected_alert_revision=2, now_utc=NOW
    )
    assert resolved.lifecycle_state == after.lifecycle_state == "RESOLVED"
    assert after.fact_state == "MANUALLY_RESOLVED"


@pytest.mark.parametrize(
    "change",
    [
        {"lifecycle_state": "RESOLVED"},
        {"acknowledged": True, "acknowledged_at_utc": "2025-01-01T00:00:00Z"},
        {
            "suppression": SuppressionState(
                True,
                "account-alerts",
                "2025-01-01T00:00:00Z",
                "2025-01-01T01:00:00Z",
                "OPERATOR_POLICY",
                1,
            )
        },
        {
            "delivery": DeliveryState(
                state="DELIVERED", last_transition_at_utc="2025-01-01T00:00:00Z"
            )
        },
    ],
)
def test_restore_rejects_invalid_genesis(tmp_path, change):
    *_, store, _ = initial(tmp_path)
    snap = store.snapshot()
    bad = _seal_alert(replace(snap.accepted_revisions[0], **change, content_fingerprint_sha256=""))
    edge = _seal_history(
        replace(
            snap.mutation_history[0],
            after_fingerprint=bad.content_fingerprint_sha256,
            mutation_id="",
        )
    )
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.validate_snapshot(
            replace(snap, accepted_revisions=(bad,), mutation_history=(edge,))
        )


def test_history_bijection_identity_source_binding_and_store_revision(tmp_path):
    *_, store, _ = initial(tmp_path)
    snap = store.snapshot()
    ghost = _seal_history(
        MutationHistoryEntry(
            "",
            "ghost",
            0,
            1,
            "SOURCE_FAILING_OBSERVATION",
            "GENESIS",
            "a" * 64,
            NOW.isoformat().replace("+00:00", "Z"),
            "ref",
        )
    )
    for bad in (
        replace(snap, mutation_history=snap.mutation_history + (ghost,), store_revision=2),
        replace(snap, store_revision=999),
        replace(snap, mutation_history=(replace(snap.mutation_history[0], mutation_id="f" * 64),)),
    ):
        with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
            AlertStore.validate_snapshot(bad)


def test_restore_rejects_source_reference_swapped_between_conditions(tmp_path):
    auth, authority, sources, carrier, store, _ = initial(tmp_path)
    second = accept(sources, "FAILING", condition="other-condition")
    store.observe(alert_id="alert-002", evidence_set=second, now_utc=NOW)
    snap = store.snapshot()
    first, second_alert = snap.accepted_revisions
    wrong = _seal_alert(
        replace(
            first,
            source_evidence_reference=second_alert.source_evidence_reference,
            content_fingerprint_sha256="",
        )
    )
    first_edge = _seal_history(
        replace(
            snap.mutation_history[0],
            after_fingerprint=wrong.content_fingerprint_sha256,
            source_evidence_reference=second_alert.source_evidence_reference,
            mutation_id="",
        )
    )
    carrier._snapshot = replace(
        snap,
        accepted_revisions=(wrong, second_alert),
        mutation_history=(first_edge, snap.mutation_history[1]),
    )
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_UNACCEPTED"):
        AlertStore.restore(authority, sources, carrier)


def test_delivery_attempt_replay_conflict_and_adapter_time(tmp_path):
    *_, store, _ = initial(tmp_path, adapter=Adapter("DELIVERED"))
    first = store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="same",
        now_utc=NOW,
    )
    retry = store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="same",
        now_utc=NOW,
    )
    assert retry == first and len(store.snapshot().delivery_attempts) == 1
    with pytest.raises(AlertStoreError, match="REPLAY_CONFLICT"):
        store.request_delivery(
            "alert-001",
            expected_alert_revision=2,
            expected_delivery_revision=1,
            attempt_id="same",
            now_utc=NOW,
        )
    *_, bad, _ = initial(
        tmp_path / "time",
        adapter=Adapter(
            "DELIVERED",
            {"attempted_at_utc": (NOW + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")},
        ),
    )
    with pytest.raises(AlertStoreError, match="DELIVERY_RESULT_INVALID"):
        bad.request_delivery(
            "alert-001",
            expected_alert_revision=1,
            expected_delivery_revision=0,
            attempt_id="x",
            now_utc=NOW,
        )


@pytest.mark.parametrize(
    "suppression",
    [
        SuppressionState(
            True,
            "account-alerts",
            NOW.isoformat().replace("+00:00", "Z"),
            (NOW + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
            "OPERATOR_POLICY",
            1,
        ),
        SuppressionState(
            True,
            "account-alerts",
            NOW.isoformat().replace("+00:00", "Z"),
            (NOW + timedelta(days=36500)).isoformat().replace("+00:00", "Z"),
            "OPERATOR_POLICY",
            1,
        ),
        SuppressionState(
            True,
            "wrong",
            NOW.isoformat().replace("+00:00", "Z"),
            (NOW + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "OPERATOR_POLICY",
            1,
        ),
        SuppressionState(
            True,
            "account-alerts",
            NOW.isoformat().replace("+00:00", "Z"),
            (NOW + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "WRONG",
            1,
        ),
    ],
)
def test_restore_enforces_exact_suppression_policy(tmp_path, suppression):
    auth, authority, _, _, store, item = initial(tmp_path, operation="M0.12/ALERT_SET_SUPPRESSION")
    req, proof = authorized(auth, authority, item, 1)
    store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
    )
    snap = store.snapshot()
    bad = _seal_alert(
        replace(snap.accepted_revisions[-1], suppression=suppression, content_fingerprint_sha256="")
    )
    edge = _seal_history(
        replace(
            snap.mutation_history[-1],
            after_fingerprint=bad.content_fingerprint_sha256,
            mutation_id="",
        )
    )
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.validate_snapshot(
            replace(
                snap,
                accepted_revisions=(snap.accepted_revisions[0], bad),
                mutation_history=(snap.mutation_history[0], edge),
            )
        )


@pytest.mark.parametrize(
    "delivery_change",
    [
        {"state": "PENDING"},
        {"destination": "wrong"},
        {"failure_count": 99},
        {"attempt": 7},
    ],
)
def test_restore_binds_exact_delivery_result(tmp_path, delivery_change):
    *_, store, _ = initial(tmp_path, adapter=Adapter("FAILED"))
    store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="bound",
        now_utc=NOW,
    )
    snap = store.snapshot()
    after = snap.accepted_revisions[-1]
    bad = _seal_alert(
        replace(
            after,
            delivery=replace(after.delivery, **delivery_change),
            content_fingerprint_sha256="",
        )
    )
    edge = _seal_history(
        replace(
            snap.mutation_history[-1],
            after_fingerprint=bad.content_fingerprint_sha256,
            mutation_id="",
        )
    )
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.validate_snapshot(
            replace(
                snap,
                accepted_revisions=(snap.accepted_revisions[0], bad),
                mutation_history=(snap.mutation_history[0], edge),
            )
        )
