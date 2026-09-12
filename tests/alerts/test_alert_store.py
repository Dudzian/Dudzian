from dataclasses import asdict, replace
from datetime import timedelta

import pytest

from bot_core.alerts.store import (
    AlertStore,
    AlertStoreError,
    DeliveryAttempt,
    InMemoryAlertStoreCarrier,
    InMemorySourceEvidenceAuthority,
    SourceEvidence,
    SourceEvidenceSet,
    SourceSelector,
    _seal_alert,
    _seal_obligation,
    _source_fingerprint,
    _seed_trusted_source_evidence,
)
from bot_core.security.authentication import (
    _seed_trusted_downstream_operation_definition,
    downstream_mutation_fingerprint,
    downstream_scope_fingerprint,
)
from bot_core.security.authorization import _seed_trusted_operation_entitlement
from tests.security.test_authentication import NOW, RAW_PIN
from tests.security.test_downstream_operation_authorization import (
    arranged,
    definition,
    entitlement,
    exact_request,
)


def source_authority(alert_type="TEST_MULTI_SOURCE", healthy=True, severity="ERROR"):
    selector = SourceSelector(
        alert_type,
        "OBSERVATION",
        "PAPER",
        "account-alerts",
        "HEALTH",
        ("source-a", "source-b"),
        healthy,
        severity,
    )
    return InMemorySourceEvidenceAuthority((selector,))


def evidence(
    source,
    result,
    revision=1,
    *,
    alert_type="TEST_MULTI_SOURCE",
    condition="condition",
    severity="ERROR",
):
    value = SourceEvidence(
        f"ev-{source}-{condition}-{revision}-{result}",
        alert_type,
        "OBSERVATION",
        source,
        "PAPER",
        "account-alerts",
        "HEALTH",
        condition,
        revision,
        1,
        result,
        NOW.isoformat().replace("+00:00", "Z"),
        severity,
        "",
    )
    return replace(value, content_fingerprint_sha256=_source_fingerprint(value))


def accept(authority, result, revision=1, **kwargs):
    values = tuple(
        evidence(source, result, revision, **kwargs) for source in ("source-a", "source-b")
    )
    for value in values:
        _seed_trusted_source_evidence(authority, value)
    return SourceEvidenceSet(tuple(value.evidence_id for value in values))


def initial(
    tmp_path,
    operation="M0.12/ALERT_ACKNOWLEDGE",
    *,
    adapter=None,
    carrier=None,
    audit_ready=lambda: True,
    severity="ERROR",
    historical_authorizations=None,
):
    authentication, authorization, item, _, _ = arranged(tmp_path, operation)
    sources = source_authority(severity=severity)
    evidence_set = accept(sources, "FAILING", severity=severity)
    carrier = carrier or InMemoryAlertStoreCarrier()
    store = AlertStore(
        authorization,
        sources,
        carrier,
        delivery_adapter=adapter,
        audit_ready=audit_ready,
        historical_authorizations=historical_authorizations,
    )
    store.observe(alert_id="alert-001", evidence_set=evidence_set, now_utc=NOW)
    return authentication, authorization, sources, carrier, store, item


def authorized(authentication, authorization, item, revision, *, now=NOW):
    req = exact_request(
        item,
        revision=revision,
        intent=item.declared_intent,
        causation=f"cause-{item.operation}-{revision}",
        correlation=f"correlation-{item.operation}-{revision}",
    )
    if item.operation == "M0.12/ALERT_CLEAR_SUPPRESSION":
        target = {
            "alert_id": "alert-001",
            "expected_alert_revision": revision,
            "alert_scope": "account-alerts",
        }
        mutation = {"intent": item.declared_intent, "value": False}
        req = replace(req, scope_fingerprint_sha256=downstream_scope_fingerprint(item, req, target))
        req = replace(
            req,
            mutation_fingerprint_sha256=downstream_mutation_fingerprint(
                item, req, target, mutation
            ),
        )
    proof = authentication.issue_authentication_proof(req, RAW_PIN, now)
    _seed_trusted_operation_entitlement(authorization, entitlement(req))
    return req, proof


class Adapter:
    def __init__(self, outcome="DELIVERED", tamper=None):
        self.outcome, self.tamper = outcome, tamper
        self.results = set()
        self.attempts = {}
        self.external_effect_count = 0

    def attempt_idempotent(self, intent):
        prior = self.attempts.get(intent.attempt_id)
        context = self._semantic_key(intent)
        if prior is not None:
            if prior[0] != context:
                raise AlertStoreError("DELIVERY_ATTEMPT_REPLAY_CONFLICT")
            return prior[1]
        result = replace(intent, outcome=self.outcome)
        result = replace(result, **(self.tamper or {}))
        self.external_effect_count += 1
        self.attempts[intent.attempt_id] = (context, result)
        self.results.add(self._key(result))
        return result

    def historical_result(self, intent):
        prior = self.attempts.get(intent.attempt_id)
        if prior is None:
            return None
        if prior[0] != self._semantic_key(intent):
            raise AlertStoreError("DELIVERY_ATTEMPT_REPLAY_CONFLICT")
        return prior[1]

    @staticmethod
    def _semantic_key(result):
        body = asdict(result)
        for field in ("content_fingerprint_sha256", "outcome", "attempted_at_utc"):
            body.pop(field)
        return tuple(sorted(body.items()))

    @staticmethod
    def _key(result):
        body = asdict(result)
        body.pop("content_fingerprint_sha256")
        return tuple(sorted(body.items()))

    def validates_result(self, result):
        return self._key(result) in self.results


def test_no_complete_evidence_api_and_source_authority_is_required(tmp_path):
    _, _, sources, _, store, _ = initial(tmp_path)
    assert not hasattr(store, "observe_healthy")
    with pytest.raises(TypeError):
        store.observe(alert_id="alert-001", complete_evidence=True)  # type: ignore[call-arg]
    before = store.snapshot()
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_UNACCEPTED"):
        store.observe(
            alert_id="alert-001",
            evidence_set=SourceEvidenceSet(("forged",)),
            now_utc=NOW,
        )
    assert store.snapshot() == before
    one = evidence("source-a", "HEALTHY", 2)
    _seed_trusted_source_evidence(sources, one)
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_INCOMPLETE"):
        store.observe(
            alert_id="alert-001",
            evidence_set=SourceEvidenceSet((one.evidence_id,)),
            now_utc=NOW,
        )
    stale_a = evidence("source-a", "HEALTHY", 3)
    stale_b = evidence("source-b", "HEALTHY", 3)
    for item in (stale_a, stale_b):
        _seed_trusted_source_evidence(sources, item)
    newer = evidence("source-a", "HEALTHY", 4)
    _seed_trusted_source_evidence(sources, newer)
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_STALE"):
        store.observe(
            alert_id="alert-001",
            evidence_set=SourceEvidenceSet((stale_a.evidence_id, stale_b.evidence_id)),
            now_utc=NOW,
        )


def test_wrong_source_selector_is_rejected(tmp_path):
    _, _, sources, _, store, _ = initial(tmp_path)
    wrong = replace(
        evidence("source-a", "HEALTHY", 2),
        source_family="DOMAIN_EVENT",
        content_fingerprint_sha256="",
    )
    wrong = replace(wrong, content_fingerprint_sha256=_source_fingerprint(wrong))
    _seed_trusted_source_evidence(sources, wrong)
    with pytest.raises(AlertStoreError, match="SOURCE_SELECTOR_MISMATCH"):
        store.observe(
            alert_id="alert-001",
            evidence_set=SourceEvidenceSet((wrong.evidence_id,)),
            now_utc=NOW,
        )


def test_current_complete_evidence_resolves_only_matching_fact(tmp_path):
    _, _, sources, _, store, _ = initial(tmp_path)
    healthy = accept(sources, "HEALTHY", 2)
    resolved = store.observe(
        alert_id="ignored",
        evidence_set=healthy,
        now_utc=NOW + timedelta(seconds=3),
    )
    assert (resolved.alert_id, resolved.fact_state, resolved.lifecycle_state) == (
        "alert-001",
        "HEALTHY",
        "RESOLVED",
    )
    revision = store.snapshot().store_revision
    replay = store.observe(alert_id="x", evidence_set=healthy, now_utc=NOW + timedelta(seconds=4))
    assert replay == resolved
    assert store.snapshot().store_revision == revision


def test_domain_execution_resolution_fails_closed(tmp_path):
    selector = SourceSelector(
        "DOMAIN_EXECUTION_FAILURE",
        "DOMAIN_EVENT",
        "PAPER",
        "account-alerts",
        "EXECUTION",
        ("source-a",),
        False,
    )
    sources = InMemorySourceEvidenceAuthority((selector,))
    authentication, authorization, _, _, _ = arranged(tmp_path)
    carrier = InMemoryAlertStoreCarrier()
    store = AlertStore(authorization, sources, carrier)

    def ev(result, revision):
        x = SourceEvidence(
            f"d-{result}-{revision}",
            selector.alert_type,
            selector.source_family,
            "source-a",
            selector.environment,
            selector.alert_scope,
            selector.fact_type,
            "order",
            revision,
            1,
            result,
            NOW.isoformat().replace("+00:00", "Z"),
            "ERROR",
            "",
        )
        return replace(x, content_fingerprint_sha256=_source_fingerprint(x))

    failing = ev("FAILING", 1)
    _seed_trusted_source_evidence(sources, failing)
    store.observe(
        alert_id="domain",
        evidence_set=SourceEvidenceSet((failing.evidence_id,)),
        now_utc=NOW,
    )
    healthy = ev("HEALTHY", 2)
    _seed_trusted_source_evidence(sources, healthy)
    with pytest.raises(AlertStoreError, match="SOURCE_RESOLUTION_AUTHORITY_OPEN"):
        store.observe(
            alert_id="domain",
            evidence_set=SourceEvidenceSet((healthy.evidence_id,)),
            now_utc=NOW,
        )


def test_domain_execution_manual_resolution_is_denied(tmp_path):
    # Operator authorization cannot override the closed alert-type policy.
    auth, authority, sources, carrier, store, _ = initial(tmp_path)
    item = definition("M0.12/ALERT_MANUAL_FACT_RESOLUTION")
    _seed_trusted_downstream_operation_definition(auth, item)
    req, proof = authorized(auth, authority, item, 1)
    before = store.snapshot()
    with pytest.raises(AlertStoreError, match="MANUAL_RESOLUTION_DENIED"):
        store.mutate(
            alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
        )
    assert store.snapshot() == before


def test_bounded_suppression_real_set_clear_and_expiry(tmp_path):
    auth, authority, _, _, store, set_item = initial(tmp_path, "M0.12/ALERT_SET_SUPPRESSION")
    req, proof = authorized(auth, authority, set_item, 1)
    set_alert = store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
    )
    assert set_alert.suppression.expires_at_utc is not None
    assert set_alert.suppression.expires_at_utc > set_alert.suppression.effective_at_utc
    clear_item = definition("M0.12/ALERT_CLEAR_SUPPRESSION")
    _seed_trusted_downstream_operation_definition(auth, clear_item)
    req2, proof2 = authorized(auth, authority, clear_item, 2)
    cleared = store.mutate(
        alert_id="alert-001",
        proof=proof2,
        request=req2,
        expected_alert_revision=2,
        now_utc=NOW + timedelta(seconds=1),
    )
    assert (cleared.alert_revision, cleared.suppression.suppressed) == (3, False)
    assert [(x.pre_revision, x.post_revision) for x in store.snapshot().audit_outbox] == [
        (1, 2),
        (2, 3),
    ]

    _, _, _, _, store2, item2 = initial(tmp_path / "expiry", "M0.12/ALERT_SET_SUPPRESSION")
    req3, proof3 = authorized(
        store2._authorization._authentication, store2._authorization, item2, 1
    )
    active = store2.mutate(
        alert_id="alert-001", proof=proof3, request=req3, expected_alert_revision=1, now_utc=NOW
    )
    with pytest.raises(AlertStoreError, match="SUPPRESSION_NOT_EXPIRED|TIME_ROLLBACK"):
        store2.expire_suppression(
            "alert-001", expected_alert_revision=2, now_utc=NOW - timedelta(seconds=1)
        )
    expired = store2.expire_suppression(
        "alert-001", expected_alert_revision=2, now_utc=NOW + timedelta(hours=1)
    )
    assert not expired.suppression.suppressed and expired.alert_revision == 3
    assert (expired.fact_state, expired.acknowledged) == (active.fact_state, active.acknowledged)


def test_delivery_requires_adapter_exact_result_and_destination(tmp_path):
    *_, store, _ = initial(tmp_path, adapter=None)
    assert not hasattr(store, "record_delivery")
    with pytest.raises(AlertStoreError, match="DELIVERY_AUTHORITY_UNAVAILABLE"):
        store.request_delivery(
            "alert-001",
            expected_alert_revision=1,
            expected_delivery_revision=0,
            attempt_id="attempt-1",
            now_utc=NOW,
        )
    *_, store2, _ = initial(tmp_path / "bad", adapter=Adapter(tamper={"destination": "evil"}))
    with pytest.raises(AlertStoreError, match="DELIVERY_RESULT_INVALID"):
        store2.request_delivery(
            "alert-001",
            expected_alert_revision=1,
            expected_delivery_revision=0,
            attempt_id="attempt-1",
            now_utc=NOW,
        )
    *_, store3, _ = initial(tmp_path / "ok", adapter=Adapter("DELIVERED"))
    delivered = store3.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="attempt-1",
        now_utc=NOW + timedelta(seconds=3),
    )
    assert delivered.delivery.state == "DELIVERED"


def test_escalation_policy_early_rollback_duplicate(tmp_path):
    *_, store, _ = initial(tmp_path, adapter=Adapter("FAILED"))
    failed = store.request_delivery(
        "alert-001",
        expected_alert_revision=1,
        expected_delivery_revision=0,
        attempt_id="a",
        now_utc=NOW + timedelta(seconds=3),
    )
    with pytest.raises(AlertStoreError, match="ESCALATION_POLICY_NOT_SATISFIED"):
        store.escalate(
            "alert-001",
            expected_alert_revision=2,
            expected_delivery_revision=1,
            expected_escalation_revision=0,
            now_utc=NOW + timedelta(minutes=1),
        )
    with pytest.raises(AlertStoreError, match="TIME_ROLLBACK"):
        store.escalate(
            "alert-001",
            expected_alert_revision=2,
            expected_delivery_revision=1,
            expected_escalation_revision=0,
            now_utc=NOW,
        )
    escalated = store.escalate(
        "alert-001",
        expected_alert_revision=2,
        expected_delivery_revision=1,
        expected_escalation_revision=0,
        now_utc=NOW + timedelta(minutes=6),
    )
    assert escalated.delivery.escalation_revision == 1
    with pytest.raises(AlertStoreError, match="STALE_ESCALATION"):
        store.escalate(
            "alert-001",
            expected_alert_revision=2,
            expected_delivery_revision=1,
            expected_escalation_revision=0,
            now_utc=NOW + timedelta(minutes=7),
        )


def test_carrier_failures_never_publish_partial_operator_commit(tmp_path):
    carrier = InMemoryAlertStoreCarrier()
    auth, authority, _, _, store, item = initial(tmp_path, carrier=carrier)
    req, proof = authorized(auth, authority, item, 1)
    before = store.snapshot()
    for point in ("BEFORE", "DURING"):
        carrier.fail_next = point
        with pytest.raises(AlertStoreError, match="CARRIER_COMMIT_FAILED"):
            store.mutate(
                alert_id="alert-001",
                proof=proof,
                request=req,
                expected_alert_revision=1,
                now_utc=NOW,
            )
        assert store.snapshot() == carrier.load() == before
    committed = store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
    )
    assert committed.alert_revision == 2 and len(carrier.load().audit_outbox) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("operation", "wrong"),
        ("declared_intent", "wrong"),
        ("operator_id", "wrong"),
        ("device_installation_id", "wrong"),
        ("causation_id", "wrong"),
        ("correlation_id", "wrong"),
        ("pre_revision", 0),
        ("post_revision", 9),
    ],
)
def test_restore_rejects_complete_audit_tampering(tmp_path, field, value):
    auth, authority, sources, carrier, store, item = initial(tmp_path)
    req, proof = authorized(auth, authority, item, 1)
    store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
    )
    snap = store.snapshot()
    bad = replace(snap.audit_outbox[0], **{field: value})
    bad = _seal_obligation(bad)
    carrier._snapshot = replace(snap, audit_outbox=(bad,))
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.restore(authority, sources, carrier)


def test_restore_rejects_ack_with_suppression_and_dedup_collision(tmp_path):
    auth, authority, sources, carrier, store, item = initial(tmp_path)
    req, proof = authorized(auth, authority, item, 1)
    after = store.mutate(
        alert_id="alert-001", proof=proof, request=req, expected_alert_revision=1, now_utc=NOW
    )
    tampered = _seal_alert(
        replace(
            after,
            suppression=replace(
                after.suppression,
                suppressed=True,
                effective_at_utc=NOW.isoformat(),
                expires_at_utc=(NOW + timedelta(hours=1)).isoformat(),
                revision=1,
            ),
            content_fingerprint_sha256="",
        )
    )
    entry = replace(
        store.snapshot().mutation_history[-1], after_fingerprint=tampered.content_fingerprint_sha256
    )
    carrier._snapshot = replace(
        store.snapshot(),
        accepted_revisions=(store.snapshot().accepted_revisions[0], tampered),
        mutation_history=(store.snapshot().mutation_history[0], entry),
    )
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.restore(authority, sources, carrier)

    # A second active incident with the same semantic identity is rejected before index comparison.
    snap = store.snapshot()
    first = snap.accepted_revisions[0]
    duplicate = _seal_alert(replace(first, alert_id="alert-002", content_fingerprint_sha256=""))
    dup_entry = replace(
        snap.mutation_history[0],
        alert_id="alert-002",
        after_fingerprint=duplicate.content_fingerprint_sha256,
    )
    bad = replace(
        snap,
        accepted_revisions=snap.accepted_revisions + (duplicate,),
        current_designations={**snap.current_designations, "alert-002": 1},
        mutation_history=snap.mutation_history + (dup_entry,),
    )
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.validate_snapshot(bad)
