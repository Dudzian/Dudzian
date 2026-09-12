"""S9D-C9 adversarial source-policy, chronology, and replay-provenance tests."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from bot_core.alerts.store import (
    AlertStore,
    AlertStoreError,
    AtomicAlertAuthorityState,
    InMemoryAlertStoreCarrier,
    InMemoryHistoricalAuthorizationDecisionAuthority,
    InMemorySourceEvidenceAuthority,
    SourceEvidenceSet,
    SourceSelector,
    _seal_alert,
    _seal_history,
    _seal_replay,
    _seed_trusted_source_evidence,
    _source_fingerprint,
)
from bot_core.security.authorization import _seed_trusted_operation_entitlement
from tests.alerts.test_alert_store import (
    accept,
    authorized,
    evidence,
    initial,
    source_authority,
)
from tests.security.test_authentication import NOW, RAW_PIN
from tests.security.test_downstream_operation_authorization import entitlement, exact_request

BASE = datetime(2026, 9, 11, 20, 0, tzinfo=timezone.utc)


def _selector(**changes):
    value = SourceSelector(
        "TEST_MULTI_SOURCE",
        "OBSERVATION",
        "PAPER",
        "account-alerts",
        "HEALTH",
        ("source-a", "source-b"),
        True,
        "ERROR",
    )
    return replace(value, **changes)


@pytest.mark.parametrize("value", ("false", 1))
def test_truthy_non_boolean_resolution_selector_is_rejected(value):
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        InMemorySourceEvidenceAuthority((_selector(healthy_resolution_supported=value),))


def test_domain_selector_cannot_promote_automatic_resolution(tmp_path):
    promoted = SourceSelector(
        "DOMAIN_EXECUTION_FAILURE",
        "DOMAIN_EVENT",
        "PAPER",
        "account-alerts",
        "EXECUTION",
        ("source-a",),
        True,
        "ERROR",
    )
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        InMemorySourceEvidenceAuthority((promoted,))

    selector = replace(promoted, healthy_resolution_supported=False)
    sources = InMemorySourceEvidenceAuthority((selector,))
    authentication, authorization, _, _, _ = initial(tmp_path)[:5]
    carrier = InMemoryAlertStoreCarrier()
    store = AlertStore(authorization, sources, carrier)
    failing = evidence("source-a", "FAILING", alert_type="DOMAIN_EXECUTION_FAILURE")
    failing = replace(
        failing,
        source_family="DOMAIN_EVENT",
        fact_type="EXECUTION",
        observed_at_utc=BASE.isoformat().replace("+00:00", "Z"),
        content_fingerprint_sha256="",
    )
    failing = replace(failing, content_fingerprint_sha256=_source_fingerprint(failing))
    _seed_trusted_source_evidence(sources, failing)
    store.observe(
        alert_id="domain-alert",
        evidence_set=SourceEvidenceSet((failing.evidence_id,)),
        now_utc=BASE,
    )
    healthy = replace(
        failing,
        evidence_id="domain-healthy-2",
        source_revision=2,
        observed_result="HEALTHY",
        observed_at_utc=(BASE + timedelta(seconds=1)).isoformat().replace("+00:00", "Z"),
        content_fingerprint_sha256="",
    )
    healthy = replace(healthy, content_fingerprint_sha256=_source_fingerprint(healthy))
    _seed_trusted_source_evidence(sources, healthy)
    before = carrier.load()
    with pytest.raises(AlertStoreError, match="SOURCE_RESOLUTION_AUTHORITY_OPEN"):
        store.observe(
            alert_id="domain-alert",
            evidence_set=SourceEvidenceSet((healthy.evidence_id,)),
            now_utc=BASE + timedelta(seconds=1),
        )
    assert carrier.load() == before


def test_selector_shape_and_duplicate_identity_are_rejected():
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        InMemorySourceEvidenceAuthority((_selector(), _selector()))
    for source_ids in (("", "source-b"), ("source-a", 2)):
        with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
            InMemorySourceEvidenceAuthority((_selector(required_source_ids=source_ids),))


def _at(value, observed_at):
    value = replace(
        value,
        observed_at_utc=observed_at.isoformat().replace("+00:00", "Z"),
        content_fingerprint_sha256="",
    )
    return replace(value, content_fingerprint_sha256=_source_fingerprint(value))


@pytest.mark.parametrize("result", ("FAILING", "HEALTHY"))
def test_subsecond_source_chronology_and_restore(tmp_path, result):
    _, authorization, _, _, _, _ = initial(tmp_path)
    sources = source_authority()
    carrier = InMemoryAlertStoreCarrier()
    store = AlertStore(authorization, sources, carrier)
    first = tuple(_at(evidence(s, "FAILING", 1), BASE) for s in ("source-a", "source-b"))
    for value in first:
        _seed_trusted_source_evidence(sources, value)
    store.observe(
        alert_id="alert-time",
        evidence_set=SourceEvidenceSet(tuple(x.evidence_id for x in first)),
        now_utc=BASE,
    )
    later = BASE + timedelta(microseconds=500_000)
    second = tuple(_at(evidence(s, result, 2), later) for s in ("source-a", "source-b"))
    for value in second:
        _seed_trusted_source_evidence(sources, value)
    current = store.observe(
        alert_id="alert-time",
        evidence_set=SourceEvidenceSet(tuple(x.evidence_id for x in second)),
        now_utc=later,
    )
    assert current.last_seen_at_utc == "2026-09-11T20:00:00.500000Z"
    restored = AlertStore.restore(authorization, sources, carrier)
    assert restored.current("alert-time").last_seen_at_utc == current.last_seen_at_utc

    snapshot = carrier.load()
    revisions = list(snapshot.accepted_revisions)
    revisions[-1] = _seal_alert(replace(revisions[-1], last_seen_at_utc="2026-09-11T20:00:00Z"))
    histories = list(snapshot.mutation_history)
    histories[-1] = _seal_history(
        replace(
            histories[-1],
            after_fingerprint=revisions[-1].content_fingerprint_sha256,
        )
    )
    tampered = replace(
        snapshot,
        accepted_revisions=tuple(revisions),
        mutation_history=tuple(histories),
    )

    class FrozenCarrier:
        def load(self):
            return tampered

        def load_historical_decisions(self):
            return {}

        def load_atomic_state(self):
            return AtomicAlertAuthorityState(tampered, {})

        def commit(self, expected_store_revision, candidate):  # pragma: no cover
            raise AssertionError((expected_store_revision, candidate))

    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_UNACCEPTED"):
        AlertStore.restore(authorization, sources, FrozenCarrier())


def test_replay_cannot_swap_audit_obligations_between_alerts(tmp_path):
    decisions = InMemoryHistoricalAuthorizationDecisionAuthority()
    authentication, authorization, sources, carrier, store, item = initial(
        tmp_path, historical_authorizations=decisions
    )
    second_set = accept(sources, "FAILING", condition="condition-b")
    store.observe(alert_id="alert-002", evidence_set=second_set, now_utc=NOW)

    request_a, proof_a = authorized(authentication, authorization, item, 1)
    store.mutate(
        alert_id="alert-001",
        proof=proof_a,
        request=request_a,
        expected_alert_revision=1,
        now_utc=NOW,
    )
    request_b = exact_request(
        item,
        alert_id="alert-002",
        revision=1,
        causation="cause-b",
        correlation="correlation-b",
    )
    proof_b = authentication.issue_authentication_proof(request_b, RAW_PIN, NOW)
    _seed_trusted_operation_entitlement(authorization, entitlement(request_b))
    store.mutate(
        alert_id="alert-002",
        proof=proof_b,
        request=request_b,
        expected_alert_revision=1,
        now_utc=NOW,
    )

    snapshot = carrier.load()
    replay_a, replay_b = snapshot.operator_replays
    obligation_a, obligation_b = snapshot.audit_outbox
    replay_a = _seal_replay(
        replace(
            replay_a,
            audit_obligation_id=obligation_b.obligation_id,
            causation_id=obligation_b.causation_id,
            correlation_id=obligation_b.correlation_id,
        )
    )
    replay_b = _seal_replay(
        replace(
            replay_b,
            audit_obligation_id=obligation_a.obligation_id,
            causation_id=obligation_a.causation_id,
            correlation_id=obligation_a.correlation_id,
        )
    )
    carrier._snapshot = replace(snapshot, operator_replays=(replay_a, replay_b))
    with pytest.raises(AlertStoreError, match="CONTRACT_INCONSISTENT"):
        AlertStore.restore(
            authorization,
            sources,
            carrier,
            historical_authorizations=decisions,
        )
