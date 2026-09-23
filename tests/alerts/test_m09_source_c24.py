"""C24 production M0.9 -> AlertStore authority integration regressions."""

from dataclasses import replace
from datetime import datetime, timezone
from threading import Event, Thread

import pytest

from bot_core.alerts.m09_source import M09KillSwitchSourceAuthority
from bot_core.alerts.store import (
    AlertStore,
    AlertStoreError,
    HistoricalSourceDecision,
    InMemoryAlertStoreCarrier,
    SourceEvidenceSet,
    _historical_source_decision_id,
)
from bot_core.m09_kill_switch_authority import KillSwitchAuthority
from tests.architecture.test_m09_kill_switch_authority import Rig, WS


NOW = datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
AFTER_5 = datetime(2026, 1, 1, 0, 0, 5, 500000, tzinfo=timezone.utc)
AFTER_6 = datetime(2026, 1, 1, 0, 0, 6, 500000, tzinfo=timezone.utc)


def _accept(
    rig: Rig,
    state: str,
    generation: int,
    revision: int,
    membership: str,
    *,
    accepted_at: str | None = None,
):
    record = rig.record(state=state, generation=generation, source_revision=revision)
    return rig.accept(
        rig.context(record, membership_id=membership),
        now_utc=accepted_at or f"2026-01-01T00:00:{generation:02d}Z",
    )


def test_active_then_inactive_resolves_with_exact_record_fence_and_survives_restore():
    rig = Rig()
    _accept(rig, "ACTIVE", 5, 41, "active-5")
    adapter = M09KillSwitchSourceAuthority(rig.authority)
    carrier = InMemoryAlertStoreCarrier()
    store = AlertStore(object(), adapter, carrier)
    active_set = adapter.evidence_set_for_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    )
    failing = store.observe(alert_id="alert", evidence_set=active_set, now_utc=AFTER_5)
    active_fact = adapter.historical_fact(active_set.evidence_ids[0])
    assert active_fact.result == "FAILING"
    assert failing.source_fence[0][1:] == (5, 41)

    _accept(rig, "INACTIVE", 6, 42, "inactive-6")
    inactive_set = adapter.evidence_set_for_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    )
    healthy = store.observe(alert_id="ignored", evidence_set=inactive_set, now_utc=AFTER_6)
    inactive_fact = adapter.historical_fact(inactive_set.evidence_ids[0])
    assert inactive_fact.result == "HEALTHY"
    assert inactive_fact.evidence[0].source_generation == 6
    assert inactive_fact.evidence[0].source_revision == 42
    assert active_fact.evidence[0].source_id == inactive_fact.evidence[0].source_id
    assert healthy.lifecycle_state == "RESOLVED"

    restored_authority = KillSwitchAuthority(rig.carrier, core_membership=rig.core)
    restored_adapter = M09KillSwitchSourceAuthority(restored_authority)
    restored = AlertStore.restore(object(), restored_adapter, carrier)
    assert restored.current("alert") == healthy
    assert all(
        restored_adapter.validates_historical_decision(decision)
        for decision in carrier.load_atomic_state().committed_historical_source_decisions.values()
    )
    assert restored_adapter.historical_fact(active_set.evidence_ids[0]).result == "FAILING"


def test_non_current_and_stale_corrective_reject_but_exact_committed_replay_remains():
    rig = Rig()
    _accept(rig, "ACTIVE", 5, 5, "active-5")
    adapter = M09KillSwitchSourceAuthority(rig.authority)
    store = AlertStore(object(), adapter, InMemoryAlertStoreCarrier())
    active = adapter.evidence_set_for_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    )
    original = store.observe(alert_id="alert", evidence_set=active, now_utc=AFTER_5)
    _accept(rig, "INACTIVE", 6, 6, "inactive-6")
    inactive = adapter.evidence_set_for_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    )
    assert store.observe(alert_id="replay", evidence_set=active, now_utc=NOW) == original
    _accept(rig, "ACTIVE", 7, 7, "active-7")
    before = store.snapshot()
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_STALE"):
        store.observe(alert_id="ignored", evidence_set=inactive, now_utc=NOW)
    assert store.snapshot() == before


def test_reference_forgery_and_exact_scope_environment_are_fail_closed():
    rig = Rig()
    _accept(rig, "ACTIVE", 5, 9, "active-5")
    adapter = M09KillSwitchSourceAuthority(rig.authority)
    evidence = adapter.evidence_set_for_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    )
    fact = adapter.validate_current(evidence, NOW)
    forged = replace(
        fact.evidence[0], source_generation=6, source_revision=10, observed_result="HEALTHY"
    )
    assert forged != fact.evidence[0]  # Raw SourceEvidence is transport, never an input authority.
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_UNACCEPTED"):
        adapter.historical_fact(evidence.evidence_ids[0] + "forged")
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_UNACCEPTED"):
        adapter.evidence_set_for_current(scope_type="WORKSPACE", scope_id=WS, environment="LIVE")
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_UNACCEPTED"):
        adapter.evidence_set_for_current(
            scope_type="WORKSPACE",
            scope_id="ws_01890f3a-2b4c-7abc-8def-0123456789ac",
            environment="TESTNET",
        )


def test_stale_reference_cannot_mint_hsd_after_successor_acceptance():
    rig = Rig()
    _accept(rig, "ACTIVE", 5, 5, "active-5", accepted_at="2026-01-01T12:00:05Z")
    adapter = M09KillSwitchSourceAuthority(rig.authority)
    reference = adapter.evidence_set_for_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    ).evidence_ids[0]
    _accept(rig, "INACTIVE", 6, 6, "inactive-6", accepted_at="2026-01-01T12:00:06Z")

    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_STALE"):
        adapter.authorize_historical_transition(reference, "2026-01-01T12:00:07Z")


def test_valid_historical_hsd_survives_supersession_and_authority_restart():
    rig = Rig()
    _accept(rig, "ACTIVE", 5, 100, "active-5", accepted_at="2026-01-01T12:00:05Z")
    adapter = M09KillSwitchSourceAuthority(rig.authority)
    reference = adapter.evidence_set_for_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    ).evidence_ids[0]
    decision = adapter.authorize_historical_transition(reference, "2026-01-01T12:00:05.500000Z")
    _accept(rig, "INACTIVE", 6, 1, "inactive-6", accepted_at="2026-01-01T12:00:06Z")

    assert adapter.validates_historical_decision(decision)
    assert adapter.historical_fact_for_decision(decision).result == "FAILING"
    restored = M09KillSwitchSourceAuthority(
        KillSwitchAuthority(rig.carrier, core_membership=rig.core)
    )
    assert restored.validates_historical_decision(decision)
    assert restored.historical_fact_for_decision(decision).evidence[0].source_generation == 5


def test_coherently_resealed_hsd_for_stale_at_transaction_time_is_rejected():
    rig = Rig()
    _accept(rig, "ACTIVE", 5, 5, "active-5")
    adapter = M09KillSwitchSourceAuthority(rig.authority)
    reference = adapter.evidence_set_for_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    ).evidence_ids[0]
    fact = adapter.historical_fact(reference)
    item = fact.evidence[0]
    _accept(rig, "INACTIVE", 6, 6, "inactive-6")
    forged = HistoricalSourceDecision(
        "",
        reference,
        (reference,),
        "2026-01-01T00:00:07Z",
        fact.result,
        item.source_severity,
        fact.selector.resolution_policy_id,
        ((item.source_id, item.source_generation, item.source_revision),),
    )
    forged = replace(forged, decision_id=_historical_source_decision_id(forged))
    assert not adapter.validates_historical_decision(forged)
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_STALE"):
        adapter.historical_fact_for_decision(forged)


def test_generation_not_source_revision_controls_corrective_fence():
    rig = Rig()
    _accept(rig, "ACTIVE", 5, 100, "active-5")
    adapter = M09KillSwitchSourceAuthority(rig.authority)
    store = AlertStore(object(), adapter, InMemoryAlertStoreCarrier())
    failing = store.observe(
        alert_id="alert",
        evidence_set=adapter.evidence_set_for_current(
            scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
        ),
        now_utc=AFTER_5,
    )
    _accept(rig, "INACTIVE", 6, 1, "inactive-6")
    resolved = store.observe(
        alert_id="ignored",
        evidence_set=adapter.evidence_set_for_current(
            scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
        ),
        now_utc=AFTER_6,
    )
    assert failing.source_fence[0][1:] == (5, 100)
    assert resolved.source_fence[0][1:] == (6, 1)
    assert resolved.lifecycle_state == "RESOLVED"


def test_alertstore_nested_hsd_is_reentrant_and_inside_source_fence(monkeypatch):
    rig = Rig()
    _accept(rig, "ACTIVE", 5, 5, "active-5")
    adapter = M09KillSwitchSourceAuthority(rig.authority)
    carrier = InMemoryAlertStoreCarrier()
    store = AlertStore(object(), adapter, carrier)
    nested = adapter.authorize_historical_transition
    observed_inside_fence = []

    def checked(reference, transaction_time_utc):
        assert rig.carrier._lock._is_owned()  # type: ignore[attr-defined]  # noqa: SLF001
        observed_inside_fence.append(reference)
        return nested(reference, transaction_time_utc)

    monkeypatch.setattr(adapter, "authorize_historical_transition", checked)
    evidence = adapter.evidence_set_for_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    )
    alert = store.observe(alert_id="alert", evidence_set=evidence, now_utc=NOW)
    decisions = carrier.load_atomic_state().committed_historical_source_decisions
    assert observed_inside_fence == [evidence.evidence_ids[0]]
    assert decisions[evidence.evidence_ids[0]].source_fence == alert.source_fence


def test_cross_runtime_successor_waits_for_complete_alertstore_callback(monkeypatch):
    rig = Rig()
    _accept(rig, "ACTIVE", 5, 5, "active-5")
    authority_b, writer_b = KillSwitchAuthority.compose(rig.carrier, core_membership=rig.core)
    assert authority_b is not rig.authority
    adapter = M09KillSwitchSourceAuthority(rig.authority)
    store = AlertStore(object(), adapter, InMemoryAlertStoreCarrier())
    evidence = adapter.evidence_set_for_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    )
    successor = rig.context(
        rig.record(state="INACTIVE", generation=6, source_revision=6),
        membership_id="inactive-6",
    )
    entered, release, published = Event(), Event(), Event()
    original = store._observe_fact  # noqa: SLF001

    def blocked(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)

    monkeypatch.setattr(store, "_observe_fact", blocked)
    result = []
    observing = Thread(
        target=lambda: result.append(
            store.observe(alert_id="alert", evidence_set=evidence, now_utc=NOW)
        )
    )
    successor_thread = Thread(
        target=lambda: (writer_b.accept(successor, now_utc="2026-01-01T00:00:11Z"), published.set())
    )
    observing.start()
    assert entered.wait(2)
    successor_thread.start()
    assert not published.wait(0.1)
    release.set()
    observing.join(2)
    successor_thread.join(2)
    assert result[0].source_fence[0][1] == 5
    assert published.is_set()


def test_full_history_multi_scope_projection_hsd_and_restore():
    rig = Rig()
    system_1 = rig.record(
        scope_type="PRODUCT_SYSTEM", scope_id="product", state="ACTIVE", generation=1
    )
    workspace_1 = rig.record(state="ACTIVE", generation=1)
    accepted_a = rig.accept(
        rig.context(system_1, workspace_1, membership_id="snapshot-A"),
        now_utc="2026-01-01T12:00:01Z",
    )
    workspace_2 = rig.record(state="INACTIVE", generation=2)
    accepted_b = rig.accept(
        rig.context(system_1, workspace_1, workspace_2, membership_id="snapshot-B"),
        now_utc="2026-01-01T12:00:02Z",
    )
    adapter = M09KillSwitchSourceAuthority(rig.authority)
    system_ref = adapter.evidence_set_for_current(
        scope_type="PRODUCT_SYSTEM", scope_id="product", environment="TESTNET"
    ).evidence_ids[0]
    workspace_ref = adapter.evidence_set_for_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    ).evidence_ids[0]
    assert accepted_a.context.membership_id in system_ref
    assert accepted_b.context.membership_id in workspace_ref
    assert (
        adapter.validate_current(
            SourceEvidenceSet((system_ref,)),
            datetime(2026, 1, 1, 12, 0, 3, tzinfo=timezone.utc),
        ).result
        == "FAILING"
    )
    assert adapter.historical_fact(workspace_ref).result == "HEALTHY"
    system_hsd = adapter.authorize_historical_transition(system_ref, "2026-01-01T12:00:03Z")
    workspace_hsd = adapter.authorize_historical_transition(workspace_ref, "2026-01-01T12:00:03Z")
    restored = M09KillSwitchSourceAuthority(
        KillSwitchAuthority(rig.carrier, core_membership=rig.core)
    )
    assert restored.validates_historical_decision(system_hsd)
    assert restored.validates_historical_decision(workspace_hsd)
    assert restored.historical_fact_for_decision(system_hsd).evidence[0].source_generation == 1
    assert restored.historical_fact_for_decision(workspace_hsd).evidence[0].source_generation == 2
