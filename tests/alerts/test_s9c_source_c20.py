"""S9D-C20 executable S9C -> AlertStore integration regressions."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from bot_core.alerts.s9c_source import S9CObservationSourceAuthority
from bot_core.alerts.store import AlertStore, AlertStoreError, InMemoryAlertStoreCarrier, SourceEvidenceSet
from bot_core.observability.authority import (
    FreshnessPolicy,
    FrozenEnvironmentRegistryBinding,
    InMemoryObservationAuthorityCarrier,
    ObservationAuthority,
)

U = "018f1f10-7b2c-7abc-8def-123456789abc"
RUN_B = "run_018f1f10-7b2c-7abc-8def-123456789abd"


def _policy(category: str) -> FreshnessPolicy:
    return FreshnessPolicy(f"{category}_TEST_V1", 1, category, "core_host", 30, 4, 5, 4)


def _raw(category: str, *, condition: str, run: str = f"run_{U}", sequence: int = 1,
         second: int = 0, environment: str = "TESTNET", scope_suffix: str = "") -> dict:
    stamp = f"2030-01-01T00:00:{second:02d}Z"
    scope = ({"market_data_route_id": f"mdr_{U}{scope_suffix}", "instrument_id": f"instr_{U}"}
             if category == "MARKET_DATA_FRESHNESS" else
             {"exchange_account_id": f"xacc_{U}", "instrument_id": f"instr_{U}",
              "execution_route_id": f"xroute_{U}{scope_suffix}"})
    value = ({"last_data_at_utc": stamp, "sequence_state": "CURRENT"}
             if category == "MARKET_DATA_FRESHNESS" else {"path_state": "AVAILABLE"})
    return {"observation_id": f"observation-{run[-4:]}-{second}", "category": category,
            "source_component": "core_host", "source_instance_id": run,
            "environment": environment, "scope": scope, "source_event_at_utc": stamp,
            "observed_at_utc": stamp, "ingested_at_utc": stamp,
            "expires_at_utc": f"2030-01-01T00:00:{second + 30:02d}Z",
            "freshness_policy_id": f"{category}_TEST_V1", "source_sequence": sequence,
            "condition": condition, "reason_code": f"PROBE_{condition}", "value": value,
            "source_quality": "DIRECT", "correlation_reference": None}


def _system():
    carrier = InMemoryObservationAuthorityCarrier()
    authority, publisher = ObservationAuthority.compose(
        carrier, policies=(_policy("MARKET_DATA_FRESHNESS"), _policy("EXECUTION_PATH_HEALTH")),
        environment_binding=FrozenEnvironmentRegistryBinding(), enabled_environments=frozenset({"TESTNET"}),
    )
    alerts = InMemoryAlertStoreCarrier()
    adapter = S9CObservationSourceAuthority(authority)
    store = AlertStore(object(), adapter, alerts)  # authorization is unused by source transactions
    return authority, publisher, adapter, alerts, store


@pytest.mark.parametrize("category,expected_type,severity", [
    ("MARKET_DATA_FRESHNESS", "MARKET_DATA_CURRENT_CONDITION", "WARNING"),
    ("EXECUTION_PATH_HEALTH", "EXECUTION_ROUTE_CONDITION", "ERROR"),
])
def test_failing_then_cross_runtime_ok_resolves_stable_identity(category, expected_type, severity):
    _, publish, adapter, _, store = _system()
    failing = publish.publish(_raw(category, condition="DEGRADED"), now_utc="2030-01-01T00:00:00Z")
    first = store.observe(alert_id="alert-1", evidence_set=SourceEvidenceSet((failing.acceptance_id,)),
                          now_utc=datetime(2030, 1, 1, tzinfo=timezone.utc))
    healthy = publish.publish(_raw(category, condition="OK", run=RUN_B, second=1), now_utc="2030-01-01T00:00:01Z")
    second = store.observe(alert_id="ignored", evidence_set=SourceEvidenceSet((healthy.acceptance_id,)),
                           now_utc=datetime(2030, 1, 1, 0, 0, 2, tzinfo=timezone.utc))
    assert (first.alert_type, first.severity, second.lifecycle_state) == (expected_type, severity, "RESOLVED")
    assert first.dedup_identity == second.dedup_identity
    assert first.source_fence[0][2] == failing.transaction_revision
    assert second.source_fence[0][2] == healthy.transaction_revision
    assert adapter.historical_fact(failing.acceptance_id).evidence[0].source_generation == 1


@pytest.mark.parametrize("category,condition,severity", [
    ("MARKET_DATA_FRESHNESS", "UNKNOWN", "ERROR"),
    ("MARKET_DATA_FRESHNESS", "BLOCKED", "ERROR"),
    ("EXECUTION_PATH_HEALTH", "UNKNOWN", "ERROR"),
    ("EXECUTION_PATH_HEALTH", "BLOCKED", "CRITICAL"),
])
def test_effective_condition_and_frozen_severity(category, condition, severity):
    _, publish, _, _, store = _system()
    accepted = publish.publish(_raw(category, condition=condition), now_utc="2030-01-01T00:00:00Z")
    alert = store.observe(alert_id="alert", evidence_set=SourceEvidenceSet((accepted.acceptance_id,)),
                          now_utc=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert (alert.fact_state, alert.severity) == ("FAILING", severity)


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
def test_unconsumed_old_runtime_is_stale_and_cannot_mutate_store(category):
    _, publish, _, alerts, store = _system()
    old = publish.publish(_raw(category, condition="BLOCKED"), now_utc="2030-01-01T00:00:00Z")
    new = publish.publish(_raw(category, condition="OK", run=RUN_B, second=1), now_utc="2030-01-01T00:00:01Z")
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_STALE"):
        store.observe(alert_id="forged", evidence_set=SourceEvidenceSet((old.acceptance_id,)),
                      now_utc=datetime(2030, 1, 1, 0, 0, 2, tzinfo=timezone.utc))
    assert alerts.load().store_revision == 0
    assert alerts.load_atomic_state().committed_historical_source_decisions == {}
    with pytest.raises(AlertStoreError, match="ALERT_NOT_FOUND"):
        store.observe(alert_id="none", evidence_set=SourceEvidenceSet((new.acceptance_id,)),
                      now_utc=datetime(2030, 1, 1, 0, 0, 2, tzinfo=timezone.utc))


def test_acceptance_membership_not_fingerprint_and_committed_replay_survives_supersession():
    _, publish, _, alerts, store = _system()
    first = publish.publish(_raw("MARKET_DATA_FRESHNESS", condition="BLOCKED"), now_utc="2030-01-01T00:00:00Z")
    original = store.observe(alert_id="alert", evidence_set=SourceEvidenceSet((first.acceptance_id,)),
                             now_utc=datetime(2030, 1, 1, tzinfo=timezone.utc))
    before = alerts.load().store_revision
    publish.publish(_raw("MARKET_DATA_FRESHNESS", condition="OK", run=RUN_B, second=1), now_utc="2030-01-01T00:00:01Z")
    replay = store.observe(alert_id="ignored", evidence_set=SourceEvidenceSet((first.acceptance_id,)),
                           now_utc=datetime(2030, 1, 1, 0, 0, 2, tzinfo=timezone.utc))
    assert replay == original and alerts.load().store_revision == before
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_UNACCEPTED"):
        store.observe(alert_id="forged", evidence_set=SourceEvidenceSet((first.content_fingerprint,)),
                      now_utc=datetime(2030, 1, 1, 0, 0, 2, tzinfo=timezone.utc))


def test_alert_carrier_failure_has_no_partial_decision_and_retry_succeeds():
    _, publish, _, alerts, store = _system()
    accepted = publish.publish(_raw("EXECUTION_PATH_HEALTH", condition="BLOCKED"), now_utc="2030-01-01T00:00:00Z")
    alerts.fail_next = "SOURCE_BEFORE_PUBLICATION"
    with pytest.raises(AlertStoreError, match="CARRIER_COMMIT_FAILED"):
        store.observe(alert_id="alert", evidence_set=SourceEvidenceSet((accepted.acceptance_id,)),
                      now_utc=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert alerts.load().store_revision == 0
    assert not alerts.load_atomic_state().committed_historical_source_decisions
    assert store.observe(alert_id="alert", evidence_set=SourceEvidenceSet((accepted.acceptance_id,)),
                         now_utc=datetime(2030, 1, 1, tzinfo=timezone.utc)).alert_revision == 1


def test_historical_semantic_supersession_and_decision_tampering_fail_closed():
    authority, publish, adapter, _, _ = _system()
    first = publish.publish(_raw("MARKET_DATA_FRESHNESS", condition="BLOCKED"), now_utc="2030-01-01T00:00:00Z")
    decision = adapter.authorize_historical_transition(first.acceptance_id, "2030-01-01T00:00:00Z")
    publish.publish(_raw("MARKET_DATA_FRESHNESS", condition="OK", run=RUN_B, second=1), now_utc="2030-01-01T00:00:01Z")
    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_UNACCEPTED"):
        adapter.authorize_historical_transition(first.acceptance_id, "2030-01-01T00:00:02Z")
    assert not adapter.validates_historical_decision(replace(decision, severity="INFO"))
    assert authority.resolve_historical_acceptance(first.acceptance_id) == first
