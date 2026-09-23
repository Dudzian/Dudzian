from __future__ import annotations
from dataclasses import replace
from threading import Event, Thread
import pytest
from bot_core.observability.authority import *

U = "018f1f10-7b2c-7abc-8def-123456789abc"


def policy(cat="MARKET_DATA_FRESHNESS"):
    return FreshnessPolicy(cat + "_TEST_V1", 1, cat, "core_host", 30, 4, 5, 4)


def raw(cat="MARKET_DATA_FRESHNESS", seq=1, observed="2030-01-01T00:00:00Z", **change):
    scope = (
        {"market_data_route_id": "mdr_" + U, "instrument_id": "instr_" + U}
        if cat == "MARKET_DATA_FRESHNESS"
        else {
            "exchange_account_id": "xacc_" + U,
            "instrument_id": "instr_" + U,
            "execution_route_id": "xroute_" + U,
        }
    )
    value = (
        {"last_data_at_utc": observed, "sequence_state": "CURRENT"}
        if cat == "MARKET_DATA_FRESHNESS"
        else {"path_state": "AVAILABLE"}
    )
    d = dict(
        observation_id="local.observation-1",
        category=cat,
        source_component="core_host",
        source_instance_id="run_" + U,
        environment="TESTNET",
        scope=scope,
        source_event_at_utc=observed,
        observed_at_utc=observed,
        ingested_at_utc=observed,
        expires_at_utc="2030-01-01T00:00:30Z"
        if observed.endswith("00Z")
        else "2030-01-01T00:00:31Z",
        freshness_policy_id=cat + "_TEST_V1",
        source_sequence=seq,
        condition="OK",
        reason_code="PROBE_OK",
        value=value,
        source_quality="DIRECT",
        correlation_reference={"kind": "LOCAL", "value": "request.1"},
    )
    d.update(change)
    return d


def composed(carrier=None, policies=None, enabled=None):
    return ObservationAuthority.compose(
        carrier or InMemoryObservationAuthorityCarrier(),
        policies=policies or (policy(), policy("EXECUTION_PATH_HEALTH")),
        environment_binding=FrozenEnvironmentRegistryBinding(),
        enabled_environments=frozenset({"TESTNET"}) if enabled is None else enabled,
    )


def test_history_restart_replay_gap_and_exact_current_projection():
    carrier = InMemoryObservationAuthorityCarrier()
    authority, publisher = composed(carrier)
    a = publisher.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    assert publisher.publish(raw(), now_utc="2030-01-01T00:00:01Z") is a
    b = publisher.publish(
        raw(seq=3, observed="2030-01-01T00:00:01Z", observation_id="local.observation-2"),
        now_utc="2030-01-01T00:00:01Z",
    )
    assert (b.sequence_gap, b.effective_condition, b.effective_reason_codes) == (
        True,
        "DEGRADED",
        ("PROBE_OK", "SEQUENCE_GAP"),
    )
    restored, _ = composed(carrier)
    assert restored.resolve_historical_acceptance(a.acceptance_id) == a
    key = b.key
    assert restored.resolve_current(key, now_utc="2030-01-01T00:00:29Z").accepted == b
    assert restored.resolve_current(key, now_utc="2030-01-01T00:00:30Z").status == "FRESH"
    assert restored.resolve_current(key, now_utc="2030-01-01T00:00:31Z").status == "STALE"
    assert (
        restored.resolve_current(key, now_utc="2030-01-01T00:00:31Z").projected_condition
        == "UNKNOWN"
    )


def test_conflict_regression_scope_and_time_validation():
    _, p = composed()
    p.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="DUPLICATE_SEQUENCE_CONFLICT"):
        p.publish(raw(reason_code="CHANGED"), now_utc="2030-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="SEQUENCE_REGRESSION"):
        p.publish(raw(seq=0, observed="2030-01-01T00:00:01Z"), now_utc="2030-01-01T00:00:01Z")
    for changes, error in [
        ({"environment": "OTHER"}, "ILLEGAL_ENVIRONMENT"),
        ({"source_component": "tray_agent"}, "WRONG_SOURCE"),
        ({"source_instance_id": "local"}, "WRONG_SOURCE_INSTANCE"),
        ({"freshness_policy_id": "NOPE"}, "UNKNOWN_FRESHNESS_POLICY"),
        ({"expires_at_utc": "2030-01-01T00:00:29Z"}, "INVALID_EXPIRY"),
        (
            {"observed_at_utc": "2030-01-01T00:00:05Z", "expires_at_utc": "2030-01-01T00:00:35Z"},
            "FUTURE_TIMESTAMP",
        ),
        ({"source_event_at_utc": "2030-01-01T00:00:06Z"}, "FUTURE_SOURCE_EVENT"),
        ({"ingested_at_utc": "2029-12-31T23:59:55Z"}, "INGEST_SKEW"),
    ]:
        _, q = composed()
        with pytest.raises(ValueError, match=error):
            q.publish(raw(**changes), now_utc="2030-01-01T00:00:00Z")


def test_sequence_none_remains_historical_after_successor():
    carrier = InMemoryObservationAuthorityCarrier()
    a, p = composed(carrier)
    first = p.publish(raw(seq=None), now_utc="2030-01-01T00:00:00Z")
    p.publish(
        raw(seq=None, observed="2030-01-01T00:00:01Z", observation_id="next"),
        now_utc="2030-01-01T00:00:01Z",
    )
    restored, _ = composed(carrier)
    assert restored.resolve_historical_acceptance(first.acceptance_id) == first


def test_carrier_failure_is_atomic_and_retry_single_transaction():
    carrier = InMemoryObservationAuthorityCarrier()
    a, p = composed(carrier)
    carrier.fail_next = True
    with pytest.raises(OSError):
        p.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    assert (
        carrier.read().store_revision == 0 and a.resolve_historical_acceptance("anything") is None
    )
    accepted = p.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    assert carrier.read().accepted == (accepted,)


def test_real_authority_fence_orders_superseding_publication_after_consumer():
    carrier = InMemoryObservationAuthorityCarrier()
    a, p = composed(carrier)
    other, q = composed(carrier)
    first = p.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    entered = Event()
    release = Event()
    attempted = Event()
    finished = Event()
    order = []

    def consume(record):
        assert record == first
        order.append("consumer_enter")
        entered.set()
        assert release.wait(2)
        order.append("downstream_commit")

    def consuming():
        a.consume_effective_current(first.key, now_utc="2030-01-01T00:00:01Z", consumer=consume)

    def supersede():
        assert entered.wait(2)
        attempted.set()
        q.publish(
            raw(seq=2, observed="2030-01-01T00:00:01Z", observation_id="next"),
            now_utc="2030-01-01T00:00:01Z",
        )
        order.append("b_current")
        finished.set()

    t1 = Thread(target=consuming)
    t2 = Thread(target=supersede)
    t1.start()
    t2.start()
    assert attempted.wait(2)
    assert not finished.wait(0.1)
    release.set()
    t1.join()
    t2.join()
    assert order == ["consumer_enter", "downstream_commit", "b_current"]
    assert (
        other.resolve_current(
            first.key, now_utc="2030-01-01T00:00:02Z"
        ).accepted.observation.observation_id
        == "next"
    )


def test_semantic_live_winner_crosses_runtime_sessions():
    authority, publisher = composed()
    first = publisher.publish(raw(condition="BLOCKED"), now_utc="2030-01-01T00:00:00Z")
    second_raw = raw(
        seq=1,
        source_instance_id="run_018f1f10-7b2c-7abc-8def-123456789abd",
        condition="OK",
        observed="2030-01-01T00:00:01Z",
        observation_id="run-b",
    )
    second = publisher.publish(second_raw, now_utc="2030-01-01T00:00:01Z")
    semantic = ObservationSemanticKey.from_observation(first.observation)
    seen = []
    authority.consume_semantic_effective_current(
        semantic, now_utc="2030-01-01T00:00:02Z", consumer=seen.append
    )
    assert seen == [second]
    assert authority.resolve_current(first.key, now_utc="2030-01-01T00:00:02Z").accepted == first


def test_historical_semantic_winner_crosses_runtime_sessions():
    authority, publisher = composed()
    first = publisher.publish(raw(condition="BLOCKED"), now_utc="2030-01-01T00:00:00Z")
    second = publisher.publish(
        raw(
            seq=1,
            source_instance_id="run_018f1f10-7b2c-7abc-8def-123456789abd",
            observed="2030-01-01T00:00:01Z",
            observation_id="run-b",
        ),
        now_utc="2030-01-01T00:00:01Z",
    )
    with pytest.raises(ValueError, match="ACCEPTANCE_NOT_HISTORICALLY_EFFECTIVE"):
        authority.validate_historical_semantic_effective_acceptance(
            first.acceptance_id, at_utc="2030-01-01T00:00:02Z"
        )
    assert (
        authority.validate_historical_semantic_effective_acceptance(
            second.acceptance_id, at_utc="2030-01-01T00:00:02Z"
        )
        == second
    )


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize("cross_runtime", [False, True])
def test_same_semantic_same_second_new_acceptance_is_rejected(category, cross_runtime):
    carrier = InMemoryObservationAuthorityCarrier()
    authority, publisher = composed(carrier)
    first = publisher.publish(raw(category, condition="BLOCKED"), now_utc="2030-01-01T00:00:00Z")
    before = carrier.read()
    changes = dict(seq=2, observation_id="next")
    if cross_runtime:
        changes.update(seq=1, source_instance_id="run_018f1f10-7b2c-7abc-8def-123456789abd")
    with pytest.raises(ValueError, match="SEMANTIC_TRANSACTION_TIME_COLLISION"):
        publisher.publish(raw(category, **changes), now_utc="2030-01-01T00:00:00Z")
    assert carrier.read() == before
    assert (
        authority.validate_historical_semantic_effective_acceptance(
            first.acceptance_id, at_utc="2030-01-01T00:00:00Z"
        )
        == first
    )


def test_distinct_semantic_keys_may_share_transaction_second():
    authority, publisher = composed()
    first = publisher.publish(raw("MARKET_DATA_FRESHNESS"), now_utc="2030-01-01T00:00:00Z")
    second = publisher.publish(raw("EXECUTION_PATH_HEALTH", seq=2), now_utc="2030-01-01T00:00:00Z")
    assert (first.transaction_revision, second.transaction_revision) == (1, 2)
    assert (
        authority.validate_historical_semantic_effective_acceptance(
            first.acceptance_id, at_utc="2030-01-01T00:00:00Z"
        )
        == first
    )
    assert (
        authority.validate_historical_semantic_effective_acceptance(
            second.acceptance_id, at_utc="2030-01-01T00:00:00Z"
        )
        == second
    )


def test_expired_semantic_winner_never_falls_back_to_older_runtime():
    authority, publisher = composed()
    older = publisher.publish(
        raw(observed="2030-01-01T00:00:04Z", expires_at_utc="2030-01-01T00:00:34Z"),
        now_utc="2030-01-01T00:00:00Z",
    )
    publisher.publish(
        raw(
            seq=1,
            source_instance_id="run_018f1f10-7b2c-7abc-8def-123456789abd",
            observation_id="run-b",
        ),
        now_utc="2030-01-01T00:00:01Z",
    )
    assert authority.resolve_current(older.key, now_utc="2030-01-01T00:00:31Z").status == "FRESH"
    with pytest.raises(ValueError, match="OBSERVATION_NOT_EFFECTIVE_CURRENT"):
        authority.consume_semantic_effective_current(
            ObservationSemanticKey.from_observation(older.observation),
            now_utc="2030-01-01T00:00:31Z",
            consumer=lambda item: item,
        )


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
def test_cross_runtime_transaction_time_rollback_is_atomic(category):
    carrier = InMemoryObservationAuthorityCarrier()
    authority, publisher = composed(carrier)
    first = publisher.publish(
        raw(category, observed="2030-01-01T00:00:10Z", expires_at_utc="2030-01-01T00:00:40Z"),
        now_utc="2030-01-01T00:00:10Z",
    )
    before = carrier.read()
    candidate = raw(
        category,
        seq=1,
        source_instance_id="run_018f1f10-7b2c-7abc-8def-123456789abd",
        observed="2030-01-01T00:00:09Z",
        expires_at_utc="2030-01-01T00:00:39Z",
        observation_id="run-b",
    )
    with pytest.raises(ValueError, match="TRANSACTION_TIME_ROLLBACK"):
        publisher.publish(candidate, now_utc="2030-01-01T00:00:09Z")
    assert carrier.read() == before
    seen = []
    authority.consume_semantic_effective_current(
        ObservationSemanticKey.from_observation(first.observation),
        now_utc="2030-01-01T00:00:11Z",
        consumer=seen.append,
    )
    assert seen == [first]


def test_same_runtime_rollback_rejects_but_distinct_semantic_same_second_and_replay_remain_valid():
    carrier = InMemoryObservationAuthorityCarrier()
    authority, publisher = composed(carrier)
    original = raw(observed="2030-01-01T00:00:09Z", expires_at_utc="2030-01-01T00:00:39Z")
    first = publisher.publish(original, now_utc="2030-01-01T00:00:10Z")
    rollback = raw(
        seq=2,
        observed="2030-01-01T00:00:10Z",
        expires_at_utc="2030-01-01T00:00:40Z",
        observation_id="next",
    )
    with pytest.raises(ValueError, match="TRANSACTION_TIME_ROLLBACK"):
        publisher.publish(rollback, now_utc="2030-01-01T00:00:09Z")
    assert publisher.publish(original, now_utc="malformed") == first
    second = publisher.publish(
        raw(
            "EXECUTION_PATH_HEALTH",
            seq=2,
            observed="2030-01-01T00:00:10Z",
            expires_at_utc="2030-01-01T00:00:40Z",
        ),
        now_utc="2030-01-01T00:00:10Z",
    )
    assert second.transaction_revision == 2
    assert (
        authority.validate_historical_semantic_effective_acceptance(
            second.acceptance_id, at_utc="2030-01-01T00:00:10Z"
        )
        == second
    )


def test_restore_rejects_coherently_resealed_transaction_time_rollback():
    carrier = InMemoryObservationAuthorityCarrier()
    authority, publisher = composed(carrier)
    first = publisher.publish(
        raw(observed="2030-01-01T00:00:08Z", expires_at_utc="2030-01-01T00:00:38Z"),
        now_utc="2030-01-01T00:00:09Z",
    )
    second = publisher.publish(
        raw(
            seq=2,
            observed="2030-01-01T00:00:10Z",
            expires_at_utc="2030-01-01T00:00:40Z",
            observation_id="next",
        ),
        now_utc="2030-01-01T00:00:10Z",
    )
    state = carrier.read()
    backdated = "2030-01-01T00:00:08Z"
    forged_id = authority._acceptance_id(
        2, backdated, second.content_fingerprint, second.freshness_policy_fingerprint_sha256
    )
    forged = replace(second, acceptance_id=forged_id, accepted_at_utc=backdated)
    current = tuple(
        (key, forged_id if value == second.acceptance_id else value) for key, value in state.current
    )
    replay = tuple(
        (key, forged_id if value == second.acceptance_id else value) for key, value in state.replay
    )
    bad = replace(state, accepted=(first, forged), current=current, replay=replay)
    with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
        composed(InMemoryObservationAuthorityCarrier(bad))


def test_restore_rejects_coherently_resealed_semantic_time_collision():
    carrier = InMemoryObservationAuthorityCarrier()
    authority, publisher = composed(carrier)
    first = publisher.publish(
        raw(observed="2030-01-01T00:00:08Z", expires_at_utc="2030-01-01T00:00:38Z"),
        now_utc="2030-01-01T00:00:09Z",
    )
    second = publisher.publish(
        raw(
            seq=2,
            observed="2030-01-01T00:00:10Z",
            expires_at_utc="2030-01-01T00:00:40Z",
            observation_id="next",
        ),
        now_utc="2030-01-01T00:00:10Z",
    )
    state = carrier.read()
    collided = first.accepted_at_utc
    forged_id = authority._acceptance_id(
        2, collided, second.content_fingerprint, second.freshness_policy_fingerprint_sha256
    )
    forged = replace(second, acceptance_id=forged_id, accepted_at_utc=collided)
    current = tuple(
        (key, forged_id if value == second.acceptance_id else value) for key, value in state.current
    )
    replay = tuple(
        (key, forged_id if value == second.acceptance_id else value) for key, value in state.replay
    )
    bad = replace(state, accepted=(first, forged), current=current, replay=replay)
    with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
        composed(InMemoryObservationAuthorityCarrier(bad))


def test_cross_instance_visibility_history_and_opposite_linearization():
    carrier = InMemoryObservationAuthorityCarrier()
    a, p = composed(carrier)
    b, q = composed(carrier)
    first = p.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    assert a.resolve_current(first.key, now_utc="2030-01-01T00:00:00Z").accepted == first
    second = q.publish(
        raw(seq=2, observed="2030-01-01T00:00:01Z", observation_id="next"),
        now_utc="2030-01-01T00:00:01Z",
    )
    assert a.resolve_current(first.key, now_utc="2030-01-01T00:00:02Z").accepted == second
    seen = []
    a.consume_effective_current(first.key, now_utc="2030-01-01T00:00:02Z", consumer=seen.append)
    assert seen == [second]
    restored, _ = composed(carrier)
    assert restored.resolve_historical_acceptance(first.acceptance_id) == first
    assert restored.resolve_current(first.key, now_utc="2030-01-01T00:00:02Z").accepted == second


def test_scope_order_canonical_and_changed_scope_exact():
    _, p = composed()
    d = raw()
    d["scope"] = {"instrument_id": "instr_" + U, "market_data_route_id": "mdr_" + U}
    r = p.publish(d, now_utc="2030-01-01T00:00:00Z")
    assert r.key == ObservationKey.exact(
        category=d["category"],
        source_component="core_host",
        source_instance_id="run_" + U,
        environment="TESTNET",
        scope=raw()["scope"],
    )


def test_restore_rejects_corruption_and_fingerprint_is_not_membership():
    carrier = InMemoryObservationAuthorityCarrier()
    a, p = composed(carrier)
    r = p.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    assert a.resolve_historical_acceptance("s9c_" + r.content_fingerprint) is None
    bad = replace(carrier.read(), current=((r.key, "missing"),))
    with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
        composed(InMemoryObservationAuthorityCarrier(bad))


def test_restore_requires_complete_derived_indexes_and_acceptance_id():
    carrier = InMemoryObservationAuthorityCarrier()
    _, p = composed(carrier)
    record = p.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    state = carrier.read()
    corruptions = [
        replace(state, replay=()),
        replace(state, last_sequence=()),
        replace(state, replay=(), last_sequence=()),
        replace(state, current=()),
        replace(state, replay=((state.replay[0][0], "wrong"),)),
    ]
    forged = replace(record, acceptance_id="forged")
    corruptions.append(
        replace(
            state,
            accepted=(forged,),
            current=((record.key, "forged"),),
            replay=((state.replay[0][0], "forged"),),
        )
    )
    for bad in corruptions:
        with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
            composed(InMemoryObservationAuthorityCarrier(bad))


def test_safe_scalar_and_canonical_correlation_validation():
    for change, error in [
        (
            {"value": {"last_data_at_utc": "api_secret=bad", "sequence_state": "CURRENT"}},
            "SECRET_CONTENT",
        ),
        ({"correlation_reference": {"kind": "LOCAL", "value": "token.value"}}, "SECRET_CONTENT"),
        (
            {
                "correlation_reference": {
                    "kind": "CANONICAL",
                    "entity": "Unknown",
                    "value": "run_" + U,
                }
            },
            "INVALID_CORRELATION",
        ),
        (
            {
                "correlation_reference": {
                    "kind": "CANONICAL",
                    "entity": "RuntimeSession",
                    "value": "xacc_" + U,
                }
            },
            "INVALID_CORRELATION",
        ),
    ]:
        _, p = composed()
        with pytest.raises(ValueError, match=error):
            p.publish(raw(**change), now_utc="2030-01-01T00:00:00Z")


def test_writer_capability_has_no_public_constructor():
    import bot_core.observability.authority as production

    assert not hasattr(production, "ObservationPublisher")


def test_m04_binding_and_all_canonical_environments():
    with pytest.raises(ValueError, match="ILLEGAL_ENVIRONMENT"):
        composed(enabled=frozenset({"OTHER"}))
    for environment in ("PAPER", "TESTNET", "LIVE"):
        _, p = composed(enabled=frozenset({environment}))
        accepted = p.publish(raw(environment=environment), now_utc="2030-01-01T00:00:00Z")
        assert accepted.observation.environment == environment


def test_policy_provenance_restore_evolution_and_replay():
    carrier = InMemoryObservationAuthorityCarrier()
    v1 = policy()
    _, p = composed(carrier, policies=(v1,), enabled=frozenset({"TESTNET"}))
    original = p.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    revision = carrier.read().store_revision
    v2 = replace(v1, version=2, validity_horizon_seconds=40)
    restored, q = composed(carrier, policies=(v2,), enabled=frozenset({"TESTNET"}))
    assert restored.resolve_historical_acceptance(original.acceptance_id) == original
    assert q.publish(raw(), now_utc="2030-01-01T00:00:01Z") == original
    assert carrier.read().store_revision == revision
    successor = q.publish(
        raw(
            seq=2,
            observed="2030-01-01T00:00:01Z",
            expires_at_utc="2030-01-01T00:00:41Z",
            observation_id="next",
        ),
        now_utc="2030-01-01T00:00:01Z",
    )
    assert (
        successor.freshness_policy.version == 2
        and successor.freshness_policy_fingerprint_sha256
        != original.freshness_policy_fingerprint_sha256
    )
    assert successor.acceptance_id != original.acceptance_id


def test_restore_rejects_coherently_resealed_semantically_illegal_policies():
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    record = publisher.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    state = carrier.read()
    mutations = [
        {"policy_id": "OTHER_POLICY"},
        {"policy_id": "MALFORMED/POLICY"},
        {"version": 0},
        {"version": True},
        {"validity_horizon_seconds": True},
        {"allowed_observed_future_skew_seconds": True},
        {"allowed_source_event_future_skew_seconds": True},
        {"allowed_ingest_before_observed_skew_seconds": True},
        {"allowed_observed_future_skew_seconds": -1},
        {"category": "EXECUTION_PATH_HEALTH"},
        {"source_class": "tray_agent"},
    ]
    for changes in mutations:
        forged_policy = replace(record.freshness_policy, **changes)
        policy_fingerprint = ObservationAuthority._policy_fingerprint(forged_policy)
        acceptance_id = ObservationAuthority._acceptance_id(
            record.transaction_revision,
            record.accepted_at_utc,
            record.content_fingerprint,
            policy_fingerprint,
        )
        forged = replace(
            record,
            acceptance_id=acceptance_id,
            freshness_policy=forged_policy,
            freshness_policy_fingerprint_sha256=policy_fingerprint,
        )
        replay_key = state.replay[0][0]
        resealed = replace(
            state,
            accepted=(forged,),
            current=((record.key, acceptance_id),),
            replay=((replay_key, acceptance_id),),
        )
        with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
            composed(InMemoryObservationAuthorityCarrier(resealed))


def _append_forged_transition(authority, state, raw_observation, *, now="2030-01-01T00:00:01Z"):
    observation = authority._canonical(raw_observation, now)
    policy = state.accepted[0].freshness_policy
    fingerprint = authority._fingerprint(observation)
    policy_fingerprint = authority._policy_fingerprint(policy)
    revision = state.store_revision + 1
    acceptance_id = authority._acceptance_id(revision, now, fingerprint, policy_fingerprint)
    record = AcceptedObservation(
        acceptance_id,
        revision,
        now,
        observation,
        authority._key(observation),
        observation.condition,
        (observation.reason_code,),
        False,
        fingerprint,
        policy,
        policy_fingerprint,
    )
    current = dict(state.current)
    current[record.key] = acceptance_id
    replay = dict(state.replay)
    last = dict(state.last_sequence)
    if observation.source_sequence is not None:
        source = (observation.source_component, observation.source_instance_id)
        replay[(*source, observation.source_sequence)] = acceptance_id
        last[source] = observation.source_sequence
    return replace(
        state,
        store_revision=revision,
        accepted=state.accepted + (record,),
        current=tuple(current.items()),
        replay=tuple(replay.items()),
        last_sequence=tuple(last.items()),
    )


def test_restore_replays_clock_transition_rules_and_keeps_distinct_keys_independent():
    carrier = InMemoryObservationAuthorityCarrier()
    authority, publisher = composed(carrier)
    publisher.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    earlier = raw(
        seq=2,
        observed="2029-12-31T23:59:59Z",
        ingested_at_utc="2029-12-31T23:59:59Z",
        source_event_at_utc="2029-12-31T23:59:59Z",
        expires_at_utc="2030-01-01T00:00:29Z",
        observation_id="earlier",
    )
    forged = _append_forged_transition(authority, carrier.read(), earlier)
    with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
        composed(InMemoryObservationAuthorityCarrier(forged))
    different = raw(
        seq=2,
        observed="2029-12-31T23:59:59Z",
        ingested_at_utc="2029-12-31T23:59:59Z",
        source_event_at_utc="2029-12-31T23:59:59Z",
        expires_at_utc="2030-01-01T00:00:29Z",
        observation_id="different",
        scope={
            "market_data_route_id": "mdr_018f1f10-7b2c-7abc-8def-123456789abd",
            "instrument_id": "instr_" + U,
        },
    )
    # Exact-key clocks are independent; source sequence remains source-session global.
    publisher.publish(different, now_utc="2030-01-01T00:00:01Z")
    composed(carrier)


@pytest.mark.parametrize("kind", ["equal_clock", "source_event_regression"])
def test_restore_replays_unsequenced_transition_rules(kind):
    carrier = InMemoryObservationAuthorityCarrier()
    authority, publisher = composed(carrier)
    publisher.publish(raw(seq=None), now_utc="2030-01-01T00:00:00Z")
    if kind == "equal_clock":
        second = raw(seq=None, observation_id="equal")
    else:
        second = raw(
            seq=None,
            observation_id="source-regression",
            observed="2030-01-01T00:00:01Z",
            expires_at_utc="2030-01-01T00:00:31Z",
            source_event_at_utc="2029-12-31T23:59:59Z",
        )
    forged = _append_forged_transition(authority, carrier.read(), second)
    with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
        composed(InMemoryObservationAuthorityCarrier(forged))


def test_positive_transition_histories_restore():
    for sequences in ((1, 2), (1, 3)):
        carrier = InMemoryObservationAuthorityCarrier()
        _, publisher = composed(carrier)
        publisher.publish(raw(seq=sequences[0]), now_utc="2030-01-01T00:00:00Z")
        last = publisher.publish(
            raw(
                seq=sequences[1],
                observed="2030-01-01T00:00:01Z",
                expires_at_utc="2030-01-01T00:00:31Z",
                source_event_at_utc="2030-01-01T00:00:01Z",
                observation_id="next",
            ),
            now_utc="2030-01-01T00:00:01Z",
        )
        restored, _ = composed(carrier)
        assert restored.resolve_current(last.key, now_utc="2030-01-01T00:00:02Z").accepted == last
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    publisher.publish(raw(seq=None), now_utc="2030-01-01T00:00:00Z")
    publisher.publish(
        raw(
            seq=None,
            observed="2030-01-01T00:00:01Z",
            expires_at_utc="2030-01-01T00:00:31Z",
            source_event_at_utc="2030-01-01T00:00:01Z",
            observation_id="next",
        ),
        now_utc="2030-01-01T00:00:01Z",
    )
    composed(carrier)


def test_environment_binding_rejects_mutable_alias():
    mutable = {"PAPER", "TESTNET", "LIVE"}
    with pytest.raises(ValueError, match="INVALID_M04_ENVIRONMENT_BINDING"):
        FrozenEnvironmentRegistryBinding(canonical_environments=mutable)


@pytest.mark.parametrize(
    "field,value",
    [
        ("transaction_revision", True),
        ("transaction_revision", 1.0),
        ("sequence_gap", 0),
        ("sequence_gap", 0.0),
    ],
)
def test_restore_rejects_record_numeric_aliases(field, value):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    record = publisher.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    state = carrier.read()
    corrupted = replace(record, **{field: value})
    with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
        composed(InMemoryObservationAuthorityCarrier(replace(state, accepted=(corrupted,))))


@pytest.mark.parametrize("value", [1, 1.0])
def test_restore_rejects_truthy_gap_numeric_aliases(value):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    publisher.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    gap = publisher.publish(
        raw(
            seq=3,
            observed="2030-01-01T00:00:01Z",
            expires_at_utc="2030-01-01T00:00:31Z",
            observation_id="gap",
        ),
        now_utc="2030-01-01T00:00:01Z",
    )
    state = carrier.read()
    corrupted = replace(gap, sequence_gap=value)
    with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
        composed(
            InMemoryObservationAuthorityCarrier(
                replace(state, accepted=(state.accepted[0], corrupted))
            )
        )


@pytest.mark.parametrize("alias", [True, 1.0])
def test_restore_rejects_projection_sequence_aliases(alias):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    publisher.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    state = carrier.read()
    source = state.last_sequence[0][0]
    acceptance = state.replay[0][1]
    for corrupted in (
        replace(state, replay=(((source[0], source[1], alias), acceptance),)),
        replace(state, last_sequence=((source, alias),)),
    ):
        with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
            composed(InMemoryObservationAuthorityCarrier(corrupted))


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize("original,changed", [(1, 1.0), (0, False), (1, True)])
def test_exact_replay_distinguishes_json_scalar_types(category, original, changed):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    item = raw(category)
    field = next(iter(item["value"]))
    item["value"][field] = original
    publisher.publish(item, now_utc="2030-01-01T00:00:00Z")
    conflicting = {**item, "value": {**item["value"], field: changed}}
    with pytest.raises(ValueError, match="DUPLICATE_SEQUENCE_CONFLICT"):
        publisher.publish(conflicting, now_utc="2030-01-01T00:00:01Z")


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
def test_exact_replay_is_order_independent_and_does_not_mutate(category):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    item = raw(category)
    accepted = publisher.publish(item, now_utc="2030-01-01T00:00:00Z")
    before = carrier.read()
    reordered = dict(reversed(list(item.items())))
    reordered["scope"] = dict(reversed(list(item["scope"].items())))
    reordered["value"] = dict(reversed(list(item["value"].items())))
    assert publisher.publish(reordered, now_utc="2031-01-01T00:00:00Z") == accepted
    assert carrier.read() == before


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize(
    "changes",
    [
        {"observed_at_utc": "malformed"},
        {"ingested_at_utc": "malformed"},
        {"expires_at_utc": "2030-01-01T00:00:29Z"},
        {"source_event_at_utc": "malformed"},
        {"observed_at_utc": "2031-01-01T00:00:00Z"},
        {"source_event_at_utc": "2031-01-01T00:00:00Z"},
        {"ingested_at_utc": "2029-12-31T23:59:00Z"},
        {"freshness_policy_id": "UNKNOWN"},
    ],
)
def test_known_sequence_conflict_precedes_temporal_freshness_validation(category, changes):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    item = raw(category)
    publisher.publish(item, now_utc="2030-01-01T00:00:00Z")
    before = carrier.read()
    with pytest.raises(ValueError, match="DUPLICATE_SEQUENCE_CONFLICT"):
        publisher.publish({**item, **changes}, now_utc="malformed")
    assert carrier.read() == before


def test_exact_replay_ignores_malformed_now_and_disabled_current_environment():
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier, enabled=frozenset({"TESTNET"}))
    item = raw()
    accepted = publisher.publish(item, now_utc="2030-01-01T00:00:00Z")
    _, restricted = composed(carrier, enabled=frozenset({"PAPER"}))
    before = carrier.read()
    assert restricted.publish(item, now_utc="malformed") == accepted
    assert carrier.read() == before


@pytest.mark.parametrize(
    "field,value,error",
    [("source_component", [], "WRONG_SOURCE"), ("source_instance_id", [], "WRONG_SOURCE_INSTANCE")],
)
def test_unhashable_replay_source_fields_are_controlled(field, value, error):
    _, publisher = composed()
    with pytest.raises(ValueError, match=error):
        publisher.publish(raw(**{field: value}), now_utc="2030-01-01T00:00:00Z")


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize(
    "changes,error",
    [
        ({"category": []}, "UNKNOWN_CATEGORY"),
        ({"category": {}}, "UNKNOWN_CATEGORY"),
        ({"environment": []}, "ILLEGAL_ENVIRONMENT"),
        ({"environment": {}}, "ILLEGAL_ENVIRONMENT"),
        ({"freshness_policy_id": []}, "UNKNOWN_FRESHNESS_POLICY"),
        ({"freshness_policy_id": {}}, "UNKNOWN_FRESHNESS_POLICY"),
        ({"freshness_policy_id": 1}, "UNKNOWN_FRESHNESS_POLICY"),
        ({"freshness_policy_id": True}, "UNKNOWN_FRESHNESS_POLICY"),
        (
            {"correlation_reference": {"kind": "CANONICAL", "entity": [], "value": "run_" + U}},
            "INVALID_CORRELATION",
        ),
        (
            {"correlation_reference": {"kind": "CANONICAL", "entity": {}, "value": "run_" + U}},
            "INVALID_CORRELATION",
        ),
    ],
)
def test_ingress_hash_lookup_fields_fail_with_controlled_errors(category, changes, error):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    before = carrier.read()
    item = raw(category)
    item.update(changes)
    with pytest.raises(ValueError, match=error):
        publisher.publish(item, now_utc="2030-01-01T00:00:00Z")
    assert carrier.read() == before


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected_without_state_change(category, nonfinite):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    item = raw(category)
    field = next(iter(item["value"]))
    item["value"][field] = nonfinite
    before = carrier.read()
    with pytest.raises(ValueError, match="INVALID_VALUE"):
        publisher.publish(item, now_utc="2030-01-01T00:00:00Z")
    assert carrier.read() == before


@pytest.mark.parametrize("finite", [0.0, 1.5, -1.5, 1e-6])
def test_finite_decimal_values_have_deterministic_fingerprints(finite):
    item = raw()
    field = next(iter(item["value"]))
    item["value"][field] = finite
    _, first = composed()
    accepted = first.publish(item, now_utc="2030-01-01T00:00:00Z")
    _, second = composed()
    again = second.publish(item, now_utc="2030-01-01T00:00:00Z")
    assert accepted.content_fingerprint == again.content_fingerprint


def test_canonical_json_serializer_is_strict_for_nonfinite_numbers():
    import bot_core.observability.authority as production

    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            production._json({"value": value})


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_restore_rejects_nonfinite_historical_observation(nonfinite):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    accepted = publisher.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    state = carrier.read()
    field = accepted.observation.value[0][0]
    forged_observation = replace(
        accepted.observation, value=((field, nonfinite),) + accepted.observation.value[1:]
    )
    forged = replace(accepted, observation=forged_observation)
    with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
        composed(InMemoryObservationAuthorityCarrier(replace(state, accepted=(forged,))))


INVALID_TIMESTAMPS = [
    "2030-01-01T00:00Z",
    "2030-01-01T00Z",
    "2030-01-01Z",
    "2030-W01-1T00:00:00Z",
    "20300101T000000Z",
    "2030-01-01T000000Z",
    "20300101T00:00:00Z",
    "2030-02-30T00:00:00Z",
    "2030-13-01T00:00:00Z",
    "2030-01-01T24:00:00Z",
    "2030-01-01T00:60:00Z",
    "2030-01-01T00:00:61Z",
]


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize("timestamp", INVALID_TIMESTAMPS)
def test_unseen_publication_rejects_noncanonical_timestamps(category, timestamp):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    before = carrier.read()
    with pytest.raises(ValueError, match="MALFORMED_TIMESTAMP"):
        publisher.publish(raw(category, observed_at_utc=timestamp), now_utc="2030-01-01T00:00:00Z")
    assert carrier.read() == before


@pytest.mark.parametrize(
    "field,timestamp",
    [
        ("observed_at_utc", "2030-01-01T00:00Z"),
        ("observed_at_utc", "2030-W01-1T00:00:00Z"),
        ("ingested_at_utc", "20300101T000000Z"),
        ("expires_at_utc", "2030-01-01T00Z"),
        ("source_event_at_utc", "2030-01-01Z"),
    ],
)
def test_restore_rejects_coherently_resealed_noncanonical_timestamps(field, timestamp):
    carrier = InMemoryObservationAuthorityCarrier()
    authority, publisher = composed(carrier)
    accepted = publisher.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    state = carrier.read()
    observation = replace(accepted.observation, **{field: timestamp})
    fingerprint = authority._fingerprint(observation)
    acceptance_id = authority._acceptance_id(
        1, accepted.accepted_at_utc, fingerprint, accepted.freshness_policy_fingerprint_sha256
    )
    forged = replace(
        accepted,
        observation=observation,
        content_fingerprint=fingerprint,
        acceptance_id=acceptance_id,
    )
    resealed = replace(
        state,
        accepted=(forged,),
        current=((accepted.key, acceptance_id),),
        replay=((state.replay[0][0], acceptance_id),),
    )
    with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
        composed(InMemoryObservationAuthorityCarrier(resealed))


def test_restore_rejects_noncanonical_acceptance_time():
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    accepted = publisher.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    state = carrier.read()
    with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
        composed(
            InMemoryObservationAuthorityCarrier(
                replace(state, accepted=(replace(accepted, accepted_at_utc="2030-01-01T00:00Z"),))
            )
        )


@pytest.mark.parametrize("query_time", ["2030-W01-1T00:00:00Z", "2030-01-01T00:00Z"])
def test_current_queries_reject_noncanonical_query_time(query_time):
    authority, publisher = composed()
    accepted = publisher.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="MALFORMED_TIMESTAMP"):
        authority.resolve_current(accepted.key, now_utc=query_time)
    with pytest.raises(ValueError, match="MALFORMED_TIMESTAMP"):
        authority.consume_effective_current(
            accepted.key, now_utc=query_time, consumer=lambda _: None
        )


def test_positive_canonical_timestamp_boundaries():
    _, publisher = composed()
    assert (
        publisher.publish(raw(), now_utc="2030-12-31T23:59:59Z").observation.observed_at_utc
        == "2030-01-01T00:00:00Z"
    )


@pytest.mark.parametrize(
    "changed", ["2029-12-31T23:59:59Z", "2030-01-01T00:00:01Z", "2030-01-01T00:00:04Z"]
)
def test_acceptance_identity_binds_acceptance_time(changed):
    carrier = InMemoryObservationAuthorityCarrier()
    authority, publisher = composed(carrier)
    accepted = publisher.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    state = carrier.read()
    assert (
        authority._acceptance_id(
            1, changed, accepted.content_fingerprint, accepted.freshness_policy_fingerprint_sha256
        )
        != accepted.acceptance_id
    )
    with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
        composed(
            InMemoryObservationAuthorityCarrier(
                replace(state, accepted=(replace(accepted, accepted_at_utc=changed),))
            )
        )


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize("digits", [4301, 5001])
def test_oversized_integer_values_fail_closed(category, digits):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    item = raw(category)
    item["value"][next(iter(item["value"]))] = 10**digits
    before = carrier.read()
    with pytest.raises(ValueError, match="INVALID_VALUE"):
        publisher.publish(item, now_utc="2030-01-01T00:00:00Z")
    assert carrier.read() == before


@pytest.mark.parametrize(
    "field",
    [
        "validity_horizon_seconds",
        "allowed_observed_future_skew_seconds",
        "allowed_source_event_future_skew_seconds",
        "allowed_ingest_before_observed_skew_seconds",
    ],
)
def test_unrepresentable_policy_durations_fail_at_composition(field):
    huge = 10**1000
    with pytest.raises(ValueError, match="INVALID_FRESHNESS_POLICY"):
        composed(policies=(replace(policy(), **{field: huge}),))


def test_oversized_policy_version_fails_at_composition():
    with pytest.raises(ValueError, match="INVALID_FRESHNESS_POLICY"):
        composed(policies=(replace(policy(), version=10**5000),))


@pytest.mark.parametrize("changes", [{"validity_horizon_seconds": 10**1000}, {"version": 10**5000}])
def test_restore_rejects_unrepresentable_historical_policy_numbers(changes):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    accepted = publisher.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    state = carrier.read()
    forged = replace(accepted, freshness_policy=replace(accepted.freshness_policy, **changes))
    with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
        composed(InMemoryObservationAuthorityCarrier(replace(state, accepted=(forged,))))


@pytest.mark.parametrize("category", ["MARKET_DATA_FRESHNESS", "EXECUTION_PATH_HEALTH"])
@pytest.mark.parametrize(
    "sequence",
    [10**4301, 10**5001, True, 1.0, -1],
    ids=["digits4302", "digits5002", "bool", "float", "negative"],
)
def test_invalid_or_unrepresentable_source_sequence_fails_closed(category, sequence):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    before = carrier.read()
    with pytest.raises(ValueError, match="INVALID_SOURCE_SEQUENCE"):
        publisher.publish(raw(category, seq=sequence), now_utc="2030-01-01T00:00:00Z")
    assert carrier.read() == before


@pytest.mark.parametrize("sequence", [10**4301, 10**5001], ids=["digits4302", "digits5002"])
def test_restore_rejects_unrepresentable_historical_source_sequence(sequence):
    carrier = InMemoryObservationAuthorityCarrier()
    _, publisher = composed(carrier)
    accepted = publisher.publish(raw(), now_utc="2030-01-01T00:00:00Z")
    state = carrier.read()
    forged = replace(accepted, observation=replace(accepted.observation, source_sequence=sequence))
    with pytest.raises(ValueError, match="CORRUPT_AUTHORITY_STATE"):
        composed(InMemoryObservationAuthorityCarrier(replace(state, accepted=(forged,))))
