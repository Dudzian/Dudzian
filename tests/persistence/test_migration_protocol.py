from dataclasses import replace

import pytest

from bot_core.persistence.state_store import StateStoreMetadata, StateStoreSnapshot
from bot_core.persistence.migration_execution import MigrationExecutionAuthority

from bot_core.persistence.migration_protocol import (
    MigrationCoordinator,
    MigrationDefinition,
    MigrationError,
    MigrationRecord,
    MigrationRegistry,
    migration_current,
    migration_transition,
    validate_migration_lifecycle,
    bind_runtime_migration_instance,
)

H = "a" * 64
ACCOUNT = "acct_01890f3a-2b4c-7abc-8def-0123456789ab"
DEVICE = "dev_01890f3a-2b4c-7abc-8def-0123456789ab"
ACCOUNT_B = "acct_01890f3a-2b4c-7abc-8def-0123456789ac"
DEVICE_B = "dev_01890f3a-2b4c-7abc-8def-0123456789ac"


def definition():
    return MigrationDefinition("migration-1", 1, 2, ("test-step",))


def registry(step=lambda value: value):
    value = definition()
    authority = MigrationExecutionAuthority(
        value.migration_id,
        value.source_schema_version,
        value.target_schema_version,
        value.ordered_path,
        value.rollback_policy,
        value.fingerprint(),
        H,
        H,
        H,
    )
    return MigrationRegistry(((value, authority, step),))


def plan(**changes):
    values = dict(
        migration_id="migration-1",
        source_schema_version=1,
        target_schema_version=2,
        ordered_path=("test-step",),
        scope=(ACCOUNT, DEVICE),
        environment="PAPER",
        pre_state_fingerprint_sha256=H,
        post_state_fingerprint_sha256=H,
        transaction_fingerprint_sha256=H,
        protected_freshness_generation=2,
        rollback_policy="FORWARD_ONLY",
    )
    values.update(changes)
    return MigrationRecord(**values)


def lifecycle(states=("PREPARED",)):
    history = []
    for revision, state in enumerate(states, 1):
        history.append(
            migration_transition(
                migration_id="migration-1",
                transition_revision=revision,
                previous_state=None if revision == 1 else states[revision - 2],
                state=state,
                transaction_fingerprint_sha256=H,
                state_fingerprint_sha256=H,
                protected_freshness_generation=2,
            )
        )
    current = migration_current(
        migration_id="migration-1",
        current_transition_revision=len(history),
        state=states[-1],
        authoritative_state_fingerprint_sha256=H,
        protected_freshness_generation=2,
    )
    return history, current


@pytest.mark.parametrize(
    "states",
    [
        ("PREPARED",),
        ("PREPARED", "APPLYING"),
        ("PREPARED", "APPLYING", "DURABLE_MIGRATED"),
        ("PREPARED", "APPLYING", "DURABLE_MIGRATED", "COMPLETED"),
        ("PREPARED", "FAILED"),
        ("PREPARED", "APPLYING", "FAILED"),
    ],
)
def test_valid_frozen_transitions(states):
    validate_migration_lifecycle(plan(), *lifecycle(states))


def test_corruption_gap_illegal_and_mismatch_fail_closed():
    history, current = lifecycle(("PREPARED", "APPLYING"))
    for corrupt in (
        [history[0], {**history[1], "transition_revision": 3}],
        [
            history[0],
            migration_transition(
                migration_id="migration-1",
                transition_revision=2,
                previous_state="PREPARED",
                state="COMPLETED",
                transaction_fingerprint_sha256=H,
                state_fingerprint_sha256=H,
                protected_freshness_generation=2,
            ),
        ],
        [{**history[0], "transition_fingerprint_sha256": "b" * 64}],
    ):
        with pytest.raises(MigrationError):
            validate_migration_lifecycle(plan(), corrupt, current)
    with pytest.raises(MigrationError):
        validate_migration_lifecycle(plan(), history, {**current, "state": "FAILED"})
    with pytest.raises(MigrationError):
        validate_migration_lifecycle(
            plan(), history, {**current, "designation_fingerprint_sha256": "b" * 64}
        )


def test_exact_registry_and_recovery_do_not_reapply_durable_or_terminal():
    coordinator = MigrationCoordinator(registry())
    calls = []
    history, current = lifecycle(("PREPARED", "APPLYING"))
    with pytest.raises(MigrationError, match="sealed execution coordinator"):
        coordinator.resume(
            plan(),
            history,
            current,
            verified_schema_version=1,
            apply_once=lambda step: calls.append(step),
            finalize=lambda: calls.append("finalize"),
        )
    assert calls == []
    history, current = lifecycle(("PREPARED", "APPLYING", "DURABLE_MIGRATED"))
    assert (
        coordinator.resume(
            plan(),
            history,
            current,
            verified_schema_version=2,
            apply_once=lambda step: calls.append(step),
            finalize=lambda: calls.append("finalize"),
        )
        == "COMPLETED"
    )
    assert len(calls) == 1
    history, current = lifecycle(("PREPARED", "APPLYING", "DURABLE_MIGRATED", "COMPLETED"))
    assert (
        coordinator.resume(
            plan(),
            history,
            current,
            verified_schema_version=2,
            apply_once=lambda step: calls.append(step),
            finalize=lambda: calls.append("finalize"),
        )
        == "COMPLETED"
    )
    assert len(calls) == 1


def test_descriptor_is_not_authority_and_unknown_path_denied():
    coordinator = MigrationCoordinator(MigrationRegistry())
    with pytest.raises(MigrationError):
        coordinator.resume(
            plan(),
            [],
            None,
            verified_schema_version=1,
            apply_once=lambda _: None,
            finalize=lambda: None,
        )
    history, current = lifecycle(("PREPARED", "APPLYING"))
    with pytest.raises(MigrationError):
        coordinator.resume(
            plan(),
            history,
            current,
            verified_schema_version=1,
            apply_once=lambda _: None,
            finalize=lambda: None,
        )


def test_lifecycle_carriers_pass_stage_one_and_use_existing_buckets():
    from bot_core.persistence.migration_protocol import (
        migration_current_carrier,
        migration_transition_carrier,
    )

    history, current = lifecycle()
    assert (
        migration_transition_carrier(history[0]).record_key == "migration-transition:migration-1:1"
    )
    assert migration_current_carrier(current).record_key == "migration-current:migration-1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_schema_version", 9),
        ("target_schema_version", 9),
        ("ordered_path", ("other",)),
        ("rollback_policy", "OTHER"),
    ],
)
def test_static_definition_rejects_substitution(field, value):
    with pytest.raises(MigrationError):
        registry().assert_static_match(plan(**{field: value}))


def snapshot(
    *,
    account=ACCOUNT,
    device=DEVICE,
    environment="PAPER",
    generation=2,
    state=H,
    transaction=H,
):
    metadata = StateStoreMetadata(
        account,
        device,
        1,
        "d" * 64,
        environment,
        generation,
        state,
        transaction,
        "e" * 64,
    )
    return StateStoreSnapshot(metadata, (), (), ())


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", (ACCOUNT_B, DEVICE)),
        ("scope", (ACCOUNT, DEVICE_B)),
        ("environment", "LIVE"),
        ("pre_state_fingerprint_sha256", "b" * 64),
        ("transaction_fingerprint_sha256", "b" * 64),
        ("protected_freshness_generation", 3),
    ],
)
def test_runtime_source_binding_rejects_substitution(field, value):
    with pytest.raises(MigrationError):
        bind_runtime_migration_instance(
            definition(),
            plan(**{field: value}),
            snapshot(),
            derived_post_state_fingerprint_sha256=H,
        )


def test_target_binding_rejects_caller_substitution():
    with pytest.raises(MigrationError):
        bind_runtime_migration_instance(
            definition(),
            plan(post_state_fingerprint_sha256="b" * 64),
            snapshot(),
            derived_post_state_fingerprint_sha256=H,
        )


def test_one_static_definition_supports_multiple_runtime_installations():
    a = snapshot(
        account=ACCOUNT,
        device=DEVICE,
        generation=5,
        state="1" * 64,
        transaction="2" * 64,
    )
    b = snapshot(
        account=ACCOUNT_B,
        device=DEVICE_B,
        environment="LIVE",
        generation=17,
        state="3" * 64,
        transaction="4" * 64,
    )
    for source, post in ((a, "5" * 64), (b, "6" * 64)):
        metadata = source.metadata
        candidate = plan(
            scope=(metadata.account_id, metadata.device_installation_id),
            environment=metadata.environment,
            pre_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
            transaction_fingerprint_sha256=metadata.transaction_fingerprint_sha256,
            protected_freshness_generation=metadata.protected_freshness_generation,
            post_state_fingerprint_sha256=post,
        )
        bind_runtime_migration_instance(
            definition(), candidate, source, derived_post_state_fingerprint_sha256=post
        )


def test_empty_registry_and_path_only_authority_fail_closed():
    with pytest.raises(MigrationError):
        MigrationRegistry().resolve(plan())


@pytest.mark.parametrize("state", ["PREPARED", "DURABLE_MIGRATED", "COMPLETED", "FAILED"])
def test_static_substitution_rejected_before_all_lifecycle_states(state):
    states = {
        "PREPARED": ("PREPARED",),
        "DURABLE_MIGRATED": ("PREPARED", "APPLYING", "DURABLE_MIGRATED"),
        "COMPLETED": ("PREPARED", "APPLYING", "DURABLE_MIGRATED", "COMPLETED"),
        "FAILED": ("PREPARED", "FAILED"),
    }[state]
    history, current = lifecycle(states)
    coordinator = MigrationCoordinator(registry())
    with pytest.raises(MigrationError):
        coordinator.resume(
            plan(target_schema_version=9),
            history,
            current,
            verified_schema_version=1,
            apply_once=lambda _: None,
            finalize=lambda: None,
        )
