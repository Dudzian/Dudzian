from concurrent.futures import ThreadPoolExecutor

import pytest

from bot_core.persistence.secret_handoff import (
    ExternalOutcome,
    SecretHandoffCoordinator,
    SecretHandoffError,
    SecretHandoffRecord,
    secret_metadata_fingerprint,
    secret_operation_fingerprint,
    handoff_current,
    handoff_transition,
    validate_secret_handoff_lifecycle,
)

H = "a" * 64


def descriptor(**changes):
    metadata = changes.pop("reconciliation_metadata", {"cleanup": True})
    scope = changes.pop("scope", ("acct", "dev"))
    operation = changes.pop("operation", "ROTATE")
    old_reference = changes.pop("old_reference", "ref:old")
    new_reference = changes.pop("new_reference", "ref:new")
    metadata_hash = secret_metadata_fingerprint(metadata)
    operation_hash = secret_operation_fingerprint(
        scope=scope,
        operation=operation,
        old_reference=old_reference,
        new_reference=new_reference,
        metadata_fingerprint_sha256=metadata_hash,
    )
    values = dict(
        handoff_id="handoff-1",
        scope=scope,
        operation=operation,
        old_reference=old_reference,
        new_reference=new_reference,
        metadata_fingerprint_sha256=metadata_hash,
        operation_fingerprint_sha256=operation_hash,
        reconciliation_metadata=metadata,
    )
    values.update(changes)
    return SecretHandoffRecord(**values)


def lifecycle(states=("PREPARED",)):
    bound = descriptor()
    history = []
    for revision, state in enumerate(states, 1):
        history.append(
            handoff_transition(
                handoff_id="handoff-1",
                transition_revision=revision,
                previous_state=None if revision == 1 else states[revision - 2],
                state=state,
                operation_fingerprint_sha256=bound.operation_fingerprint_sha256,
                metadata_fingerprint_sha256=bound.metadata_fingerprint_sha256,
            )
        )
    return history, handoff_current(
        handoff_id="handoff-1",
        current_transition_revision=len(history),
        state=states[-1],
        operation_fingerprint_sha256=bound.operation_fingerprint_sha256,
    )


@pytest.mark.parametrize(
    "states",
    [
        ("PREPARED",),
        ("PREPARED", "COMMITTED"),
        ("PREPARED", "COMMITTED", "CLEANUP_PENDING"),
        ("PREPARED", "UNKNOWN_RECONCILIATION"),
    ],
)
def test_valid_lifecycle(states):
    validate_secret_handoff_lifecycle(descriptor(), *lifecycle(states))


def test_illegal_corrupt_and_descriptor_only_fail_closed():
    history, current = lifecycle()
    illegal = [
        history[0],
        handoff_transition(
            handoff_id="handoff-1",
            transition_revision=2,
            previous_state="PREPARED",
            state="CLEANUP_PENDING",
            operation_fingerprint_sha256=H,
            metadata_fingerprint_sha256=H,
        ),
    ]
    with pytest.raises(SecretHandoffError):
        validate_secret_handoff_lifecycle(descriptor(), illegal, current)
    with pytest.raises(SecretHandoffError):
        validate_secret_handoff_lifecycle(
            descriptor(), [{**history[0], "transition_fingerprint_sha256": "b" * 64}], current
        )
    with pytest.raises(SecretHandoffError):
        SecretHandoffCoordinator(Fake()).resume(descriptor(), [], None)


class Fake:
    def __init__(self, outcome=ExternalOutcome.COMMITTED):
        self.outcome = outcome
        self.mutations = 0
        self.reconciles = 0
        self.cleanups = 0

    def begin(self, descriptor):
        self.mutations += 1
        return self.outcome

    def reconcile(self, descriptor):
        self.reconciles += 1
        return self.outcome

    def cleanup(self, descriptor):
        self.cleanups += 1


def test_lost_ack_reconciles_without_second_mutation_and_unknown_is_terminal():
    fake = Fake()
    coordinator = SecretHandoffCoordinator(fake)
    history, current = lifecycle()
    assert coordinator.resume(descriptor(), history, current, first_dispatch=True) == "COMMITTED"
    assert coordinator.resume(descriptor(), history, current, first_dispatch=True) == "COMMITTED"
    assert fake.mutations == 1 and fake.reconciles == 1
    unknown = Fake(ExternalOutcome.UNRESOLVED)
    c = SecretHandoffCoordinator(unknown)
    assert c.resume(descriptor(), history, current, first_dispatch=True) == "UNKNOWN_RECONCILIATION"
    h, u = lifecycle(("PREPARED", "UNKNOWN_RECONCILIATION"))
    assert c.resume(descriptor(), h, u) == "UNKNOWN_RECONCILIATION" and unknown.mutations == 1


def test_concurrent_resume_mutates_at_most_once():
    fake = Fake()
    coordinator = SecretHandoffCoordinator(fake)
    history, current = lifecycle()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: coordinator.resume(descriptor(), history, current, first_dispatch=True),
                range(2),
            )
        )
    assert results == ["COMMITTED", "COMMITTED"] and fake.mutations == 1


def test_committed_cleanup_does_not_change_descriptor():
    fake = Fake()
    coordinator = SecretHandoffCoordinator(fake)
    record = descriptor()
    history, current = lifecycle(("PREPARED", "COMMITTED"))
    assert coordinator.resume(record, history, current) == "CLEANUP_PENDING"
    assert record.new_reference == "ref:new" and fake.cleanups == 1


def test_lifecycle_carriers_pass_stage_one_and_contain_references_only():
    from bot_core.persistence.secret_handoff import (
        handoff_current_carrier,
        handoff_transition_carrier,
    )

    history, current = lifecycle()
    assert handoff_transition_carrier(history[0]).record_key == "handoff-transition:handoff-1:1"
    assert handoff_current_carrier(current).record_key == "handoff-current:handoff-1"


def test_descriptor_deeply_snapshots_reconciliation_metadata():
    metadata = {"cleanup": True, "nested": [1, {"mode": "exact"}]}
    record = descriptor(reconciliation_metadata=metadata)
    before = record.to_mapping()
    metadata["cleanup"] = False
    metadata["nested"][1]["mode"] = "changed"
    assert record.to_mapping() == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scope", ("other", "dev")),
        ("scope", ("acct", "other")),
        ("operation", "REPLACE"),
        ("old_reference", "ref:different-old"),
        ("new_reference", "ref:different-new"),
        ("reconciliation_metadata", {"cleanup": False}),
    ],
)
def test_secret_descriptor_substitution_fails_against_durable_lifecycle(field, value):
    history, current = lifecycle()
    changed = descriptor(**{field: value})
    with pytest.raises(SecretHandoffError):
        validate_secret_handoff_lifecycle(changed, history, current)


def test_secret_descriptor_rejects_arbitrary_well_formed_hashes():
    with pytest.raises(SecretHandoffError):
        descriptor(metadata_fingerprint_sha256="f" * 64)
    with pytest.raises(SecretHandoffError):
        descriptor(operation_fingerprint_sha256="f" * 64)
