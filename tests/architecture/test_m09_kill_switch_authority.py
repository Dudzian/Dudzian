"""C23-R1 executable M0.9 membership-boundary regressions."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
from pathlib import Path
from threading import Event, Thread

import pytest

from bot_core.m09_kill_switch_authority import (
    AtomicCoreAcceptedContentState,
    AtomicKillSwitchAuthorityState,
    CoreAcceptedContentAuthority,
    CoreAcceptedContentBinding,
    InMemoryCoreAcceptedContentCarrier,
    InMemoryKillSwitchAuthorityCarrier,
    KillSwitchAuthority,
    KillSwitchAuthorityError,
    KillSwitchRecord,
    PrevalidatedKillSwitchContext,
)
from bot_core.persistence.fingerprints import canonical_json_sha256

CONTRACT = Path(__file__).parents[2] / "docs/architecture/cryptohunter_product_architecture/risk_hierarchy_kill_switch_and_execution_lease.json"
WS = "ws_01890f3a-2b4c-7abc-8def-0123456789ab"
OTHER_WS = "ws_01890f3a-2b4c-7abc-8def-0123456789ac"


def raw_record(*, authority_fingerprint: str, state: str = "INACTIVE", generation: int = 1,
               source_revision: int = 1, scope_type: str = "WORKSPACE",
               scope_id: str = WS, environment: str = "TESTNET",
               effective_at_utc: str = "2026-01-01T00:00:00Z") -> KillSwitchRecord:
    candidate = KillSwitchRecord(scope_type, scope_id, environment, state, source_revision,
                                 effective_at_utc, generation,
                                 authority_fingerprint, "0" * 64)
    payload = asdict(candidate)
    payload.pop("record_fingerprint_sha256")
    return replace(candidate, record_fingerprint_sha256=canonical_json_sha256(payload))


class Rig:
    def __init__(self) -> None:
        self.core_carrier = InMemoryCoreAcceptedContentCarrier()
        self.core, self.core_writer = CoreAcceptedContentAuthority.compose(self.core_carrier)
        self.carrier = InMemoryKillSwitchAuthorityCarrier()
        self.authority, self.writer = KillSwitchAuthority.compose(
            self.carrier, core_membership=self.core
        )
        self.authority_binding = CoreAcceptedContentBinding(
            "core-authority-1", canonical_json_sha256({"owner": "CoreHost", "kind": "M0.9"})
        )

    def context(self, *records: KillSwitchRecord, membership_id: str | None = None,
                enroll_authority: bool = True, enroll_history: bool = True) -> PrevalidatedKillSwitchContext:
        if enroll_authority:
            self.core_writer.accept(self.authority_binding)
        history = tuple(records)
        membership = membership_id or f"switch-history-{len(self.core_carrier.read().accepted)}"
        if enroll_history:
            self.core_writer.accept(CoreAcceptedContentBinding(
                membership, KillSwitchAuthority.history_fingerprint(history)
            ))
        provisional = PrevalidatedKillSwitchContext(history, membership, "0" * 64)
        return replace(
            provisional,
            context_fingerprint_sha256=KillSwitchAuthority.context_fingerprint(provisional),
        )

    def record(self, **changes: object) -> KillSwitchRecord:
        return raw_record(
            authority_fingerprint=self.authority_binding.content_fingerprint_sha256, **changes
        )


def test_exact_production_schema_parity_with_frozen_machine_source() -> None:
    frozen = json.loads(CONTRACT.read_text())
    schemas = frozen["executable_boundary_schemas"]
    assert [item.name for item in fields(CoreAcceptedContentBinding)] == schemas["CoreAcceptedContentBinding"]
    assert [item.name for item in fields(PrevalidatedKillSwitchContext)] == schemas["PrevalidatedKillSwitchContext"]
    assert [item.name for item in fields(KillSwitchRecord)] == frozen["kill_switch_contract"]["record_fields"]


def test_exploit_self_hashed_record_with_arbitrary_sha_cannot_self_enroll() -> None:
    rig = Rig()
    forged = raw_record(authority_fingerprint="a" * 64)
    context = rig.context(forged, enroll_authority=False)
    before = rig.carrier.read()
    with pytest.raises(KillSwitchAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        rig.writer.accept(context)
    assert rig.carrier.read() == before == AtomicKillSwitchAuthorityState()
    assert rig.authority.resolve_current(scope_type="WORKSPACE", scope_id=WS, environment="TESTNET") is None
    assert rig.authority.resolve_historical(context.membership_id) is None


def test_nominal_binding_dto_without_carrier_membership_is_rejected() -> None:
    rig = Rig()
    record = rig.record()
    context = rig.context(record, enroll_history=False)
    with pytest.raises(KillSwitchAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        rig.writer.accept(context)


def test_unknown_accepted_authority_fingerprint_is_rejected() -> None:
    rig = Rig()
    unknown = raw_record(authority_fingerprint="b" * 64)
    context = rig.context(unknown)
    assert rig.core.has_content_fingerprint("b" * 64) is False
    with pytest.raises(KillSwitchAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        rig.writer.accept(context)


@pytest.mark.parametrize("defect", ["unknown_membership", "wrong_content", "wrong_authority", "context_fingerprint"])
def test_complete_core_membership_chain_rejects_each_forgery(defect: str) -> None:
    rig = Rig()
    record = rig.record()
    context = rig.context(record)
    if defect == "unknown_membership":
        context = replace(context, membership_id="unknown")
    elif defect == "wrong_content":
        rig = Rig()
        rig.core_writer.accept(rig.authority_binding)
        rig.core_writer.accept(CoreAcceptedContentBinding("known-wrong", "f" * 64))
        context = rig.context(record, membership_id="known-wrong", enroll_history=False)
    elif defect == "wrong_authority":
        wrong = raw_record(authority_fingerprint="b" * 64)
        context = rig.context(wrong)
    else:
        context = replace(context, context_fingerprint_sha256="f" * 64)
    with pytest.raises(KillSwitchAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        rig.writer.accept(context)


@pytest.mark.parametrize("state", ["INACTIVE", "ACTIVE"])
def test_known_exact_core_binding_accepts_both_frozen_states(state: str) -> None:
    rig = Rig()
    context = rig.context(rig.record(state=state))
    accepted = rig.writer.accept(context)
    assert rig.authority.resolve_historical(context.membership_id) == accepted
    assert rig.authority.resolve_current(scope_type="WORKSPACE", scope_id=WS, environment="TESTNET") == accepted


def test_current_lookup_is_exact_for_scope_and_environment() -> None:
    rig = Rig()
    accepted = rig.writer.accept(rig.context(rig.record()))
    assert rig.authority.resolve_current(scope_type="WORKSPACE", scope_id=WS, environment="TESTNET") == accepted
    assert rig.authority.resolve_current(scope_type="WORKSPACE", scope_id=WS, environment="LIVE") is None
    assert rig.authority.resolve_current(scope_type="WORKSPACE", scope_id=OTHER_WS, environment="TESTNET") is None


def test_generation_advance_preserves_superseded_historical_membership() -> None:
    rig = Rig()
    first_context = rig.context(rig.record(generation=1), membership_id="switch-1")
    first = rig.writer.accept(first_context)
    second_context = rig.context(rig.record(state="ACTIVE", generation=2), membership_id="switch-2")
    second = rig.writer.accept(second_context)
    assert rig.authority.resolve_historical("switch-1") == first
    assert rig.authority.resolve_current(scope_type="WORKSPACE", scope_id=WS, environment="TESTNET") == second


def test_duplicate_core_content_fingerprint_is_legal_and_nonretroactive() -> None:
    rig = Rig()
    context = rig.context(rig.record(), membership_id="switch-1")
    accepted = rig.writer.accept(context)
    fingerprint = rig.authority_binding.content_fingerprint_sha256

    rig.core_writer.accept(CoreAcceptedContentBinding("authority-B", fingerprint))

    assert rig.core.has_content_fingerprint(fingerprint) is True
    assert len(rig.core_carrier.read().accepted) == 3
    assert rig.authority.resolve_historical("switch-1") == accepted
    assert rig.authority.resolve_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    ) == accepted
    restored = KillSwitchAuthority(rig.carrier, core_membership=rig.core)
    assert restored.resolve_historical("switch-1") == accepted
    assert restored.resolve_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    ) == accepted


def test_duplicate_core_fingerprint_preserves_superseded_history_on_restore() -> None:
    rig = Rig()
    old = rig.writer.accept(
        rig.context(rig.record(generation=1), membership_id="switch-1")
    )
    new = rig.writer.accept(
        rig.context(
            rig.record(state="ACTIVE", generation=2), membership_id="switch-2"
        )
    )
    fingerprint = rig.authority_binding.content_fingerprint_sha256
    rig.core_writer.accept(CoreAcceptedContentBinding("authority-B", fingerprint))

    restored = KillSwitchAuthority(rig.carrier, core_membership=rig.core)
    assert restored.resolve_historical("switch-1") == old
    assert restored.resolve_historical("switch-2") == new
    assert restored.resolve_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    ) == new


def test_full_history_multi_scope_overlap_advances_only_changed_scope() -> None:
    rig = Rig()
    system_1 = rig.record(
        scope_type="PRODUCT_SYSTEM", scope_id="product", state="ACTIVE", generation=1
    )
    workspace_1 = rig.record(state="INACTIVE", generation=1)
    context_a = rig.context(system_1, workspace_1, membership_id="snapshot-A")
    accepted_a = rig.writer.accept(context_a)
    system_before = rig.authority.resolve_current(
        scope_type="PRODUCT_SYSTEM", scope_id="product", environment="TESTNET"
    )

    workspace_2 = rig.record(state="ACTIVE", generation=2)
    context_b = rig.context(
        system_1, workspace_1, workspace_2, membership_id="snapshot-B"
    )
    accepted_b = rig.writer.accept(context_b)

    assert accepted_a != accepted_b
    assert rig.authority.resolve_historical("snapshot-A") == accepted_a
    assert rig.authority.resolve_historical("snapshot-B") == accepted_b
    assert rig.authority.resolve_current(
        scope_type="PRODUCT_SYSTEM", scope_id="product", environment="TESTNET"
    ) == system_before == accepted_a
    assert rig.authority.resolve_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    ) == accepted_b
    restored = KillSwitchAuthority(rig.carrier, core_membership=rig.core)
    assert restored.resolve_current(
        scope_type="PRODUCT_SYSTEM", scope_id="product", environment="TESTNET"
    ) == accepted_a
    assert restored.resolve_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    ) == accepted_b


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state", "ACTIVE"),
        ("source_revision", 2),
        ("effective_at_utc", "2026-01-02T00:00:00Z"),
        ("accepted_authority_fingerprint_sha256", "c" * 64),
    ],
)
def test_overlap_semantic_rewrite_rejects_without_carrier_mutation(
    field: str, value: object
) -> None:
    rig = Rig()
    original = rig.record(generation=1)
    rig.writer.accept(rig.context(original, membership_id="snapshot-A"))
    if field == "accepted_authority_fingerprint_sha256":
        rig.core_writer.accept(CoreAcceptedContentBinding("alternate-authority", str(value)))
    rewritten_values = asdict(original)
    rewritten_values[field] = value
    rewritten_values.pop("record_fingerprint_sha256")
    rewritten = KillSwitchRecord(
        **rewritten_values,
        record_fingerprint_sha256=canonical_json_sha256(rewritten_values),
    )
    context = rig.context(rewritten, membership_id="snapshot-B")
    before = rig.carrier.read()
    with pytest.raises(KillSwitchAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        rig.writer.accept(context)
    assert rig.carrier.read() == before


def test_restore_rejects_overlap_semantic_rewrite() -> None:
    rig = Rig()
    original = rig.record(generation=1)
    first = rig.writer.accept(rig.context(original, membership_id="snapshot-A"))
    rewritten = rig.record(state="ACTIVE", generation=1)
    second_context = rig.context(rewritten, membership_id="snapshot-B")
    forged = replace(
        rig.carrier.read(),
        store_revision=2,
        accepted=(first, replace(first, transaction_revision=2, context=second_context)),
    )
    rig.carrier._state = forged  # noqa: SLF001 - deliberate durable corruption fixture
    with pytest.raises(KillSwitchAuthorityError, match="CORRUPT_AUTHORITY_STATE"):
        KillSwitchAuthority(rig.carrier, core_membership=rig.core)


def test_unseen_historical_backfill_below_prior_max_is_rejected() -> None:
    rig = Rig()
    rig.writer.accept(rig.context(rig.record(generation=1), membership_id="snapshot-1"))
    rig.writer.accept(rig.context(rig.record(generation=3), membership_id="snapshot-3"))
    before = rig.carrier.read()
    with pytest.raises(KillSwitchAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        rig.writer.accept(rig.context(rig.record(generation=2), membership_id="snapshot-2"))
    assert rig.carrier.read() == before


def test_exact_context_replay_is_idempotent_before_and_after_restore() -> None:
    rig = Rig()
    context = rig.context(rig.record(), membership_id="snapshot-A")
    accepted = rig.writer.accept(context)
    before = rig.carrier.read()
    assert rig.writer.accept(context) == accepted
    assert rig.carrier.read() == before

    restored, restored_writer = KillSwitchAuthority.compose(
        rig.carrier, core_membership=rig.core
    )
    assert restored_writer.accept(context) == accepted
    assert rig.carrier.read() == before
    assert restored.resolve_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET"
    ) == accepted


def test_context_local_duplicate_and_reverse_generations_reject() -> None:
    rig = Rig()
    one = rig.record(generation=1)
    two = rig.record(generation=2)
    for index, history in enumerate(((one, one), (two, one)), 1):
        with pytest.raises(KillSwitchAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
            rig.writer.accept(rig.context(*history, membership_id=f"invalid-{index}"))


@pytest.mark.parametrize("generation", [2, 1])
def test_generation_equality_and_rollback_reject(generation: int) -> None:
    rig = Rig()
    rig.writer.accept(rig.context(rig.record(generation=2), membership_id="switch-1"))
    candidate = rig.record(
        generation=generation, state="ACTIVE" if generation == 2 else "INACTIVE"
    )
    with pytest.raises(KillSwitchAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        rig.writer.accept(rig.context(candidate, membership_id="switch-2"))


@pytest.mark.parametrize(("field", "value"), [("generation", True), ("source_revision", True)])
def test_generation_and_source_revision_reject_bool_aliasing(field: str, value: object) -> None:
    rig = Rig()
    invalid = replace(rig.record(), **{field: value})
    with pytest.raises(KillSwitchAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        rig.writer.accept(rig.context(invalid))


def test_restore_current_and_shared_carrier_visibility() -> None:
    rig = Rig()
    accepted = rig.writer.accept(rig.context(rig.record(state="ACTIVE")))
    restored = KillSwitchAuthority(rig.carrier, core_membership=rig.core)
    assert restored.resolve_current(scope_type="WORKSPACE", scope_id=WS, environment="TESTNET") == accepted
    assert rig.authority.resolve_historical(accepted.context.membership_id) == accepted


@pytest.mark.parametrize("damage", ["generation", "fingerprint", "state", "projection", "membership"])
def test_restore_revalidates_full_history_and_membership_chain(damage: str) -> None:
    rig = Rig()
    accepted = rig.writer.accept(rig.context(rig.record()))
    state = rig.carrier.read()
    if damage == "projection":
        corrupted = replace(state, current=())
    else:
        record = accepted.context.history[0]
        if damage == "generation": record = replace(record, generation=0)
        elif damage == "fingerprint": record = replace(record, record_fingerprint_sha256="f" * 64)
        elif damage == "state": record = replace(record, state="UNKNOWN")
        context = replace(accepted.context, history=(record,))
        if damage == "membership": context = replace(context, membership_id="forged-membership")
        corrupted = replace(state, accepted=(replace(accepted, context=context),))
    rig.carrier._state = corrupted  # noqa: SLF001 - deliberate durable corruption fixture
    with pytest.raises(KillSwitchAuthorityError, match="CORRUPT_AUTHORITY_STATE"):
        KillSwitchAuthority(rig.carrier, core_membership=rig.core)


def test_restore_rejects_missing_core_membership() -> None:
    rig = Rig()
    rig.writer.accept(rig.context(rig.record()))
    empty_core = CoreAcceptedContentAuthority(InMemoryCoreAcceptedContentCarrier(AtomicCoreAcceptedContentState()))
    with pytest.raises(KillSwitchAuthorityError, match="CORRUPT_AUTHORITY_STATE"):
        KillSwitchAuthority(rig.carrier, core_membership=empty_core)


def test_carrier_failure_is_atomic() -> None:
    rig = Rig()
    context = rig.context(rig.record())
    rig.carrier.fail_next = True
    with pytest.raises(OSError, match="INJECTED_CARRIER_FAILURE"):
        rig.writer.accept(context)
    assert rig.carrier.read() == AtomicKillSwitchAuthorityState()


def test_consumer_keeps_carrier_fence_until_callback_returns() -> None:
    rig = Rig()
    first = rig.writer.accept(rig.context(rig.record(), membership_id="switch-1"))
    second = rig.context(rig.record(state="ACTIVE", generation=2), membership_id="switch-2")
    entered, release, published = Event(), Event(), Event()

    def consume(current):
        assert current == first
        entered.set()
        assert release.wait(2)
        return current.context.history[-1].state

    result: list[str] = []
    consuming = Thread(target=lambda: result.append(rig.authority.consume_current(
        scope_type="WORKSPACE", scope_id=WS, environment="TESTNET", consumer=consume)))
    publishing = Thread(target=lambda: (rig.writer.accept(second), published.set()))
    consuming.start(); assert entered.wait(2)
    publishing.start(); assert not published.wait(0.1)
    release.set(); consuming.join(2); publishing.join(2)
    assert result == ["INACTIVE"] and published.is_set()
