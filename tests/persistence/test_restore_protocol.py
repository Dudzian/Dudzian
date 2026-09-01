from __future__ import annotations

import os
import threading
from pathlib import Path

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.restore_protocol import RestoreDecision, S7CRestoreCoordinator
from bot_core.persistence.state_store import SQLiteStateStore
from tests.persistence.test_backup_envelope import _candidate
from tests.persistence.test_state_store_records import (
    OTHER_ACCOUNT_ID,
    OTHER_DEVICE_ID,
    _account,
    _commit,
    _metadata,
    _runtime,
)


def _record(candidate, lifecycle: str, **changes):  # type: ignore[no-untyped-def]
    value = {
        "account_id": candidate.account_id,
        "device_installation_id": candidate.device_installation_id,
        "state_store_identity_fingerprint_sha256": (
            candidate.state_store_identity_fingerprint_sha256
        ),
        "lifecycle": lifecycle,
        "committed_generation": None,
        "committed_state_fingerprint_sha256": None,
        "prepared_generation": None,
        "prepared_state_fingerprint_sha256": None,
        "prepared_transaction_fingerprint_sha256": None,
        "authority_revision": 1,
        "authority_source": "EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY",
    }
    value.update(changes)
    value["content_fingerprint_sha256"] = canonical_json_sha256(value)
    return value


class Boundary:
    def __init__(self, candidate, lifecycle="COMMITTED"):  # type: ignore[no-untyped-def]
        self.ref = object()
        self.calls: list[str] = []
        self.ack_loss = False
        if lifecycle == "COMMITTED":
            self.value = _record(
                candidate,
                lifecycle,
                committed_generation=candidate.local_protected_freshness_generation,
                committed_state_fingerprint_sha256=candidate.state_fingerprint_sha256,
            )
        elif lifecycle == "PREPARED":
            self.value = _record(
                candidate,
                lifecycle,
                prepared_generation=candidate.local_protected_freshness_generation,
                prepared_state_fingerprint_sha256=candidate.state_fingerprint_sha256,
                prepared_transaction_fingerprint_sha256=(candidate.transaction_fingerprint_sha256),
            )
        else:
            self.value = _record(candidate, lifecycle)

    def resolve_current(self, scope):  # type: ignore[no-untyped-def]
        self.calls.append("resolve")
        expected = (
            self.value["account_id"],
            self.value["device_installation_id"],
            self.value["state_store_identity_fingerprint_sha256"],
        )
        return (self.ref, self.value) if scope == expected else None

    def finalize(self, current_ref, scope, *, evidence_ref, evidence_resolver):  # type: ignore[no-untyped-def]
        self.calls.append("finalize")
        evidence = evidence_resolver(evidence_ref)
        expected_scope = (
            self.value["account_id"],
            self.value["device_installation_id"],
            self.value["state_store_identity_fingerprint_sha256"],
        )
        if (
            current_ref is not self.ref
            or scope != expected_scope
            or evidence is None
            or evidence.durability_state != "DURABLE_COMMITTED"
            or evidence.generation != self.value["prepared_generation"]
            or evidence.state_fingerprint_sha256 != self.value["prepared_state_fingerprint_sha256"]
            or evidence.transaction_fingerprint_sha256
            != self.value["prepared_transaction_fingerprint_sha256"]
        ):
            raise RuntimeError("invalid FINALIZE")
        self.value = _record(
            type(
                "Candidate",
                (),
                {
                    "account_id": self.value["account_id"],
                    "device_installation_id": self.value["device_installation_id"],
                    "state_store_identity_fingerprint_sha256": self.value[
                        "state_store_identity_fingerprint_sha256"
                    ],
                },
            )(),
            "COMMITTED",
            authority_revision=2,
            committed_generation=evidence.generation,
            committed_state_fingerprint_sha256=evidence.state_fingerprint_sha256,
        )
        if self.ack_loss:
            raise RuntimeError("ack lost after effect")

    def prepare(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append("prepare")
        raise AssertionError("restore cannot PREPARE")

    def abort(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append("abort")
        raise AssertionError("restore cannot ABORT")


def _coordinator(path: Path, boundary: Boundary):
    return S7CRestoreCoordinator(path, LocalDurableEvidenceRegistry(), boundary)


def _custom_candidate(
    path: Path,
    *,
    environment: str = "PAPER",
    account_id: str | None = None,
    device_id: str | None = None,
    identity: str = "1" * 64,
    repeat_current_at_g2: bool = False,
):  # type: ignore[no-untyped-def]
    account = account_id or _metadata().account_id
    device = device_id or _metadata().device_installation_id
    metadata = _metadata(
        account_id=account,
        device_installation_id=device,
        state_store_identity_fingerprint_sha256=identity,
        environment=environment,
    )
    with SQLiteStateStore(path) as store:
        _commit(
            store, metadata, current=(_account(account),), history=(_runtime(device_id=device),)
        )
        if repeat_current_at_g2:
            _commit(
                store,
                _metadata(
                    2,
                    account_id=account,
                    device_installation_id=device,
                    state_store_identity_fingerprint_sha256=identity,
                    environment=environment,
                ),
                current=(_account(account),),
                history=(_runtime("run_01890f4c-7b9a-7cc2-8a2b-123456789abc", device_id=device),),
                expected=1,
            )
        from bot_core.persistence.backup_envelope import create_backup_envelope

        candidate = create_backup_envelope(store)
    assert candidate is not None
    return candidate


def test_committed_empty_restores_exact_isolated_store(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "source.db")
    boundary = Boundary(candidate)
    live = tmp_path / "live.db"
    result = _coordinator(live, boundary).restore(candidate.to_mapping())
    assert result.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    with SQLiteStateStore(live) as store:
        assert store.read_verified_snapshot().metadata.state_fingerprint_sha256 == (  # type: ignore[union-attr]
            candidate.state_fingerprint_sha256
        )
    assert "finalize" not in boundary.calls


def test_committed_exact_is_idempotent_no_restore_write(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "live.db")
    boundary = Boundary(candidate)
    before = (tmp_path / "live.db").stat().st_ino
    result = _coordinator(tmp_path / "live.db", boundary).restore(candidate)
    assert result.decision is RestoreDecision.NOOP_ALREADY_CURRENT
    assert (tmp_path / "live.db").stat().st_ino == before


def test_missing_membership_and_uninitialized_deny_without_live_write(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "source.db")
    boundary = Boundary(candidate, "UNINITIALIZED")
    live = tmp_path / "live.db"
    assert _coordinator(live, boundary).restore(candidate).decision is RestoreDecision.DENY
    assert not live.exists()


def test_missing_current_membership_denies_without_live_write(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "source.db")
    boundary = Boundary(candidate)
    boundary.resolve_current = lambda scope: None  # type: ignore[method-assign]
    live = tmp_path / "live.db"
    assert _coordinator(live, boundary).restore(candidate).decision is RestoreDecision.DENY
    assert not live.exists()


def test_invalid_backup_denies_before_live_write(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "source.db")
    boundary = Boundary(candidate)
    mapping = candidate.to_mapping()
    mapping["state_fingerprint_sha256"] = "0" * 64
    live = tmp_path / "live.db"
    assert _coordinator(live, boundary).restore(mapping).decision is RestoreDecision.DENY
    assert not live.exists()


def test_wrong_current_payload_denies(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "source.db")
    boundary = Boundary(candidate)
    boundary.value = _record(
        candidate,
        "COMMITTED",
        committed_generation=1,
        committed_state_fingerprint_sha256="0" * 64,
    )
    assert _coordinator(tmp_path / "live.db", boundary).restore(candidate).decision is (
        RestoreDecision.DENY
    )


def test_genesis_prepared_restores_and_uses_only_existing_finalize(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "source.db")
    boundary = Boundary(candidate, "PREPARED")
    result = _coordinator(tmp_path / "live.db", boundary).restore(candidate)
    assert result.decision is RestoreDecision.RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE
    assert boundary.calls.count("finalize") == 1
    assert "prepare" not in boundary.calls and "abort" not in boundary.calls


def test_local_ahead_is_not_rolled_back(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "candidate.db", generations=1)
    _candidate(tmp_path / "live.db", generations=2)
    boundary = Boundary(candidate)
    result = _coordinator(tmp_path / "live.db", boundary).restore(candidate)
    assert result.decision is RestoreDecision.DENY
    with SQLiteStateStore(tmp_path / "live.db") as store:
        assert store.read_verified_snapshot().metadata.protected_freshness_generation == 2  # type: ignore[union-attr]


def test_committed_exact_invalidates_old_and_rebuilds_fresh_evidence(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "live.db")
    boundary = Boundary(candidate)
    registry = LocalDurableEvidenceRegistry()
    scope = (
        candidate.account_id,
        candidate.device_installation_id,
        candidate.state_store_identity_fingerprint_sha256,
    )
    with SQLiteStateStore(tmp_path / "live.db") as store:
        old_ref = registry.publish_verified_state(store)
    published: list[str] = []
    original_publish = registry.publish_verified_state

    def capture(store):  # type: ignore[no-untyped-def]
        ref = original_publish(store)
        assert ref is not None
        published.append(ref)
        return ref

    registry.publish_verified_state = capture  # type: ignore[method-assign]
    result = S7CRestoreCoordinator(tmp_path / "live.db", registry, boundary).restore(candidate)
    assert result.decision is RestoreDecision.NOOP_ALREADY_CURRENT
    assert registry.resolve_current(scope, old_ref) is None
    assert not registry.verify_current(scope, old_ref)
    assert len(published) == 1 and published[0] != old_ref
    evidence = registry.resolve_current(scope, published[0])
    assert evidence is not None
    assert (
        evidence.account_id,
        evidence.device_installation_id,
        evidence.state_store_identity_fingerprint_sha256,
    ) == scope
    assert (
        evidence.generation,
        evidence.state_fingerprint_sha256,
        evidence.transaction_fingerprint_sha256,
        evidence.durability_state,
    ) == (
        candidate.local_protected_freshness_generation,
        candidate.state_fingerprint_sha256,
        candidate.transaction_fingerprint_sha256,
        "DURABLE_COMMITTED",
    )


def test_install_before_evidence_failure_retry_rebuilds_without_second_replace(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    candidate = _candidate(tmp_path / "source.db")
    boundary = Boundary(candidate)
    registry = LocalDurableEvidenceRegistry()
    live = tmp_path / "live.db"
    original_publish = registry.publish_verified_state
    monkeypatch.setattr(registry, "publish_verified_state", lambda store: None)
    assert S7CRestoreCoordinator(live, registry, boundary).restore(candidate).decision is (
        RestoreDecision.DENY
    )
    with SQLiteStateStore(live) as store:
        assert store.read_verified_snapshot() is not None
    monkeypatch.setattr(registry, "publish_verified_state", original_publish)
    replacements = 0
    original_replace = SQLiteStateStore.atomic_replace

    def counted(source, target):  # type: ignore[no-untyped-def]
        nonlocal replacements
        replacements += 1
        return original_replace(source, target)

    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", counted)
    assert S7CRestoreCoordinator(live, registry, boundary).restore(candidate).decision is (
        RestoreDecision.NOOP_ALREADY_CURRENT
    )
    assert replacements == 0


def test_failed_atomic_replace_preserves_verified_old_live(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    live = tmp_path / "live.db"
    old = _candidate(live, generations=1)
    candidate = _candidate(tmp_path / "candidate.db", generations=2)
    boundary = Boundary(candidate)
    coordinator = _coordinator(live, boundary)
    original_classify = coordinator._classify
    classifications = []

    def classified(value):  # type: ignore[no-untyped-def]
        result = original_classify(value)
        classifications.append(result[0].value)
        return result

    with SQLiteStateStore(live) as store:
        before = store.read_verified_snapshot()

    def fail(source, target):  # type: ignore[no-untyped-def]
        raise OSError("injected replacement failure")

    monkeypatch.setattr(coordinator, "_classify", classified)
    monkeypatch.setattr("bot_core.persistence.state_store.os.replace", fail)
    assert coordinator.restore(candidate).decision is RestoreDecision.DENY
    assert classifications == ["BEHIND", "BEHIND"]
    with SQLiteStateStore(live) as store:
        assert store.read_verified_snapshot() == before
    assert old.state_fingerprint_sha256 == before.metadata.state_fingerprint_sha256  # type: ignore[union-attr]


def test_open_live_handle_fences_install_without_split_brain(tmp_path: Path) -> None:
    live = tmp_path / "live.db"
    _candidate(live, generations=1)
    candidate = _candidate(tmp_path / "candidate.db", generations=2)
    boundary = Boundary(candidate)
    stale = SQLiteStateStore(live)
    before = stale.read_verified_snapshot()
    try:
        assert _coordinator(live, boundary).restore(candidate).decision is RestoreDecision.DENY
        assert stale.read_verified_snapshot() == before
    finally:
        stale.close()


def test_isolated_write_failure_closes_and_cleans_artifacts(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    candidate = _candidate(tmp_path / "source.db")
    boundary = Boundary(candidate)
    live = tmp_path / "live.db"

    def fail(self, snapshot):  # type: ignore[no-untyped-def]
        raise StateStoreError("injected isolated write failure")

    from bot_core.persistence.state_store import StateStoreError

    monkeypatch.setattr(SQLiteStateStore, "write_restored_snapshot", fail)
    assert _coordinator(live, boundary).restore(candidate).decision is RestoreDecision.DENY
    assert not live.exists()
    assert not list(tmp_path.glob(".live.db.s7c-*"))


def test_current_ref_change_before_install_denies_and_preserves_live(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    live = tmp_path / "live.db"
    _candidate(live, generations=1)
    candidate = _candidate(tmp_path / "candidate.db", generations=2)
    boundary = Boundary(candidate)
    original = boundary.resolve_current
    calls = 0

    def changing(scope):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls >= 2:
            boundary.ref = object()
        return original(scope)

    monkeypatch.setattr(boundary, "resolve_current", changing)
    assert _coordinator(live, boundary).restore(candidate).decision is RestoreDecision.DENY
    with SQLiteStateStore(live) as store:
        assert store.read_verified_snapshot().metadata.protected_freshness_generation == 1  # type: ignore[union-attr]


def test_normal_prepared_g_plus_one_restores_and_finalizes(tmp_path: Path) -> None:
    baseline = _candidate(tmp_path / "baseline-source.db", generations=1)
    candidate = _candidate(tmp_path / "candidate.db", generations=2)
    live = tmp_path / "live.db"
    _candidate(live, generations=1)
    boundary = Boundary(candidate)
    boundary.value = _record(
        candidate,
        "PREPARED",
        committed_generation=1,
        committed_state_fingerprint_sha256=baseline.state_fingerprint_sha256,
        prepared_generation=2,
        prepared_state_fingerprint_sha256=candidate.state_fingerprint_sha256,
        prepared_transaction_fingerprint_sha256=candidate.transaction_fingerprint_sha256,
    )
    result = _coordinator(live, boundary).restore(candidate)
    assert result.decision is RestoreDecision.RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE
    assert boundary.calls.count("finalize") == 1
    assert boundary.value["lifecycle"] == "COMMITTED"


def test_prepared_exact_uses_zero_replacements(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    candidate = _candidate(tmp_path / "live.db")
    boundary = Boundary(candidate, "PREPARED")
    replacements = 0

    def forbidden(source, target):  # type: ignore[no-untyped-def]
        nonlocal replacements
        replacements += 1
        raise AssertionError("exact restore must not replace")

    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", forbidden)
    result = _coordinator(tmp_path / "live.db", boundary).restore(candidate)
    assert result.decision is RestoreDecision.RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE
    assert replacements == 0
    assert boundary.calls.count("finalize") == 1
    assert boundary.calls.count("prepare") == 0
    assert boundary.calls.count("abort") == 0
    assert boundary.value["lifecycle"] == "COMMITTED"


def test_finalize_ack_loss_is_reconciled_without_blind_retry(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "live.db")
    boundary = Boundary(candidate, "PREPARED")
    boundary.ack_loss = True
    result = _coordinator(tmp_path / "live.db", boundary).restore(candidate)
    assert result.decision is RestoreDecision.RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE
    assert boundary.calls.count("finalize") == 1
    assert boundary.value["lifecycle"] == "COMMITTED"


def test_fresh_behind_to_ahead_race_denies_without_replace(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    live = tmp_path / "live.db"
    _candidate(live, generations=1)
    candidate = _candidate(tmp_path / "candidate.db", generations=2)
    ahead_path = tmp_path / "ahead.db"
    _candidate(ahead_path, generations=3)
    boundary = Boundary(candidate)
    coordinator = _coordinator(live, boundary)
    original_classify = coordinator._classify
    classifications = 0
    replacements = 0

    def racing_classify(value):  # type: ignore[no-untyped-def]
        nonlocal classifications
        classifications += 1
        if classifications == 2:
            os.replace(ahead_path, live)
        return original_classify(value)

    def counted_replace(source, target):  # type: ignore[no-untyped-def]
        nonlocal replacements
        replacements += 1

    monkeypatch.setattr(coordinator, "_classify", racing_classify)
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", counted_replace)
    assert coordinator.restore(candidate).decision is RestoreDecision.DENY
    assert replacements == 0
    with SQLiteStateStore(live) as store:
        assert store.read_verified_snapshot().metadata.protected_freshness_generation == 3  # type: ignore[union-attr]


def test_no_trusted_creation_race_to_exact_skips_replace(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "source.db"
    candidate = _candidate(source)
    live = tmp_path / "live.db"
    boundary = Boundary(candidate)
    coordinator = _coordinator(live, boundary)
    original_classify = coordinator._classify
    classifications = 0
    replacements = 0

    def racing_classify(value):  # type: ignore[no-untyped-def]
        nonlocal classifications
        classifications += 1
        if classifications == 2:
            os.replace(source, live)
        return original_classify(value)

    def counted_replace(source_path, target):  # type: ignore[no-untyped-def]
        nonlocal replacements
        replacements += 1

    monkeypatch.setattr(coordinator, "_classify", racing_classify)
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", counted_replace)
    assert coordinator.restore(candidate).decision is RestoreDecision.NOOP_ALREADY_CURRENT
    assert replacements == 0


def test_connect_registration_is_atomic_with_replacement_gate(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import bot_core.persistence.state_store as state_store_module

    live = tmp_path / "live.db"
    _candidate(live)
    isolated = tmp_path / "isolated.db"
    _candidate(isolated, generations=2)
    entered_connect = threading.Event()
    release_connect = threading.Event()
    opener_generations: list[int] = []
    replace_errors: list[BaseException] = []
    replacement_done = threading.Event()
    original_connect = state_store_module.sqlite3.connect

    def gated_connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        entered_connect.set()
        assert release_connect.wait(timeout=5)
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(state_store_module.sqlite3, "connect", gated_connect)

    def open_and_hold() -> None:
        with SQLiteStateStore(live) as store:
            assert replacement_done.wait(timeout=5)
            snapshot = store.read_verified_snapshot()
            assert snapshot is not None
            opener_generations.append(snapshot.metadata.protected_freshness_generation)

    opener = threading.Thread(target=open_and_hold)
    opener.start()
    assert entered_connect.wait(timeout=5)

    def replace() -> None:
        try:
            SQLiteStateStore.atomic_replace(isolated, live)
        except BaseException as exc:
            replace_errors.append(exc)
        finally:
            replacement_done.set()

    replacer = threading.Thread(target=replace)
    replacer.start()
    release_connect.set()
    opener.join(timeout=5)
    replacer.join(timeout=5)
    assert not opener.is_alive() and not replacer.is_alive()
    assert opener_generations == [1]
    assert len(replace_errors) == 1


def test_preexisting_transient_handle_is_denied_at_quiescence_before_replace(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    live = tmp_path / "live.db"
    _candidate(live, generations=1)
    candidate = _candidate(tmp_path / "candidate.db", generations=2)
    boundary = Boundary(candidate)
    transient = SQLiteStateStore(live)
    replacements = 0

    def counted(source, target):  # type: ignore[no-untyped-def]
        nonlocal replacements
        replacements += 1

    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", counted)
    try:
        assert _coordinator(live, boundary).restore(candidate).decision is RestoreDecision.DENY
        assert replacements == 0
        assert "finalize" not in boundary.calls
        assert transient.read_verified_snapshot().metadata.protected_freshness_generation == 1  # type: ignore[union-attr]
    finally:
        transient.close()


def test_final_in_gate_external_ref_change_denies_before_publication(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    live = tmp_path / "live.db"
    _candidate(live, generations=1)
    candidate = _candidate(tmp_path / "candidate.db", generations=2)
    boundary = Boundary(candidate)
    original_resolve = boundary.resolve_current
    resolves = 0
    replacements = 0

    def changing(scope):  # type: ignore[no-untyped-def]
        nonlocal resolves
        resolves += 1
        if resolves == 3:
            boundary.ref = object()
        return original_resolve(scope)

    def counted(source, target):  # type: ignore[no-untyped-def]
        nonlocal replacements
        replacements += 1

    monkeypatch.setattr(boundary, "resolve_current", changing)
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", counted)
    assert _coordinator(live, boundary).restore(candidate).decision is RestoreDecision.DENY
    assert replacements == 0 and "finalize" not in boundary.calls
    with SQLiteStateStore(live) as store:
        assert store.read_verified_snapshot().metadata.protected_freshness_generation == 1  # type: ignore[union-attr]


def test_committed_same_generation_wrong_state_denies_without_replace(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    candidate = _candidate(tmp_path / "candidate.db")
    live = tmp_path / "live.db"
    with SQLiteStateStore(live) as store:
        _commit(
            store,
            _metadata(),
            current=(_account(),),
            history=(_runtime("run_01890f4c-7b9a-7cc3-8a2b-123456789abc"),),
        )
        before = store.read_verified_snapshot()
    boundary = Boundary(candidate)
    monkeypatch.setattr(
        SQLiteStateStore,
        "atomic_replace",
        lambda source, target: (_ for _ in ()).throw(AssertionError("unexpected replace")),
    )
    assert _coordinator(live, boundary).restore(candidate).decision is RestoreDecision.DENY
    assert "finalize" not in boundary.calls
    with SQLiteStateStore(live) as store:
        assert store.read_verified_snapshot() == before


def test_committed_same_state_wrong_transaction_denies_as_lineage_mismatch(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    candidate = _candidate(tmp_path / "candidate.db", generations=2)
    local = _custom_candidate(tmp_path / "live.db", repeat_current_at_g2=True)
    assert local.state_fingerprint_sha256 == candidate.state_fingerprint_sha256
    assert local.history_tail_fingerprint_sha256 == candidate.history_tail_fingerprint_sha256
    assert local.transaction_fingerprint_sha256 != candidate.transaction_fingerprint_sha256
    monkeypatch.setattr(
        SQLiteStateStore,
        "atomic_replace",
        lambda source, target: (_ for _ in ()).throw(AssertionError("unexpected replace")),
    )
    assert _coordinator(tmp_path / "live.db", Boundary(candidate)).restore(candidate).decision is (
        RestoreDecision.DENY
    )


def test_committed_wrong_history_lineage_denies_without_replace(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    candidate = _candidate(tmp_path / "candidate.db")
    live = tmp_path / "live.db"
    with SQLiteStateStore(live) as store:
        _commit(
            store,
            _metadata(),
            current=(_account(),),
            history=(_runtime("run_01890f4c-7b9a-7cc3-8a2b-123456789abc"),),
        )
    monkeypatch.setattr(
        SQLiteStateStore,
        "atomic_replace",
        lambda source, target: (_ for _ in ()).throw(AssertionError("unexpected replace")),
    )
    # Canonical state binds history_tail, so a self-consistent different
    # history is the equivalent history-lineage conflict and is denied early.
    assert _coordinator(live, Boundary(candidate)).restore(candidate).decision is (
        RestoreDecision.DENY
    )


def test_committed_scope_conflict_denies_without_replace(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    candidate = _candidate(tmp_path / "candidate.db")
    local = _custom_candidate(
        tmp_path / "live.db",
        account_id=OTHER_ACCOUNT_ID,
        device_id=OTHER_DEVICE_ID,
        identity="2" * 64,
    )
    monkeypatch.setattr(
        SQLiteStateStore,
        "atomic_replace",
        lambda source, target: (_ for _ in ()).throw(AssertionError("unexpected replace")),
    )
    assert _coordinator(tmp_path / "live.db", Boundary(candidate)).restore(candidate).decision is (
        RestoreDecision.DENY
    )
    with SQLiteStateStore(tmp_path / "live.db") as store:
        assert store.read_verified_snapshot().metadata.account_id == local.account_id  # type: ignore[union-attr]


def test_committed_environment_conflict_has_no_fallback(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    candidate = _candidate(tmp_path / "candidate.db")
    _custom_candidate(tmp_path / "live.db", environment="TESTNET")
    monkeypatch.setattr(
        SQLiteStateStore,
        "atomic_replace",
        lambda source, target: (_ for _ in ()).throw(AssertionError("unexpected replace")),
    )
    assert _coordinator(tmp_path / "live.db", Boundary(candidate)).restore(candidate).decision is (
        RestoreDecision.DENY
    )
    with SQLiteStateStore(tmp_path / "live.db") as store:
        assert store.read_verified_snapshot().metadata.environment == "TESTNET"  # type: ignore[union-attr]


def _normal_pending(baseline, pending):  # type: ignore[no-untyped-def]
    boundary = Boundary(pending)
    boundary.value = _record(
        pending,
        "PREPARED",
        committed_generation=baseline.local_protected_freshness_generation,
        committed_state_fingerprint_sha256=baseline.state_fingerprint_sha256,
        prepared_generation=pending.local_protected_freshness_generation,
        prepared_state_fingerprint_sha256=pending.state_fingerprint_sha256,
        prepared_transaction_fingerprint_sha256=pending.transaction_fingerprint_sha256,
    )
    return boundary


def test_prepared_pending_g_plus_one_denies_baseline_g_without_abort(tmp_path: Path) -> None:
    baseline = _candidate(tmp_path / "baseline.db", generations=1)
    pending = _candidate(tmp_path / "pending.db", generations=2)
    boundary = _normal_pending(baseline, pending)
    live = tmp_path / "live.db"
    assert _coordinator(live, boundary).restore(baseline).decision is RestoreDecision.DENY
    assert not live.exists()
    assert "abort" not in boundary.calls and "finalize" not in boundary.calls
    assert boundary.value["lifecycle"] == "PREPARED"


def test_prepared_wrong_transaction_valid_candidate_denied(tmp_path: Path) -> None:
    baseline = _candidate(tmp_path / "baseline.db", generations=1)
    pending = _candidate(tmp_path / "pending.db", generations=2)
    wrong_tx = _custom_candidate(tmp_path / "wrong-tx.db", repeat_current_at_g2=True)
    assert wrong_tx.state_fingerprint_sha256 == pending.state_fingerprint_sha256
    assert wrong_tx.transaction_fingerprint_sha256 != pending.transaction_fingerprint_sha256
    boundary = _normal_pending(baseline, pending)
    assert _coordinator(tmp_path / "live.db", boundary).restore(wrong_tx).decision is (
        RestoreDecision.DENY
    )
    assert "abort" not in boundary.calls and "finalize" not in boundary.calls


def test_prepared_pending_g_plus_one_denies_valid_g_plus_two(tmp_path: Path) -> None:
    baseline = _candidate(tmp_path / "baseline.db", generations=1)
    pending = _candidate(tmp_path / "pending.db", generations=2)
    future = _candidate(tmp_path / "future.db", generations=3)
    boundary = _normal_pending(baseline, pending)
    assert _coordinator(tmp_path / "live.db", boundary).restore(future).decision is (
        RestoreDecision.DENY
    )
    assert not {"prepare", "abort", "finalize"}.intersection(boundary.calls)


def test_current_ref_change_after_install_denies_without_second_replace(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    candidate = _candidate(tmp_path / "candidate.db")
    boundary = Boundary(candidate)
    original_resolve = boundary.resolve_current
    resolves = 0
    replacements = 0
    original_replace = SQLiteStateStore.atomic_replace

    def changing(scope):  # type: ignore[no-untyped-def]
        nonlocal resolves
        resolves += 1
        if resolves == 4:
            boundary.ref = object()
        return original_resolve(scope)

    def counted(source, target):  # type: ignore[no-untyped-def]
        nonlocal replacements
        replacements += 1
        return original_replace(source, target)

    monkeypatch.setattr(boundary, "resolve_current", changing)
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", counted)
    live = tmp_path / "live.db"
    assert _coordinator(live, boundary).restore(candidate).decision is RestoreDecision.DENY
    assert replacements == 1 and "finalize" not in boundary.calls
    with SQLiteStateStore(live) as store:
        assert store.read_verified_snapshot().metadata.state_fingerprint_sha256 == (  # type: ignore[union-attr]
            candidate.state_fingerprint_sha256
        )


def test_prepared_ref_change_before_recovery_denies_without_finalize(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    candidate = _candidate(tmp_path / "candidate.db")
    boundary = Boundary(candidate, "PREPARED")
    original_resolve = boundary.resolve_current
    resolves = 0

    def changing(scope):  # type: ignore[no-untyped-def]
        nonlocal resolves
        resolves += 1
        if resolves == 4:
            boundary.ref = object()
        return original_resolve(scope)

    monkeypatch.setattr(boundary, "resolve_current", changing)
    assert _coordinator(tmp_path / "live.db", boundary).restore(candidate).decision is (
        RestoreDecision.DENY
    )
    assert "finalize" not in boundary.calls


def test_corrupt_unreadable_live_disaster_restore_isolated_and_exact(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "candidate.db")
    live = tmp_path / "live.db"
    live.write_bytes(b"not-a-sqlite-state-store")
    Path(f"{live}-wal").write_bytes(b"orphan-old-wal")
    Path(f"{live}-shm").write_bytes(b"orphan-old-shm")
    registry = LocalDurableEvidenceRegistry()
    result = S7CRestoreCoordinator(live, registry, Boundary(candidate)).restore(candidate)
    assert result.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    assert not Path(f"{live}-wal").exists() and not Path(f"{live}-shm").exists()
    with SQLiteStateStore(live) as store:
        snapshot = store.read_verified_snapshot()
    assert snapshot is not None and snapshot.metadata.state_fingerprint_sha256 == (
        candidate.state_fingerprint_sha256
    )


def test_orphan_sidecars_without_main_cannot_contaminate_restore(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "candidate.db")
    live = tmp_path / "live.db"
    Path(f"{live}-wal").write_bytes(b"orphan-old-wal")
    Path(f"{live}-shm").write_bytes(b"orphan-old-shm")
    result = _coordinator(live, Boundary(candidate)).restore(candidate)
    assert result.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    with SQLiteStateStore(live) as store:
        snapshot = store.read_verified_snapshot()
    assert snapshot is not None
    assert snapshot.metadata.state_fingerprint_sha256 == candidate.state_fingerprint_sha256


def test_successful_replace_failure_before_reopen_retries_without_second_replace(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from bot_core.persistence.state_store import StateStoreError

    candidate = _candidate(tmp_path / "candidate.db")
    live = tmp_path / "live.db"
    boundary = Boundary(candidate)
    original_replace = SQLiteStateStore.atomic_replace
    original_init = SQLiteStateStore.__init__
    replaced = False
    fail_reopen_once = True
    replacements = 0

    def counted_replace(source, target):  # type: ignore[no-untyped-def]
        nonlocal replaced, replacements
        original_replace(source, target)
        replacements += 1
        replaced = True

    def failing_init(self, path, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal fail_reopen_once
        if replaced and fail_reopen_once and Path(path).resolve() == live.resolve():
            fail_reopen_once = False
            raise StateStoreError("injected crash before post-install reopen")
        original_init(self, path, *args, **kwargs)

    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", counted_replace)
    monkeypatch.setattr(SQLiteStateStore, "__init__", failing_init)
    assert _coordinator(live, boundary).restore(candidate).decision is RestoreDecision.DENY
    assert replacements == 1
    assert not Path(f"{live}-wal").exists() and not Path(f"{live}-shm").exists()
    assert _coordinator(live, boundary).restore(candidate).decision is (
        RestoreDecision.NOOP_ALREADY_CURRENT
    )
    assert replacements == 1
