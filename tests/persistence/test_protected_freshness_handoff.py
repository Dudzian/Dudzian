from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.persistence.local_durable_evidence import (
    LocalDurableEvidenceRegistry,
    LocalDurableStateEvidence,
)
from bot_core.persistence.protected_freshness_handoff import (
    ProtectedFreshnessAuthorityRecord,
    ProtectedFreshnessHandoffCoordinator,
    ProtectedFreshnessHandoffError,
)
from bot_core.persistence.state_store import SQLiteStateStore
from tests.persistence.test_state_store_records import _account, _commit, _metadata, _runtime

SCOPE = (_metadata().account_id, _metadata().device_installation_id, "1" * 64)


def record(lifecycle: str, *, revision: int = 1, **changes: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "account_id": SCOPE[0],
        "device_installation_id": SCOPE[1],
        "state_store_identity_fingerprint_sha256": SCOPE[2],
        "lifecycle": lifecycle,
        "committed_generation": None,
        "committed_state_fingerprint_sha256": None,
        "prepared_generation": None,
        "prepared_state_fingerprint_sha256": None,
        "prepared_transaction_fingerprint_sha256": None,
        "authority_revision": revision,
        "authority_source": "EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY",
    }
    values.update(changes)
    values["content_fingerprint_sha256"] = canonical_json_sha256(values)
    return values


class Boundary:
    def __init__(self, initial: dict[str, Any]) -> None:
        self.ref: object = object()
        self.value = initial
        self.calls: list[str] = []
        self.fail_finalize = False
        self.after_publish = None
        self.wrong_pending: dict[str, Any] = {}
        self.before_prepare = None
        self.ack_lost = False
        self.change_ref_after_prepare = False
        self.change_ref_after_finalize = False
        self.post_finalize_changes: dict[str, Any] = {}
        self._finalized = False

    def resolve_current(self, scope):  # type: ignore[no-untyped-def]
        self.calls.append("resolve")
        current_scope = (
            self.value["account_id"],
            self.value["device_installation_id"],
            self.value["state_store_identity_fingerprint_sha256"],
        )
        if scope != current_scope:
            return None
        if self._finalized and self.post_finalize_changes:
            changed = {**self.value, **self.post_finalize_changes}
            if "content_fingerprint_sha256" not in self.post_finalize_changes:
                changed["content_fingerprint_sha256"] = canonical_json_sha256(
                    {k: v for k, v in changed.items() if k != "content_fingerprint_sha256"}
                )
            return self.ref, changed
        return self.ref, self.value

    def prepare(self, current_ref, scope, **candidate):  # type: ignore[no-untyped-def]
        self.calls.append("prepare")
        if self.before_prepare:
            self.before_prepare()
        current_scope = (
            self.value["account_id"],
            self.value["device_installation_id"],
            self.value["state_store_identity_fingerprint_sha256"],
        )
        if current_ref is not self.ref or scope != current_scope:
            raise RuntimeError("stale")
        lifecycle = self.value["lifecycle"]
        expected = (
            candidate["expected_committed_generation"],
            candidate["expected_committed_state_fingerprint_sha256"],
        )
        committed = (
            self.value["committed_generation"],
            self.value["committed_state_fingerprint_sha256"],
        )
        generation = candidate["candidate_generation"]
        if lifecycle not in {"UNINITIALIZED", "COMMITTED"} or expected != committed:
            raise RuntimeError("prepare denied")
        if (lifecycle == "UNINITIALIZED" and generation != 1) or (
            lifecycle == "COMMITTED" and generation != committed[0] + 1
        ):
            raise RuntimeError("prepare denied")
        if any(
            not isinstance(candidate[name], str)
            or len(candidate[name]) != 64
            or candidate[name].lower() != candidate[name]
            for name in (
                "candidate_state_fingerprint_sha256",
                "candidate_transaction_fingerprint_sha256",
            )
        ):
            raise RuntimeError("prepare denied")
        pending = {
            "committed_generation": candidate["expected_committed_generation"],
            "committed_state_fingerprint_sha256": candidate[
                "expected_committed_state_fingerprint_sha256"
            ],
            "prepared_generation": candidate["candidate_generation"],
            "prepared_state_fingerprint_sha256": candidate["candidate_state_fingerprint_sha256"],
            "prepared_transaction_fingerprint_sha256": candidate[
                "candidate_transaction_fingerprint_sha256"
            ],
        }
        pending.update(self.wrong_pending)
        self.value = record("PREPARED", revision=2, **pending)
        if self.change_ref_after_prepare:
            self.ref = object()

    def finalize(self, current_ref, scope, *, evidence_ref, evidence_resolver):  # type: ignore[no-untyped-def]
        self.calls.append("finalize")
        if self.after_publish:
            self.after_publish()
        evidence = evidence_resolver(evidence_ref)
        pending_scope = (
            self.value["account_id"],
            self.value["device_installation_id"],
            self.value["state_store_identity_fingerprint_sha256"],
        )
        if (
            current_ref is not self.ref
            or scope != pending_scope
            or self.value["lifecycle"] != "PREPARED"
        ):
            raise RuntimeError("finalize denied")
        if evidence is None or self.fail_finalize:
            raise RuntimeError("finalize denied")
        if (
            evidence.durability_state != "DURABLE_COMMITTED"
            or (
                evidence.account_id,
                evidence.device_installation_id,
                evidence.state_store_identity_fingerprint_sha256,
            )
            != pending_scope
            or evidence.generation != self.value["prepared_generation"]
            or evidence.state_fingerprint_sha256 != self.value["prepared_state_fingerprint_sha256"]
            or evidence.transaction_fingerprint_sha256
            != self.value["prepared_transaction_fingerprint_sha256"]
        ):
            raise RuntimeError("finalize evidence mismatch")
        self.value = record(
            "COMMITTED",
            revision=3,
            committed_generation=evidence.generation,
            committed_state_fingerprint_sha256=evidence.state_fingerprint_sha256,
        )
        self._finalized = True
        if self.change_ref_after_finalize:
            self.ref = object()
        if self.ack_lost:
            raise RuntimeError("ack lost")


def test_genesis_and_normal_advance_use_ordered_handoff(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "s.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        registry = LocalDurableEvidenceRegistry()
        coordinator = ProtectedFreshnessHandoffCoordinator(store, registry, boundary)
        one = coordinator.advance_protected_state(_metadata(), current_records=(_account(),))
        assert one.protected_freshness_generation == 1
        assert boundary.value["lifecycle"] == "COMMITTED"
        two = coordinator.advance_protected_state(_metadata(2), immutable_history=(_runtime(),))
        assert two.protected_freshness_generation == 2
        assert boundary.value["committed_state_fingerprint_sha256"] == two.state_fingerprint_sha256
        assert boundary.calls.count("prepare") == boundary.calls.count("finalize") == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authority_revision", True),
        ("authority_revision", 0),
        ("lifecycle", "BAD"),
        ("committed_generation", 0),
        ("state_store_identity_fingerprint_sha256", "A" * 64),
        ("authority_source", "LOCAL"),
    ],
)
def test_external_record_validation_fails_closed(field: str, value: object) -> None:
    candidate = record(
        "COMMITTED", committed_generation=1, committed_state_fingerprint_sha256="a" * 64
    )
    candidate[field] = value
    candidate["content_fingerprint_sha256"] = canonical_json_sha256(
        {k: v for k, v in candidate.items() if k != "content_fingerprint_sha256"}
    )
    with pytest.raises(ProtectedFreshnessHandoffError):
        ProtectedFreshnessAuthorityRecord.from_mapping(candidate)


def test_external_record_rejects_extra_and_bad_content_fingerprint() -> None:
    candidate = record("UNINITIALIZED")
    with pytest.raises(ProtectedFreshnessHandoffError):
        ProtectedFreshnessAuthorityRecord.from_mapping({**candidate, "trusted": True})
    with pytest.raises(ProtectedFreshnessHandoffError):
        ProtectedFreshnessAuthorityRecord.from_mapping(
            {**candidate, "content_fingerprint_sha256": "f" * 64}
        )


@pytest.mark.parametrize("scope_index", range(3))
def test_external_scope_mismatch_precedes_prepare(tmp_path: Path, scope_index: int) -> None:
    initial = record("UNINITIALIZED")
    key = ("account_id", "device_installation_id", "state_store_identity_fingerprint_sha256")[
        scope_index
    ]
    initial[key] = (
        "acct_01890f4c-7b9a-7cc1-8a2b-123456789abd"
        if scope_index == 0
        else "dev_01890f4c-7b9a-7cc1-8a2b-123456789abd"
        if scope_index == 1
        else "e" * 64
    )
    initial["content_fingerprint_sha256"] = canonical_json_sha256(
        {k: v for k, v in initial.items() if k != "content_fingerprint_sha256"}
    )
    with SQLiteStateStore(tmp_path / f"{scope_index}.db") as store:
        boundary = Boundary(initial)
        with pytest.raises(ProtectedFreshnessHandoffError):
            ProtectedFreshnessHandoffCoordinator(
                store, LocalDurableEvidenceRegistry(), boundary
            ).advance_protected_state(_metadata())
        assert "prepare" not in boundary.calls
        assert store.read_verified_snapshot() is None


def test_prepared_at_entry_is_not_overwritten_or_aborted(tmp_path: Path) -> None:
    pending = record(
        "PREPARED",
        prepared_generation=1,
        prepared_state_fingerprint_sha256="a" * 64,
        prepared_transaction_fingerprint_sha256="b" * 64,
    )
    with SQLiteStateStore(tmp_path / "s.db") as store:
        boundary = Boundary(pending)
        with pytest.raises(ProtectedFreshnessHandoffError):
            ProtectedFreshnessHandoffCoordinator(
                store, LocalDurableEvidenceRegistry(), boundary
            ).advance_protected_state(_metadata())
        assert boundary.value == pending and "prepare" not in boundary.calls


def test_wrong_prepare_ack_prevents_local_commit(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "s.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        boundary.wrong_pending = {"prepared_state_fingerprint_sha256": "f" * 64}
        coordinator = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        with pytest.raises(ProtectedFreshnessHandoffError):
            coordinator.advance_protected_state(_metadata())
        assert store.read_verified_snapshot() is None
        assert boundary.value["lifecycle"] == "PREPARED"
        assert coordinator._recovery_required is True


def test_finalize_failure_retains_local_evidence_and_external_pending(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "s.db") as store:
        registry = LocalDurableEvidenceRegistry()
        boundary = Boundary(record("UNINITIALIZED"))
        boundary.fail_finalize = True
        with pytest.raises(RuntimeError):
            ProtectedFreshnessHandoffCoordinator(store, registry, boundary).advance_protected_state(
                _metadata()
            )
        snapshot = store.read_verified_snapshot()
        assert snapshot is not None
        assert boundary.value["lifecycle"] == "PREPARED"


def test_fresh_store_gate_denies_evidence_after_out_of_band_advance(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "s.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))

        def advance_again() -> None:
            boundary.after_publish = None
            _commit(store, _metadata(2), expected=1)

        boundary.after_publish = advance_again
        with pytest.raises(RuntimeError, match="finalize denied"):
            ProtectedFreshnessHandoffCoordinator(
                store, LocalDurableEvidenceRegistry(), boundary
            ).advance_protected_state(_metadata())
        assert store.read_verified_snapshot().metadata.protected_freshness_generation == 2  # type: ignore[union-attr]
        assert boundary.value["lifecycle"] == "PREPARED"


def test_registry_resolver_denies_same_generation_stale_and_manual_refs(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "s.db") as store:
        _commit(store, _metadata())
        registry = LocalDurableEvidenceRegistry()
        first = registry.publish_verified_state(store)
        second = registry.publish_verified_state(store)
        assert registry.resolve_current(SCOPE, first) is None
        assert registry.resolve_current(SCOPE, second) is not None
        valid = registry.resolve_current(SCOPE, second)
        assert valid is not None
        assert registry.resolve_current(SCOPE, replace(valid)) is None


def test_production_surface_has_no_owner_or_recovery_operations() -> None:
    forbidden = {
        "provision",
        "create_membership",
        "mint_membership",
        "register_scope",
        "select_current",
        "replace_reference",
        "retire",
        "unretire",
        "set_generation",
        "abort",
        "clear_pending",
        "rollback_prepare",
    }
    assert forbidden.isdisjoint(vars(ProtectedFreshnessHandoffCoordinator))
    assert forbidden.isdisjoint(
        vars(__import__("bot_core.persistence.protected_freshness_handoff", fromlist=["*"]))
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", "acct_01890f4c-7b9a-7cc1-8a2b-123456789abd"),
        ("device_installation_id", "dev_01890f4c-7b9a-7cc1-8a2b-123456789abd"),
        ("state_store_identity_fingerprint_sha256", "9" * 64),
        ("state_store_schema_version", 2),
    ],
)
def test_existing_store_metadata_preflight_denies_before_prepare(
    tmp_path: Path, field: str, value: object
) -> None:
    with SQLiteStateStore(tmp_path / f"{field}.db") as store:
        local = _commit(store, _metadata())
        candidate = replace(_metadata(2), **{field: value})
        external_scope = (
            candidate.account_id,
            candidate.device_installation_id,
            candidate.state_store_identity_fingerprint_sha256,
        )
        initial = record(
            "COMMITTED",
            committed_generation=1,
            committed_state_fingerprint_sha256=local.state_fingerprint_sha256,
        )
        initial.update(
            dict(
                zip(
                    (
                        "account_id",
                        "device_installation_id",
                        "state_store_identity_fingerprint_sha256",
                    ),
                    external_scope,
                    strict=True,
                )
            )
        )
        initial["content_fingerprint_sha256"] = canonical_json_sha256(
            {k: v for k, v in initial.items() if k != "content_fingerprint_sha256"}
        )
        boundary = Boundary(initial)
        coordinator = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        with pytest.raises(ProtectedFreshnessHandoffError):
            coordinator.advance_protected_state(candidate)
        assert "prepare" not in boundary.calls
        snapshot = store.read_verified_snapshot()
        assert snapshot is not None and snapshot.metadata.protected_freshness_generation == 1
        assert len(snapshot.transaction_descriptors) == 1
        assert coordinator._recovery_required is False


def test_finalize_ack_lost_latches_and_denies_second_business_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SQLiteStateStore(tmp_path / "ack.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        boundary.ack_lost = True
        coordinator = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        original = store.commit_prepared_state
        commits = 0

        def counted(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal commits
            commits += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(store, "commit_prepared_state", counted)
        with pytest.raises(RuntimeError, match="ack lost"):
            coordinator.advance_protected_state(_metadata())
        assert boundary.value["lifecycle"] == "COMMITTED"
        assert coordinator._recovery_required is True and commits == 1
        with pytest.raises(ProtectedFreshnessHandoffError, match="requires recovery"):
            coordinator.advance_protected_state(_metadata(2))
        assert commits == 1
        assert store.read_verified_snapshot().metadata.protected_freshness_generation == 1  # type: ignore[union-attr]


def test_local_commit_failure_after_prepare_latches_without_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SQLiteStateStore(tmp_path / "commit.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        coordinator = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        calls = 0

        def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            raise RuntimeError("local commit failed")

        monkeypatch.setattr(store, "commit_prepared_state", fail)
        with pytest.raises(RuntimeError, match="local commit failed"):
            coordinator.advance_protected_state(_metadata())
        assert calls == 1 and boundary.value["lifecycle"] == "PREPARED"
        assert store.read_verified_snapshot() is None
        assert "finalize" not in boundary.calls and coordinator._recovery_required is True
        with pytest.raises(ProtectedFreshnessHandoffError, match="requires recovery"):
            coordinator.advance_protected_state(_metadata())
        assert calls == 1


def test_evidence_publication_failure_latches_without_local_rollback(tmp_path: Path) -> None:
    class FailingRegistry(LocalDurableEvidenceRegistry):
        def publish_verified_state(self, store):  # type: ignore[no-untyped-def]
            raise RuntimeError("publication failed")

    with SQLiteStateStore(tmp_path / "evidence.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        coordinator = ProtectedFreshnessHandoffCoordinator(store, FailingRegistry(), boundary)
        with pytest.raises(RuntimeError, match="publication failed"):
            coordinator.advance_protected_state(_metadata())
        assert boundary.value["lifecycle"] == "PREPARED"
        assert store.read_verified_snapshot() is not None
        assert "finalize" not in boundary.calls and coordinator._recovery_required is True


def test_finalize_pending_failure_latches_and_denies_retry(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "finalize.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        boundary.fail_finalize = True
        coordinator = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        with pytest.raises(RuntimeError, match="finalize denied"):
            coordinator.advance_protected_state(_metadata())
        assert boundary.value["lifecycle"] == "PREPARED" and coordinator._recovery_required
        with pytest.raises(ProtectedFreshnessHandoffError, match="requires recovery"):
            coordinator.advance_protected_state(_metadata(2))


def test_stale_external_ref_during_prepare_denied_without_local_commit(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "stale.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        boundary.before_prepare = lambda: setattr(boundary, "ref", object())
        coordinator = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        with pytest.raises(RuntimeError, match="stale"):
            coordinator.advance_protected_state(_metadata())
        assert store.read_verified_snapshot() is None
        assert boundary.value["lifecycle"] == "UNINITIALIZED"
        assert coordinator._recovery_required


@pytest.mark.parametrize(
    "wrong",
    [
        {"prepared_generation": 2},
        {"prepared_state_fingerprint_sha256": "f" * 64},
        {"prepared_transaction_fingerprint_sha256": "f" * 64},
    ],
)
def test_wrong_prepare_ack_matrix_latches_before_local_commit(
    tmp_path: Path, wrong: dict[str, Any]
) -> None:
    with SQLiteStateStore(tmp_path / (next(iter(wrong)) + ".db")) as store:
        boundary = Boundary(record("UNINITIALIZED"))
        boundary.wrong_pending = wrong
        coordinator = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        with pytest.raises(ProtectedFreshnessHandoffError):
            coordinator.advance_protected_state(_metadata())
        assert store.read_verified_snapshot() is None and coordinator._recovery_required


def test_prepare_changed_ref_and_finalize_changed_ref_are_latched(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "prepare-ref.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        boundary.change_ref_after_prepare = True
        coordinator = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        with pytest.raises(ProtectedFreshnessHandoffError):
            coordinator.advance_protected_state(_metadata())
        assert store.read_verified_snapshot() is None and coordinator._recovery_required
    with SQLiteStateStore(tmp_path / "finalize-ref.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        boundary.change_ref_after_finalize = True
        coordinator = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        with pytest.raises(ProtectedFreshnessHandoffError):
            coordinator.advance_protected_state(_metadata())
        assert store.read_verified_snapshot() is not None and coordinator._recovery_required


@pytest.mark.parametrize(
    "changes",
    [
        {
            "lifecycle": "PREPARED",
            "prepared_generation": 2,
            "prepared_state_fingerprint_sha256": "a" * 64,
            "prepared_transaction_fingerprint_sha256": "b" * 64,
        },
        {"committed_generation": 2},
        {"committed_state_fingerprint_sha256": "f" * 64},
        {"content_fingerprint_sha256": "f" * 64},
    ],
)
def test_post_finalize_wrong_ack_matrix_is_not_success(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    with SQLiteStateStore(tmp_path / (next(iter(changes)) + ".db")) as store:
        boundary = Boundary(record("UNINITIALIZED"))
        boundary.post_finalize_changes = changes
        coordinator = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        with pytest.raises(ProtectedFreshnessHandoffError):
            coordinator.advance_protected_state(_metadata())
        assert store.read_verified_snapshot() is not None and coordinator._recovery_required


def test_same_generation_republish_during_finalize_denies_stale_ref(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "republish.db") as store:
        registry = LocalDurableEvidenceRegistry()
        boundary = Boundary(record("UNINITIALIZED"))
        boundary.after_publish = lambda: registry.publish_verified_state(store)
        coordinator = ProtectedFreshnessHandoffCoordinator(store, registry, boundary)
        with pytest.raises(RuntimeError, match="finalize denied"):
            coordinator.advance_protected_state(_metadata())
        assert boundary.value["lifecycle"] == "PREPARED" and coordinator._recovery_required


@pytest.mark.parametrize("external_generation", [1, 3])
def test_external_generation_mismatch_at_local_g2_has_zero_prepare(
    tmp_path: Path, external_generation: int
) -> None:
    with SQLiteStateStore(tmp_path / f"g{external_generation}.db") as store:
        _commit(store, _metadata())
        local = _commit(store, _metadata(2), expected=1)
        boundary = Boundary(
            record(
                "COMMITTED",
                committed_generation=external_generation,
                committed_state_fingerprint_sha256=local.state_fingerprint_sha256,
            )
        )
        coordinator = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        with pytest.raises(ProtectedFreshnessHandoffError):
            coordinator.advance_protected_state(_metadata(3))
        assert "prepare" not in boundary.calls and coordinator._recovery_required is False


def test_same_generation_different_external_state_has_zero_prepare(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        _commit(store, _metadata())
        _commit(store, _metadata(2), expected=1)
        boundary = Boundary(
            record("COMMITTED", committed_generation=2, committed_state_fingerprint_sha256="f" * 64)
        )
        coordinator = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        with pytest.raises(ProtectedFreshnessHandoffError):
            coordinator.advance_protected_state(_metadata(3))
        assert "prepare" not in boundary.calls and coordinator._recovery_required is False


def test_faithful_finalize_denies_valid_evidence_that_differs_from_pending(
    tmp_path: Path,
) -> None:
    with SQLiteStateStore(tmp_path / "wrong-evidence.db") as store:
        local = _commit(store, _metadata())
        registry = LocalDurableEvidenceRegistry()
        evidence_ref = registry.publish_verified_state(store)
        boundary = Boundary(
            record(
                "PREPARED",
                prepared_generation=1,
                prepared_state_fingerprint_sha256="f" * 64,
                prepared_transaction_fingerprint_sha256=local.transaction_fingerprint_sha256,
            )
        )
        resolver = ProtectedFreshnessHandoffCoordinator(
            store, registry, boundary
        )._store_bound_resolver(SCOPE)
        with pytest.raises(RuntimeError, match="evidence mismatch"):
            boundary.finalize(
                boundary.ref, SCOPE, evidence_ref=evidence_ref, evidence_resolver=resolver
            )
        assert boundary.value["lifecycle"] == "PREPARED"


def test_manual_self_hashed_evidence_has_no_finalize_membership(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "manual.db") as store:
        local = _commit(store, _metadata())
        projection = {
            "account_id": SCOPE[0],
            "device_installation_id": SCOPE[1],
            "state_store_identity_fingerprint_sha256": SCOPE[2],
            "generation": 1,
            "state_fingerprint_sha256": local.state_fingerprint_sha256,
            "transaction_fingerprint_sha256": local.transaction_fingerprint_sha256,
            "durability_state": "DURABLE_COMMITTED",
            "evidence_revision": 1,
        }
        manual = LocalDurableStateEvidence(
            **projection, evidence_fingerprint_sha256=canonical_json_sha256(projection)
        )
        boundary = Boundary(
            record(
                "PREPARED",
                prepared_generation=1,
                prepared_state_fingerprint_sha256=local.state_fingerprint_sha256,
                prepared_transaction_fingerprint_sha256=local.transaction_fingerprint_sha256,
            )
        )
        resolver = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )._store_bound_resolver(SCOPE)
        with pytest.raises(RuntimeError, match="finalize denied"):
            boundary.finalize(boundary.ref, SCOPE, evidence_ref=manual, evidence_resolver=resolver)
        assert boundary.value["lifecycle"] == "PREPARED"


def test_successful_path_has_exact_relative_call_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SQLiteStateStore(tmp_path / "order.db") as store:
        log: list[str] = []
        registry = LocalDurableEvidenceRegistry()

        class LoggingBoundary(Boundary):
            def resolve_current(self, scope):  # type: ignore[no-untyped-def]
                log.append("external resolve")
                return super().resolve_current(scope)

            def prepare(self, current_ref, scope, **candidate):  # type: ignore[no-untyped-def]
                log.append("prepare")
                return super().prepare(current_ref, scope, **candidate)

            def finalize(self, current_ref, scope, *, evidence_ref, evidence_resolver):  # type: ignore[no-untyped-def]
                log.append("finalize")
                return super().finalize(
                    current_ref,
                    scope,
                    evidence_ref=evidence_ref,
                    evidence_resolver=evidence_resolver,
                )

        boundary = LoggingBoundary(record("UNINITIALIZED"))
        originals = {
            "read": store.read_verified_snapshot,
            "derive": store.derive_prepared_metadata,
            "commit": store.commit_prepared_state,
            "publish": registry.publish_verified_state,
            "resolve": registry.resolve_current,
        }

        deriving = False

        def read():  # type: ignore[no-untyped-def]
            log.append("derive internal read" if deriving else "local verify")
            return originals["read"]()

        def derive(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal deriving
            log.append("derive")
            deriving = True
            try:
                return originals["derive"](*args, **kwargs)
            finally:
                deriving = False

        def commit(*args, **kwargs):  # type: ignore[no-untyped-def]
            log.append("local commit")
            return originals["commit"](*args, **kwargs)

        def publish(*args, **kwargs):  # type: ignore[no-untyped-def]
            log.append("publish")
            return originals["publish"](*args, **kwargs)

        def resolve(*args, **kwargs):  # type: ignore[no-untyped-def]
            log.append("evidence resolve")
            return originals["resolve"](*args, **kwargs)

        monkeypatch.setattr(store, "read_verified_snapshot", read)
        monkeypatch.setattr(store, "derive_prepared_metadata", derive)
        monkeypatch.setattr(store, "commit_prepared_state", commit)
        monkeypatch.setattr(registry, "publish_verified_state", publish)
        monkeypatch.setattr(registry, "resolve_current", resolve)
        ProtectedFreshnessHandoffCoordinator(store, registry, boundary).advance_protected_state(
            _metadata()
        )
        first = {name: log.index(name) for name in set(log)}
        assert first["external resolve"] < first["local verify"] < first["derive"]
        stability_reads = [
            index
            for index, event in enumerate(log)
            if event == "local verify" and first["derive"] < index < first["prepare"]
        ]
        assert len(stability_reads) == 1
        assert first["derive"] < stability_reads[0] < first["prepare"]
        assert first["prepare"] < first["local commit"]
        assert first["local commit"] < first["publish"] < first["evidence resolve"]
        assert first["evidence resolve"] < first["finalize"]
        assert log[-1] == "external resolve"
        assert log.index("external resolve", first["finalize"]) > first["finalize"]


def test_s3_observation_is_not_part_of_s5_public_surface() -> None:
    import inspect

    signatures = " ".join(
        str(inspect.signature(member))
        for member in (
            ProtectedFreshnessHandoffCoordinator.advance_protected_state,
            Boundary.finalize,
        )
    )
    assert "DurableStateObservation" not in signatures


def test_existing_g1_advance_during_derivation_fails_before_external_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SQLiteStateStore(tmp_path / "derive-race.db") as store:
        local_g1 = _commit(store, _metadata())
        boundary = Boundary(
            record(
                "COMMITTED",
                committed_generation=1,
                committed_state_fingerprint_sha256=local_g1.state_fingerprint_sha256,
            )
        )
        registry = LocalDurableEvidenceRegistry()
        coordinator = ProtectedFreshnessHandoffCoordinator(store, registry, boundary)
        original = store.derive_prepared_metadata
        raced = False

        def derive_with_race(metadata, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal raced
            if not raced:
                raced = True
                out_of_band = original(_metadata(2), expected_current_generation=1)
                store.commit_prepared_state(
                    out_of_band,
                    current_records=(),
                    immutable_history=(),
                    expected_current_generation=1,
                )
            return original(metadata, **kwargs)

        monkeypatch.setattr(store, "derive_prepared_metadata", derive_with_race)
        with pytest.raises(ProtectedFreshnessHandoffError, match="baseline changed"):
            coordinator.advance_protected_state(_metadata(2))
        assert boundary.value["lifecycle"] == "COMMITTED"
        assert boundary.value["committed_generation"] == 1
        assert "prepare" not in boundary.calls
        assert coordinator._recovery_required is False
        assert registry._accepted == {}
        snapshot = store.read_verified_snapshot()
        assert snapshot is not None and snapshot.metadata.protected_freshness_generation == 2


def test_genesis_becomes_g1_during_derivation_fails_before_external_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SQLiteStateStore(tmp_path / "genesis-derive-race.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        registry = LocalDurableEvidenceRegistry()
        coordinator = ProtectedFreshnessHandoffCoordinator(store, registry, boundary)
        original = store.derive_prepared_metadata
        raced = False

        def derive_with_race(metadata, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal raced
            if not raced:
                raced = True
                out_of_band = original(_metadata(), expected_current_generation=None)
                store.commit_prepared_state(
                    out_of_band,
                    current_records=(),
                    immutable_history=(),
                    expected_current_generation=None,
                )
            return original(metadata, **kwargs)

        monkeypatch.setattr(store, "derive_prepared_metadata", derive_with_race)
        with pytest.raises(ProtectedFreshnessHandoffError, match="baseline changed"):
            coordinator.advance_protected_state(_metadata())
        assert boundary.value["lifecycle"] == "UNINITIALIZED"
        assert "prepare" not in boundary.calls
        assert coordinator._recovery_required is False
        assert registry._accepted == {}
        snapshot = store.read_verified_snapshot()
        assert snapshot is not None and snapshot.metadata.protected_freshness_generation == 1
