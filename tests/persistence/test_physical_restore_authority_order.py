from __future__ import annotations

from pathlib import Path

import pytest

from bot_core.persistence.backup_authentication import (
    BackupArtifactAuthenticator,
    BackupArtifactVerifier,
)
from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.physical_backup import (
    PhysicalBackupAdmissionValidator,
    PhysicalBackupCreator,
)
from bot_core.persistence.protected_freshness_handoff import (
    ProtectedFreshnessHandoffCoordinator,
)
from bot_core.persistence.restore_protocol import (
    RestoreDecision,
    RestoreLifecycleAuthorityBundle,
    SecretHandoffRestoreObservation,
    TrustedPhysicalRestoreCoordinator,
)
from bot_core.persistence.secret_handoff import (
    DurableSecretHandoffLifecycleCoordinator,
    SecretHandoffError,
)
from bot_core.persistence.state_store import SQLiteStateStore
from tests.persistence.physical_backup_helpers import authority_for
from tests.persistence.test_protected_freshness_handoff import (
    Boundary as ProtectedBoundary,
    record,
)
from tests.persistence.test_restore_protocol import Boundary as RestoreBoundary, _record
from tests.persistence.test_secret_handoff import descriptor
from tests.persistence.test_state_store_records import _account, _metadata


class RecordingSecretRestoreAuthority:
    def __init__(
        self,
        events: list[str],
        fail_at: int | None = None,
        rounds: tuple[str, ...] = ("INITIAL", "PRE_INSTALL", "FINAL"),
    ) -> None:
        self.events = events
        self.calls: list[str] = []
        self.fail_at = fail_at
        self.rounds = rounds

    def observe(self, value):  # type: ignore[no-untyped-def]
        self.calls.append(value.handoff_id)
        call = len(self.calls)
        round_name = self.rounds[(call - 1) // 2]
        self.events.append(f"{round_name}:{value.handoff_id}")
        if call == self.fail_at:
            raise SecretHandoffError("injected secret observation failure")
        return SecretHandoffRestoreObservation(
            value.handoff_id,
            value.scope,
            value.operation_fingerprint_sha256,
            value.metadata_fingerprint_sha256,
            "NOT_STARTED",
        )


class RecordingEvidence(LocalDurableEvidenceRegistry):
    def __init__(
        self, events: list[str], recovery: dict[str, bool] | None = None
    ) -> None:
        super().__init__()
        self.events = events
        self.recovery = recovery or {"active": False}

    def publish_verified_state(self, store):  # type: ignore[no-untyped-def]
        self.events.append(
            "m03_recovery_evidence"
            if self.recovery["active"]
            else "restore_final_evidence"
        )
        return super().publish_verified_state(store)


def instrument_recovery(monkeypatch, events, recovery):  # type: ignore[no-untyped-def]
    original = ProtectedFreshnessHandoffCoordinator.recover_protected_state

    def wrapped(self, scope):  # type: ignore[no-untyped-def]
        events.append("recover_m03")
        recovery["active"] = True
        try:
            return original(self, scope)
        finally:
            recovery["active"] = False

    monkeypatch.setattr(
        ProtectedFreshnessHandoffCoordinator, "recover_protected_state", wrapped
    )


def two_secret_artifact(tmp_path: Path):
    store = SQLiteStateStore(tmp_path / "two-secret-source.sqlite")
    protected_boundary = ProtectedBoundary(record("UNINITIALIZED"))
    protected = ProtectedFreshnessHandoffCoordinator(
        store, LocalDurableEvidenceRegistry(), protected_boundary
    )
    protected.advance_protected_state(_metadata(), current_records=(_account(),))
    lifecycle = DurableSecretHandoffLifecycleCoordinator(store, protected)
    scope = (_metadata().account_id, _metadata().device_installation_id)
    lifecycle.prepare(descriptor(handoff_id="handoff-a", scope=scope))
    lifecycle.prepare(descriptor(handoff_id="handoff-b", scope=scope))
    authority = authority_for(tmp_path, store)
    artifact = PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
        store, tmp_path / "two-secret-backup.sqlite"
    )
    return store, artifact, BackupArtifactVerifier(authority)


def make_coordinator(live, artifact, verifier, secret, evidence):  # type: ignore[no-untyped-def]
    return TrustedPhysicalRestoreCoordinator(
        live,
        evidence,
        RestoreBoundary(artifact.backup_envelope),
        PhysicalBackupAdmissionValidator(verifier),
        RestoreLifecycleAuthorityBundle(secret_handoff_restore_authority=secret),
    )


def prepared_boundary(artifact):  # type: ignore[no-untyped-def]
    boundary = RestoreBoundary(artifact.backup_envelope)
    descriptor = (
        artifact.backup_envelope.integrity_metadata.state_store_transaction_descriptors[
            -1
        ]
    )
    boundary.value = _record(
        artifact.backup_envelope,
        "PREPARED",
        committed_generation=descriptor.expected_current_generation,
        committed_state_fingerprint_sha256=descriptor.pre_state_fingerprint_sha256,
        prepared_generation=artifact.backup_envelope.local_protected_freshness_generation,
        prepared_state_fingerprint_sha256=artifact.backup_envelope.state_fingerprint_sha256,
        prepared_transaction_fingerprint_sha256=artifact.backup_envelope.transaction_fingerprint_sha256,
    )
    return boundary


def test_two_secret_install_observes_every_fence_in_exact_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, artifact, verifier = two_secret_artifact(tmp_path)
    expected = source.read_verified_snapshot()
    source.close()
    events: list[str] = []
    secret = RecordingSecretRestoreAuthority(events)
    evidence = RecordingEvidence(events)
    original = SQLiteStateStore.atomic_replace

    def atomic(staged, live):  # type: ignore[no-untyped-def]
        events.append("atomic_replace")
        return original(staged, live)

    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", atomic)
    live = tmp_path / "two-secret-live.sqlite"
    result = make_coordinator(
        live, artifact, verifier, secret, evidence
    ).restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    assert secret.calls == ["handoff-a", "handoff-b"] * 3
    assert events == [
        "INITIAL:handoff-a",
        "INITIAL:handoff-b",
        "PRE_INSTALL:handoff-a",
        "PRE_INSTALL:handoff-b",
        "atomic_replace",
        "FINAL:handoff-a",
        "FINAL:handoff-b",
        "restore_final_evidence",
    ]
    assert events.index("PRE_INSTALL:handoff-b") + 1 == events.index("atomic_replace")
    with SQLiteStateStore(live) as reopened:
        assert reopened.read_verified_snapshot() == expected


def _assert_secret_failure_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: int
) -> None:
    source, artifact, verifier = two_secret_artifact(tmp_path)
    source.close()
    events: list[str] = []
    secret = RecordingSecretRestoreAuthority(events, fail_at)
    evidence = RecordingEvidence(events)
    atomic_calls = 0
    original = SQLiteStateStore.atomic_replace

    def atomic(staged, live):  # type: ignore[no-untyped-def]
        nonlocal atomic_calls
        atomic_calls += 1
        return original(staged, live)

    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", atomic)
    result = make_coordinator(
        tmp_path / f"failure-{fail_at}.sqlite", artifact, verifier, secret, evidence
    ).restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.DENY
    assert "restore_final_evidence" not in events
    assert atomic_calls == (1 if fail_at == 5 else 0)


def test_second_secret_initial_failure_stops_before_stage3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_secret_failure_position(tmp_path, monkeypatch, 2)


def test_second_secret_preinstall_failure_prevents_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_secret_failure_position(tmp_path, monkeypatch, 4)


def test_ordinary_final_failure_denies_without_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_secret_failure_position(tmp_path, monkeypatch, 5)


def test_prepared_install_uses_two_fresh_final_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, artifact, verifier = two_secret_artifact(tmp_path)
    source.close()
    events: list[str] = []
    secret = RecordingSecretRestoreAuthority(
        events, rounds=("INITIAL", "PRE_INSTALL", "FINAL_1", "FINAL_2")
    )
    recovery = {"active": False}
    instrument_recovery(monkeypatch, events, recovery)
    evidence = RecordingEvidence(events, recovery)
    coordinator = TrustedPhysicalRestoreCoordinator(
        tmp_path / "prepared-install.sqlite",
        evidence,
        prepared_boundary(artifact),
        PhysicalBackupAdmissionValidator(verifier),
        RestoreLifecycleAuthorityBundle(secret_handoff_restore_authority=secret),
    )

    result = coordinator.restore_trusted_artifact(artifact)

    assert (
        result.decision is RestoreDecision.RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE
    )
    assert secret.calls == ["handoff-a", "handoff-b"] * 4
    assert events[-3:] == [
        "FINAL_2:handoff-a",
        "FINAL_2:handoff-b",
        "restore_final_evidence",
    ]
    assert events.index("FINAL_1:handoff-b") + 1 == events.index("recover_m03")
    assert events.count("recover_m03") == 1
    assert events.count("m03_recovery_evidence") == 1


def test_prepared_true_noop_skips_c2d_and_preinstall_but_runs_two_finals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, artifact, verifier = two_secret_artifact(tmp_path)
    events: list[str] = []
    secret = RecordingSecretRestoreAuthority(
        events, rounds=("INITIAL", "FINAL_1", "FINAL_2")
    )
    admission = PhysicalBackupAdmissionValidator(verifier)
    admission.admit = pytest.fail  # type: ignore[method-assign]
    recovery = {"active": False}
    instrument_recovery(monkeypatch, events, recovery)
    coordinator = TrustedPhysicalRestoreCoordinator(
        source.path,
        RecordingEvidence(events, recovery),
        prepared_boundary(artifact),
        admission,
        RestoreLifecycleAuthorityBundle(secret_handoff_restore_authority=secret),
    )
    try:
        result = coordinator.restore_trusted_artifact(artifact)
    finally:
        source.close()

    assert (
        result.decision is RestoreDecision.RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE
    )
    assert secret.calls == ["handoff-a", "handoff-b"] * 3
    assert not any("PRE_INSTALL" in event for event in events)
    assert events[-1] == "restore_final_evidence"
    assert events.index("FINAL_1:handoff-b") + 1 == events.index("recover_m03")


def _assert_prepared_final_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: int
) -> None:
    source, artifact, verifier = two_secret_artifact(tmp_path)
    source.close()
    events: list[str] = []
    secret = RecordingSecretRestoreAuthority(
        events,
        fail_at,
        rounds=("INITIAL", "PRE_INSTALL", "FINAL_1", "FINAL_2"),
    )
    recovery = {"active": False}
    instrument_recovery(monkeypatch, events, recovery)
    boundary = prepared_boundary(artifact)
    coordinator = TrustedPhysicalRestoreCoordinator(
        tmp_path / f"prepared-failure-{fail_at}.sqlite",
        RecordingEvidence(events, recovery),
        boundary,
        PhysicalBackupAdmissionValidator(verifier),
        RestoreLifecycleAuthorityBundle(secret_handoff_restore_authority=secret),
    )

    result = coordinator.restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.DENY
    assert events.count("recover_m03") == (1 if fail_at == 7 else 0)
    assert events.count("m03_recovery_evidence") == (1 if fail_at == 7 else 0)
    assert events.count("restore_final_evidence") == 0
    assert len(secret.calls) == fail_at
    assert boundary.value["lifecycle"] == ("COMMITTED" if fail_at == 7 else "PREPARED")
    assert "abort" not in boundary.calls and "prepare" not in boundary.calls


def test_prepared_first_final_failure_does_not_finalize_m03(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_prepared_final_failure(tmp_path, monkeypatch, 5)


def test_prepared_second_final_failure_preserves_completed_m03_but_denies_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_prepared_final_failure(tmp_path, monkeypatch, 7)
