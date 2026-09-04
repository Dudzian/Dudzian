from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from unittest.mock import Mock

import pytest

from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.physical_backup import PhysicalBackupAdmissionValidator
from bot_core.persistence.restore_protocol import (
    RestoreDecision,
    RestoreLifecycleAuthorityBundle,
    TrustedPhysicalRestoreCoordinator,
)
from bot_core.persistence.state_store import SQLiteStateStore
from tests.persistence.physical_backup_helpers import artifact_fixture
from tests.persistence.test_restore_protocol import Boundary


def _coordinator(tmp_path: Path, live: Path, boundary: Boundary, verifier):  # type: ignore[no-untyped-def]
    return TrustedPhysicalRestoreCoordinator(
        live,
        LocalDurableEvidenceRegistry(),
        boundary,
        PhysicalBackupAdmissionValidator(verifier),
        RestoreLifecycleAuthorityBundle(),
    )


def test_zero_secret_physical_candidate_installs_and_reopens_exactly(
    tmp_path: Path,
) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    candidate_snapshot = source.read_verified_snapshot()
    assert candidate_snapshot is not None
    source.close()
    live = tmp_path / "restored.sqlite"
    boundary = Boundary(artifact.backup_envelope)

    result = _coordinator(tmp_path, live, boundary, verifier).restore_trusted_artifact(
        artifact
    )

    assert result.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    with SQLiteStateStore(live) as reopened:
        assert reopened.read_verified_snapshot() == candidate_snapshot


def test_true_noop_does_not_replace_live_physical_bytes(tmp_path: Path) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    before = source.path.read_bytes()
    boundary = Boundary(artifact.backup_envelope)

    result = _coordinator(
        tmp_path, source.path, boundary, verifier
    ).restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.NOOP_ALREADY_CURRENT
    assert source.path.read_bytes() == before
    source.close()


def test_true_noop_never_enters_physical_admission(tmp_path: Path) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    boundary = Boundary(artifact.backup_envelope)
    admission = PhysicalBackupAdmissionValidator(verifier)
    admission.admit = Mock(side_effect=AssertionError("C2D must not run"))  # type: ignore[method-assign]
    coordinator = TrustedPhysicalRestoreCoordinator(
        source.path,
        LocalDurableEvidenceRegistry(),
        boundary,
        admission,
        RestoreLifecycleAuthorityBundle(),
    )

    result = coordinator.restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.NOOP_ALREADY_CURRENT
    admission.admit.assert_not_called()
    source.close()


def test_true_noop_calls_are_process_serialized(tmp_path: Path) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)

    class BlockingBoundary(Boundary):
        def __init__(self):
            super().__init__(artifact.backup_envelope)
            self.entered = Event()
            self.release = Event()
            self.second_entered = Event()
            self.guard = Lock()
            self.resolve_count = 0

        def resolve_current(self, scope):  # type: ignore[no-untyped-def]
            with self.guard:
                self.resolve_count += 1
                count = self.resolve_count
            if count == 1:
                self.entered.set()
                assert self.release.wait(2)
            elif count == 2:
                # Reaching call two means the second restore entered authority work.
                self.second_entered.set()
            return super().resolve_current(scope)

    boundary = BlockingBoundary()
    coordinator = _coordinator(tmp_path, source.path, boundary, verifier)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(coordinator.restore_trusted_artifact, artifact)
        assert boundary.entered.wait(2)
        second = pool.submit(coordinator.restore_trusted_artifact, artifact)
        assert not boundary.second_entered.wait(0.2)
        boundary.release.set()
        assert first.result().decision is RestoreDecision.NOOP_ALREADY_CURRENT
        assert second.result().decision is RestoreDecision.NOOP_ALREADY_CURRENT
    assert boundary.second_entered.is_set()
    source.close()


def test_entry_continuity_failure_prevents_install(tmp_path: Path) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    source.close()
    admission = PhysicalBackupAdmissionValidator(verifier)
    candidate = admission.admit(artifact)
    candidate.private_path.write_bytes(b"not sqlite")
    live = tmp_path / "restored.sqlite"
    boundary = Boundary(artifact.backup_envelope)
    coordinator = TrustedPhysicalRestoreCoordinator(
        live,
        LocalDurableEvidenceRegistry(),
        boundary,
        admission,
        RestoreLifecycleAuthorityBundle(),
    )
    try:
        result = coordinator.restore_admitted(candidate)
    finally:
        candidate.close()

    assert result.decision is RestoreDecision.DENY
    assert not live.exists()


@pytest.mark.parametrize("fail", [False, True])
def test_high_level_restore_always_closes_owned_candidate(
    tmp_path: Path, fail: bool
) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    source.close()
    admission = PhysicalBackupAdmissionValidator(verifier)
    admitted = admission.admit(artifact)
    original_admit = admission.admit
    admission.admit = Mock(return_value=admitted)  # type: ignore[method-assign]
    if fail:
        admitted.private_path.write_bytes(b"tampered")
    coordinator = TrustedPhysicalRestoreCoordinator(
        tmp_path / "owned.sqlite",
        LocalDurableEvidenceRegistry(),
        Boundary(artifact.backup_envelope),
        admission,
        RestoreLifecycleAuthorityBundle(),
    )

    result = coordinator.restore_trusted_artifact(artifact)

    assert result.decision is (
        RestoreDecision.DENY
        if fail
        else RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    )
    assert admitted._directory is None
    admission.admit = original_admit  # type: ignore[method-assign]


def test_restore_admitted_borrows_candidate_lease(tmp_path: Path) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    source.close()
    admission = PhysicalBackupAdmissionValidator(verifier)
    candidate = admission.admit(artifact)
    coordinator = TrustedPhysicalRestoreCoordinator(
        tmp_path / "borrowed.sqlite",
        LocalDurableEvidenceRegistry(),
        Boundary(artifact.backup_envelope),
        admission,
        RestoreLifecycleAuthorityBundle(),
    )
    try:
        result = coordinator.restore_admitted(candidate)
        assert result.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
        candidate.verify_physical_continuity()
        assert candidate._directory is not None
    finally:
        candidate.close()


def test_atomic_replace_failure_denies_without_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    source.close()
    live = tmp_path / "atomic-failure.sqlite"
    calls = Mock(side_effect=OSError("injected pre-replacement failure"))
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", calls)

    result = _coordinator(
        tmp_path, live, Boundary(artifact.backup_envelope), verifier
    ).restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.DENY
    calls.assert_called_once()
    assert not live.exists()
