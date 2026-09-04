from __future__ import annotations

from pathlib import Path

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
