from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.physical_backup import PhysicalBackupAdmissionValidator
from bot_core.persistence.restore_protocol import (
    RestoreDecision,
    RestoreLifecycleAuthorityBundle,
    TrustedPhysicalRestoreCoordinator,
)
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreError
from tests.persistence.physical_backup_helpers import artifact_fixture
from tests.persistence.test_restore_protocol import Boundary


def coordinator(live, boundary, admission):  # type: ignore[no-untyped-def]
    return TrustedPhysicalRestoreCoordinator(
        live,
        LocalDurableEvidenceRegistry(),
        boundary,
        admission,
        RestoreLifecycleAuthorityBundle(),
    )


def test_final_staged_tamper_denies_before_atomic_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    source.close()
    live = tmp_path / "staged-tamper.sqlite"
    original = TrustedPhysicalRestoreCoordinator._copy_for_install

    def tamper(candidate, target):  # type: ignore[no-untyped-def]
        staged = original(candidate, target)
        with staged.open("ab") as stream:
            stream.write(b"tamper")
        return staged

    monkeypatch.setattr(
        TrustedPhysicalRestoreCoordinator, "_copy_for_install", staticmethod(tamper)
    )
    atomic = Mock(side_effect=AssertionError("must not install"))
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", atomic)

    result = coordinator(
        live,
        Boundary(artifact.backup_envelope),
        PhysicalBackupAdmissionValidator(verifier),
    ).restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.DENY
    atomic.assert_not_called()
    assert not live.exists()


def test_open_live_handle_blocks_physical_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    source.close()
    live = tmp_path / "open-live.sqlite"
    open_live = SQLiteStateStore(live)
    atomic = Mock(side_effect=AssertionError("must not install"))
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", atomic)
    try:
        result = coordinator(
            live,
            Boundary(artifact.backup_envelope),
            PhysicalBackupAdmissionValidator(verifier),
        ).restore_trusted_artifact(artifact)
    finally:
        open_live.close()

    assert result.decision is RestoreDecision.DENY
    atomic.assert_not_called()


def test_original_artifact_swap_after_admission_uses_private_candidate(
    tmp_path: Path,
) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    expected = source.read_verified_snapshot()
    source.close()
    admission = PhysicalBackupAdmissionValidator(verifier)
    original = admission.admit

    def admit_and_swap(value):  # type: ignore[no-untyped-def]
        candidate = original(value)
        value.physical_artifact.path.write_bytes(b"substituted original")
        return candidate

    admission.admit = admit_and_swap  # type: ignore[method-assign]
    live = tmp_path / "artifact-swap-live.sqlite"
    result = coordinator(
        live, Boundary(artifact.backup_envelope), admission
    ).restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    with SQLiteStateStore(live) as installed:
        assert installed.read_verified_snapshot() == expected


def test_late_exact_discards_staging_without_atomic_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    source.close()
    live = tmp_path / "late-exact.sqlite"
    original_copy = TrustedPhysicalRestoreCoordinator._copy_for_install

    def make_live_exact(candidate, target):  # type: ignore[no-untyped-def]
        staged = original_copy(candidate, target)
        concurrent = target.with_suffix(".concurrent.sqlite")
        shutil.copy2(candidate.private_path, concurrent)
        SQLiteStateStore.atomic_replace(concurrent, target)
        return staged

    monkeypatch.setattr(
        TrustedPhysicalRestoreCoordinator,
        "_copy_for_install",
        staticmethod(make_live_exact),
    )
    original_atomic = SQLiteStateStore.atomic_replace
    installs = Mock(wraps=original_atomic)
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", installs)

    result = coordinator(
        live,
        Boundary(artifact.backup_envelope),
        PhysicalBackupAdmissionValidator(verifier),
    ).restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.NOOP_ALREADY_CURRENT
    assert installs.call_count == 1  # only the simulated concurrent actor


def test_restore_admitted_deny_preserves_borrowed_lease(tmp_path: Path) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    source.close()
    admission = PhysicalBackupAdmissionValidator(verifier)
    candidate = admission.admit(artifact)
    live = tmp_path / "borrowed-deny.sqlite"
    boundary = Boundary(artifact.backup_envelope)
    boundary.value["committed_state_fingerprint_sha256"] = "f" * 64
    try:
        result = coordinator(live, boundary, admission).restore_admitted(candidate)
        assert result.decision is RestoreDecision.DENY
        candidate.verify_physical_continuity()
    finally:
        candidate.close()


def test_stage3_failure_stops_before_m03(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    source.close()
    boundary = Boundary(artifact.backup_envelope)
    coordinator_ = coordinator(
        tmp_path / "stage3.sqlite", boundary, PhysicalBackupAdmissionValidator(verifier)
    )
    original = coordinator_._secret_lifecycles
    calls = 0

    def fail_second(snapshot):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            from bot_core.persistence.secret_handoff import SecretHandoffError

            raise SecretHandoffError("injected Stage3 failure")
        return original(snapshot)

    monkeypatch.setattr(coordinator_, "_secret_lifecycles", fail_second)
    result = coordinator_.restore_trusted_artifact(artifact)
    assert result.decision is RestoreDecision.DENY
    assert boundary.calls == []


def test_m03_failure_stops_before_preinstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    source.close()
    boundary = Boundary(artifact.backup_envelope)
    boundary.resolve_current = Mock(return_value=None)  # type: ignore[method-assign]
    atomic = Mock(side_effect=AssertionError("must not install"))
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", atomic)
    result = coordinator(
        tmp_path / "m03-failure.sqlite",
        boundary,
        PhysicalBackupAdmissionValidator(verifier),
    ).restore_trusted_artifact(artifact)
    assert result.decision is RestoreDecision.DENY
    atomic.assert_not_called()


def test_post_install_reopen_failure_denies_before_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    source.close()
    live = tmp_path / "reopen-failure.sqlite"
    installed = False
    original_atomic = SQLiteStateStore.atomic_replace
    original_read = SQLiteStateStore.read_verified_snapshot

    def atomic(staged, target):  # type: ignore[no-untyped-def]
        nonlocal installed
        original_atomic(staged, target)
        installed = True

    def read(store):  # type: ignore[no-untyped-def]
        if installed and store.path == live:
            raise StateStoreError("injected reopen failure")
        return original_read(store)

    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", atomic)
    monkeypatch.setattr(SQLiteStateStore, "read_verified_snapshot", read)
    result = coordinator(
        live,
        Boundary(artifact.backup_envelope),
        PhysicalBackupAdmissionValidator(verifier),
    ).restore_trusted_artifact(artifact)
    assert result.decision is RestoreDecision.DENY


def test_post_install_reopen_mismatch_denies_before_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    source.close()
    live = tmp_path / "reopen-mismatch.sqlite"
    installed = False
    original_atomic = SQLiteStateStore.atomic_replace
    original_read = SQLiteStateStore.read_verified_snapshot

    def atomic(staged, target):  # type: ignore[no-untyped-def]
        nonlocal installed
        original_atomic(staged, target)
        installed = True

    def read(store):  # type: ignore[no-untyped-def]
        if installed and store.path == live:
            return None
        return original_read(store)

    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", atomic)
    monkeypatch.setattr(SQLiteStateStore, "read_verified_snapshot", read)
    result = coordinator(
        live,
        Boundary(artifact.backup_envelope),
        PhysicalBackupAdmissionValidator(verifier),
    ).restore_trusted_artifact(artifact)
    assert result.decision is RestoreDecision.DENY


def test_fresh_forbidden_live_change_prevents_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bot_core.persistence.restore_protocol import LocalRestoreClassification

    source, _authentication, artifact, verifier = artifact_fixture(tmp_path)
    source.close()
    coordinator_ = coordinator(
        tmp_path / "fresh-forbidden.sqlite",
        Boundary(artifact.backup_envelope),
        PhysicalBackupAdmissionValidator(verifier),
    )
    original = coordinator_._classify
    calls = 0

    def classify(envelope):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls >= 3:
            return LocalRestoreClassification.AHEAD, None
        return original(envelope)

    monkeypatch.setattr(coordinator_, "_classify", classify)
    atomic = Mock(side_effect=AssertionError("must not overwrite fresh ahead state"))
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", atomic)
    result = coordinator_.restore_trusted_artifact(artifact)
    assert result.decision is RestoreDecision.DENY
    atomic.assert_not_called()
