from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3

import pytest

from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.restore_migration_install import (
    RestoreMigrationInstallCoordinator,
    RestoreMigrationInstallDisposition,
    RestoreMigrationInstallError,
)
from bot_core.persistence.restore_migration_resume import RestoreMigrationResumeCoordinator
from bot_core.persistence.restore_migration_staging import (
    restore_migration_manifest_path,
    restore_migration_staged_sqlite_path,
    restore_migration_staging_id,
)
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreError
from tests.persistence.test_protected_freshness_handoff import record
from tests.persistence.test_restore_migration_resume import _boundary_for
from tests.persistence.test_restore_migration_staging import _legacy_backup


def _paths(backup, live: Path) -> tuple[Path, Path]:  # type: ignore[no-untyped-def]
    staging_id = restore_migration_staging_id(
        state_store_identity_fingerprint_sha256=backup.state_store_identity_fingerprint_sha256,
        source_backup_envelope_fingerprint_sha256=backup.envelope_fingerprint_sha256,
    )
    return (
        restore_migration_staged_sqlite_path(live, staging_id),
        restore_migration_manifest_path(live, staging_id),
    )


def _coordinator(backup, live: Path, boundary=None):  # type: ignore[no-untyped-def]
    return RestoreMigrationInstallCoordinator(
        live,
        LocalDurableEvidenceRegistry(),
        boundary or _boundary_for(backup),
    )


def test_happy_install_and_manifest_free_retry_are_idempotent(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)

    first = _coordinator(backup, live, boundary).install(backup)
    staged, manifest = _paths(backup, live)
    second = _coordinator(backup, live, boundary).install(backup)

    assert first.disposition is RestoreMigrationInstallDisposition.INSTALLED
    assert second.disposition is RestoreMigrationInstallDisposition.ALREADY_INSTALLED
    assert second.installed_snapshot == first.installed_snapshot
    assert live.exists() and not staged.exists() and not manifest.exists()


def test_gate_order_wraps_real_atomic_replace(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    staged, _ = _paths(backup, live)
    events: list[str] = []
    original_gate = SQLiteStateStore.installation_gate.__func__  # type: ignore[attr-defined]
    original_replace = SQLiteStateStore.atomic_replace

    from contextlib import contextmanager

    @contextmanager
    def gate(cls, path):  # type: ignore[no-untyped-def]
        name = "staging" if Path(path).resolve() == staged else "live"
        events.append(f"enter {name}")
        with original_gate(cls, path):
            yield
        events.append(f"exit {name}")

    def replace(source, target):  # type: ignore[no-untyped-def]
        events.append("atomic_replace")
        original_replace(source, target)

    monkeypatch.setattr(SQLiteStateStore, "installation_gate", classmethod(gate))
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", replace)
    _coordinator(backup, live).install(backup)

    assert events == [
        "enter staging",
        "enter live",
        "atomic_replace",
        "exit live",
        "exit staging",
        "enter staging",
        "exit staging",
    ]


def test_extra_staging_handle_is_not_force_closed_and_retry_succeeds(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)
    extra = SQLiteStateStore(completed.sqlite_path)
    try:
        with pytest.raises(RestoreMigrationInstallError, match="STAGING_PATH_NOT_QUIESCENT"):
            _coordinator(backup, live, boundary).install(backup)
        assert completed.sqlite_path.exists()
    finally:
        extra.close()
    assert _coordinator(backup, live, boundary).install(backup).disposition is (
        RestoreMigrationInstallDisposition.INSTALLED
    )


def test_real_replace_failure_after_rename_recovers_before_resume(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    original = SQLiteStateStore.atomic_replace

    def crash_after_replace(source, target):  # type: ignore[no-untyped-def]
        original(source, target)
        raise StateStoreError("injected crash after rename")

    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", crash_after_replace)
    with pytest.raises(StateStoreError, match="injected crash"):
        _coordinator(backup, live, boundary).install(backup)
    staged, manifest = _paths(backup, live)
    assert live.exists() and not staged.exists() and manifest.exists()

    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", original)
    result = _coordinator(backup, live, boundary).install(backup)
    assert result.disposition is RestoreMigrationInstallDisposition.ALREADY_INSTALLED
    assert not manifest.exists()


def test_manifest_without_staging_and_non_target_live_requires_manual_recovery(
    tmp_path: Path,
) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), _boundary_for(backup)
    ).resume(backup)
    completed.sqlite_path.unlink()
    live.write_bytes(b"not-the-completed-target")

    with pytest.raises(RestoreMigrationInstallError, match="MANUAL_RECOVERY_REQUIRED"):
        _coordinator(backup, live).install(backup)


def test_closed_main_is_wal_independent_after_install_preparation(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), _boundary_for(backup)
    ).resume(backup)
    with SQLiteStateStore(completed.sqlite_path) as staged:
        expected = staged.read_verified_snapshot()
        staged.prepare_for_atomic_install()
    assert SQLiteStateStore.read_isolated_verified_snapshot(completed.sqlite_path) == expected


@pytest.mark.parametrize("change", ["revision", "ref"])
def test_external_observation_change_after_f1_prevents_replace(
    tmp_path: Path, monkeypatch, change: str
) -> None:  # type: ignore[no-untyped-def]
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    coordinator = _coordinator(backup, live, boundary)
    original_resolve = coordinator._resolve_external_exact
    calls = 0
    replacements = 0

    def changed(source, snapshot):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            if change == "revision":
                boundary.value = record(
                    "COMMITTED",
                    revision=boundary.value["authority_revision"] + 1,
                    committed_generation=snapshot.metadata.protected_freshness_generation,
                    committed_state_fingerprint_sha256=snapshot.metadata.state_fingerprint_sha256,
                )
            else:
                boundary.ref = object()
        return original_resolve(source, snapshot)

    def forbidden(*_args):  # type: ignore[no-untyped-def]
        nonlocal replacements
        replacements += 1

    monkeypatch.setattr(coordinator, "_resolve_external_exact", changed)
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", forbidden)
    with pytest.raises(RestoreMigrationInstallError, match="EXTERNAL_CHANGED"):
        coordinator.install(backup)
    staged, manifest = _paths(backup, live)
    assert replacements == 0
    assert staged.exists() and manifest.exists() and not live.exists()


def test_f3_recomputes_closed_physical_schema_and_rejects_tamper(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    staged_path, manifest = _paths(backup, live)
    original_prepare = SQLiteStateStore.prepare_for_atomic_install
    replacements = 0

    def prepare_then_tamper(store):  # type: ignore[no-untyped-def]
        original_prepare(store)
        if store.path == staged_path:
            with sqlite3.connect(staged_path) as connection:
                connection.execute("CREATE TABLE unrelated_physical_tamper(id INTEGER)")
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def forbidden(*_args):  # type: ignore[no-untyped-def]
        nonlocal replacements
        replacements += 1

    monkeypatch.setattr(SQLiteStateStore, "prepare_for_atomic_install", prepare_then_tamper)
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", forbidden)
    with pytest.raises(RestoreMigrationInstallError, match="TARGET_MISMATCH"):
        _coordinator(backup, live).install(backup)
    assert replacements == 0
    assert staged_path.exists() and manifest.exists() and not live.exists()


def test_exact_target_cleanup_requires_reacquired_staging_quiescence(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)
    _, manifest_path = _paths(backup, live)
    with SQLiteStateStore(completed.sqlite_path) as staged:
        staged.prepare_for_atomic_install()
    shutil.copyfile(completed.sqlite_path, live)
    coordinator = _coordinator(backup, live, boundary)
    original_proof = coordinator._read_and_prove
    extra: SQLiteStateStore | None = None

    def proof_then_open_staging(path, source, manifest):  # type: ignore[no-untyped-def]
        nonlocal extra
        result = original_proof(path, source, manifest)
        if path == live and extra is None:
            extra = SQLiteStateStore(completed.sqlite_path)
        return result

    monkeypatch.setattr(coordinator, "_read_and_prove", proof_then_open_staging)
    with pytest.raises(RestoreMigrationInstallError, match="CLEANUP_PENDING"):
        coordinator.install(backup)
    assert extra is not None
    assert completed.sqlite_path.exists() and manifest_path.exists()
    extra.close()

    result = _coordinator(backup, live, boundary).install(backup)
    assert result.disposition is RestoreMigrationInstallDisposition.ALREADY_INSTALLED
    assert not completed.sqlite_path.exists() and not manifest_path.exists()


def test_live_gate_handle_is_not_mislabeled_as_staging_conflict(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live_path = tmp_path / "live.sqlite3"
    live = SQLiteStateStore(live_path)
    try:
        with pytest.raises(RestoreMigrationInstallError, match="LIVE_PATH_NOT_QUIESCENT"):
            _coordinator(backup, live_path).install(backup)
    finally:
        live.close()


@pytest.mark.parametrize("change", ["revision", "ref"])
def test_external_change_after_live_checkpoint_prevents_replace(
    tmp_path: Path, monkeypatch, change: str
) -> None:  # type: ignore[no-untyped-def]
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live_path = tmp_path / "live.sqlite3"
    with SQLiteStateStore(live_path):
        pass
    boundary = _boundary_for(backup)
    original_prepare = SQLiteStateStore.prepare_for_atomic_install
    original_replace = SQLiteStateStore.atomic_replace
    replacements = 0

    def prepare_then_change(store):  # type: ignore[no-untyped-def]
        original_prepare(store)
        if store.path == live_path:
            if change == "revision":
                boundary.value = record(
                    "COMMITTED",
                    revision=boundary.value["authority_revision"] + 1,
                    committed_generation=boundary.value["committed_generation"],
                    committed_state_fingerprint_sha256=boundary.value[
                        "committed_state_fingerprint_sha256"
                    ],
                )
            else:
                boundary.ref = object()

    def counted_replace(source, target):  # type: ignore[no-untyped-def]
        nonlocal replacements
        replacements += 1
        original_replace(source, target)

    monkeypatch.setattr(SQLiteStateStore, "prepare_for_atomic_install", prepare_then_change)
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", counted_replace)
    with pytest.raises(RestoreMigrationInstallError, match="EXTERNAL_CHANGED"):
        _coordinator(backup, live_path, boundary).install(backup)

    staged, manifest = _paths(backup, live_path)
    assert replacements == 0
    assert staged.exists() and manifest.exists() and live_path.exists()
    assert SQLiteStateStore.read_isolated_verified_snapshot(live_path) is None


def test_exact_target_runs_final_secret_fence_before_noop_cleanup(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)
    _, manifest = _paths(backup, live)
    with SQLiteStateStore(completed.sqlite_path) as staged:
        staged.prepare_for_atomic_install()
    shutil.copyfile(completed.sqlite_path, live)
    coordinator = _coordinator(backup, live, boundary)
    fences: list[str] = []
    replacements = 0

    def reject_final(_snapshot, fence):  # type: ignore[no-untyped-def]
        nonlocal replacements
        fences.append(fence.value)
        if fence.value == "FINAL_PROMOTION":
            raise RestoreMigrationInstallError("secret final fence rejected")

    def forbidden(*_args):  # type: ignore[no-untyped-def]
        nonlocal replacements
        replacements += 1

    monkeypatch.setattr(coordinator, "_observe_secrets", reject_final)
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", forbidden)
    with pytest.raises(RestoreMigrationInstallError, match="secret final fence rejected"):
        coordinator.install(backup)

    assert fences == ["PRE_INSTALL", "FINAL_PROMOTION"]
    assert replacements == 0
    assert completed.sqlite_path.exists() and manifest.exists()


def test_staging_lost_does_not_abort_prepared_external_from_non_target_live(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.sqlite3"
    backup = _legacy_backup(source_path)
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)
    _, manifest = _paths(backup, live)
    completed.sqlite_path.unlink()
    shutil.copyfile(source_path, live)
    live_before = live.read_bytes()
    target = completed.completed_snapshot.metadata
    boundary.value = record(
        "PREPARED",
        revision=boundary.value["authority_revision"] + 1,
        committed_generation=backup.local_protected_freshness_generation,
        committed_state_fingerprint_sha256=backup.state_fingerprint_sha256,
        prepared_generation=target.protected_freshness_generation,
        prepared_state_fingerprint_sha256=target.state_fingerprint_sha256,
        prepared_transaction_fingerprint_sha256=target.transaction_fingerprint_sha256,
    )
    before_ref, before_record = boundary.ref, dict(boundary.value)
    boundary.calls.clear()

    with pytest.raises(RestoreMigrationInstallError, match="STAGING_LOST"):
        _coordinator(backup, live, boundary).install(backup)

    assert "abort" not in boundary.calls and "finalize" not in boundary.calls
    assert boundary.ref is before_ref and boundary.value == before_record
    assert live.read_bytes() == live_before
    assert manifest.exists()


def test_exact_c7_prepared_recovers_only_after_local_target_proof(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)
    _, manifest = _paths(backup, live)
    target = completed.completed_snapshot.metadata
    completed_descriptor = completed.completed_snapshot.transaction_descriptors[-1]
    SQLiteStateStore.atomic_replace(completed.sqlite_path, live)
    boundary.value = record(
        "PREPARED",
        revision=boundary.value["authority_revision"] + 1,
        committed_generation=target.protected_freshness_generation - 1,
        committed_state_fingerprint_sha256=completed_descriptor.pre_state_fingerprint_sha256,
        prepared_generation=target.protected_freshness_generation,
        prepared_state_fingerprint_sha256=target.state_fingerprint_sha256,
        prepared_transaction_fingerprint_sha256=target.transaction_fingerprint_sha256,
    )
    generation = target.protected_freshness_generation
    boundary.calls.clear()
    coordinator = _coordinator(backup, live, boundary)
    events: list[str] = []
    original_gate = SQLiteStateStore.installation_gate.__func__  # type: ignore[attr-defined]
    original_proof = coordinator._read_and_prove_isolated
    original_recover = coordinator._recover_external_if_prepared
    original_observe = coordinator._observe_secrets

    from contextlib import contextmanager

    @contextmanager
    def gate(cls, path):  # type: ignore[no-untyped-def]
        name = "live" if Path(path).resolve() == live else "staging"
        events.append(f"enter {name}")
        with original_gate(cls, path):
            yield
        events.append(f"exit {name}")

    def proof(*args):  # type: ignore[no-untyped-def]
        events.append("local proof")
        return original_proof(*args)

    def recover(*args):  # type: ignore[no-untyped-def]
        events.append("recover")
        return original_recover(*args)

    def observe(*args):  # type: ignore[no-untyped-def]
        events.append("FINAL_PROMOTION")
        return original_observe(*args)

    monkeypatch.setattr(SQLiteStateStore, "installation_gate", classmethod(gate))
    monkeypatch.setattr(coordinator, "_read_and_prove_isolated", proof)
    monkeypatch.setattr(coordinator, "_recover_external_if_prepared", recover)
    monkeypatch.setattr(coordinator, "_observe_secrets", observe)

    result = coordinator.install(backup)

    assert result.disposition is RestoreMigrationInstallDisposition.ALREADY_INSTALLED
    assert result.installed_snapshot.metadata.protected_freshness_generation == generation
    assert boundary.calls.count("finalize") == 1 and "abort" not in boundary.calls
    assert boundary.value["lifecycle"] == "COMMITTED"
    assert not manifest.exists()
    assert events == [
        "enter live",
        "local proof",
        "recover",
        "local proof",
        "FINAL_PROMOTION",
        "exit live",
        "enter staging",
        "exit staging",
    ]


def test_c7_secret_rejection_preserves_manifest(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)
    _, manifest = _paths(backup, live)
    SQLiteStateStore.atomic_replace(completed.sqlite_path, live)
    coordinator = _coordinator(backup, live, boundary)

    def reject(_snapshot, fence):  # type: ignore[no-untyped-def]
        assert fence.value == "FINAL_PROMOTION"
        raise RestoreMigrationInstallError("C7 secret rejection")

    monkeypatch.setattr(coordinator, "_observe_secrets", reject)
    with pytest.raises(RestoreMigrationInstallError, match="C7 secret rejection"):
        coordinator.install(backup)
    assert manifest.exists()


@pytest.mark.parametrize("path_kind", ["c7", "installed"])
def test_reappeared_staged_path_blocks_cleanup_without_deletion(
    tmp_path: Path, monkeypatch, path_kind: str
) -> None:  # type: ignore[no-untyped-def]
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)
    staged, manifest = _paths(backup, live)
    if path_kind == "c7":
        SQLiteStateStore.atomic_replace(staged, live)
    coordinator = _coordinator(backup, live, boundary)
    original_finish = coordinator._finish_cleanup_expect_staged_absent
    residue: SQLiteStateStore | None = None

    def recreate_before_cleanup(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal residue
        if residue is None:
            residue = SQLiteStateStore(staged)
        return original_finish(*args, **kwargs)

    monkeypatch.setattr(
        coordinator, "_finish_cleanup_expect_staged_absent", recreate_before_cleanup
    )
    with pytest.raises(RestoreMigrationInstallError, match="CLEANUP_PENDING"):
        coordinator.install(backup)
    assert residue is not None and staged.exists() and manifest.exists()
    residue.close()
    staged.unlink()

    result = _coordinator(backup, live, boundary).install(backup)
    assert result.disposition is RestoreMigrationInstallDisposition.ALREADY_INSTALLED
    assert not manifest.exists()


def test_live_exact_replaced_staged_path_is_reproved_before_cleanup(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)
    staged, manifest = _paths(backup, live)
    with SQLiteStateStore(staged) as target:
        target.prepare_for_atomic_install()
    shutil.copyfile(staged, live)
    coordinator = _coordinator(backup, live, boundary)
    original_cleanup = coordinator._finish_cleanup_exact_staged
    replacements = 0
    replacement_bytes = b""
    manifest_bytes = manifest.read_bytes()

    def replace_path_then_cleanup(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal replacement_bytes
        staged.unlink()
        replacement = sqlite3.connect(staged)
        try:
            with replacement:
                replacement.execute("PRAGMA journal_mode=DELETE")
                replacement.execute("CREATE TABLE marker(value TEXT)")
                replacement.execute("INSERT INTO marker VALUES ('foreign')")
        finally:
            replacement.close()
        replacement_bytes = staged.read_bytes()
        return original_cleanup(*args, **kwargs)

    def forbidden(*_args):  # type: ignore[no-untyped-def]
        nonlocal replacements
        replacements += 1

    monkeypatch.setattr(coordinator, "_finish_cleanup_exact_staged", replace_path_then_cleanup)
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", forbidden)
    with pytest.raises(RestoreMigrationInstallError, match="CLEANUP_PENDING"):
        coordinator.install(backup)

    assert replacements == 0
    assert staged.exists() and staged.read_bytes() == replacement_bytes
    assert manifest.exists() and manifest.read_bytes() == manifest_bytes
    assert SQLiteStateStore.read_isolated_verified_snapshot(live) == completed.completed_snapshot

    staged.unlink()
    result = _coordinator(backup, live, boundary).install(backup)
    assert result.disposition is RestoreMigrationInstallDisposition.ALREADY_INSTALLED
    assert not manifest.exists()


def test_live_exact_manifest_change_preserves_verified_staged_target(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)
    staged, manifest = _paths(backup, live)
    with SQLiteStateStore(staged) as target:
        target.prepare_for_atomic_install()
    shutil.copyfile(staged, live)
    staged_bytes = staged.read_bytes()
    coordinator = _coordinator(backup, live, boundary)
    original_cleanup = coordinator._finish_cleanup_exact_staged

    def corrupt_manifest_then_cleanup(*args, **kwargs):  # type: ignore[no-untyped-def]
        manifest.write_bytes(b"changed-manifest")
        return original_cleanup(*args, **kwargs)

    monkeypatch.setattr(coordinator, "_finish_cleanup_exact_staged", corrupt_manifest_then_cleanup)
    with pytest.raises(RestoreMigrationInstallError, match="CLEANUP_PENDING"):
        coordinator.install(backup)
    assert staged.exists() and staged.read_bytes() == staged_bytes
    assert manifest.read_bytes() == b"changed-manifest"


def test_c7_non_target_sqlite_is_byte_exact_and_receives_no_state_store_schema(
    tmp_path: Path,
) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)
    _, manifest = _paths(backup, live)
    completed.sqlite_path.unlink()
    with sqlite3.connect(live) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("CREATE TABLE marker(value TEXT)")
        connection.execute("INSERT INTO marker VALUES ('unchanged')")
    before_bytes = live.read_bytes()
    with sqlite3.connect(f"file:{live}?mode=ro", uri=True) as connection:
        before_schema = connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
        ).fetchall()
        before_marker = connection.execute("SELECT value FROM marker").fetchall()
    before_external = boundary.ref, dict(boundary.value)
    assert not Path(f"{live}-wal").exists() and not Path(f"{live}-shm").exists()

    with pytest.raises(RestoreMigrationInstallError, match="STAGING_LOST"):
        _coordinator(backup, live, boundary).install(backup)

    with sqlite3.connect(f"file:{live}?mode=ro", uri=True) as connection:
        after_schema = connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
        ).fetchall()
        after_marker = connection.execute("SELECT value FROM marker").fetchall()
    assert live.read_bytes() == before_bytes
    assert after_schema == before_schema and after_marker == before_marker
    assert not any(name.startswith("state_store_") for _, name, _, _ in after_schema)
    assert not Path(f"{live}-wal").exists() and not Path(f"{live}-shm").exists()
    assert boundary.ref is before_external[0] and boundary.value == before_external[1]
    assert manifest.exists()


def test_c7_live_handle_blocks_before_prepared_recovery_and_retry_succeeds(
    tmp_path: Path,
) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)
    _, manifest = _paths(backup, live)
    target = completed.completed_snapshot.metadata
    descriptor = completed.completed_snapshot.transaction_descriptors[-1]
    SQLiteStateStore.atomic_replace(completed.sqlite_path, live)
    boundary.value = record(
        "PREPARED",
        revision=boundary.value["authority_revision"] + 1,
        committed_generation=target.protected_freshness_generation - 1,
        committed_state_fingerprint_sha256=descriptor.pre_state_fingerprint_sha256,
        prepared_generation=target.protected_freshness_generation,
        prepared_state_fingerprint_sha256=target.state_fingerprint_sha256,
        prepared_transaction_fingerprint_sha256=target.transaction_fingerprint_sha256,
    )
    before = boundary.ref, dict(boundary.value)
    boundary.calls.clear()
    extra = SQLiteStateStore(live)
    try:
        with pytest.raises(RestoreMigrationInstallError, match="LIVE_PATH_NOT_QUIESCENT"):
            _coordinator(backup, live, boundary).install(backup)
        assert "abort" not in boundary.calls and "finalize" not in boundary.calls
        assert boundary.ref is before[0] and boundary.value == before[1]
        assert manifest.exists()
    finally:
        extra.close()

    result = _coordinator(backup, live, boundary).install(backup)
    assert result.disposition is RestoreMigrationInstallDisposition.ALREADY_INSTALLED
    assert boundary.calls.count("finalize") == 1
