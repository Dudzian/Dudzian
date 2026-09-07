from __future__ import annotations

import json
import sqlite3
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from bot_core.persistence import physical_durability
from bot_core.persistence import restore_migration_staging
from bot_core.persistence.backup_envelope import (
    BackupCandidateDisposition,
    BackupEnvelopeError,
    classify_authenticated_backup,
    create_backup_envelope,
)
from bot_core.persistence.fingerprints import canonical_json, canonical_json_sha256
from bot_core.persistence.physical_backup import (
    PhysicalSQLiteArtifact,
    physical_sqlite_artifact_fingerprint,
)
from bot_core.persistence.restore_migration_staging import (
    RestoreMigrationStagingDisposition,
    RestoreMigrationStagingError,
    RestoreMigrationStagingManifest,
    prepare_legacy_restore_staging,
    restore_migration_staging_id,
    restore_migration_manifest_path,
    restore_migration_staged_sqlite_path,
    restore_migration_unpublished_sqlite_path,
)
from bot_core.persistence.state_store import SQLiteStateStore
from tests.persistence.test_state_store_records import _account, _commit, _metadata, _runtime


def _legacy_backup(path: Path):  # type: ignore[no-untyped-def]
    with SQLiteStateStore(path) as store:
        _commit(
            store,
            _metadata(state_store_schema_version=1),
            current=(_account(),),
            history=(_runtime(),),
        )
        backup = create_backup_envelope(store)
    assert backup is not None
    return backup


def test_source_ready_is_durable_exact_and_idempotent(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    live.write_bytes(b"untouched-live")
    before = live.read_bytes()

    first = prepare_legacy_restore_staging(backup, live_state_store_path=live)
    second = prepare_legacy_restore_staging(backup, live_state_store_path=live)

    assert classify_authenticated_backup(backup) is (
        BackupCandidateDisposition.AUTHENTICATED_LEGACY_CANDIDATE
    )
    assert first.disposition is RestoreMigrationStagingDisposition.SOURCE_READY
    assert second.staging_id == first.staging_id
    assert second.manifest == first.manifest
    assert first.snapshot.metadata.state_store_schema_version == 1
    assert first.snapshot.metadata.state_fingerprint_sha256 == backup.state_fingerprint_sha256
    assert first.sqlite_path.parent.resolve() == live.parent.resolve()
    assert first.manifest_path.parent.resolve() == live.parent.resolve()
    assert not (live.parent / ".cryptohunter-restore-staging").exists()
    assert live.read_bytes() == before
    assert len(fields(RestoreMigrationStagingManifest)) == 18
    with pytest.raises(FrozenInstanceError):
        first.manifest.staging_id = "a" * 64  # type: ignore[misc]


def test_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    artifact = prepare_legacy_restore_staging(
        backup, live_state_store_path=tmp_path / "live.sqlite3"
    )
    raw = json.loads(artifact.manifest_path.read_text())
    raw["source_generation"] = 2
    artifact.manifest_path.write_text(canonical_json(raw))
    with pytest.raises(RestoreMigrationStagingError):
        prepare_legacy_restore_staging(backup, live_state_store_path=tmp_path / "live.sqlite3")


def test_sealed_migration_binding_tamper_fails(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    artifact = prepare_legacy_restore_staging(
        backup, live_state_store_path=tmp_path / "live.sqlite3"
    )
    raw = json.loads(artifact.manifest_path.read_text())
    raw["operation_plan_fingerprint_sha256"] = "a" * 64
    projection = {key: value for key, value in raw.items() if key != "manifest_fingerprint_sha256"}
    raw["manifest_fingerprint_sha256"] = canonical_json_sha256(projection)
    artifact.manifest_path.write_text(canonical_json(raw))
    with pytest.raises(RestoreMigrationStagingError, match="sealed"):
        prepare_legacy_restore_staging(backup, live_state_store_path=tmp_path / "live.sqlite3")


def test_current_backup_is_not_applicable_and_creates_nothing(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "source.sqlite3") as store:
        _commit(
            store,
            _metadata(state_store_schema_version=2),
            current=(_account(),),
            history=(_runtime(),),
        )
        backup = create_backup_envelope(store)
    assert backup is not None
    assert classify_authenticated_backup(backup) is (
        BackupCandidateDisposition.CURRENT_AUTHENTICATED_CANDIDATE
    )
    with pytest.raises(RestoreMigrationStagingError, match="not applicable"):
        prepare_legacy_restore_staging(backup, live_state_store_path=tmp_path / "live.sqlite3")
    assert not (tmp_path / ".cryptohunter-restore-staging").exists()


@pytest.mark.parametrize("residue", ["unpublished", "staged", "manifest-temporary"])
def test_c0_prepublication_crash_residue_is_rebuilt(tmp_path: Path, residue: str) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    staging_id = restore_migration_staging_id(
        state_store_identity_fingerprint_sha256=backup.state_store_identity_fingerprint_sha256,
        source_backup_envelope_fingerprint_sha256=backup.envelope_fingerprint_sha256,
    )
    unpublished = restore_migration_unpublished_sqlite_path(live, staging_id)
    staged = restore_migration_staged_sqlite_path(live, staging_id)
    manifest = restore_migration_manifest_path(live, staging_id)
    if residue == "unpublished":
        unpublished.write_bytes(b"crash debris")
    elif residue == "staged":
        staged.write_bytes(b"crash debris")
    else:
        Path(manifest.parent / f".{manifest.name}.owned-temp").write_bytes(b"partial")

    artifact = prepare_legacy_restore_staging(backup, live_state_store_path=live)
    assert artifact.disposition is RestoreMigrationStagingDisposition.SOURCE_READY
    assert not unpublished.exists()
    assert staged.exists() and manifest.exists()
    assert not Path(manifest.parent / f".{manifest.name}.owned-temp").exists()


def test_present_invalid_manifest_is_preserved_as_publication_barrier(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    staging_id = restore_migration_staging_id(
        state_store_identity_fingerprint_sha256=backup.state_store_identity_fingerprint_sha256,
        source_backup_envelope_fingerprint_sha256=backup.envelope_fingerprint_sha256,
    )
    live = tmp_path / "live.sqlite3"
    manifest = restore_migration_manifest_path(live, staging_id)
    manifest.write_bytes(b"not-json")
    with pytest.raises(RestoreMigrationStagingError, match="STAGING_CONFLICT"):
        prepare_legacy_restore_staging(backup, live_state_store_path=tmp_path / "live.sqlite3")
    assert manifest.read_bytes() == b"not-json"
    assert manifest.exists()


def test_trailing_publication_failure_is_refenced_on_retry_without_rematerializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")

    original_publish = physical_durability.publish_file_atomically_durably

    def fail_after_replace(temporary: Path, target: Path, **_kwargs: object) -> None:
        # Model the exact uncertain point: final pathname exists, but the
        # trailing regular-file/metadata fence did not complete.
        temporary.replace(target)
        raise physical_durability.PhysicalDurabilityError("trailing fence failed")

    monkeypatch.setattr(physical_durability, "publish_file_atomically_durably", fail_after_replace)
    with pytest.raises(physical_durability.PhysicalDurabilityError):
        prepare_legacy_restore_staging(backup, live_state_store_path=tmp_path / "live.sqlite3")
    staging_id = restore_migration_staging_id(
        state_store_identity_fingerprint_sha256=backup.state_store_identity_fingerprint_sha256,
        source_backup_envelope_fingerprint_sha256=backup.envelope_fingerprint_sha256,
    )
    live = tmp_path / "live.sqlite3"
    manifest_path = restore_migration_manifest_path(live, staging_id)
    sqlite_path = restore_migration_staged_sqlite_path(live, staging_id)
    assert manifest_path.exists()
    assert sqlite_path.exists()
    manifest_bytes, sqlite_bytes = manifest_path.read_bytes(), sqlite_path.read_bytes()

    monkeypatch.setattr(physical_durability, "publish_file_atomically_durably", original_publish)
    refences = 0
    original_reinforce = restore_migration_staging.reinforce_published_file_durability

    def reinforce(path: Path, payload: bytes) -> None:
        nonlocal refences
        refences += 1
        original_reinforce(path, payload)

    monkeypatch.setattr(restore_migration_staging, "reinforce_published_file_durability", reinforce)
    result = prepare_legacy_restore_staging(backup, live_state_store_path=tmp_path / "live.sqlite3")
    assert result.disposition is RestoreMigrationStagingDisposition.SOURCE_READY
    assert refences == 1
    assert result.manifest_path.read_bytes() == manifest_bytes
    assert result.sqlite_path.read_bytes() == sqlite_bytes


def test_existing_publication_refence_failure_preserves_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    artifact = prepare_legacy_restore_staging(
        backup, live_state_store_path=tmp_path / "live.sqlite3"
    )
    before = (artifact.manifest_path.read_bytes(), artifact.sqlite_path.read_bytes())

    def fail(_path: Path, _payload: bytes) -> None:
        raise physical_durability.PhysicalDurabilityError("re-fence failed")

    monkeypatch.setattr(restore_migration_staging, "reinforce_published_file_durability", fail)
    for _ in range(2):
        with pytest.raises(physical_durability.PhysicalDurabilityError, match="re-fence"):
            prepare_legacy_restore_staging(backup, live_state_store_path=tmp_path / "live.sqlite3")
        assert (artifact.manifest_path.read_bytes(), artifact.sqlite_path.read_bytes()) == before


def test_source_sqlite_byte_tamper_fails_closed(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    artifact = prepare_legacy_restore_staging(
        backup, live_state_store_path=tmp_path / "live.sqlite3"
    )
    with artifact.sqlite_path.open("ab") as output:
        output.write(b"physical-tamper")
    with pytest.raises(RestoreMigrationStagingError, match="bytes mismatch"):
        prepare_legacy_restore_staging(backup, live_state_store_path=tmp_path / "live.sqlite3")


def test_source_descriptor_anchor_tamper_fails_closed(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    artifact = prepare_legacy_restore_staging(
        backup, live_state_store_path=tmp_path / "live.sqlite3"
    )
    connection = sqlite3.connect(artifact.sqlite_path)
    try:
        connection.execute(
            "UPDATE state_store_transaction_descriptors "
            "SET descriptor_json = replace(descriptor_json, ?, ?)",
            (backup.transaction_fingerprint_sha256, "a" * 64),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(RestoreMigrationStagingError, match="staged SQLite is invalid"):
        prepare_legacy_restore_staging(backup, live_state_store_path=tmp_path / "live.sqlite3")


def test_different_source_backup_uses_distinct_workspace(tmp_path: Path) -> None:
    first = _legacy_backup(tmp_path / "source-a.sqlite3")
    with SQLiteStateStore(tmp_path / "source-b.sqlite3") as store:
        _commit(
            store,
            _metadata(
                state_store_schema_version=1,
                state_store_identity_fingerprint_sha256="5" * 64,
            ),
            current=(_account(),),
            history=(_runtime(),),
        )
        second = create_backup_envelope(store)
    assert second is not None
    first_artifact = prepare_legacy_restore_staging(
        first, live_state_store_path=tmp_path / "live.sqlite3"
    )
    second_artifact = prepare_legacy_restore_staging(
        second, live_state_store_path=tmp_path / "live.sqlite3"
    )
    assert first_artifact.sqlite_path != second_artifact.sqlite_path
    assert first_artifact.manifest.source_backup_envelope_fingerprint_sha256 != (
        second_artifact.manifest.source_backup_envelope_fingerprint_sha256
    )


def test_unknown_backup_version_has_no_staging_side_effect(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    mapping = backup.to_mapping()
    mapping["state_store_schema_version"] = 3
    projection = {
        key: value for key, value in mapping.items() if key != "envelope_fingerprint_sha256"
    }
    mapping["envelope_fingerprint_sha256"] = canonical_json_sha256(projection)
    with pytest.raises(BackupEnvelopeError, match="unsupported state_store_schema_version"):
        prepare_legacy_restore_staging(mapping, live_state_store_path=tmp_path / "live.sqlite3")
    assert not (tmp_path / ".cryptohunter-restore-staging").exists()


def test_paths_are_same_directory_distinct_and_live_basename_collision_safe(tmp_path: Path) -> None:
    staging_id = "a" * 64
    alpha = tmp_path / "alpha.sqlite3"
    beta = tmp_path / "beta.sqlite3"
    alpha_paths = (
        restore_migration_unpublished_sqlite_path(alpha, staging_id),
        restore_migration_staged_sqlite_path(alpha, staging_id),
        restore_migration_manifest_path(alpha, staging_id),
    )
    beta_paths = (
        restore_migration_unpublished_sqlite_path(beta, staging_id),
        restore_migration_staged_sqlite_path(beta, staging_id),
        restore_migration_manifest_path(beta, staging_id),
    )
    assert all(path.parent.resolve() == alpha.parent.resolve() for path in alpha_paths)
    assert len({alpha, *alpha_paths}) == 4
    assert all(alpha.name in path.name for path in alpha_paths)
    assert all(beta.name in path.name for path in beta_paths)
    assert all(alpha_path != beta_path for alpha_path, beta_path in zip(alpha_paths, beta_paths))


def test_initial_materialization_uses_unpublished_then_durable_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    calls: list[tuple[Path, Path]] = []
    original = restore_migration_staging.publish_file_atomically_durably

    def publish(source: Path, target: Path) -> None:
        calls.append((Path(source), Path(target)))
        original(source, target)

    monkeypatch.setattr(restore_migration_staging, "publish_file_atomically_durably", publish)
    artifact = prepare_legacy_restore_staging(backup, live_state_store_path=live)
    unpublished = restore_migration_unpublished_sqlite_path(live, artifact.staging_id)
    assert calls == [(unpublished, artifact.sqlite_path)]
    assert not unpublished.exists()
    assert artifact.manifest.source_staged_sqlite_artifact_fingerprint_sha256 == (
        physical_sqlite_artifact_fingerprint(PhysicalSQLiteArtifact(artifact.sqlite_path))
    )


def test_manifest_with_unpublished_or_missing_staged_fails_closed(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    artifact = prepare_legacy_restore_staging(backup, live_state_store_path=live)
    unpublished = restore_migration_unpublished_sqlite_path(live, artifact.staging_id)
    unpublished.write_bytes(b"preserve")
    with pytest.raises(RestoreMigrationStagingError, match="unpublished"):
        prepare_legacy_restore_staging(backup, live_state_store_path=live)
    assert unpublished.read_bytes() == b"preserve"
    unpublished.unlink()
    artifact.sqlite_path.unlink()
    with pytest.raises(RestoreMigrationStagingError, match="higher-level recovery"):
        prepare_legacy_restore_staging(backup, live_state_store_path=live)
    assert artifact.manifest_path.exists()
