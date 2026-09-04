from __future__ import annotations

from pathlib import Path

from bot_core.persistence.backup_authentication import (
    BackupArtifactAuthenticationAuthority,
    BackupArtifactAuthenticationScope,
    BackupArtifactAuthenticator,
    BackupArtifactVerifier,
    SQLiteBackupAuthenticationMetadataStore,
    SecretStorageBackupAuthenticationSecureCustody,
)
from bot_core.persistence.physical_backup import PhysicalBackupCreator
from bot_core.security.base import SecretStorage
from tests.persistence.test_backup_envelope import _store


class MemoryStorage(SecretStorage):
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get_secret(self, key: str) -> str | None:
        return self.values.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self.values[key] = value

    def delete_secret(self, key: str) -> None:
        self.values.pop(key, None)


def artifact_fixture(tmp_path: Path):
    store = _store(tmp_path / "live.sqlite")
    metadata = store.read_metadata()
    assert metadata is not None
    scope = BackupArtifactAuthenticationScope(
        metadata.account_id, metadata.device_installation_id, metadata.environment
    )
    authority = BackupArtifactAuthenticationAuthority(
        SQLiteBackupAuthenticationMetadataStore(tmp_path / "authority.sqlite"),
        SecretStorageBackupAuthenticationSecureCustody(MemoryStorage()),
    )
    authority.provision(scope)
    artifact = PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
        store, tmp_path / "backup.sqlite"
    )
    return store, authority, artifact, BackupArtifactVerifier(authority)


def scope_for(store):
    metadata = store.read_metadata()
    assert metadata is not None
    return BackupArtifactAuthenticationScope(
        metadata.account_id, metadata.device_installation_id, metadata.environment
    )


def authority_for(tmp_path: Path, store):
    authority = BackupArtifactAuthenticationAuthority(
        SQLiteBackupAuthenticationMetadataStore(tmp_path / "authority.sqlite"),
        SecretStorageBackupAuthenticationSecureCustody(MemoryStorage()),
    )
    authority.provision(scope_for(store))
    return authority
