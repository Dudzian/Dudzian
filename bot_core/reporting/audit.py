"""Obsługa synchronizacji dziennika JSONL smoke testów do magazynu audytowego."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping

from bot_core.config.models import (
    CoreReportingConfig,
    PaperSmokeJsonSyncConfig,
    PaperSmokeJsonSyncLocalConfig,
    PaperSmokeJsonSyncS3Config,
)
from bot_core.reporting._s3 import load_s3_credentials
from bot_core.security import SecretManager, SecretStorageError


@dataclass(slots=True)
class PaperSmokeJsonSyncResult:
    """Informacje o udanej synchronizacji dziennika JSONL."""

    backend: str
    location: str
    metadata: Mapping[str, str]


class PaperSmokeJsonSynchronizer:
    """Realizuje synchronizację dziennika JSONL smoke testów."""

    def __init__(
        self,
        config: PaperSmokeJsonSyncConfig,
        *,
        secret_manager: SecretManager | None = None,
    ) -> None:
        self._config = config
        self._secret_manager = secret_manager

    def sync(
        self,
        json_log_path: Path,
        *,
        environment: str,
        record_id: str,
        timestamp: datetime,
    ) -> PaperSmokeJsonSyncResult:
        if not json_log_path.exists():
            raise FileNotFoundError(json_log_path)

        backend = self._config.backend.lower()
        timestamp = timestamp.astimezone(timezone.utc)

        contents = json_log_path.read_bytes()
        log_sha256 = self._hash_bytes(contents)
        placeholders = self._build_placeholders(
            environment=environment,
            record_id=record_id or "unknown",
            timestamp=timestamp,
            log_sha256=log_sha256,
        )

        if backend == "local":
            if self._config.local is None:
                raise ValueError("Brak konfiguracji local dla backendu 'local'")
            destination, verification = self._sync_local(
                contents,
                config=self._config.local,
                placeholders=placeholders,
                expected_hash=log_sha256,
            )
            metadata: MutableMapping[str, str] = {
                "timestamp": timestamp.isoformat(),
                "log_sha256": log_sha256,
            }
            metadata.update(verification)
            return PaperSmokeJsonSyncResult(
                backend="local",
                location=str(destination),
                metadata=metadata,
            )

        if backend == "s3":
            if self._config.s3 is None:
                raise ValueError("Brak konfiguracji s3 dla backendu 's3'")
            location, verification = self._sync_s3(
                contents,
                config=self._config.s3,
                placeholders=placeholders,
                expected_hash=log_sha256,
            )
            metadata = {
                "timestamp": timestamp.isoformat(),
                "log_sha256": log_sha256,
            }
            metadata.update(verification)
            return PaperSmokeJsonSyncResult(
                backend="s3",
                location=location,
                metadata=metadata,
            )

        raise ValueError(f"Nieobsługiwany backend synchronizacji dziennika: {self._config.backend}")

    @staticmethod
    def resolve_config(
        reporting: CoreReportingConfig | object | None,
    ) -> PaperSmokeJsonSyncConfig | None:
        if reporting is None:
            return None
        return getattr(reporting, "paper_smoke_json_sync", None)

    @staticmethod
    def _hash_bytes(payload: bytes) -> str:
        import hashlib

        digest = hashlib.sha256()
        digest.update(payload)
        return digest.hexdigest()

    @staticmethod
    def _build_placeholders(
        *,
        environment: str,
        record_id: str,
        timestamp: datetime,
        log_sha256: str,
    ) -> MutableMapping[str, str]:
        return {
            "environment": environment,
            "record": record_id,
            "timestamp": timestamp.strftime("%Y%m%dT%H%M%SZ"),
            "date": timestamp.strftime("%Y-%m-%d"),
            "log_hash": log_sha256,
        }

    def _sync_local(
        self,
        payload: bytes,
        *,
        config: PaperSmokeJsonSyncLocalConfig,
        placeholders: Mapping[str, str],
        expected_hash: str,
    ) -> tuple[Path, Mapping[str, str]]:
        destination_dir = Path(config.directory)
        destination_dir.mkdir(parents=True, exist_ok=True)
        filename = config.filename_pattern.format(**placeholders)
        target_path = destination_dir / filename
        with target_path.open("wb") as handle:
            handle.write(payload)
            if config.fsync:
                handle.flush()
                os.fsync(handle.fileno())
        verification_hash = self._hash_bytes(target_path.read_bytes())
        if verification_hash != expected_hash:
            raise RuntimeError(
                "Niezgodność sumy kontrolnej przy synchronizacji lokalnego dziennika smoke"
            )
        metadata: MutableMapping[str, str] = {
            "verified": "true",
            "verified_hash": verification_hash,
            "acknowledged": "true",
            "ack_mechanism": "local_copy",
        }
        return target_path, metadata

    def _sync_s3(
        self,
        payload: bytes,
        *,
        config: PaperSmokeJsonSyncS3Config,
        placeholders: Mapping[str, str],
        expected_hash: str,
    ) -> tuple[str, Mapping[str, str]]:
        credentials = self._load_s3_credentials()

        try:
            import boto3
        except ModuleNotFoundError as exc:  # pragma: no cover - zależne od środowiska
            raise RuntimeError("Backend 's3' wymaga zainstalowanego pakietu boto3") from exc

        object_key = self._build_object_key(config, placeholders)
        session = boto3.session.Session(
            aws_access_key_id=credentials.get("access_key_id"),
            aws_secret_access_key=credentials.get("secret_access_key"),
            aws_session_token=credentials.get("session_token"),
            region_name=config.region,
        )
        client = session.client(
            "s3",
            endpoint_url=config.endpoint_url,
            use_ssl=config.use_ssl,
        )
        extra_args = dict(config.extra_args)
        metadata = dict(extra_args.get("Metadata") or {})
        metadata.setdefault("sha256", expected_hash)
        extra_args["Metadata"] = metadata
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:  # pragma: no cover - zależne od środowiska
            temp_file.write(payload)
            temp_file.flush()
            upload_path = Path(temp_file.name)
        try:
            client.upload_file(str(upload_path), config.bucket, object_key, ExtraArgs=extra_args)
        finally:
            try:
                upload_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
        head = client.head_object(Bucket=config.bucket, Key=object_key)
        remote_metadata = head.get("Metadata", {}) if isinstance(head, Mapping) else {}
        remote_hash = remote_metadata.get("sha256")
        if remote_hash and remote_hash.lower() != expected_hash.lower():
            raise RuntimeError("Hash pliku w S3 nie zgadza się z lokalną sumą SHA-256")

        verification: MutableMapping[str, str] = {
            "verified": "true",
            "acknowledged": "true",
        }
        if remote_hash:
            verification["remote_sha256"] = remote_hash
        version_id = head.get("VersionId") if isinstance(head, Mapping) else None
        if version_id:
            verification["version_id"] = str(version_id)
        response_meta = head.get("ResponseMetadata") if isinstance(head, Mapping) else None
        if isinstance(response_meta, Mapping):
            status_code = response_meta.get("HTTPStatusCode")
            request_id = response_meta.get("RequestId") or response_meta.get("HostId")
            if status_code is not None:
                verification["ack_http_status"] = str(status_code)
            if request_id:
                verification["ack_request_id"] = str(request_id)
        return f"s3://{config.bucket}/{object_key}", verification

    def _load_s3_credentials(self) -> Mapping[str, str]:
        """Utrzymuje zgodność z poprzednimi testami jednostkowymi."""

        return load_s3_credentials(self._secret_manager, self._config.credential_secret)

    @staticmethod
    def _build_object_key(
        config: PaperSmokeJsonSyncS3Config,
        placeholders: Mapping[str, str],
    ) -> str:
        filename = "{environment}_{timestamp}_{record}_{log_hash}.jsonl".format(**placeholders)
        if config.object_prefix:
            prefix = config.object_prefix.rstrip("/")
            return f"{prefix}/{filename}"
        return filename


__all__ = [
    "PaperSmokeJsonSynchronizer",
    "PaperSmokeJsonSyncResult",
]

