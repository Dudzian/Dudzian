"""Obsługa wysyłki archiwów smoke testu do bezpiecznego magazynu."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping

from bot_core.config.models import (
    CoreReportingConfig,
    SmokeArchiveLocalConfig,
    SmokeArchiveS3Config,
    SmokeArchiveUploadConfig,
)
from bot_core.reporting._s3 import load_s3_credentials
from bot_core.security import SecretManager, SecretStorageError


@dataclass(slots=True)
class SmokeArchiveUploadResult:
    """Szczegóły udanego przesłania archiwum smoke testu."""

    backend: str
    location: str
    metadata: Mapping[str, str]


class SmokeArchiveUploader:
    """Realizuje wysyłkę artefaktów smoke testu zgodnie z konfiguracją."""

    def __init__(
        self,
        config: SmokeArchiveUploadConfig,
        *,
        secret_manager: SecretManager | None = None,
    ) -> None:
        self._config = config
        self._secret_manager = secret_manager

    def upload(
        self,
        archive_path: Path,
        *,
        environment: str,
        summary_sha256: str,
        window: Mapping[str, str],
    ) -> SmokeArchiveUploadResult:
        if not archive_path.exists():
            raise FileNotFoundError(archive_path)

        backend = self._config.backend.lower()
        archive_sha256 = self._hash_file(archive_path)
        timestamp = datetime.now(timezone.utc)
        placeholders = self._build_placeholders(
            environment=environment,
            summary_sha256=summary_sha256,
            window=window,
            timestamp=timestamp,
        )

        metadata: MutableMapping[str, str] = {
            "timestamp": timestamp.isoformat(),
            "summary_sha256": summary_sha256,
            "archive_sha256": archive_sha256,
        }

        if backend == "local":
            if self._config.local is None:
                raise ValueError("Brak konfiguracji local dla backendu 'local'")
            destination, verification = self._upload_local(
                archive_path,
                local_cfg=self._config.local,
                placeholders=placeholders,
                expected_hash=archive_sha256,
            )
            metadata.update(verification)
            return SmokeArchiveUploadResult(
                backend="local",
                location=str(destination),
                metadata=metadata,
            )

        if backend == "s3":
            if self._config.s3 is None:
                raise ValueError("Brak konfiguracji s3 dla backendu 's3'")
            location, verification = self._upload_s3(
                archive_path,
                s3_cfg=self._config.s3,
                placeholders=placeholders,
                expected_hash=archive_sha256,
            )
            metadata.update(verification)
            return SmokeArchiveUploadResult(
                backend="s3",
                location=location,
                metadata=metadata,
            )

        raise ValueError(f"Nieobsługiwany backend wysyłki archiwum: {self._config.backend}")

    @staticmethod
    def resolve_config(reporting: CoreReportingConfig | object | None) -> SmokeArchiveUploadConfig | None:
        """Zwraca konfigurację uploadu, jeśli została zdefiniowana w configu."""

        if reporting is None:
            return None
        return getattr(reporting, "smoke_archive_upload", None)

    @staticmethod
    def _build_placeholders(
        *,
        environment: str,
        summary_sha256: str,
        window: Mapping[str, str],
        timestamp: datetime,
    ) -> MutableMapping[str, str]:
        values: MutableMapping[str, str] = {
            "environment": environment,
            "hash": summary_sha256,
            "timestamp": timestamp.strftime("%Y%m%dT%H%M%SZ"),
            "date": timestamp.strftime("%Y-%m-%d"),
            "start": str(window.get("start", "")),
            "end": str(window.get("end", "")),
        }
        return values

    def _upload_local(
        self,
        archive_path: Path,
        *,
        local_cfg: SmokeArchiveLocalConfig,
        placeholders: Mapping[str, str],
        expected_hash: str,
    ) -> tuple[Path, Mapping[str, str]]:
        destination_dir = Path(local_cfg.directory)
        destination_dir.mkdir(parents=True, exist_ok=True)
        filename = local_cfg.filename_pattern.format(**placeholders)
        target_path = destination_dir / filename
        shutil.copy2(archive_path, target_path)
        if local_cfg.fsync:
            with target_path.open("rb+") as handle:  # pragma: no cover - zależne od platformy
                handle.flush()
                os.fsync(handle.fileno())
        verification_hash = self._hash_file(target_path)
        if verification_hash != expected_hash:
            raise RuntimeError(
                "Niezgodność sumy kontrolnej przy kopiowaniu archiwum smoke do magazynu lokalnego"
            )
        metadata: MutableMapping[str, str] = {
            "verified": "true",
            "verified_hash": verification_hash,
            "acknowledged": "true",
            "ack_mechanism": "local_copy",
        }
        return target_path, metadata

    def _upload_s3(
        self,
        archive_path: Path,
        *,
        s3_cfg: SmokeArchiveS3Config,
        placeholders: Mapping[str, str],
        expected_hash: str,
    ) -> tuple[str, Mapping[str, str]]:
        credentials = self._load_s3_credentials()

        try:
            import boto3
        except ModuleNotFoundError as exc:  # pragma: no cover - zależne od środowiska
            raise RuntimeError("Backend 's3' wymaga zainstalowanego pakietu boto3") from exc

        object_key = self._build_object_key(s3_cfg, placeholders)
        session = boto3.session.Session(
            aws_access_key_id=credentials.get("access_key_id"),
            aws_secret_access_key=credentials.get("secret_access_key"),
            aws_session_token=credentials.get("session_token"),
            region_name=s3_cfg.region,
        )
        client = session.client(
            "s3",
            endpoint_url=s3_cfg.endpoint_url,
            use_ssl=s3_cfg.use_ssl,
        )
        extra_args = dict(s3_cfg.extra_args)
        metadata = dict(extra_args.get("Metadata") or {})
        existing_hash = metadata.get("sha256")
        if existing_hash and existing_hash.lower() != expected_hash.lower():
            raise RuntimeError(
                "Konfiguracja extra_args zawiera hash niezgodny z obliczoną sumą SHA-256 archiwum"
            )
        metadata["sha256"] = expected_hash
        extra_args["Metadata"] = metadata
        client.upload_file(str(archive_path), s3_cfg.bucket, object_key, ExtraArgs=extra_args)
        head = client.head_object(Bucket=s3_cfg.bucket, Key=object_key)
        remote_metadata = head.get("Metadata", {}) if isinstance(head, Mapping) else {}
        remote_hash = remote_metadata.get("sha256")
        if remote_hash and remote_hash.lower() != expected_hash.lower():
            raise RuntimeError("Hash pliku w S3 nie zgadza się z lokalną sumą SHA-256 archiwum")

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

        location = f"s3://{s3_cfg.bucket}/{object_key}"
        return location, verification

    @staticmethod
    def _build_object_key(
        config: SmokeArchiveS3Config,
        placeholders: Mapping[str, str],
    ) -> str:
        filename = "{environment}_{timestamp}_{hash}.zip".format(**placeholders)
        if config.object_prefix:
            prefix = config.object_prefix.rstrip("/")
            return f"{prefix}/{filename}"
        return filename

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_s3_credentials(self) -> Mapping[str, str]:
        """Zachowuje kompatybilność z dotychczasowymi testami."""

        return load_s3_credentials(self._secret_manager, self._config.credential_secret)


__all__ = [
    "SmokeArchiveUploader",
    "SmokeArchiveUploadResult",
]

