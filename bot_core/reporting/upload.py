"""Obsługa wysyłki archiwów smoke testu do bezpiecznego magazynu."""
from __future__ import annotations

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
        timestamp = datetime.now(timezone.utc)
        placeholders = self._build_placeholders(
            environment=environment,
            summary_sha256=summary_sha256,
            window=window,
            timestamp=timestamp,
        )

        if backend == "local":
            if self._config.local is None:
                raise ValueError("Brak konfiguracji local dla backendu 'local'")
            destination = self._upload_local(
                archive_path,
                local_cfg=self._config.local,
                placeholders=placeholders,
            )
            return SmokeArchiveUploadResult(
                backend="local",
                location=str(destination),
                metadata={"timestamp": timestamp.isoformat()},
            )

        if backend == "s3":
            if self._config.s3 is None:
                raise ValueError("Brak konfiguracji s3 dla backendu 's3'")
            location = self._upload_s3(
                archive_path,
                s3_cfg=self._config.s3,
                placeholders=placeholders,
            )
            return SmokeArchiveUploadResult(
                backend="s3",
                location=location,
                metadata={"timestamp": timestamp.isoformat()},
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

    @staticmethod
    def _upload_local(
        archive_path: Path,
        *,
        local_cfg: SmokeArchiveLocalConfig,
        placeholders: Mapping[str, str],
    ) -> Path:
        destination_dir = Path(local_cfg.directory)
        destination_dir.mkdir(parents=True, exist_ok=True)
        filename = local_cfg.filename_pattern.format(**placeholders)
        target_path = destination_dir / filename
        shutil.copy2(archive_path, target_path)
        if local_cfg.fsync:
            with target_path.open("rb+") as handle:  # pragma: no cover - zależne od platformy
                handle.flush()
                os.fsync(handle.fileno())
        return target_path

    def _upload_s3(
        self,
        archive_path: Path,
        *,
        s3_cfg: SmokeArchiveS3Config,
        placeholders: Mapping[str, str],
    ) -> str:
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
        client.upload_file(str(archive_path), s3_cfg.bucket, object_key, ExtraArgs=extra_args)
        location = f"s3://{s3_cfg.bucket}/{object_key}"
        return location

    def _load_s3_credentials(self) -> Mapping[str, str]:
        if not self._config.credential_secret:
            raise SecretStorageError(
                "Konfiguracja backendu 's3' wymaga podania credential_secret z kluczami dostępowymi"
            )
        if self._secret_manager is None:
            raise SecretStorageError("Brak dostępu do SecretManager przy backendzie 's3'")

        raw_value = self._secret_manager.load_secret_value(self._config.credential_secret)
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise SecretStorageError("Sekret S3 ma nieprawidłowy format JSON") from exc

        expected_keys = {"access_key_id", "secret_access_key"}
        missing = sorted(key for key in expected_keys if key not in payload)
        if missing:
            raise SecretStorageError(
                "Sekret S3 nie zawiera wymaganych pól: " + ", ".join(missing)
            )
        return {str(key): str(value) for key, value in payload.items()}

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


__all__ = [
    "SmokeArchiveUploader",
    "SmokeArchiveUploadResult",
]

