from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.config.models import (
    SmokeArchiveLocalConfig,
    SmokeArchiveS3Config,
    SmokeArchiveUploadConfig,
)
from bot_core.reporting import upload
from bot_core.reporting.upload import SmokeArchiveUploader
from bot_core.security import SecretManager, SecretStorage


class _InMemorySecretStorage(SecretStorage):
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get_secret(self, key: str) -> str | None:  # pragma: no cover - deleguje logikę testu
        return self._data.get(key)

    def set_secret(self, key: str, value: str) -> None:  # pragma: no cover - deleguje logikę testu
        self._data[key] = value

    def delete_secret(self, key: str) -> None:  # pragma: no cover - deleguje logikę testu
        self._data.pop(key, None)


class _FixedDateTime(datetime):
    """Pomocnicza klasa umożliwiająca deterministyczne testy timestampów."""

    _fixed_now = datetime(2024, 1, 2, 12, 34, 56, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        if tz is None:
            return cls._fixed_now.replace(tzinfo=None)
        return cls._fixed_now.astimezone(tz)


def _patch_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upload, "datetime", _FixedDateTime)


def test_smoke_archive_upload_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_datetime(monkeypatch)
    archive_path = tmp_path / "report.zip"
    archive_path.write_bytes(b"dummy-data")

    config = SmokeArchiveUploadConfig(
        backend="local",
        local=SmokeArchiveLocalConfig(
            directory=str(tmp_path / "dest"),
            filename_pattern="{environment}_{date}_{hash}.zip",
            fsync=False,
        ),
    )

    uploader = SmokeArchiveUploader(config)
    result = uploader.upload(
        archive_path,
        environment="binance_paper",
        summary_sha256="abc123",
        window={"start": "2024-01-01", "end": "2024-02-01"},
    )

    assert result.backend == "local"
    assert result.metadata["timestamp"].startswith("2024-01-02T12:34:56")

    expected_name = "binance_paper_2024-01-02_abc123.zip"
    destination = Path(result.location)
    assert destination.name == expected_name
    assert destination.exists()
    assert destination.read_bytes() == b"dummy-data"


def test_smoke_archive_upload_s3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_datetime(monkeypatch)

    archive_path = tmp_path / "report.zip"
    archive_path.write_bytes(b"dummy-data")

    storage = _InMemorySecretStorage()
    secret_manager = SecretManager(storage, namespace="tests")
    secret_manager.store_secret_value(
        "s3_creds",
        json.dumps({"access_key_id": "abc", "secret_access_key": "def"}),
    )

    uploads: list[tuple[str, str, str, dict[str, str]]] = []

    class _DummyS3Client:
        def upload_file(self, filename: str, bucket: str, object_key: str, ExtraArgs=None):  # noqa: N803
            uploads.append((filename, bucket, object_key, ExtraArgs or {}))

    class _DummySession:
        def __init__(self, **kwargs) -> None:  # noqa: D401
            self.kwargs = kwargs

        def client(self, service_name: str, **kwargs):
            assert service_name == "s3"
            self.client_kwargs = kwargs
            return _DummyS3Client()

    dummy_module = types.SimpleNamespace(session=types.SimpleNamespace(Session=_DummySession))
    monkeypatch.setitem(sys.modules, "boto3", dummy_module)

    config = SmokeArchiveUploadConfig(
        backend="s3",
        credential_secret="s3_creds",
        s3=SmokeArchiveS3Config(
            bucket="smoke-bucket",
            object_prefix="reports",
            endpoint_url="https://example.com",
            region="eu-central-1",
            use_ssl=True,
            extra_args={"ACL": "private"},
        ),
    )

    uploader = SmokeArchiveUploader(config, secret_manager=secret_manager)
    result = uploader.upload(
        archive_path,
        environment="binance_paper",
        summary_sha256="abc123",
        window={"start": "2024-01-01", "end": "2024-02-01"},
    )

    assert result.backend == "s3"
    assert result.location == "s3://smoke-bucket/reports/binance_paper_20240102T123456Z_abc123.zip"
    assert result.metadata["timestamp"].startswith("2024-01-02T12:34:56")

    assert uploads == [
        (
            str(archive_path),
            "smoke-bucket",
            "reports/binance_paper_20240102T123456Z_abc123.zip",
            {"ACL": "private"},
        )
    ]
