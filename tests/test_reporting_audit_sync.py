from datetime import datetime, timezone
from pathlib import Path
import hashlib
import sys
import types

import pytest

from bot_core.config.models import (
    CoreReportingConfig,
    PaperSmokeJsonSyncConfig,
    PaperSmokeJsonSyncLocalConfig,
    PaperSmokeJsonSyncS3Config,
)
from bot_core.reporting.audit import PaperSmokeJsonSynchronizer


def test_sync_local_creates_copy(tmp_path: Path) -> None:
    json_log = tmp_path / "paper_trading_log.jsonl"
    json_log.write_text("{\"status\": \"ok\"}\n", encoding="utf-8")

    destination_dir = tmp_path / "remote"
    config = PaperSmokeJsonSyncConfig(
        backend="local",
        credential_secret=None,
        local=PaperSmokeJsonSyncLocalConfig(
            directory=str(destination_dir),
            filename_pattern="{environment}_{timestamp}_{log_hash}.jsonl",
            fsync=False,
        ),
        s3=None,
    )

    synchronizer = PaperSmokeJsonSynchronizer(config)
    result = synchronizer.sync(
        json_log,
        environment="binance_paper",
        record_id="J-0001",
        timestamp=datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )

    assert result.backend == "local"
    assert "log_sha256" in result.metadata
    assert result.metadata.get("verified") == "true"
    assert result.metadata.get("verified_hash") == result.metadata.get("log_sha256")
    assert result.metadata.get("acknowledged") == "true"
    assert result.metadata.get("ack_mechanism") == "local_copy"
    destination_path = Path(result.location)
    assert destination_path.exists()
    assert destination_path.read_text(encoding="utf-8") == json_log.read_text(encoding="utf-8")


def test_resolve_config_returns_json_sync() -> None:
    local_cfg = PaperSmokeJsonSyncLocalConfig(directory="/tmp", filename_pattern="foo.jsonl", fsync=False)
    config = CoreReportingConfig(
        daily_report_time_utc=None,
        weekly_report_day=None,
        retention_months=None,
        smoke_archive_upload=None,
        paper_smoke_json_sync=PaperSmokeJsonSyncConfig(
            backend="local",
            credential_secret=None,
            local=local_cfg,
            s3=None,
        ),
    )

    resolved = PaperSmokeJsonSynchronizer.resolve_config(config)
    assert resolved is config.paper_smoke_json_sync


def test_sync_local_detects_hash_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    json_log = tmp_path / "paper_trading_log.jsonl"
    json_log.write_text("{\"status\": \"ok\"}\n", encoding="utf-8")

    destination_dir = tmp_path / "remote"
    config = PaperSmokeJsonSyncConfig(
        backend="local",
        credential_secret=None,
        local=PaperSmokeJsonSyncLocalConfig(
            directory=str(destination_dir),
            filename_pattern="{environment}_{timestamp}_{log_hash}.jsonl",
            fsync=False,
        ),
        s3=None,
    )

    synchronizer = PaperSmokeJsonSynchronizer(config)
    hashes = iter(["hash_a", "hash_b"])
    monkeypatch.setattr(synchronizer, "_hash_bytes", lambda payload: next(hashes))

    with pytest.raises(RuntimeError):
        synchronizer.sync(
            json_log,
            environment="binance_paper",
            record_id="J-0001",
            timestamp=datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        )


def test_sync_s3_provides_version_and_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    json_log = tmp_path / "paper_trading_log.jsonl"
    json_log.write_text("{\"status\": \"ok\"}\n", encoding="utf-8")

    uploads: list[tuple[str, str, str, dict[str, str]]] = []
    head_calls: list[tuple[str, str]] = []
    expected_hash = hashlib.sha256(json_log.read_bytes()).hexdigest()

    class _DummyS3Client:
        def upload_file(self, filename: str, bucket: str, object_key: str, ExtraArgs=None):  # noqa: N803
            uploads.append((filename, bucket, object_key, ExtraArgs or {}))

        def head_object(self, Bucket: str, Key: str):  # noqa: N803
            head_calls.append((Bucket, Key))
            return {
                "Metadata": {"sha256": expected_hash},
                "VersionId": "ver-001",
                "ResponseMetadata": {"HTTPStatusCode": 200, "RequestId": "req-123"},
            }

    class _DummySession:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def client(self, service_name: str, **kwargs):
            assert service_name == "s3"
            return _DummyS3Client()

    dummy_module = types.SimpleNamespace(session=types.SimpleNamespace(Session=_DummySession))
    monkeypatch.setitem(sys.modules, "boto3", dummy_module)

    config = PaperSmokeJsonSyncConfig(
        backend="s3",
        credential_secret="dummy",
        local=None,
        s3=PaperSmokeJsonSyncS3Config(
            bucket="audit-bucket",
            object_prefix="logs",
            endpoint_url="https://example.com",
            region="eu-central-1",
            use_ssl=True,
            extra_args={"ACL": "private"},
        ),
    )

    synchronizer = PaperSmokeJsonSynchronizer(config)
    monkeypatch.setattr(synchronizer, "_load_s3_credentials", lambda: {})

    result = synchronizer.sync(
        json_log,
        environment="binance_paper",
        record_id="J-0009",
        timestamp=datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )

    assert result.backend == "s3"
    assert result.location == (
        f"s3://audit-bucket/logs/binance_paper_20250102T030405Z_J-0009_{expected_hash}.jsonl"
    )
    assert result.metadata.get("verified") == "true"
    assert result.metadata.get("acknowledged") == "true"
    assert result.metadata.get("version_id") == "ver-001"
    assert result.metadata.get("ack_request_id") == "req-123"
    assert result.metadata.get("remote_sha256") == expected_hash
    assert uploads and uploads[0][1:] == (
        "audit-bucket",
        f"logs/binance_paper_20250102T030405Z_J-0009_{expected_hash}.jsonl",
        {"ACL": "private", "Metadata": {"sha256": expected_hash}},
    )
    assert head_calls == [
        ("audit-bucket", f"logs/binance_paper_20250102T030405Z_J-0009_{expected_hash}.jsonl")
    ]
