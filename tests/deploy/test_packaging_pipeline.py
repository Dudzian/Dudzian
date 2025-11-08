"""Testy pipeline'u pakowania bundla OEM."""
from __future__ import annotations

import base64
import json
import zipfile
from pathlib import Path

import pytest

from deploy.packaging.pipeline import (
    HardwareFingerprintValidator,
    PackagingContext,
    build_pipeline_from_mapping,
)
from bot_core.security.signing import build_hmac_signature


def _write_fingerprint_document(path: Path, fingerprint: str, key: bytes) -> None:
    payload = {"fingerprint": fingerprint, "generated_at": "2024-06-01T00:00:00Z"}
    document = {"payload": payload, "signature": build_hmac_signature(payload, key=key)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_hardware_fingerprint_validator_reports_missing_document(tmp_path: Path) -> None:
    context = PackagingContext(
        staging_root=tmp_path,
        archive_path=tmp_path / "bundle.zip",
        manifest={"bundle": "core-oem", "version": "1.0.0", "platform": "linux"},
    )

    validator = HardwareFingerprintValidator(fail_on_missing=False)
    report = validator.validate(context)

    assert report.document_path.name == "fingerprint.expected.json"
    assert "fingerprint.expected.json is missing" in report.issues[0]
    assert report.signature_valid is None


def test_pipeline_executes_all_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_manifest = {
        "version": "1.0.0",
        "files": [
            {"path": "daemon/app.bin", "sha384": "old-digest"},
            {"path": "obsolete.txt", "sha384": "deadbeef"},
        ],
    }
    base_manifest_dir = tmp_path / "base-bundle"
    base_manifest_dir.mkdir()
    (base_manifest_dir / "manifest.json").write_text(json.dumps(base_manifest), encoding="utf-8")

    pipeline_config = {
        "fingerprint_validation": {
            "expected": "OEM-FP-001",
            "hmac_key": "env:OEM_PIPELINE_HMAC",
        },
        "delta_updates": {
            "bases": [base_manifest_dir.name],
            "output_dir": "delta",
        },
        "notarization": {
            "bundle_id": "com.example.core",
            "profile": "notary-profile",
            "dry_run": True,
            "log_path": "notary-log.json",
        },
    }

    hmac_key = b"pipeline-secret-key"
    monkeypatch.setenv("OEM_PIPELINE_HMAC", base64.b64encode(hmac_key).decode("ascii"))

    pipeline = build_pipeline_from_mapping(pipeline_config, base_dir=tmp_path)

    staging_root = tmp_path / "staging"
    (staging_root / "daemon").mkdir(parents=True)
    (staging_root / "daemon" / "app.bin").write_bytes(b"new payload")

    fingerprint_path = staging_root / "config" / "fingerprint.expected.json"
    _write_fingerprint_document(fingerprint_path, "OEM-FP-001", hmac_key)

    archive_path = tmp_path / "core-oem-1.1.0-linux.zip"
    archive_path.write_bytes(b"placeholder")

    manifest = {
        "bundle": "core-oem",
        "version": "1.1.0",
        "platform": "linux",
        "files": [
            {"path": "daemon/app.bin", "sha384": "new-digest"},
            {"path": "config/fingerprint.expected.json", "sha384": "fingerprint-digest"},
        ],
    }
    context = PackagingContext(staging_root=staging_root, archive_path=archive_path, manifest=manifest)

    report = pipeline.execute(context)

    assert report.fingerprint is not None
    assert report.fingerprint.issues == []
    assert report.fingerprint.signature_valid is True

    assert report.delta_updates, "Oczekiwano wygenerowania paczki delta"
    delta = report.delta_updates[0]
    assert delta.changed_files == [
        "config/fingerprint.expected.json",
        "daemon/app.bin",
    ]
    assert delta.removed_files == ["obsolete.txt"]
    assert delta.archive_path.exists()

    with zipfile.ZipFile(delta.archive_path) as archive:
        assert "delta.json" in archive.namelist()
        assert "daemon/app.bin" in archive.namelist()

    assert report.notarization is not None
    assert report.notarization.dry_run is True
    assert report.notarization.command[0] == "xcrun"
    assert report.notarization.command[1:3] == ["notarytool", "submit"]

    log_path = tmp_path / "notary-log.json"
    assert log_path.exists()
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "dry-run"

