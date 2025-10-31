from __future__ import annotations

import json
import tarfile
from pathlib import Path

from bot_core.security.update import verify_update_bundle
from scripts.build.desktop_distribution import build_distribution
from scripts.build.prepare_update_package import create_update_package


def _extract_archive(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(destination)
    return next(destination.iterdir())


def test_prepare_update_package(tmp_path: Path) -> None:
    runtime_base = tmp_path / "runtime_base"
    runtime_new = tmp_path / "runtime_new"
    ui_base = tmp_path / "ui_base"
    ui_new = tmp_path / "ui_new"
    runtime_base.mkdir()
    runtime_new.mkdir()
    ui_base.mkdir()
    ui_new.mkdir()

    (runtime_base / "bin").mkdir()
    (runtime_new / "bin").mkdir()
    (runtime_base / "bin" / "start.sh").write_text("#!/bin/sh\necho v1", encoding="utf-8")
    (runtime_new / "bin" / "start.sh").write_text("#!/bin/sh\necho v2", encoding="utf-8")
    (runtime_base / "data.txt").write_text("base", encoding="utf-8")
    (runtime_new / "data.txt").write_text("updated", encoding="utf-8")

    (ui_base / "Main.qml").write_text("import QtQuick 2.15\nText { text: 'v1' }", encoding="utf-8")
    (ui_new / "Main.qml").write_text("import QtQuick 2.15\nText { text: 'v2' }", encoding="utf-8")

    license_json = tmp_path / "license.json"
    license_payload = {
        "license_id": "TEST-LIC-002",
        "profile": "desktop.pro",
        "issuer": "unit-test",
        "schema": "core.oem.license",
        "schema_version": 1,
        "issued_at": "2024-01-01T00:00:00Z",
        "expires_at": "2026-01-01T00:00:00Z",
    }
    license_json.write_text(json.dumps(license_payload, ensure_ascii=False), encoding="utf-8")

    base_archive = build_distribution(
        version="1.0.0",
        platform="linux",
        runtime_dir=runtime_base,
        ui_dir=ui_base,
        includes=[],
        license_json=license_json,
        license_fingerprint="FP-123",
        output_dir=tmp_path,
    )
    new_archive = build_distribution(
        version="1.1.0",
        platform="linux",
        runtime_dir=runtime_new,
        ui_dir=ui_new,
        includes=[],
        license_json=license_json,
        license_fingerprint="FP-123",
        output_dir=tmp_path,
    )

    base_install = _extract_archive(base_archive, tmp_path / "install_base")
    new_install = _extract_archive(new_archive, tmp_path / "install_new")

    package_root = create_update_package(
        base_dir=base_install,
        target_dir=new_install,
        output_dir=tmp_path / "updates",
        package_id="bot-suite",
        version="1.1.0",
        platform="linux",
        runtime="desktop",
        base_id="bot-suite-1.0.0",
        signing_key="update=hex:31323334",
        metadata={"notes": "regression"},
    )

    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == "1.1.0"
    assert manifest["metadata"]["notes"] == "regression"
    assert manifest["artifacts"], "Manifest powinien zawierać artefakty"

    verification = verify_update_bundle(
        manifest_path=manifest_path,
        base_dir=package_root,
        signature_path=None,
        hmac_key=bytes.fromhex("31323334"),
        license_result=None,
    )
    assert verification.is_successful
