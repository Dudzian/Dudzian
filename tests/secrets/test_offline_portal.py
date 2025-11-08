"""Testy portalu offline do zarządzania licencjami."""
from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

from bot_core.security.license_store import LicenseStore
from bot_core.security.signing import build_hmac_signature

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "secrets" / "licensing" / "offline_portal.py"
spec = importlib.util.spec_from_file_location("offline_portal", MODULE_PATH)
offline_portal = importlib.util.module_from_spec(spec)
sys.modules.setdefault("offline_portal", offline_portal)
assert spec and spec.loader
spec.loader.exec_module(offline_portal)  # type: ignore[union-attr]


def _create_license_store(path: Path, fingerprint: str) -> dict[str, object]:
    store = LicenseStore(path=path, fingerprint_override=fingerprint)
    payload = {
        "licenses": {
            "LIC-001": {
                "status": "active",
                "fingerprint": fingerprint,
                "issues": [],
            }
        }
    }
    document = store.save(payload)
    return {"payload": payload, "document": document}


def test_status_outputs_summary(tmp_path, capsys) -> None:
    store_path = tmp_path / "store.json"
    fingerprint = "OEM-FP-STATUS"
    _create_license_store(store_path, fingerprint)

    exit_code = offline_portal.main(
        [
            "status",
            "--store",
            str(store_path),
            "--fingerprint",
            fingerprint,
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["fingerprint"] == fingerprint
    assert summary["store_path"] == str(store_path)
    assert summary["licenses"]


def test_verify_reports_signature_and_membership(tmp_path, capsys) -> None:
    store_path = tmp_path / "store.json"
    fingerprint = "OEM-FP-VERIFY"
    _create_license_store(store_path, fingerprint)
    key = b"offline-secret-key"
    signed_license = {
        "payload": {
            "license_id": "LIC-001",
            "fingerprint": fingerprint,
        },
        "signature": build_hmac_signature(
            {"license_id": "LIC-001", "fingerprint": fingerprint}, key=key
        ),
    }
    license_path = tmp_path / "license.json"
    license_path.write_text(json.dumps(signed_license), encoding="utf-8")

    exit_code = offline_portal.main(
        [
            "verify",
            "--store",
            str(store_path),
            "--fingerprint",
            fingerprint,
            "--license",
            str(license_path),
            "--hmac-key",
            "base64:" + base64.b64encode(key).decode("ascii"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    report = json.loads(captured.out)
    assert report["store_contains"] is True
    assert not report["issues"]
    assert report["signature"]["valid"] is True
    assert report["signature"]["errors"] == []


def test_recover_reencrypts_store(tmp_path, capsys) -> None:
    store_path = tmp_path / "store.json"
    old_fp = "OEM-FP-OLD"
    new_fp = "OEM-FP-NEW"
    _create_license_store(store_path, old_fp)
    output_path = tmp_path / "store-new.json"
    report_path = tmp_path / "report.json"

    exit_code = offline_portal.main(
        [
            "recover",
            "--store",
            str(store_path),
            "--old-fingerprint",
            old_fp,
            "--new-fingerprint",
            new_fp,
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    summary = json.loads(captured.out)
    assert Path(summary["output_path"]).exists()
    assert report_path.exists()

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["old_fingerprint"] == old_fp
    assert report_payload["new_fingerprint"] == new_fp

    reloaded = LicenseStore(path=output_path, fingerprint_override=new_fp).load()
    assert reloaded.data["licenses"]
