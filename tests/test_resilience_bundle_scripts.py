from __future__ import annotations

import json
import os
from pathlib import Path


from bot_core.resilience.audit import audit_bundles as audit_bundles_fn
from bot_core.resilience.policy import load_policy
from scripts.audit_resilience_bundles import run as audit_bundles_cli
from scripts.export_resilience_bundle import run as export_bundle
from scripts.verify_resilience_bundle import run as verify_bundle
from bot_core.security.signing import build_hmac_signature


ROOT = Path(__file__).resolve().parents[1]
def _write_file(base: Path, name: str, data: str) -> Path:
    path = base / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return path


def test_export_and_verify_resilience_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_file(source, "drills/failover.json", json.dumps({"status": "ok"}))
    _write_file(source, "reports/tco.csv", "header,value\ncore,42")

    key_path = tmp_path / "keys" / "bundle.key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)
    key_path.write_bytes(key)
    if os.name != "nt":
        key_path.chmod(0o600)

    output_dir = tmp_path / "out"

    exit_code = export_bundle(
        [
            "--source",
            str(source),
            "--output-dir",
            str(output_dir),
            "--bundle-name",
            "stage6-resilience-test",
            "--metadata",
            "run_id=42",
            "--hmac-key-file",
            str(key_path),
            "--hmac-key-id",
            "ops-stage6",
        ]
    )

    assert exit_code == 0

    artifacts = list(output_dir.glob("stage6-resilience-test-*.zip"))
    assert len(artifacts) == 1
    bundle = artifacts[0]
    manifest = bundle.with_suffix(".manifest.json")
    signature = bundle.with_suffix(".manifest.sig")

    assert manifest.exists()
    assert signature.exists()

    verify_code = verify_bundle(
        [
            str(bundle),
            "--manifest",
            str(manifest),
            "--signature",
            str(signature),
            "--hmac-key-file",
            str(key_path),
        ]
    )
    assert verify_code == 0


def test_verify_resilience_bundle_detects_tamper(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_file(source, "ok.txt", "ok")

    export_bundle([
        "--source",
        str(source),
        "--output-dir",
        str(tmp_path / "out"),
        "--bundle-name",
        "stage6-resilience-test",
    ])

    bundle = next((tmp_path / "out").glob("*.zip"))
    manifest = bundle.with_suffix(".manifest.json")

    import zipfile

    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr("extra.txt", "tamper")

    exit_code = verify_bundle([
        str(bundle),
        "--manifest",
        str(manifest),
    ])

    assert exit_code == 2


def test_audit_resilience_bundles_reports(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_file(source, "drills/failover.json", json.dumps({"status": "ok"}))

    key_path = tmp_path / "keys" / "bundle.key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)
    key_path.write_bytes(key)
    if os.name != "nt":
        key_path.chmod(0o600)

    output_dir = tmp_path / "out"

    export_bundle(
        [
            "--source",
            str(source),
            "--output-dir",
            str(output_dir),
            "--bundle-name",
            "stage6-resilience-test",
            "--metadata",
            "drill=primary",
            "--hmac-key-file",
            str(key_path),
        ]
    )

    csv_path = tmp_path / "reports" / "audit.csv"
    json_path = tmp_path / "reports" / "audit.json"
    json_sig_path = tmp_path / "reports" / "audit.json.sig"

    report_key_path = tmp_path / "keys" / "audit.key"
    report_key_path.parent.mkdir(parents=True, exist_ok=True)
    report_key = os.urandom(32)
    report_key_path.write_bytes(report_key)
    if os.name != "nt":
        report_key_path.chmod(0o600)

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "required_patterns": [
                    {
                        "pattern": "drills/*.json",
                        "description": "Raporty z failover drill",
                        "min_matches": 1,
                    }
                ],
                "metadata": [
                    {
                        "key": "drill",
                        "description": "Typ scenariusza DR",
                        "allowed_values": ["primary", "secondary"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = audit_bundles_cli(
        [
            "--directory",
            str(output_dir),
            "--hmac-key-file",
            str(key_path),
            "--csv-report",
            str(csv_path),
            "--json-report",
            str(json_path),
            "--json-report-signature",
            str(json_sig_path),
            "--report-hmac-key-file",
            str(report_key_path),
            "--report-hmac-key-id",
            "stage6-audit",
            "--policy",
            str(policy_path),
        ]
    )

    assert exit_code == 0
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "stage6-resilience-test" in csv_text

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["schema"].startswith("stage6.resilience.audit")
    assert data["audited"] == 1
    assert data["warnings"] == 0
    assert data["results"][0]["warnings"] == []

    signature_doc = json.loads(json_sig_path.read_text(encoding="utf-8"))
    assert signature_doc["target"] == json_path.name
    expected_signature = build_hmac_signature(data, key=report_key, algorithm="HMAC-SHA256", key_id="stage6-audit")
    assert signature_doc["signature"] == expected_signature


def test_audit_resilience_bundles_policy_failure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_file(source, "reports/summary.txt", "brak drill")

    export_bundle(
        [
            "--source",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--bundle-name",
            "stage6-resilience-test",
        ]
    )

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "required_patterns": [
                    {
                        "pattern": "drills/*.json",
                        "description": "Raport failover",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = audit_bundles_cli(
        [
            "--directory",
            str(tmp_path / "out"),
            "--policy",
            str(policy_path),
        ]
    )

    assert exit_code == 2
    results = audit_bundles_fn(Path(tmp_path / "out"), policy=load_policy(policy_path))
    assert not results[0].is_successful()
    assert any("Raport failover" in error for error in results[0].errors)


def test_audit_resilience_bundles_detects_issues(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_file(source, "ok.txt", "ok")

    export_bundle(
        [
            "--source",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--bundle-name",
            "stage6-resilience-test",
        ]
    )

    bundle = next((tmp_path / "out").glob("*.zip"))

    import zipfile

    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr("extra.txt", "tamper")

    json_path = tmp_path / "reports" / "audit.json"
    exit_code = audit_bundles_cli(
        [
            "--directory",
            str(tmp_path / "out"),
            "--json-report",
            str(json_path),
            "--require-signature",
        ]
    )

    assert exit_code == 2
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["failed"] == 1
    assert any("Brak podpisu" in msg for msg in payload["results"][0]["errors"])
