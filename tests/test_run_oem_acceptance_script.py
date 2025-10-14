from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_oem_acceptance import main as run_acceptance


def _write_signing_key(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(os.urandom(48))
    if os.name != "nt":
        path.chmod(0o600)


def _write_core_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "risk_profiles": {
            "conservative": {
                "max_daily_loss_pct": 0.02,
                "max_position_pct": 0.05,
                "target_volatility": 0.1,
                "max_leverage": 3.0,
                "stop_loss_atr_multiple": 1.0,
                "max_open_positions": 3,
                "hard_drawdown_pct": 0.1,
                "data_quality": {"max_gap_minutes": 1440, "min_ok_ratio": 0.9},
                "strategy_allocations": {},
                "instrument_buckets": ["core"],
            }
        },
        "environments": {
            "paper": {
                "exchange": "binance",
                "environment": "paper",
                "keychain_key": "paper",
                "data_cache_path": str(path.parent / "data"),
                "risk_profile": "conservative",
                "alert_channels": [],
            }
        },
    }
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def test_run_oem_acceptance_end_to_end(tmp_path: Path) -> None:
    daemon_artifact = tmp_path / "daemon" / "botd"
    daemon_artifact.parent.mkdir(parents=True, exist_ok=True)
    daemon_artifact.write_text("binary", encoding="utf-8")

    ui_artifact = tmp_path / "ui" / "app"
    ui_artifact.parent.mkdir(parents=True, exist_ok=True)
    ui_artifact.write_text("qt", encoding="utf-8")

    core_config = tmp_path / "config" / "core.yaml"
    _write_core_config(core_config)

    bundle_key = tmp_path / "keys" / "bundle.key"
    license_key = tmp_path / "keys" / "license.key"
    decision_key = tmp_path / "keys" / "decision.key"

    _write_signing_key(bundle_key)
    _write_signing_key(license_key)
    _write_signing_key(decision_key)

    summary_path = tmp_path / "summary" / "acceptance.json"
    artifact_root = tmp_path / "artifacts"
    bundle_output = tmp_path / "dist"
    license_registry = tmp_path / "licenses" / "registry.jsonl"
    risk_output = tmp_path / "reports"
    mtls_output = tmp_path / "mtls"

    decision_log_path = tmp_path / "audit" / "decision_log.jsonl"

    exit_code = run_acceptance(
        [
            "--bundle-platform",
            "linux",
            "--bundle-version",
            "1.2.3",
            "--bundle-signing-key",
            str(bundle_key),
            "--bundle-daemon",
            str(daemon_artifact),
            "--bundle-ui",
            str(ui_artifact),
            "--bundle-config",
            f"core.yaml={core_config}",
            "--bundle-output-dir",
            str(bundle_output),
            "--bundle-fingerprint-placeholder",
            "PLACEHOLDER-FP",
            "--license-signing-key",
            str(license_key),
            "--license-fingerprint",
            "ABCDEF123456",
            "--license-registry",
            str(license_registry),
            "--license-bundle-version",
            "1.2.3",
            "--license-valid-days",
            "30",
            "--license-feature",
            "paper",
            "--risk-config",
            str(core_config),
            "--risk-environment",
            "paper",
            "--risk-output-dir",
            str(risk_output),
            "--risk-json-name",
            "report.json",
            "--risk-pdf-name",
            "report.pdf",
            "--mtls-output-dir",
            str(mtls_output),
            "--mtls-bundle-name",
            "core-oem",
            "--summary-path",
            str(summary_path),
            "--artifact-root",
            str(artifact_root),
            "--decision-log-path",
            str(decision_log_path),
            "--decision-log-hmac-key-file",
            str(decision_key),
            "--decision-log-key-id",
            "oem-1",
            "--decision-log-category",
            "release.oem.acceptance",
            "--decision-log-notes",
            "Dry-run release 2024-Phase2",
        ]
    )

    assert exit_code == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert {entry["step"] for entry in summary} == {"bundle", "license", "risk", "mtls"}
    assert all(entry["status"] == "ok" for entry in summary)

    bundle_details = next(item for item in summary if item["step"] == "bundle")
    assert Path(bundle_details["details"]["archive"]).exists()

    license_details = next(item for item in summary if item["step"] == "license")
    registry_path = Path(license_details["details"]["registry"])
    assert registry_path.exists()
    registry_line = registry_path.read_text(encoding="utf-8").strip()
    assert "ABCDEF123456" in registry_line

    risk_details = next(item for item in summary if item["step"] == "risk")
    risk_json = Path(risk_details["details"]["json_report"])
    risk_pdf = Path(risk_details["details"]["pdf_report"])
    assert risk_json.exists()
    assert risk_pdf.exists()

    mtls_details = next(item for item in summary if item["step"] == "mtls")
    metadata_path = Path(mtls_details["details"]["metadata"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["bundle"] == "core-oem"
    assert Path(mtls_details["details"]["ca_certificate"]).exists()
    assert Path(mtls_details["details"]["server_certificate"]).exists()
    assert Path(mtls_details["details"]["client_certificate"]).exists()

    decision_log_lines = decision_log_path.read_text(encoding="utf-8").splitlines()
    assert len(decision_log_lines) == 1
    decision_entry = json.loads(decision_log_lines[0])
    assert decision_entry["status"] == "ok"
    assert decision_entry["category"] == "release.oem.acceptance"
    assert decision_entry["context"]["bundle_version"] == "1.2.3"
    assert decision_entry["notes"] == "Dry-run release 2024-Phase2"

    signature = decision_entry.get("signature")
    assert signature is not None
    assert signature["algorithm"] == "HMAC-SHA256"
    assert signature["key_id"] == "oem-1"

    key_bytes = decision_key.read_bytes().strip()
    entry_copy = dict(decision_entry)
    entry_copy.pop("signature", None)
    canonical = json.dumps(entry_copy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = base64.b64encode(hmac.new(key_bytes, canonical, hashlib.sha256).digest()).decode("ascii")
    assert signature["value"] == expected

    artifact_runs = list(artifact_root.iterdir())
    assert len(artifact_runs) == 1
    acceptance_dir = artifact_runs[0]
    metadata_path = acceptance_dir / "metadata.json"
    summary_artifact = acceptance_dir / "summary.json"
    assert metadata_path.exists()
    assert summary_artifact.exists()

    copied_summary = json.loads(summary_artifact.read_text(encoding="utf-8"))
    assert {entry["step"] for entry in copied_summary} == {"bundle", "license", "risk", "mtls"}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["exit_code"] == 0
    assert metadata["bundle"]["archive"].endswith(Path(bundle_details["details"]["archive"]).name)

    bundle_dir = acceptance_dir / "bundle"
    assert (bundle_dir / Path(bundle_details["details"]["archive"]).name).exists()
    manifest_copy = bundle_dir / "manifest.json"
    assert manifest_copy.exists()
    signatures = list((bundle_dir / "signatures").rglob("*.sig"))
    assert signatures

    license_dir = acceptance_dir / "license"
    assert (license_dir / Path(license_details["details"]["registry"]).name).exists()

    risk_dir = acceptance_dir / "paper_labs"
    assert (risk_dir / Path(risk_details["details"]["json_report"]).name).exists()
    assert (risk_dir / Path(risk_details["details"]["pdf_report"]).name).exists()

    mtls_dir = acceptance_dir / "mtls"
    for key, value in mtls_details["details"].items():
        assert (mtls_dir / Path(value).name).exists()

    decision_dir = acceptance_dir / "decision_log"
    assert (decision_dir / "entry.json").exists()
    assert (decision_dir / decision_log_path.name).exists()
