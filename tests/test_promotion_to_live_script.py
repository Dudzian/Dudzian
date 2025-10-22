from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from scripts.promotion_to_live import build_promotion_report


def _write_core_config(path: Path, *, document_root: Path | None = None) -> None:
    base_dir = document_root or path.parent
    checklist_sig = base_dir / "compliance/live/binance/checklist.sig"
    checklist_sig.parent.mkdir(parents=True, exist_ok=True)
    checklist_sig.write_text("checklist-sig", encoding="utf-8")

    kyc_path = base_dir / "compliance/live/binance/kyc_packet.pdf"
    kyc_path.parent.mkdir(parents=True, exist_ok=True)
    kyc_content = b"kyc-packet"
    kyc_path.write_bytes(kyc_content)
    kyc_digest = hashlib.sha256(kyc_content).hexdigest()
    kyc_signature = base_dir / "compliance/live/binance/kyc_packet.sig"
    kyc_signature.write_text("kyc-sig", encoding="utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            f"""
            risk_profiles:
              conservative:
                max_daily_loss_pct: 0.01
                max_position_pct: 0.03
                target_volatility: 0.07
                max_leverage: 2.0
                stop_loss_atr_multiple: 1.0
                max_open_positions: 3
                hard_drawdown_pct: 0.05
            environments:
              binance_live:
                exchange: binance_spot
                environment: live
                keychain_key: binance_live_key
                data_cache_path: ./var/data/binance_live
                risk_profile: conservative
                alert_channels: [telegram:primary]
                alert_throttle:
                  window_seconds: 120
                alert_audit:
                  backend: file
                  directory: audit/alerts
                live_readiness:
                  checklist_id: stage6-binance
                  signed: true
                  signed_by: [compliance, security]
                  signature_path: compliance/live/binance/checklist.sig
                  required_documents: [kyc_packet]
                  documents:
                    - name: kyc_packet
                      path: compliance/live/binance/kyc_packet.pdf
                      sha256: {kyc_digest}
                      signature_path: compliance/live/binance/kyc_packet.sig
                      signed: true
                      signed_by: [compliance]
            runtime_entrypoints:
              auto_trader:
                environment: binance_live
                risk_profile: conservative
                compliance:
                  live_allowed: true
                  signed: true
                  signoffs: [kyc_packet]
                  require_signoff: true
            """
        ),
        encoding="utf-8",
    )


def _write_multi_core_config(path: Path) -> None:
    base_dir = path.parent

    # Dokumenty dla środowiska binance
    checklist_sig = base_dir / "compliance/live/binance/checklist.sig"
    checklist_sig.parent.mkdir(parents=True, exist_ok=True)
    checklist_sig.write_text("checklist-sig", encoding="utf-8")

    kyc_path = base_dir / "compliance/live/binance/kyc_packet.pdf"
    kyc_path.parent.mkdir(parents=True, exist_ok=True)
    kyc_content = b"kyc-packet"
    kyc_path.write_bytes(kyc_content)
    kyc_digest = hashlib.sha256(kyc_content).hexdigest()
    kyc_signature = base_dir / "compliance/live/binance/kyc_packet.sig"
    kyc_signature.write_text("kyc-sig", encoding="utf-8")

    # Ścieżki dla środowiska coinbase (brak dokumentu wywoła blokadę)
    coinbase_dir = base_dir / "compliance/live/coinbase"
    coinbase_dir.mkdir(parents=True, exist_ok=True)
    coinbase_sig = coinbase_dir / "aml_packet.sig"
    coinbase_sig.write_text("aml-sig", encoding="utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            f"""
            risk_profiles:
              conservative:
                max_daily_loss_pct: 0.01
                max_position_pct: 0.03
                target_volatility: 0.07
                max_leverage: 2.0
                stop_loss_atr_multiple: 1.0
                max_open_positions: 3
                hard_drawdown_pct: 0.05
            environments:
              binance_live:
                exchange: binance_spot
                environment: live
                keychain_key: binance_live_key
                data_cache_path: ./var/data/binance_live
                risk_profile: conservative
                alert_channels: [telegram:primary]
                alert_throttle:
                  window_seconds: 120
                alert_audit:
                  backend: file
                  directory: audit/alerts
                live_readiness:
                  checklist_id: stage6-binance
                  signed: true
                  signed_by: [compliance, security]
                  signature_path: compliance/live/binance/checklist.sig
                  required_documents: [kyc_packet]
                  documents:
                    - name: kyc_packet
                      path: compliance/live/binance/kyc_packet.pdf
                      sha256: {kyc_digest}
                      signature_path: compliance/live/binance/kyc_packet.sig
                      signed: true
                      signed_by: [compliance]
              coinbase_live:
                exchange: coinbase_spot
                environment: live
                keychain_key: coinbase_live_key
                data_cache_path: ./var/data/coinbase_live
                risk_profile: conservative
                alert_channels: [telegram:primary]
                live_readiness:
                  checklist_id: stage6-coinbase
                  signed: false
                  signed_by: []
                  required_documents: [aml_packet]
                  documents:
                    - name: aml_packet
                      path: compliance/live/coinbase/aml_packet.pdf
                      signature_path: compliance/live/coinbase/aml_packet.sig
                      signed: false
            runtime_entrypoints:
              auto_trader:
                environment: binance_live
                risk_profile: conservative
                compliance:
                  live_allowed: true
                  signed: true
                  signoffs: [kyc_packet]
                  require_signoff: true
            """
        ),
        encoding="utf-8",
    )


def test_build_promotion_report_produces_summary(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    _write_core_config(config_path)

    report = build_promotion_report(
        "binance_live", config_path=config_path, skip_license=True
    )

    assert report["environment"] == "binance_live"
    assert report["alerting"]["channels"] == ["telegram:primary"]
    assert report["license"]["status"] == "skipped"
    assert report["license"]["reason"] == "cli_skip_requested"
    summary = report["live_readiness_summary"]
    assert summary["status"] == "ok"
    assert "blocked_items" not in summary
    checklist_entries = {entry["item"]: entry for entry in report["live_readiness_checklist"]}
    assert checklist_entries["live_checklist"]["status"] == "ok"
    metadata = report["live_readiness_metadata"]
    assert metadata["checklist_id"] == "stage6-binance"
    assert metadata["status"] == "ok"
    assert metadata.get("checklist_status") == "ok"
    assert metadata.get("resolved_signature_path").endswith("checklist.sig")
    assert Path(metadata["document_root"]) == config_path.parent
    documents = {doc["name"]: doc for doc in metadata["documents"]}
    assert documents["kyc_packet"]["signed"] is True
    assert documents["kyc_packet"]["status"] == "ok"
    assert documents["kyc_packet"]["resolved_path"].endswith("kyc_packet.pdf")
    assert documents["kyc_packet"]["computed_sha256"] == documents["kyc_packet"]["sha256"]


def test_build_promotion_report_supports_custom_document_root(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "core.yaml"
    document_root = tmp_path / "artifacts"
    _write_core_config(config_path, document_root=document_root)

    report = build_promotion_report(
        "binance_live",
        config_path=config_path,
        skip_license=True,
        document_root=document_root,
    )

    summary = report["live_readiness_summary"]
    assert summary["status"] == "ok"
    metadata = report["live_readiness_metadata"]
    assert metadata["status"] == "ok"
    documents = {doc["name"]: doc for doc in metadata["documents"]}
    resolved_path = Path(documents["kyc_packet"]["resolved_path"])
    assert resolved_path.is_relative_to(document_root)

def test_cli_execution_writes_report(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    _write_core_config(config_path)
    output_path = tmp_path / "promotion.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/promotion_to_live.py",
            "binance_live",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--skip-license",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["environment"] == "binance_live"
    live_metadata = payload["live_readiness_metadata"]
    assert live_metadata["status"] == "ok"
    assert live_metadata["documents"][0]["name"] == "kyc_packet"
    assert live_metadata["documents"][0]["status"] == "ok"
    assert Path(live_metadata["document_root"]) == config_path.parent


def test_cli_supports_document_root_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "core.yaml"
    document_root = tmp_path / "artifacts"
    _write_core_config(config_path, document_root=document_root)
    output_path = tmp_path / "promotion.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/promotion_to_live.py",
            "binance_live",
            "--config",
            str(config_path),
            "--document-root",
            str(document_root),
            "--output",
            str(output_path),
            "--skip-license",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    documents = {doc["name"]: doc for doc in payload["live_readiness_metadata"]["documents"]}
    resolved_path = Path(documents["kyc_packet"]["resolved_path"])
    assert resolved_path.is_relative_to(document_root)
    assert Path(payload["live_readiness_metadata"]["document_root"]) == document_root.resolve()


def test_cli_fail_on_blocked_returns_error(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    _write_core_config(config_path)

    # Usuń dokument, aby checklisty zgłosiły blokadę
    missing_doc = config_path.parent / "compliance/live/binance/kyc_packet.pdf"
    missing_doc.unlink()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/promotion_to_live.py",
            "binance_live",
            "--config",
            str(config_path),
            "--fail-on-blocked",
            "--skip-license",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 2
    assert "blokady" in result.stderr


def test_cli_supports_markdown_output(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    _write_core_config(config_path)
    output_path = tmp_path / "promotion.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/promotion_to_live.py",
            "binance_live",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--format",
            "markdown",
            "--skip-license",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert content.startswith("# Raport promotion-to-live")
    assert "## Metadane checklisty" in content
    assert "### Dokumenty" in content
    assert "kyc_packet" in content


def test_cli_all_live_outputs_aggregate_and_individual_reports(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    _write_multi_core_config(config_path)
    aggregate_output = tmp_path / "promotion.json"
    output_dir = tmp_path / "reports"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/promotion_to_live.py",
            "--all-live",
            "--config",
            str(config_path),
            "--output",
            str(aggregate_output),
            "--output-dir",
            str(output_dir),
            "--skip-license",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    assert aggregate_output.exists()
    payload = json.loads(aggregate_output.read_text(encoding="utf-8"))
    assert payload["summary"]["status"] == "blocked"
    assert payload["summary"]["total_environments"] == 2
    assert payload["summary"]["ok_count"] == 1
    assert payload["summary"]["blocked_count"] == 1
    assert "coinbase_live" in payload["summary"]["blocked_environments"]
    assert "live_checklist" in payload["summary"]["blocked_items"]
    assert "kyc_aml_signoff" in payload["summary"]["blocked_items"]
    assert payload["summary"]["blocked_items"] != payload["summary"]["blocked_environments"]
    assert "aml_packet" in payload["summary"]["blocked_documents"]
    assert payload["summary"]["blocked_items_count"] == 3
    assert payload["summary"]["blocked_documents_count"] == 1
    assert payload["summary"]["license_issue_count"] == 0
    assert payload["summary"]["license_skipped_count"] == 2
    assert payload["summary"]["license_skipped_reasons"] == ["cli_skip_requested"]
    skipped_envs = {entry["environment"] for entry in payload["summary"]["license_skipped"]}
    assert skipped_envs == {"binance_live", "coinbase_live"}
    assert {
        entry["reason"] for entry in payload["summary"]["license_skipped"]
    } == {"cli_skip_requested"}
    assert len(payload["reports"]) == 2

    assert output_dir.is_dir()
    binance_report = json.loads(
        (output_dir / "binance_live.json").read_text(encoding="utf-8")
    )
    coinbase_report = json.loads(
        (output_dir / "coinbase_live.json").read_text(encoding="utf-8")
    )
    assert binance_report["live_readiness_summary"]["status"] == "ok"
    assert coinbase_report["live_readiness_summary"]["status"] == "blocked"
    assert "aml_packet" in coinbase_report["live_readiness_summary"]["blocked_documents"]


def test_cli_all_live_fail_on_blocked(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    _write_multi_core_config(config_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/promotion_to_live.py",
            "--all-live",
            "--config",
            str(config_path),
            "--fail-on-blocked",
            "--skip-license",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 2
    assert "środowiska: coinbase_live" in result.stderr


def test_cli_fail_on_license_single_environment(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    _write_core_config(config_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/promotion_to_live.py",
            "binance_live",
            "--config",
            str(config_path),
            "--fail-on-license",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 3
    assert "problemy licencyjne" in result.stderr


def test_cli_fail_on_license_with_skip_reports_skipped(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    _write_core_config(config_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/promotion_to_live.py",
            "binance_live",
            "--config",
            str(config_path),
            "--fail-on-license",
            "--skip-license",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    assert "Walidacja licencji została pominięta" in result.stderr


def test_cli_fail_on_skipped_license(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    _write_core_config(config_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/promotion_to_live.py",
            "binance_live",
            "--config",
            str(config_path),
            "--skip-license",
            "--fail-on-skipped-license",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 4
    assert "cli_skip_requested" in result.stderr


def test_cli_all_live_fail_on_license(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    _write_multi_core_config(config_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/promotion_to_live.py",
            "--all-live",
            "--config",
            str(config_path),
            "--fail-on-license",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 3
    assert "problemy licencyjne" in result.stderr
