from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from scripts.promotion_to_live import build_promotion_report


def _write_core_config(path: Path) -> None:
    base_dir = path.parent
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


def test_build_promotion_report_produces_summary(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    _write_core_config(config_path)

    report = build_promotion_report(
        "binance_live", config_path=config_path, skip_license=True
    )

    assert report["environment"] == "binance_live"
    assert report["alerting"]["channels"] == ["telegram:primary"]
    checklist_entries = {entry["item"]: entry for entry in report["live_readiness_checklist"]}
    assert checklist_entries["live_checklist"]["status"] == "ok"
    metadata = report["live_readiness_metadata"]
    assert metadata["checklist_id"] == "stage6-binance"
    assert metadata["status"] == "ok"
    assert metadata.get("checklist_status") == "ok"
    assert metadata.get("resolved_signature_path").endswith("checklist.sig")
    documents = {doc["name"]: doc for doc in metadata["documents"]}
    assert documents["kyc_packet"]["signed"] is True
    assert documents["kyc_packet"]["status"] == "ok"
    assert documents["kyc_packet"]["resolved_path"].endswith("kyc_packet.pdf")
    assert documents["kyc_packet"]["computed_sha256"] == documents["kyc_packet"]["sha256"]


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
