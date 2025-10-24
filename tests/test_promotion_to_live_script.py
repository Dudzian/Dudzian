from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.promotion_to_live import build_promotion_report


def _write_core_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
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
                      sha256: 79b8d3b02b3e29f4c4a0d428c2a7c4de48bd97f49deed19c7ef287c93883bf94
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
    documents = {doc["name"]: doc for doc in metadata["documents"]}
    assert documents["kyc_packet"]["signed"] is True


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
    assert payload["live_readiness_metadata"]["documents"][0]["name"] == "kyc_packet"


def test_promotion_report_blocks_on_guardrail(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    _write_core_config(config_path)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({"guardrails": {"allowed": False, "reason": "max drawdown breached"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        build_promotion_report(
            "binance_live",
            config_path=config_path,
            skip_license=True,
            backtest_summary_path=summary_path,
        )
