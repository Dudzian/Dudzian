from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from core.compliance import ComplianceAuditor
from core.monitoring.events import ComplianceViolation


def _write_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def test_compliance_auditor_passes_when_no_findings(tmp_path: Path) -> None:
    config_path = tmp_path / "audit.yml"
    _write_config(
        config_path,
        {
            "kyc": {"required_fields": ["full_name", "address"], "severity": "high"},
            "aml": {
                "blocked_countries": ["IR"],
                "suspicious_tags": ["sanctioned"],
                "forbidden_data_sources": ["darkpool"],
                "severity": "critical",
            },
            "transaction_limits": {
                "max_single_trade_usd": 5000,
                "max_daily_volume_usd": 20000,
                "lookback_days": 2,
                "severity": "warning",
            },
        },
    )
    auditor = ComplianceAuditor(config_path=config_path)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    result = auditor.audit(
        strategy_config={"name": "grid", "tags": ["low_risk"]},
        data_sources=["ohlcv"],
        transactions=[
            {
                "id": "tx-1",
                "timestamp": now.isoformat(),
                "usd_value": 1200.0,
                "counterparty_country": "PL",
            }
        ],
        kyc_profile={"full_name": "Jan Kowalski", "address": "Warszawa", "country": "PL"},
        as_of=now,
    )
    assert result.passed is True
    assert result.findings == ()
    assert result.context_summary["strategy"] == "grid"
    assert result.context_summary["transactions_analyzed"] == 1


def test_compliance_auditor_detects_violations_and_emits_events(tmp_path: Path) -> None:
    config_path = tmp_path / "audit.yml"
    _write_config(
        config_path,
        {
            "kyc": {"required_fields": ["full_name", "address"], "severity": "high"},
            "aml": {
                "blocked_countries": ["IR"],
                "high_risk_jurisdictions": ["RU"],
                "suspicious_tags": ["sanctioned"],
                "max_unverified_volume_usd": 1000,
                "severity": "critical",
            },
            "transaction_limits": {
                "max_single_trade_usd": 500,
                "max_daily_volume_usd": 800,
                "lookback_days": 1,
                "severity": "warning",
            },
        },
    )
    auditor = ComplianceAuditor(config_path=config_path)
    now = datetime(2025, 1, 2, tzinfo=timezone.utc)
    events: list[ComplianceViolation] = []
    result = auditor.audit(
        strategy_config={"name": "mixer", "tags": ["sanctioned"]},
        data_sources=["darkpool"],
        transactions=[
            {
                "id": "tx-1",
                "timestamp": now.isoformat(),
                "usd_value": 900.0,
                "counterparty_country": "RU",
            },
            {
                "id": "tx-2",
                "timestamp": now.isoformat(),
                "usd_value": 200.0,
                "counterparty_country": "PL",
            },
        ],
        kyc_profile={"country": "IR"},
        as_of=now,
        event_publisher=lambda event: events.append(event) if isinstance(event, ComplianceViolation) else None,
    )

    assert result.passed is False
    assert len(result.findings) >= 3
    rule_ids = {finding.rule_id for finding in result.findings}
    assert "KYC_MISSING_FIELDS" in rule_ids
    assert "AML_BLOCKED_COUNTRY" in rule_ids
    assert "TX_DAILY_LIMIT_EXCEEDED" in rule_ids or "TX_SINGLE_LIMIT_EXCEEDED" in rule_ids

    emitted_rules = {event.rule_id for event in events}
    assert "AML_BLOCKED_COUNTRY" in emitted_rules
    assert any(event.severity.lower() in {"critical", "high"} for event in events)
