from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from bot_core.tco.models import CostBreakdown, ProfileCostSummary, StrategyCostSummary, TCOReport
from bot_core.tco.reporting import SignedArtifact, TCOReportWriter


@pytest.fixture()
def sample_report() -> TCOReport:
    breakdown_total = CostBreakdown(
        commission=Decimal("5.1"),
        slippage=Decimal("1.3"),
        funding=Decimal("0.4"),
        other=Decimal("0.2"),
    )
    breakdown_profile = CostBreakdown(
        commission=Decimal("3.4"),
        slippage=Decimal("0.8"),
        funding=Decimal("0.1"),
        other=Decimal("0.05"),
    )
    profile_summary = ProfileCostSummary(
        profile="balanced",
        trade_count=3,
        notional=Decimal("25000"),
        breakdown=breakdown_profile,
    )
    total_summary = ProfileCostSummary(
        profile="__total__",
        trade_count=3,
        notional=Decimal("25000"),
        breakdown=breakdown_total,
    )
    strategy_summary = StrategyCostSummary(
        strategy="mean_reversion",
        profiles={"balanced": profile_summary},
        total=total_summary,
    )
    return TCOReport(
        generated_at=datetime(2024, 3, 14, 9, 26, tzinfo=timezone.utc),
        metadata={"environment": "paper", "events_count": 3},
        strategies={"mean_reversion": strategy_summary},
        total=total_summary,
        alerts=["cost_limit_exceeded"],
    )


def test_report_writer_builds_csv_pdf_and_json(sample_report: TCOReport) -> None:
    writer = TCOReportWriter(sample_report)

    csv_output = writer.build_csv()
    lines = [line for line in csv_output.splitlines() if line]
    assert lines[0].startswith("strategy,profile,trade_count")
    assert any("mean_reversion,balanced" in line for line in lines[1:])
    assert lines[-1].startswith("__TOTAL__")

    pdf_bytes = writer.build_pdf()
    assert pdf_bytes.startswith(b"%PDF-1.4")
    pdf_text = pdf_bytes.decode("latin-1", errors="ignore")
    assert "Raport koszt" in pdf_text

    json_payload = writer.build_json()
    assert json_payload == sample_report.to_dict()


def test_report_writer_writes_and_signs_artifacts(
    tmp_path: Path, sample_report: TCOReport
) -> None:
    writer = TCOReportWriter(sample_report)

    artifacts = writer.write_outputs(tmp_path, basename="custom_report")
    assert set(artifacts.keys()) == {"csv", "pdf", "json"}
    for path in artifacts.values():
        assert path.exists()

    json_payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
    assert json_payload["metadata"]["environment"] == "paper"

    signing_key = b"k" * 32
    signed = writer.sign_artifacts(artifacts, signing_key=signing_key, key_id="key-1")
    assert set(signed.keys()) == {"csv", "pdf", "json"}
    for label, artifact in signed.items():
        assert isinstance(artifact, SignedArtifact)
        expected_hash = hashlib.sha256(artifacts[label].read_bytes()).hexdigest()
        assert artifact.payload["sha256"] == expected_hash
        assert artifact.signature_path.exists()
        signature_doc = json.loads(artifact.signature_path.read_text(encoding="utf-8"))
        assert signature_doc["payload"]["artifact_type"] == label
        assert signature_doc["signature"]["algorithm"] == "HMAC-SHA256"

    with pytest.raises(ValueError):
        writer.sign_artifacts(artifacts, signing_key=b"short")
