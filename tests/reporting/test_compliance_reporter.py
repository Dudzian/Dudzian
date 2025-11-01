from datetime import datetime, timezone
from pathlib import Path

import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from core.compliance import ComplianceAuditResult, ComplianceFinding
from core.reporting import ComplianceReport


def _sample_result() -> ComplianceAuditResult:
    finding = ComplianceFinding(
        rule_id="AML_SUSPICIOUS_TAG",
        severity="critical",
        message="Strategia zawiera tagi wysokiego ryzyka",
        metadata={"tags": ("sanctioned",)},
    )
    return ComplianceAuditResult(
        generated_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        passed=False,
        findings=(finding,),
        context_summary={"strategy": "demo", "transactions_analyzed": 0},
        config_path=Path("config/compliance/audit.yml"),
    )


def test_compliance_report_markdown_and_json(tmp_path: Path) -> None:
    result = _sample_result()
    report = ComplianceReport.from_audit(result)

    markdown = report.to_markdown()
    assert "Raport audytu zgodności" in markdown
    assert "AML_SUSPICIOUS_TAG" in markdown
    assert "NEGATYWNY" in markdown

    md_path = report.write_markdown(tmp_path)
    json_path = report.write_json(tmp_path)

    assert md_path.exists()
    assert json_path.exists()

    payload = json_path.read_text(encoding="utf-8")
    assert "AML_SUSPICIOUS_TAG" in payload
    assert "critical" in payload


def test_compliance_report_recommendations(tmp_path: Path) -> None:
    result = _sample_result()
    report = ComplianceReport.from_audit(result)
    assert any("Skontaktuj" in item for item in report.recommendations)
    assert report.severity_counts()["critical"] == 1
    report.write_markdown(tmp_path / "reports")
