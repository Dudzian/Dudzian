from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import render_paper_smoke_summary as module


def _sample_summary() -> dict:
    return {
        "environment": "binance_paper",
        "operator": "CI Agent",
        "severity": "info",
        "timestamp": "2024-02-01T00:00:00Z",
        "window": {"start": "2024-02-01T00:00:00Z", "end": "2024-02-02T00:00:00Z"},
        "report": {
            "directory": "/tmp/report",
            "summary_path": "/tmp/report/summary.json",
            "summary_sha256": "deadbeef",
        },
        "precheck": {"status": "ok", "coverage_status": "ok", "risk_status": "ok"},
        "manifest": {
            "manifest_path": "/tmp/cache/ohlcv_manifest.sqlite",
            "metrics_path": "/tmp/cache/manifest.prom",
            "summary_path": "/tmp/cache/manifest_summary.json",
            "worst_status": "missing_metadata",
            "status_counts": {"missing_metadata": 3, "ok": 5},
            "total_entries": 8,
            "exit_code": 2,
            "stage": "paper",
            "risk_profile": "balanced",
            "deny_status": ["warning", "missing_metadata"],
            "summary": {
                "status_counts": {"missing_metadata": 3, "ok": 5},
                "worst_status": "missing_metadata",
            },
            "summary_signature": {
                "algorithm": "HMAC-SHA256",
                "value": "YWJj",
                "key_id": "ci-manifest",
            },
        },
        "token_audit": {
            "report_path": "/tmp/security/token_report.json",
            "status": "ok",
            "exit_code": 0,
            "warnings": ["Legacy auth token"],
            "errors": [],
            "report": {
                "warnings": ["Legacy auth token"],
                "errors": [],
                "services": [
                    {
                        "service": "metrics_service",
                        "findings": [
                            {"level": "warning", "message": "Legacy auth token"},
                        ],
                    }
                ],
            },
        },
        "tls_audit": {
            "report_path": "/tmp/security/tls_report.json",
            "status": "warning",
            "exit_code": 0,
            "warnings": ["MetricsService działa bez TLS"],
            "errors": [],
            "report": {
                "warnings": ["MetricsService działa bez TLS"],
                "errors": [],
                "services": {
                    "metrics_service": {
                        "enabled": True,
                        "auth_token_configured": False,
                        "tls": {"enabled": False},
                        "warnings": ["MetricsService działa bez TLS"],
                        "errors": [],
                    }
                },
            },
        },
    }


def test_render_manifest_section_present(tmp_path: Path) -> None:
    summary = _sample_summary()
    rendered = module.render_summary_markdown(summary, max_json_chars=2000)

    assert "## Manifest danych OHLCV" in rendered
    assert "missing_metadata" in rendered
    assert "Łączna liczba wpisów" in rendered
    assert "Blokowane statusy" in rendered
    assert "### Liczba wpisów manifestu wg statusu" in rendered
    assert "Podpis – algorytm" in rendered
    assert "HMAC-SHA256" in rendered
    assert "## Audyt tokenów RBAC" in rendered
    assert "## Audyt TLS usług runtime" in rendered
    assert "Ostrzeżenia TLS" in rendered


def test_cli_integration(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_sample_summary()), encoding="utf-8")
    output_path = tmp_path / "report.md"

    exit_code = module.main([
        "--summary-json",
        str(summary_path),
        "--output",
        str(output_path),
        "--max-json-chars",
        "500",
    ])

    assert exit_code == 0
    report = output_path.read_text(encoding="utf-8")
    assert "Manifest danych OHLCV" in report
    assert "Audyt tokenów RBAC" in report
    assert "Audyt TLS usług runtime" in report
