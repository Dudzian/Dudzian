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
        "telemetry": {
            "summary_path": "/tmp/telemetry/summary.json",
            "decision_log_path": "/tmp/telemetry/decision.log",
            "metrics_source_path": "/tmp/telemetry/metrics.prom",
            "risk_profile": {
                "name": "balanced",
                "source": "core.yaml",
                "environment_fallback": "paper",
                "profiles_file": "/tmp/config/risk_profiles.yaml",
            },
            "snippets": {
                "env_path": "/tmp/telemetry/risk.env",
                "yaml_path": "/tmp/telemetry/risk.yaml",
            },
            "decision_log_report": {
                "status": "ok",
                "exit_code": 0,
                "path": "/tmp/telemetry/report.json",
                "command": ["python", "verify_decision_log.py"],
                "payload": {"status": "ok"},
            },
            "required_auth_scopes": ["metrics.read", "risk.read"],
            "auth_scope_requirements": {
                "metrics_service": {
                    "required_scopes": ["metrics.read"],
                    "sources": [
                        {
                            "source": "summary",
                            "metadata": {"auth_token_scope_required": "metrics.read"},
                        }
                    ],
                },
                "risk_service": {
                    "required_scopes": ["risk.read"],
                    "sources": [
                        {
                            "source": "summary",
                            "metadata": {"auth_token_scope_required": "risk.read"},
                        }
                    ],
                },
            },
            "risk_service_requirements": {
                "details": {
                    "require_tls": True,
                    "tls_materials": ["root_cert", "client_cert"],
                    "expected_server_sha256": ["aa:bb"],
                    "required_scopes": ["risk.read"],
                    "required_token_ids": ["risk-reader"],
                    "require_auth_token": True,
                },
                "cli_args": ["--require-risk-service-tls"],
                "combined_metadata": {
                    "tls_enabled": True,
                    "pinned_fingerprints": ["aa:bb"],
                },
                "metadata": [
                    {
                        "source": "summary",
                        "metadata": {"tls_enabled": True, "auth_token_scope_checked": True},
                    }
                ],
            },
            "bundle": {
                "output_dir": "/tmp/telemetry/bundle",
                "manifest_path": "/tmp/telemetry/bundle/manifest.json",
            },
        },
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
        "security_baseline": {
            "report_path": "/tmp/security/baseline.json",
            "status": "warning",
            "baseline_status": "warning",
            "exit_code": 0,
            "warnings": ["MetricsService działa bez TLS"],
            "errors": [],
            "summary_signature": {
                "algorithm": "HMAC-SHA256",
                "value": "YmFzZWxpbmUtc2lnbmF0dXJl",
                "key_id": "baseline-ci",
            },
            "report": {
                "status": "warning",
                "warnings": ["MetricsService działa bez TLS"],
                "errors": [],
                "tls": {"services": {"metrics_service": {"warnings": ["MetricsService działa bez TLS"]}}},
                "tokens": {"services": []},
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
    assert "## Telemetria runtime" in rendered
    assert "### Wymagania RiskService" in rendered
    assert "metrics.read" in rendered
    assert "risk.read" in rendered
    assert "risk-reader" in rendered
    assert "Flagi verify_decision_log" in rendered
    assert "## Audyt tokenów RBAC" in rendered
    assert "## Audyt TLS usług runtime" in rendered
    assert "Ostrzeżenia TLS" in rendered
    assert "## Audyt bezpieczeństwa (TLS + RBAC)" in rendered
    assert "Ostrzeżenia bezpieczeństwa" in rendered
    assert "Szczegóły podpisu audytu bezpieczeństwa" in rendered
    assert "baseline-ci" in rendered


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
    assert "Telemetria runtime" in report
    assert "Audyt tokenów RBAC" in report
    assert "Audyt TLS usług runtime" in report
    assert "Audyt bezpieczeństwa (TLS + RBAC)" in report
