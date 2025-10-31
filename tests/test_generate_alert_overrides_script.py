from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


from scripts.generate_alert_overrides import run as generate_overrides


ROOT = Path(__file__).resolve().parents[1]
def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sample_slo_report() -> dict[str, object]:
    now = datetime.now(timezone.utc)
    return {
        "definitions": [
            {
                "name": "latency",
                "indicator": "latency_p95",
                "target": 0.5,
                "comparison": "<=",
                "warning_threshold": 0.4,
                "severity": "critical",
                "tags": ["stage6"],
            },
            {
                "name": "throughput",
                "indicator": "orders_per_minute",
                "target": 120,
                "comparison": ">=",
                "warning_threshold": 140,
                "severity": "warning",
            },
        ],
        "results": {
            "latency": {
                "indicator": "latency_p95",
                "value": 0.7,
                "target": 0.5,
                "comparison": "<=",
                "status": "breach",
                "severity": "critical",
                "warning_threshold": 0.4,
                "window_start": (now - timedelta(minutes=30)).isoformat(),
                "window_end": now.isoformat(),
                "sample_size": 300,
                "reason": "latency high",
                "metadata": {"latency_p95": 0.7},
            },
            "throughput": {
                "indicator": "orders_per_minute",
                "value": 130,
                "target": 120,
                "comparison": ">=",
                "status": "warning",
                "severity": "warning",
                "warning_threshold": 140,
                "window_start": (now - timedelta(minutes=30)).isoformat(),
                "window_end": now.isoformat(),
                "sample_size": 600,
                "reason": "throughput dip",
            },
        },
        "composites": {
            "definitions": [
                {
                    "name": "core_stack",
                    "objectives": ["latency", "throughput"],
                    "max_breaches": 0,
                    "severity": "critical",
                }
            ],
            "results": {
                "core_stack": {
                    "name": "core_stack",
                    "status": "breach",
                    "severity": "critical",
                    "counts": {"breach": 1, "warning": 1, "ok": 0, "unknown": 0},
                    "objectives": ["latency", "throughput"],
                    "reason": "latency breach + throughput warning",
                }
            },
        },
    }


def test_generate_alert_overrides_cli(tmp_path: Path) -> None:
    report_path = tmp_path / "slo_report.json"
    _write_json(report_path, _sample_slo_report())

    output_path = tmp_path / "alert_overrides.json"
    signature_path = tmp_path / "alert_overrides.sig"

    exit_code = generate_overrides(
        [
            "--slo-report",
            str(report_path),
            "--output",
            str(output_path),
            "--signature",
            str(signature_path),
            "--expires-in",
            "60",
            "--signing-key",
            "override-secret",
            "--signing-key-id",
            "ops",
            "--tag",
            "stage6",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert signature_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["active"] == 3
    alerts = {entry["alert"] for entry in payload["annotations"]}
    assert alerts == {"latency", "throughput", "core_stack"}


def test_generate_alert_overrides_cli_with_existing(tmp_path: Path) -> None:
    report_path = tmp_path / "slo_report.json"
    _write_json(report_path, _sample_slo_report())

    existing_override = {
        "schema": "stage6.observability.alert_overrides",
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overrides": [
            {
                "alert": "latency",
                "status": "breach",
                "severity": "critical",
                "reason": "previous",
                "indicator": "latency_p95",
                "created_at": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
                "expires_at": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
                "tags": ["stage6"],
                "metadata": {"error_budget_pct": 0.5},
            },
            {
                "alert": "throughput",
                "status": "warning",
                "severity": "warning",
                "reason": "active",
                "indicator": "orders_per_minute",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
            },
        ],
        "summary": {},
        "annotations": [],
    }
    existing_path = tmp_path / "existing.json"
    _write_json(existing_path, existing_override)

    output_path = tmp_path / "merged.json"

    exit_code = generate_overrides(
        [
            "--slo-report",
            str(report_path),
            "--existing",
            str(existing_path),
            "--output",
            str(output_path),
            "--skip-warnings",
            "--expires-in",
            "0",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    overrides = payload["overrides"]
    # warning został pominięty, ale istniejący aktywny override zachował się
    alerts = {entry["alert"] for entry in overrides}
    assert alerts == {"latency", "throughput", "core_stack"}
    assert payload["summary"]["active"] >= 2
