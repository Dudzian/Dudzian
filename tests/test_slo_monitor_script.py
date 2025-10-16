import json
import subprocess
import sys
from pathlib import Path

import pytest

from bot_core.security.signing import build_hmac_signature


def test_slo_monitor_cli_generates_signed_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    definitions_path = tmp_path / "definitions.yaml"
    definitions_path.write_text(
        """
        slo:
          - name: availability
            indicator: router_availability_pct
            target: 99.0
            comparison: ">="
            warning_threshold: 99.5
            severity: critical
          - name: latency
            indicator: router_latency_ms
            target: 250.0
            comparison: "<="
            warning_threshold: 200.0
        composites:
          - name: core_stack
            objectives:
              - availability
              - latency
            max_breaches: 0
            severity: critical
        """,
        encoding="utf-8",
    )

    metrics_path = tmp_path / "metrics.json"
    metrics_payload = {
        "router_availability_pct": {
            "indicator": "router_availability_pct",
            "value": 98.8,
            "window_start": "2024-01-01T00:00:00Z",
            "window_end": "2024-01-02T00:00:00Z",
            "sample_size": 86400,
        },
        "router_latency_ms": {
            "indicator": "router_latency_ms",
            "value": 275.0,
            "window_start": "2024-01-01T00:00:00Z",
            "window_end": "2024-01-01T12:00:00Z",
            "sample_size": 5000,
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")

    output_path = tmp_path / "report.json"
    signature_path = tmp_path / "report.sig"
    csv_path = tmp_path / "report.csv"
    key_value = "stage6-secret-key"
    monkeypatch.setenv("SLO_MONITOR_KEY", key_value)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/slo_monitor.py",
            "--definitions",
            str(definitions_path),
            "--metrics",
            str(metrics_path),
            "--output",
            str(output_path),
            "--output-csv",
            str(csv_path),
            "--signature",
            str(signature_path),
            "--signing-key-env",
            "SLO_MONITOR_KEY",
            "--signing-key-id",
            "local-stage6",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
    assert signature_path.exists()
    assert csv_path.exists()
    assert "podpisem" in result.stdout

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(payload["results"]) == {"availability", "latency"}
    assert payload["results"]["availability"]["status"] == "breach"
    assert payload["results"]["latency"]["status"] == "breach"
    assert "composites" in payload
    composite_results = payload["composites"]["results"]
    assert composite_results["core_stack"]["status"] == "breach"
    assert payload["summary"]["composites"]["status_counts"]["breach"] == 1

    expected_signature = build_hmac_signature(payload, key=key_value.encode("utf-8"), key_id="local-stage6")
    recorded_signature = json.loads(signature_path.read_text(encoding="utf-8"))
    assert recorded_signature == expected_signature

    csv_rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_rows[0].startswith("type,name,status")
    assert any(row.startswith("slo,availability") for row in csv_rows[1:])
