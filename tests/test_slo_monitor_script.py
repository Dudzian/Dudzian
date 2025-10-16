from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Wspólne
from bot_core.security.signing import build_hmac_signature

# Detekcja wariantu skryptu:
# - wariant "main" eksportuje run(argv) i używa --config/--metrics JSONL
# - wariant "HEAD" nie eksportuje run(argv) i używa --definitions/--metrics JSON + podpis HMAC
try:
    from scripts.slo_monitor import run as slo_monitor_run  # type: ignore[attr-defined]
    _HAS_RUN = True
except Exception:  # pragma: no cover
    slo_monitor_run = None  # type: ignore[assignment]
    _HAS_RUN = False


# ==================== Test dla wariantu HEAD (definitions + metrics + HMAC) ====================
@pytest.mark.skipif(_HAS_RUN, reason="Wariant main wykryty – pomijam test CLI definitions/metrics")
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
    assert "podpisem" in result.stdout  # komunikat o podpisie

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    # Oczekujemy struktur z wynikami SLO i kompozytów (zgodnie z implementacją HEAD)
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


# ==================== Test dla wariantu main (config + JSONL metrics) ====================
@pytest.mark.skipif(not _HAS_RUN, reason="Wariant HEAD wykryty – pomijam test config/JSONL")
def test_slo_monitor_generates_report(tmp_path: Path) -> None:
    assert slo_monitor_run is not None  # dla type-checkerów

    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        (
            "environments:\n"
            "  demo:\n"
            "    exchange: binance\n"
            "    environment: paper\n"
            "    keychain_key: demo\n"
            "    data_cache_path: cache\n"
            "    risk_profile: conservative\n"
            "    alert_channels: []\n"
            "risk_profiles:\n"
            "  conservative:\n"
            "    max_daily_loss_pct: 0.01\n"
            "    max_position_pct: 0.2\n"
            "    target_volatility: 0.08\n"
            "    max_leverage: 2.0\n"
            "    stop_loss_atr_multiple: 2.5\n"
            "    max_open_positions: 3\n"
            "    hard_drawdown_pct: 0.12\n"
            "observability:\n"
            "  slo:\n"
            "    latency:\n"
            "      metric: bot_core_decision_latency_ms\n"
            "      objective: 220\n"
            "      comparator: \"<=\"\n"
            "      aggregation: p95\n"
            "      window_minutes: 120\n"
            "    cost:\n"
            "      metric: bot_core_trade_cost_bps\n"
            "      objective: 12\n"
            "      comparator: \"<=\"\n"
            "      aggregation: average\n"
            "      window_minutes: 120\n"
            "    fill:\n"
            "      metric: bot_core_fill_rate_pct\n"
            "      objective: 0.94\n"
            "      comparator: \">=\"\n"
            "      aggregation: average\n"
            "      window_minutes: 120\n"
            "  key_rotation:\n"
            "    registry_path: registry.json\n"
            "    entries:\n"
            "      - key: api\n"
            "        purpose: trading\n"
        ),
        encoding="utf-8",
    )

    def _ts(minutes_ago: float) -> str:
        dt = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        return dt.isoformat().replace("+00:00", "Z")

    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "metric": "bot_core_decision_latency_ms",
                        "value": 180,
                        "timestamp": _ts(10),
                        "labels": {"schedule": "core"},
                    }
                ),
                json.dumps(
                    {
                        "metric": "bot_core_trade_cost_bps",
                        "value": 9.5,
                        "timestamp": _ts(5),
                    }
                ),
                json.dumps(
                    {
                        "metric": "bot_core_fill_rate_pct",
                        "value": 0.97,
                        "timestamp": _ts(8),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    exit_code = slo_monitor_run(
        [
            "--metrics",
            str(metrics_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--basename",
            "report",
        ]
    )
    assert exit_code == 0
    report_path = output_dir / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # Wariant main zwraca listę wyników z nazwą i statusem
    statuses = {entry["name"]: entry["status"] for entry in report["results"]}
    assert statuses == {"latency": "pass", "cost": "pass", "fill": "pass"}
