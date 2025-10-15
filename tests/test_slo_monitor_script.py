from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import textwrap

from scripts.slo_monitor import run as slo_monitor_run


def _timestamp(minutes_ago: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return dt.isoformat().replace("+00:00", "Z")


def test_slo_monitor_generates_report(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            environments:
              demo:
                exchange: binance
                environment: paper
                keychain_key: demo
                data_cache_path: cache
                risk_profile: conservative
                alert_channels: []
            risk_profiles:
              conservative:
                max_daily_loss_pct: 0.01
                max_position_pct: 0.2
                target_volatility: 0.08
                max_leverage: 2.0
                stop_loss_atr_multiple: 2.5
                max_open_positions: 3
                hard_drawdown_pct: 0.12
            observability:
              slo:
                latency:
                  metric: bot_core_decision_latency_ms
                  objective: 220
                  comparator: "<="
                  aggregation: p95
                  window_minutes: 120
                cost:
                  metric: bot_core_trade_cost_bps
                  objective: 12
                  comparator: "<="
                  aggregation: average
                  window_minutes: 120
                fill:
                  metric: bot_core_fill_rate_pct
                  objective: 0.94
                  comparator: ">="
                  aggregation: average
                  window_minutes: 120
              key_rotation:
                registry_path: registry.json
                entries:
                  - key: api
                    purpose: trading
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "metric": "bot_core_decision_latency_ms",
                        "value": 180,
                        "timestamp": _timestamp(10),
                        "labels": {"schedule": "core"},
                    }
                ),
                json.dumps(
                    {
                        "metric": "bot_core_trade_cost_bps",
                        "value": 9.5,
                        "timestamp": _timestamp(5),
                    }
                ),
                json.dumps(
                    {
                        "metric": "bot_core_fill_rate_pct",
                        "value": 0.97,
                        "timestamp": _timestamp(8),
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
    statuses = {entry["name"]: entry["status"] for entry in report["results"]}
    assert statuses == {"latency": "pass", "cost": "pass", "fill": "pass"}
