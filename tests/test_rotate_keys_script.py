from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import textwrap

from bot_core.security import RotationRegistry
from scripts.rotate_keys import run as rotate_keys_run


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def test_rotate_keys_generates_plan_and_updates_registry(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "api::trading": _iso(datetime.now(timezone.utc) - timedelta(days=120))
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        textwrap.dedent(
            f"""
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
              key_rotation:
                registry_path: {registry_path}
                default_interval_days: 60
                default_warn_within_days: 7
                entries:
                  - key: api
                    purpose: trading
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    exit_code = rotate_keys_run(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--basename",
            "plan",
        ]
    )
    assert exit_code == 0
    report_path = output_dir / "plan.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["results"][0]["state"] in {"due", "overdue"}

    exit_code = rotate_keys_run(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--basename",
            "plan_execute",
            "--execute",
        ]
    )
    assert exit_code == 0
    registry = RotationRegistry(registry_path)
    status = registry.status("api", "trading", interval_days=60.0)
    assert not status.is_due
