from __future__ import annotations

from pathlib import Path

import textwrap

from bot_core.config import load_core_config


def test_load_core_config_observability_section(tmp_path: Path) -> None:
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
                  objective: 200
                  comparator: "<="
                  aggregation: avg
                  window_minutes: 60
                  label_filters:
                    schedule: test
              key_rotation:
                registry_path: registry.json
                default_interval_days: 30
                default_warn_within_days: 5
                entries:
                  - key: api
                    purpose: trading
                    interval_days: 25
                    warn_within_days: 4
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_core_config(config_path)
    assert config.observability is not None
    slo_config = config.observability.slo["latency"]
    assert slo_config.metric == "bot_core_decision_latency_ms"
    assert slo_config.aggregation == "average"
    assert slo_config.label_filters == {"schedule": "test"}
    rotation = config.observability.key_rotation
    assert rotation is not None
    assert rotation.registry_path.endswith("registry.json")
    assert len(rotation.entries) == 1
    entry = rotation.entries[0]
    assert entry.key == "api"
    assert entry.purpose == "trading"
    assert entry.interval_days == 25
    assert entry.warn_within_days == 4
