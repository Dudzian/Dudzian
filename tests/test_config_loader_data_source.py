from __future__ import annotations

from pathlib import Path

from bot_core.config.loader import load_core_config


def test_load_core_config_reads_environment_data_source(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles:
          manual:
            max_daily_loss_pct: 0.0
            max_position_pct: 0.0
            target_volatility: 0.0
            max_leverage: 0.0
            stop_loss_atr_multiple: 0.0
            max_open_positions: 0
            hard_drawdown_pct: 0.0
        environments:
          coinbase_offline:
            exchange: coinbase_spot
            environment: paper
            keychain_key: coinbase_offline_key
            data_cache_path: ./var/data/offline/coinbase
            risk_profile: manual
            alert_channels: []
            offline_mode: true
            data_source:
              enable_snapshots: false
              cache_namespace: research_offline
            report_storage:
              backend: file
              directory: reports
              filename_pattern: offline-%Y%m%d.json
              retention_days: 30
              fsync: true
        reporting: {}
        alerts: {}
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    environment = config.environments["coinbase_offline"]
    assert environment.offline_mode is True
    assert environment.data_source is not None
    assert environment.data_source.enable_snapshots is False
    assert environment.data_source.cache_namespace == "research_offline"
    assert environment.report_storage is not None
    assert environment.report_storage.backend == "file"
    assert environment.report_storage.directory == "reports"
    assert environment.report_storage.filename_pattern == "offline-%Y%m%d.json"
    assert environment.report_storage.retention_days == 30
    assert environment.report_storage.fsync is True
