from pathlib import Path

import yaml

from bot_core.config.loader import load_core_config


def test_load_core_config_parses_decision_engine_tco(tmp_path: Path) -> None:
    config_data = {
        "risk_profiles": {
            "conservative": {
                "max_daily_loss_pct": 0.02,
                "max_position_pct": 0.1,
                "target_volatility": 0.1,
                "max_leverage": 2.0,
                "stop_loss_atr_multiple": 1.0,
                "max_open_positions": 5,
                "hard_drawdown_pct": 0.2,
            }
        },
        "alerts": {},
        "environments": {
            "paper_binance": {
                "exchange": "binance_spot",
                "environment": "paper",
                "keychain_key": "binance_key",
                "credential_purpose": "trading",
                "data_cache_path": "./cache",
                "risk_profile": "conservative",
                "alert_channels": [],
                "ip_allowlist": [],
                "required_permissions": [],
                "forbidden_permissions": [],
            }
        },
        "decision_engine": {
            "orchestrator": {
                "max_cost_bps": 25.0,
                "min_net_edge_bps": 5.0,
                "max_daily_loss_pct": 0.2,
                "max_drawdown_pct": 0.3,
                "max_position_ratio": 3.0,
                "max_open_positions": 8,
                "max_latency_ms": 250.0,
            },
            "min_probability": 0.5,
            "require_cost_data": True,
            "tco": {
                "reports": ["data/tco/baseline.json", "../shared/tco.json"],
                "require_at_startup": True,
                "warn_report_age_hours": 12,
                "max_report_age_hours": 36,
            },
        },
    }

    config_path = tmp_path / "core.yaml"
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    core_config = load_core_config(config_path)

    assert core_config.decision_engine is not None
    tco_config = core_config.decision_engine.tco
    assert tco_config is not None
    assert tco_config.require_at_startup is True
    assert len(tco_config.report_paths) == 2
    for path in tco_config.report_paths:
        assert Path(path).is_absolute()
    assert tco_config.report_paths[0].endswith("data/tco/baseline.json")
    assert tco_config.report_paths[1].endswith("shared/tco.json")
    assert tco_config.warn_report_age_hours == 12.0
    assert tco_config.max_report_age_hours == 36.0
