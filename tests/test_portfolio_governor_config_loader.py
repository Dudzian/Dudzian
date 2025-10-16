from pathlib import Path

import yaml
import pytest

from bot_core.config.loader import load_core_config


def test_load_core_config_parses_portfolio_governor_section(tmp_path: Path) -> None:
    config_data = {
        "risk_profiles": {
            "balanced": {
                "max_daily_loss_pct": 0.02,
                "max_position_pct": 0.1,
                "target_volatility": 0.1,
                "max_leverage": 3.0,
                "stop_loss_atr_multiple": 1.5,
                "max_open_positions": 5,
                "hard_drawdown_pct": 0.15,
            }
        },
        "alerts": {},
        "environments": {
            "paper_test": {
                "exchange": "binance_spot",
                "environment": "paper",
                "keychain_key": "binance_key",
                "credential_purpose": "trading",
                "data_cache_path": "./cache",
                "risk_profile": "balanced",
                "alert_channels": [],
                "ip_allowlist": [],
                "required_permissions": [],
                "forbidden_permissions": [],
            }
        },
        "portfolio_governor": {
            "enabled": True,
            "rebalance_interval_minutes": 20,
            "smoothing": 0.75,
            "default_baseline_weight": 0.4,
            "default_min_weight": 0.1,
            "default_max_weight": 0.7,
            "require_complete_metrics": True,
            "min_score_threshold": 0.05,
            "default_cost_bps": 2.5,
            "max_signal_floor": 2,
            "scoring": {
                "alpha": 1.0,
                "cost": 0.8,
                "slo": 1.1,
                "risk": 0.4,
            },
            "strategies": {
                "trend": {
                    "baseline_weight": 0.5,
                    "min_weight": 0.2,
                    "max_weight": 0.7,
                    "baseline_max_signals": 4,
                    "max_signal_factor": 1.4,
                    "risk_profile": "balanced",
                    "tags": ["core", "trend"],
                },
                "mean_reversion": {
                    "baseline_weight": 0.3,
                    "min_weight": 0.1,
                    "max_weight": 0.5,
                    "baseline_max_signals": 3,
                    "max_signal_factor": 1.2,
                },
            },
        },
    }

    config_path = tmp_path / "core.yaml"
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    core_config = load_core_config(config_path)

    governor_config = core_config.portfolio_governor
    assert governor_config is not None
    assert governor_config.enabled is True
    assert governor_config.rebalance_interval_minutes == pytest.approx(20.0)
    assert governor_config.smoothing == pytest.approx(0.75)
    assert governor_config.default_baseline_weight == pytest.approx(0.4)
    assert governor_config.max_signal_floor == 2
    assert governor_config.scoring.alpha == pytest.approx(1.0)
    assert governor_config.scoring.cost == pytest.approx(0.8)
    assert set(governor_config.strategies) == {"trend", "mean_reversion"}

    trend_cfg = governor_config.strategies["trend"]
    assert trend_cfg.baseline_weight == pytest.approx(0.5)
    assert trend_cfg.baseline_max_signals == 4
    assert trend_cfg.tags == ("core", "trend")

    mean_cfg = governor_config.strategies["mean_reversion"]
    assert mean_cfg.baseline_max_signals == 3
    assert mean_cfg.risk_profile is None
