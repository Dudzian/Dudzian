from pathlib import Path

import json

import pytest
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
    report_paths = getattr(tco_config, "report_paths", None)
    if not report_paths:
        report_paths = getattr(tco_config, "reports", ())
    assert len(report_paths) == 2
    for path in report_paths:
        assert Path(path).is_absolute()
    assert report_paths[0].endswith("data/tco/baseline.json")
    assert report_paths[1].endswith("shared/tco.json")


def test_load_core_config_maps_thresholds_and_overrides(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_path = reports_dir / "tco_summary.json"
    report_path.write_text(
        json.dumps(
            {
                "total": {"cost_bps": 6.5},
                "strategies": {
                    "core_daily_trend": {
                        "total": {"cost_bps": 7.1},
                        "profiles": {"balanced": {"cost_bps": 7.1}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config_data = {
        "risk_profiles": {
            "balanced": {
                "max_daily_loss_pct": 0.015,
                "max_position_pct": 0.05,
                "target_volatility": 0.11,
                "max_leverage": 3.0,
                "stop_loss_atr_multiple": 1.5,
                "max_open_positions": 5,
                "hard_drawdown_pct": 0.1,
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
                "risk_profile": "balanced",
                "alert_channels": [],
                "ip_allowlist": [],
                "required_permissions": [],
                "forbidden_permissions": [],
            }
        },
        "decision_engine": {
            "orchestrator": {
                "max_cost_bps": 18.0,
                "min_net_edge_bps": 4.5,
                "max_daily_loss_pct": 0.025,
                "max_drawdown_pct": 0.12,
                "max_position_ratio": 0.35,
                "max_open_positions": 6,
                "max_latency_ms": 240.0,
            },
            "profile_overrides": {
                "balanced": {
                    "max_cost_bps": 16.0,
                    "max_latency_ms": 200.0,
                    "max_trade_notional": 15000.0,
                }
            },
            "stress_tests": {
                "cost_shock_bps": 3.5,
                "latency_spike_ms": 75.0,
                "slippage_multiplier": 1.25,
            },
            "min_probability": 0.6,
            "require_cost_data": True,
            "penalty_cost_bps": 1.75,
            "tco": {
                "reports": [f"reports/{report_path.name}"],
            },
        },
    }

    config_path = tmp_path / "core.yaml"
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    core_config = load_core_config(config_path)
    decision_config = core_config.decision_engine
    assert decision_config is not None

    orchestrator = decision_config.orchestrator
    assert orchestrator.max_cost_bps == pytest.approx(18.0)
    assert orchestrator.max_latency_ms == pytest.approx(240.0)

    overrides = decision_config.profile_overrides
    assert "balanced" in overrides
    balanced = overrides["balanced"]
    # values not provided in override should fall back to base orchestrator thresholds
    assert balanced.max_drawdown_pct == pytest.approx(orchestrator.max_drawdown_pct)
    assert balanced.max_latency_ms == pytest.approx(200.0)
    assert balanced.max_trade_notional == pytest.approx(15000.0)
    assert balanced.min_net_edge_bps == pytest.approx(orchestrator.min_net_edge_bps)

    stress = decision_config.stress_tests
    assert stress is not None
    assert stress.cost_shock_bps == pytest.approx(3.5)
    assert stress.latency_spike_ms == pytest.approx(75.0)
    assert stress.slippage_multiplier == pytest.approx(1.25)

    assert decision_config.min_probability == pytest.approx(0.6)
    assert decision_config.require_cost_data is True
    assert decision_config.penalty_cost_bps == pytest.approx(1.75)

    tco_config = decision_config.tco
    assert tco_config is not None
    normalized_paths = getattr(tco_config, "report_paths", None) or getattr(tco_config, "reports", ())
    assert len(normalized_paths) == 1
    assert Path(normalized_paths[0]).resolve() == report_path.resolve()
