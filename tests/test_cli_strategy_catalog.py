import copy
from argparse import Namespace
from pathlib import Path

import pytest

from bot_core.cli import show_strategy_catalog


def _build_args(**overrides):
    defaults = {
        "output_format": "text",
        "engines": [],
        "capabilities": [],
        "tags": [],
        "config": None,
        "scheduler": None,
        "include_parameters": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_show_strategy_catalog_prints_engine_metadata(capfd) -> None:
    args = _build_args()
    assert show_strategy_catalog(args) == 0
    captured = capfd.readouterr().out
    assert "capability=" in captured
    assert "license=" in captured
    assert "risk_classes=[" in captured
    assert "required_data=[" in captured


def test_show_strategy_catalog_prints_definition_metadata(capfd) -> None:
    config_path = Path("config/core.yaml")
    args = _build_args(config=str(config_path))
    assert show_strategy_catalog(args) == 0
    captured = capfd.readouterr().out
    assert "Definicje strategii:" in captured
    assert "license=" in captured
    assert "required_data=" in captured


def test_show_strategy_catalog_reports_blocked_entries(
    monkeypatch: pytest.MonkeyPatch, capfd
) -> None:
    config_path = Path("config/core.yaml")
    plan = {
        "config_path": str(config_path),
        "scheduler": "demo",
        "schedules": [
            {
                "name": "trend-plan",
                "strategy": "trend",
            }
        ],
        "strategies": [
            {
                "name": "trend",
                "engine": "daily_trend_momentum",
                "tags": ["trend"],
                "capability": "trend_d1",
                "license_tier": "standard",
                "risk_classes": ["directional"],
                "required_data": ["ohlcv"],
            }
        ],
        "blocked_schedules": ["blocked-schedule"],
        "blocked_strategies": ["blocked-strategy"],
        "blocked_capabilities": {"blocked-strategy": "scalping"},
        "blocked_schedule_capabilities": {"blocked-schedule": "scalping"},
        "blocked_initial_signal_limits": {"blocked-strategy": ["balanced"]},
        "blocked_initial_signal_limit_capabilities": {"blocked-strategy": "scalping"},
        "blocked_signal_limits": {"blocked-strategy": ["aggressive"]},
        "blocked_signal_limit_capabilities": {"blocked-strategy": "scalping"},
        "blocked_suspensions": [
            {
                "kind": "schedule",
                "target": "blocked-schedule",
                "reason": "license",
                "capability": "scalping",
            }
        ],
        "blocked_suspension_capabilities": {"schedule:blocked-schedule": "scalping"},
    }

    monkeypatch.setattr(
        "bot_core.cli.describe_multi_strategy_configuration",
        lambda **kwargs: copy.deepcopy(plan),
    )

    args = _build_args(config=str(config_path), scheduler="demo")
    assert show_strategy_catalog(args) == 0
    captured = capfd.readouterr().out
    assert "Pominięte przez strażnika licencji:" in captured
    assert "blocked-schedule" in captured
    assert "blocked-strategy" in captured
    assert "capability:" in captured
    assert "Limity sygnałów (początkowe" in captured
    assert "aggressive" in captured
    assert "license" in captured
