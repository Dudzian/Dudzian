import copy
from argparse import Namespace

import pytest

from bot_core.cli import show_scheduler_plan


def _build_args(**overrides):
    defaults = {
        "config": "config/core.yaml",
        "scheduler": "demo",
        "output_format": "text",
        "filter_tags": [],
        "filter_strategies": [],
        "include_definitions": True,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_show_scheduler_plan_reports_blocked_entries(
    monkeypatch: pytest.MonkeyPatch, capfd
) -> None:
    plan = {
        "config_path": "config/core.yaml",
        "scheduler": "demo",
        "capital_policy": {"name": "demo"},
        "schedules": [
            {
                "name": "trend-run",
                "strategy": "trend",
                "risk_profile": "balanced",
                "cadence_seconds": 60,
                "max_drift_seconds": 15,
                "max_signals": 3,
                "interval": "5m",
                "license_tier": "standard",
                "risk_classes": ["directional"],
                "required_data": ["ohlcv"],
                "tags": ["trend"],
            }
        ],
        "strategies": [
            {
                "name": "trend",
                "engine": "daily_trend_momentum",
                "capability": "trend_d1",
                "license_tier": "standard",
                "risk_classes": ["directional"],
                "required_data": ["ohlcv"],
                "tags": ["trend"],
            }
        ],
        "initial_suspensions": [],
        "initial_signal_limits": {},
        "signal_limits": {},
        "blocked_schedules": ["blocked-schedule"],
        "blocked_strategies": ["blocked-strategy"],
        "blocked_capabilities": {"blocked-strategy": "scalping"},
        "blocked_capability_reasons": {
            "blocked-strategy": "Strategia 'scalping' nie jest dostępna."
        },
        "blocked_schedule_capabilities": {"blocked-schedule": "scalping"},
        "blocked_schedule_capability_reasons": {
            "blocked-schedule": "Strategia 'scalping' nie jest dostępna."
        },
        "blocked_initial_signal_limits": {"blocked-strategy": ["balanced"]},
        "blocked_initial_signal_limit_capabilities": {"blocked-strategy": "scalping"},
        "blocked_initial_signal_limit_reasons": {
            "blocked-strategy": "Strategia 'scalping' nie jest dostępna."
        },
        "blocked_signal_limits": {"blocked-strategy": ["aggressive"]},
        "blocked_signal_limit_capabilities": {"blocked-strategy": "scalping"},
        "blocked_signal_limit_reasons": {
            "blocked-strategy": "Strategia 'scalping' nie jest dostępna."
        },
        "blocked_suspensions": [
            {
                "kind": "schedule",
                "target": "blocked-schedule",
                "reason": "license",
                "capability": "scalping",
                "guard_reason": "Strategia 'scalping' nie jest dostępna.",
            }
        ],
        "blocked_suspension_capabilities": {"schedule:blocked-schedule": "scalping"},
        "blocked_suspension_reasons": {
            "schedule:blocked-schedule": "Strategia 'scalping' nie jest dostępna."
        },
    }

    monkeypatch.setattr(
        "bot_core.cli.describe_multi_strategy_configuration",
        lambda **kwargs: copy.deepcopy(plan),
    )

    args = _build_args()
    assert show_scheduler_plan(args) == 0
    captured = capfd.readouterr().out
    assert "Pominięte przez strażnika licencji:" in captured
    assert "blocked-schedule" in captured
    assert "blocked-strategy" in captured
    assert "capability:" in captured
    assert "Limity sygnałów (początkowe" in captured
    assert "aggressive" in captured
    assert "license" in captured
    assert "Strategia 'scalping' nie jest dostępna." in captured
    assert "Podsumowanie blokad strażnika:" in captured
    assert "Ogółem: 5" in captured
    assert "Szczegóły blokad wg capability:" in captured
    assert "capability scalping" in captured
    assert "Strategie:" in captured
    assert "Podsumowanie blokad wg capability:" in captured
    assert "powody: Strategia 'scalping' nie jest dostępna. (5)" in captured
    assert "Podsumowanie blokad wg kategorii:" in captured
    assert "- Strategie: 1" in captured
    assert "capabilities: scalping (1)" in captured
    assert "Podsumowanie blokad wg capability i powodów:" in captured
    assert "  - scalping:" in captured
    assert "    - Strategia 'scalping' nie jest dostępna.: 5" in captured
    assert "Podsumowanie blokad wg powodów:" in captured
    assert "Strategia 'scalping' nie jest dostępna.: 5" in captured
    assert "kategorie: Strategie (1), Harmonogramy (1), Limity sygnałów (początkowe) (1), Limity sygnałów (1), Zawieszenia (1)" in captured
    assert "Szczegóły blokad wg powodów:" in captured
    assert "- Strategia 'scalping' nie jest dostępna.: 5" in captured
