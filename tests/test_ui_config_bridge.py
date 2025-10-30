import io
import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.guards import install_capability_guard, reset_capability_guard

from scripts import ui_config_bridge


@pytest.fixture()
def sample_config(tmp_path: Path) -> Path:
    content = {
        "decision_engine": {
            "orchestrator": {
                "max_cost_bps": 12,
                "min_net_edge_bps": 7,
                "max_daily_loss_pct": 2.5,
                "max_drawdown_pct": 4.1,
                "max_position_ratio": 0.3,
                "max_open_positions": 5,
                "max_latency_ms": 120,
                "max_trade_notional": 150000,
                "stress_tests": [
                    {
                        "name": "latency",
                        "type": "latency",
                        "parameters": {"max_ms": 150},
                    }
                ],
            },
            "min_probability": 0.6,
            "require_cost_data": True,
            "penalty_cost_bps": 3,
            "profile_overrides": {
                "conservative": {
                    "max_cost_bps": 8,
                    "max_latency_ms": 90,
                }
            },
        },
        "strategies": {
            "trend-follow": {
                "engine": "daily_trend_momentum",
                "license_tier": "standard",
                "risk_classes": ["directional"],
                "required_data": ["ohlcv"],
                "tags": ["custom-tag"],
            }
        },
        "futures_spread_strategies": {
            "basis_guard": {
                "parameters": {"entry_z": 1.4, "exit_z": 0.35, "max_bars": 24}
            }
        },
        "cross_exchange_hedge_strategies": {
            "delta_guard": {
                "parameters": {"basis_scale": 0.009, "inventory_scale": 0.3}
            }
        },
        "multi_strategy_schedulers": {
            "default": {
                "telemetry_namespace": "telemetry/main",
                "decision_log_category": "decision.default",
                "health_check_interval": 120,
                "portfolio_governor": "governor-a",
                "schedules": [
                    {
                        "name": "open-session",
                        "strategy": "trend-follow",
                        "cadence_seconds": 60,
                        "max_drift_seconds": 5,
                        "warmup_bars": 120,
                        "risk_profile": "BALANCED",
                        "max_signals": 10,
                        "interval": "PT1M",
                    }
                ],
            }
        },
    }
    path = tmp_path / "core.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def test_dump_config_sections(sample_config: Path) -> None:
    raw = yaml.safe_load(sample_config.read_text(encoding="utf-8"))
    dumped = ui_config_bridge.dump_config(raw, section="all", scheduler=None)

    assert dumped["decision"]["max_cost_bps"] == 12
    assert dumped["decision"]["profile_overrides"] == [
        {
            "profile": "conservative",
            "max_cost_bps": 8,
            "max_latency_ms": 90,
        }
    ]

    assert list(dumped["schedulers"].keys()) == ["default"]
    default_scheduler = dumped["schedulers"]["default"]
    assert default_scheduler["telemetry_namespace"] == "telemetry/main"
    assert default_scheduler["schedules"][0]["name"] == "open-session"
    first_schedule = default_scheduler["schedules"][0]
    assert first_schedule["engine"] == "daily_trend_momentum"
    assert first_schedule["capability"] == "trend_d1"
    assert first_schedule["license_tier"] == "standard"
    assert first_schedule["risk_classes"] == ["directional", "momentum"]
    assert first_schedule["required_data"] == ["ohlcv", "technical_indicators"]
    assert first_schedule["tags"] == ["trend", "momentum", "custom-tag"]
    assert default_scheduler["initial_suspensions"] == []
    assert default_scheduler["initial_signal_limits"] == {}
    assert "blocked_schedules" not in default_scheduler
    assert "blocked_initial_signal_limits" not in default_scheduler


def test_apply_updates_writes_yaml(sample_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_load(path: Path) -> None:
        calls.append(path)

    monkeypatch.setattr(ui_config_bridge, "load_core_config", fake_load)

    payload = {
        "decision": {
            "max_cost_bps": 9,
            "profile_overrides": [
                {
                    "profile": "conservative",
                    "max_cost_bps": 6,
                    "max_latency_ms": 80,
                },
                {
                    "profile": "aggressive",
                    "max_open_positions": 12,
                },
            ],
        },
        "schedulers": {
            "default": {
                "health_check_interval": 90,
                "schedules": [
                    {
                        "name": "open-session",
                        "max_signals": 6,
                    }
                ],
            }
        },
    }

    ui_config_bridge.apply_updates(sample_config, payload)

    assert calls == [sample_config]

    updated = yaml.safe_load(sample_config.read_text(encoding="utf-8"))
    overrides = updated["decision_engine"]["profile_overrides"]
    assert overrides["conservative"]["max_latency_ms"] == 80
    assert overrides["aggressive"]["max_open_positions"] == 12

    schedules = updated["multi_strategy_schedulers"]["default"]["schedules"]
    assert schedules[0]["max_signals"] == 6
    assert updated["decision_engine"]["orchestrator"]["max_cost_bps"] == 9
    assert updated["multi_strategy_schedulers"]["default"]["health_check_interval"] == 90


def test_main_apply_reads_stdin(sample_config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[Path] = []

    def fake_load(path: Path) -> None:
        calls.append(path)

    monkeypatch.setattr(ui_config_bridge, "load_core_config", fake_load)

    update_payload = {
        "decision": {"max_open_positions": 7},
    }

    monkeypatch.setattr(ui_config_bridge.sys, "stdin", io.StringIO(json.dumps(update_payload)))

    exit_code = ui_config_bridge.main([
        "--config",
        str(sample_config),
        "--apply",
    ])

    assert exit_code == 0
    assert calls == [sample_config]

    updated = yaml.safe_load(sample_config.read_text(encoding="utf-8"))
    assert updated["decision_engine"]["orchestrator"]["max_open_positions"] == 7


def test_main_dump_outputs_json(sample_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = ui_config_bridge.main([
        "--config",
        str(sample_config),
        "--dump",
        "--section",
        "scheduler",
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "schedulers" in data
    assert list(data["schedulers"].keys()) == ["default"]


def test_dump_config_filters_blocked_capabilities() -> None:
    raw = {
        "strategies": {
            "trend-follow": {
                "engine": "daily_trend_momentum",
                "tags": ["custom"],
            }
        },
        "scalping_strategies": {
            "quick_scalp": {}
        },
        "multi_strategy_schedulers": {
            "default": {
                "schedules": [
                    {"name": "trend", "strategy": "trend-follow"},
                    {"name": "scalp", "strategy": "quick_scalp"},
                ]
            }
        },
    }

    try:
        capabilities = build_capabilities_from_payload(
            {
                "edition": "pro",
                "environments": ["paper"],
                "exchanges": {},
                "strategies": {"trend_d1": True, "scalping": False},
                "runtime": {},
                "modules": {},
                "limits": {},
            },
            effective_date=date(2025, 1, 1),
        )
        install_capability_guard(capabilities)

        dumped = ui_config_bridge.dump_config(raw, section="scheduler", scheduler=None)
        scheduler = dumped["schedulers"]["default"]
        schedules = scheduler["schedules"]

        assert len(schedules) == 1
        trend_entry = schedules[0]
        assert trend_entry["name"] == "trend"
        assert trend_entry["engine"] == "daily_trend_momentum"
        assert trend_entry["capability"] == "trend_d1"

        assert scheduler.get("blocked_schedules") == ["scalp"]
        assert scheduler.get("blocked_strategies") == ["quick_scalp"]
        assert scheduler.get("blocked_capabilities") == {"quick_scalp": "scalping"}
        assert scheduler.get("blocked_schedule_capabilities") == {"scalp": "scalping"}
        assert "blocked_initial_signal_limits" not in scheduler
        assert "blocked_signal_limits" not in scheduler
    finally:
        reset_capability_guard()


def test_dump_config_reports_blocked_limits_and_suspensions() -> None:
    raw = {
        "strategies": {
            "trend": {"engine": "daily_trend_momentum"},
            "blocked-strategy": {"engine": "scalping"},
        },
        "multi_strategy_schedulers": {
            "default": {
                "schedules": [
                    {"name": "trend-run", "strategy": "trend"},
                    {"name": "blocked-run", "strategy": "blocked-strategy"},
                ],
                "initial_signal_limits": {
                    "trend": {"balanced": {"limit": 2}},
                    "blocked-strategy": {"balanced": {"limit": 1}},
                },
                "signal_limits": {
                    "trend": {"balanced": {"limit": 3}},
                    "blocked-strategy": {"balanced": {"limit": 2}},
                },
                "initial_suspensions": [
                    {"kind": "schedule", "target": "blocked-run", "reason": "maintenance"},
                    {"kind": "tag", "target": "intraday", "reason": "tag-pause"},
                ],
            }
        },
    }

    try:
        capabilities = build_capabilities_from_payload(
            {
                "edition": "pro",
                "environments": ["paper"],
                "exchanges": {},
                "strategies": {"trend_d1": True, "scalping": False},
                "runtime": {},
                "modules": {},
                "limits": {},
            },
            effective_date=date(2025, 1, 1),
        )
        install_capability_guard(capabilities)

        dumped = ui_config_bridge.dump_config(raw, section="scheduler", scheduler=None)
        scheduler = dumped["schedulers"]["default"]

        schedules = scheduler["schedules"]
        assert len(schedules) == 1
        assert schedules[0]["strategy"] == "trend"

        assert scheduler["initial_signal_limits"] == {
            "trend": {"balanced": {"limit": 2}}
        }
        assert scheduler["signal_limits"] == {
            "trend": {"balanced": {"limit": 3}}
        }
        assert scheduler["initial_suspensions"] == [
            {"kind": "tag", "target": "intraday", "reason": "tag-pause"}
        ]
        assert scheduler["blocked_schedules"] == ["blocked-run"]
        assert scheduler["blocked_strategies"] == ["blocked-strategy"]
        assert scheduler["blocked_capabilities"] == {"blocked-strategy": "scalping"}
        assert scheduler["blocked_schedule_capabilities"] == {"blocked-run": "scalping"}
        assert scheduler["blocked_initial_signal_limits"] == {
            "blocked-strategy": ["balanced"]
        }
        assert scheduler["blocked_initial_signal_limit_capabilities"] == {
            "blocked-strategy": "scalping"
        }
        assert scheduler["blocked_signal_limits"] == {
            "blocked-strategy": ["balanced"]
        }
        assert scheduler["blocked_signal_limit_capabilities"] == {
            "blocked-strategy": "scalping"
        }
        assert scheduler["blocked_suspensions"] == [
            {
                "kind": "schedule",
                "target": "blocked-run",
                "reason": "maintenance",
                "capability": "scalping",
            }
        ]
        assert scheduler["blocked_suspension_capabilities"] == {
            "schedule:blocked-run": "scalping"
        }
    finally:
        reset_capability_guard()

