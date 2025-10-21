import io
import json
from pathlib import Path

import pytest
import yaml

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

