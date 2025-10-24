import json
from types import SimpleNamespace

from scripts.local_orchestrator import EnvironmentDefinition, cmd_status


def _build_fake_core_config() -> SimpleNamespace:
    definition_cfg = SimpleNamespace(
        name="trend_strategy",
        engine="daily_trend_momentum",
        parameters={"fast_ma": 12},
        license_tier="",  # wymusi backfill z katalogu
        risk_classes=(),
        required_data=(),
        capability=None,
        risk_profile="balanced",
        tags=("demo",),
        metadata={},
    )
    schedule_cfg = SimpleNamespace(
        name="demo_scheduler",
        schedules=[
            SimpleNamespace(
                name="trend_schedule",
                strategy="trend_strategy",
                cadence_seconds=900,
                max_drift_seconds=45,
                warmup_bars=10,
                risk_profile="balanced",
                max_signals=20,
            )
        ],
    )
    return SimpleNamespace(
        strategy_definitions={"trend_strategy": definition_cfg},
        multi_strategy_schedulers={"demo_sched": schedule_cfg},
    )


def test_cmd_status_includes_strategy_metadata(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "scripts.local_orchestrator.load_core_config",
        lambda path: _build_fake_core_config(),
    )

    env = EnvironmentDefinition(
        name="demo",
        environment_key="demo",
        config_path=tmp_path / "core.yaml",
        scheduler_name="demo_sched",
    )

    cmd_status(SimpleNamespace(), {"demo": env}, state_path=tmp_path / "state.json")

    output = capsys.readouterr().out
    report = json.loads(output)
    env_report = report["demo"]

    strategies = env_report["strategies"]
    assert strategies
    strategy_payload = strategies[0]
    assert strategy_payload["license_tier"] == "standard"
    assert "directional" in strategy_payload["risk_classes"]
    assert strategy_payload["capability"] == "trend_d1"

    scheduler = env_report["scheduler_plan"]
    assert scheduler["name"] == "demo_scheduler"
    schedule_entry = scheduler["strategies"][0]
    assert schedule_entry["capability"] == "trend_d1"
    assert "technical_indicators" in schedule_entry["required_data"]
    assert scheduler["license_tiers"] == ["standard"]
    assert scheduler["capabilities"] == ["trend_d1"]

    assert env_report["license_tiers"] == ["standard"]
    assert env_report["capabilities"] == ["trend_d1"]
