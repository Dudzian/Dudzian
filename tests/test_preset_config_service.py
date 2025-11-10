from pathlib import Path

import pytest
import yaml

from bot_core.config.models import RiskServiceConfig
from bot_core.runtime.preset_service import PresetConfigService


@pytest.fixture()
def core_config_copy(tmp_path: Path) -> Path:
    source = Path("config/core.yaml")
    target = tmp_path / "core.yaml"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_import_gui_preset_updates_core_config(core_config_copy: Path) -> None:
    preset = {
        "fraction": 0.35,
        "risk": {
            "max_daily_loss_pct": 0.04,
            "risk_per_trade": 0.12,
            "portfolio_risk": 0.18,
            "max_open_positions": 6,
            "max_leverage": 3.5,
            "hard_drawdown_pct": 0.22,
        },
    }
    service = PresetConfigService(core_config_copy)
    service.core_config.risk_service = RiskServiceConfig(
        enabled=True,
        host="127.0.0.1",
        port=9100,
        history_size=128,
        profiles=("balanced",),
    )
    profile = service.import_gui_preset(preset, profile_name="Stage6 GUI")

    assert profile.name == "stage6_gui"
    assert pytest.approx(profile.max_daily_loss_pct, rel=1e-6) == 0.04
    assert pytest.approx(profile.max_position_pct, rel=1e-6) == 0.12
    assert pytest.approx(profile.target_volatility, rel=1e-6) == 0.18
    assert pytest.approx(profile.max_leverage, rel=1e-6) == 3.5
    assert pytest.approx(profile.hard_drawdown_pct, rel=1e-6) == 0.22

    updated_entrypoint = service.core_config.runtime_entrypoints["trading_gui"]
    assert updated_entrypoint.risk_profile == "stage6_gui"

    decision_override = service.core_config.decision_engine.profile_overrides["stage6_gui"]
    assert pytest.approx(decision_override.max_daily_loss_pct, rel=1e-6) == 0.04
    assert pytest.approx(decision_override.max_position_ratio, rel=1e-6) == 0.12
    assert decision_override.max_open_positions == 6
    assert pytest.approx(decision_override.max_drawdown_pct, rel=1e-6) == 0.22

    risk_profiles = service.core_config.risk_service.profiles
    assert risk_profiles[-1] == "stage6_gui"

    rendered = service.save(dry_run=True)
    assert "stage6_gui" in rendered

    payload = yaml.safe_load(rendered)
    budgets = payload["portfolio_governors"]["stage6_core"]["risk_budgets"]
    assert "stage6_gui" in budgets
    assert pytest.approx(budgets["stage6_gui"]["max_drawdown_pct"], rel=1e-6) == 0.22
    assert pytest.approx(budgets["stage6_gui"]["max_leverage"], rel=1e-6) == 3.5
