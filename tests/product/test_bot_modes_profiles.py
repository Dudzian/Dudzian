from __future__ import annotations

import ast
from pathlib import Path

import json
import pytest

from bot_core.config.models import RuntimeExecutionLiveSettings, RuntimeExecutionSettings
from bot_core.execution.mode_policy import ExecutionModeSelector
from bot_core.product.bot_modes import builtin_bot_modes, load_bot_mode_profiles


class _Env:
    def __init__(self, *, environment: str, offline_mode: bool = False) -> None:
        self.environment = environment
        self.offline_mode = offline_mode


def _autotrade_preset_engines() -> set[str]:
    source = Path("bot_core/trading/auto_trade.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_PRESET_ENGINE_MAPPING":
                    mapping = ast.literal_eval(node.value)
                    return {str(value) for value in mapping.values()}
    raise AssertionError("Nie znaleziono _PRESET_ENGINE_MAPPING")


def test_builtin_profiles_are_bootstrapable_and_complete() -> None:
    profiles = {profile.id: profile for profile in builtin_bot_modes()}

    assert set(profiles) == {"signal_grid", "paper_monitoring", "rule_auto_router"}
    assert profiles["signal_grid"].execution_mode == "paper"
    assert profiles["rule_auto_router"].execution_mode == "auto"


def test_builtin_profiles_work_outside_repo_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    profiles = builtin_bot_modes()

    assert len(profiles) == 3
    assert {profile.id for profile in profiles} == {"signal_grid", "paper_monitoring", "rule_auto_router"}


def test_signal_grid_uses_engine_available_in_runtime_contract() -> None:
    profiles = {profile.id: profile for profile in builtin_bot_modes()}
    runtime_engines = _autotrade_preset_engines()

    assert profiles["signal_grid"].strategy_engine == "grid_trading"
    assert profiles["signal_grid"].strategy_engine in runtime_engines


def test_rule_auto_router_and_paper_monitoring_follow_mode_policy() -> None:
    selector = ExecutionModeSelector()
    profiles = {profile.id: profile for profile in builtin_bot_modes()}

    paper_settings = RuntimeExecutionSettings(default_mode=profiles["paper_monitoring"].execution_mode)
    assert selector.resolve(paper_settings, _Env(environment="paper")) == "paper"

    auto_settings = RuntimeExecutionSettings(
        default_mode=profiles["rule_auto_router"].execution_mode,
        live=RuntimeExecutionLiveSettings(enabled=True),
        force_paper_when_offline=True,
    )
    assert selector.resolve(auto_settings, _Env(environment="live")) == "live"
    assert selector.resolve(auto_settings, _Env(environment="live", offline_mode=True)) == "paper"


def test_loader_rejects_duplicate_profile_ids(tmp_path: Path) -> None:
    payload = {
        "id": "duplicate",
        "label": "Duplicate",
        "execution_mode": "paper",
        "description": "test",
    }
    (tmp_path / "a.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Zduplikowany identyfikator"):
        load_bot_mode_profiles(tmp_path)


def test_loader_rejects_missing_required_fields(tmp_path: Path) -> None:
    broken = {"id": "broken", "execution_mode": "paper", "description": "x"}
    (tmp_path / "broken.json").write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ValueError, match="polach obowiązkowych"):
        load_bot_mode_profiles(tmp_path)


def test_builtin_loader_fails_when_resource_set_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    import bot_core.product.bot_modes as bot_modes_module

    class _BrokenTraversable:
        def joinpath(self, name: str):
            if name == "profiles":
                return self
            raise FileNotFoundError(name)

        def iterdir(self):
            return iter(())

    monkeypatch.setattr(bot_modes_module.resources, "files", lambda _pkg: _BrokenTraversable())

    with pytest.raises(RuntimeError, match="Brak kompletnego zestawu"):
        bot_modes_module.builtin_bot_modes()
