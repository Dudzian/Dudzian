from __future__ import annotations

from pathlib import Path
import time
from textwrap import dedent

import pytest

from KryptoLowca.paper_auto_trade_app import HeadlessTradingStub, PaperAutoTradeApp


def _write_core_config(path: Path, *, max_daily_loss: float) -> None:
    path.write_text(
        dedent(
            f"""
            risk_profiles:
              balanced:
                max_daily_loss_pct: {max_daily_loss}
                max_position_pct: 0.05
                target_volatility: 0.1
                max_leverage: 3.0
                stop_loss_atr_multiple: 1.5
                max_open_positions: 5
                hard_drawdown_pct: 0.08
            environments:
              paper_env:
                name: paper_env
                exchange: binance_spot
                environment: paper
                keychain_key: dummy
                data_cache_path: ./cache
                risk_profile: balanced
                alert_channels: []
                required_permissions: []
                forbidden_permissions: []
            """
        ).strip()
    )


def test_reload_updates_headless_stub(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(gui=None, headless_stub=stub, core_config_path=core_path, core_environment="paper_env")

    profile, settings, _ = app.reload_risk_settings()
    assert profile == "balanced"
    assert settings["max_daily_loss_pct"] == pytest.approx(0.02)
    assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.02)
    first_update = stub.update_count

    _write_core_config(core_path, max_daily_loss=0.03)
    assert app.handle_cli_command(f"reload-risk config={core_path} env=paper_env")
    assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.03)
    assert stub.update_count == first_update + 1


def test_app_uses_gui_reload() -> None:
    responses = [
        ("balanced", {"max_daily_loss_pct": 0.02, "max_portfolio_risk": 0.08}, object()),
    ]

    class _DummyGUI:
        def __init__(self) -> None:
            self.calls: list[tuple[object, str | None]] = []
            self.core_config_path: Path | None = None
            self.core_environment: str | None = None

        def reload_risk_manager_settings(self, *, config_path=None, environment: str | None = None):
            self.calls.append((config_path, environment))
            if config_path is not None:
                self.core_config_path = Path(config_path).expanduser().resolve()
            if environment:
                self.core_environment = environment
            elif self.core_environment is None:
                self.core_environment = "paper_env"
            return responses.pop(0)

    gui = _DummyGUI()
    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(gui=gui, headless_stub=stub, core_environment="paper_env")

    profile, settings, _ = app.reload_risk_settings()
    assert profile == "balanced"
    assert settings["max_daily_loss_pct"] == pytest.approx(0.02)
    assert len(gui.calls) == 1
    cfg_path, env_name = gui.calls[0]
    assert env_name == "paper_env"
    assert Path(cfg_path) == app.core_config_path
    assert app.core_environment == "paper_env"
    assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.02)


def test_listeners_receive_updates(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    app = PaperAutoTradeApp(core_config_path=core_path, core_environment="paper_env")
    received: list[tuple[dict[str, float], str | None]] = []

    def _listener(settings, profile_name, _profile_cfg) -> None:
        received.append((dict(settings), profile_name))

    app.add_listener(_listener)

    profile, settings, _ = app.reload_risk_settings()
    assert profile == "balanced"
    assert settings["max_daily_loss_pct"] == pytest.approx(0.02)
    assert received
    stored_settings, stored_profile = received[-1]
    assert stored_profile == "balanced"
    assert stored_settings["max_daily_loss_pct"] == pytest.approx(0.02)


def test_auto_reload_detects_changes(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(core_config_path=core_path, core_environment="paper_env", headless_stub=stub)

    profile, settings, _ = app.reload_risk_settings()
    assert profile == "balanced"
    assert settings["max_daily_loss_pct"] == pytest.approx(0.02)
    first_updates = stub.update_count

    app.start_auto_reload(interval=0.05)
    try:
        _write_core_config(core_path, max_daily_loss=0.03)

        deadline = time.time() + 2.0
        while time.time() < deadline and stub.update_count < first_updates + 1:
            time.sleep(0.05)

        assert stub.update_count >= first_updates + 1
        assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.03)
    finally:
        app.stop_auto_reload()


def test_auto_reload_recovers_after_missing_file(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(core_config_path=core_path, core_environment="paper_env", headless_stub=stub)

    profile, settings, _ = app.reload_risk_settings()
    assert profile == "balanced"
    assert settings["max_daily_loss_pct"] == pytest.approx(0.02)
    first_updates = stub.update_count

    app.start_auto_reload(interval=0.05)
    try:
        try:
            core_path.unlink()
        except FileNotFoundError:
            pass

        deadline = time.time() + 2.0
        while time.time() < deadline and app._watch_last_mtime is not None:
            time.sleep(0.05)

        _write_core_config(core_path, max_daily_loss=0.04)

        deadline = time.time() + 2.0
        while time.time() < deadline and stub.update_count < first_updates + 1:
            time.sleep(0.05)

        assert stub.update_count >= first_updates + 1
        assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.04)
    finally:
        app.stop_auto_reload()


def test_cli_parsing_with_flags_and_quotes(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(core_config_path=core_path, core_environment="paper_env", headless_stub=stub)
    app.reload_risk_settings()

    fancy_dir = tmp_path / "alt config"
    fancy_dir.mkdir()
    alt_path = fancy_dir / "core.yaml"
    _write_core_config(alt_path, max_daily_loss=0.05)

    quoted_command = f'reload-risk --config "{alt_path}" --env paper_env'
    assert app.handle_cli_command(quoted_command)
    assert app.core_config_path == alt_path.resolve()
    assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.05)

    _write_core_config(alt_path, max_daily_loss=0.06)
    flag_command = f'reload-risk -c "{alt_path}" -e paper_env'
    assert app.handle_cli_command(flag_command)
    assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.06)


def test_reload_skips_duplicate_updates(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(core_config_path=core_path, core_environment="paper_env", headless_stub=stub)

    app.reload_risk_settings()
    first_count = stub.update_count
    first_signature = app._last_signature
    first_timestamp = app.last_reload_at

    assert app.handle_cli_command(f"reload-risk --config {core_path} --env paper_env")

    assert stub.update_count == first_count
    assert app._last_signature == first_signature
    assert app.last_reload_at is not None
    assert app.last_reload_at >= first_timestamp
    assert app.reload_count == 2


def test_auto_reload_ignores_duplicate_content(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(core_config_path=core_path, core_environment="paper_env", headless_stub=stub)

    app.reload_risk_settings()
    first_count = stub.update_count

    app.start_auto_reload(interval=0.05)
    try:
        # Zapisz tę samą konfigurację jeszcze raz, aby wywołać zmianę mtime bez zmiany ustawień.
        _write_core_config(core_path, max_daily_loss=0.02)

        deadline = time.time() + 1.5
        while time.time() < deadline and stub.update_count == first_count:
            time.sleep(0.05)

        assert stub.update_count == first_count
        assert app.reload_count >= 2
    finally:
        app.stop_auto_reload()
