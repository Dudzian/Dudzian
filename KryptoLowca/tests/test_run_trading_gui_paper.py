from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_core.runtime.metadata import RiskManagerSettings

import KryptoLowca.run_trading_gui_paper as paper_launcher
from KryptoLowca.ui.trading.risk_helpers import RiskSnapshot


def test_derive_risk_defaults_respects_runtime_settings() -> None:
    snapshot = RiskSnapshot(
        paper_balance=50_000.0,
        profile_name="balanced",
        settings=RiskManagerSettings(
            max_risk_per_trade=0.04,
            max_daily_loss_pct=0.12,
            max_portfolio_risk=0.25,
            max_positions=6,
            emergency_stop_drawdown=0.2,
        ),
    )

    capital, risk_pct, portfolio_pct, notional = paper_launcher._derive_risk_defaults(snapshot)

    assert capital == pytest.approx(50_000.0)
    assert risk_pct == pytest.approx(4.0)
    assert portfolio_pct == pytest.approx(25.0)
    assert notional == pytest.approx(2_000.0)


def test_derive_risk_defaults_fallback_to_defaults() -> None:
    snapshot = RiskSnapshot(paper_balance=0.0, profile_name=None, settings=None)

    capital, risk_pct, portfolio_pct, notional = paper_launcher._derive_risk_defaults(snapshot)

    assert capital == pytest.approx(paper_launcher.DEFAULT_CAPITAL_USDT)
    assert risk_pct == pytest.approx(paper_launcher.DEFAULT_RISK_PCT)
    assert portfolio_pct == pytest.approx(paper_launcher.DEFAULT_PORTFOLIO_PCT)
    assert notional == pytest.approx(paper_launcher.DEFAULT_NOTIONAL_USDT)


def test_build_frontend_bootstrap_uses_runtime_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paper_launcher, "_FRONTEND_PATHS", None)

    captured: dict[str, object] = {}
    dummy_paths = SimpleNamespace(app_root=Path("/tmp/paper"), db_file=Path("/tmp/paper.db"))

    def fake_build(app_file: Path, *, logs_dir, text_log_file):  # type: ignore[override]
        captured["build_args"] = (app_file, logs_dir, text_log_file)
        return dummy_paths

    dummy_services = paper_launcher.FrontendBootstrap(exchange_manager="mgr", market_intel="intel")

    def fake_bootstrap(**kwargs: object) -> paper_launcher.FrontendBootstrap:
        captured["bootstrap_kwargs"] = kwargs
        return dummy_services

    monkeypatch.setattr(paper_launcher, "build_desktop_app_paths", fake_build)
    monkeypatch.setattr(paper_launcher, "bootstrap_frontend_services", fake_bootstrap)

    paths, services = paper_launcher._build_frontend_bootstrap(core_config_path="cfg.yaml")

    assert paths is dummy_paths
    assert services is dummy_services
    assert captured["bootstrap_kwargs"] == {
        "paths": dummy_paths,
        "config_path": "cfg.yaml",
        "environment": None,
    }
