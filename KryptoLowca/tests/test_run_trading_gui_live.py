from types import SimpleNamespace

import pytest

import KryptoLowca.run_trading_gui_live as launcher
from bot_core.runtime.metadata import RiskManagerSettings


def test_compute_default_notional_uses_risk_profile() -> None:
    app = SimpleNamespace(
        paper_balance=25_000.0,
        risk_manager_settings=RiskManagerSettings(
            max_risk_per_trade=0.04,
            max_daily_loss_pct=0.12,
            max_portfolio_risk=0.25,
            max_positions=6,
            emergency_stop_drawdown=0.2,
        ),
    )

    value = launcher._compute_default_notional(app)  # type: ignore[arg-type]
    assert value == pytest.approx(1_000.0)


def test_compute_default_notional_falls_back_to_default() -> None:
    app = SimpleNamespace(paper_balance=5_000.0, risk_manager_settings=None)

    value = launcher._compute_default_notional(app)  # type: ignore[arg-type]
    assert value == pytest.approx(launcher.DEFAULT_NOTIONAL_USDT)


class DummyGUI:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.calls: list[tuple[str, str, float]] = []
        self.logged: list[tuple[str, str]] = []
        self.should_fail = should_fail

    def _bridge_execute_trade(self, symbol: str, side: str, price: float) -> None:
        if self.should_fail:
            raise RuntimeError("boom")
        self.calls.append((symbol, side, price))

    def _log(self, message: str, level: str = "INFO") -> None:
        self.logged.append((level, message))


def test_paper_trade_executor_delegates_to_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    gui = DummyGUI()
    captured: dict[str, tuple[str, str]] = {}

    monkeypatch.setattr(launcher.messagebox, "showerror", lambda title, msg: captured.setdefault("error", (title, msg)))

    launcher._paper_trade_executor(gui, "ETH/USDT", "buy", 2_000.0)

    assert gui.calls == [("ETH/USDT", "buy", 2_000.0)]
    assert "error" not in captured


def test_paper_trade_executor_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    gui = DummyGUI(should_fail=True)
    captured: dict[str, tuple[str, str]] = {}

    monkeypatch.setattr(launcher.messagebox, "showerror", lambda title, msg: captured.setdefault("error", (title, msg)))
    monkeypatch.setattr(launcher.traceback, "format_exc", lambda: "traceback")

    launcher._paper_trade_executor(gui, "BTC/USDT", "sell", 25_000.0)

    assert captured["error"][0] == "Paper"
    assert "boom" in captured["error"][1]
    assert any(level == "ERROR" for level, _ in gui.logged)
