"""Testy kontrolera Trading GUI z naciskiem na konfigurację ExchangeManagera."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import pytest

from KryptoLowca.managers.exchange_manager import ExchangeManager, Mode
from KryptoLowca.ui.trading.controller import TradingSessionController


@dataclass
class _DummyVar:
    value: Any

    def get(self) -> Any:
        return self.value

    def set(self, new_value: Any) -> None:
        self.value = new_value


class _DummySecurity:
    def __init__(self, payload: Optional[Dict[str, str]] = None) -> None:
        self._payload = payload or {}
        self.calls: list[Tuple[str, ...]] = []

    def load_encrypted_keys(self, password: str) -> Dict[str, str]:
        self.calls.append((password,))
        return dict(self._payload)


def _build_state(
    *,
    network: str,
    mode: str = "Spot",
    paper_balance: str = "15 000",
    market_intel_label: Any | None = None,
    logs_dir: Optional[Path] = None,
) -> SimpleNamespace:
    intel_label = market_intel_label if market_intel_label is not None else _DummyVar("Market intel: —")
    resolved_logs = logs_dir or Path.cwd()
    return SimpleNamespace(
        network=_DummyVar(network),
        mode=_DummyVar(mode),
        paper_balance=_DummyVar(paper_balance),
        status=_DummyVar("Idle"),
        market_intel_label=intel_label,
        market_intel_summary="Market intel: —",
        market_intel_history=[],
        market_intel_history_label=None,
        market_intel_history_display="Brak historii market intel",
        market_intel_history_destination=None,
        market_intel_history_destination_display="Plik historii: domyślny",
        market_intel_history_path_label=_DummyVar("Plik historii: domyślny"),
        market_intel_auto_save=_DummyVar(False),
        running=False,
        open_positions={},
        risk_manager_settings=None,
        risk_manager_config=None,
        paths=SimpleNamespace(logs_dir=resolved_logs),
    )


def _build_controller(state: SimpleNamespace, security: _DummySecurity) -> TradingSessionController:
    return TradingSessionController(
        state,
        db=SimpleNamespace(),
        security=security,
        config=SimpleNamespace(),
        reporter=SimpleNamespace(),
        risk=SimpleNamespace(),
        ai_manager=SimpleNamespace(),
    )


def test_ensure_exchange_falls_back_to_paper_when_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KRYPTLOWCA_KEYS_PASSWORD", raising=False)
    monkeypatch.delenv("TRADING_GUI_KEYS_PASSWORD", raising=False)
    state = _build_state(network="Live", paper_balance="12 500")
    controller = _build_controller(state, _DummySecurity())

    ex_mgr = controller.ensure_exchange()

    assert isinstance(ex_mgr, ExchangeManager)
    assert ex_mgr.mode is Mode.PAPER
    assert ex_mgr._paper_initial_cash == pytest.approx(12_500.0)  # type: ignore[attr-defined]


def test_ensure_exchange_configures_testnet_with_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KRYPTLOWCA_KEYS_PASSWORD", "secret")
    payload = {"testnet_key": "TEST_KEY", "testnet_secret": "TEST_SECRET"}
    security = _DummySecurity(payload)
    state = _build_state(network="Testnet", mode="Futures")
    controller = _build_controller(state, security)

    ex_mgr = controller.ensure_exchange()

    assert ex_mgr.mode is Mode.FUTURES
    assert ex_mgr._testnet is True  # type: ignore[attr-defined]
    assert ex_mgr._api_key == "TEST_KEY"  # type: ignore[attr-defined]
    assert ex_mgr._secret == "TEST_SECRET"  # type: ignore[attr-defined]
    assert security.calls, "Security manager should be queried for keys"


def test_plan_event_updates_market_intel_label() -> None:
    state = _build_state(network="Testnet")
    controller = _build_controller(state, _DummySecurity())

    controller._handle_engine_event(
        {
            "type": "plan_created",
            "plan": {
                "market_intel": {
                    "mid_price": 101.1234,
                    "liquidity_usd": 2_500_000,
                    "momentum_score": 0.75,
                    "volatility_pct": 3.5,
                }
            },
        }
    )

    label_value = state.market_intel_label.value
    assert "price≈101.12" in label_value
    assert "liq≈2.50M USD" in label_value
    assert "mom≈+0.75" in label_value
    assert "vol≈3.50%" in label_value
    assert getattr(state, "market_intel_summary") == label_value
    history = getattr(state, "market_intel_history")
    assert len(history) == 1
    assert history[0].endswith(label_value)
    display = getattr(state, "market_intel_history_display")
    assert label_value in display


def test_plan_event_stores_summary_without_label() -> None:
    state = _build_state(network="Testnet", market_intel_label=None)
    controller = _build_controller(state, _DummySecurity())

    controller._handle_engine_event(
        {
            "type": "plan_created",
            "plan": {"market_intel": {"mid_price": 42.0}},
        }
    )

    summary = getattr(state, "market_intel_summary")
    assert isinstance(summary, str)
    assert summary.startswith("Market intel:")
    history = getattr(state, "market_intel_history")
    assert len(history) == 1
    assert history[0].endswith(summary)
    display = getattr(state, "market_intel_history_display")
    assert summary in display


def test_clear_market_intel_history_resets_state() -> None:
    state = _build_state(network="Testnet")
    state.market_intel_history_label = _DummyVar("Historia market intel: ...")
    controller = _build_controller(state, _DummySecurity())

    controller._handle_engine_event(
        {
            "type": "plan_created",
            "plan": {"market_intel": {"mid_price": 120.0}},
        }
    )
    assert getattr(state, "market_intel_history"), "Historia powinna zawierać wpis"

    controller.clear_market_intel_history()

    assert getattr(state, "market_intel_history") == []
    assert getattr(state, "market_intel_history_display") == "Brak historii market intel"
    label = getattr(state, "market_intel_history_label")
    assert isinstance(label, _DummyVar)
    assert label.value == "Brak historii market intel"


def test_get_market_intel_history_text_returns_normalised_entries() -> None:
    state = _build_state(network="Testnet")
    controller = _build_controller(state, _DummySecurity())

    controller._handle_engine_event(
        {
            "type": "plan_created",
            "plan": {"market_intel": {"mid_price": 89.5}},
        }
    )

    text = controller.get_market_intel_history_text()
    assert "Market intel:" in text
    assert "89.50" in text or "89.5000" in text
    assert "\n" not in text  # jedna pozycja historii, brak dodatkowych znaków końca linii


def test_get_market_intel_history_text_handles_empty_history() -> None:
    state = _build_state(network="Testnet")
    state.market_intel_history = []
    controller = _build_controller(state, _DummySecurity())

    assert controller.get_market_intel_history_text() == "Brak historii market intel"


def test_export_market_intel_history_writes_file(tmp_path: Path) -> None:
    state = _build_state(network="Testnet", logs_dir=tmp_path)
    controller = _build_controller(state, _DummySecurity())

    controller._handle_engine_event(
        {
            "type": "plan_created",
            "plan": {"market_intel": {"mid_price": 64.25}},
        }
    )

    target = controller.export_market_intel_history()

    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "Market intel" in content
    assert "64.25" in content or "64.2500" in content
    assert "Zapisano historię" in state.status.value
    assert state.market_intel_history_destination is None
    display = getattr(state, "market_intel_history_destination_display")
    assert "domyślny" in display
    assert str(target) in display


def test_export_market_intel_history_without_logs_dir_raises() -> None:
    state = _build_state(network="Testnet")
    state.paths = None
    controller = _build_controller(state, _DummySecurity())

    with pytest.raises(RuntimeError):
        controller.export_market_intel_history()


def test_load_market_intel_history_populates_state(tmp_path: Path) -> None:
    history_file = tmp_path / "market_intel_history.txt"
    history_file.write_text(
        "\n".join(
            [
                "12:00:00 UTC | Market intel: price≈25.00 liq≈1.00M USD",
                "12:05:00 UTC | Market intel: price≈26.00 liq≈1.50M USD",
            ]
        ),
        encoding="utf-8",
    )
    state = _build_state(network="Live", logs_dir=tmp_path)
    controller = _build_controller(state, _DummySecurity())

    entries = controller.load_market_intel_history()

    assert entries == [
        "12:00:00 UTC | Market intel: price≈25.00 liq≈1.00M USD",
        "12:05:00 UTC | Market intel: price≈26.00 liq≈1.50M USD",
    ]
    assert getattr(state, "market_intel_history") == entries
    assert getattr(state, "market_intel_history_display").endswith("liq≈1.50M USD")
    assert getattr(state, "market_intel_summary") == "Market intel: price≈26.00 liq≈1.50M USD"
    assert "Wczytano 2 wpisów" in state.status.value
    assert getattr(state, "market_intel_history_destination") is None


def test_load_market_intel_history_handles_missing_file(tmp_path: Path) -> None:
    state = _build_state(network="Live", logs_dir=tmp_path)
    controller = _build_controller(state, _DummySecurity())

    entries = controller.load_market_intel_history()

    assert entries == []
    assert getattr(state, "market_intel_history") == []
    assert getattr(state, "market_intel_history_display") == "Brak historii market intel"
    assert state.status.value == "Brak zapisanej historii market intel"


def test_set_market_intel_auto_save_persists_existing_history(tmp_path: Path) -> None:
    state = _build_state(network="Testnet", logs_dir=tmp_path)
    controller = _build_controller(state, _DummySecurity())

    controller._handle_engine_event(
        {
            "type": "plan_created",
            "plan": {"market_intel": {"mid_price": 77.0}},
        }
    )

    history_file = tmp_path / "market_intel_history.txt"
    assert not history_file.exists()

    controller.set_market_intel_auto_save(True)

    assert history_file.exists()
    content = history_file.read_text(encoding="utf-8")
    assert "Market intel" in content
    assert "77.00" in content or "77.0000" in content
    assert state.status.value == "Auto-zapis historii market intel włączony"


def test_auto_save_history_on_plan_event(tmp_path: Path) -> None:
    state = _build_state(network="Live", logs_dir=tmp_path)
    controller = _build_controller(state, _DummySecurity())

    controller.set_market_intel_auto_save(True)
    history_file = tmp_path / "market_intel_history.txt"
    if history_file.exists():
        history_file.unlink()

    controller._handle_engine_event(
        {
            "type": "plan_created",
            "plan": {"market_intel": {"mid_price": 91.5, "liquidity_usd": 500_000}},
        }
    )

    assert history_file.exists()
    content = history_file.read_text(encoding="utf-8")
    assert "91.50" in content or "91.5000" in content
    assert "500000" in content or "500.00K" in content or "500000.00" in content


def test_set_market_intel_history_destination_updates_state_and_label(tmp_path: Path) -> None:
    state = _build_state(network="Testnet", logs_dir=tmp_path)
    controller = _build_controller(state, _DummySecurity())

    custom_file = tmp_path / "custom_history.txt"
    returned_path = controller.set_market_intel_history_destination(custom_file)

    assert returned_path == custom_file.resolve()
    assert state.market_intel_history_destination == str(custom_file.resolve())
    display = getattr(state, "market_intel_history_destination_display")
    assert str(custom_file.name) in display
    label = getattr(state, "market_intel_history_path_label")
    assert isinstance(label, _DummyVar)
    assert str(custom_file.name) in label.value


def test_reset_history_destination_restores_default(tmp_path: Path) -> None:
    state = _build_state(network="Testnet", logs_dir=tmp_path)
    controller = _build_controller(state, _DummySecurity())

    controller.set_market_intel_history_destination(tmp_path / "custom.txt")
    controller.set_market_intel_history_destination(None)

    assert state.market_intel_history_destination is None
    display = getattr(state, "market_intel_history_destination_display")
    default_path = tmp_path / "market_intel_history.txt"
    assert "domyślny" in display
    assert str(default_path) in display


def test_auto_save_uses_custom_destination(tmp_path: Path) -> None:
    state = _build_state(network="Live", logs_dir=tmp_path)
    controller = _build_controller(state, _DummySecurity())

    custom_path = tmp_path / "intel_history.txt"
    controller.set_market_intel_history_destination(custom_path)
    controller.set_market_intel_auto_save(True)

    controller._handle_engine_event(
        {
            "type": "plan_created",
            "plan": {"market_intel": {"mid_price": 101.5, "liquidity_usd": 123_000}},
        }
    )

    assert custom_path.exists()
    content = custom_path.read_text(encoding="utf-8")
    assert "Market intel" in content
    assert "101.50" in content or "101.5000" in content
    assert "123000" in content or "123.00K" in content or "123000.00" in content


def test_reveal_market_intel_history_uses_opener(tmp_path: Path) -> None:
    history_file = tmp_path / "market_intel_history.txt"
    history_file.write_text("Market intel: price≈1.00", encoding="utf-8")
    state = _build_state(network="Live", logs_dir=tmp_path)
    controller = _build_controller(state, _DummySecurity())

    opened: list[Path] = []

    def opener(path: Path) -> bool:
        opened.append(path)
        return True

    returned = controller.reveal_market_intel_history(opener=opener)

    assert returned == history_file
    assert opened == [history_file]
    assert "Otwarto plik historii market intel" in state.status.value


def test_reveal_market_intel_history_missing_file(tmp_path: Path) -> None:
    history_file = tmp_path / "market_intel_history.txt"
    if history_file.exists():
        history_file.unlink()
    state = _build_state(network="Testnet", logs_dir=tmp_path)
    controller = _build_controller(state, _DummySecurity())

    with pytest.raises(FileNotFoundError):
        controller.reveal_market_intel_history(opener=lambda _: True)

    assert state.status.value == "Brak zapisanej historii market intel"


def test_reveal_market_intel_history_handles_opener_failure(tmp_path: Path) -> None:
    history_file = tmp_path / "market_intel_history.txt"
    history_file.write_text("Market intel: price≈2.00", encoding="utf-8")
    state = _build_state(network="Live", logs_dir=tmp_path)
    controller = _build_controller(state, _DummySecurity())

    def opener(_: Path) -> bool:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        controller.reveal_market_intel_history(opener=opener)

    assert state.status.value == "Nie udało się otworzyć pliku historii market intel"


def test_reveal_market_intel_history_fallbacks_to_webbrowser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    history_file = tmp_path / "market_intel_history.txt"
    history_file.write_text("Market intel: price≈3.00", encoding="utf-8")
    state = _build_state(network="Live", logs_dir=tmp_path)
    controller = _build_controller(state, _DummySecurity())

    opened_urls: list[str] = []

    def fake_open(url: str) -> bool:
        opened_urls.append(url)
        return True

    monkeypatch.setattr("KryptoLowca.ui.trading.controller.webbrowser.open", fake_open)

    returned = controller.reveal_market_intel_history()

    assert returned == history_file
    assert opened_urls == [history_file.as_uri()]
