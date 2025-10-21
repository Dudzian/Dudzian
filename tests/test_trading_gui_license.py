from __future__ import annotations

from datetime import date
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from bot_core.runtime.paths import DesktopAppPaths
from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.guards import CapabilityGuard

if "reporting" not in sys.modules:
    reporting_stub = types.ModuleType("reporting")

    class _StubReporter:
        def __init__(self, db_path: str | None = None, **_: object) -> None:
            self.db_path = db_path

        def log_trade(self, *args: object, **kwargs: object) -> None:
            return None

        def export_to_pdf(self, *args: object, **kwargs: object) -> None:
            return None

    reporting_stub.EnhancedReporting = _StubReporter
    reporting_stub.TradeInfo = dict
    sys.modules["reporting"] = reporting_stub

from KryptoLowca.ui.trading.controller import TradingSessionController
from KryptoLowca.ui.trading.license_context import COMMUNITY_NOTICE, build_license_ui_context
from KryptoLowca.ui.trading.state import AppState


class DummyVar:
    def __init__(self, value: object = "") -> None:
        self._value = value

    def get(self) -> object:
        return self._value

    def set(self, value: object) -> None:
        self._value = value


def _paths() -> DesktopAppPaths:
    base = Path(".")
    return DesktopAppPaths(
        app_root=base,
        logs_dir=base,
        text_log_file=base / "trading.log",
        db_file=base / "trading.db",
        open_positions_file=base / "open_positions.json",
        favorites_file=base / "favorites.json",
        presets_dir=base,
        models_dir=base,
        keys_file=base / "keys.enc",
        salt_file=base / "salt.bin",
        secret_vault_file=base / "api_keys.vault",
    )


def _build_state(
    *,
    network: str = "Live",
    mode: str = "Spot",
    guard: CapabilityGuard | None = None,
    capabilities=None,
) -> AppState:
    return AppState(
        paths=_paths(),
        runtime_metadata=None,
        symbol=DummyVar("BTC/USDT"),
        network=DummyVar(network),
        mode=DummyVar(mode),
        timeframe=DummyVar("1m"),
        fraction=DummyVar(0.1),
        paper_balance=DummyVar("10000"),
        account_balance=DummyVar("0"),
        status=DummyVar(""),
        license_capabilities=capabilities,
        capability_guard=guard,
        license_summary=DummyVar(""),
        license_notice=DummyVar(""),
    )


def _make_controller(state: AppState) -> TradingSessionController:
    controller = TradingSessionController(
        state,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    controller.ensure_exchange = MagicMock(return_value=MagicMock())
    return controller


def test_build_license_ui_context_for_pro_license() -> None:
    payload = {
        "edition": "pro",
        "environments": ["demo", "paper", "live"],
        "modules": {
            "futures": True,
            "observability_ui": True,
        },
        "runtime": {"auto_trader": True},
        "maintenance_until": "2026-07-01",
        "license_id": "DUD-1",
        "holder": {"name": "ACME"},
    }
    capabilities = build_capabilities_from_payload(payload, effective_date=date(2025, 7, 1))
    context = build_license_ui_context(capabilities)

    assert context.live_enabled is True
    assert context.futures_enabled is True
    assert context.auto_trader_enabled is True
    assert "Licencja: Pro" in context.summary
    assert "Licencja aktywna" in context.notice


def test_build_license_ui_context_when_license_missing() -> None:
    context = build_license_ui_context(None)
    assert context.live_enabled is False
    assert context.futures_enabled is False
    assert context.auto_trader_enabled is False
    assert context.notice == COMMUNITY_NOTICE


def test_build_license_ui_context_flags_missing_modules() -> None:
    payload = {
        "edition": "standard",
        "environments": ["demo", "paper"],
        "modules": {
            "futures": False,
            "observability_ui": False,
        },
        "runtime": {"auto_trader": False},
        "maintenance_until": "2024-01-01",
        "trial": {"enabled": True, "expires_at": "2023-12-01"},
    }
    capabilities = build_capabilities_from_payload(payload, effective_date=date(2025, 1, 2))
    context = build_license_ui_context(capabilities)

    assert context.live_enabled is False
    assert context.futures_enabled is False
    assert context.auto_trader_enabled is False
    assert context.maintenance_active is False
    assert "Dodaj moduł Futures" in context.notice
    assert "Licencja nie obejmuje modułu AutoTrader" in context.notice
    assert "Licencja wygasła" in context.notice


@patch("KryptoLowca.ui.trading.controller.messagebox.showerror")
def test_controller_blocks_live_without_permission(mock_messagebox: MagicMock) -> None:
    payload = {
        "edition": "standard",
        "environments": ["demo", "paper"],
        "modules": {"futures": True},
        "runtime": {"auto_trader": True},
    }
    capabilities = build_capabilities_from_payload(payload, effective_date=date.today())
    guard = CapabilityGuard(capabilities)
    state = _build_state(network="Live", guard=guard, capabilities=capabilities)
    controller = _make_controller(state)

    controller.start()

    assert state.running is False
    assert "Tryb live" in str(state.status.get())
    assert controller._reserved_slot is None
    mock_messagebox.assert_called()


@patch("KryptoLowca.ui.trading.controller.messagebox.showerror")
def test_controller_reserves_and_releases_slot(mock_messagebox: MagicMock) -> None:
    payload = {
        "edition": "pro",
        "environments": ["demo", "paper", "live"],
        "modules": {"futures": True},
        "runtime": {"auto_trader": True},
    }
    capabilities = build_capabilities_from_payload(payload, effective_date=date.today())
    guard = CapabilityGuard(capabilities)
    state = _build_state(network="Live", guard=guard, capabilities=capabilities)
    controller = _make_controller(state)

    controller.start()
    assert state.running is True
    assert controller._reserved_slot == "live_controller"
    assert guard._slots["live_controller"] == 1

    controller.stop()
    assert state.running is False
    assert guard._slots["live_controller"] == 0
    assert mock_messagebox.call_count == 0


@patch("KryptoLowca.ui.trading.controller.messagebox.showerror")
def test_controller_blocks_futures_without_module(mock_messagebox: MagicMock) -> None:
    payload = {
        "edition": "pro",
        "environments": ["demo", "paper"],
        "modules": {"futures": False},
        "runtime": {"auto_trader": True},
    }
    capabilities = build_capabilities_from_payload(payload, effective_date=date.today())
    guard = CapabilityGuard(capabilities)
    state = _build_state(network="Testnet", mode="Futures", guard=guard, capabilities=capabilities)
    controller = _make_controller(state)

    controller.start()

    assert state.running is False
    assert "Dodaj moduł Futures" in str(state.status.get())
    assert controller._reserved_slot is None
    mock_messagebox.assert_called()


@patch("KryptoLowca.ui.trading.controller.messagebox.showerror")
def test_controller_blocks_autotrader_when_disabled(mock_messagebox: MagicMock) -> None:
    payload = {
        "edition": "standard",
        "environments": ["demo", "paper"],
        "modules": {"futures": True},
        "runtime": {"auto_trader": False},
    }
    capabilities = build_capabilities_from_payload(payload, effective_date=date.today())
    guard = CapabilityGuard(capabilities)
    state = _build_state(network="Testnet", mode="Spot", guard=guard, capabilities=capabilities)
    controller = _make_controller(state)

    controller.start()

    assert state.running is False
    assert "AutoTrader" in str(state.status.get())
    assert controller._reserved_slot is None
    mock_messagebox.assert_called()
