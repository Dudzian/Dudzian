from __future__ import annotations

from pathlib import Path

from core.config.ui_settings import UISettings, UISettingsStore
from tests.ui._qt import require_pyside6

require_pyside6()

from ui.backend.dashboard_settings import DashboardSettingsController


def test_dashboard_cards_contract_matches_runtime_overview_defaults(tmp_path: Path) -> None:
    controller = DashboardSettingsController(store=UISettingsStore(tmp_path / "ui_settings.json"))

    assert "feed_sla" in controller.availableCards
    assert "risk_journal" in controller.availableCards

    defaults = UISettings().dashboard.card_order
    assert {"feed_sla", "risk_journal"} <= set(defaults)


def test_visible_order_contains_new_cards_after_reset_defaults(tmp_path: Path) -> None:
    store = UISettingsStore(tmp_path / "ui_settings.json")
    controller = DashboardSettingsController(store=store)

    assert "feed_sla" in controller.visibleCardOrder
    assert "risk_journal" in controller.visibleCardOrder

    controller.setCardVisibility("feed_sla", False)
    controller.setCardVisibility("risk_journal", False)

    assert "feed_sla" not in controller.visibleCardOrder
    assert "risk_journal" not in controller.visibleCardOrder

    controller.resetDefaults()

    assert "feed_sla" in controller.visibleCardOrder
    assert "risk_journal" in controller.visibleCardOrder
