from __future__ import annotations

from core.config.ui_settings import UISettings


def test_ui_settings_default_order_includes_feed_sla_and_risk_journal() -> None:
    defaults = UISettings().dashboard.card_order

    assert "feed_sla" in defaults
    assert "risk_journal" in defaults
