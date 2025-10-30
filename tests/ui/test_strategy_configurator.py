from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")

from PySide6.QtCore import QObject, Qt, QMetaObject, QUrl
from PySide6.QtQml import QQmlApplicationEngine

try:  # pragma: no cover - zależy od środowiska CI
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # brak bibliotek systemowych (np. libGL)
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)


@pytest.mark.timeout(20)
def test_strategy_configurator_lists_new_strategies() -> None:
    app = QApplication.instance() or QApplication([])

    engine = QQmlApplicationEngine()
    view_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "views" / "StrategyConfigurator.qml"
    engine.load(QUrl.fromLocalFile(str(view_path)))
    assert engine.rootObjects(), "Nie udało się załadować StrategyConfigurator"

    root = engine.rootObjects()[0]
    assert isinstance(root, QObject)

    view_model = root.findChild(QObject, "strategyConfiguratorViewModel")
    assert view_model is not None

    catalog_entries = [
        {
            "name": "theta_income_balanced",
            "engine": "options_income",
            "description": "Dochód theta",
            "license_tier": "enterprise",
            "risk_classes": ["derivatives", "income"],
            "required_data": ["options_chain", "greeks", "ohlcv"],
            "tags": ["options", "income"],
            "parameters": {"min_iv": 0.32, "max_delta": 0.28},
        },
        {
            "name": "futures_basis_defender",
            "engine": "futures_spread",
            "description": "Hedging rozjazdów futures",
            "license_tier": "enterprise",
            "risk_classes": ["derivatives", "market_neutral"],
            "required_data": ["futures_curve", "funding_rates", "ohlcv"],
            "tags": ["futures", "hedge"],
            "parameters": {"entry_z": 1.3, "exit_z": 0.35},
        },
        {
            "name": "cross_exchange_delta_guard",
            "engine": "cross_exchange_hedge",
            "description": "Delta neutral cross-venue",
            "license_tier": "enterprise",
            "risk_classes": ["hedging", "liquidity"],
            "required_data": ["spot_basis", "inventory_skew", "latency_metrics"],
            "tags": ["hedge", "multi_venue"],
            "parameters": {"basis_scale": 0.009, "max_hedge_ratio": 0.85},
        },
    ]

    view_model.setProperty("catalogDefinitions", catalog_entries)
    QMetaObject.invokeMethod(root, "rebuildFiltered", Qt.DirectConnection)

    strategy_names = root.property("strategyNames")
    assert isinstance(strategy_names, list)
    assert {"theta_income_balanced", "futures_basis_defender", "cross_exchange_delta_guard"} <= set(strategy_names)

    list_view = root.findChild(QObject, "strategyConfiguratorList")
    assert list_view is not None
    assert list_view.property("count") == 3

    list_view.setProperty("currentIndex", 1)
    app.processEvents()

    parameter_view = root.findChild(QObject, "strategyParameterView")
    assert parameter_view is not None
    text = parameter_view.property("text")
    assert "entry_z" in text

    for obj in engine.rootObjects():
        obj.deleteLater()
    engine.deleteLater()
    app.processEvents()
