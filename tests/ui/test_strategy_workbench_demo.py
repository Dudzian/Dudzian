from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")
pytest.importorskip("playwright.sync_api", reason="Wymagany Playwright do testów end-to-end")

from PySide6.QtCore import QObject, QUrl, QMetaObject, Qt, Q_ARG
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication
from playwright.sync_api import sync_playwright


@pytest.mark.timeout(30)
def test_workbench_demo_mode_roundtrip() -> None:
    app = QApplication.instance() or QApplication([])

    engine = QQmlApplicationEngine()
    workbench_qml = (
        Path(__file__).resolve().parents[2]
        / "ui"
        / "qml"
        / "components"
        / "workbench"
        / "StrategyWorkbench.qml"
    )
    engine.load(QUrl.fromLocalFile(str(workbench_qml)))
    assert engine.rootObjects(), "Nie udało się załadować widoku StrategyWorkbench"

    root_object = engine.rootObjects()[0]
    assert isinstance(root_object, QObject)

    view_model = root_object.findChild(QObject, "strategyWorkbenchViewModel")
    assert view_model is not None, "ViewModel nie został odnaleziony"

    assert QMetaObject.invokeMethod(
        view_model,
        "activateDemoMode",
        Qt.DirectConnection,
        Q_ARG(str, "momentum"),
    ) is True
    assert view_model.property("demoModeActive") is True
    assert view_model.property("demoModeTitle") == "Momentum Pro"

    instrument_details = view_model.property("instrumentDetails")
    assert instrument_details["exchange"] == "BINANCE"
    assert instrument_details["venueSymbol"] == "BTCUSDT"
    assert instrument_details["quoteCurrency"] == "USDT"

    portfolio_summary = view_model.property("portfolioSummary")
    assert portfolio_summary["maxExposureUtilization"] == pytest.approx(0.82, rel=1e-3)

    risk_timeline = view_model.property("riskTimeline")
    assert isinstance(risk_timeline, list)
    assert len(risk_timeline) >= 4
    latest_sample = risk_timeline[0]
    assert latest_sample["portfolioValue"] == pytest.approx(129800, rel=1e-3)
    first_timestamp = latest_sample["timestamp"]

    open_positions = view_model.property("openPositions")
    assert isinstance(open_positions, list)
    assert len(open_positions) >= 2
    assert open_positions[0]["symbol"] == "BTC/USDT"
    assert open_positions[0]["side"] == "Long"

    pending_orders = view_model.property("pendingOrders")
    assert isinstance(pending_orders, list)
    assert len(pending_orders) >= 2
    assert pending_orders[0]["status"] == "PartiallyFilled"
    assert pending_orders[1]["timeInForce"].upper() == "GTC"

    trade_history = view_model.property("tradeHistory")
    assert isinstance(trade_history, list)
    assert len(trade_history) >= 3
    assert trade_history[0]["status"].lower() == "filled"

    signal_alerts = view_model.property("signalAlerts")
    assert isinstance(signal_alerts, list)
    assert len(signal_alerts) >= 2
    assert signal_alerts[0]["direction"] == "Long"
    assert signal_alerts[0]["confidence"] == pytest.approx(0.82, rel=1e-3)

    market_sentiment = view_model.property("marketSentiment")
    assert market_sentiment["trend"] == "Bullish"
    assert market_sentiment["globalScore"] == pytest.approx(0.64, rel=1e-3)
    assert isinstance(market_sentiment["sources"], list)
    assert market_sentiment["sources"][0]["name"].startswith("Fear & Greed")

    news_headlines = view_model.property("newsHeadlines")
    assert isinstance(news_headlines, list)
    assert len(news_headlines) >= 2
    assert news_headlines[0]["source"] == "CryptoTimes"
    assert news_headlines[0]["title"].startswith("ETF-y spot")

    capital_allocation = view_model.property("capitalAllocation")
    assert isinstance(capital_allocation, list)
    assert len(capital_allocation) >= 3
    assert capital_allocation[0]["segment"] == "Momentum Core"
    assert capital_allocation[0]["weight"] == pytest.approx(0.38, rel=1e-3)

    performance_comparison = view_model.property("performanceComparison")
    assert isinstance(performance_comparison, dict)
    assert performance_comparison["benchmarkName"] == "BTC Spot Index"
    assert performance_comparison["strategyReturn"] == pytest.approx(0.186, rel=1e-3)
    assert performance_comparison["alpha"] == pytest.approx(0.044, rel=1e-3)

    scenario_tests = view_model.property("scenarioTests")
    assert isinstance(scenario_tests, list)
    assert len(scenario_tests) >= 2
    assert scenario_tests[0]["success"] is True
    assert scenario_tests[1]["success"] is False
    assert scenario_tests[1]["maxDrawdown"] == pytest.approx(0.128, rel=1e-3)
    assert scenario_tests[1]["runCount"] >= 2
    initial_scenario_runs = [entry.get("runCount", 0) for entry in scenario_tests]
    initial_last_run = scenario_tests[0].get("lastRunAt", "")

    automation_rules = view_model.property("automationRules")
    assert isinstance(automation_rules, list)
    assert len(automation_rules) >= 3
    assert automation_rules[0]["enabled"] is True
    assert automation_rules[0]["name"].startswith("Strażnik")
    assert automation_rules[1]["trigger"].startswith("Odchylenie")
    disabled_rules = sum(1 for rule in automation_rules if not rule.get("enabled", True))

    execution_diagnostics = view_model.property("executionDiagnostics")
    assert isinstance(execution_diagnostics, dict)
    assert execution_diagnostics["provider"] == "OMS Simulator"
    assert execution_diagnostics["avgLatencyMs"] == pytest.approx(38, rel=1e-3)
    assert execution_diagnostics["fillRate"] == pytest.approx(0.912, rel=1e-3)
    assert execution_diagnostics["recentIncidents"][0]["type"] == "reject"
    initial_execution_updated = execution_diagnostics.get("lastUpdated", "")

    compliance_summary = view_model.property("complianceSummary")
    assert isinstance(compliance_summary, dict)
    assert compliance_summary["licenseActive"] is True
    assert compliance_summary["automationPaused"] is False
    assert compliance_summary["openAlerts"] == len(signal_alerts)
    assert compliance_summary.get("scenarioFailures", 0) == 1
    assert compliance_summary.get("disabledAutomationRules", 0) == disabled_rules
    exposure_utilization = risk_timeline[0].get(
        "exposureUtilization",
        portfolio_summary.get("maxExposureUtilization", 0.0),
    )
    baseline = 0.9 if compliance_summary["licenseActive"] else 0.45
    expected_score = baseline
    expected_score -= max(0.0, float(exposure_utilization) - 0.85) * 0.5
    expected_score -= min(float(compliance_summary.get("outstandingBreaches", 0)) * 0.1, 0.5)
    expected_score -= min(compliance_summary["openAlerts"] * 0.06, 0.3)
    if compliance_summary["automationPaused"]:
        expected_score -= 0.08
    expected_score -= min(float(compliance_summary.get("scenarioFailures", 0)) * 0.05, 0.2)
    expected_score -= min(disabled_rules * 0.02, 0.1)
    expected_score = max(0.0, min(1.0, expected_score))
    assert compliance_summary["complianceScore"] == pytest.approx(expected_score, rel=1e-6)
    assert compliance_summary["notes"], "Powinny istnieć notatki z podsumowaniem zgodności"

    activity_log = view_model.property("activityLog")
    assert isinstance(activity_log, list)
    assert activity_log[0]["type"] == "scheduler:start"

    assert QMetaObject.invokeMethod(
        view_model,
        "startScheduler",
        Qt.DirectConnection,
    ) is True

    control_state = view_model.property("controlState")
    assert control_state["schedulerRunning"] is True
    assert control_state["lastActionSuccess"] is True

    activity_log = view_model.property("activityLog")
    assert activity_log[0]["type"] == "scheduler:start"
    assert activity_log[0]["success"] is True

    assert QMetaObject.invokeMethod(
        view_model,
        "triggerRiskRefresh",
        Qt.DirectConnection,
    ) is True

    control_state = view_model.property("controlState")
    assert control_state["manualRefreshCount"] == 3
    assert control_state["lastActionMessage"] == "Zainicjowano odświeżenie ryzyka"

    refreshed_timeline = view_model.property("riskTimeline")
    assert len(refreshed_timeline) == len(risk_timeline) + 1
    assert refreshed_timeline[0]["timestamp"] != first_timestamp
    assert refreshed_timeline[0]["source"] == "demo"

    refreshed_scenarios = view_model.property("scenarioTests")
    assert isinstance(refreshed_scenarios, list)
    assert refreshed_scenarios[0]["runCount"] >= initial_scenario_runs[0]
    assert refreshed_scenarios[0]["lastRunAt"] != initial_last_run

    refreshed_diagnostics = view_model.property("executionDiagnostics")
    assert refreshed_diagnostics["lastUpdated"] != initial_execution_updated
    assert refreshed_diagnostics["recentIncidents"][0]["type"] == "refresh"

    activity_log = view_model.property("activityLog")
    assert activity_log[0]["type"] == "risk:refresh"
    assert activity_log[0]["success"] is True

    assert QMetaObject.invokeMethod(
        view_model,
        "stopScheduler",
        Qt.DirectConnection,
    ) is True

    control_state = view_model.property("controlState")
    assert control_state["schedulerRunning"] is False
    assert control_state["lastActionSuccess"] is True

    activity_log = view_model.property("activityLog")
    assert activity_log[0]["type"] == "scheduler:stop"
    assert activity_log[0]["success"] is True

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - zależne od środowiska CI
            pytest.skip(f"Brak zainstalowanej przeglądarki Playwright: {exc}")
        page = browser.new_page()
        page.set_content("<html><body><span id='mode'></span></body></html>")
        page.eval_on_selector("#mode", "(el, value) => el.textContent = value", view_model.property("demoModeTitle"))
        assert page.text_content("#mode") == "Momentum Pro"
        browser.close()

    for obj in engine.rootObjects():
        obj.deleteLater()
    engine.deleteLater()
    app.processEvents()
