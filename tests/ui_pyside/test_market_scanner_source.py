"""Source checks for UI-PREVIEW-8.0F Market Scanner / Okazje."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QML_ROOT = REPO_ROOT / "ui" / "pyside_app" / "qml"
SMOKE_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "smoke.py"

FORBIDDEN_SCANNER_TOKENS = (
    "create" + "_" + "order",
    "fetch" + "_" + "balance",
    "load" + "_" + "markets",
    "key" + "ring",
    "dot" + "env",
    "shell" + "=True",
    "subprocess" + "." + "run",
    "os" + "." + "environ",
    "get" + "env",
    "c" + "cxt",
)


def _source() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(QML_ROOT.rglob("*.qml")))


def test_market_scanner_tab_state_and_functions_exist() -> None:
    source = _source()
    for token in (
        "nav.marketScanner",
        "Okazje",
        "Market Scanner",
        "marketScannerPanel",
        "marketScannerPanelComponent",
        "scannerStatus",
        "scannerActive",
        "scannerLastScanAt",
        "scannerTickCount",
        "scannerSelectedExchange",
        "scannerUniverseCount",
        "scannerCandidateCount",
        "scannerRejectedCount",
        "scannerWatchlistCount",
        "scannerBestOpportunity",
        "scannerRows",
        "scannerRejectedRows",
        "scannerWatchlistPairs",
        "scannerWatchlistRows",
        "scannerAiCandidateRows",
        "scannerFilterMode",
        "scannerSortMode",
        "scannerMinAiScore",
        "scannerMinLiquidityScore",
        "scannerMaxRiskScore",
        "scannerSelectedPair",
        "scannerExplanation",
        "startMarketScannerPreview",
        "pauseMarketScannerPreview",
        "stopMarketScannerPreview",
        "resetMarketScannerPreview",
        "runMarketScannerTick",
        "runMarketScannerBurst",
        "selectScannerPair",
        "addScannerPairToWatchlist",
        "removeScannerPairFromWatchlist",
        "blacklistScannerPair",
        "setScannerFilterMode",
        "setScannerSortMode",
        "setScannerThreshold",
        "explainScannerCandidate",
    ):
        assert token in source


def test_market_scanner_table_filters_safety_tooltips_and_glossary_exist() -> None:
    source = _source()
    for token in (
        "Pair",
        "Exchange",
        "Price",
        "Volume",
        "Trend",
        "Spread",
        "Liquidity",
        "AI score",
        "Risk",
        "Strategy",
        "Recommendation",
        "Reason",
        "All",
        "AI candidates",
        "Trade candidates",
        "Watchlist",
        "Rejected",
        "Blocked",
        "High liquidity",
        "Low risk",
        "Top score",
        "Risk score",
        "Trend strength",
        "min AI score",
        "min liquidity score",
        "max risk score",
        "Dlaczego bot wybrał / odrzucił tę parę?",
        "Safe preview scanner",
        "Live trading disabled",
        "Exchange I/O disabled",
        "Order submission disabled",
        "API keys not required",
        "No real orders",
        "No network/API calls",
        "Local preview catalog only",
        "Skaner działa lokalnie w preview",
        "Brak realnych połączeń API",
        "Start scanner",
        "Pause scanner",
        "Run scan tick",
        "Run scan burst",
        "Liquidity score",
        "Volatility",
        "Strategy match",
        "Blacklist",
        "Explain candidate",
        "Market Scanner",
        "Candidate",
        "Rejected setup",
    ):
        assert token in source


def test_market_scanner_smoke_contract_fields_exist() -> None:
    smoke = SMOKE_SOURCE.read_text(encoding="utf-8")
    for token in (
        "market_scanner_tab_present",
        "market_scanner_state_present",
        "market_scanner_rows_present",
        "market_scanner_start_sets_scanning",
        "market_scanner_pause_sets_paused",
        "market_scanner_tick_updates_rows",
        "market_scanner_burst_updates_count",
        "market_scanner_explain_updates_explanation",
        "market_scanner_watchlist_updates_count",
        "market_scanner_watchlist_separate_from_whitelist",
        "market_scanner_watchlist_add_does_not_mutate_whitelist",
        "market_scanner_watchlist_remove_does_not_mutate_whitelist",
        "market_scanner_watchlist_filter_uses_scanner_watchlist",
        "market_scanner_blacklist_updates_rejected",
        "market_scanner_filter_sort_threshold_present",
        "market_scanner_safety_boundary_ok",
        "market_scanner_no_network_api_calls",
        "market_scanner_no_order_submission",
        "market_scanner_no_secret_reads",
        "simulation_can_use_scanner_candidate_local_only",
        "top_navigation_default_order_unique",
    ):
        assert token in smoke


def test_market_scanner_sources_avoid_forbidden_runtime_tokens() -> None:
    source = (QML_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    source += (QML_ROOT / "views" / "MarketScanner.qml").read_text(encoding="utf-8")
    for token in FORBIDDEN_SCANNER_TOKENS:
        assert token not in source


def _qml_function_body(source: str, function_name: str) -> str:
    marker = f"function {function_name}"
    start = source.index(marker)
    next_function = source.find("\n    function ", start + len(marker))
    return source[start:] if next_function == -1 else source[start:next_function]


def test_market_scanner_watchlist_is_separate_from_whitelist_source_contract() -> None:
    main_window = (QML_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    market_scanner = (QML_ROOT / "views" / "MarketScanner.qml").read_text(encoding="utf-8")

    assert "property var scannerWatchlistPairs: []" in main_window
    assert "Watchlist = obserwacja. Whitelist = dopuszczone pary" in market_scanner
    assert "Watchlist is for observation. Whitelist is for allowed pairs" in market_scanner
    assert "preview-local blocklist shared with Trading Universe" in market_scanner

    refresh_body = _qml_function_body(main_window, "refreshScannerBuckets")
    visible_body = _qml_function_body(main_window, "visibleScannerRows")
    add_body = _qml_function_body(main_window, "addScannerPairToWatchlist")
    remove_body = _qml_function_body(main_window, "removeScannerPairFromWatchlist")

    assert "scannerWatchlistPairs" in refresh_body
    assert "scannerWatchlistPairs.length" in refresh_body
    assert "scannerWatchlistPairs" in visible_body
    assert 'scannerFilterMode === "Watchlist"' in visible_body
    assert "scannerWatchlistPairs" in add_body
    assert "scannerWatchlistPairs" in remove_body
    assert "whitelistPairs" not in add_body
    assert "whitelistPairs" not in remove_body
    assert "selectedPairs" not in add_body
    assert "selectedPairs" not in remove_body


def test_market_scanner_top_navigation_default_order_is_unique() -> None:
    main_window = (QML_ROOT / "MainWindow.qml").read_text(encoding="utf-8")
    expected_order = {
        "sidePanel": 0,
        "aiCenterPanel": 1,
        "tradingUniversePanel": 2,
        "marketScannerPanel": 3,
        "portfolioPerformancePanel": 4,
        "terminalPanel": 5,
        "strategiesPanel": 6,
        "riskControlsPanel": 7,
        "aiDecisionsPanel": 8,
        "telemetryPanel": 9,
        "diagnosticsPanel": 10,
        "helpGlossaryPanel": 11,
    }
    matches = re.findall(r'panelId: "([^"]+)",[^\n]+defaultOrder: (\d+)', main_window)
    panel_orders = {panel_id: int(order) for panel_id, order in matches}

    assert panel_orders == expected_order
    assert len(set(panel_orders.values())) == len(panel_orders)
