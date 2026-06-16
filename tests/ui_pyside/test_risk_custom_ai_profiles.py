"""Source-level checks for custom and AI recommended risk preview profiles."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QML_ROOT = REPO_ROOT / "ui" / "pyside_app" / "qml"
SMOKE_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "smoke.py"
UI_PYSIDE_ROOT = REPO_ROOT / "ui" / "pyside_app"
APP_SOURCE = UI_PYSIDE_ROOT / "app.py"
ALLOWED_QT_QUICK_CONTROLS_STYLE_BOOTSTRAP = (
    'os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")'
)


def _qml_source() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(QML_ROOT.rglob("*.qml")))


def _scoped_source() -> str:
    paths = [*UI_PYSIDE_ROOT.rglob("*.py"), *UI_PYSIDE_ROOT.rglob("*.qml")]
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(paths))


def _source_without_allowed_qt_style_bootstrap(source: str) -> str:
    return source.replace(ALLOWED_QT_QUICK_CONTROLS_STYLE_BOOTSTRAP, "", 1)


def test_risk_profiles_and_apply_functions_exist() -> None:
    source = _qml_source()

    for profile in ("Conservative", "Balanced", "Aggressive", "Custom", "AI Recommended"):
        assert profile in source
    assert "function setRiskProfile(profile)" in source
    assert "function applyAiRecommendedRiskProfile()" in source
    assert "customRiskState" in source


def test_custom_risk_fields_explanation_and_active_limits_exist() -> None:
    source = _qml_source()

    for field in (
        "max position",
        "max open positions",
        "stop loss",
        "take profit",
        "max slippage",
        "max drawdown",
        "daily loss limit",
        "per-symbol exposure",
        "confidence floor",
        "cooldown",
        "max allocation",
        "allow AI override",
    ):
        assert field in source
    assert "riskExplanationCard" in source
    assert "Dlaczego takie ustawienia?" in source
    assert "riskActiveLimitsTable" in source
    assert "Aktywne limity" in source


def test_risk_tooltips_glossary_and_safety_copy_exist() -> None:
    source = _qml_source()

    for tooltip in (
        "Risk profile Conservative",
        "Risk profile Balanced",
        "Risk profile Aggressive",
        "Risk profile Custom",
        "Risk profile AI Recommended",
        "max position",
        "stop loss",
        "take profit",
        "slippage",
        "drawdown",
        "daily loss limit",
        "confidence floor",
        "kill-switch",
        "allow AI override",
    ):
        assert tooltip in source
    for term in (
        "Custom risk",
        "AI Recommended risk",
        "confidence floor",
        "exposure",
        "daily loss limit",
        "cooldown",
        "risk override",
    ):
        assert term in source
    for safety_label in (
        "Live trading disabled",
        "Exchange I/O disabled",
        "Order submission disabled",
        "API keys not required",
        "No real orders",
        "Runtime loop not started",
        "production runtime loop not started",
        "Risk settings are local preview only",
        "Blocked events update audit/logs only",
    ):
        assert safety_label in source


def test_smoke_audit_contract_has_risk_8_0e_flags() -> None:
    source = SMOKE_SOURCE.read_text(encoding="utf-8")

    for flag in (
        "risk_custom_profile_present",
        "risk_ai_recommended_present",
        "risk_ai_recommended_updates_values",
        "risk_custom_does_not_write_runtime_config",
        "risk_ai_recommended_explanation_present",
        "risk_active_limits_present",
        "risk_tooltips_present",
        "risk_safety_boundary_ok",
        "risk_blocked_tick_does_not_mutate_paper_pnl",
        "risk_blocked_tick_does_not_mutate_paper_equity",
        "risk_blocked_tick_increments_blocked_count",
        "risk_blocked_tick_appends_decision",
        "risk_blocked_tick_appends_telemetry",
        "risk_blocked_tick_creates_no_filled_order",
        "risk_unlocked_tick_can_update_financial_state",
        "simulation_respects_risk_preview_state",
    ):
        assert flag in source


def test_forbidden_tokens_do_not_appear_in_ui_preview_scope() -> None:
    source = _source_without_allowed_qt_style_bootstrap(_scoped_source())
    forbidden = (
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

    for token in forbidden:
        assert token not in source


def test_app_bootstrap_allows_only_basic_qt_quick_controls_style_env_default() -> None:
    app_source = APP_SOURCE.read_text(encoding="utf-8")

    assert "QT_QUICK_CONTROLS_STYLE" in app_source
    assert "os.environ.setdefault" in app_source
    assert ALLOWED_QT_QUICK_CONTROLS_STYLE_BOOTSTRAP in app_source
    assert 'os.environ["QT_QUICK_CONTROLS_STYLE"]' not in app_source
    assert "os.environ['QT_QUICK_CONTROLS_STYLE']" not in app_source
    assert "os.getenv" not in app_source
    assert "getenv" not in app_source


def test_app_bootstrap_forbidden_env_guard_still_catches_other_environ_usage() -> None:
    app_source = APP_SOURCE.read_text(encoding="utf-8")
    unsafe_source = (
        f'{app_source}\nos.environ.setdefault("TRADING_API_SECRET", "unsafe-preview-secret")\n'
    )

    normalized_source = _source_without_allowed_qt_style_bootstrap(unsafe_source)

    assert "os.environ" in normalized_source
