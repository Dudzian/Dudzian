from __future__ import annotations

from pathlib import Path

QML_ROOT = Path("ui/pyside_app/qml")
MAIN = QML_ROOT / "MainWindow.qml"
SMOKE = Path("ui/pyside_app/smoke.py")


def _source() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(QML_ROOT.rglob("*.qml")))


def test_decision_explainability_state_and_functions_declared() -> None:
    text = MAIN.read_text(encoding="utf-8")
    for token in (
        "property bool decisionExplainDrawerOpen",
        "property string selectedDecisionId",
        "property string selectedDecisionPair",
        "property string selectedDecisionAction",
        "property string selectedDecisionSource",
        "property string selectedDecisionConfidence",
        "property string selectedDecisionRiskState",
        "property string selectedDecisionStrategy",
        "property string selectedDecisionReason",
        "property var selectedDecisionAuditRows",
        "property var selectedDecisionInputSnapshot",
        "property var selectedDecisionAlternatives",
        "property var selectedDecisionRiskChecks",
        "property var selectedDecisionLineageLinks",
        "property string selectedDecisionPaperImpact",
        "property string selectedDecisionSafetySummary",
        "function openDecisionExplainDrawer(row)",
        "function closeDecisionExplainDrawer()",
        "function explainDecisionRow(row)",
        "function explainScannerCandidateDecision(pair)",
        "function explainPaperOrderDecision(order)",
        "function buildDecisionAuditRows(row)",
        "function buildDecisionRiskChecks(row)",
        "function buildDecisionAlternatives(row)",
        "function buildDecisionInputSnapshot(row)",
    ):
        assert token in text


def test_decision_explainability_drawer_copy_and_sections_present() -> None:
    source = _source()
    for token in (
        "decisionExplainabilityDrawer",
        "Dlaczego bot tak zdecydował?",
        "source: %1 • confidence: %2",
        "AI score: %1 • risk score: %2 • liquidity score: %3",
        "strategy match: %1",
        "risk profile / kill-switch / risk lock",
        "expected paper action",
        "paper impact",
        "Human explanation",
        "Audit trail",
        "Risk checks",
        "Input snapshot",
        "Alternatywy",
        "Lineage links",
        "Explanation is local preview only",
        "No backend AI inference",
        "No exchange/API call",
        "No order submission",
        "No real orders",
        "No secrets read",
        "Wyjaśnienie działa lokalnie w preview",
        "Brak backendowej inferencji AI",
        "Brak połączenia z giełdą/API",
        "Brak składania zleceń",
        "Brak prawdziwych zleceń",
        "Brak odczytu sekretów",
    ):
        assert token in source


def test_decision_explainability_integrations_and_tooltips_present() -> None:
    source = _source()
    for token in (
        "root.previewState.openDecisionExplainDrawer(modelData)",
        "previewState.explainScannerCandidateDecision(previewState.scannerSelectedPair)",
        "previewState.explainPaperOrderDecision(modelData.sourceRow)",
        "Wyjaśnij ostatnią decyzję",
        "Explain decision",
        "Explain candidate",
        "Audit trail",
        "Risk checks",
        "Input snapshot",
        "Paper impact",
        "explainability",
        "audit trail",
        "lineage",
        "input snapshot",
        "risk check",
        "decision source",
        "alternative candidate",
        "paper impact",
    ):
        assert token in source


def test_decision_explainability_smoke_fields_present() -> None:
    smoke = SMOKE.read_text(encoding="utf-8")
    for token in (
        "decision_explainability_state_present",
        "decision_explain_drawer_present",
        "decision_explain_open_close_works",
        "decision_explain_builds_audit_rows",
        "decision_explain_has_risk_checks",
        "decision_explain_has_input_snapshot",
        "decision_explain_has_alternatives",
        "decision_explain_has_paper_impact",
        "decision_explain_safety_boundary_ok",
        "scanner_candidate_explain_opens_shared_drawer",
        "paper_order_explain_local_only",
        "explainability_no_backend_inference",
        "explainability_no_network_api_calls",
        "explainability_no_order_submission",
        "explainability_no_secret_reads",
    ):
        assert token in smoke


def test_decision_explainability_sources_avoid_forbidden_tokens() -> None:
    source = _source()
    for token in (
        "create_order",
        "fetch_balance",
        "load_markets",
        "keyring",
        "dotenv",
        "shell=True",
        "subprocess.run",
        "os.environ",
        "getenv",
        "ccxt",
    ):
        assert token not in source
