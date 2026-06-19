from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/ci/functional_preview_readiness.py"
REPORT = REPO_ROOT / "reports/ci/functional_preview_readiness.json"
REQUIRED_SECTIONS = {
    "preview_mode_contract",
    "data_source_market_feed",
    "scanner_opportunity_pipeline",
    "ai_decision_governor",
    "paper_terminal_order_lifecycle",
    "portfolio_positions_trades",
    "alerts_telemetry_audit",
    "settings_config_api_keys",
    "runtime_session_control_plane",
    "strategy_model_backtest_replay",
    "live_safety_hard_gate",
}
VALID_STATUSES = {"functional", "partial", "static_mock_only", "missing", "unknown"}


def _load_report() -> dict[str, object]:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_script_generates_deterministic_report(tmp_path: Path) -> None:
    output = tmp_path / "readiness.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    generated = json.loads(output.read_text(encoding="utf-8"))
    tracked = _load_report()
    assert generated == tracked
    assert generated["evaluated_at"] == "2026-06-16T00:00:00Z"


@pytest.mark.parametrize("section_name", sorted(REQUIRED_SECTIONS))
def test_report_section_schema(section_name: str) -> None:
    payload = _load_report()
    sections = payload["sections"]
    assert isinstance(sections, dict)
    assert section_name in sections
    section = sections[section_name]
    assert isinstance(section, dict)
    assert section["status"] in VALID_STATUSES
    assert isinstance(section["evidence_files"], list)
    assert section["evidence_files"], f"{section_name} needs evidence files"
    assert all(isinstance(item, str) for item in section["evidence_files"])
    assert all((REPO_ROOT / item).exists() for item in section["evidence_files"])
    assert isinstance(section["runtime_backed"], bool)
    assert isinstance(section["static_qml_only"], bool)
    assert isinstance(section["supports_test_server"], bool)
    assert isinstance(section["supports_read_only_real_data"], bool)
    assert isinstance(section["paper_only_execution_safe"], bool)
    assert isinstance(section["gaps"], list)
    assert all(isinstance(item, str) and item for item in section["gaps"])
    assert isinstance(section["recommended_next_step"], str)
    assert section["recommended_next_step"].strip()


def test_report_does_not_overstate_functional_readiness() -> None:
    payload = _load_report()
    sections = payload["sections"]
    assert isinstance(sections, dict)
    assert all(section["status"] != "functional" for section in sections.values())
    assert all(not section["supports_test_server"] for section in sections.values())
    assert all(not section["supports_read_only_real_data"] for section in sections.values())
    for name, section in sections.items():
        if section["status"] == "static_mock_only":
            assert section["static_qml_only"] is True
            assert section["runtime_backed"] is False
        if section["runtime_backed"] is False:
            assert section["status"] in {"partial", "static_mock_only", "missing", "unknown"}, name


def test_live_safety_is_not_marked_as_complete() -> None:
    payload = _load_report()
    sections = payload["sections"]
    assert isinstance(sections, dict)
    safety = sections["live_safety_hard_gate"]
    assert safety["paper_only_execution_safe"] is True
    assert safety["status"] == "partial"
    assert safety["status"] != "functional"
    assert safety["supports_test_server"] is False
    assert safety["supports_read_only_real_data"] is False
    assert "tests/scripts/test_preview_process_safety_hard_gate.py" in safety["evidence_files"]
    assert "tests/test_live_execution_router_preview_safety_guard.py" in safety["evidence_files"]
    caution_text = "\n".join([*safety["gaps"], safety["recommended_next_step"]])
    caution_text_lower = caution_text.lower()
    assert "subprocess/source/payload" in caution_text_lower
    assert "LiveExecutionRouter" in caution_text
    assert "DI canary" in caution_text
    assert "disabled/test-mode" in caution_text_lower
    assert "not a full end-to-end preview proof" in caution_text_lower


def test_runtime_session_control_plane_includes_frontend_parity_11_evidence() -> None:
    payload = _load_report()
    section = payload["sections"]["runtime_session_control_plane"]
    evidence = set(section["evidence_files"])
    assert {
        "ui/pyside_app/smoke.py",
        "ui/pyside_app/qml/MainWindow.qml",
        "tests/ui_pyside/test_source_smoke.py",
    }.issubset(evidence)
    assert section["status"] == "partial"
    assert section["supports_test_server"] is False
    assert section["supports_read_only_real_data"] is False


def test_runtime_session_control_next_step_stays_non_live() -> None:
    payload = _load_report()
    step = payload["sections"]["runtime_session_control_plane"]["recommended_next_step"].lower()
    assert "mock heartbeat" in step
    assert "worker states" in step
    assert "without live scheduler" in step
    assert "live workers" in step
    assert "reconnect" in step
    assert "recovery side effects" in step
    assert "start live" not in step


def test_no_functional_status_without_zero_gaps() -> None:
    payload = _load_report()
    for name, section in payload["sections"].items():
        if section["status"] == "functional":
            assert section["gaps"] == [], name


def test_test_server_and_read_only_flags_need_explicit_tested_source() -> None:
    payload = _load_report()
    for name, section in payload["sections"].items():
        joined = "\n".join(
            [*section["evidence_files"], *section["gaps"], section["recommended_next_step"]]
        ).lower()
        if section["supports_test_server"]:
            assert "test-server" in joined or "test server" in joined or "sandbox" in joined, name
            assert "test" in "\n".join(section["evidence_files"]).lower(), name
        if section["supports_read_only_real_data"]:
            assert "read-only" in joined or "read only" in joined, name
            assert "test" in "\n".join(section["evidence_files"]).lower(), name


def test_settings_and_strategy_frontend_parity_evidence_is_not_overstated() -> None:
    payload = _load_report()
    settings = payload["sections"]["settings_config_api_keys"]
    strategy = payload["sections"]["strategy_model_backtest_replay"]
    for section in (settings, strategy):
        assert section["status"] in {"partial", "static_mock_only"}
        assert section["supports_test_server"] is False
        assert section["supports_read_only_real_data"] is False
        assert any("FRONTEND-PARITY" in gap for gap in section["gaps"])
    assert strategy["static_qml_only"] is True
    assert strategy["runtime_backed"] is False


def test_preview_mode_contract_includes_block_c_controlled_ui_state_proof() -> None:
    payload = _load_report()
    section = payload["sections"]["preview_mode_contract"]
    evidence = set(section["evidence_files"])
    assert {
        "ui/pyside_app/preview_read_only_binding.py",
        "ui/pyside_app/smoke.py",
        "ui/pyside_app/qml/views/OperatorDashboard.qml",
        "tests/ui_pyside/test_preview_read_only_binding.py",
        "tests/ui_pyside/test_source_smoke.py",
    }.issubset(evidence)
    assert section["status"] == "partial"
    assert section["runtime_backed"] is False
    assert section["supports_read_only_real_data"] is False
    joined = "\n".join([*section["gaps"], section["recommended_next_step"]])
    assert (
        "BLOK C now maps PreviewReadOnlyBindingSnapshot into controlled read-only UI state values"
        in joined
    )
    assert "BLOK C QML panel now consumes controlled read-only UI state values" in joined
    assert "safe blocked/false/none/zero fallbacks" in joined
    assert "BLOK C now has a read-only UI state boundary/no-action matrix" in joined
    assert "matrix refuses action handlers, command dispatch, lifecycle execution" in joined
    assert "source smoke confirms controlled QML value consumption" in joined
    assert "values originate from PreviewReadOnlyBindingSnapshot via controlled UI state" in joined
    assert "UI proof does not start runtime loop" in joined
    assert "integration gate remains blocked" in joined
    assert "ready_for_ui_runtime_integration remains false" in joined
    assert "values copied from static-local BLOK B closure evidence" in joined
    assert (
        "state does not expose commands/actions/export/live/testnet/account/secret handles"
        in joined
    )
    assert "QML/source proof remains read-only" in joined
    assert (
        "next step should be read-only UI state runtime bridge preflight or controlled PySide bridge exposure, still without actions"
        in joined
    )


def test_data_source_market_feed_does_not_use_generic_pyside_live_shape_as_feed_proof() -> None:
    payload = _load_report()
    section = payload["sections"]["data_source_market_feed"]
    evidence = set(section["evidence_files"])
    assert "ui/pyside_app/smoke.py" not in evidence
    assert "ui/pyside_app/qml/MainWindow.qml" not in evidence
    assert "tests/ui_pyside/test_source_smoke.py" not in evidence
    assert section["status"] == "partial"
    assert section["supports_test_server"] is False
    assert section["supports_read_only_real_data"] is False


def test_strategy_model_backtest_replay_evidence_files_are_existing_and_tracked() -> None:
    payload = _load_report()
    section = payload["sections"]["strategy_model_backtest_replay"]
    evidence = section["evidence_files"]
    assert evidence == [
        "ui/pyside_app/smoke.py",
        "ui/pyside_app/qml/MainWindow.qml",
        "tests/ui_pyside/test_source_smoke.py",
        "ui/qml/views/StrategyExperience.qml",
    ]
    assert "ui/qml/views/Strategies.qml" not in evidence
    assert all((REPO_ROOT / item).exists() for item in evidence)


def test_functional_preview_3_scope_remains_local_unit_only() -> None:
    payload = _load_report()
    scope = payload["scope"]
    assert "FUNCTIONAL-PREVIEW-3.16" in scope
    for phrase in (
        "local paper event spine",
        "read-only market data contract unit evidence",
        "deterministic local context/artifact/audit bundle",
        "local preview bundle read model evidence",
        "local read model boundary refusal matrix evidence",
        "local/static UI/runtime preflight audit evidence",
        "local/static integration readiness checklist gate evidence",
    ):
        assert phrase in scope
    assert (
        "integration readiness checklist gate blocks transition to real UI/runtime while preflight has blocking checks; BLOK A is contractually complete but not functional UI/runtime integration; no runtime loop, UI integration, file loader/export, secrets, real market fetches, live account access, cloud/export sink, external export, serialization export, engine handoff, DecisionEnvelope handoff, TradingController handoff, order generation, or live order I/O executed"
        in scope
    )


def test_runtime_and_strategy_sections_keep_conservative_status() -> None:
    payload = _load_report()
    runtime = payload["sections"]["runtime_session_control_plane"]
    strategy = payload["sections"]["strategy_model_backtest_replay"]
    assert runtime["status"] == "partial"
    assert runtime["status"] != "functional"
    assert runtime["supports_test_server"] is False
    assert runtime["supports_read_only_real_data"] is False
    assert strategy["status"] == "static_mock_only"
    assert strategy["runtime_backed"] is False
    assert strategy["supports_test_server"] is False
    assert strategy["supports_read_only_real_data"] is False


def test_preview_mode_contract_evidence_is_static_and_not_mock_only() -> None:
    payload = _load_report()
    section = payload["sections"]["preview_mode_contract"]
    assert section["status"] == "partial"
    assert section["runtime_backed"] is False
    assert section["static_qml_only"] is False
    assert section["supports_test_server"] is False
    assert section["supports_read_only_real_data"] is False
    assert "bot_core/runtime/preview_modes.py" in section["evidence_files"]
    assert "tests/runtime/test_preview_modes.py" in section["evidence_files"]
    text = "\n".join([*section["gaps"], section["recommended_next_step"]]).lower()
    assert "static mode contract exists" in text
    assert "enforcement helper exists" in text
    assert "preview is not mock-only" in text
    assert "paper" in text
    assert "testnet" in text or "sandbox" in text
    assert "read_only_market" in text or "read-only feed" in text
    assert "real runtime implementations/proofs" in text
    assert "live-production capabilities are blocked" in text
    assert "live account balance fetch" in text
    assert "live account snapshot read" in text


def test_paper_terminal_order_lifecycle_includes_local_spine_without_overstatement() -> None:
    payload = _load_report()
    section = payload["sections"]["paper_terminal_order_lifecycle"]
    assert section["status"] == "partial"
    assert section["runtime_backed"] is False
    assert section["supports_test_server"] is False
    assert section["supports_read_only_real_data"] is False
    assert section["paper_only_execution_safe"] is True
    assert "bot_core/runtime/paper_event_spine.py" in section["evidence_files"]
    assert "tests/runtime/test_paper_event_spine.py" in section["evidence_files"]
    text = "\n".join([*section["gaps"], section["recommended_next_step"]]).lower()
    assert "local paper event spine exists" in text
    assert "no live exchange/order/account side effects" in text
    assert "scenario fixture runner" in text
    assert "external export" in text
    assert "local paper portfolio reducer now exists" in text
    assert "local paper audit/alerts consumer now exists" in text
    assert "ui/runtime integration still missing" in text
    assert "testnet" in text
    assert "read-only market feed" in text
    assert "local scenario fixture runner exists" in text
    assert "deterministic in-memory" in text


def test_portfolio_positions_trades_includes_local_reducer_without_overstatement() -> None:
    payload = _load_report()
    section = payload["sections"]["portfolio_positions_trades"]
    assert section["status"] == "partial"
    assert section["runtime_backed"] is False
    assert section["static_qml_only"] is False
    assert section["supports_test_server"] is False
    assert section["supports_read_only_real_data"] is False
    assert section["paper_only_execution_safe"] is True
    assert "bot_core/runtime/paper_portfolio_reducer.py" in section["evidence_files"]
    assert "tests/runtime/test_paper_portfolio_reducer.py" in section["evidence_files"]
    text = "\n".join([*section["gaps"], section["recommended_next_step"]]).lower()
    assert "local paper portfolio reducer exists" in text
    assert "paper fills produce deterministic trades/positions" in text
    assert "non-fill events do not mutate portfolio" in text
    assert "app runtime/ui integration still missing" in text
    assert "local paper audit/alerts consumer" in text
    assert "testnet/read-only market feed" in text
    assert "live exchange/order/account" in text
    assert "scenario runner" in text
    assert "file loader/export" in text


def test_alerts_telemetry_audit_is_local_unit_evidence_only() -> None:
    payload = _load_report()
    section = payload["sections"]["alerts_telemetry_audit"]
    assert section["status"] == "partial"
    assert section["runtime_backed"] is False
    assert section["static_qml_only"] is False
    assert section["supports_test_server"] is False
    assert section["supports_read_only_real_data"] is False
    assert section["paper_only_execution_safe"] is True
    assert "bot_core/runtime/paper_audit_journal.py" in section["evidence_files"]
    assert "tests/runtime/test_paper_audit_journal.py" in section["evidence_files"]
    text = "\n".join([*section["gaps"], section["recommended_next_step"]]).lower()
    assert "local paper audit/alerts consumer exists" in text
    assert "paperorderevent" in text
    assert "papertrade" in text
    assert "no cloud sink" in text
    assert "no external export" in text
    assert "no runtime loop" in text
    assert "app runtime/ui integration still missing" in text
    assert "testnet/read-only market feed still missing" in text
    assert "no live exchange/order/account side effects" in text
    assert "scenario fixture runner" in text
    assert "external export" in text


def test_read_only_market_contract_is_static_local_evidence_only() -> None:
    payload = _load_report()
    preview = payload["sections"]["preview_mode_contract"]
    feed = payload["sections"]["data_source_market_feed"]

    for section in (preview, feed):
        assert section["status"] == "partial"
        assert (
            section["runtime_backed"] is False
            if section is feed
            else section["runtime_backed"] is False
        )
        assert section["static_qml_only"] is False
        assert section["supports_test_server"] is False
        assert section["supports_read_only_real_data"] is False
        assert section["paper_only_execution_safe"] is True
        assert "bot_core/runtime/read_only_market_data.py" in section["evidence_files"]
        assert "tests/runtime/test_read_only_market_data.py" in section["evidence_files"]
        assert "bot_core/runtime/paper_preview_scenario.py" in section["evidence_files"]
        assert "tests/runtime/test_paper_preview_scenario.py" in section["evidence_files"]

    joined = "\n".join([*feed["gaps"], feed["recommended_next_step"]]).lower()
    assert "read-only market data contract" in joined
    assert "read_only_market_fetch" in joined
    assert "account/balance/credentials/order/fill/live side effects remain blocked" in joined
    assert "no real market adapter/fetch" in joined
    assert "app runtime loop" in joined
    assert "ui integration" in joined
    assert "testnet/sandbox adapter" in joined
    assert "cloud sink" in joined
    assert "external export" in joined
    assert "scenario runner can carry deterministic read-only market context" in joined
    assert "in-memory/static-local fixture" in joined


def test_decision_dry_run_artifact_readiness_evidence_stays_partial_static_local() -> None:
    payload = _load_report()
    sections = payload["sections"]
    for name in (
        "ai_decision_governor",
        "paper_terminal_order_lifecycle",
        "data_source_market_feed",
    ):
        section = sections[name]
        joined = "\n".join(
            [*section["evidence_files"], *section["gaps"], section["recommended_next_step"]]
        ).lower()
        assert section["status"] == "partial"
        assert section["runtime_backed"] is False
        assert section["supports_read_only_real_data"] is False
        assert "paper_preview_scenario.py" in joined
        assert "dry-run decision artifact" in joined
        assert "context-only" in joined
        assert "generate no orders/decisions" in joined or "generates no orders/decisions" in joined
        assert "no scoring" in joined
        assert "no recommendation" in joined
        assert "no strategy engine" in joined
        assert "ai/model inference" in joined
        assert "decisionenvelope integration" in joined
        assert "tradingcontroller integration" in joined


def test_bundle_boundary_refusal_readiness_evidence_stays_static_local() -> None:
    payload = _load_report()
    sections = payload["sections"]
    for name in (
        "ai_decision_governor",
        "alerts_telemetry_audit",
        "preview_mode_contract",
        "paper_terminal_order_lifecycle",
        "data_source_market_feed",
    ):
        section = sections[name]
        joined = "\n".join(
            [*section["evidence_files"], *section["gaps"], section["recommended_next_step"]]
        ).lower()
        assert section["status"] == "partial"
        assert section["runtime_backed"] is False
        assert section["supports_read_only_real_data"] is False
        assert "paper_preview_bundle_boundary.py" in joined
        assert "bundle boundary/export refusal" in joined
        assert "static/local" in joined or "static-local" in joined
        assert "file export" in joined
        assert "serialization export" in joined
        assert "cloud sink" in joined
        assert "external export" in joined
        assert "engine handoff" in joined
        assert (
            "no generated orders/decisions" in joined
            or "generate no orders/decisions" in joined
            or "generates no orders/decisions" in joined
        )
        assert "no scoring" in joined
        assert "no recommendation" in joined
        assert "decisionenvelope" in joined
        assert "tradingcontroller" in joined


def test_bundle_read_model_readiness_evidence_stays_partial_static_local() -> None:
    payload = _load_report()
    sections = payload["sections"]
    for name in (
        "ai_decision_governor",
        "alerts_telemetry_audit",
        "preview_mode_contract",
        "paper_terminal_order_lifecycle",
        "data_source_market_feed",
    ):
        section = sections[name]
        joined = "\n".join(
            [*section["evidence_files"], *section["gaps"], section["recommended_next_step"]]
        ).lower()
        assert section["status"] == "partial"
        assert section["runtime_backed"] is False
        assert section["supports_read_only_real_data"] is False
        assert "paper_preview_bundle_read_model.py" in joined
        assert "read model" in joined
        assert "future ui/runtime integration" in joined
        assert (
            "read model boundary matrix refuses qml/pyside/ui/runtime/export/cloud/engine" in joined
        )
        assert "local/static" in joined or "static-local" in joined
        assert "not qml/pyside/ui-bound" in joined or "no ui integration" in joined
        assert "not runtime-backed" in joined or "no app runtime loop" in joined
        assert "no file export" in joined
        assert "no serialization export" in joined
        assert "no cloud sink" in joined
        assert "no external export" in joined
        assert "no scoring" in joined
        assert "no recommendation" in joined
        assert "decisionenvelope integration" in joined
        assert "tradingcontroller integration" in joined


def test_ui_runtime_preflight_readiness_evidence_stays_partial_static_local() -> None:
    payload = _load_report()
    sections = payload["sections"]
    for name in (
        "ai_decision_governor",
        "alerts_telemetry_audit",
        "preview_mode_contract",
        "paper_terminal_order_lifecycle",
        "data_source_market_feed",
    ):
        section = sections[name]
        joined = "\n".join(
            [*section["evidence_files"], *section["gaps"], section["recommended_next_step"]]
        ).lower()
        assert section["status"] == "partial"
        assert section["runtime_backed"] is False
        assert section["supports_read_only_real_data"] is False
        assert "paper_preview_ui_runtime_preflight.py" in joined
        assert (
            "ui/runtime preflight audit lists missing requirements before real integration"
            in joined
        )
        assert "preflight/read model/matrix are local/static-only" in joined
        assert "preflight is not qml/pyside/ui-bound" in joined
        assert "preflight is not runtime-backed" in joined
        assert "generate no orders/decisions" in joined
        assert "no scoring" in joined
        assert "no recommendation" in joined
        assert "no strategy engine" in joined
        assert "no ai/model inference" in joined
        assert "no decisionenvelope integration" in joined
        assert "no tradingcontroller integration" in joined
        assert "no file export" in joined
        assert "no serialization export" in joined
        assert "no cloud sink" in joined
        assert "no external export" in joined
        assert "no app runtime loop" in joined
        assert "no ui integration" in joined
        assert "no testnet/sandbox adapter" in joined


def test_service_snapshot_history_readiness_evidence_stays_partial_static_local() -> None:
    payload = _load_report()
    sections = payload["sections"]
    for name in (
        "ai_decision_governor",
        "alerts_telemetry_audit",
        "preview_mode_contract",
        "paper_terminal_order_lifecycle",
        "data_source_market_feed",
    ):
        section = sections[name]
        joined = "\n".join(
            [*section["evidence_files"], *section["gaps"], section["recommended_next_step"]]
        ).lower()
        assert section["status"] == "partial"
        assert section["runtime_backed"] is False
        assert section["static_qml_only"] is False
        assert section["supports_test_server"] is False
        assert section["supports_read_only_real_data"] is False
        assert section["paper_only_execution_safe"] is True
        assert "paper_preview_runtime_service_history.py" in joined
        assert (
            "service snapshot history contract keeps a bounded immutable in-memory history"
            in joined
        )
        assert "already-built read api evidence" in joined
        assert "history does not collect in background" in joined
        assert "run loops" in joined
        assert "serialize/export/write files/open sockets" in joined
        assert "execute commands" in joined
        assert "refusal executor proof remains non-executing" in joined
        assert "read api remains immutable in-memory projection only" in joined
        assert "wrapper remains single-shot/static-local" in joined
        assert "no app runtime loop" in joined
        assert "no ui binding" in joined
        assert "no controller handoff" in joined
        assert "no decision engine" in joined
        assert "no export/cloud/serialization" in joined
        assert "no generated decisions/orders" in joined
        assert "no real market adapter/fetch" in joined
        assert "no testnet/sandbox adapter" in joined
        assert (
            "blok c now maps previewreadonlybindingsnapshot into controlled read-only ui state values"
            in joined
        )


def test_block_b_closure_readiness_evidence_stays_contract_complete_static_local() -> None:
    payload = _load_report()
    sections = payload["sections"]
    for name in (
        "ai_decision_governor",
        "alerts_telemetry_audit",
        "preview_mode_contract",
        "paper_terminal_order_lifecycle",
        "data_source_market_feed",
    ):
        section = sections[name]
        joined = "\n".join(
            [*section["evidence_files"], *section["gaps"], section["recommended_next_step"]]
        ).lower()
        assert section["status"] == "partial"
        assert section["runtime_backed"] is False
        assert section["static_qml_only"] is False
        assert section["supports_test_server"] is False
        assert section["supports_read_only_real_data"] is False
        assert section["paper_only_execution_safe"] is True
        assert "paper_preview_runtime_service_closure.py" in joined
        assert "contract-complete for static-local evidence" in joined
        assert "closure audit aggregates service snapshot" in joined
        assert "ready_for_block_c=true" in joined
        assert "ready_for_ui_runtime_integration=false" in joined
        assert "ready_for_decision_engine=false" in joined
        assert "ready_for_export=false" in joined
        assert "ready_for_live=false" in joined
        assert "integration gate remains blocked" in joined
        assert "wrapper remains single-shot/static-local" in joined
        assert "no app runtime loop" in joined
        assert "no ui binding" in joined
        assert "no controller handoff" in joined
        assert "no decision engine" in joined
        assert "no export/cloud/serialization" in joined
        assert "no generated decisions/orders" in joined
        assert "no real market adapter/fetch" in joined
        assert "no testnet/sandbox adapter" in joined
        assert (
            "blok c now maps previewreadonlybindingsnapshot into controlled read-only ui state values"
            in joined
        )
