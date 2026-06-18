#!/usr/bin/env python3
"""Emit a deterministic functional-preview readiness audit.

This script is intentionally static/read-only: it does not import runtime UI modules,
open network connections, read secrets, or start preview/runtime loops.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
STATUSES = {"functional", "partial", "static_mock_only", "missing", "unknown"}
REQUIRED_KEYS = {
    "status",
    "evidence_files",
    "runtime_backed",
    "static_qml_only",
    "supports_test_server",
    "supports_read_only_real_data",
    "paper_only_execution_safe",
    "gaps",
    "recommended_next_step",
}


def _existing(paths: list[str]) -> list[str]:
    """Return only evidence files that exist, preserving deterministic order."""

    return [path for path in paths if (REPO_ROOT / path).exists()]


def build_report() -> dict[str, Any]:
    """Build the brutal, non-promotional preview functionality classification."""

    sections: dict[str, dict[str, Any]] = {
        "preview_mode_contract": {
            "status": "partial",
            "evidence_files": _existing(
                [
                    "bot_core/runtime/preview_modes.py",
                    "tests/runtime/test_preview_modes.py",
                    "bot_core/runtime/read_only_market_data.py",
                    "tests/runtime/test_read_only_market_data.py",
                    "bot_core/runtime/paper_preview_scenario.py",
                    "bot_core/runtime/paper_preview_bundle_boundary.py",
                    "bot_core/runtime/paper_preview_bundle_read_model.py",
                    "bot_core/runtime/paper_preview_ui_runtime_preflight.py",
                    "tests/runtime/test_paper_preview_scenario.py",
                ]
            ),
            "runtime_backed": False,
            "static_qml_only": False,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Static mode contract exists and states that preview is not mock-only: local_mock, recorded replay, paper, testnet/sandbox, and read_only_market are allowed by policy.",
                "Enforcement helper exists and accepts enum/string inputs while failing closed for unknown or ambiguous preview mode/capability declarations.",
                "Paper, testnet/sandbox, read-only market, recorded replay, and local telemetry audit capabilities are allowed by policy.",
                "Live-production capabilities are blocked by policy, including real exchange orders/fills, live account balance mutation, live account balance fetch, live account snapshot read, live credentials, production cloud telemetry, external export, and live scheduler/worker effects.",
                "Read-only market policy is distinct from live-production account/balance access; this is static/read-only policy evidence only, and real runtime implementations/proofs for paper spine, read-only feed, and testnet execution still remain.",
                "Read-only market data contract exists as local/static unit evidence and uses READ_ONLY_MARKET_FETCH while account, balance, credentials, order, fill, cloud/export, and live scheduler side effects remain blocked.",
                "Local scenario runner can carry deterministic read-only market context from in-memory/static-local fixtures and can produce a deterministic local context/artifact/audit bundle plus fail-closed bundle boundary/export refusal and local bundle boundary matrix, and local read model can summarize bundle + boundary matrix for future UI/runtime integration; read model boundary matrix refuses QML/PySide/UI/runtime/export/cloud/engine boundaries without treating it as live account/balance/order access and with no generated orders/decisions, no scoring, no recommendation, no DecisionEnvelope integration, and no TradingController integration; bundle boundary matrix shows all export/serialization/cloud/engine handoff boundaries are refused; matrix/report/read model is local/static-only; read model is not QML/PySide/UI-bound; read model is not runtime-backed; file export, serialization export, cloud sink, external export, and engine handoff remain refused.",
            ],
            "recommended_next_step": "Keep the static mode contract and enforcement helper as the foundation that allows paper/testnet/read-only/recorded capabilities while blocking live-production capabilities; the local scenario runner can carry deterministic read-only market context from in-memory/static-local fixtures, READ_ONLY_MARKET_FETCH remains preview-safe, and account/balance/credentials/order/fill/live side effects remain blocked; no real market adapter/fetch yet, no app runtime loop, UI integration, testnet/sandbox adapter, file export, serialization export, cloud sink, external export, serialization payload, or engine handoff exists yet.",
        },
        "data_source_market_feed": {
            "status": "partial",
            "evidence_files": _existing(
                [
                    "ui/config/preview_local.yaml",
                    "ui/backend/runtime_service.py",
                    "ui/backend/demo_data.py",
                    "ui/src/grpc/TradingClient.cpp",
                    "config/e2e/core_loopback.yaml",
                    "tests/scripts/test_run_rest_market_data_poller_script.py",
                    "bot_core/runtime/read_only_market_data.py",
                    "tests/runtime/test_read_only_market_data.py",
                    "bot_core/runtime/paper_preview_scenario.py",
                    "bot_core/runtime/paper_preview_bundle_boundary.py",
                    "bot_core/runtime/paper_preview_bundle_read_model.py",
                    "bot_core/runtime/paper_preview_ui_runtime_preflight.py",
                    "tests/runtime/test_paper_preview_scenario.py",
                ]
            ),
            "runtime_backed": False,
            "static_qml_only": False,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Preview profile is explicitly in-process/local and points at a sample OHLCV dataset, not a test-server or read-only real-market source.",
                "A preview-scoped read-only market data contract now exists for static/local in-memory quote/candle/snapshot proof, but it is not a real adapter or real feed.",
                "Local scenario runner can carry deterministic read-only market context from in-memory/static-local fixtures before paper step execution.",
                "Local scenario runner can produce deterministic context-only dry-run decision artifact, local in-memory artifact audit trail, and deterministic local context/artifact/audit bundle plus fail-closed bundle boundary/export refusal local bundle boundary matrix, and local read model after execution; artifact, bundle, matrix, and read model are context-only/static-local and generate no orders/decisions, with no scoring, no recommendation, no strategy engine, AI/model inference, DecisionEnvelope integration, or TradingController integration.",
                "gRPC/UI fallback code can expose runtime-shaped snapshots, but this audit found no real-data-backed preview proof of read-only feed ingestion.",
                "No real market adapter/fetch, app runtime loop, UI integration, testnet/sandbox adapter, cloud sink, external export, serialization payload, or engine handoff is implemented for this contract.",
            ],
            "recommended_next_step": "Keep the read-only market data contract, scenario market context, scenario decision dry-run artifact, local artifact audit trail, and local context/artifact/audit bundle plus fail-closed bundle boundary/export refusal, local bundle boundary matrix, and local read model as partial/static-local evidence: the local scenario runner can carry deterministic read-only market context and produce deterministic local context/artifact/audit bundle plus fail-closed bundle boundary/export refusal, local bundle boundary matrix, and local read model that generate no orders/decisions, market context is in-memory/static-local fixture only, READ_ONLY_MARKET_FETCH is preview-safe, and account/balance/credentials/order/fill/live side effects remain blocked; no scoring, no recommendation, no strategy engine, AI/model inference, DecisionEnvelope integration, TradingController integration, file export, serialization export, cloud sink, external export, real market adapter/fetch, app runtime loop, UI integration, or testnet/sandbox adapter exists yet.",
        },
        "scanner_opportunity_pipeline": {
            "status": "static_mock_only",
            "evidence_files": _existing(
                [
                    "ui/backend/runtime_service.py",
                    "ui/backend/demo_data.py",
                    "ui/qml/views/StrategyExperience.qml",
                    "ui/qml/views/AnalyticsDashboard.qml",
                    "tests/scripts/test_controller_mock_preview.py",
                ]
            ),
            "runtime_backed": False,
            "static_qml_only": True,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Candidates visible in preview are not proven to originate from candles/ticker/orderbook scanner runtime.",
                "Controller mock preview processes bounded synthetic signals and explicitly does not start the runtime loop.",
                "No test-server path for scanner candidates was identified in preview scope.",
            ],
            "recommended_next_step": "Introduce a scanner preview adapter fed by recorded/read-only market snapshots, then assert candidate provenance in CI.",
        },
        "ai_decision_governor": {
            "status": "partial",
            "evidence_files": _existing(
                [
                    "ui/backend/runtime_service.py",
                    "ui/backend/ai_governor_demo.py",
                    "ui/backend/demo_data.py",
                    "tests/test_run_decision_engine_smoke_script.py",
                    "data/decision_engine/paper/candidates.json",
                    "bot_core/runtime/paper_preview_scenario.py",
                    "bot_core/runtime/paper_preview_bundle_boundary.py",
                    "bot_core/runtime/paper_preview_bundle_read_model.py",
                    "bot_core/runtime/paper_preview_ui_runtime_preflight.py",
                    "tests/runtime/test_paper_preview_scenario.py",
                    "bot_core/runtime/read_only_market_data.py",
                    "tests/runtime/test_read_only_market_data.py",
                ]
            ),
            "runtime_backed": False,
            "static_qml_only": False,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Preview decision rows can be loaded from runtime journal/stream plumbing, but default first-run data is a static demo snapshot.",
                "Paper decision-engine smoke inputs are file fixtures, not a preview proof of live read-only market input.",
                "Confidence/reason/risk fields in the preview default are supplied as serialized demo values unless a runtime source is configured.",
                "Local scenario runner can produce a deterministic local context/artifact/audit bundle plus fail-closed bundle boundary/export refusal and local bundle boundary matrix after paper scenario execution, including scenario state, optional in-memory/static-local market context summary, decision context, dry-run artifact, local artifact audit trail, no-action markers, no-export/no-cloud markers, and blocked engine integration markers.",
                "Dry-run artifact, local bundle, bundle boundary matrix, and local read model are context-only/static-local and generate no orders/decisions; they are not scoring, recommendation, strategy engine, AI/model inference path, DecisionEnvelope integration, TradingController integration, file export, serialization export, cloud sink, external export, or engine handoff.",
                "Market context is in-memory/static-local fixture only; no real market adapter/fetch yet, no app runtime loop, UI integration, testnet/sandbox adapter, file export, serialization export, cloud sink, external export, serialization payload, or engine handoff exists yet.",
            ],
            "recommended_next_step": "Keep ai_decision_governor partial/static-local: local scenario runner can produce deterministic context-only dry-run decision artifact and local in-memory audit trail, dry-run artifact audit trail, bundle boundary matrix, and local read model are static-local only and generate no orders/decisions, no scoring, no recommendation, no strategy engine, no AI/model inference, no DecisionEnvelope integration, no TradingController integration, market context is in-memory/static-local fixture only, READ_ONLY_MARKET_FETCH is preview-safe, and account/balance/credentials/order/fill/live side effects remain blocked; no real market adapter/fetch yet, no app runtime loop, UI integration, testnet/sandbox adapter, file export, serialization export, cloud sink, external export, serialization payload, or engine handoff exists yet.",
        },
        "paper_terminal_order_lifecycle": {
            "status": "partial",
            "evidence_files": _existing(
                [
                    "scripts/controller_mock_preview.py",
                    "scripts/mock_runtime_preview.py",
                    "tests/scripts/test_controller_mock_preview.py",
                    "tests/scripts/test_mock_runtime_preview.py",
                    "tests/test_paper_execution.py",
                    "bot_core/exchanges/base.py",
                    "bot_core/runtime/paper_event_spine.py",
                    "tests/runtime/test_paper_event_spine.py",
                    "bot_core/runtime/paper_preview_flow.py",
                    "tests/runtime/test_paper_preview_flow.py",
                    "bot_core/runtime/paper_preview_scenario.py",
                    "bot_core/runtime/paper_preview_bundle_boundary.py",
                    "bot_core/runtime/paper_preview_bundle_read_model.py",
                    "bot_core/runtime/paper_preview_ui_runtime_preflight.py",
                    "tests/runtime/test_paper_preview_scenario.py",
                ]
            ),
            "runtime_backed": False,
            "static_qml_only": False,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Local paper event spine exists with deterministic accepted, rejected, partial fill, fill, and cancel unit-level lifecycle tests and no live exchange/order/account side effects.",
                "Local paper portfolio reducer now exists as separate unit-level evidence; app runtime/UI integration still missing.",
                "Local paper audit/alerts consumer now exists as separate unit-level evidence and consumes paper order/trade events locally without cloud/export/live side effects.",
                "Local composition proof exists: paper spine, portfolio reducer, and audit journal are wired together locally for submit/reject/cancel/partial/fill snapshot evidence.",
                "Local scenario fixture runner exists and drives PaperPreviewFlow with deterministic in-memory scenarios; it can carry deterministic read-only market context from an in-memory/static-local fixture; no file loader/export is implemented.",
                "Local scenario runner can produce deterministic context-only dry-run decision artifact, local in-memory artifact audit trail, and deterministic local context/artifact/audit bundle plus fail-closed bundle boundary/export refusal local bundle boundary matrix, and local read model after execution; dry-run artifact, bundle, matrix, and read model are context-only/static-local and generate no orders/decisions with no scoring, no recommendation, no strategy engine, AI/model inference, DecisionEnvelope integration, or TradingController integration.",
                "UI/runtime integration still missing; this is not app-runtime-backed evidence and no runtime loop is started.",
                "No cloud sink, no external export, no testnet/read-only market feed, and no live exchange/order/account side effects are proven or introduced here.",
                "Testnet execution and read-only market feed still missing; this paper spine does not fetch market data or use sandbox/testnet adapters.",
            ],
            "recommended_next_step": "Keep the local composition proof partial: paper spine + portfolio reducer + audit journal are wired together locally, a deterministic in-memory local scenario fixture runner now drives PaperPreviewFlow, and the runner can produce a context-only dry-run decision artifact plus local in-memory audit trail and local context/artifact/audit bundle plus fail-closed bundle boundary/export refusal, local bundle boundary matrix, and local read model with no generated orders/decisions, but there is no scoring, no recommendation, no strategy engine, no AI/model inference, no DecisionEnvelope integration, no TradingController integration, no app runtime loop, no UI integration, no file loader/export, no file export, no serialization export, no cloud sink, no external export, no engine handoff, no testnet/sandbox adapter, no real market adapter/fetch, and no live exchange/order/account side effects.",
        },
        "portfolio_positions_trades": {
            "status": "partial",
            "evidence_files": _existing(
                [
                    "bot_core/runtime/paper_portfolio_reducer.py",
                    "tests/runtime/test_paper_portfolio_reducer.py",
                    "bot_core/runtime/paper_preview_scenario.py",
                    "bot_core/runtime/paper_preview_bundle_boundary.py",
                    "bot_core/runtime/paper_preview_bundle_read_model.py",
                    "bot_core/runtime/paper_preview_ui_runtime_preflight.py",
                    "tests/runtime/test_paper_preview_scenario.py",
                    "bot_core/runtime/paper_preview_flow.py",
                    "tests/runtime/test_paper_preview_flow.py",
                    "bot_core/runtime/paper_event_spine.py",
                    "tests/runtime/test_paper_event_spine.py",
                    "ui/qml/views/PortfolioDashboard.qml",
                    "ui/qml/components/PortfolioManagerView.qml",
                    "ui/src/grpc/TradingClient.cpp",
                    "ui/src/app/PortfolioManagerController.cpp",
                ]
            ),
            "runtime_backed": False,
            "static_qml_only": False,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Local paper portfolio reducer exists and consumes PaperOrderEvent fill events as static/local unit evidence.",
                "Paper fills produce deterministic trades/positions and basic realized PnL for long closes without live exchange/order/account side effects.",
                "Rejected, cancelled, accepted, and other non-fill paper events do not mutate portfolio trades/positions.",
                "App runtime/UI integration still missing; this is not app-runtime-backed portfolio evidence.",
                "Local paper audit/alerts consumer now consumes PaperTrade records as static/local unit evidence without mutating portfolio state.",
                "Local composition proof exists and wires paper spine + portfolio reducer + audit journal together locally; this remains unit evidence, not app-runtime-backed proof.",
                "Local scenario fixture runner exists and produces deterministic PaperPreviewFlow snapshots/summaries from in-memory steps without a file loader or export.",
                "Testnet/read-only market feed still missing; no test-server, sandbox, read-only real-data, live account, or balance path is proven here.",
            ],
            "recommended_next_step": "Keep the local paper portfolio reducer partial: paper fills produce deterministic trades/positions, non-fill events do not mutate portfolio, local composition proof wires trades/positions to audit locally, and a deterministic in-memory scenario runner exists, but there is no app runtime loop, no UI integration, no file loader/export, no file export, no serialization export, no cloud sink, no external export, no testnet/read-only market feed, and no live exchange/order/account side effects.",
        },
        "alerts_telemetry_audit": {
            "status": "partial",
            "evidence_files": _existing(
                [
                    "bot_core/runtime/paper_audit_journal.py",
                    "tests/runtime/test_paper_audit_journal.py",
                    "bot_core/runtime/paper_preview_flow.py",
                    "tests/runtime/test_paper_preview_flow.py",
                    "bot_core/runtime/paper_event_spine.py",
                    "tests/runtime/test_paper_event_spine.py",
                    "bot_core/runtime/paper_portfolio_reducer.py",
                    "tests/runtime/test_paper_portfolio_reducer.py",
                    "bot_core/runtime/paper_preview_scenario.py",
                    "bot_core/runtime/paper_preview_bundle_boundary.py",
                    "bot_core/runtime/paper_preview_bundle_read_model.py",
                    "bot_core/runtime/paper_preview_ui_runtime_preflight.py",
                    "tests/runtime/test_paper_preview_scenario.py",
                ]
            ),
            "runtime_backed": False,
            "static_qml_only": False,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Local paper audit/alerts consumer exists and consumes PaperOrderEvent lifecycle events plus PaperTrade records locally as in-memory unit evidence.",
                "No cloud sink, no external export, no file export, and no runtime loop are implemented for the local paper audit journal.",
                "App runtime/UI integration still missing; this is static/local unit evidence, not app-runtime-backed telemetry proof.",
                "Local composition proof exists and wires paper spine + portfolio reducer + audit journal together locally without a cloud sink or external export.",
                "Local scenario fixture runner exists as static/local unit evidence and drives PaperPreviewFlow deterministically without file export, serialization export, cloud sink, external export, or engine handoff.",
                "Local scenario runner can produce a deterministic context-only dry-run artifact audit trail and fail-closed bundle boundary/export refusal and local bundle boundary matrix and local read model as local/in-memory/static-local evidence only; it is not cloud telemetry, file export, serialization export, external export, engine handoff, strategy evaluation, AI/model inference, scoring, recommendation, DecisionEnvelope integration, or TradingController integration, and it generates no orders/decisions.",
                "Testnet/read-only market feed still missing; no live exchange/order/account side effects are introduced or proven here.",
            ],
            "recommended_next_step": "Keep local paper audit/alerts consumer partial: local composition proof wires paper spine + portfolio reducer + audit journal locally, and a deterministic in-memory scenario runner plus static/local bundle boundary/export refusal, local matrix, and local read model exist, but there is no app runtime loop, no UI integration, no file loader/export, no serialization export, no cloud sink, no external export, no engine handoff, no scoring, no recommendation, no DecisionEnvelope integration, no TradingController integration, no testnet/read-only market feed, and no live exchange/order/account side effects.",
        },
        "settings_config_api_keys": {
            "status": "partial",
            "evidence_files": _existing(
                [
                    "ui/config/preview_local.yaml",
                    "config/e2e/demo_paper.yml",
                    "tests/scripts/test_mock_runtime_preview.py",
                    "tests/scripts/test_controller_mock_preview.py",
                    "tests/scripts/test_credential_reference_readiness.py",
                    "ui/backend/dashboard_settings.py",
                    "ui/pyside_app/smoke.py",
                    "ui/pyside_app/qml/MainWindow.qml",
                    "tests/ui_pyside/test_source_smoke.py",
                ]
            ),
            "runtime_backed": False,
            "static_qml_only": False,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Safe preview config disables TLS/secret endpoints and mock-preview tests assert no API keys are required.",
                "No preview config entry was found for read-only/test credentials or explicit sandbox/test-server API selection.",
                "FRONTEND-PARITY-9.0 adds settings/config/API-key live-shape proof, but it is QML/smoke/source evidence only and does not prove credential I/O, sandbox auth, or read-only account access.",
                "Live credential separation is safety-oriented but not equivalent to a functional read-only credential workflow.",
            ],
            "recommended_next_step": "After process-wide preview safety hard gate proof, define non-secret credential references for preview read-only/test-server profiles and validate that live credential keys are rejected.",
        },
        "strategy_model_backtest_replay": {
            "status": "static_mock_only",
            "evidence_files": _existing(
                [
                    "ui/pyside_app/smoke.py",
                    "ui/pyside_app/qml/MainWindow.qml",
                    "tests/ui_pyside/test_source_smoke.py",
                    "ui/qml/views/StrategyExperience.qml",
                ]
            ),
            "runtime_backed": False,
            "static_qml_only": True,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "FRONTEND-PARITY-10.0 proves strategy/model/backtest/replay live shape in PySide/QML/source-smoke only.",
                "No audited preview path trains a real model, promotes an artifact, starts live inference, fetches live market data, or executes replay against a test server.",
                "Local replay/backtest cards remain static/mock readiness evidence, not functional strategy runtime readiness.",
            ],
            "recommended_next_step": "After process-wide preview safety hard gate proof, connect strategy/model/replay to the local paper event spine and preview data-source contract using deterministic fixtures only.",
        },
        "runtime_session_control_plane": {
            "status": "partial",
            "evidence_files": _existing(
                [
                    "scripts/mock_runtime_preview.py",
                    "scripts/controller_mock_preview.py",
                    "ui/backend/runtime_service.py",
                    "ui/qml/dashboard/RuntimeOverview.qml",
                    "ui/pyside_app/smoke.py",
                    "ui/pyside_app/qml/MainWindow.qml",
                    "tests/ui_pyside/test_source_smoke.py",
                    "tests/scripts/test_mock_runtime_preview.py",
                    "tests/scripts/test_controller_mock_preview.py",
                ]
            ),
            "runtime_backed": True,
            "static_qml_only": False,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Mock runtime preview can start a safe local preview simulation, but controller mock preview explicitly reports runtime_loop_started=false.",
                "FRONTEND-PARITY-11.0 adds PySide/QML/source-smoke proof for runtime/session/control-plane live shape, including heartbeat, scheduler, worker, failover, and disabled recovery/reconnect controls; this is not runtime-loop proof.",
                "Scheduler, heartbeat, worker, recovery, reconnect, and failover controls are preview-shaped unless a separate runtime is attached; no start path is enabled here.",
            ],
            "recommended_next_step": "First prove a process-wide preview safety hard gate, then add a local paper harness that emits mock heartbeat/worker states without live scheduler, live workers, reconnect, or recovery side effects.",
        },
        "live_safety_hard_gate": {
            "status": "partial",
            "evidence_files": _existing(
                [
                    "scripts/mock_runtime_preview.py",
                    "scripts/controller_mock_preview.py",
                    "tests/scripts/test_mock_runtime_preview.py",
                    "tests/scripts/test_controller_mock_preview.py",
                    "tests/scripts/test_preview_process_safety_hard_gate.py",
                    "tests/test_live_execution_router_preview_safety_guard.py",
                    "config/e2e/demo_paper.yml",
                    "bot_core/exchanges/base.py",
                ]
            ),
            "runtime_backed": True,
            "static_qml_only": False,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Preview helper tests prove live mode/config is blocked, safe payload invariants report exchange/order/runtime-loop disabled, and forbidden exchange/secret/stream/process tokens are absent from preview scripts.",
                "A subprocess/source/payload preview hard gate proof exists for mock/controller preview helpers, and a LiveExecutionRouter DI canary proof exists for the disabled/test-mode path.",
                "The LiveExecutionRouter disabled/test-mode proof shows get_runtime_stats(), execute(...), cancel(...), flush(), and close() do not touch the injected canary adapter; the closed-path cancel guard is covered.",
                "This still is not a full end-to-end preview proof across every backend/runtime/live adapter route under a running preview application.",
                "Paper event spine, preview data-source contract, and read-only market feed proof remain missing, so this is not real live readiness.",
            ],
            "recommended_next_step": "Keep the live safety hard gate partial: preserve the existing subprocess/source/payload helper proof and LiveExecutionRouter DI canary disabled/test-mode proof, then add full end-to-end preview coverage for all backend/runtime/live adapter routes plus a paper event spine, preview data-source contract, and read-only market feed proof.",
        },
    }
    read_model_gap = (
        "Local scenario runner can produce deterministic local context/artifact/audit bundle; "
        "local read model can summarize bundle + boundary matrix for future UI/runtime "
        "integration; read model boundary matrix refuses QML/PySide/UI/runtime/export/cloud/engine "
        "boundaries; UI/runtime preflight audit lists missing requirements before real "
        "integration; preflight/read model/matrix are local/static-only; preflight is not "
        "QML/PySide/UI-bound; preflight is not runtime-backed; bundle/read model/matrix/preflight "
        "generate no orders/decisions; no scoring, no recommendation, no strategy engine, "
        "no AI/model inference, no DecisionEnvelope integration, no TradingController integration, "
        "no file export, no serialization export, no cloud sink, no external export; market context "
        "is in-memory/static-local fixture only; READ_ONLY_MARKET_FETCH is preview-safe; "
        "account/balance/credentials/order/fill/live side effects remain blocked; no real market "
        "adapter/fetch yet; no app runtime loop; no UI integration; no testnet/sandbox adapter."
    )
    for section_name in (
        "ai_decision_governor",
        "alerts_telemetry_audit",
        "preview_mode_contract",
        "paper_terminal_order_lifecycle",
        "data_source_market_feed",
    ):
        sections[section_name]["gaps"].append(read_model_gap)

    payload = {
        "schema_version": "functional_preview_readiness.v1",
        "evaluated_at": "2026-06-16T00:00:00Z",
        "scope": "FUNCTIONAL-PREVIEW-3.15 local paper event spine, portfolio reducer, local audit/alerts consumer, local composition proof, deterministic in-memory local scenario fixture runner, read-only market data contract unit evidence, static/local scenario-level read-only market context evidence, context-only paper scenario decision-context/dry-run artifact contract evidence, local in-memory dry-run artifact audit-trail evidence, and deterministic local context/artifact/audit bundle plus fail-closed bundle boundary/export refusal and local bundle boundary matrix contract evidence, local bundle boundary refusal matrix evidence, and local preview bundle read model evidence, local read model boundary refusal matrix evidence, and local/static UI/runtime preflight audit evidence; no runtime loop, UI integration, file loader/export, secrets, real market fetches, live account access, cloud/export sink, external export, serialization export, engine handoff, DecisionEnvelope handoff, TradingController handoff, order generation, or live order I/O executed",
        "sections": sections,
    }
    validate_report(payload)
    return payload


def validate_report(payload: dict[str, Any]) -> None:
    sections = payload.get("sections")
    if not isinstance(sections, dict) or not sections:
        raise ValueError("sections must be a non-empty object")
    for name, section in sections.items():
        if not isinstance(name, str) or not isinstance(section, dict):
            raise ValueError("section names and values must be objects")
        missing = REQUIRED_KEYS - set(section)
        if missing:
            raise ValueError(f"{name}: missing keys: {sorted(missing)}")
        if section["status"] not in STATUSES:
            raise ValueError(f"{name}: invalid status {section['status']!r}")
        for key in (
            "runtime_backed",
            "static_qml_only",
            "supports_test_server",
            "supports_read_only_real_data",
            "paper_only_execution_safe",
        ):
            if not isinstance(section[key], bool):
                raise ValueError(f"{name}: {key} must be bool")
        if section["status"] == "functional" and section["gaps"]:
            raise ValueError(f"{name}: functional status cannot have known gaps")
        if section["status"] == "static_mock_only" and section["runtime_backed"]:
            raise ValueError(f"{name}: static_mock_only cannot be runtime_backed")
        if section["supports_read_only_real_data"] and not section["supports_test_server"]:
            raise ValueError(f"{name}: read-only real-data support needs a tested source path")
        if not isinstance(section["evidence_files"], list) or not all(
            isinstance(item, str) for item in section["evidence_files"]
        ):
            raise ValueError(f"{name}: evidence_files must be list[str]")
        if not isinstance(section["gaps"], list) or not all(
            isinstance(item, str) for item in section["gaps"]
        ):
            raise ValueError(f"{name}: gaps must be list[str]")
        if (
            not isinstance(section["recommended_next_step"], str)
            or not section["recommended_next_step"].strip()
        ):
            raise ValueError(f"{name}: recommended_next_step must be non-empty")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON report output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_report()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
