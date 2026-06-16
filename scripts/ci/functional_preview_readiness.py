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
                ]
            ),
            "runtime_backed": True,
            "static_qml_only": False,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Preview profile is explicitly in-process/local and points at a sample OHLCV dataset, not a test-server or read-only real-market source.",
                "No preview-scoped abstraction was found that selects local_mock/test_server/sandbox/read_only_market as interchangeable data sources.",
                "gRPC/UI fallback code can expose runtime-shaped snapshots, but this audit found no deterministic preview contract proving real read-only feed ingestion.",
            ],
            "recommended_next_step": "Add a preview data-source contract with explicit local_mock/test_server/sandbox/read_only_market modes and tests that prove read-only feed use without order I/O.",
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
                ]
            ),
            "runtime_backed": True,
            "static_qml_only": False,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Preview decision rows can be loaded from runtime journal/stream plumbing, but default first-run data is a static demo snapshot.",
                "Paper decision-engine smoke inputs are file fixtures, not a preview proof of live read-only market input.",
                "Confidence/reason/risk fields in the preview default are supplied as serialized demo values unless a runtime source is configured.",
            ],
            "recommended_next_step": "Wire a paper/preview decision source to deterministic read-only market fixtures and assert computed confidence/risk provenance.",
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
                ]
            ),
            "runtime_backed": True,
            "static_qml_only": False,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Mock/controller preview proves a bounded filled synthetic order path with exchange I/O disabled, but not a full accepted/rejected/partial/fill/cancel UI lifecycle.",
                "No evidence was found that preview portfolio and alerts are updated by the same paper order event stream end-to-end.",
            ],
            "recommended_next_step": "Add a paper broker simulator contract covering accepted, rejected, partial fill, fill, and cancel events through UI-facing models.",
        },
        "portfolio_positions_trades": {
            "status": "static_mock_only",
            "evidence_files": _existing(
                [
                    "ui/qml/views/PortfolioDashboard.qml",
                    "ui/qml/components/PortfolioManagerView.qml",
                    "ui/src/grpc/TradingClient.cpp",
                    "ui/src/app/PortfolioManagerController.cpp",
                ]
            ),
            "runtime_backed": False,
            "static_qml_only": True,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Preview portfolio presentation has UI/model shape, but this audit found no proof of mutation after simulated fill.",
                "Read-only balance retrieval for preview/test server is not implemented as an allowed, tested preview path.",
            ],
            "recommended_next_step": "Create a preview portfolio state reducer fed by paper fill events and a separate read-only balance adapter with secret-free tests.",
        },
        "alerts_telemetry_audit": {
            "status": "partial",
            "evidence_files": _existing(
                [
                    "ui/backend/alert_manager.py",
                    "ui/backend/telemetry_provider.py",
                    "ui/backend/privacy_settings.py",
                    "ui/src/telemetry/UiTelemetryReporter.cpp",
                    "core/telemetry/anonymous_collector.py",
                    "tests/scripts/test_mock_runtime_preview.py",
                ]
            ),
            "runtime_backed": True,
            "static_qml_only": False,
            "supports_test_server": False,
            "supports_read_only_real_data": False,
            "paper_only_execution_safe": True,
            "gaps": [
                "Telemetry/alerts have runtime plumbing and local preview contracts, but some rows remain demo/static unless connected to runtime events.",
                "This audit did not find a single hard CI assertion covering all preview cloud/export sinks; existing safety tests focus on mock preview source tokens and config.",
            ],
            "recommended_next_step": "Add a preview telemetry sink contract proving local-only storage/export disabled while consuming runtime paper events.",
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
                "Preview helper tests prove live mode/config is blocked and forbidden exchange/secret tokens are absent from preview scripts.",
                "The hard gate is proven for mock/controller preview helpers, not for every backend/runtime/live adapter route under a running preview application.",
                "Runtime assertions or adapter-level mocks proving no live exchange I/O across the full preview process were not identified in this audit.",
            ],
            "recommended_next_step": "First add process-wide preview safety hard gate proof around exchange adapters/create_order; then add a local paper event spine for order lifecycle → portfolio → alerts; then add the preview data-source contract.",
        },
    }
    payload = {
        "schema_version": "functional_preview_readiness.v1",
        "evaluated_at": "2026-06-16T00:00:00Z",
        "scope": "FUNCTIONAL-PREVIEW-2.0 static/read-only audit refresh after FRONTEND-PARITY-9.0/10.0/11.0; no runtime loop, secrets, market fetches, or order I/O executed",
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
        if not isinstance(section["recommended_next_step"], str) or not section[
            "recommended_next_step"
        ].strip():
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
