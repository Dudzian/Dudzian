from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from bot_core.alerts import AlertMessage
from bot_core.exchanges.base import AccountSnapshot, OrderRequest, OrderResult
from bot_core.execution import ExecutionContext, ExecutionService
from bot_core.risk import RiskCheckResult, RiskEngine, RiskProfile
from bot_core.runtime.controller import TradingController
from bot_core.strategies.base import StrategySignal

_EXPECTED_RULES: tuple[tuple[str, Any], ...] = (
    ("trading.enable_paper_mode", True),
    ("trading.enable_live_mode", False),
    ("execution.default_mode", "paper"),
    ("execution.force_paper_when_offline", True),
    ("execution.live.enabled", False),
)

_DEFAULT_MAX_SIGNALS = 1
_MAX_SIGNALS = 3


class _NoopAlertRouter:
    def dispatch(self, message: AlertMessage) -> None:
        del message

    def health_snapshot(self) -> Mapping[str, Mapping[str, object]]:
        return {"noop": {"status": "ok"}}


class _AllowAllMockRiskEngine(RiskEngine):
    def register_profile(self, profile: RiskProfile) -> None:
        del profile

    def apply_pre_trade_checks(
        self, request: OrderRequest, *, account: AccountSnapshot, profile_name: str
    ) -> RiskCheckResult:
        del request, account, profile_name
        return RiskCheckResult(allowed=True)

    def snapshot_state(self, profile_name: str) -> Mapping[str, object]:
        return {"profile": profile_name, "total_equity": 100_000.0, "available_margin": 90_000.0}

    def on_fill(
        self,
        *,
        profile_name: str,
        symbol: str,
        side: str,
        position_value: float,
        pnl: float,
        timestamp: object | None = None,
    ) -> None:
        del profile_name, symbol, side, position_value, pnl, timestamp

    def should_liquidate(self, *, profile_name: str) -> bool:
        del profile_name
        return False


class _RecordingExecutionService(ExecutionService):
    def __init__(self) -> None:
        self.requests: list[OrderRequest] = []
        self.statuses: list[str] = []

    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        self.requests.append(request)
        result = OrderResult(
            order_id="controller-mock-preview-order",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=request.price,
            raw_response={"context": dict(context.metadata)},
        )
        self.statuses.append(result.status)
        return result

    def cancel(self, order_id: str, context: ExecutionContext) -> None:
        del order_id, context

    def flush(self) -> None:
        return None


def _get_nested_mapping_value(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for segment in dotted_path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controller-backed mock preview (bounded one-shot, no exchange io, no real orders, no api keys)."
        )
    )
    parser.add_argument("--config", default="config/e2e/demo_paper.yml")
    parser.add_argument("--mode", choices=("demo", "paper", "live"), default="demo")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--side", choices=("BUY", "SELL"), default="BUY")
    parser.add_argument("--quantity", type=float, default=0.01)
    parser.add_argument("--max-signals", type=int, default=_DEFAULT_MAX_SIGNALS)
    return parser.parse_args(argv)


def _validate_config(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    loaded: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        return {}, ["config_not_mapping"]

    checks: dict[str, Any] = {}
    issues: list[str] = []
    for dotted_path, expected in _EXPECTED_RULES:
        observed = _get_nested_mapping_value(loaded, dotted_path)
        checks[dotted_path] = observed
        if observed != expected:
            issues.append(f"unsafe_config:{dotted_path}")
    return checks, issues


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(payload)


def _blocked_payload(
    args: argparse.Namespace, reason: str, issues: list[str], checks: dict[str, Any] | None = None
) -> dict[str, Any]:
    events_observed: list[str] = []
    safety_invariants = {
        "exchange_io_disabled": True,
        "real_orders_submitted": False,
        "api_keys_required": False,
        "runtime_loop_started": False,
    }
    payload: dict[str, Any] = {
        "status": "blocked",
        "reason": reason,
        "mode": args.mode,
        "config": str(Path(args.config)),
        "controller_backed_preview_started": False,
        "synthetic_signals_processed": 0,
        "exchange_io": "disabled",
        "order_execution": "mocked_or_disabled",
        "api_keys_required": False,
        "live_mode_allowed": False,
        "real_orders_submitted": False,
        "runtime_loop_started": False,
        "events_observed": events_observed,
        "events_observed_count": len(events_observed),
        "controller_results_count": 0,
        "controller_result_statuses": [],
        "mock_execution_requests_count": 0,
        "mock_execution_statuses": [],
        "journal_summary": "N/A (controller mock wrapper has no journal export in this stage)",
        "journal_events_count": None,
        "journal_event_types": [],
        "safety_invariants": safety_invariants,
        "issues": issues,
        "safety_contract_version": "controller_mock_preview.v1",
    }
    if checks is not None:
        payload["checks"] = checks
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.mode == "live":
        _emit(
            _blocked_payload(
                args,
                reason="controller_mock_preview_forbids_live_mode",
                issues=["live_mode_not_allowed"],
            ),
            args.json,
        )
        return 2

    if args.max_signals <= 0 or args.max_signals > _MAX_SIGNALS:
        _emit(
            _blocked_payload(
                args,
                reason="max_signals_out_of_bounds",
                issues=["max_signals_out_of_bounds"],
            ),
            args.json,
        )
        return 2

    if args.quantity <= 0:
        _emit(
            _blocked_payload(
                args,
                reason="controller_mock_preview_invalid_quantity",
                issues=["invalid_quantity"],
            ),
            args.json,
        )
        return 2

    args.symbol = args.symbol.strip()
    if not args.symbol:
        _emit(
            _blocked_payload(
                args,
                reason="controller_mock_preview_invalid_symbol",
                issues=["invalid_symbol"],
            ),
            args.json,
        )
        return 2

    config_path = Path(args.config)
    if not config_path.exists():
        payload = _blocked_payload(
            args,
            reason="config_not_found",
            issues=[f"config_not_found:{config_path}"],
        )
        payload["status"] = "error"
        _emit(payload, args.json)
        return 1

    checks, issues = _validate_config(config_path)
    if issues:
        _emit(
            _blocked_payload(args, reason="unsafe_config", issues=issues, checks=checks), args.json
        )
        return 2

    risk_engine = _AllowAllMockRiskEngine()
    execution = _RecordingExecutionService()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=_NoopAlertRouter(),
        account_snapshot_provider=lambda: AccountSnapshot(
            balances={},
            total_equity=100_000.0,
            available_margin=90_000.0,
            maintenance_margin=10_000.0,
        ),
        portfolio_id="controller-mock-preview",
        environment="paper",
        risk_profile="balanced",
    )

    signal = StrategySignal(
        symbol=args.symbol,
        side=args.side,
        confidence=1.0,
        metadata={"source": "controller_mock_preview", "synthetic": True},
        quantity=args.quantity,
    )
    signals = [signal for _ in range(args.max_signals)]
    results = controller.process_signals(signals)

    controller_result_statuses = [result.status for result in results]
    payload = {
        "status": "ok",
        "mode": args.mode,
        "config": str(config_path),
        "controller_backed_preview_started": True,
        "synthetic_signals_processed": len(signals),
        "exchange_io": "disabled",
        "order_execution": "mocked_or_disabled",
        "api_keys_required": False,
        "live_mode_allowed": False,
        "real_orders_submitted": False,
        "runtime_loop_started": False,
        "events_observed": controller_result_statuses,
        "events_observed_count": len(controller_result_statuses),
        "controller_results_count": len(results),
        "controller_result_statuses": controller_result_statuses,
        "mock_execution_requests_count": len(execution.requests),
        "mock_execution_statuses": list(execution.statuses),
        "journal_summary": "N/A (controller mock wrapper has no journal export in this stage)",
        "journal_events_count": None,
        "journal_event_types": [],
        "safety_invariants": {
            "exchange_io_disabled": True,
            "real_orders_submitted": False,
            "api_keys_required": False,
            "runtime_loop_started": False,
        },
        "issues": [],
        "checks": checks,
        "recorded_mock_requests": len(execution.requests),
        "safety_contract_version": "controller_mock_preview.v1",
    }
    _emit(payload, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
