"""Run CI backtest regressions and export audit artefacts."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import pandas as pd

from bot_core.backtest.engine import BacktestEngine, MatchingConfig
from bot_core.backtest.reporting import export_report
from bot_core.risk.base import RiskCheckResult
from bot_core.risk.guardrails import evaluate_backtest_guardrails, summarize_guardrail_results
from bot_core.risk.profiles import AggressiveProfile, BalancedProfile, ConservativeProfile
from scripts.backtest_ci_config import BACKTEST_CI_SCENARIOS


@dataclass(slots=True)
class RegressionSignal:
    action: str
    size: float | None
    stop_loss: float | None
    take_profit: float | None


@dataclass(slots=True)
class RegressionContext:
    symbol: str
    timeframe: str
    portfolio_value: float
    position: float
    timestamp: datetime
    metadata: Mapping[str, Any] | None
    extra: MutableMapping[str, Any]


class MovingAverageRegressionStrategy:
    """Simple moving average crossover used for regression testing."""

    def __init__(
        self,
        *,
        fast_window: int = 8,
        slow_window: int = 21,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float = 0.03,
    ) -> None:
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError("Moving average windows must be positive")
        if fast_window >= slow_window:
            raise ValueError("Fast window must be shorter than slow window")
        self._fast = int(fast_window)
        self._slow = int(slow_window)
        self._stop_loss_pct = float(stop_loss_pct)
        self._take_profit_pct = float(take_profit_pct)
        self._bias = 0.001

    async def prepare(self, context: RegressionContext, data_provider: Any) -> None:
        return None

    async def handle_market_data(
        self, context: RegressionContext, market_payload: Mapping[str, Any]
    ) -> RegressionSignal:
        ohlcv = market_payload.get("ohlcv")
        if not isinstance(ohlcv, pd.DataFrame) or len(ohlcv) < self._slow:
            return RegressionSignal("HOLD", None, self._stop_loss_pct, self._take_profit_pct)
        closes = ohlcv.get("close")
        if closes is None:
            return RegressionSignal("HOLD", None, self._stop_loss_pct, self._take_profit_pct)
        close_series = pd.Series(closes, dtype="float64")
        fast_ma = close_series.rolling(self._fast, min_periods=self._fast).mean().iloc[-1]
        slow_ma = close_series.rolling(self._slow, min_periods=self._slow).mean().iloc[-1]
        if pd.isna(fast_ma) or pd.isna(slow_ma):
            return RegressionSignal("HOLD", None, self._stop_loss_pct, self._take_profit_pct)
        if fast_ma > slow_ma * (1.0 + self._bias) and context.position <= 0:
            return RegressionSignal("BUY", None, self._stop_loss_pct, self._take_profit_pct)
        if fast_ma < slow_ma * (1.0 - self._bias) and context.position >= 0:
            return RegressionSignal("SELL", None, self._stop_loss_pct, self._take_profit_pct)
        return RegressionSignal("HOLD", None, self._stop_loss_pct, self._take_profit_pct)

    async def notify_fill(self, context: RegressionContext, fill: Mapping[str, Any]) -> None:
        return None

    async def shutdown(self) -> None:
        return None


_PROFILE_REGISTRY = {
    "aggressive": AggressiveProfile(),
    "balanced": BalancedProfile(),
    "conservative": ConservativeProfile(),
}


def _resolve_risk_profile(name: str | None):
    if not name:
        return None
    return _PROFILE_REGISTRY.get(str(name).lower())


def _build_context(payload: Mapping[str, Any]) -> RegressionContext:
    timestamp = payload["timestamp"]
    if isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.to_pydatetime()
    if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    metadata_raw = payload.get("metadata") if isinstance(payload, Mapping) else None
    metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else metadata_raw
    extra_raw = payload.get("extra", {})
    if isinstance(extra_raw, MutableMapping):
        extra = extra_raw  # type: ignore[assignment]
    elif isinstance(extra_raw, Mapping):
        extra = dict(extra_raw)
    else:
        extra = {}
    return RegressionContext(
        symbol=str(payload["symbol"]),
        timeframe=str(payload["timeframe"]),
        portfolio_value=float(payload["portfolio_value"]),
        position=float(payload["position"]),
        timestamp=timestamp,
        metadata=metadata,
        extra=extra,
    )


def _load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    return df


def _select_scenarios(target: str | None) -> Sequence[Dict[str, Any]]:
    if not target:
        return BACKTEST_CI_SCENARIOS
    selected = [cfg for cfg in BACKTEST_CI_SCENARIOS if cfg.get("name") == target]
    if not selected:
        raise KeyError(f"Scenario '{target}' is not defined in BACKTEST_CI_SCENARIOS")
    return selected


def run_backtest_regressions(
    output_dir: Path,
    *,
    scenario: str | None = None,
    fail_on_block: bool = False,
) -> Sequence[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[Path] = []
    guardrail_results: dict[str, RiskCheckResult] = {}
    blocked: list[str] = []
    for cfg in _select_scenarios(scenario):
        dataset_path = Path(cfg["dataset"])
        data = _load_dataset(dataset_path)
        metadata = {
            "risk_profile": cfg.get("risk_profile"),
            "required_data": tuple(cfg.get("required_data", ())),
        }
        matching = MatchingConfig(**cfg.get("matching", {}))
        context_extra = cfg.get("context_extra", {})
        strategy_cfg = cfg.get("strategy", {})

        def _factory() -> MovingAverageRegressionStrategy:
            return MovingAverageRegressionStrategy(
                fast_window=int(strategy_cfg.get("fast_window", 8)),
                slow_window=int(strategy_cfg.get("slow_window", 21)),
                stop_loss_pct=float(strategy_cfg.get("stop_loss_pct", 0.01)),
                take_profit_pct=float(strategy_cfg.get("take_profit_pct", 0.03)),
            )

        engine = BacktestEngine(
            strategy_factory=_factory,
            context_builder=_build_context,
            data=data,
            symbol=str(cfg["symbol"]),
            timeframe=str(cfg["timeframe"]),
            initial_balance=float(cfg.get("initial_balance", 10_000.0)),
            matching=matching,
            allow_short=bool(cfg.get("allow_short", False)),
            context_extra=context_extra,
            metadata=metadata,
        )
        report = engine.run()
        scenario_dir = output_dir / str(cfg["name"])
        artefacts = export_report(
            report,
            scenario_dir,
            title=f"Backtest report · {cfg['name']}",
            html_name="report.html",
            pdf_name="report.pdf",
        )
        guardrail_result = evaluate_backtest_guardrails(
            report,
            risk_profile=_resolve_risk_profile(cfg.get("risk_profile")),
            max_drawdown_pct=cfg.get("max_drawdown_pct"),
            max_exposure_pct=cfg.get("max_exposure_pct"),
            min_sortino_ratio=cfg.get("min_sortino_ratio"),
            min_omega_ratio=cfg.get("min_omega_ratio"),
        )
        guardrail_results[str(cfg["name"])] = guardrail_result

        metadata = guardrail_result.metadata or {}
        summary = {
            "scenario": cfg["name"],
            "symbol": cfg["symbol"],
            "timeframe": cfg["timeframe"],
            "strategy_metadata": report.strategy_metadata,
            "parameters": report.parameters,
            "metrics": asdict(report.metrics) if report.metrics else None,
            "warnings": list(report.warnings),
            "trades": len(report.trades),
            "artefacts": {key: str(path) for key, path in artefacts.items()},
            "guardrails": {
                "allowed": guardrail_result.allowed,
                "reason": guardrail_result.reason,
                "thresholds": dict(metadata.get("thresholds", {})),
                "threshold_sources": dict(metadata.get("threshold_sources", {})),
                "observed": dict(metadata.get("observed", {})),
                "violations": list(metadata.get("violations", ())),
                "risk_profile": metadata.get("risk_profile"),
                "strategy_metadata": dict(metadata.get("strategy_metadata", {})),
                "warnings": list(metadata.get("warnings", ())),
            },
        }
        summary_path = scenario_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(summary_path)
        if not guardrail_result.allowed:
            blocked.append(f"{cfg['name']}: {guardrail_result.reason or 'blocked by guardrails'}")

    aggregate = summarize_guardrail_results(guardrail_results)
    (output_dir / "guardrail_summary.json").write_text(
        json.dumps(aggregate.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if fail_on_block and blocked:
        raise RuntimeError("; ".join(blocked))
    return summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit/backtests/ci"),
        help="Directory where reports should be written",
    )
    parser.add_argument(
        "--scenario",
        help="Run only the specified scenario from BACKTEST_CI_SCENARIOS",
    )
    parser.add_argument(
        "--fail-on-block",
        action="store_true",
        help="Exit with non-zero status when guardrails reject a scenario",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_backtest_regressions(args.output, scenario=args.scenario, fail_on_block=args.fail_on_block)
    except RuntimeError as exc:
        print(f"Guardrail failure: {exc}")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
