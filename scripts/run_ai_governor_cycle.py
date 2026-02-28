"""Uruchamia pojedynczy cykl/serię cykli AutoTraderAIGovernorRunner.

Skrypt buduje lekką konfigurację orchestratorem opartym o snapshot JSON lub
domyślny zestaw strategii demo i wypisuje decyzje w formacie JSON.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from bot_core.ai.regime import MarketRegime
from bot_core.auto_trader.ai_governor import AutoTraderAIGovernorRunner
from bot_core.decision.orchestrator import StrategyPerformanceSummary


def _load_snapshot(path: Path | None) -> Mapping[str, StrategyPerformanceSummary]:
    if path is None:
        now = datetime.now(timezone.utc)
        return {
            "scalping_alpha": StrategyPerformanceSummary(
                strategy="scalping_alpha",
                regime=MarketRegime.TREND,
                hit_rate=0.78,
                pnl=12.0,
                sharpe=1.1,
                updated_at=now,
                observations=8,
            ),
            "grid_balanced": StrategyPerformanceSummary(
                strategy="grid_balanced",
                regime=MarketRegime.DAILY,
                hit_rate=0.64,
                pnl=4.2,
                sharpe=0.4,
                updated_at=now,
                observations=5,
            ),
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    snapshot = {}
    for entry in payload:
        snapshot[entry["strategy"]] = StrategyPerformanceSummary(
            strategy=entry["strategy"],
            regime=MarketRegime(entry.get("regime", MarketRegime.TREND.value)),
            hit_rate=float(entry.get("hit_rate", 0.5)),
            pnl=float(entry.get("pnl", 0.0)),
            sharpe=float(entry.get("sharpe", 0.0)),
            updated_at=now,
            observations=int(entry.get("observations", 1)),
        )
    return snapshot


def _parse_modes(values: Sequence[str] | None) -> Iterable[MarketRegime]:
    if not values:
        return ()
    return tuple(MarketRegime(value) for value in values)


def _serialize(decisions: Sequence[Any]) -> str:
    def _to_mapping(decision: Any) -> Any:
        if decision is None:
            return None
        if hasattr(decision, "model_dump"):
            try:
                return decision.model_dump()
            except Exception:
                pass
        if hasattr(decision, "to_mapping"):
            try:
                return decision.to_mapping()  # type: ignore[no-any-return]
            except Exception:
                pass
        if is_dataclass(decision) and not isinstance(decision, type):
            return asdict(decision)
        if isinstance(decision, Mapping):
            return dict(decision)
        plain = getattr(decision, "__dict__", None)
        if isinstance(plain, Mapping):
            return dict(plain)
        return str(decision)

    payload = [_to_mapping(decision) for decision in decisions]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Plik JSON ze snapshotem strategii")
    parser.add_argument("--mode", type=str, help="Docelowy tryb (scalping/grid/hedge)")
    parser.add_argument("--limit", type=int, default=1, help="Limit cykli przed przerwaniem")
    parser.add_argument(
        "--regime-cycle",
        type=str,
        nargs="*",
        help="Lista reżimów do iteracji (np. trend daily mean_reversion)",
    )
    args = parser.parse_args(argv)

    snapshot = _load_snapshot(args.snapshot)
    orchestrator = type("_Orchestrator", (), {"strategy_performance_snapshot": lambda self: snapshot})()
    runner = AutoTraderAIGovernorRunner(orchestrator)

    decisions = runner.run_until(
        mode=args.mode,
        limit=args.limit,
        regimes=tuple(_parse_modes(args.regime_cycle)),
    )
    print(_serialize(decisions))
    return 0


if __name__ == "__main__":  # pragma: no cover - wywołanie z CLI
    raise SystemExit(main())
