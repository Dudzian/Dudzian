"""Adapter warstwy opcjonalnych zależności PortfolioGovernora."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping

try:  # pragma: no cover - moduły mogą być nieobecne w lżejszych buildach
    from bot_core.market_intel import MarketIntelSnapshot  # type: ignore
except Exception:  # pragma: no cover - fallback minimalny

    @dataclass(slots=True)
    class MarketIntelSnapshot:  # type: ignore[redefinition]
        symbol: str
        interval: str
        start: datetime | None
        end: datetime | None
        bar_count: int
        price_change_pct: float | None = None
        volatility_pct: float | None = None
        max_drawdown_pct: float | None = None
        average_volume: float | None = None
        liquidity_usd: float | None = None
        momentum_score: float | None = None
        metadata: Mapping[str, float] = field(default_factory=dict)


try:  # pragma: no cover - opcjonalne metryki SLO
    from bot_core.observability.slo import SLOStatus  # type: ignore
except Exception:  # pragma: no cover - fallback minimalny

    @dataclass(slots=True)
    class SLOStatus:  # type: ignore[redefinition]
        status: str | None = None
        severity: str = "warning"
        error_budget_pct: float | None = None

        @property
        def is_breach(self) -> bool:
            return (self.status or "").lower() == "breach"


try:  # pragma: no cover - zależność od warstwy ryzyka
    from bot_core.risk import StressOverrideRecommendation  # type: ignore
except Exception:  # pragma: no cover - fallback minimalny

    @dataclass(slots=True)
    class StressOverrideRecommendation:  # type: ignore[redefinition]
        symbol: str | None = None
        risk_budget: str | None = None
        reason: str = ""
        severity: str | None = "warning"
        weight_multiplier: float | None = None
        min_weight: float | None = None
        max_weight: float | None = None
        force_rebalance: bool = False


__all__ = [
    "MarketIntelSnapshot",
    "SLOStatus",
    "StressOverrideRecommendation",
]
