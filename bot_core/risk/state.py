"""Modele stanu i metryki wspólne dla runtime i symulacji."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, Mapping, MutableMapping


def normalize_position_side(side: str, *, default: str = "long") -> str:
    """Sprowadza oznaczenie strony pozycji do wartości "long"/"short"."""

    normalized = str(side or "").strip().lower()
    if normalized in {"long", "buy"}:
        return "long"
    if normalized in {"short", "sell"}:
        return "short"
    return default


@dataclass(slots=True)
class PositionState:
    """Stan pojedynczej pozycji wykorzystywany przez silnik ryzyka."""

    side: str
    notional: float

    def __post_init__(self) -> None:
        self.side = normalize_position_side(self.side)
        self.notional = max(0.0, float(self.notional))

    def to_mapping(self) -> Mapping[str, object]:
        return {"side": self.side, "notional": self.notional}

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "PositionState":
        side = str(data.get("side", "long"))
        notional = float(data.get("notional", 0.0))
        return cls(side=side, notional=notional)


@dataclass(slots=True)
class RiskMetrics:
    """Zestaw kluczowych metryk używanych w runtime i backteście."""

    drawdown_pct: float
    daily_loss_pct: float
    weekly_loss_pct: float
    gross_notional: float
    active_positions: int
    used_leverage: float


@dataclass(slots=True)
class RiskState:
    """Stan profilu wykorzystywany do egzekwowania limitów."""

    profile: str
    current_day: date
    start_of_week: date | None = None
    start_of_day_equity: float = 0.0
    last_equity: float = 0.0
    daily_realized_pnl: float = 0.0
    weekly_realized_pnl: float = 0.0
    peak_equity: float = 0.0
    force_liquidation: bool = False
    positions: Dict[str, PositionState] = field(default_factory=dict)
    start_of_week_equity: float = 0.0
    rolling_profit_30d: float = 0.0
    rolling_costs_30d: float = 0.0
    hedge_mode: bool = False
    hedge_reason: str | None = None
    hedge_activated_at: datetime | None = None
    hedge_cooldown_until: datetime | None = None

    def gross_notional(self) -> float:
        return sum(position.notional for position in self.positions.values())

    def active_positions(self) -> int:
        return sum(1 for position in self.positions.values() if position.notional > 0.0)

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "profile": self.profile,
            "current_day": self.current_day.isoformat(),
            "start_of_week": self.start_of_week.isoformat() if self.start_of_week else None,
            "start_of_day_equity": self.start_of_day_equity,
            "daily_realized_pnl": self.daily_realized_pnl,
            "weekly_realized_pnl": self.weekly_realized_pnl,
            "peak_equity": self.peak_equity,
            "force_liquidation": self.force_liquidation,
            "last_equity": self.last_equity,
            "positions": {
                symbol: position.to_mapping() for symbol, position in self.positions.items()
            },
            "start_of_week_equity": self.start_of_week_equity,
            "rolling_profit_30d": self.rolling_profit_30d,
            "rolling_costs_30d": self.rolling_costs_30d,
            "hedge_mode": self.hedge_mode,
            "hedge_reason": self.hedge_reason,
            "hedge_activated_at": self.hedge_activated_at.isoformat()
            if self.hedge_activated_at
            else None,
            "hedge_cooldown_until": self.hedge_cooldown_until.isoformat()
            if self.hedge_cooldown_until
            else None,
        }

    @classmethod
    def from_mapping(cls, profile: str, data: Mapping[str, object]) -> "RiskState":
        day_str = str(data.get("current_day", date.today().isoformat()))
        parsed_day = datetime.fromisoformat(day_str).date()
        start_of_week_str = data.get("start_of_week")
        start_of_week = (
            datetime.fromisoformat(str(start_of_week_str)).date() if start_of_week_str else None
        )
        positions_raw = data.get("positions", {})
        positions: Dict[str, PositionState] = {}
        if isinstance(positions_raw, Mapping):
            for symbol, raw in positions_raw.items():
                if isinstance(raw, Mapping):
                    positions[str(symbol)] = PositionState.from_mapping(raw)
        hedge_activated_at_raw = data.get("hedge_activated_at")
        hedge_activated_at: datetime | None = None
        if hedge_activated_at_raw:
            try:
                hedge_activated_at = datetime.fromisoformat(str(hedge_activated_at_raw))
            except ValueError:
                hedge_activated_at = None

        hedge_cooldown_raw = data.get("hedge_cooldown_until")
        hedge_cooldown_until: datetime | None = None
        if hedge_cooldown_raw:
            try:
                hedge_cooldown_until = datetime.fromisoformat(str(hedge_cooldown_raw))
            except ValueError:
                hedge_cooldown_until = None

        return cls(
            profile=profile,
            current_day=parsed_day,
            start_of_week=start_of_week,
            start_of_day_equity=float(data.get("start_of_day_equity", 0.0)),
            daily_realized_pnl=float(data.get("daily_realized_pnl", 0.0)),
            weekly_realized_pnl=float(data.get("weekly_realized_pnl", 0.0)),
            peak_equity=float(data.get("peak_equity", 0.0)),
            force_liquidation=bool(data.get("force_liquidation", False)),
            last_equity=float(data.get("last_equity", 0.0)),
            positions=positions,
            start_of_week_equity=float(data.get("start_of_week_equity", 0.0)),
            rolling_profit_30d=float(data.get("rolling_profit_30d", 0.0)),
            rolling_costs_30d=float(data.get("rolling_costs_30d", 0.0)),
            hedge_mode=bool(data.get("hedge_mode", False)),
            hedge_reason=str(data.get("hedge_reason"))
            if data.get("hedge_reason") not in (None, "")
            else None,
            hedge_activated_at=hedge_activated_at,
            hedge_cooldown_until=hedge_cooldown_until,
        )

    def reset_for_new_day(self, *, equity: float, day: date) -> None:
        self.current_day = day
        self.start_of_day_equity = equity
        self.daily_realized_pnl = 0.0
        self.force_liquidation = False
        self.last_equity = equity

    def reset_for_new_week(self, *, equity: float, week_start: date) -> None:
        self.start_of_week = week_start
        self.start_of_week_equity = equity
        self.weekly_realized_pnl = 0.0

    def ensure_week_alignment(self, *, day: date, equity: float) -> None:
        week_start = day - timedelta(days=day.weekday())
        if self.start_of_week is None or self.start_of_week != week_start:
            self.reset_for_new_week(equity=equity, week_start=week_start)

    def update_peak_equity(self, equity: float) -> None:
        self.peak_equity = max(self.peak_equity, equity)
        self.last_equity = equity

    def daily_loss_pct(self) -> float:
        if self.start_of_day_equity <= 0:
            return 0.0
        loss = min(0.0, self.daily_realized_pnl)
        return abs(loss) / self.start_of_day_equity

    def drawdown_pct(self, current_equity: float) -> float:
        if self.peak_equity <= 0:
            return 0.0
        drawdown_value = self.peak_equity - current_equity
        if drawdown_value <= 0:
            return 0.0
        return drawdown_value / self.peak_equity

    def weekly_loss_pct(self, *, equity: float | None = None) -> float:
        weekly_equity = self.start_of_week_equity or (equity or 0.0)
        weekly_loss_value = -min(0.0, self.weekly_realized_pnl)
        if weekly_equity <= 0:
            return 0.0
        return weekly_loss_value / weekly_equity

    def update_position(self, symbol: str, side: str, notional: float) -> None:
        if notional <= 0:
            self.positions.pop(symbol, None)
            return
        self.positions[symbol] = PositionState(side=side, notional=notional)

    def activate_hedge(
        self,
        *,
        reason: str | None,
        timestamp: datetime,
        cooldown_until: datetime | None,
    ) -> None:
        self.hedge_mode = True
        self.hedge_reason = reason
        self.hedge_activated_at = timestamp
        self.hedge_cooldown_until = cooldown_until

    def deactivate_hedge(self) -> None:
        self.hedge_mode = False
        self.hedge_reason = None
        self.hedge_cooldown_until = None

    def metrics(
        self,
        *,
        equity: float,
        gross_notional: float | None = None,
        active_positions: int | None = None,
    ) -> RiskMetrics:
        gross = gross_notional if gross_notional is not None else self.gross_notional()
        positions = active_positions if active_positions is not None else self.active_positions()
        used_leverage = gross / equity if equity > 0 else 0.0
        return RiskMetrics(
            drawdown_pct=self.drawdown_pct(equity),
            daily_loss_pct=self.daily_loss_pct(),
            weekly_loss_pct=self.weekly_loss_pct(equity=equity),
            gross_notional=gross,
            active_positions=positions,
            used_leverage=used_leverage,
        )


def build_risk_snapshot(
    state: RiskState,
    *,
    equity: float,
    limits: Mapping[str, float] | None = None,
    gross_notional: float | None = None,
    active_positions: int | None = None,
    cost_breakdown: Mapping[str, float] | None = None,
) -> "RiskSnapshot":
    metrics = state.metrics(
        equity=equity,
        gross_notional=gross_notional,
        active_positions=active_positions,
    )
    return RiskSnapshot(
        profile=state.profile,
        state=state.to_mapping(),
        metrics=metrics,
        limits=limits,
        cost_breakdown=cost_breakdown,
    )


@dataclass(slots=True)
class RiskSnapshot:
    """Zrzut stanu profilu wraz z metrykami do użycia w runtime i backteście."""

    profile: str
    state: Mapping[str, object]
    metrics: RiskMetrics
    limits: Mapping[str, float] | None = None
    cost_breakdown: Mapping[str, float] | None = None

    def to_mapping(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = dict(self.state)
        payload.update(
            {
                "gross_notional": self.metrics.gross_notional,
                "active_positions": self.metrics.active_positions,
                "daily_loss_pct": self.metrics.daily_loss_pct,
                "drawdown_pct": self.metrics.drawdown_pct,
                "weekly_loss_pct": self.metrics.weekly_loss_pct,
                "used_leverage": self.metrics.used_leverage,
            }
        )
        payload["statistics"] = {
            "dailyRealizedPnl": float(self.state.get("daily_realized_pnl", 0.0) or 0.0),
            "grossNotional": self.metrics.gross_notional,
            "activePositions": int(payload.get("active_positions", 0)),
            "dailyLossPct": float(payload.get("daily_loss_pct", 0.0) or 0.0),
            "drawdownPct": float(payload.get("drawdown_pct", 0.0) or 0.0),
            "usedLeverage": self.metrics.used_leverage,
        }
        if self.limits is not None:
            payload["limits"] = dict(self.limits)
        if self.cost_breakdown is not None:
            payload["cost_breakdown"] = dict(self.cost_breakdown)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "RiskSnapshot":
        profile = str(payload.get("profile", ""))
        metrics = RiskMetrics(
            drawdown_pct=float(payload.get("drawdown_pct", 0.0) or 0.0),
            daily_loss_pct=float(payload.get("daily_loss_pct", 0.0) or 0.0),
            weekly_loss_pct=float(payload.get("weekly_loss_pct", 0.0) or 0.0),
            gross_notional=float(payload.get("gross_notional", 0.0) or 0.0),
            active_positions=int(payload.get("active_positions", 0) or 0),
            used_leverage=float(payload.get("used_leverage", 0.0) or 0.0),
        )
        limits_raw = payload.get("limits") if isinstance(payload, Mapping) else None
        limits = dict(limits_raw) if isinstance(limits_raw, Mapping) else None
        cost_raw = payload.get("cost_breakdown") if isinstance(payload, Mapping) else None
        cost_breakdown = dict(cost_raw) if isinstance(cost_raw, Mapping) else None
        return cls(
            profile=profile,
            state=dict(payload),
            metrics=metrics,
            limits=limits,
            cost_breakdown=cost_breakdown,
        )


__all__ = [
    "PositionState",
    "RiskMetrics",
    "RiskSnapshot",
    "RiskState",
    "build_risk_snapshot",
    "normalize_position_side",
]
