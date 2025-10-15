"""Modele danych DecisionOrchestratora (Etap 5)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Sequence


@dataclass(slots=True)
class DecisionCandidate:
    """Kandydat decyzji handlowej oceniany przez orchestratora."""

    strategy: str
    action: str
    risk_profile: str
    symbol: str | None
    notional: float
    expected_return_bps: float
    expected_probability: float = 1.0
    cost_bps_override: float | None = None
    latency_ms: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.strategy = str(self.strategy)
        self.action = str(self.action)
        self.risk_profile = str(self.risk_profile)
        self.symbol = str(self.symbol) if self.symbol not in (None, "") else None
        self.notional = float(self.notional)
        self.expected_return_bps = float(self.expected_return_bps)
        self.expected_probability = max(0.0, min(1.0, float(self.expected_probability)))
        if self.cost_bps_override is not None:
            self.cost_bps_override = float(self.cost_bps_override)
        if self.latency_ms is not None:
            self.latency_ms = float(self.latency_ms)
        self.metadata = dict(self.metadata)

    @property
    def expected_value_bps(self) -> float:
        """Zwrot oczekiwany po uwzględnieniu prawdopodobieństwa sukcesu."""

        return self.expected_return_bps * self.expected_probability

    def to_mapping(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "strategy": self.strategy,
            "action": self.action,
            "risk_profile": self.risk_profile,
            "notional": self.notional,
            "expected_return_bps": self.expected_return_bps,
            "expected_probability": self.expected_probability,
            "metadata": dict(self.metadata),
        }
        if self.symbol is not None:
            payload["symbol"] = self.symbol
        if self.cost_bps_override is not None:
            payload["cost_bps_override"] = self.cost_bps_override
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        return payload

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "DecisionCandidate":
        return cls(
            strategy=str(raw["strategy"]),
            action=str(raw.get("action", "enter")),
            risk_profile=str(raw.get("risk_profile", raw.get("profile", ""))),
            symbol=raw.get("symbol"),
            notional=float(raw.get("notional", 0.0)),
            expected_return_bps=float(raw.get("expected_return_bps", 0.0)),
            expected_probability=float(raw.get("expected_probability", 1.0)),
            cost_bps_override=(
                None
                if raw.get("cost_bps_override") is None
                else float(raw.get("cost_bps_override"))
            ),
            latency_ms=(
                None if raw.get("latency_ms") is None else float(raw.get("latency_ms"))
            ),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(slots=True)
class RiskSnapshot:
    """Minimalny snapshot stanu profilu ryzyka."""

    profile: str
    start_of_day_equity: float
    daily_realized_pnl: float
    peak_equity: float
    last_equity: float
    gross_notional: float
    active_positions: int
    symbols: Sequence[str] = field(default_factory=tuple)
    force_liquidation: bool = False

    @classmethod
    def from_mapping(cls, profile: str, raw: Mapping[str, object]) -> "RiskSnapshot":
        positions_raw = raw.get("positions", {})
        symbols: list[str] = []
        gross = 0.0
        if isinstance(positions_raw, Mapping):
            for name, payload in positions_raw.items():
                if not isinstance(payload, Mapping):
                    continue
                notional = float(payload.get("notional", 0.0))
                if notional <= 0:
                    continue
                symbols.append(str(name))
                gross += max(0.0, notional)
        start_equity = float(raw.get("start_of_day_equity", 0.0))
        last_equity = float(raw.get("last_equity", start_equity))
        peak_equity = float(raw.get("peak_equity", last_equity))
        daily_realized = float(raw.get("daily_realized_pnl", 0.0))
        return cls(
            profile=profile,
            start_of_day_equity=start_equity,
            daily_realized_pnl=daily_realized,
            peak_equity=peak_equity,
            last_equity=last_equity,
            gross_notional=gross,
            active_positions=len(symbols),
            symbols=tuple(symbols),
            force_liquidation=bool(raw.get("force_liquidation", False)),
        )

    @property
    def daily_loss_pct(self) -> float:
        if self.start_of_day_equity <= 0:
            return 0.0
        loss = min(0.0, self.daily_realized_pnl)
        return abs(loss) / self.start_of_day_equity

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        drawdown_value = self.peak_equity - self.last_equity
        if drawdown_value <= 0:
            return 0.0
        return drawdown_value / self.peak_equity

    def contains_symbol(self, symbol: str | None) -> bool:
        if symbol is None:
            return False
        return symbol in set(self.symbols)


@dataclass(slots=True)
class DecisionEvaluation:
    """Wynik oceny kandydata przez orchestratora."""

    candidate: DecisionCandidate
    accepted: bool
    cost_bps: float | None
    net_edge_bps: float | None
    reasons: Sequence[str] = field(default_factory=tuple)
    risk_flags: Sequence[str] = field(default_factory=tuple)
    stress_failures: Sequence[str] = field(default_factory=tuple)

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "candidate": self.candidate.to_mapping(),
            "accepted": self.accepted,
            "cost_bps": self.cost_bps,
            "net_edge_bps": self.net_edge_bps,
            "reasons": list(self.reasons),
            "risk_flags": list(self.risk_flags),
            "stress_failures": list(self.stress_failures),
        }


__all__ = [
    "DecisionCandidate",
    "DecisionEvaluation",
    "RiskSnapshot",
]
