"""Strategia opcyjna typu covered-call generująca dochód z premii."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from bot_core.strategies.base import (
    BaseStrategy,
    MarketSnapshot,
    StrategySignal,
    clamp_range,
    ensure_positive_int,
)


@dataclass(slots=True)
class OptionsIncomeSettings:
    """Parametry strategii covered-call."""

    min_iv: float = 0.35
    max_delta: float = 0.35
    min_days_to_expiry: int = 7
    roll_threshold_iv: float = 0.25

    def __post_init__(self) -> None:
        self.min_iv = clamp_range(float(self.min_iv), field="min_iv", lower=0.0, upper=5.0)
        self.max_delta = clamp_range(float(self.max_delta), field="max_delta", lower=0.0, upper=1.0)
        self.min_days_to_expiry = ensure_positive_int(
            int(self.min_days_to_expiry), field="min_days_to_expiry"
        )
        self.roll_threshold_iv = clamp_range(
            float(self.roll_threshold_iv), field="roll_threshold_iv", lower=0.0, upper=5.0
        )

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[str, Any] | None = None
    ) -> "OptionsIncomeSettings":
        params = dict(parameters or {})
        defaults = cls()
        return cls(
            min_iv=float(params.get("min_iv", defaults.min_iv)),
            max_delta=float(params.get("max_delta", defaults.max_delta)),
            min_days_to_expiry=int(params.get("min_days_to_expiry", defaults.min_days_to_expiry)),
            roll_threshold_iv=float(params.get("roll_threshold_iv", defaults.roll_threshold_iv)),
        )


@dataclass(slots=True)
class _OptionsState:
    in_position: bool = False
    entry_iv: float = 0.0


class OptionsIncomeStrategy(BaseStrategy):
    """Generuje sygnały sprzedaży covered-call na podstawie warunków rynkowych."""

    def __init__(self, settings: OptionsIncomeSettings | None = None) -> None:
        super().__init__(
            required_data=("options_chain", "ohlcv"),
            metadata={"capability": "options_income", "tags": ("income", "volatility")},
        )
        self._settings = settings or OptionsIncomeSettings()
        self._states: Dict[str, _OptionsState] = {}

    def warmup(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            self._ensure_state(snapshot.symbol)

    def decide(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        state = self._ensure_state(snapshot.symbol)
        iv = float(snapshot.indicators.get("option_iv", 0.0))
        delta = abs(float(snapshot.indicators.get("option_delta", 0.0)))
        days_to_expiry = int(snapshot.indicators.get("days_to_expiry", 0))

        signals: List[StrategySignal] = []
        if not state.in_position:
            if self._should_open(iv, delta, days_to_expiry):
                state.in_position = True
                state.entry_iv = iv
                option_side = "sell_call"
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side=option_side,
                        confidence=self._entry_confidence(iv, delta),
                        metadata=self._build_metadata(
                            snapshot,
                            iv=iv,
                            delta=delta,
                            days_to_expiry=days_to_expiry,
                            action="enter",
                            option_side=option_side,
                            legacy_side="SELL",
                        ),
                    )
                )
            return signals

        if self._should_close(state, iv, days_to_expiry):
            state.in_position = False
            option_side = "buy_to_close"
            signals.append(
                StrategySignal(
                    symbol=snapshot.symbol,
                    side=option_side,
                    confidence=1.0,
                    metadata=self._build_metadata(
                        snapshot,
                        iv=iv,
                        delta=delta,
                        days_to_expiry=days_to_expiry,
                        action="exit",
                        option_side=option_side,
                        legacy_side="BUY",
                    ),
                )
            )
        return signals

    def teardown(self) -> None:
        self._states.clear()

    # ------------------------------------------------------------------
    def _ensure_state(self, symbol: str) -> _OptionsState:
        if symbol not in self._states:
            self._states[symbol] = _OptionsState()
        return self._states[symbol]

    def _should_open(self, iv: float, delta: float, days: int) -> bool:
        return (
            iv >= self._settings.min_iv
            and delta <= self._settings.max_delta
            and days >= self._settings.min_days_to_expiry
        )

    def _should_close(self, state: _OptionsState, iv: float, days: int) -> bool:
        iv_condition = iv <= self._settings.roll_threshold_iv or iv <= state.entry_iv * 0.6
        expiry_condition = days <= self._settings.min_days_to_expiry
        return iv_condition or expiry_condition

    def _entry_confidence(self, iv: float, delta: float) -> float:
        iv_component = min(1.0, iv / max(self._settings.min_iv, 1e-6))
        delta_component = 1.0 - min(1.0, delta / max(self._settings.max_delta, 1e-6))
        return max(0.0, (iv_component + delta_component) / 2.0)

    def _build_metadata(
        self,
        snapshot: MarketSnapshot,
        *,
        iv: float,
        delta: float,
        days_to_expiry: int,
        action: str,
        option_side: str,
        legacy_side: str | None = None,
    ) -> Dict[str, object]:
        metadata: Dict[str, object] = {
            "strategy": {
                "type": "options_income",
                "profile": "options_income_conservative",
                "risk_label": "income",
            },
            "iv": iv,
            "delta": delta,
            "days_to_expiry": days_to_expiry,
            "action": action,
            "underlying_price": snapshot.close,
            "option_side": option_side,
        }
        if legacy_side:
            metadata["legacy_side"] = legacy_side
        return metadata


__all__ = ["OptionsIncomeSettings", "OptionsIncomeStrategy"]
