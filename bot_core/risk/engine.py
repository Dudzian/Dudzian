"""Implementacja silnika ryzyka egzekwującego limity profili."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Dict, Mapping, MutableMapping

from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.base import RiskCheckResult, RiskEngine, RiskProfile, RiskRepository

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PositionState:
    """Stan pojedynczej pozycji wykorzystywany przez silnik ryzyka."""

    side: str
    notional: float

    def to_mapping(self) -> Mapping[str, object]:
        return {"side": self.side, "notional": self.notional}

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "PositionState":
        side = str(data.get("side", "long"))
        notional = float(data.get("notional", 0.0))
        return cls(side=side, notional=max(0.0, notional))


@dataclass(slots=True)
class RiskState:
    """Stan profilu wykorzystywany do egzekwowania limitów."""

    profile: str
    current_day: date
    start_of_day_equity: float = 0.0
    last_equity: float = 0.0
    daily_realized_pnl: float = 0.0
    peak_equity: float = 0.0
    force_liquidation: bool = False
    positions: Dict[str, PositionState] = field(default_factory=dict)

    def gross_notional(self) -> float:
        return sum(position.notional for position in self.positions.values())

    def active_positions(self) -> int:
        return sum(1 for position in self.positions.values() if position.notional > 0.0)

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "profile": self.profile,
            "current_day": self.current_day.isoformat(),
            "start_of_day_equity": self.start_of_day_equity,
            "daily_realized_pnl": self.daily_realized_pnl,
            "peak_equity": self.peak_equity,
            "force_liquidation": self.force_liquidation,
            "last_equity": self.last_equity,
            "positions": {symbol: position.to_mapping() for symbol, position in self.positions.items()},
        }

    @classmethod
    def from_mapping(cls, profile: str, data: Mapping[str, object]) -> "RiskState":
        day_str = str(data.get("current_day", date.today().isoformat()))
        parsed_day = datetime.fromisoformat(day_str).date()
        positions_raw = data.get("positions", {})
        positions: Dict[str, PositionState] = {}
        if isinstance(positions_raw, Mapping):
            for symbol, raw in positions_raw.items():
                if isinstance(raw, Mapping):
                    positions[str(symbol)] = PositionState.from_mapping(raw)
        return cls(
            profile=profile,
            current_day=parsed_day,
            start_of_day_equity=float(data.get("start_of_day_equity", 0.0)),
            daily_realized_pnl=float(data.get("daily_realized_pnl", 0.0)),
            peak_equity=float(data.get("peak_equity", 0.0)),
            force_liquidation=bool(data.get("force_liquidation", False)),
            last_equity=float(data.get("last_equity", 0.0)),
            positions=positions,
        )

    def reset_for_new_day(self, *, equity: float, day: date) -> None:
        self.current_day = day
        self.start_of_day_equity = equity
        self.daily_realized_pnl = 0.0
        self.force_liquidation = False
        self.last_equity = equity

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

    def update_position(self, symbol: str, side: str, notional: float) -> None:
        if notional <= 0:
            self.positions.pop(symbol, None)
            return
        self.positions[symbol] = PositionState(side=side, notional=notional)


class InMemoryRiskRepository(RiskRepository):
    """Najprostsze repozytorium używane do testów i trybu offline."""

    def __init__(self) -> None:
        self._storage: Dict[str, MutableMapping[str, object]] = {}

    def load(self, profile: str) -> Mapping[str, object] | None:
        return self._storage.get(profile)

    def store(self, profile: str, state: Mapping[str, object]) -> None:
        self._storage[profile] = dict(state)


class ThresholdRiskEngine(RiskEngine):
    """Silnik egzekwujący limity dzienne, ekspozycji i dźwigni."""

    def __init__(
        self,
        repository: RiskRepository | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository or InMemoryRiskRepository()
        self._clock = clock or datetime.utcnow
        self._profiles: Dict[str, RiskProfile] = {}
        self._states: Dict[str, RiskState] = {}

    def register_profile(self, profile: RiskProfile) -> None:
        self._profiles[profile.name] = profile
        state = self._repository.load(profile.name)
        if state is None:
            today = self._clock().date()
            self._states[profile.name] = RiskState(profile=profile.name, current_day=today)
            self._repository.store(profile.name, self._states[profile.name].to_mapping())
        else:
            self._states[profile.name] = RiskState.from_mapping(profile.name, state)
        _LOGGER.info("Zarejestrowano profil ryzyka %s", profile.name)

    def apply_pre_trade_checks(
        self,
        request: OrderRequest,
        *,
        account: AccountSnapshot,
        profile_name: str,
    ) -> RiskCheckResult:
        profile = self._profiles.get(profile_name)
        if profile is None:
            raise KeyError(f"Profil ryzyka {profile_name} nie został zarejestrowany")

        state = self._states.get(profile_name)
        if state is None:
            raise KeyError(f"Brak stanu ryzyka dla profilu {profile_name}")

        now = self._clock()
        current_day = now.date()
        if state.start_of_day_equity <= 0:
            state.start_of_day_equity = account.total_equity
            state.last_equity = account.total_equity

        if state.peak_equity <= 0:
            state.peak_equity = account.total_equity

        if state.current_day != current_day:
            state.reset_for_new_day(equity=account.total_equity, day=current_day)
            _LOGGER.info("Reset dziennego licznika PnL dla profilu %s", profile_name)

        state.update_peak_equity(account.total_equity)

        drawdown = state.drawdown_pct(account.total_equity)
        daily_loss = state.daily_loss_pct()

        force_reason: str | None = None
        if profile.drawdown_limit() > 0 and drawdown >= profile.drawdown_limit():
            state.force_liquidation = True
            force_reason = "Przekroczono limit obsunięcia portfela."

        if profile.daily_loss_limit() > 0 and daily_loss >= profile.daily_loss_limit():
            state.force_liquidation = True
            force_reason = "Przekroczono dzienny limit straty."

        price = request.price
        if price is None or price <= 0:
            self._persist_state(profile_name)
            return RiskCheckResult(
                allowed=False,
                reason=(
                    "Brak ceny referencyjnej uniemożliwia obliczenie ekspozycji. Podaj midpoint z orderbooka lub cenę limit."
                ),
            )

        symbol = request.symbol
        position = state.positions.get(symbol)
        is_buy = request.side.lower() == "buy"
        notional = abs(request.quantity) * price
        current_notional = position.notional if position else 0.0
        current_side = position.side if position else ("long" if is_buy else "short")

        if position:
            if position.side == "long" and not is_buy and notional > current_notional:
                self._persist_state(profile_name)
                return RiskCheckResult(
                    allowed=False,
                    reason=(
                        "Zlecenie sprzedaży przekracza wielkość istniejącej pozycji. Zamknij pozycję przed odwróceniem kierunku."
                    ),
                )
            if position.side == "short" and is_buy and notional > current_notional:
                self._persist_state(profile_name)
                return RiskCheckResult(
                    allowed=False,
                    reason=(
                        "Zlecenie kupna przekracza wielkość krótkiej pozycji. Zamknij pozycję przed odwróceniem kierunku."
                    ),
                )

        if is_buy:
            if current_side == "long":
                new_notional = current_notional + notional
            else:
                new_notional = max(current_notional - notional, 0.0)
        else:
            if current_side == "short":
                new_notional = current_notional + notional
            else:
                new_notional = max(current_notional - notional, 0.0)

        is_reducing = new_notional < current_notional

        is_new_position = current_notional == 0.0 and new_notional > 0.0 and not is_reducing

        if state.force_liquidation and not is_reducing:
            self._persist_state(profile_name)
            return RiskCheckResult(
                allowed=False,
                reason=(
                    (force_reason + " Dozwolone są wyłącznie transakcje redukujące ekspozycję.")
                    if force_reason
                    else "Profil w trybie awaryjnym – dozwolone są wyłącznie transakcje redukujące ekspozycję."
                ),
            )

        if is_new_position:
            if state.active_positions() >= profile.max_positions():
                self._persist_state(profile_name)
                return RiskCheckResult(
                    allowed=False,
                    reason="Limit liczby równoległych pozycji został osiągnięty.",
                )

        incremental_notional = max(0.0, new_notional - current_notional)
        if not is_reducing and incremental_notional > 0:
            effective_leverage = profile.max_leverage()
            if effective_leverage <= 0:
                effective_leverage = 1.0
            effective_leverage = max(1.0, effective_leverage)
            usable_margin = max(0.0, account.available_margin - account.maintenance_margin)
            required_margin = incremental_notional / effective_leverage
            if required_margin > usable_margin:
                allowed_additional_notional = usable_margin * effective_leverage
                allowed_notional = current_notional + allowed_additional_notional
                allowed_quantity = max(0.0, allowed_notional - current_notional) / price
                self._persist_state(profile_name)
                return RiskCheckResult(
                    allowed=False,
                    reason="Niewystarczający wolny margines na otwarcie lub powiększenie pozycji.",
                    adjustments={"max_quantity": max(0.0, allowed_quantity)},
                )

        max_position_pct = profile.max_position_exposure()
        if not is_reducing and max_position_pct > 0:
            max_notional = max_position_pct * account.total_equity
            if max_notional <= 0:
                self._persist_state(profile_name)
                return RiskCheckResult(
                    allowed=False,
                    reason="Kapitał konta jest zbyt niski do otwierania nowych pozycji.",
                )
            if new_notional > max_notional:
                allowed_notional = max_notional
                allowed_quantity = max(0.0, allowed_notional - current_notional) / price
                self._persist_state(profile_name)
                return RiskCheckResult(
                    allowed=False,
                    reason="Wielkość zlecenia przekracza limit ekspozycji na instrument.",
                    adjustments={"max_quantity": max(0.0, allowed_quantity)},
                )

        max_leverage = profile.max_leverage()
        if not is_reducing and max_leverage > 0:
            gross_without_symbol = state.gross_notional() - current_notional
            max_total_notional = max_leverage * account.total_equity
            if max_total_notional <= 0:
                self._persist_state(profile_name)
                return RiskCheckResult(
                    allowed=False,
                    reason="Kapitał konta nie pozwala na otwieranie pozycji przy zadanej dźwigni.",
                )
            new_gross = gross_without_symbol + new_notional
            if new_gross > max_total_notional:
                remaining_notional = max(0.0, max_total_notional - gross_without_symbol)
                allowed_quantity = remaining_notional / price
                self._persist_state(profile_name)
                return RiskCheckResult(
                    allowed=False,
                    reason="Przekroczono maksymalną dźwignię portfela.",
                    adjustments={"max_quantity": max(0.0, allowed_quantity)},
                )

        self._persist_state(profile_name)

        return RiskCheckResult(allowed=True)

    def on_fill(
        self,
        *,
        profile_name: str,
        symbol: str,
        side: str,
        position_value: float,
        pnl: float,
        timestamp: datetime | None = None,
    ) -> None:
        profile = self._profiles.get(profile_name)
        if profile is None:
            raise KeyError(f"Profil ryzyka {profile_name} nie został zarejestrowany")

        state = self._states.get(profile_name)
        if state is None:
            raise KeyError(f"Brak stanu ryzyka dla profilu {profile_name}")

        now = timestamp or self._clock()
        current_day = now.date()
        if state.current_day != current_day:
            state.reset_for_new_day(equity=state.last_equity, day=current_day)

        state.daily_realized_pnl += pnl
        state.update_position(symbol, side=side, notional=max(0.0, position_value))
        self._persist_state(profile_name)

    def should_liquidate(self, *, profile_name: str) -> bool:
        state = self._states.get(profile_name)
        if state is None:
            raise KeyError(f"Brak stanu ryzyka dla profilu {profile_name}")
        return state.force_liquidation

    def _persist_state(self, profile_name: str) -> None:
        state = self._states[profile_name]
        self._repository.store(profile_name, state.to_mapping())


__all__ = ["InMemoryRiskRepository", "ThresholdRiskEngine", "RiskState", "PositionState"]
