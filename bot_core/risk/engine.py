"""Implementacja silnika ryzyka egzekwującego limity profili."""
from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date, datetime
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Sequence

from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.base import RiskCheckResult, RiskEngine, RiskProfile, RiskRepository
from bot_core.risk.events import RiskDecisionLog
from bot_core.risk.simulation import (
    DEFAULT_PROFILES,
    RiskSimulationSuite,
    build_profile,
    load_orders_from_parquet,
    run_profile_scenario,
)

try:  # pragma: no cover - moduły decision/tco mogą być opcjonalne w innych gałęziach
    from bot_core.decision import DecisionOrchestrator
    from bot_core.decision.models import (
        DecisionCandidate,
        DecisionEvaluation,
        ModelSelectionDetail,
        ModelSelectionMetadata,
        RiskSnapshot,
    )
except Exception:  # pragma: no cover - decyzje mogą nie być dostępne
    DecisionOrchestrator = None  # type: ignore
    DecisionCandidate = None  # type: ignore
    DecisionEvaluation = None  # type: ignore
    ModelSelectionDetail = None  # type: ignore
    ModelSelectionMetadata = None  # type: ignore
    RiskSnapshot = None  # type: ignore

_LOGGER = logging.getLogger(__name__)


def _normalize_position_side(side: str, *, default: str = "long") -> str:
    """Sprowadza oznaczenie strony pozycji do wartości "long"/"short"."""

    normalized = str(side or "").strip().lower()
    if normalized in {"long", "buy"}:
        return "long"
    if normalized in {"short", "sell"}:
        return "short"
    return default


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class PositionState:
    """Stan pojedynczej pozycji wykorzystywany przez silnik ryzyka."""

    side: str
    notional: float

    def __post_init__(self) -> None:
        self.side = _normalize_position_side(self.side)
        self.notional = max(0.0, float(self.notional))

    def to_mapping(self) -> Mapping[str, object]:
        return {"side": self.side, "notional": self.notional}

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "PositionState":
        side = str(data.get("side", "long"))
        notional = float(data.get("notional", 0.0))
        return cls(side=side, notional=notional)


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
        decision_log: RiskDecisionLog | None = None,
        decision_orchestrator: Any | None = None,
    ) -> None:
        self._repository = repository or InMemoryRiskRepository()
        self._clock = clock or datetime.utcnow
        self._profiles: Dict[str, RiskProfile] = {}
        self._states: Dict[str, RiskState] = {}
        self._decision_log = decision_log
        self._decision_orchestrator: Any | None = decision_orchestrator
        self._decision_model_outcomes: MutableMapping[str, MutableMapping[str, int]] = {}
        self._decision_model_rejections: MutableMapping[
            str, MutableMapping[str, int]
        ] = {}
        self._decision_model_metrics: MutableMapping[
            str, MutableMapping[str, float | int]
        ] = {}
        self._decision_model_selection: MutableMapping[
            str,
            MutableMapping[str, object],
        ] = {}
        self._decision_orchestrator_activity: MutableMapping[
            str, float | int | None
        ] = self._init_decision_orchestrator_activity()
        self._decision_orchestrator_errors: MutableMapping[str, int] = {}
        self._decision_orchestrator_activity_profiles: MutableMapping[
            str,
            MutableMapping[str, float | int | None],
        ] = {}
        self._decision_orchestrator_errors_by_profile: MutableMapping[
            str,
            MutableMapping[str, int],
        ] = {}

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

    def _init_decision_orchestrator_activity(
        self,
    ) -> MutableMapping[str, float | int | None]:
        return {
            "attempts": 0,
            "evaluated": 0,
            "accepted": 0,
            "rejected": 0,
            "errors": 0,
            "skipped": 0,
            "duration_ms_sum": 0.0,
            "duration_ms_count": 0,
            "duration_ms_max": None,
            "duration_ms_min": None,
        }

    def attach_decision_orchestrator(self, orchestrator: Any | None) -> None:
        """Podłącza DecisionOrchestrator do dodatkowych kontroli kosztowych."""

        self._decision_orchestrator = orchestrator

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

        def deny(
            reason: str,
            *,
            adjustments: Mapping[str, float] | None = None,
            metadata: Mapping[str, object] | None = None,
        ) -> RiskCheckResult:
            result = RiskCheckResult(
                allowed=False,
                reason=reason,
                adjustments=adjustments,
                metadata=metadata,
            )
            self._persist_state(profile_name)
            return self._finalize_decision(
                profile_name=profile_name,
                request=request,
                account=account,
                result=result,
                state=state,
            )

        price = request.price
        if price is None or price <= 0:
            return deny(
                "Brak ceny referencyjnej uniemożliwia obliczenie ekspozycji. Podaj midpoint z orderbooka lub cenę limit."
            )

        symbol = request.symbol
        position = state.positions.get(symbol)
        is_buy = request.side.lower() == "buy"
        notional = abs(request.quantity) * price
        current_notional = position.notional if position else 0.0
        current_side = (
            _normalize_position_side(position.side)
            if position
            else ("long" if is_buy else "short")
        )

        if position:
            position_side = _normalize_position_side(position.side)
            if position_side == "long" and not is_buy and notional > current_notional:
                return deny(
                    "Zlecenie sprzedaży przekracza wielkość istniejącej pozycji. Zamknij pozycję przed odwróceniem kierunku."
                )
            if position_side == "short" and is_buy and notional > current_notional:
                return deny(
                    "Zlecenie kupna przekracza wielkość krótkiej pozycji. Zamknij pozycję przed odwróceniem kierunku."
                )

        metadata: Mapping[str, object] = request.metadata or {}

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
            message = (
                force_reason + " Dozwolone są wyłącznie transakcje redukujące ekspozycję."
            ) if force_reason else "Profil w trybie awaryjnym – dozwolone są wyłącznie transakcje redukujące ekspozycję."
            return deny(message)

        if not is_reducing:
            atr_value = _coerce_float(metadata.get("atr")) if metadata else None
            if atr_value is None:
                atr_value = _coerce_float(getattr(request, "atr", None))
            stop_price_value = _coerce_float(metadata.get("stop_price")) if metadata else None
            if stop_price_value is None:
                stop_price_value = _coerce_float(getattr(request, "stop_price", None))
            target_vol = profile.target_volatility()
            atr_multiple = profile.stop_loss_atr_multiple()

            if target_vol > 0 and account.total_equity > 0:
                if atr_value is None or atr_value <= 0:
                    return deny(
                        "Brak prawidłowego ATR w metadanych zlecenia – nie można wyznaczyć wielkości pozycji."
                    )

                if stop_price_value is None:
                    return deny(
                        "Metadane zlecenia nie zawierają stop_price wymaganej do kontroli ryzyka."
                    )

                if is_buy:
                    stop_distance = price - stop_price_value
                else:
                    stop_distance = stop_price_value - price

                if stop_distance <= 0:
                    return deny(
                        "Stop loss musi znajdować się po stronie ograniczającej stratę (nieprawidłowy stop_price)."
                    )

                minimum_distance = atr_value * max(atr_multiple, 1.0)
                if minimum_distance > 0 and stop_distance + 1e-9 < minimum_distance:
                    return deny("Stop loss jest ciaśniejszy niż wymaga tego polityka profilu ryzyka.")

                max_risk_capital = target_vol * account.total_equity
                if max_risk_capital <= 0:
                    return deny("Kapitał lub target volatility profilu uniemożliwia otwarcie pozycji.")

                allowed_total_quantity = max_risk_capital / max(stop_distance, 1e-12)
                current_quantity = current_notional / price
                new_quantity = new_notional / price if price > 0 else 0.0

                if allowed_total_quantity <= 0:
                    return deny(
                        "Target volatility profilu blokuje otwarcie pozycji przy zadanych parametrach ATR/stop."
                    )

                if new_quantity > allowed_total_quantity:
                    allowed_increment = max(0.0, allowed_total_quantity - current_quantity)
                    return deny(
                        "Zlecenie przekracza limit wynikający z target volatility profilu ryzyka.",
                        adjustments={"max_quantity": max(0.0, allowed_increment)},
                    )

        if is_new_position:
            if state.active_positions() >= profile.max_positions():
                return deny("Limit liczby równoległych pozycji został osiągnięty.")

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
                return deny(
                    "Niewystarczający wolny margines na otwarcie lub powiększenie pozycji.",
                    adjustments={"max_quantity": max(0.0, allowed_quantity)},
                )

        max_position_pct = profile.max_position_exposure()
        if not is_reducing and max_position_pct > 0:
            max_notional = max_position_pct * account.total_equity
            if max_notional <= 0:
                return deny("Kapitał konta jest zbyt niski do otwierania nowych pozycji.")
            if new_notional > max_notional:
                allowed_notional = max_notional
                allowed_quantity = max(0.0, allowed_notional - current_notional) / price
                return deny(
                    "Wielkość zlecenia przekracza limit ekspozycji na instrument.",
                    adjustments={"max_quantity": max(0.0, allowed_quantity)},
                )

        max_leverage = profile.max_leverage()
        if not is_reducing and max_leverage > 0:
            gross_without_symbol = state.gross_notional() - current_notional
            max_total_notional = max_leverage * account.total_equity
            if max_total_notional <= 0:
                return deny("Kapitał konta nie pozwala na otwieranie pozycji przy zadanej dźwigni.")
            new_gross = gross_without_symbol + new_notional
            if new_gross > max_total_notional:
                remaining_notional = max(0.0, max_total_notional - gross_without_symbol)
                allowed_quantity = remaining_notional / price
                return deny(
                    "Przekroczono maksymalną dźwignię portfela.",
                    adjustments={"max_quantity": max(0.0, allowed_quantity)},
                )

        if not is_reducing:
            target_vol = profile.target_volatility()
            stop_multiple = profile.stop_loss_atr_multiple()
            if target_vol > 0 and stop_multiple > 0:
                atr_raw = request.atr
                try:
                    atr_value = float(atr_raw) if atr_raw is not None else None
                except (TypeError, ValueError):  # pragma: no cover - defensywnie
                    atr_value = None
                if atr_value is None or atr_value <= 0:
                    return deny(
                        "Brak wartości ATR uniemożliwia wyznaczenie wielkości pozycji w oparciu o profil ryzyka."
                    )

                stop_raw = request.stop_price
                try:
                    stop_price = float(stop_raw) if stop_raw is not None else None
                except (TypeError, ValueError):  # pragma: no cover - defensywnie
                    stop_price = None
                if stop_price is None or stop_price <= 0:
                    return deny(
                        "Zlecenie wymaga ustawienia stop loss opartego o ATR zgodnie z profilem ryzyka."
                    )

                entering_long = is_buy and current_side == "long"
                entering_short = (not is_buy) and current_side == "short"

                if not entering_long and not entering_short:
                    entering_long = is_buy
                    entering_short = not is_buy

                expected_stop_distance = atr_value * stop_multiple
                if expected_stop_distance <= 0:
                    return deny(
                        "Parametry ATR i mnożnik stop loss prowadzą do nieprawidłowego dystansu zabezpieczenia."
                    )

                if entering_long:
                    actual_distance = price - stop_price
                else:
                    actual_distance = stop_price - price

                if actual_distance <= 0:
                    return deny("Cena stop loss musi znajdować się po właściwej stronie ceny wejścia.")

                tolerance = max(expected_stop_distance * 1e-4, 1e-8)
                if actual_distance + tolerance < expected_stop_distance:
                    return deny(
                        "Cena stop loss musi odpowiadać wielokrotności ATR określonej w profilu ryzyka."
                    )

                risk_budget = target_vol * account.total_equity
                if risk_budget <= 0:
                    return deny(
                        "Kapitał lub docelowa zmienność profilu uniemożliwiają otwarcie nowej pozycji."
                    )

                allowed_total_quantity = risk_budget / max(actual_distance, 1e-12)
                if allowed_total_quantity <= 0:
                    return deny("Docelowa zmienność profilu ogranicza wielkość pozycji do zera.")

                current_quantity = current_notional / price if price > 0 else 0.0
                new_quantity = new_notional / price if price > 0 else 0.0
                if new_quantity > allowed_total_quantity + 1e-9:
                    allowed_additional_quantity = max(0.0, allowed_total_quantity - current_quantity)
                    return deny(
                        "Wielkość zlecenia przekracza limit wynikający z docelowej zmienności profilu.",
                        adjustments={"max_quantity": max(0.0, allowed_additional_quantity)},
                    )

        evaluation_metadata: Mapping[str, object] | None = None
        if self._decision_orchestrator is not None:
            (
                evaluation,
                evaluation_payload,
                orchestrator_stats,
            ) = self._maybe_run_decision_orchestrator(
                request=request,
                profile_name=profile_name,
                account=account,
                state=state,
            )
            if evaluation_payload:
                evaluation_metadata = {"decision_orchestrator": evaluation_payload}
            accepted_flag: bool | None = None
            raw_model_name: str | None = None
            if evaluation is not None:
                raw_model_name, accepted_value = self._extract_model_and_acceptance(
                    evaluation
                )
                if accepted_value is not None:
                    accepted_flag = bool(accepted_value)
            if orchestrator_stats:
                self._record_decision_orchestrator_activity(
                    orchestrator_stats,
                    accepted=accepted_flag,
                    profile_name=profile_name,
                )
            if evaluation is None:
                if evaluation_payload and evaluation_payload.get("status") == "error":
                    reason = self._format_decision_error(evaluation_payload)
                    return deny(reason, metadata=evaluation_metadata)
            else:
                outcome = self._record_decision_model_outcome(
                    evaluation,
                    model_name=raw_model_name,
                    accepted=accepted_flag,
                )
                if outcome is not None:
                    normalized_model, accepted_flag = outcome
                else:
                    normalized_model = str(raw_model_name or "__unknown__").lower()
                    accepted_flag = accepted_flag
                self._record_decision_model_metrics(
                    normalized_model, evaluation, accepted_flag
                )
                self._record_decision_model_selection(evaluation)
                if accepted_flag is False:
                    self._record_decision_model_rejection(
                        normalized_model, evaluation
                    )
                    reason = self._format_decision_denial_reason(evaluation)
                    return deny(reason, metadata=evaluation_metadata)

        self._persist_state(profile_name)
        result = RiskCheckResult(allowed=True, metadata=evaluation_metadata)
        return self._finalize_decision(
            profile_name=profile_name,
            request=request,
            account=account,
            result=result,
            state=state,
        )

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
        normalized_side = _normalize_position_side(side, default="long")
        state.update_position(symbol, side=normalized_side, notional=max(0.0, position_value))
        self._persist_state(profile_name)

    def snapshot_state(self, profile_name: str) -> Mapping[str, object] | None:
        state = self._states.get(profile_name)
        if state is None:
            return None

        snapshot: MutableMapping[str, object] = dict(state.to_mapping())
        snapshot["gross_notional"] = state.gross_notional()
        snapshot["active_positions"] = state.active_positions()
        snapshot["daily_loss_pct"] = state.daily_loss_pct()
        current_equity = state.last_equity or state.start_of_day_equity
        snapshot["drawdown_pct"] = state.drawdown_pct(current_equity)

        profile = self._profiles.get(profile_name)
        if profile is not None:
            snapshot["limits"] = {
                "max_positions": profile.max_positions(),
                "max_leverage": profile.max_leverage(),
                "daily_loss_limit": profile.daily_loss_limit(),
                "drawdown_limit": profile.drawdown_limit(),
                "max_position_pct": profile.max_position_exposure(),
                "target_volatility": profile.target_volatility(),
                "stop_loss_atr_multiple": profile.stop_loss_atr_multiple(),
            }

        return snapshot

    def should_liquidate(self, *, profile_name: str) -> bool:
        state = self._states.get(profile_name)
        if state is None:
            raise KeyError(f"Brak stanu ryzyka dla profilu {profile_name}")
        return state.force_liquidation

    def recent_decisions(
        self,
        *,
        profile_name: str | None = None,
        limit: int = 20,
    ) -> Sequence[Mapping[str, object]]:
        if self._decision_log is None:
            return ()
        return self._decision_log.tail(profile=profile_name, limit=limit)

    def decision_orchestrator_activity(
        self, *, reset: bool = False
    ) -> Mapping[str, object]:
        """Zwraca zbiorcze statystyki aktywności DecisionOrchestratora."""

        snapshot = self._build_decision_orchestrator_activity_snapshot(
            self._decision_orchestrator_activity,
            self._decision_orchestrator_errors,
        )

        if reset:
            self._decision_orchestrator_activity = (
                self._init_decision_orchestrator_activity()
            )
            self._decision_orchestrator_errors = {}

        return snapshot

    def decision_orchestrator_activity_by_profile(
        self, *, reset: bool = False
    ) -> Mapping[str, Mapping[str, object]]:
        """Zwraca statystyki aktywności orchestratora pogrupowane per profil."""

        snapshot: dict[str, Mapping[str, object]] = {}
        for profile, counters in self._decision_orchestrator_activity_profiles.items():
            profile_errors = self._decision_orchestrator_errors_by_profile.get(
                profile,
                {},
            )
            snapshot[profile] = self._build_decision_orchestrator_activity_snapshot(
                counters,
                profile_errors,
            )

        if reset:
            self._decision_orchestrator_activity_profiles = {}
            self._decision_orchestrator_errors_by_profile = {}

        return snapshot

    def decision_model_outcomes(self, *, reset: bool = False) -> Mapping[str, Mapping[str, int]]:
        """Zwraca agregację przyjęć i odrzuceń per model DecisionOrchestratora."""

        snapshot: dict[str, Mapping[str, int]] = {}
        for name, counters in self._decision_model_outcomes.items():
            snapshot[name] = {
                "accepted": int(counters.get("accepted", 0)),
                "rejected": int(counters.get("rejected", 0)),
            }
        if reset:
            self._decision_model_outcomes.clear()
        return snapshot

    def decision_model_rejection_reasons(
        self, *, reset: bool = False
    ) -> Mapping[str, Mapping[str, int]]:
        """Zwraca liczniki powodów odrzuceń decyzji pogrupowane per model."""

        snapshot: dict[str, Mapping[str, int]] = {}
        for name, counters in self._decision_model_rejections.items():
            snapshot[name] = {
                str(reason): int(count)
                for reason, count in counters.items()
            }
        if reset:
            self._decision_model_rejections.clear()
        return snapshot

    def decision_model_metrics(
        self, *, reset: bool = False
    ) -> Mapping[str, Mapping[str, object]]:
        """Zwraca zagregowane metryki ewaluacji modeli orchestratora."""

        snapshot: dict[str, Mapping[str, object]] = {}
        metric_keys = (
            "cost_bps",
            "net_edge_bps",
            "model_expected_return_bps",
            "model_success_probability",
        )
        for name, counters in self._decision_model_metrics.items():
            entry: dict[str, object] = {
                "evaluations": int(counters.get("evaluations", 0)),
                "accepted": int(counters.get("accepted", 0)),
                "rejected": int(counters.get("rejected", 0)),
            }
            for metric in metric_keys:
                sum_key = f"{metric}_sum"
                count_key = f"{metric}_count"
                total = float(counters.get(sum_key, 0.0) or 0.0)
                count = int(counters.get(count_key, 0))
                entry[metric] = {
                    "sum": total if count else 0.0,
                    "average": (total / count) if count else None,
                    "count": count,
                }
            snapshot[name] = entry
        if reset:
            self._decision_model_metrics.clear()
        return snapshot

    def decision_model_selection_stats(
        self, *, reset: bool = False
    ) -> Mapping[str, Mapping[str, object]]:
        """Zwraca statystyki wyboru modeli przez orchestratora."""

        snapshot: dict[str, Mapping[str, object]] = {}
        metric_keys = ("score", "weight", "effective_score")
        for name, counters in self._decision_model_selection.items():
            reasons_raw = counters.get("reasons")
            if isinstance(reasons_raw, Mapping):
                reasons_snapshot = {
                    str(reason): int(count)
                    for reason, count in reasons_raw.items()
                }
            else:
                reasons_snapshot = {}
            entry: dict[str, object] = {
                "considered": int(counters.get("considered", 0)),
                "selected": int(counters.get("selected", 0)),
                "available": int(counters.get("available", 0)),
                "unavailable": int(counters.get("unavailable", 0)),
                "reasons": reasons_snapshot,
            }
            for metric in metric_keys:
                sum_key = f"{metric}_sum"
                count_key = f"{metric}_count"
                total = float(counters.get(sum_key, 0.0) or 0.0)
                count = int(counters.get(count_key, 0))
                entry[metric] = {
                    "sum": total if count else 0.0,
                    "average": (total / count) if count else None,
                    "count": count,
                }
            snapshot[name] = entry
        if reset:
            self._decision_model_selection.clear()
        return snapshot

    def _maybe_run_decision_orchestrator(
        self,
        *,
        request: OrderRequest,
        profile_name: str,
        account: AccountSnapshot,
        state: RiskState,
    ) -> tuple[
        Any | None,
        Mapping[str, object] | None,
        Mapping[str, object] | None,
    ]:
        if self._decision_orchestrator is None:
            return None, None, None
        if not _ensure_decision_models():
            return None, None, {
                "status": "skipped",
                "attempted": False,
                "duration_ms": 0.0,
            }

        start = perf_counter()

        def build_stats(
            status: str, *, attempted: bool, error: str | None = None
        ) -> Mapping[str, object]:
            stats: dict[str, object] = {
                "status": status,
                "attempted": attempted,
                "duration_ms": (perf_counter() - start) * 1000.0,
            }
            if error is not None:
                stats["error"] = error
            return stats

        candidate_payload, error_payload = self._extract_candidate_payload(
            request.metadata
        )
        if error_payload:
            error_code = str(error_payload.get("error", "unknown_error"))
            return None, error_payload, build_stats(
                "error", attempted=False, error=error_code
            )
        if candidate_payload is None:
            return None, None, build_stats("skipped", attempted=False)

        payload = dict(candidate_payload)
        payload.setdefault("risk_profile", profile_name)
        if request.symbol and "symbol" not in payload:
            payload["symbol"] = request.symbol

        if "notional" not in payload or payload["notional"] in (None, 0, 0.0):
            price_value = _coerce_float(request.price)
            if price_value and price_value > 0:
                payload["notional"] = abs(request.quantity) * price_value

        try:
            candidate = DecisionCandidate.from_mapping(payload)
        except Exception:
            _LOGGER.warning(
                "DecisionOrchestrator: nieprawidłowe dane kandydata", exc_info=True
            )
            return (
                None,
                {"status": "error", "error": "invalid_candidate"},
                build_stats("error", attempted=True, error="invalid_candidate"),
            )

        snapshot = self._build_decision_snapshot(profile_name, state, account)
        try:
            evaluation = self._decision_orchestrator.evaluate_candidate(  # type: ignore[union-attr]
                candidate,
                snapshot,
            )
        except Exception:
            _LOGGER.exception("DecisionOrchestrator: błąd ewaluacji")
            return (
                None,
                {"status": "error", "error": "evaluation_failed"},
                build_stats("error", attempted=True, error="evaluation_failed"),
            )

        return (
            evaluation,
            self._serialize_decision_evaluation(evaluation),
            build_stats("evaluated", attempted=True),
        )

    def _extract_candidate_payload(
        self, metadata: Mapping[str, object] | None
    ) -> tuple[Mapping[str, object] | None, Mapping[str, object] | None]:
        if not isinstance(metadata, Mapping):
            return None, None

        candidate_raw: object | None = metadata.get("decision_candidate")
        if candidate_raw is None:
            engine_section = metadata.get("decision_engine")
            if isinstance(engine_section, Mapping):
                nested = engine_section.get("candidate")
                if isinstance(nested, Mapping):
                    candidate_raw = nested
                else:
                    candidate_raw = engine_section

        if candidate_raw is None:
            return None, None

        if not isinstance(candidate_raw, Mapping):
            return None, {"status": "error", "error": "invalid_candidate_payload"}

        return dict(candidate_raw), None

    def _build_decision_snapshot(
        self,
        profile_name: str,
        state: RiskState,
        account: AccountSnapshot,
    ) -> Mapping[str, object] | Any:
        snapshot_payload: MutableMapping[str, object] = dict(state.to_mapping())
        last_equity = state.last_equity or account.total_equity
        snapshot_payload["last_equity"] = last_equity
        snapshot_payload["start_of_day_equity"] = (
            state.start_of_day_equity or account.total_equity
        )
        snapshot_payload["daily_realized_pnl"] = state.daily_realized_pnl
        snapshot_payload["gross_notional"] = state.gross_notional()
        snapshot_payload["active_positions"] = state.active_positions()
        snapshot_payload["daily_loss_pct"] = state.daily_loss_pct()
        snapshot_payload["drawdown_pct"] = state.drawdown_pct(last_equity)
        snapshot_payload["force_liquidation"] = state.force_liquidation

        if RiskSnapshot is not None:
            try:
                return RiskSnapshot.from_mapping(profile_name, snapshot_payload)
            except Exception:  # pragma: no cover - diagnostyka snapshotu
                _LOGGER.debug(
                    "DecisionOrchestrator: nie udało się zbudować RiskSnapshot",
                    exc_info=True,
                )

        snapshot_payload["profile"] = profile_name
        return snapshot_payload

    def _serialize_decision_evaluation(
        self, evaluation: Any
    ) -> Mapping[str, object]:
        if DecisionEvaluation is not None and isinstance(evaluation, DecisionEvaluation):
            return {
                "status": "evaluated",
                "accepted": evaluation.accepted,
                "cost_bps": evaluation.cost_bps,
                "net_edge_bps": evaluation.net_edge_bps,
                "reasons": list(evaluation.reasons),
                "risk_flags": list(evaluation.risk_flags),
                "stress_failures": list(evaluation.stress_failures),
                "candidate": evaluation.candidate.to_mapping(),
                "model_expected_return_bps": evaluation.model_expected_return_bps,
                "model_success_probability": evaluation.model_success_probability,
                "model_name": evaluation.model_name,
                "model_selection": (
                    evaluation.model_selection.to_mapping()
                    if evaluation.model_selection is not None
                    else None
                ),
            }
        if isinstance(evaluation, Mapping):
            return dict(evaluation)
        return {"status": "evaluated", "accepted": bool(getattr(evaluation, "accepted", False))}

    def _format_decision_denial_reason(self, evaluation: Any) -> str:
        if DecisionEvaluation is not None and isinstance(evaluation, DecisionEvaluation):
            reasons = list(evaluation.reasons)
            reasons.extend(str(flag) for flag in evaluation.risk_flags)
            reasons.extend(str(flag) for flag in evaluation.stress_failures)
        else:
            reasons = []
            if isinstance(evaluation, Mapping):
                reasons = [
                    *(str(reason) for reason in evaluation.get("reasons", []) or []),
                    *(str(reason) for reason in evaluation.get("risk_flags", []) or []),
                    *(str(reason) for reason in evaluation.get("stress_failures", []) or []),
                ]

        reasons = [reason for reason in reasons if reason]
        if not reasons:
            return "DecisionOrchestrator odrzucił decyzję bez szczegółów."
        return "DecisionOrchestrator odrzucił decyzję: " + "; ".join(reasons)

    def _format_decision_error(self, payload: Mapping[str, object]) -> str:
        detail = str(payload.get("error", "unknown_error"))
        return f"DecisionOrchestrator nie mógł ocenić kandydata ({detail})."

    def _finalize_decision(
        self,
        *,
        profile_name: str,
        request: OrderRequest,
        account: AccountSnapshot,
        result: RiskCheckResult,
        state: RiskState,
    ) -> RiskCheckResult:
        if self._decision_log is not None:
            price_value = _coerce_float(request.price)
            quantity_value = _coerce_float(request.quantity) or 0.0
            notional = None
            if price_value and price_value > 0:
                notional = abs(quantity_value) * price_value

            metadata: dict[str, object] = {
                "account": {
                    "total_equity": float(account.total_equity),
                    "available_margin": float(account.available_margin),
                    "maintenance_margin": float(account.maintenance_margin),
                },
                "state": {
                    "force_liquidation": state.force_liquidation,
                    "gross_notional": state.gross_notional(),
                    "active_positions": state.active_positions(),
                    "daily_loss_pct": state.daily_loss_pct(),
                    "drawdown_pct": state.drawdown_pct(state.last_equity or state.start_of_day_equity),
                    "current_day": state.current_day.isoformat(),
                },
            }
            if isinstance(request.metadata, Mapping) and request.metadata:
                metadata["request_metadata"] = {str(k): v for k, v in request.metadata.items()}

            if result.metadata:
                metadata.update({str(k): v for k, v in result.metadata.items()})

            profile = self._profiles.get(profile_name)
            if profile is not None:
                metadata["limits"] = {
                    "max_positions": profile.max_positions(),
                    "max_leverage": profile.max_leverage(),
                    "daily_loss_limit": profile.daily_loss_limit(),
                    "drawdown_limit": profile.drawdown_limit(),
                    "max_position_pct": profile.max_position_exposure(),
                    "target_volatility": profile.target_volatility(),
                    "stop_loss_atr_multiple": profile.stop_loss_atr_multiple(),
                }

            self._decision_log.record(
                profile=profile_name,
                symbol=request.symbol,
                side=request.side,
                quantity=quantity_value,
                price=price_value,
                notional=notional,
                allowed=result.allowed,
                reason=result.reason,
                adjustments=result.adjustments,
                metadata=metadata,
            )

        return result

    def _persist_state(self, profile_name: str) -> None:
        state = self._states[profile_name]
        self._repository.store(profile_name, state.to_mapping())

    def _record_decision_model_outcome(
        self,
        evaluation: Any,
        *,
        model_name: str | None = None,
        accepted: bool | None = None,
    ) -> tuple[str, bool] | None:
        if accepted is None or model_name is None:
            extracted_name, extracted_accepted = self._extract_model_and_acceptance(
                evaluation
            )
            if model_name is None:
                model_name = extracted_name
            if accepted is None:
                accepted = extracted_accepted

        if accepted is None:
            return None

        normalized_name = str(model_name or "__unknown__").lower()
        counters = self._decision_model_outcomes.setdefault(
            normalized_name,
            {"accepted": 0, "rejected": 0},
        )
        bucket = "accepted" if accepted else "rejected"
        counters[bucket] = counters.get(bucket, 0) + 1
        return normalized_name, bool(accepted)

    def _record_decision_model_rejection(
        self, model_name: str, evaluation: Any
    ) -> None:
        reasons = self._extract_rejection_reasons(evaluation)
        if not reasons:
            return
        counters = self._decision_model_rejections.setdefault(model_name, {})
        for reason in reasons:
            counters[reason] = counters.get(reason, 0) + 1

    def _record_decision_model_metrics(
        self, model_name: str, evaluation: Any, accepted: bool | None
    ) -> None:
        counters = self._decision_model_metrics.setdefault(
            model_name,
            {
                "evaluations": 0,
                "accepted": 0,
                "rejected": 0,
                "cost_bps_sum": 0.0,
                "cost_bps_count": 0,
                "net_edge_bps_sum": 0.0,
                "net_edge_bps_count": 0,
                "model_expected_return_bps_sum": 0.0,
                "model_expected_return_bps_count": 0,
                "model_success_probability_sum": 0.0,
                "model_success_probability_count": 0,
            },
        )
        counters["evaluations"] = int(counters.get("evaluations", 0)) + 1
        if accepted is True:
            counters["accepted"] = int(counters.get("accepted", 0)) + 1
        elif accepted is False:
            counters["rejected"] = int(counters.get("rejected", 0)) + 1

        metrics = self._extract_evaluation_metrics(evaluation)
        for metric, value in metrics.items():
            if value is None:
                continue
            sum_key = f"{metric}_sum"
            count_key = f"{metric}_count"
            counters[sum_key] = float(counters.get(sum_key, 0.0) or 0.0) + float(value)
            counters[count_key] = int(counters.get(count_key, 0)) + 1

    def _record_decision_orchestrator_activity(
        self,
        stats: Mapping[str, object] | None,
        *,
        accepted: bool | None,
        profile_name: str,
    ) -> None:
        if not stats:
            return

        status = str(stats.get("status", "")).lower()
        attempted = bool(stats.get("attempted"))
        duration_value = _coerce_float(stats.get("duration_ms"))

        self._update_orchestrator_activity_counters(
            self._decision_orchestrator_activity,
            status=status,
            attempted=attempted,
            accepted=accepted,
            duration_value=duration_value,
        )

        if profile_name:
            profile_counters = self._decision_orchestrator_activity_profiles.setdefault(
                profile_name,
                self._init_decision_orchestrator_activity(),
            )
            self._update_orchestrator_activity_counters(
                profile_counters,
                status=status,
                attempted=attempted,
                accepted=accepted,
                duration_value=duration_value,
            )

        if status == "error":
            reason = stats.get("error")
            if not reason:
                return
            reason_key = str(reason)
            current = self._decision_orchestrator_errors.get(reason_key, 0)
            self._decision_orchestrator_errors[reason_key] = int(current) + 1
            if profile_name:
                profile_errors = self._decision_orchestrator_errors_by_profile.setdefault(
                    profile_name,
                    {},
                )
                profile_errors[reason_key] = int(
                    profile_errors.get(reason_key, 0)
                ) + 1

    def _update_orchestrator_activity_counters(
        self,
        counters: MutableMapping[str, float | int | None],
        *,
        status: str,
        attempted: bool,
        accepted: bool | None,
        duration_value: float | None,
    ) -> None:
        if attempted:
            counters["attempts"] = int(counters.get("attempts", 0)) + 1

        if status == "evaluated":
            counters["evaluated"] = int(counters.get("evaluated", 0)) + 1
        elif status == "error":
            counters["errors"] = int(counters.get("errors", 0)) + 1
        elif status == "skipped":
            counters["skipped"] = int(counters.get("skipped", 0)) + 1

        if accepted is True:
            counters["accepted"] = int(counters.get("accepted", 0)) + 1
        elif accepted is False:
            counters["rejected"] = int(counters.get("rejected", 0)) + 1

        if attempted and duration_value is not None:
            counters["duration_ms_sum"] = float(
                counters.get("duration_ms_sum", 0.0) or 0.0
            ) + float(duration_value)
            count = int(counters.get("duration_ms_count", 0)) + 1
            counters["duration_ms_count"] = count

            previous_max = counters.get("duration_ms_max")
            previous_min = counters.get("duration_ms_min")
            counters["duration_ms_max"] = (
                float(duration_value)
                if previous_max is None
                else max(float(previous_max), float(duration_value))
            )
            counters["duration_ms_min"] = (
                float(duration_value)
                if previous_min is None
                else min(float(previous_min), float(duration_value))
            )

    def _build_decision_orchestrator_activity_snapshot(
        self,
        counters: Mapping[str, float | int | None],
        errors: Mapping[str, int],
    ) -> Mapping[str, object]:
        duration_count = int(counters.get("duration_ms_count", 0))
        duration_sum = float(counters.get("duration_ms_sum", 0.0) or 0.0)
        duration_average = (
            duration_sum / duration_count if duration_count else None
        )
        duration_max = counters.get("duration_ms_max") if duration_count else None
        duration_min = counters.get("duration_ms_min") if duration_count else None

        snapshot: dict[str, object] = {
            "attempts": int(counters.get("attempts", 0)),
            "evaluated": int(counters.get("evaluated", 0)),
            "accepted": int(counters.get("accepted", 0)),
            "rejected": int(counters.get("rejected", 0)),
            "errors": int(counters.get("errors", 0)),
            "skipped": int(counters.get("skipped", 0)),
            "duration_ms": {
                "sum": duration_sum if duration_count else 0.0,
                "average": duration_average,
                "max": (
                    float(duration_max)
                    if duration_max is not None and duration_count
                    else None
                ),
                "min": (
                    float(duration_min)
                    if duration_min is not None and duration_count
                    else None
                ),
                "count": duration_count,
            },
            "error_reasons": {
                str(reason): int(count)
                for reason, count in errors.items()
            },
        }

        return snapshot

    def _record_decision_model_selection(self, evaluation: Any) -> None:
        selection = self._extract_model_selection(evaluation)
        if selection is None:
            return
        selected_name, candidates = selection
        normalized_selected = (
            str(selected_name).lower() if selected_name is not None else None
        )
        for detail in candidates:
            name_raw = detail.get("name")
            if not name_raw:
                continue
            normalized_name = str(name_raw).lower()
            counters = self._decision_model_selection.setdefault(
                normalized_name,
                {
                    "considered": 0,
                    "selected": 0,
                    "available": 0,
                    "unavailable": 0,
                    "reasons": {},
                },
            )
            counters["considered"] = int(counters.get("considered", 0)) + 1
            if normalized_selected is not None and normalized_name == normalized_selected:
                counters["selected"] = int(counters.get("selected", 0)) + 1

            available_value = detail.get("available")
            if available_value is True:
                counters["available"] = int(counters.get("available", 0)) + 1
            elif available_value is False:
                counters["unavailable"] = int(counters.get("unavailable", 0)) + 1

            reason_value = detail.get("reason")
            if reason_value:
                reasons = counters.setdefault("reasons", {})
                reason_key = str(reason_value)
                reasons[reason_key] = int(reasons.get(reason_key, 0)) + 1

            for metric in ("score", "weight", "effective_score"):
                metric_value = _coerce_float(detail.get(metric))
                if metric_value is None:
                    continue
                sum_key = f"{metric}_sum"
                count_key = f"{metric}_count"
                counters[sum_key] = float(counters.get(sum_key, 0.0) or 0.0) + float(
                    metric_value
                )
                counters[count_key] = int(counters.get(count_key, 0)) + 1

    def _extract_model_and_acceptance(
        self, evaluation: Any
    ) -> tuple[str | None, bool | None]:
        model_name: str | None = None
        accepted: bool | None = None

        if DecisionEvaluation is not None and isinstance(evaluation, DecisionEvaluation):
            model_name = evaluation.model_name
            accepted = evaluation.accepted
        elif isinstance(evaluation, Mapping):
            model_raw = evaluation.get("model_name")  # type: ignore[index]
            if model_raw is not None:
                model_name = str(model_raw)
            accepted_value = evaluation.get("accepted")  # type: ignore[index]
            if accepted_value is not None:
                accepted = bool(accepted_value)
        else:
            model_attr = getattr(evaluation, "model_name", None)
            if model_attr is not None:
                model_name = str(model_attr)
            accepted_attr = getattr(evaluation, "accepted", None)
            if accepted_attr is not None:
                accepted = bool(accepted_attr)

        return model_name, accepted

    def _extract_evaluation_metrics(self, evaluation: Any) -> Mapping[str, float | None]:
        if (
            DecisionEvaluation is not None
            and isinstance(evaluation, DecisionEvaluation)
        ):
            return {
                "cost_bps": _coerce_float(evaluation.cost_bps),
                "net_edge_bps": _coerce_float(evaluation.net_edge_bps),
                "model_expected_return_bps": _coerce_float(
                    evaluation.model_expected_return_bps
                ),
                "model_success_probability": _coerce_float(
                    evaluation.model_success_probability
                ),
            }
        if isinstance(evaluation, Mapping):
            return {
                "cost_bps": _coerce_float(evaluation.get("cost_bps")),
                "net_edge_bps": _coerce_float(evaluation.get("net_edge_bps")),
                "model_expected_return_bps": _coerce_float(
                    evaluation.get("model_expected_return_bps")
                ),
                "model_success_probability": _coerce_float(
                    evaluation.get("model_success_probability")
                ),
            }

        return {
            "cost_bps": _coerce_float(getattr(evaluation, "cost_bps", None)),
            "net_edge_bps": _coerce_float(getattr(evaluation, "net_edge_bps", None)),
            "model_expected_return_bps": _coerce_float(
                getattr(evaluation, "model_expected_return_bps", None)
            ),
            "model_success_probability": _coerce_float(
                getattr(evaluation, "model_success_probability", None)
            ),
        }

    def _extract_rejection_reasons(self, evaluation: Any) -> Sequence[str]:
        if DecisionEvaluation is not None and isinstance(evaluation, DecisionEvaluation):
            sources: Iterable[object] = (
                evaluation.reasons,
                evaluation.risk_flags,
                evaluation.stress_failures,
            )
        elif isinstance(evaluation, Mapping):
            sources = (
                evaluation.get("reasons"),
                evaluation.get("risk_flags"),
                evaluation.get("stress_failures"),
            )
        else:
            sources = (
                getattr(evaluation, "reasons", ()),
                getattr(evaluation, "risk_flags", ()),
                getattr(evaluation, "stress_failures", ()),
            )

        collected: list[str] = []
        for source in sources:
            collected.extend(self._coerce_reason_entries(source))
        return [reason for reason in collected if reason]

    def _coerce_reason_entries(self, payload: object) -> Sequence[str]:
        if payload is None:
            return ()
        if isinstance(payload, str):
            return (payload,)
        if isinstance(payload, Mapping):
            return tuple(str(value) for value in payload.values())
        try:
            return tuple(str(item) for item in payload)  # type: ignore[arg-type]
        except TypeError:
            return (str(payload),)

    def _extract_model_selection(
        self, evaluation: Any
    ) -> tuple[str | None, Sequence[Mapping[str, object]]] | None:
        selection: object | None
        if DecisionEvaluation is not None and isinstance(evaluation, DecisionEvaluation):
            selection = evaluation.model_selection
        elif isinstance(evaluation, Mapping):
            selection = evaluation.get("model_selection")  # type: ignore[index]
        else:
            selection = getattr(evaluation, "model_selection", None)

        if selection is None:
            return None

        if (
            ModelSelectionMetadata is not None
            and isinstance(selection, ModelSelectionMetadata)
        ):
            selected = selection.selected
            candidates = self._coerce_selection_candidates(selection.candidates)
        elif isinstance(selection, Mapping):
            selected = selection.get("selected")  # type: ignore[index]
            candidates = self._coerce_selection_candidates(selection.get("candidates"))  # type: ignore[index]
        else:
            selected = getattr(selection, "selected", None)
            candidates = self._coerce_selection_candidates(
                getattr(selection, "candidates", None)
            )

        normalized_selected = str(selected) if selected is not None else None
        return normalized_selected, candidates

    def _coerce_selection_candidates(
        self, payload: object
    ) -> Sequence[Mapping[str, object]]:
        if payload is None:
            return ()
        if isinstance(payload, Mapping):
            values = list(payload.values())
        else:
            try:
                values = list(payload)  # type: ignore[arg-type]
            except TypeError:
                values = [payload]

        candidates: list[Mapping[str, object]] = []
        for item in values:
            if item is None:
                continue
            if (
                ModelSelectionDetail is not None
                and isinstance(item, ModelSelectionDetail)
            ):
                candidates.append(
                    {
                        "name": item.name,
                        "score": item.score,
                        "weight": item.weight,
                        "effective_score": item.effective_score,
                        "available": item.available,
                        "reason": item.reason,
                    }
                )
                continue
            if isinstance(item, Mapping):
                candidates.append(dict(item))
                continue
            candidates.append(
                {
                    "name": getattr(item, "name", None),
                    "score": getattr(item, "score", None),
                    "weight": getattr(item, "weight", None),
                    "effective_score": getattr(item, "effective_score", None),
                    "available": getattr(item, "available", None),
                    "reason": getattr(item, "reason", None),
                }
            )
        return tuple(candidates)

    def profile_names(self) -> Sequence[str]:
        """Zwraca listę zarejestrowanych profili ryzyka."""

        return tuple(self._profiles.keys())

    @classmethod
    def run_simulations_from_parquet(
        cls,
        parquet_path: str | Path,
        *,
        profiles: Sequence[str] | None = None,
        manual_overrides: Mapping[str, object] | None = None,
    ) -> RiskSimulationSuite:
        """Uruchamia scenariusze symulacyjne dla wskazanych profili.

        Symulacja korzysta z danych w formacie Parquet, oczekując kolumn
        zgodnych z :class:`bot_core.risk.simulation.SimulationOrder`. Każdy
        scenariusz wykonywany jest na świeżej instancji silnika, aby uniknąć
        przecieków stanu pomiędzy profilami.
        """

        orders = load_orders_from_parquet(parquet_path)
        scenarios: list[str] = [profile.lower() for profile in (profiles or DEFAULT_PROFILES)]
        results = []
        for profile_name in scenarios:
            profile = build_profile(profile_name, manual_overrides=manual_overrides)
            engine = cls(repository=InMemoryRiskRepository())
            result = run_profile_scenario(engine, profile, orders)
            results.append(result)
        return RiskSimulationSuite(tuple(results))

    @classmethod
    def generate_simulation_reports(
        cls,
        parquet_path: str | Path,
        *,
        output_dir: str | Path,
        profiles: Sequence[str] | None = None,
        manual_overrides: Mapping[str, object] | None = None,
        json_name: str = "risk_simulation_report.json",
        pdf_name: str = "risk_simulation_report.pdf",
    ) -> Mapping[str, object]:
        """Generuje raporty JSON oraz PDF z przebiegu symulacji."""

        suite = cls.run_simulations_from_parquet(
            parquet_path,
            profiles=profiles,
            manual_overrides=manual_overrides,
        )
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        json_path = target_dir / json_name
        pdf_path = target_dir / pdf_name
        suite.write_json(json_path)
        suite.render_pdf(pdf_path)
        return {
            "summary": suite.to_mapping(),
            "json_path": str(json_path),
            "pdf_path": str(pdf_path),
        }


def _ensure_decision_models() -> bool:
    """Gwarantuje, że zależności modułu decision są załadowane."""

    global DecisionCandidate, DecisionEvaluation, RiskSnapshot

    if all(dependency is not None for dependency in (DecisionCandidate, DecisionEvaluation, RiskSnapshot)):
        return True

    try:  # pragma: no cover - fallback ładowania w środowiskach z modułem decision
        from bot_core.decision.models import (  # type: ignore import-not-found
            DecisionCandidate as _DecisionCandidate,
            DecisionEvaluation as _DecisionEvaluation,
            RiskSnapshot as _RiskSnapshot,
        )
    except Exception:  # pragma: no cover - środowiska bez modułu decision
        return False

    DecisionCandidate = _DecisionCandidate  # type: ignore[assignment]
    DecisionEvaluation = _DecisionEvaluation  # type: ignore[assignment]
    RiskSnapshot = _RiskSnapshot  # type: ignore[assignment]
    return True


__all__ = ["InMemoryRiskRepository", "ThresholdRiskEngine", "RiskState", "PositionState"]
