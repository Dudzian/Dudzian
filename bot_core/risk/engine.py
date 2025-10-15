"""Implementacja silnika ryzyka egzekwującego limity profili."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence

from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.base import RiskCheckResult, RiskEngine, RiskProfile, RiskRepository
from bot_core.risk.events import RiskDecisionLog

try:  # pragma: no cover - moduły decision/tco mogą być opcjonalne w innych gałęziach
    from bot_core.decision import DecisionOrchestrator
    from bot_core.decision.models import DecisionCandidate, DecisionEvaluation, RiskSnapshot
except Exception:  # pragma: no cover - decyzje mogą nie być dostępne
    DecisionOrchestrator = None  # type: ignore
    DecisionCandidate = None  # type: ignore
    DecisionEvaluation = None  # type: ignore
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
            evaluation, evaluation_payload = self._maybe_run_decision_orchestrator(
                request=request,
                profile_name=profile_name,
                account=account,
                state=state,
            )
            if evaluation_payload:
                evaluation_metadata = {"decision_orchestrator": evaluation_payload}
            if evaluation is None:
                if evaluation_payload and evaluation_payload.get("status") == "error":
                    reason = self._format_decision_error(evaluation_payload)
                    return deny(reason, metadata=evaluation_metadata)
            elif not evaluation.accepted:
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

    def _maybe_run_decision_orchestrator(
        self,
        *,
        request: OrderRequest,
        profile_name: str,
        account: AccountSnapshot,
        state: RiskState,
    ) -> tuple[Any | None, Mapping[str, object] | None]:
        if self._decision_orchestrator is None or DecisionCandidate is None:
            return None, None

        candidate_payload, error_payload = self._extract_candidate_payload(
            request.metadata
        )
        if error_payload:
            return None, error_payload
        if candidate_payload is None:
            return None, None

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
            return None, {"status": "error", "error": "invalid_candidate"}

        snapshot = self._build_decision_snapshot(profile_name, state, account)
        try:
            evaluation = self._decision_orchestrator.evaluate_candidate(  # type: ignore[union-attr]
                candidate,
                snapshot,
            )
        except Exception:
            _LOGGER.exception("DecisionOrchestrator: błąd ewaluacji")
            return None, {"status": "error", "error": "evaluation_failed"}

        return evaluation, self._serialize_decision_evaluation(evaluation)

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

    def profile_names(self) -> Sequence[str]:
        """Zwraca listę zarejestrowanych profili ryzyka."""

        return tuple(self._profiles.keys())


__all__ = ["InMemoryRiskRepository", "ThresholdRiskEngine", "RiskState", "PositionState"]
