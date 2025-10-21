"""Harmonogram wielostrate-giczny obsługujący wiele silników strategii."""
from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Mapping, MutableMapping, Protocol, Sequence

if TYPE_CHECKING:
    from bot_core.runtime.portfolio_coordinator import PortfolioRuntimeCoordinator

from bot_core.runtime.journal import TradingDecisionEvent, TradingDecisionJournal
from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal

_LOGGER = logging.getLogger(__name__)


def _split_symbol_components(symbol: str | None) -> tuple[str | None, str | None]:
    if not symbol:
        return None, None
    upper_symbol = symbol.upper()
    known_quotes = (
        "USDT",
        "USDC",
        "USD",
        "EUR",
        "BTC",
        "ETH",
        "PLN",
        "GBP",
        "CHF",
    )
    for quote in known_quotes:
        if upper_symbol.endswith(quote) and len(upper_symbol) > len(quote):
            base = upper_symbol[: -len(quote)].rstrip("_-/")
            if base:
                return base, quote
    return None, None


class StrategyDataFeed(Protocol):
    """Źródło danych dla strategii."""

    def load_history(self, strategy_name: str, bars: int) -> Sequence[MarketSnapshot]:
        ...

    def fetch_latest(self, strategy_name: str) -> Sequence[MarketSnapshot]:
        ...


class StrategySignalSink(Protocol):
    """Interfejs odbiorcy sygnałów strategii."""

    def submit(
        self,
        *,
        strategy_name: str,
        schedule_name: str,
        risk_profile: str,
        timestamp: datetime,
        signals: Sequence[StrategySignal],
    ) -> None:
        ...


TelemetryEmitter = Callable[[str, Mapping[str, float]], None]

if TYPE_CHECKING:  # pragma: no cover - tylko dla typowania
    from bot_core.portfolio import PortfolioGovernor, PortfolioRebalanceDecision


@dataclass(slots=True)
class _ScheduleContext:
    name: str
    strategy_name: str
    strategy: StrategyEngine
    feed: StrategyDataFeed
    sink: StrategySignalSink
    cadence: float
    max_drift: float
    warmup_bars: int
    risk_profile: str
    base_max_signals: int
    active_max_signals: int
    portfolio_weight: float = 0.0
    allocator_weight: float = 1.0
    allocator_signal_factor: float = 1.0
    governor_signal_factor: float = 1.0
    last_run: datetime | None = None
    warmed_up: bool = False
    metrics: MutableMapping[str, float] = field(default_factory=dict)


class CapitalAllocationPolicy(Protocol):
    """Polityka wyznaczająca wagi kapitału pomiędzy strategie."""

    name: str

    def allocate(self, schedules: Sequence[_ScheduleContext]) -> Mapping[str, float]:
        ...


def _normalize_weights(candidates: Mapping[str, float]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for key, value in candidates.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric <= 0:
            continue
        weights[key] = numeric
    if not weights:
        return {}
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0:
        uniform = 1.0 / len(weights)
        return {key: uniform for key in weights}
    return {key: value / total for key, value in weights.items()}


class EqualWeightAllocation:
    """Prosta polityka przydzielająca identyczne wagi wszystkim strategiom."""

    name = "equal_weight"

    def allocate(self, schedules: Sequence[_ScheduleContext]) -> Mapping[str, float]:
        if not schedules:
            return {}
        weight = 1.0 / len(schedules)
        return {schedule.name: weight for schedule in schedules}


class RiskParityAllocation:
    """Polityka alokacji proporcjonalna do odwrotności zmienności."""

    name = "risk_parity"

    def allocate(self, schedules: Sequence[_ScheduleContext]) -> Mapping[str, float]:
        scores: dict[str, float] = {}
        for schedule in schedules:
            vol = schedule.metrics.get("avg_realized_volatility")
            if not isinstance(vol, (int, float)) or vol <= 0:
                vol = schedule.metrics.get("realized_volatility")
            if not isinstance(vol, (int, float)) or vol <= 0:
                vol = schedule.metrics.get("volatility")
            if not isinstance(vol, (int, float)) or vol <= 0:
                vol = 1.0
            scores[schedule.name] = 1.0 / max(float(vol), 1e-6)
        return _normalize_weights(scores)


class VolatilityTargetAllocation:
    """Polityka zwiększająca udział strategii trafiających w target zmienności."""

    name = "volatility_target"

    def allocate(self, schedules: Sequence[_ScheduleContext]) -> Mapping[str, float]:
        scores: dict[str, float] = {}
        for schedule in schedules:
            error = schedule.metrics.get("realized_vs_target_vol_pct")
            if not isinstance(error, (int, float)) or error < 0:
                error = schedule.metrics.get("allocation_error_pct")
            if not isinstance(error, (int, float)) or error < 0:
                error = 0.0
            weight = 1.0 / (1.0 + abs(float(error)))
            scores[schedule.name] = weight
        normalized = _normalize_weights(scores)
        if not normalized:
            return EqualWeightAllocation().allocate(schedules)
        return normalized


class SignalStrengthAllocation:
    """Preferuje strategie generujące częściej wysokiej jakości sygnały."""

    name = "signal_strength"

    def allocate(self, schedules: Sequence[_ScheduleContext]) -> Mapping[str, float]:
        scores: dict[str, float] = {}
        for schedule in schedules:
            signals = schedule.metrics.get("signals")
            confidence = schedule.metrics.get("avg_confidence")
            try:
                signal_rate = max(0.0, float(signals))
            except (TypeError, ValueError):
                signal_rate = 0.0
            try:
                confidence_score = float(confidence)
            except (TypeError, ValueError):
                confidence_score = 0.0
            if not math.isfinite(confidence_score):
                confidence_score = 0.0
            confidence_score = min(max(confidence_score, 0.0), 1.0)
            base_weight = 0.5 + confidence_score
            scores[schedule.name] = (1.0 + signal_rate) * base_weight
        normalized = _normalize_weights(scores)
        if not normalized:
            return EqualWeightAllocation().allocate(schedules)
        return normalized


class SmoothedCapitalAllocationPolicy:
    """Wygładza wagi zwracane przez wewnętrzną politykę alokacji."""

    name = "smoothed"

    def __init__(
        self,
        inner_policy: CapitalAllocationPolicy,
        *,
        smoothing_factor: float = 0.35,
        min_delta: float = 0.0,
        floor_weight: float = 0.0,
    ) -> None:
        self.inner_policy = inner_policy
        self.smoothing_factor = min(max(float(smoothing_factor), 0.0), 1.0)
        self.min_delta = max(0.0, float(min_delta))
        self.floor_weight = max(0.0, float(floor_weight))
        self._last_smoothed: dict[str, float] = {}
        self._last_raw: dict[str, float] = {}

    def allocate(self, schedules: Sequence[_ScheduleContext]) -> Mapping[str, float]:
        if not schedules:
            self._last_smoothed = {}
            self._last_raw = {}
            return {}

        try:
            raw_allocation = self.inner_policy.allocate(schedules)
        except Exception:  # pragma: no cover - diagnostyka polityk zewnętrznych
            _LOGGER.exception("Błąd wewnętrznej polityki alokacji kapitału")
            raw_allocation = {}

        raw_weights: dict[str, float] = {}
        smoothed: dict[str, float] = {}
        alpha = self.smoothing_factor

        for schedule in schedules:
            name = schedule.name
            try:
                raw_value = float(raw_allocation.get(name, 0.0))
            except (TypeError, ValueError):
                raw_value = 0.0
            if not math.isfinite(raw_value) or raw_value < 0.0:
                raw_value = 0.0
            raw_weights[name] = raw_value

            previous = self._last_smoothed.get(name)
            if previous is None:
                smoothed_value = raw_value
            else:
                delta = raw_value - previous
                if abs(delta) < self.min_delta:
                    smoothed_value = previous
                else:
                    smoothed_value = previous + alpha * delta
            smoothed[name] = max(self.floor_weight, smoothed_value)

        normalized = _normalize_weights(smoothed)
        if not normalized:
            normalized = EqualWeightAllocation().allocate(schedules)

        self._last_raw = raw_weights
        self._last_smoothed = dict(normalized)
        return normalized

    def raw_allocation_snapshot(self) -> Mapping[str, float]:
        """Zwraca ostatnie, niewygładzone wagi z wewnętrznej polityki."""

        return dict(self._last_raw)

    def smoothed_allocation_snapshot(self) -> Mapping[str, float]:
        """Zwraca ostatnie wygładzone wagi po normalizacji."""

        return dict(self._last_smoothed)


class DrawdownAdaptiveAllocation:
    """Dostosowuje wagi strategii do presji drawdownu i stresu płynności."""

    name = "drawdown_adaptive"

    def __init__(
        self,
        *,
        warning_drawdown_pct: float = 10.0,
        panic_drawdown_pct: float = 20.0,
        pressure_weight: float = 0.7,
        min_weight: float = 0.05,
        max_weight: float = 1.0,
    ) -> None:
        self.warning_drawdown_pct = max(0.0, float(warning_drawdown_pct))
        self.panic_drawdown_pct = max(0.0, float(panic_drawdown_pct))
        if self.panic_drawdown_pct <= self.warning_drawdown_pct:
            self.panic_drawdown_pct = self.warning_drawdown_pct + 1.0
        self.pressure_weight = min(max(float(pressure_weight), 0.0), 1.0)
        self.min_weight = max(0.0, float(min_weight))
        self.max_weight = max(self.min_weight, float(max_weight)) or 1.0
        self._last_snapshot: dict[str, dict[str, float]] = {}

    def allocate(self, schedules: Sequence[_ScheduleContext]) -> Mapping[str, float]:
        if not schedules:
            self._last_snapshot = {}
            return {}

        weights: dict[str, float] = {}
        snapshot: dict[str, dict[str, float]] = {}
        for schedule in schedules:
            drawdown_pct = self._extract_drawdown_pct(schedule)
            pressure = self._extract_drawdown_pressure(schedule)
            penalty = self._compute_penalty(drawdown_pct, pressure)
            candidate_weight = self.max_weight * (1.0 - penalty)
            weight = max(self.min_weight, candidate_weight)
            weights[schedule.name] = weight
            snapshot[schedule.name] = {
                "drawdown_pct": drawdown_pct,
                "drawdown_pressure": pressure,
                "penalty": penalty,
                "raw_weight": candidate_weight,
                "clamped_weight": weight,
            }

        normalized = _normalize_weights(weights)
        if not normalized:
            normalized = EqualWeightAllocation().allocate(schedules)
        self._last_snapshot = snapshot
        return normalized

    def allocation_diagnostics(self) -> Mapping[str, Mapping[str, float]]:
        """Zwraca ostatnie metryki drawdownu i zastosowane kary wagowe."""

        return {key: dict(value) for key, value in self._last_snapshot.items()}

    def _extract_drawdown_pct(self, schedule: _ScheduleContext) -> float:
        metric_keys = (
            "max_drawdown_pct",
            "drawdown_pct",
            "avg_drawdown_pct",
            "current_drawdown_pct",
            "max_drawdown_percent",
            "drawdown_percent",
            "drawdown",
            "avg_drawdown",
        )
        for key in metric_keys:
            value = schedule.metrics.get(key)
            if not isinstance(value, (int, float)):
                continue
            drawdown = abs(float(value))
            if key in {"drawdown", "avg_drawdown"} and drawdown <= 1.0:
                drawdown *= 100.0
            if math.isfinite(drawdown):
                return drawdown
        return 0.0

    def _extract_drawdown_pressure(self, schedule: _ScheduleContext) -> float:
        pressure_keys = ("drawdown_pressure", "drawdown_trend", "stress_drawdown_pressure")
        for key in pressure_keys:
            value = schedule.metrics.get(key)
            if not isinstance(value, (int, float)):
                continue
            pressure = max(0.0, float(value))
            if key == "drawdown_trend":
                pressure = max(0.0, min(1.0, pressure))
            if math.isfinite(pressure):
                return min(1.0, pressure)
        return 0.0

    def _compute_penalty(self, drawdown_pct: float, pressure: float) -> float:
        interval = max(self.panic_drawdown_pct - self.warning_drawdown_pct, 1e-6)
        drawdown_penalty = 0.0
        if drawdown_pct >= self.panic_drawdown_pct:
            drawdown_penalty = 1.0
        elif drawdown_pct > self.warning_drawdown_pct:
            drawdown_penalty = (drawdown_pct - self.warning_drawdown_pct) / interval
        pressure_penalty = min(1.0, max(0.0, pressure))
        combined = max(drawdown_penalty, pressure_penalty * self.pressure_weight)
        # jeżeli oba sygnały są wysokie, eskaluj karę szybciej
        if pressure_penalty > 0.0:
            combined = max(combined, min(1.0, drawdown_penalty + pressure_penalty * 0.5))
        return min(1.0, combined)


class FixedWeightAllocation:
    """Polityka stosująca ręcznie zdefiniowane wagi harmonogramów/strategii."""

    def __init__(self, weights: Mapping[str, float], *, label: str | None = None) -> None:
        normalized_source: dict[str, float] = {}
        for key, value in weights.items():
            if key is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            normalized_source[str(key).lower()] = numeric
        self._weights = _normalize_weights(normalized_source)
        self.name = str(label or "fixed_weight")

    def allocate(self, schedules: Sequence[_ScheduleContext]) -> Mapping[str, float]:
        if not schedules:
            return {}
        matched: dict[str, float] = {}
        for schedule in schedules:
            candidates = (
                f"{schedule.name}:{schedule.risk_profile}",
                f"{schedule.strategy_name}:{schedule.risk_profile}",
                schedule.name,
                schedule.strategy_name,
            )
            for candidate in candidates:
                lookup = candidate.lower()
                weight = self._weights.get(lookup)
                if weight is not None:
                    matched[schedule.name] = weight
                    break
        if not matched:
            return {}
        return _normalize_weights(matched)


class RiskProfileBudgetAllocation:
    """Rozdziela budżet kapitału pomiędzy profile ryzyka i strategie w ich obrębie."""

    def __init__(
        self,
        profile_weights: Mapping[str, float],
        *,
        label: str | None = None,
        profile_floor: float = 0.0,
        inner_policy_factory: Callable[[], CapitalAllocationPolicy] | None = None,
    ) -> None:
        normalized_profiles: dict[str, float] = {}
        for profile, weight in profile_weights.items():
            if profile in (None, ""):
                continue
            try:
                numeric = float(weight)
            except (TypeError, ValueError):
                continue
            if numeric <= 0:
                continue
            normalized_profiles[str(profile).lower()] = numeric
        self._profile_targets = _normalize_weights(normalized_profiles)
        self._profile_floor = max(0.0, float(profile_floor))
        if inner_policy_factory is None:
            inner_policy_factory = RiskParityAllocation
        self._inner_policy_factory = inner_policy_factory
        self._per_profile_allocators: dict[str, CapitalAllocationPolicy] = {}
        self.name = str(label or "risk_profile")
        self._last_profile_weights: dict[str, float] = {}
        self._last_floor_adjustment: bool = False

    def _profile_key(self, risk_profile: str | None) -> str:
        return str(risk_profile or "default").lower()

    def _allocator_for(self, profile_key: str) -> CapitalAllocationPolicy:
        allocator = self._per_profile_allocators.get(profile_key)
        if allocator is None:
            try:
                allocator = self._inner_policy_factory()
            except Exception:  # pragma: no cover - zabezpieczenie przed błędną fabryką
                _LOGGER.exception(
                    "Nie udało się zbudować wewnętrznej polityki alokacji dla profilu %s",
                    profile_key,
                )
                allocator = RiskParityAllocation()
            self._per_profile_allocators[profile_key] = allocator
        return allocator

    def _enforce_profile_floor(
        self,
        weights: Mapping[str, float],
        groups: Mapping[str, Sequence[_ScheduleContext]],
    ) -> dict[str, float]:
        floor = max(0.0, float(self._profile_floor))
        if floor <= 0 or not groups:
            return {key: max(float(weights.get(key, 0.0)), 0.0) for key in groups}

        keys = tuple(groups.keys())
        if not keys:
            return {}

        floor = min(floor, 1.0 / len(keys))
        if floor <= 0:
            return {key: max(float(weights.get(key, 0.0)), 0.0) for key in keys}

        normalized = {
            key: max(float(weights.get(key, 0.0)), 0.0)
            for key in keys
        }
        total = sum(normalized.values())
        if total <= 0:
            normalized = {key: 1.0 / len(keys) for key in keys}
        else:
            normalized = {
                key: value / total
                for key, value in normalized.items()
            }

        floor_total = floor * len(keys)
        if floor_total >= 1.0 - 1e-9:
            return {key: 1.0 / len(keys) for key in keys}

        remaining = 1.0 - floor_total
        residuals = {
            key: max(normalized[key] - floor, 0.0)
            for key in keys
        }
        residual_total = sum(residuals.values())
        if residual_total <= 0:
            uniform_bonus = remaining / len(keys)
            return {key: floor + uniform_bonus for key in keys}

        return {
            key: floor + remaining * (residuals[key] / residual_total)
            for key in keys
        }

    def allocate(self, schedules: Sequence[_ScheduleContext]) -> Mapping[str, float]:
        if not schedules:
            self._last_profile_weights = {}
            self._last_floor_adjustment = False
            return {}

        groups: dict[str, list[_ScheduleContext]] = {}
        for schedule in schedules:
            key = self._profile_key(schedule.risk_profile)
            groups.setdefault(key, []).append(schedule)

        configured_targets = {
            key: weight
            for key, weight in self._profile_targets.items()
            if key in groups
        }

        if configured_targets:
            remaining_profiles = [key for key in groups if key not in configured_targets]
            leftover = max(0.0, 1.0 - sum(configured_targets.values()))
            if remaining_profiles:
                if leftover > 0:
                    share = leftover / len(remaining_profiles)
                    for profile in remaining_profiles:
                        configured_targets[profile] = share
                else:
                    for profile in remaining_profiles:
                        configured_targets.setdefault(profile, 0.0)
        else:
            configured_targets = {key: 1.0 for key in groups}

        pre_floor_targets = dict(configured_targets)

        if self._profile_floor > 0:
            configured_targets = dict(configured_targets)

        normalized_profiles = _normalize_weights(configured_targets)
        if not normalized_profiles:
            normalized_profiles = {
                key: 1.0 / len(groups)
                for key in groups
            }
        adjusted_profiles = normalized_profiles
        floor_applied = False
        if self._profile_floor > 0:
            adjusted_profiles = self._enforce_profile_floor(normalized_profiles, groups)
            floor_applied = any(
                abs(adjusted_profiles.get(key, 0.0) - normalized_profiles.get(key, 0.0))
                > 1e-9
                for key in groups
            )
            if floor_applied:
                _LOGGER.debug(
                    "Risk profile floor %.4f adjusted targets from %s to %s",
                    self._profile_floor,
                    {k: round(pre_floor_targets.get(k, 0.0), 4) for k in groups},
                    {k: round(adjusted_profiles.get(k, 0.0), 4) for k in groups},
                )
        normalized_profiles = adjusted_profiles
        self._last_profile_weights = {
            key: float(value)
            for key, value in normalized_profiles.items()
        }
        self._last_floor_adjustment = floor_applied

        schedule_weights: dict[str, float] = {}
        for profile_key, schedule_group in groups.items():
            profile_weight = normalized_profiles.get(profile_key, 0.0)
            allocator = self._allocator_for(profile_key)
            try:
                inner_weights = allocator.allocate(schedule_group)
            except Exception:  # pragma: no cover - diagnostyka wewnętrznej polityki
                _LOGGER.exception(
                    "Błąd alokacji kapitału dla profilu ryzyka %s",
                    profile_key,
                )
                inner_weights = {}
            normalized_inner = _normalize_weights(inner_weights)
            if not normalized_inner and schedule_group:
                normalized_inner = {
                    schedule.name: 1.0 / len(schedule_group)
                    for schedule in schedule_group
                }
            for schedule in schedule_group:
                weight = normalized_inner.get(schedule.name)
                if weight is None:
                    weight = normalized_inner.get(schedule.strategy_name, 0.0)
                schedule_weights[schedule.name] = profile_weight * float(weight or 0.0)

        return _normalize_weights(schedule_weights)

    def profile_allocation_snapshot(self) -> Mapping[str, float]:
        """Zwraca ostatnio wyliczone udziały profili ryzyka."""

        return dict(self._last_profile_weights)

    @property
    def floor_adjustment_applied(self) -> bool:
        """Czy ostatnia alokacja wymagała korekty przez `profile_floor`."""

        return self._last_floor_adjustment


class MultiStrategyScheduler:
    """Koordynuje wykonywanie wielu strategii zgodnie z harmonogramem."""

    def __init__(
        self,
        *,
        environment: str,
        portfolio: str,
        clock: Callable[[], datetime] | None = None,
        telemetry_emitter: TelemetryEmitter | None = None,
        decision_journal: TradingDecisionJournal | None = None,
        portfolio_governor: "PortfolioGovernor | None" = None,
        capital_policy: CapitalAllocationPolicy | None = None,
        signal_limits: Mapping[str, Mapping[str, int]] | None = None,
        allocation_rebalance_seconds: float | None = 300.0,
    ) -> None:
        self._environment = environment
        self._portfolio = portfolio
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._telemetry = telemetry_emitter
        self._decision_journal = decision_journal
        self._portfolio_governor = portfolio_governor
        self._schedules: list[_ScheduleContext] = []
        self._stop_event: asyncio.Event | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._portfolio_coordinator: "PortfolioRuntimeCoordinator" | None = None
        self._portfolio_lock: asyncio.Lock | None = None
        self._last_portfolio_decision: "PortfolioRebalanceDecision | None" = None
        self._capital_policy = capital_policy or EqualWeightAllocation()
        self._allocation_lock = asyncio.Lock()
        self._allocation_rebalance_seconds = (
            float(allocation_rebalance_seconds) if allocation_rebalance_seconds else None
        )
        self._last_allocation_at: datetime | None = None
        self._signal_limits: dict[tuple[str, str], int] = {}
        for strategy, profiles in (signal_limits or {}).items():
            for profile, limit in (profiles or {}).items():
                try:
                    value = int(limit)
                except (TypeError, ValueError):
                    continue
                self._signal_limits[(strategy, profile)] = max(0, value)

    def set_capital_policy(self, policy: CapitalAllocationPolicy | None) -> None:
        """Ustawia politykę alokacji bez natychmiastowego przeliczenia."""

        self._capital_policy = policy or EqualWeightAllocation()
        self._last_allocation_at = None

    async def replace_capital_policy(
        self,
        policy: CapitalAllocationPolicy | None,
        *,
        rebalance: bool = True,
        timestamp: datetime | None = None,
    ) -> None:
        """Podmienia politykę kapitału i opcjonalnie uruchamia rebalance."""

        self.set_capital_policy(policy)
        if not rebalance:
            return
        await self._maybe_rebalance_allocation(timestamp or self._clock())

    def allocation_snapshot(self) -> Mapping[str, float]:
        """Zwraca ostatnio zastosowane wagi polityki kapitału."""

        return {
            schedule.name: float(schedule.allocator_weight)
            for schedule in self._schedules
        }

    def register_schedule(
        self,
        *,
        name: str,
        strategy_name: str,
        strategy: StrategyEngine,
        feed: StrategyDataFeed,
        sink: StrategySignalSink,
        cadence_seconds: int,
        max_drift_seconds: int,
        warmup_bars: int,
        risk_profile: str,
        max_signals: int,
    ) -> None:
        context = _ScheduleContext(
            name=name,
            strategy_name=strategy_name,
            strategy=strategy,
            feed=feed,
            sink=sink,
            cadence=float(cadence_seconds),
            max_drift=float(max(0, max_drift_seconds)),
            warmup_bars=max(0, warmup_bars),
            risk_profile=risk_profile,
            base_max_signals=max(1, max_signals),
            active_max_signals=max(1, max_signals),
        )
        self._schedules.append(context)
        _LOGGER.debug("Zarejestrowano harmonogram %s dla strategii %s", name, strategy_name)

    def configure_signal_limit(
        self, strategy_name: str, risk_profile: str, limit: int | None
    ) -> None:
        key = (strategy_name, risk_profile)
        if limit is None:
            self._signal_limits.pop(key, None)
            return
        try:
            value = int(limit)
        except (TypeError, ValueError):
            return
        self._signal_limits[key] = max(0, value)

    def configure_signal_limits(
        self, limits: Mapping[str, Mapping[str, int]]
    ) -> None:
        for strategy, profiles in limits.items():
            for profile, limit in profiles.items():
                self.configure_signal_limit(strategy, profile, limit)

    def attach_portfolio_coordinator(
        self, coordinator: "PortfolioRuntimeCoordinator"
    ) -> None:
        """Podpina koordynatora PortfolioGovernora do schedulera."""

        self._portfolio_coordinator = coordinator
        self._portfolio_lock = asyncio.Lock()

    async def run_forever(self) -> None:
        if self._tasks:
            raise RuntimeError("Scheduler został już uruchomiony")

        self._stop_event = asyncio.Event()
        self._tasks = [
            asyncio.create_task(self._run_schedule(schedule), name=f"strategy:{schedule.name}")
            for schedule in self._schedules
        ]
        if self._portfolio_coordinator is not None:
            self._tasks.append(
                asyncio.create_task(self._run_portfolio_loop(), name="portfolio:governor")
            )

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            if self._stop_event and not self._stop_event.is_set():
                self._stop_event.set()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            raise
        finally:
            self._tasks.clear()
            self._stop_event = None

    def stop(self) -> None:
        if self._stop_event and not self._stop_event.is_set():
            self._stop_event.set()

    async def run_once(self) -> None:
        """Wykonuje pojedynczy cykl wszystkich zarejestrowanych harmonogramów."""

        timestamp = self._clock()
        for schedule in self._schedules:
            await self._execute_schedule(schedule, timestamp)
        await self._evaluate_portfolio(force=True)

    async def _run_schedule(self, schedule: _ScheduleContext) -> None:
        assert self._stop_event is not None, "Scheduler musi zostać zainicjalizowany"
        cadence = max(1.0, schedule.cadence)
        while not self._stop_event.is_set():
            start_time = self._clock()
            await self._execute_schedule(schedule, start_time)
            elapsed = (self._clock() - start_time).total_seconds()
            schedule.last_run = start_time
            sleep_for = max(0.0, cadence - elapsed)
            if sleep_for < cadence - schedule.max_drift:
                _LOGGER.warning(
                    "Harmonogram %s wykonał się z dryfem (elapsed=%.2fs, cadence=%.2fs)",
                    schedule.name,
                    elapsed,
                    cadence,
                )
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_for)
            except asyncio.TimeoutError:
                continue

    async def _run_portfolio_loop(self) -> None:
        assert self._stop_event is not None, "Scheduler musi zostać zainicjalizowany"
        if self._portfolio_coordinator is None:
            return
        while not self._stop_event.is_set():
            await self._evaluate_portfolio(force=False)
            cooldown = max(1.0, float(self._portfolio_coordinator.cooldown_seconds))
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=cooldown)
            except asyncio.TimeoutError:
                continue

    async def _evaluate_portfolio(self, *, force: bool) -> None:
        coordinator = self._portfolio_coordinator
        lock = self._portfolio_lock
        if coordinator is None or lock is None:
            return
        async with lock:
            try:
                coordinator.evaluate(force=force)
            except Exception:  # pragma: no cover - diagnostyka runtime
                _LOGGER.exception("PortfolioGovernor: błąd ewaluacji w schedulerze")

    async def _execute_schedule(self, schedule: _ScheduleContext, timestamp: datetime) -> None:
        try:
            await self._maybe_rebalance_allocation(timestamp)
            if not schedule.warmed_up and schedule.warmup_bars > 0:
                history = schedule.feed.load_history(schedule.strategy_name, schedule.warmup_bars)
                if history:
                    schedule.strategy.warm_up(history)
                schedule.warmed_up = True

            schedule.metrics.clear()
            schedule.metrics["base_max_signals"] = float(schedule.base_max_signals)
            schedule.metrics["portfolio_weight_target"] = 0.0
            schedule.metrics["portfolio_signal_factor"] = 1.0
            schedule.metrics["allocator_signal_factor"] = float(schedule.allocator_signal_factor)
            schedule.metrics["allocator_weight_target"] = float(schedule.allocator_weight)
            schedule.metrics["governor_signal_factor"] = float(schedule.governor_signal_factor)
            if self._portfolio_governor is not None:
                allocation = self._portfolio_governor.resolve_allocation(
                    schedule.strategy_name,
                    schedule.risk_profile,
                )
                if allocation.max_signal_hint is not None:
                    schedule.base_max_signals = max(1, allocation.max_signal_hint)
                factor = float(allocation.signal_factor)
                schedule.governor_signal_factor = max(0.0, factor)
                schedule.portfolio_weight = float(allocation.weight)
                schedule.metrics["portfolio_weight_target"] = float(allocation.weight)
                schedule.metrics["portfolio_signal_factor"] = factor
            else:
                schedule.governor_signal_factor = 1.0
                schedule.portfolio_weight = 0.0

            schedule.metrics["portfolio_weight"] = float(schedule.portfolio_weight)
            self._apply_signal_limits(schedule)

            snapshots = schedule.feed.fetch_latest(schedule.strategy_name)
            total_signals = 0
            confidence_sum = 0.0
            confidence_count = 0
            mean_reversion_zscores: list[float] = []
            mean_reversion_vols: list[float] = []
            volatility_alloc_errors: list[float] = []
            volatility_target_errors: list[float] = []
            arbitrage_captures: list[float] = []
            arbitrage_delays: list[float] = []
            for snapshot in snapshots:
                signals = list(schedule.strategy.on_data(snapshot))
                if not signals:
                    continue
                bounded_signals = self._bounded_signals(
                    signals, schedule.active_max_signals
                )
                total_signals += len(bounded_signals)
                for signal in bounded_signals:
                    confidence_sum += float(signal.confidence)
                    confidence_count += 1
                    metadata = signal.metadata
                    if schedule.strategy_name.startswith("mean_reversion"):
                        zscore = metadata.get("zscore")
                        if isinstance(zscore, (int, float)):
                            mean_reversion_zscores.append(abs(float(zscore)))
                        volatility = metadata.get("volatility")
                        if isinstance(volatility, (int, float)):
                            mean_reversion_vols.append(float(volatility))
                    if "target_allocation" in metadata and "current_allocation" in metadata:
                        target_alloc = metadata.get("target_allocation")
                        current_alloc = metadata.get("current_allocation")
                        if isinstance(target_alloc, (int, float)) and isinstance(
                            current_alloc, (int, float)
                        ) and target_alloc:
                            diff_pct = abs(float(target_alloc) - float(current_alloc)) / max(
                                abs(float(target_alloc)), 1e-9
                            )
                            volatility_alloc_errors.append(diff_pct * 100.0)
                    realized_vol = metadata.get("realized_volatility")
                    target_vol = metadata.get("target_volatility")
                    if isinstance(realized_vol, (int, float)) and isinstance(target_vol, (int, float)) and target_vol:
                        variance_pct = abs(float(realized_vol) - float(target_vol)) / max(
                            abs(float(target_vol)), 1e-9
                        )
                        volatility_target_errors.append(variance_pct * 100.0)
                    secondary_delay = metadata.get("secondary_delay_ms")
                    if isinstance(secondary_delay, (int, float)):
                        arbitrage_delays.append(float(secondary_delay))
                    entry_spread = metadata.get("entry_spread")
                    exit_spread = metadata.get("exit_spread")
                    if isinstance(entry_spread, (int, float)) and isinstance(exit_spread, (int, float)) and entry_spread:
                        capture = (float(entry_spread) - float(exit_spread)) / abs(float(entry_spread))
                        arbitrage_captures.append(capture * 10_000.0)
                self._record_decisions(schedule, bounded_signals, timestamp, snapshot.symbol)
                schedule.sink.submit(
                    strategy_name=schedule.strategy_name,
                    schedule_name=schedule.name,
                    risk_profile=schedule.risk_profile,
                    timestamp=timestamp,
                    signals=bounded_signals,
                )
            schedule.metrics["signals"] = float(total_signals)
            schedule.metrics["active_max_signals"] = float(schedule.active_max_signals)
            schedule.metrics["last_latency_ms"] = max(
                0.0, (self._clock() - timestamp).total_seconds() * 1000
            )
            if confidence_count:
                schedule.metrics["avg_confidence"] = confidence_sum / confidence_count
            if mean_reversion_zscores:
                schedule.metrics["avg_abs_zscore"] = sum(mean_reversion_zscores) / len(
                    mean_reversion_zscores
                )
            if mean_reversion_vols:
                schedule.metrics["avg_realized_volatility"] = sum(mean_reversion_vols) / len(
                    mean_reversion_vols
                )
            if volatility_alloc_errors:
                schedule.metrics["allocation_error_pct"] = sum(volatility_alloc_errors) / len(
                    volatility_alloc_errors
                )
            if volatility_target_errors:
                schedule.metrics["realized_vs_target_vol_pct"] = sum(
                    volatility_target_errors
                ) / len(volatility_target_errors)
            if arbitrage_delays:
                schedule.metrics["secondary_delay_ms"] = max(arbitrage_delays)
            if arbitrage_captures:
                schedule.metrics["spread_capture_bps"] = sum(arbitrage_captures) / len(
                    arbitrage_captures
                )
            self._emit_metrics(schedule)

            if self._portfolio_governor is not None:
                observation_payload = dict(schedule.metrics)
                observation_payload["signals"] = float(total_signals)
                self._portfolio_governor.observe_strategy_metrics(
                    schedule.strategy_name,
                    observation_payload,
                    timestamp=timestamp,
                    risk_profile=schedule.risk_profile,
                )
                decision = self._portfolio_governor.maybe_rebalance(timestamp=timestamp)
                if decision is not None:
                    self._apply_portfolio_decision(decision)
        except Exception:  # pragma: no cover - chronimy scheduler przed przerwaniem
            _LOGGER.exception("Błąd podczas wykonywania harmonogramu %s", schedule.name)

    def _bounded_signals(
        self, signals: Sequence[StrategySignal], max_signals: int
    ) -> Sequence[StrategySignal]:
        limit = max(0, max_signals)
        if limit == 0:
            return ()
        if len(signals) <= limit:
            return signals
        ordered = sorted(signals, key=lambda signal: signal.confidence, reverse=True)
        return tuple(ordered[:limit])

    def _min_signal_floor(self) -> int:
        if self._portfolio_governor is None:
            return 1
        floor = getattr(self._portfolio_governor, "min_signal_floor", 1)
        try:
            return max(0, int(floor))
        except Exception:  # pragma: no cover - defensywnie
            return 1

    def _apply_portfolio_decision(
        self, decision: "PortfolioRebalanceDecision"
    ) -> None:
        self._last_portfolio_decision = decision
        if self._portfolio_governor is None:
            return
        for schedule in self._schedules:
            allocation = self._portfolio_governor.resolve_allocation(
                schedule.strategy_name,
                schedule.risk_profile,
            )
            if allocation.max_signal_hint is not None:
                schedule.base_max_signals = max(1, allocation.max_signal_hint)
            schedule.governor_signal_factor = max(0.0, float(allocation.signal_factor))
            schedule.portfolio_weight = float(allocation.weight)
            schedule.metrics["portfolio_weight"] = float(allocation.weight)
            schedule.metrics["portfolio_signal_factor"] = float(allocation.signal_factor)
            self._apply_signal_limits(schedule)
        allocation_map = {
            sched.name: sched.portfolio_weight for sched in self._schedules
        }
        _LOGGER.info(
            "PortfolioGovernor decision applied: weights=%s signals=%s",
            {name: round(weight, 4) for name, weight in allocation_map.items()},
            {
                sched.name: sched.active_max_signals
                for sched in self._schedules
            },
        )

    async def _maybe_rebalance_allocation(self, timestamp: datetime) -> None:
        policy = getattr(self, "_capital_policy", None)
        if policy is None:
            return
        if self._allocation_rebalance_seconds is not None and self._last_allocation_at is not None:
            delta = (timestamp - self._last_allocation_at).total_seconds()
            if delta < self._allocation_rebalance_seconds:
                return
        async with self._allocation_lock:
            schedules_snapshot = tuple(self._schedules)
            weights = policy.allocate(schedules_snapshot)
            normalized = _normalize_weights(weights)
            if not normalized and schedules_snapshot:
                normalized = {
                    schedule.name: 1.0 / len(schedules_snapshot)
                    for schedule in schedules_snapshot
                }
            self._last_allocation_at = timestamp
        if not normalized:
            return
        log_payload = {name: round(weight, 4) for name, weight in normalized.items()}
        _LOGGER.info("Capital allocator %s weights: %s", policy.name, log_payload)

        schedule_weights: dict[str, float] = {}
        profile_totals: defaultdict[str, float] = defaultdict(float)
        for schedule in self._schedules:
            weight = normalized.get(schedule.name)
            if weight is None:
                weight = normalized.get(schedule.strategy_name, 0.0)
            numeric_weight = float(weight or 0.0)
            schedule_weights[schedule.name] = numeric_weight
            profile_totals[schedule.risk_profile] += numeric_weight

        profile_snapshot: Mapping[str, float] | None = None
        snapshot_getter = getattr(policy, "profile_allocation_snapshot", None)
        if callable(snapshot_getter):
            try:
                snapshot = snapshot_getter()
            except Exception:  # pragma: no cover - diagnostyka polityki
                _LOGGER.exception(
                    "Nie udało się pobrać rozkładu profili z polityki %s",
                    policy.name,
                )
            else:
                if snapshot:
                    profile_snapshot = {
                        str(key): float(value)
                        for key, value in snapshot.items()
                        if isinstance(value, (int, float)) and math.isfinite(float(value))
                    }

        if profile_snapshot is None:
            profile_snapshot = dict(profile_totals)

        if profile_snapshot:
            normalized_profile_snapshot = _normalize_weights(profile_snapshot)
            log_profiles = {
                key: round(value, 4)
                for key, value in normalized_profile_snapshot.items()
            }
            _LOGGER.info(
                "Capital allocator %s profile weights: %s",
                policy.name,
                log_profiles,
            )
        else:
            normalized_profile_snapshot = {}

        for schedule in self._schedules:
            numeric_weight = schedule_weights.get(schedule.name, 0.0)
            schedule.allocator_weight = numeric_weight
            schedule.allocator_signal_factor = max(0.0, numeric_weight)
            schedule.metrics["allocator_weight"] = schedule.allocator_weight
            schedule.metrics["allocator_signal_factor"] = schedule.allocator_signal_factor
            schedule.metrics["allocator_profile_weight"] = normalized_profile_snapshot.get(
                schedule.risk_profile,
                0.0,
            )

    def _apply_signal_limits(self, schedule: _ScheduleContext) -> None:
        floor = self._min_signal_floor()
        factor = max(0.0, schedule.governor_signal_factor) * max(
            0.0, schedule.allocator_signal_factor
        )
        computed = int(round(schedule.base_max_signals * factor))
        if schedule.base_max_signals > 0:
            computed = max(floor if factor > 0 else 0, computed)
        override = self._signal_limits.get((schedule.strategy_name, schedule.risk_profile))
        if override is not None:
            if computed > override:
                _LOGGER.debug(
                    "Sygnal limit override dla %s/%s: %s -> %s",
                    schedule.strategy_name,
                    schedule.risk_profile,
                    computed,
                    override,
                )
            computed = min(computed, override)
        schedule.active_max_signals = max(0, computed)
        schedule.metrics["base_max_signals"] = float(schedule.base_max_signals)
        schedule.metrics["active_max_signals"] = float(schedule.active_max_signals)

    def _emit_metrics(self, schedule: _ScheduleContext) -> None:
        if not self._telemetry:
            return
        payload = {
            "signals": schedule.metrics.get("signals", 0.0),
            "latency_ms": schedule.metrics.get("last_latency_ms", 0.0),
        }
        for key, value in schedule.metrics.items():
            if key in {"signals", "last_latency_ms"}:
                continue
            if isinstance(value, (int, float)):
                payload[key] = float(value)
        self._telemetry(schedule.name, payload)

    def _record_decisions(
        self,
        schedule: _ScheduleContext,
        signals: Sequence[StrategySignal],
        timestamp: datetime,
        symbol: str,
    ) -> None:
        if not self._decision_journal:
            return
        for signal in signals:
            metadata_payload = {str(k): str(v) for k, v in signal.metadata.items()}
            schedule_run_id = metadata_payload.get(
                "schedule_run_id",
                f"{schedule.name}:{timestamp.isoformat()}",
            )
            strategy_instance_id = metadata_payload.get(
                "strategy_instance_id",
                schedule.strategy_name,
            )
            signal_identifier = metadata_payload.get(
                "signal_id",
                f"{schedule.name}:{symbol}:{timestamp.isoformat()}",
            )
            primary_exchange = metadata_payload.get("primary_exchange")
            secondary_exchange = metadata_payload.get("secondary_exchange")
            base_asset, quote_asset = _split_symbol_components(symbol)
            instrument_type = metadata_payload.get("instrument_type")
            data_feed = metadata_payload.get(
                "data_feed",
                getattr(schedule.feed, "name", schedule.feed.__class__.__name__),
            )
            risk_bucket = metadata_payload.get(
                "risk_budget_bucket",
                schedule.risk_profile,
            )
            event = TradingDecisionEvent(
                event_type="strategy_signal",
                timestamp=timestamp,
                environment=self._environment,
                portfolio=self._portfolio,
                risk_profile=schedule.risk_profile,
                symbol=symbol,
                side=signal.side,
                schedule=schedule.name,
                strategy=schedule.strategy_name,
                schedule_run_id=schedule_run_id,
                strategy_instance_id=strategy_instance_id,
                signal_id=signal_identifier,
                primary_exchange=primary_exchange,
                secondary_exchange=secondary_exchange,
                base_asset=base_asset,
                quote_asset=quote_asset,
                instrument_type=instrument_type,
                data_feed=data_feed,
                risk_budget_bucket=risk_bucket,
                confidence=float(signal.confidence),
                latency_ms=schedule.metrics.get("last_latency_ms"),
                telemetry_namespace=f"{self._environment}.multi_strategy.{schedule.name}",
                metadata=metadata_payload,
            )
            self._decision_journal.record(event)


__all__ = [
    "StrategyDataFeed",
    "StrategySignalSink",
    "TelemetryEmitter",
    "MultiStrategyScheduler",
    "CapitalAllocationPolicy",
    "EqualWeightAllocation",
    "RiskParityAllocation",
    "VolatilityTargetAllocation",
    "SignalStrengthAllocation",
    "SmoothedCapitalAllocationPolicy",
    "DrawdownAdaptiveAllocation",
    "FixedWeightAllocation",
    "RiskProfileBudgetAllocation",
]
