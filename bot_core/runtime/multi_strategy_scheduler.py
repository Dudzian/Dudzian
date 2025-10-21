"""Harmonogram wielostrate-giczny obsługujący wiele silników strategii."""
from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import (
    TYPE_CHECKING,
    Callable,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
)

from threading import RLock

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
    tags: tuple[str, ...] = ()
    primary_tag: str | None = None
    last_run: datetime | None = None
    warmed_up: bool = False
    metrics: MutableMapping[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class SuspensionRecord:
    reason: str
    applied_at: datetime
    until: datetime | None = None
    origin: str = "schedule"
    tag: str | None = None

    def is_active(self, now: datetime) -> bool:
        if self.until is None:
            return True
        return now < self.until

    def remaining_seconds(self, now: datetime) -> float | None:
        if self.until is None:
            return None
        return max(0.0, (self.until - now).total_seconds())

    def clone_for_tag(self, tag: str) -> "SuspensionRecord":
        return SuspensionRecord(
            reason=self.reason,
            applied_at=self.applied_at,
            until=self.until,
            origin="tag",
            tag=tag,
        )

    def as_dict(self, now: datetime) -> dict[str, object]:
        payload: dict[str, object] = {
            "reason": self.reason,
            "applied_at": self.applied_at.isoformat(),
            "origin": self.origin,
        }
        if self.until is not None:
            payload["until"] = self.until.isoformat()
            payload["remaining_seconds"] = self.remaining_seconds(now)
        if self.tag:
            payload["tag"] = self.tag
        return payload


@dataclass(slots=True)
class SignalLimitOverride:
    """Reprezentuje czasowe nadpisanie limitu sygnałów."""

    limit: int
    reason: str | None = None
    expires_at: datetime | None = None
    created_at: datetime | None = None

    def is_expired(self, now: datetime) -> bool:
        if self.expires_at is None:
            return False
        return now >= self.expires_at

    def remaining_seconds(self, now: datetime) -> float | None:
        if self.expires_at is None:
            return None
        return max(0.0, (self.expires_at - now).total_seconds())

    def to_snapshot(self, now: datetime) -> Mapping[str, object]:
        payload: dict[str, object] = {"limit": int(self.limit)}
        if self.reason:
            payload["reason"] = self.reason
        if self.created_at:
            payload["created_at"] = self.created_at.isoformat()
        if self.expires_at:
            payload["expires_at"] = self.expires_at.isoformat()
            remaining = self.remaining_seconds(now)
            if remaining is not None:
                payload["remaining_seconds"] = remaining
        payload["active"] = not self.is_expired(now)
        return payload


class CapitalAllocationPolicy(Protocol):
    """Polityka wyznaczająca wagi kapitału pomiędzy strategie."""

    name: str

    def allocate(self, schedules: Sequence[_ScheduleContext]) -> Mapping[str, float]:
        ...


@dataclass(slots=True)
class MetricWeightRule:
    """Definicja metryki wykorzystywanej w polityce wag telemetrycznych."""

    metric: str
    weight: float
    default: float = 0.0
    clamp_min: float | None = None
    clamp_max: float | None = None
    absolute: bool = False
    scale: float = 1.0


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


def _extract_tags(metadata: Mapping[str, object] | None) -> tuple[tuple[str, ...], str | None]:
    if not isinstance(metadata, Mapping):
        return (), None

    tags_source = metadata.get("tags")
    tags: list[str] = []
    if isinstance(tags_source, (list, tuple, set)):
        for value in tags_source:
            if not value:
                continue
            tag = str(value).strip()
            if tag:
                tags.append(tag)
    elif isinstance(tags_source, str):
        tag = tags_source.strip()
        if tag:
            tags.append(tag)

    primary_source = (
        metadata.get("primary_tag")
        or metadata.get("primary_category")
        or metadata.get("primary_group")
    )
    primary_tag: str | None = None
    if isinstance(primary_source, str) and primary_source.strip():
        primary_tag = primary_source.strip()
    elif tags:
        primary_tag = tags[0]

    return tuple(dict.fromkeys(tags)), primary_tag


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


class MetricWeightedAllocation:
    """Buduje wagi na podstawie ważonego zestawu metryk telemetrycznych."""

    name = "metric_weighted"

    def __init__(
        self,
        metrics: Sequence[MetricWeightRule],
        *,
        label: str | None = None,
        default_score: float = 0.0,
        fallback_policy: CapitalAllocationPolicy | None = None,
        shift_epsilon: float = 1e-6,
    ) -> None:
        cleaned: list[MetricWeightRule] = []
        for rule in metrics:
            if not isinstance(rule, MetricWeightRule):
                continue
            if not rule.metric:
                continue
            if not math.isfinite(float(rule.weight)):
                continue
            cleaned.append(
                MetricWeightRule(
                    metric=str(rule.metric),
                    weight=float(rule.weight),
                    default=float(rule.default),
                    clamp_min=float(rule.clamp_min) if rule.clamp_min is not None else None,
                    clamp_max=float(rule.clamp_max) if rule.clamp_max is not None else None,
                    absolute=bool(rule.absolute),
                    scale=float(rule.scale),
                )
            )
        self._metrics: tuple[MetricWeightRule, ...] = tuple(cleaned)
        self._fallback = fallback_policy
        self._default_score = float(default_score)
        self._shift_epsilon = max(1e-12, abs(float(shift_epsilon)))
        self.name = str(label or "metric_weighted")
        self._last_snapshot: dict[str, dict[str, float]] = {}

    @property
    def metrics(self) -> tuple[MetricWeightRule, ...]:
        return self._metrics

    def allocate(self, schedules: Sequence[_ScheduleContext]) -> Mapping[str, float]:
        if not schedules:
            self._last_snapshot = {}
            return {}

        if not self._metrics:
            return self._fallback_or_equal(schedules, reason="no_metrics")

        scores: dict[str, float] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        shift = 0.0
        for schedule in schedules:
            raw_score = float(self._default_score)
            details: dict[str, float] = {"bias": raw_score}
            for rule in self._metrics:
                value = schedule.metrics.get(rule.metric, rule.default)
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = float(rule.default)
                if not math.isfinite(numeric):
                    numeric = float(rule.default)
                if rule.absolute:
                    numeric = abs(numeric)
                if rule.clamp_min is not None:
                    numeric = max(rule.clamp_min, numeric)
                if rule.clamp_max is not None:
                    numeric = min(rule.clamp_max, numeric)
                contribution_value = numeric * rule.scale
                contribution = contribution_value * rule.weight
                raw_score += contribution
                details[f"metric:{rule.metric}"] = contribution_value
                details[f"contribution:{rule.metric}"] = contribution
            details["raw_score"] = raw_score
            scores[schedule.name] = raw_score
            diagnostics[schedule.name] = details

        if scores:
            min_score = min(scores.values())
            if min_score <= 0.0 or not math.isfinite(min_score):
                shift = abs(min_score) + self._shift_epsilon if math.isfinite(min_score) else self._shift_epsilon
        shifted: dict[str, float] = {}
        for schedule_name, raw_score in scores.items():
            shifted_score = raw_score + shift
            if shifted_score <= 0.0 or not math.isfinite(shifted_score):
                shifted_score = 0.0
            diagnostics[schedule_name]["shifted_score"] = shifted_score
            diagnostics[schedule_name]["shift"] = shift
            shifted[schedule_name] = shifted_score

        normalized = _normalize_weights(shifted)
        if not normalized:
            return self._fallback_or_equal(schedules, diagnostics=diagnostics, reason="normalize_failed")

        self._last_snapshot = {key: dict(value) for key, value in diagnostics.items()}
        return normalized

    def allocation_diagnostics(self) -> Mapping[str, Mapping[str, float]]:
        return {key: dict(value) for key, value in self._last_snapshot.items()}

    def _fallback_or_equal(
        self,
        schedules: Sequence[_ScheduleContext],
        *,
        diagnostics: Mapping[str, Mapping[str, float]] | None = None,
        reason: str,
    ) -> Mapping[str, float]:
        fallback_weights: Mapping[str, float] = {}
        if self._fallback is not None:
            try:
                fallback_weights = self._fallback.allocate(schedules)
            except Exception:  # pragma: no cover - defensywnie logujemy fallback
                _LOGGER.exception(
                    "Błąd fallbackowej polityki kapitału %s (%s)",
                    getattr(self._fallback, "name", self._fallback),
                    reason,
                )
                fallback_weights = {}
        if not fallback_weights and schedules:
            fallback_weights = EqualWeightAllocation().allocate(schedules)
        snapshot: dict[str, dict[str, float]] = {}
        source = diagnostics if diagnostics else {}
        for schedule in schedules:
            payload = dict(source.get(schedule.name, {}))
            payload["fallback"] = 1.0
            snapshot[schedule.name] = payload
        self._last_snapshot = snapshot
        return dict(fallback_weights)

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

        normalized_raw = _normalize_weights(raw_allocation)
        if not normalized_raw and schedules:
            normalized_raw = {
                schedule.name: 1.0 / len(schedules)
                for schedule in schedules
            }
        raw_weights: dict[str, float] = {}
        smoothed: dict[str, float] = {}
        alpha = self.smoothing_factor

        for schedule in schedules:
            name = schedule.name
            raw_value = normalized_raw.get(name)
            if raw_value is None:
                raw_value = normalized_raw.get(schedule.strategy_name, 0.0)
            try:
                numeric_raw = float(raw_value or 0.0)
            except (TypeError, ValueError):
                numeric_raw = 0.0
            if not math.isfinite(numeric_raw) or numeric_raw < 0.0:
                numeric_raw = 0.0
            raw_weights[name] = numeric_raw

            previous = self._last_smoothed.get(name)
            if previous is None:
                smoothed_value = numeric_raw
            else:
                delta = numeric_raw - previous
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

        self._last_raw = _normalize_weights(raw_weights)
        self._last_raw = raw_weights
        self._last_smoothed = dict(normalized)
        return normalized

    def raw_allocation_snapshot(self) -> Mapping[str, float]:
        """Zwraca ostatnie, niewygładzone wagi z wewnętrznej polityki."""

        return dict(self._last_raw)

    def smoothed_allocation_snapshot(self) -> Mapping[str, float]:
        """Zwraca ostatnie wygładzone wagi po normalizacji."""

        return dict(self._last_smoothed)


class BlendedCapitalAllocation:
    """Łączy kilka polityk kapitałowych w jeden miks wagowy."""

    name = "blended"

    def __init__(
        self,
        components: Sequence[tuple[CapitalAllocationPolicy, float, str | None]],
        *,
        label: str | None = None,
        normalize_components: bool = True,
        fallback_policy: CapitalAllocationPolicy | None = None,
    ) -> None:
        processed: list[tuple[CapitalAllocationPolicy, float, str | None]] = []
        for entry in components:
            if len(entry) < 2:
                continue
            policy = entry[0]
            try:
                weight = float(entry[1])
            except (TypeError, ValueError):
                continue
            if weight <= 0:
                continue
            label_entry = entry[2] if len(entry) > 2 else None
            processed.append((policy, weight, label_entry))
        self._components = processed
        self._normalize_components = bool(normalize_components)
        self._fallback = fallback_policy
        self.name = str(label or "blended")
        self._last_components: dict[str, dict[str, float]] = {}

    def allocate(self, schedules: Sequence[_ScheduleContext]) -> Mapping[str, float]:
        if not schedules:
            self._last_components = {}
            return {}

        if not self._components:
            self._last_components = {}
            if self._fallback is not None:
                return self._fallback.allocate(schedules)
            return {
                schedule.name: 1.0 / len(schedules)
                for schedule in schedules
            }

        total_weight = sum(weight for _, weight, _ in self._components)
        if total_weight <= 0:
            self._last_components = {}
            if self._fallback is not None:
                return self._fallback.allocate(schedules)
            return {
                schedule.name: 1.0 / len(schedules)
                for schedule in schedules
            }

        strategy_counts = Counter(schedule.strategy_name for schedule in schedules)
        aggregate: defaultdict[str, float] = defaultdict(float)
        diagnostics: dict[str, dict[str, float]] = {}

        for index, (policy, raw_weight, alias) in enumerate(self._components):
            try:
                component_allocation = policy.allocate(schedules) or {}
            except Exception:  # pragma: no cover - diagnostyka polityk składowych
                _LOGGER.exception(
                    "Błąd składnika polityki kapitału %s",
                    getattr(policy, "name", policy),
                )
                continue
            normalized_component = _normalize_weights(component_allocation)
            if not normalized_component:
                continue

            mix_share = raw_weight / total_weight if total_weight > 0 else 0.0
            multiplier = mix_share if self._normalize_components else raw_weight
            component_label = alias or getattr(policy, "name", f"component_{index}")
            component_diag: dict[str, float] = {"mix_weight": float(mix_share)}

            for schedule in schedules:
                weight = normalized_component.get(schedule.name)
                if weight is None:
                    strategy_weight = normalized_component.get(schedule.strategy_name)
                    if strategy_weight is not None:
                        occurrences = max(1, strategy_counts[schedule.strategy_name])
                        weight = float(strategy_weight) / occurrences
                if weight is None:
                    continue
                try:
                    numeric = float(weight)
                except (TypeError, ValueError):
                    continue
                if numeric <= 0 or not math.isfinite(numeric):
                    continue
                component_diag[schedule.name] = numeric
                aggregate[schedule.name] += multiplier * numeric

            if len(component_diag) > 1:
                diagnostics[str(component_label)] = component_diag

        normalized = _normalize_weights(aggregate)
        if not normalized and self._fallback is not None:
            try:
                fallback = self._fallback.allocate(schedules)
            except Exception:  # pragma: no cover - diagnostyka fallbacku
                _LOGGER.exception(
                    "Błąd fallbackowej polityki kapitału %s",
                    getattr(self._fallback, "name", self._fallback),
                )
                fallback = {}
            normalized = _normalize_weights(fallback)
        if not normalized and schedules:
            normalized = {
                schedule.name: 1.0 / len(schedules)
                for schedule in schedules
            }

        self._last_components = diagnostics
        return normalized

    def allocation_diagnostics(self) -> Mapping[str, Mapping[str, float]]:
        """Zwraca wkłady wagowe poszczególnych komponentów."""

        return {key: dict(value) for key, value in self._last_components.items()}


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


class TagQuotaAllocation:
    """Przydziela udziały kapitału na podstawie tagów strategii."""

    _UNASSIGNED_KEY = "__unassigned__"

    def __init__(
        self,
        tag_weights: Mapping[str, float],
        *,
        label: str | None = None,
        fallback_policy: CapitalAllocationPolicy | None = None,
        inner_policy_factory: Callable[[], CapitalAllocationPolicy] | None = None,
        default_weight: float | None = None,
        prefer_primary: bool = True,
    ) -> None:
        normalized: dict[str, float] = {}
        for tag, weight in (tag_weights or {}).items():
            if tag in (None, ""):
                continue
            try:
                numeric = float(weight)
            except (TypeError, ValueError):
                continue
            if numeric <= 0.0 or not math.isfinite(numeric):
                continue
            normalized[str(tag)] = numeric
        self.name = label or "tag_quota"
        self._raw_tag_weights = normalized
        self._fallback_policy = fallback_policy
        self._inner_policy_factory = inner_policy_factory
        parsed_default: float | None = None
        if default_weight not in (None, ""):
            try:
                candidate = float(default_weight)
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and candidate > 0.0 and math.isfinite(candidate):
                parsed_default = candidate
        self._default_weight = parsed_default
        self._prefer_primary = bool(prefer_primary)
        self._last_tag_weights: dict[str, float] = {}
        self._last_tag_counts: dict[str, int] = {}
        self._schedule_diagnostics: dict[str, dict[str, float]] = {}
        self._used_fallback = False

    def _build_inner_policy(self) -> CapitalAllocationPolicy:
        if self._inner_policy_factory is None:
            return EqualWeightAllocation()
        try:
            return self._inner_policy_factory()
        except Exception:  # pragma: no cover - defensywnie
            _LOGGER.exception("TagQuotaAllocation: nie udało się zbudować polityki wewnętrznej")
            return EqualWeightAllocation()

    def _assign_tag(self, schedule: _ScheduleContext, available_tags: Mapping[str, float]) -> str | None:
        if not available_tags:
            return None
        if self._prefer_primary and schedule.primary_tag:
            primary = schedule.primary_tag
            if primary in available_tags:
                return primary
        for tag in schedule.tags:
            if tag in available_tags:
                return tag
        return None

    def _store_snapshots(
        self,
        *,
        tag_weights: Mapping[str, float],
        tag_counts: Mapping[str, int],
        diagnostics: Mapping[str, Mapping[str, float]],
        used_fallback: bool,
    ) -> None:
        self._last_tag_weights = {self._sanitize_tag(key): float(value) for key, value in tag_weights.items()}
        self._last_tag_counts = {
            self._sanitize_tag(key): float(value)
            for key, value in tag_counts.items()
        }
        self._schedule_diagnostics = {
            str(schedule): {
                str(metric): float(value)
                for metric, value in payload.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
            for schedule, payload in diagnostics.items()
            if isinstance(payload, Mapping)
        }
        self._used_fallback = used_fallback

    def _sanitize_tag(self, tag: str) -> str:
        if tag == self._UNASSIGNED_KEY:
            return "unassigned"
        return tag

    def allocate(self, schedules: Sequence[_ScheduleContext]) -> Mapping[str, float]:
        if not schedules:
            self._store_snapshots(tag_weights={}, tag_counts={}, diagnostics={}, used_fallback=False)
            return {}

        available_tags = dict(self._raw_tag_weights)
        groups: dict[str, list[_ScheduleContext]] = {tag: [] for tag in available_tags}
        unassigned: list[_ScheduleContext] = []

        for schedule in schedules:
            tag = self._assign_tag(schedule, available_tags)
            if tag is None:
                unassigned.append(schedule)
            else:
                groups.setdefault(tag, []).append(schedule)

        if unassigned and self._default_weight is not None:
            groups.setdefault(self._UNASSIGNED_KEY, []).extend(unassigned)
            available_tags[self._UNASSIGNED_KEY] = float(self._default_weight)

        participating: dict[str, float] = {
            tag: weight for tag, weight in available_tags.items() if groups.get(tag)
        }
        normalized_tags = _normalize_weights(participating)
        if not normalized_tags:
            fallback = self._fallback_policy or EqualWeightAllocation()
            allocation = fallback.allocate(schedules)
            self._store_snapshots(tag_weights={}, tag_counts={}, diagnostics={}, used_fallback=self._fallback_policy is not None)
            return allocation

        contributions: dict[str, float] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        tag_counts: dict[str, int] = {}

        for tag, share in normalized_tags.items():
            members = groups.get(tag, [])
            if not members:
                continue
            inner_policy = self._build_inner_policy()
            inner_allocation = inner_policy.allocate(members)
            if not inner_allocation:
                inner_weight = 1.0 / len(members)
                for schedule in members:
                    contributions[schedule.name] = contributions.get(schedule.name, 0.0) + share * inner_weight
                    entry = diagnostics.setdefault(schedule.name, {})
                    entry["tag_share"] = share
                    entry["inner_fraction"] = inner_weight
            else:
                # Normalizujemy wagi wewnątrz tagu
                normalized_inner = _normalize_weights(inner_allocation)
                if not normalized_inner:
                    inner_weight = 1.0 / len(members)
                    for schedule in members:
                        contributions[schedule.name] = contributions.get(schedule.name, 0.0) + share * inner_weight
                        entry = diagnostics.setdefault(schedule.name, {})
                        entry["tag_share"] = share
                        entry["inner_fraction"] = inner_weight
                else:
                    for schedule in members:
                        inner_share = normalized_inner.get(schedule.name)
                        if inner_share is None:
                            inner_share = normalized_inner.get(schedule.strategy_name, 0.0)
                        numeric_share = float(inner_share or 0.0)
                        contributions[schedule.name] = contributions.get(schedule.name, 0.0) + share * numeric_share
                        entry = diagnostics.setdefault(schedule.name, {})
                        entry["tag_share"] = share
                        entry["inner_fraction"] = numeric_share

            tag_counts[tag] = len(members)

        normalized = _normalize_weights(contributions)
        self._store_snapshots(
            tag_weights=normalized_tags,
            tag_counts=tag_counts,
            diagnostics=diagnostics,
            used_fallback=False,
        )
        return normalized

    def allocation_diagnostics(self) -> Mapping[str, Mapping[str, float]]:
        return self._schedule_diagnostics

    def tag_allocation_snapshot(self) -> Mapping[str, float]:
        return dict(self._last_tag_weights)

    def tag_member_snapshot(self) -> Mapping[str, float]:
        return {key: float(value) for key, value in self._last_tag_counts.items()}

    @property
    def used_fallback(self) -> bool:
        return self._used_fallback

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
        self._last_allocator_weights: dict[str, float] = {}
        self._last_allocator_raw_weights: dict[str, float] = {}
        self._last_allocator_smoothed_weights: dict[str, float] = {}
        self._last_allocator_profile_weights: dict[str, float] = {}
        self._last_allocator_tag_weights: dict[str, float] = {}
        self._last_allocator_tag_counts: dict[str, float] = {}
        self._last_allocator_diagnostics: dict[str, Mapping[str, float]] = {}
        self._last_allocator_flags: dict[str, bool] = {}
        self._signal_limits: dict[tuple[str, str], SignalLimitOverride] = {}
        self._signal_limit_lock = RLock()
        self._schedule_suspensions: dict[str, SuspensionRecord] = {}
        self._tag_suspensions: dict[str, SuspensionRecord] = {}
        self._active_suspension_reasons: dict[str, str] = {}
        self._suspension_lock = RLock()
        for strategy, profiles in (signal_limits or {}).items():
            for profile, limit in (profiles or {}).items():
                self.configure_signal_limit(strategy, profile, limit)
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

    def set_allocation_rebalance_seconds(self, value: float | None) -> None:
        """Aktualizuje interwał pomiędzy kolejnymi przeliczeniami alokacji."""

        if value is None:
            self._allocation_rebalance_seconds = None
            return
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            _LOGGER.debug(
                "Nie udało się ustawić allocation_rebalance_seconds=%s", value, exc_info=True
            )
            return
        if seconds <= 0.0 or not math.isfinite(seconds):
            self._allocation_rebalance_seconds = None
        else:
            self._allocation_rebalance_seconds = seconds

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

    async def rebalance_capital(
        self,
        *,
        timestamp: datetime | None = None,
        ignore_cooldown: bool = True,
    ) -> None:
        """Wymusza przeliczenie alokacji kapitału według bieżącej polityki."""

        target_timestamp = timestamp or self._clock()
        await self._maybe_rebalance_allocation(
            target_timestamp,
            ignore_cooldown=ignore_cooldown,
        )

    def allocation_snapshot(self) -> Mapping[str, float]:
        """Zwraca ostatnio zastosowane wagi polityki kapitału."""

        return {
            schedule.name: float(schedule.allocator_weight)
            for schedule in self._schedules
        }

    def capital_allocation_state(self) -> Mapping[str, Mapping[str, float]]:
        """Zwraca ostatnie migawki wag (surowe, wygładzone, profilowe)."""

        return {
            "effective": dict(self._last_allocator_weights),
            "raw": dict(self._last_allocator_raw_weights),
            "smoothed": dict(self._last_allocator_smoothed_weights),
            "profiles": dict(self._last_allocator_profile_weights),
            "tags": dict(self._last_allocator_tag_weights),
            "tag_members": dict(self._last_allocator_tag_counts),
        }

    def capital_policy_diagnostics(self) -> Mapping[str, object]:
        """Zwraca diagnostykę ostatniej alokacji polityki kapitału."""

        payload: dict[str, object] = {
            "policy_name": getattr(self._capital_policy, "name", "unknown"),
            "flags": dict(self._last_allocator_flags),
            "details": {
                str(key): dict(value)
                for key, value in self._last_allocator_diagnostics.items()
            },
            "tag_weights": dict(self._last_allocator_tag_weights),
            "tag_members": dict(self._last_allocator_tag_counts),
        }
        if self._last_allocation_at is not None:
            payload["last_rebalance_at"] = self._last_allocation_at.isoformat()
        if self._allocation_rebalance_seconds is not None:
            payload["rebalance_cooldown_seconds"] = float(self._allocation_rebalance_seconds)
        return payload

    def describe_schedules(self) -> Mapping[str, Mapping[str, object]]:
        """Zwraca metadane zarejestrowanych harmonogramów i ich stany."""

        now = self._clock()
        with self._suspension_lock:
            schedule_suspensions = {
                name: record.as_dict(now)
                for name, record in self._schedule_suspensions.items()
                if record.is_active(now)
            }
            tag_suspensions = {
                tag: record.as_dict(now)
                for tag, record in self._tag_suspensions.items()
                if record.is_active(now)
            }

        with self._signal_limit_lock:
            expired_overrides = self._purge_expired_signal_limits(now)
            active_overrides = dict(self._signal_limits)

        if expired_overrides:
            self._handle_expired_signal_limits(expired_overrides, now)

        descriptors: dict[str, dict[str, object]] = {}
        for schedule in self._schedules:
            descriptor: dict[str, object] = {
                "strategy_name": schedule.strategy_name,
                "risk_profile": schedule.risk_profile,
                "cadence_seconds": float(schedule.cadence),
                "max_drift_seconds": float(schedule.max_drift),
                "warmup_bars": int(schedule.warmup_bars),
                "base_max_signals": int(schedule.base_max_signals),
                "active_max_signals": int(schedule.active_max_signals),
                "allocator_weight": float(schedule.allocator_weight),
                "allocator_signal_factor": float(schedule.allocator_signal_factor),
                "governor_signal_factor": float(schedule.governor_signal_factor),
                "portfolio_weight": float(schedule.portfolio_weight),
                "warmed_up": bool(schedule.warmed_up),
                "tags": list(schedule.tags),
                "primary_tag": schedule.primary_tag,
            }
            if schedule.last_run is not None:
                descriptor["last_run"] = schedule.last_run.isoformat()
            limit_override = active_overrides.get(
                (schedule.strategy_name, schedule.risk_profile)
            )
            if limit_override is not None:
                descriptor["signal_limit_override"] = int(limit_override.limit)
                descriptor["signal_limit_details"] = dict(
                    limit_override.to_snapshot(now)
                )

            suspension_info: Mapping[str, object] | None = schedule_suspensions.get(
                schedule.name
            )
            if suspension_info is None:
                for tag in schedule.tags:
                    tag_record = tag_suspensions.get(tag)
                    if tag_record is not None:
                        suspension_info = dict(tag_record)
                        break
                else:
                    if (
                        schedule.primary_tag
                        and schedule.primary_tag not in schedule.tags
                    ):
                        tag_record = tag_suspensions.get(schedule.primary_tag)
                        if tag_record is not None:
                            suspension_info = dict(tag_record)
            if suspension_info is not None:
                descriptor["active_suspension"] = dict(suspension_info)

            numeric_metrics = {
                key: float(value)
                for key, value in schedule.metrics.items()
                if isinstance(value, (int, float))
            }
            if numeric_metrics:
                descriptor["metrics"] = numeric_metrics

            descriptors[schedule.name] = descriptor

        return descriptors

    def administrative_snapshot(
        self,
        *,
        include_metrics: bool = True,
        only_active: bool = False,
        tag: str | None = None,
        strategy: str | None = None,
    ) -> Mapping[str, object]:
        """Buduje raport administracyjny łączący kluczowe migawki runtime."""

        normalized_tag = tag.lower().strip() if isinstance(tag, str) and tag.strip() else None
        normalized_strategy = (
            strategy.lower().strip()
            if isinstance(strategy, str) and strategy.strip()
            else None
        )

        schedule_map = self.describe_schedules()
        filtered: list[dict[str, object]] = []
        aggregated_profiles: defaultdict[str, float] = defaultdict(float)
        aggregated_tags: defaultdict[str, float] = defaultdict(float)

        for name, descriptor in schedule_map.items():
            if not include_metrics and "metrics" in descriptor:
                descriptor = dict(descriptor)
                descriptor.pop("metrics", None)
            entry = dict(descriptor)
            entry["name"] = name
            strategy_name = str(entry.get("strategy_name", "")).lower()
            if normalized_strategy and strategy_name != normalized_strategy:
                continue
            entry_tags = tuple(entry.get("tags", ()))
            if normalized_tag and not any(tag.lower() == normalized_tag for tag in entry_tags):
                primary = entry.get("primary_tag")
                if not (isinstance(primary, str) and primary.lower() == normalized_tag):
                    continue
            has_suspension = bool(entry.get("active_suspension"))
            if only_active and has_suspension:
                continue
            filtered.append(entry)
            try:
                weight = float(entry.get("allocator_weight", 0.0))
            except (TypeError, ValueError):  # pragma: no cover - dane diagnostyczne
                weight = 0.0
            profile = str(entry.get("risk_profile", ""))
            if profile:
                aggregated_profiles[profile] += weight
            for tag_name in entry_tags:
                aggregated_tags[tag_name] += weight
            primary_tag = entry.get("primary_tag")
            if (
                isinstance(primary_tag, str)
                and primary_tag
                and primary_tag not in entry_tags
            ):
                aggregated_tags[primary_tag] += weight

        filtered.sort(key=lambda item: item["name"])

        allocation_state = self.capital_allocation_state()
        policy_info = self.capital_policy_diagnostics()
        schedule_state = {
            "count": len(filtered),
            "schedules": filtered,
            "filters": {
                "raw": {
                    "tag": tag,
                    "strategy": strategy,
                },
                "tag": normalized_tag,
                "strategy": normalized_strategy,
                "only_active": bool(only_active),
            },
            "aggregates": {
                "risk_profiles": dict(aggregated_profiles),
                "tags": dict(aggregated_tags),
            },
        }

        return {
            "environment": self._environment,
            "portfolio": self._portfolio,
            "timestamp": self._clock().isoformat(),
            "capital_allocation": {
                "weights": self.allocation_snapshot(),
                "snapshots": allocation_state,
                "policy": policy_info,
            },
            "schedules": schedule_state,
            "suspensions": self.suspension_snapshot(),
            "signal_limits": self.signal_limit_snapshot(),
        }

    def signal_limit_snapshot(self) -> Mapping[str, Mapping[str, Mapping[str, object]]]:
        """Zwraca aktualne nadpisania limitów sygnałów wraz z metadanymi."""

        snapshot: dict[str, dict[str, Mapping[str, object]]] = {}
        now = self._clock()
        with self._signal_limit_lock:
            expired_overrides = self._purge_expired_signal_limits(now)
            for (strategy, profile), override in self._signal_limits.items():
                strategy_entry = snapshot.setdefault(strategy, {})
                strategy_entry[profile] = dict(override.to_snapshot(now))

        if expired_overrides:
            self._handle_expired_signal_limits(expired_overrides, now)
        return snapshot

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
        tags, primary_tag = _extract_tags(getattr(strategy, "metadata", None))

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
            tags=tags,
            primary_tag=primary_tag,
        )
        self._schedules.append(context)
        _LOGGER.debug("Zarejestrowano harmonogram %s dla strategii %s", name, strategy_name)

    def configure_signal_limit(
        self,
        strategy_name: str,
        risk_profile: str,
        limit: object | None,
        *,
        reason: str | None = None,
        until: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        key = (strategy_name, risk_profile)
        with self._signal_limit_lock:
            if limit in (None, ""):
                self._signal_limits.pop(key, None)
                return
            override = self._normalize_signal_limit_override(
                limit,
                reason=reason,
                until=until,
                duration_seconds=duration_seconds,
            )
            if override is None:
                return
            if override.created_at is None:
                override.created_at = self._clock()
            self._signal_limits[key] = override

    def configure_signal_limits(
        self,
        limits: Mapping[str, Mapping[str, object]],
        *,
        reason: str | None = None,
        until: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        for strategy, profiles in limits.items():
            for profile, limit in profiles.items():
                self.configure_signal_limit(
                    strategy,
                    profile,
                    limit,
                    reason=reason,
                    until=until,
                    duration_seconds=duration_seconds,
                )

    def _normalize_signal_limit_override(
        self,
        limit: object,
        *,
        reason: str | None = None,
        until: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> SignalLimitOverride | None:
        now = self._clock()
        resolved_reason: str | None = (reason or None)
        resolved_until: datetime | None = until
        resolved_duration: float | None = duration_seconds
        created_at: datetime | None = None
        if isinstance(limit, SignalLimitOverride):
            limit_value = limit.limit
            if resolved_reason is None and limit.reason:
                resolved_reason = limit.reason
            if resolved_until is None:
                resolved_until = limit.expires_at
            created_at = limit.created_at
        elif hasattr(limit, "limit") and not isinstance(limit, Mapping):
            limit_value = getattr(limit, "limit", None)
            try:
                limit_value = int(float(limit_value))
            except (TypeError, ValueError):
                return None
            if resolved_reason is None:
                reason_value = getattr(limit, "reason", None)
                if isinstance(reason_value, str):
                    resolved_reason = reason_value.strip() or None
                elif reason_value not in (None, ""):
                    resolved_reason = str(reason_value)
            if resolved_until is None:
                resolved_until = self._coerce_datetime(
                    getattr(limit, "until", None)
                    or getattr(limit, "expires_at", None)
                )
            if resolved_duration is None:
                resolved_duration = self._coerce_duration(
                    getattr(limit, "duration_seconds", None)
                    or getattr(limit, "duration", None)
                )
            created_at = self._coerce_datetime(getattr(limit, "created_at", None))
        elif isinstance(limit, Mapping):
            raw_limit = limit.get("limit", limit.get("value"))
            try:
                limit_value = int(float(raw_limit))
            except (TypeError, ValueError):
                return None
            if resolved_reason is None:
                reason_value = limit.get("reason")
                if isinstance(reason_value, str):
                    resolved_reason = reason_value.strip() or None
                elif reason_value not in (None, ""):
                    resolved_reason = str(reason_value)
            if resolved_until is None:
                resolved_until = self._coerce_datetime(
                    limit.get("until") or limit.get("expires_at")
                )
            if resolved_duration is None:
                resolved_duration = self._coerce_duration(
                    limit.get("duration_seconds") or limit.get("duration")
                )
            created_at = self._coerce_datetime(limit.get("created_at"))
        else:
            try:
                limit_value = int(limit)
            except (TypeError, ValueError):
                return None

        limit_value = max(0, int(limit_value))
        expiry = self._coerce_datetime(resolved_until)
        if expiry is None and resolved_duration not in (None, 0.0):
            try:
                seconds = float(resolved_duration)
            except (TypeError, ValueError):
                seconds = None
            if seconds is not None and math.isfinite(seconds) and seconds > 0.0:
                expiry = now + timedelta(seconds=seconds)

        reason_text = None
        if resolved_reason is not None:
            candidate = str(resolved_reason).strip()
            if candidate:
                reason_text = candidate

        created = self._coerce_datetime(created_at) or now
        return SignalLimitOverride(
            limit=limit_value,
            reason=reason_text,
            expires_at=expiry,
            created_at=created,
        )

    @staticmethod
    def _coerce_datetime(value: object | None) -> datetime | None:
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return None
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        return None

    @staticmethod
    def _coerce_duration(value: object | None) -> float | None:
        if value in (None, ""):
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(seconds) or seconds <= 0.0:
            return None
        return seconds

    def _purge_expired_signal_limits(
        self, now: datetime | None = None
    ) -> Mapping[tuple[str, str], SignalLimitOverride]:
        moment = now or self._clock()
        expired: dict[tuple[str, str], SignalLimitOverride] = {}
        for key, override in list(self._signal_limits.items()):
            if override.is_expired(moment):
                removed = self._signal_limits.pop(key, None)
                if removed is not None:
                    expired[key] = removed
        return expired

    def _handle_expired_signal_limits(
        self,
        expired: Mapping[tuple[str, str], SignalLimitOverride],
        now: datetime,
        *,
        skip: Sequence[_ScheduleContext] | None = None,
    ) -> None:
        if not expired:
            return

        skip_ids = {id(context) for context in (skip or ())}
        for (strategy, profile), override in expired.items():
            reason_part = f", powód: {override.reason}" if override.reason else ""
            expiry_part = (
                f", wygasło o {override.expires_at.isoformat()}"
                if override.expires_at
                else ""
            )
            _LOGGER.info(
                "Wygasło nadpisanie limitu sygnałów %s/%s (limit=%s%s%s)",
                strategy,
                profile,
                override.limit,
                reason_part,
                expiry_part,
            )

        for schedule in self._schedules:
            if id(schedule) in skip_ids:
                continue
            key = (schedule.strategy_name, schedule.risk_profile)
            if key in expired:
                self._apply_signal_limits(schedule)

    def suspend_schedule(
        self,
        schedule_name: str,
        *,
        reason: str | None = None,
        until: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        name = (schedule_name or "").strip()
        if not name:
            return
        reason_text = (reason or "manual").strip() or "manual"
        now = self._clock()
        expiry = self._resolve_suspension_expiry(now, until, duration_seconds)
        record = SuspensionRecord(reason=reason_text, applied_at=now, until=expiry)
        with self._suspension_lock:
            self._schedule_suspensions[name] = record
        _LOGGER.warning(
            "Zawieszono harmonogram %s z powodu: %s%s",
            name,
            reason_text,
            f" (do {expiry.isoformat()})" if expiry else "",
        )

    def resume_schedule(self, schedule_name: str) -> bool:
        name = (schedule_name or "").strip()
        if not name:
            return False
        with self._suspension_lock:
            removed = self._schedule_suspensions.pop(name, None) is not None
        if removed:
            _LOGGER.info("Wznowiono harmonogram %s", name)
            self._active_suspension_reasons.pop(name, None)
        return removed

    def suspend_tag(
        self,
        tag: str,
        *,
        reason: str | None = None,
        until: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        normalized = (tag or "").strip()
        if not normalized:
            return
        reason_text = (reason or "manual").strip() or "manual"
        now = self._clock()
        expiry = self._resolve_suspension_expiry(now, until, duration_seconds)
        record = SuspensionRecord(
            reason=reason_text,
            applied_at=now,
            until=expiry,
            origin="tag",
            tag=normalized,
        )
        with self._suspension_lock:
            self._tag_suspensions[normalized] = record
        _LOGGER.warning(
            "Zawieszono tag strategii %s z powodu: %s%s",
            normalized,
            reason_text,
            f" (do {expiry.isoformat()})" if expiry else "",
        )

    def resume_tag(self, tag: str) -> bool:
        normalized = (tag or "").strip()
        if not normalized:
            return False
        with self._suspension_lock:
            removed = self._tag_suspensions.pop(normalized, None) is not None
        if removed:
            _LOGGER.info("Wznowiono tag strategii %s", normalized)
        return removed

    def suspension_snapshot(self) -> Mapping[str, Mapping[str, object]]:
        now = self._clock()
        self._purge_expired_suspensions(now)
        schedules: dict[str, dict[str, object]] = {}
        tags: dict[str, dict[str, object]] = {}
        with self._suspension_lock:
            for name, record in self._schedule_suspensions.items():
                schedules[name] = record.as_dict(now)
            for tag_name, record in self._tag_suspensions.items():
                tags[tag_name] = record.as_dict(now)
        return {"schedules": schedules, "tags": tags}

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
            suspension = self._resolve_schedule_suspension(schedule, timestamp)
            suspended = self._update_suspension_state(schedule.name, suspension)
            schedule.last_run = timestamp
            if suspended and suspension is not None:
                await self._maybe_rebalance_allocation(timestamp)
                self._handle_suspended_schedule(schedule, timestamp, suspension)
                continue
            await self._execute_schedule(schedule, timestamp)
        await self._evaluate_portfolio(force=True)

    async def _run_schedule(self, schedule: _ScheduleContext) -> None:
        assert self._stop_event is not None, "Scheduler musi zostać zainicjalizowany"
        cadence = max(1.0, schedule.cadence)
        while not self._stop_event.is_set():
            start_time = self._clock()
            suspension = self._resolve_schedule_suspension(schedule, start_time)
            suspended = self._update_suspension_state(schedule.name, suspension)
            if suspended and suspension is not None:
                await self._maybe_rebalance_allocation(start_time)
                self._handle_suspended_schedule(schedule, start_time, suspension)
            else:
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

    def _resolve_schedule_suspension(
        self, schedule: _ScheduleContext, timestamp: datetime
    ) -> SuspensionRecord | None:
        self._purge_expired_suspensions(timestamp)
        with self._suspension_lock:
            record = self._schedule_suspensions.get(schedule.name)
            if record is not None:
                return record
            for tag in schedule.tags:
                tag_record = self._tag_suspensions.get(tag)
                if tag_record:
                    return tag_record.clone_for_tag(tag)
            if schedule.primary_tag and schedule.primary_tag not in schedule.tags:
                tag_record = self._tag_suspensions.get(schedule.primary_tag)
                if tag_record:
                    return tag_record.clone_for_tag(schedule.primary_tag)
        return None

    def _purge_expired_suspensions(self, now: datetime) -> None:
        expired_schedules: list[tuple[str, SuspensionRecord]] = []
        expired_tags: list[tuple[str, SuspensionRecord]] = []
        with self._suspension_lock:
            for name, record in list(self._schedule_suspensions.items()):
                if not record.is_active(now):
                    expired_schedules.append((name, record))
                    self._schedule_suspensions.pop(name, None)
            for tag_name, record in list(self._tag_suspensions.items()):
                if not record.is_active(now):
                    expired_tags.append((tag_name, record))
                    self._tag_suspensions.pop(tag_name, None)
        for name, record in expired_schedules:
            descriptor = self._active_suspension_reasons.pop(name, None) or record.reason
            _LOGGER.info(
                "Harmonogram %s automatycznie wznowiony po wygaśnięciu zawieszenia (%s)",
                name,
                descriptor,
            )
        for tag_name, record in expired_tags:
            descriptor = record.reason
            _LOGGER.info(
                "Tag strategii %s automatycznie wznowiony po wygaśnięciu zawieszenia (%s)",
                tag_name,
                descriptor,
            )

    def _update_suspension_state(
        self, schedule_name: str, record: SuspensionRecord | None
    ) -> bool:
        descriptor: str | None = None
        if record is not None:
            descriptor = record.reason
            if record.origin == "tag" and record.tag:
                descriptor = f"{descriptor} [tag={record.tag}]"

        previous = self._active_suspension_reasons.get(schedule_name)
        if record is None:
            if previous is not None:
                _LOGGER.info(
                    "Harmonogram %s wznowiony po zawieszeniu (%s)",
                    schedule_name,
                    previous,
                )
                self._active_suspension_reasons.pop(schedule_name, None)
            return False

        if descriptor is None:
            descriptor = "manual"

        if previous != descriptor:
            if previous is None:
                _LOGGER.warning(
                    "Harmonogram %s przechodzi w stan zawieszenia: %s",
                    schedule_name,
                    descriptor,
                )
            else:
                _LOGGER.warning(
                    "Harmonogram %s zmienia powód zawieszenia: %s -> %s",
                    schedule_name,
                    previous,
                    descriptor,
                )
            self._active_suspension_reasons[schedule_name] = descriptor
        return True

    def _handle_suspended_schedule(
        self, schedule: _ScheduleContext, timestamp: datetime, record: SuspensionRecord
    ) -> None:
        schedule.metrics.clear()
        schedule.metrics["base_max_signals"] = float(schedule.base_max_signals)
        schedule.metrics["active_max_signals"] = 0.0
        schedule.metrics["signals"] = 0.0
        schedule.metrics["suspended"] = 1.0
        schedule.metrics["allocator_signal_factor"] = float(
            schedule.allocator_signal_factor
        )
        schedule.metrics["allocator_weight_target"] = float(schedule.allocator_weight)
        schedule.metrics["governor_signal_factor"] = float(schedule.governor_signal_factor)
        schedule.metrics["portfolio_weight_target"] = float(schedule.portfolio_weight)
        schedule.metrics["portfolio_weight"] = float(schedule.portfolio_weight)
        schedule.metrics["last_latency_ms"] = 0.0
        if record.origin == "tag" and record.tag:
            schedule.metrics["suspension_tag_indicator"] = 1.0
        else:
            schedule.metrics["suspension_tag_indicator"] = 0.0
        remaining = record.remaining_seconds(timestamp)
        if remaining is not None:
            schedule.metrics["suspension_remaining_seconds"] = remaining
        self._emit_metrics(schedule)
        if self._portfolio_governor is not None:
            observation_payload = dict(schedule.metrics)
            self._portfolio_governor.observe_strategy_metrics(
                schedule.strategy_name,
                observation_payload,
                timestamp=timestamp,
                risk_profile=schedule.risk_profile,
            )
            decision = self._portfolio_governor.maybe_rebalance(timestamp=timestamp)
            if decision is not None:
                self._apply_portfolio_decision(decision)

    def _resolve_suspension_expiry(
        self,
        now: datetime,
        until: datetime | None,
        duration_seconds: float | None,
    ) -> datetime | None:
        if until is not None:
            if until.tzinfo is None:
                return until.replace(tzinfo=timezone.utc)
            return until.astimezone(timezone.utc)
        if duration_seconds is None:
            return None
        try:
            seconds = float(duration_seconds)
        except (TypeError, ValueError):
            return None
        if seconds <= 0:
            return None
        return now + timedelta(seconds=seconds)

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

    async def _maybe_rebalance_allocation(
        self,
        timestamp: datetime,
        *,
        ignore_cooldown: bool = False,
    ) -> None:
        policy = getattr(self, "_capital_policy", None)
        if policy is None:
            return
        if (
            not ignore_cooldown
            and self._allocation_rebalance_seconds is not None
            and self._last_allocation_at is not None
        ):
            delta = (timestamp - self._last_allocation_at).total_seconds()
            if delta < self._allocation_rebalance_seconds:
                return
        raw_snapshot: dict[str, float] = {}
        smoothed_snapshot: dict[str, float] = {}
        profile_snapshot: dict[str, float] | None = None
        tag_snapshot: dict[str, float] | None = None
        tag_member_snapshot: dict[str, float] | None = None
        diagnostics_snapshot: dict[str, Mapping[str, float]] = {}
        floor_adjustment_flag: bool | None = None
        fallback_flag: bool | None = None
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
            raw_getter = getattr(policy, "raw_allocation_snapshot", None)
            if callable(raw_getter):
                try:
                    raw_result = raw_getter() or {}
                except Exception:  # pragma: no cover - diagnostyka polityki
                    _LOGGER.exception(
                        "Nie udało się pobrać surowych wag z polityki %s",
                        policy.name,
                    )
                else:
                    for key, value in raw_result.items():
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            raw_snapshot[str(key)] = float(value)
            smoothed_getter = getattr(policy, "smoothed_allocation_snapshot", None)
            if callable(smoothed_getter):
                try:
                    smoothed_result = smoothed_getter() or {}
                except Exception:  # pragma: no cover - diagnostyka polityki
                    _LOGGER.exception(
                        "Nie udało się pobrać wygładzonych wag z polityki %s",
                        policy.name,
                    )
                else:
                    for key, value in smoothed_result.items():
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            smoothed_snapshot[str(key)] = float(value)
            snapshot_getter = getattr(policy, "profile_allocation_snapshot", None)
            if callable(snapshot_getter):
                try:
                    snapshot = snapshot_getter() or {}
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
                            if isinstance(value, (int, float))
                            and math.isfinite(float(value))
                        }
            diagnostics_getter = getattr(policy, "allocation_diagnostics", None)
            if callable(diagnostics_getter):
                try:
                    diagnostics = diagnostics_getter() or {}
                except Exception:  # pragma: no cover - diagnostyka polityki
                    _LOGGER.exception(
                        "Nie udało się pobrać diagnostyki alokacji z polityki %s",
                        policy.name,
                    )
                else:
                    sanitized: dict[str, Mapping[str, float]] = {}
                    for key, payload in diagnostics.items():
                        if not isinstance(payload, Mapping):
                            continue
                        inner: dict[str, float] = {}
                        for inner_key, inner_value in payload.items():
                            if isinstance(inner_value, (int, float)) and math.isfinite(
                                float(inner_value)
                            ):
                                inner[str(inner_key)] = float(inner_value)
                        if inner:
                            sanitized[str(key)] = inner
                    diagnostics_snapshot = sanitized
            tag_getter = getattr(policy, "tag_allocation_snapshot", None)
            if callable(tag_getter):
                try:
                    tag_result = tag_getter() or {}
                except Exception:  # pragma: no cover - diagnostyka polityki
                    _LOGGER.exception(
                        "Nie udało się pobrać rozkładu tagów z polityki %s",
                        policy.name,
                    )
                else:
                    if tag_result:
                        tag_snapshot = {
                            str(key): float(value)
                            for key, value in tag_result.items()
                            if isinstance(value, (int, float))
                            and math.isfinite(float(value))
                        }
            tag_member_getter = getattr(policy, "tag_member_snapshot", None)
            if callable(tag_member_getter):
                try:
                    member_result = tag_member_getter() or {}
                except Exception:  # pragma: no cover - diagnostyka polityki
                    _LOGGER.exception(
                        "Nie udało się pobrać liczności tagów z polityki %s",
                        policy.name,
                    )
                else:
                    if member_result:
                        tag_member_snapshot = {
                            str(key): float(value)
                            for key, value in member_result.items()
                            if isinstance(value, (int, float))
                            and math.isfinite(float(value))
                        }
            floor_attr = getattr(policy, "floor_adjustment_applied", None)
            if isinstance(floor_attr, bool):
                floor_adjustment_flag = floor_attr
            elif floor_attr is not None:
                floor_adjustment_flag = bool(floor_attr)
            fallback_attr = getattr(policy, "used_fallback", None)
            if isinstance(fallback_attr, bool):
                fallback_flag = fallback_attr
            self._last_allocation_at = timestamp
        if not normalized:
            self._last_allocator_weights = {}
            self._last_allocator_raw_weights = {}
            self._last_allocator_smoothed_weights = {}
            self._last_allocator_profile_weights = {}
            self._last_allocator_tag_weights = {}
            self._last_allocator_tag_counts = {}
            self._last_allocator_diagnostics = {}
            self._last_allocator_flags = {}
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

        flags: dict[str, bool] = {}
        if floor_adjustment_flag is not None:
            flags["profile_floor_adjustment"] = bool(floor_adjustment_flag)
        normalized_raw_snapshot = _normalize_weights(raw_snapshot) if raw_snapshot else {}
        if not smoothed_snapshot:
            smoothed_snapshot = dict(normalized)
        self._last_allocator_flags = flags
        self._last_allocator_weights = dict(normalized)
        self._last_allocator_raw_weights = dict(normalized_raw_snapshot)
        self._last_allocator_smoothed_weights = dict(smoothed_snapshot)
        self._last_allocator_profile_weights = dict(normalized_profile_snapshot)
        self._last_allocator_tag_weights = dict(tag_snapshot or {})
        self._last_allocator_tag_counts = dict(tag_member_snapshot or {})
        self._last_allocator_diagnostics = diagnostics_snapshot
        if fallback_flag is not None:
            self._last_allocator_flags["fallback_used"] = bool(fallback_flag)

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
            tag_key = schedule.primary_tag or (schedule.tags[0] if schedule.tags else None)
            if tag_key:
                schedule.metrics["allocator_tag_weight"] = self._last_allocator_tag_weights.get(
                    tag_key,
                    0.0,
                )
                schedule.metrics["allocator_tag_members"] = self._last_allocator_tag_counts.get(
                    tag_key,
                    0.0,
                )
            raw_weight = normalized_raw_snapshot.get(schedule.name)
            if raw_weight is None:
                raw_weight = normalized_raw_snapshot.get(schedule.strategy_name)
            if raw_weight is not None:
                schedule.metrics["allocator_raw_weight"] = float(raw_weight)
            smoothed_weight = smoothed_snapshot.get(schedule.name)
            if smoothed_weight is None:
                smoothed_weight = smoothed_snapshot.get(schedule.strategy_name)
            if smoothed_weight is not None:
                schedule.metrics["allocator_smoothed_weight"] = float(smoothed_weight)

    def _apply_signal_limits(self, schedule: _ScheduleContext) -> None:
        floor = self._min_signal_floor()
        factor = max(0.0, schedule.governor_signal_factor) * max(
            0.0, schedule.allocator_signal_factor
        )
        computed = int(round(schedule.base_max_signals * factor))
        if schedule.base_max_signals > 0:
            computed = max(floor if factor > 0 else 0, computed)
        schedule.metrics.pop("signal_limit_override", None)
        schedule.metrics.pop("signal_limit_expires_at", None)
        schedule.metrics.pop("signal_limit_reason", None)
        override_value: int | None = None
        override_reason: str | None = None
        override_until: datetime | None = None
        now = self._clock()
        expired_override: Mapping[tuple[str, str], SignalLimitOverride] | None = None
        with self._signal_limit_lock:
            key = (schedule.strategy_name, schedule.risk_profile)
            override = self._signal_limits.get(key)
            if override is not None and override.is_expired(now):
                removed = self._signal_limits.pop(key, None)
                if removed is not None:
                    expired_override = {key: removed}
                override = None
            if override is not None:
                override_value = max(0, int(override.limit))
                override_reason = override.reason
                override_until = override.expires_at
        if expired_override:
            self._handle_expired_signal_limits(expired_override, now, skip=(schedule,))
        if override_value is not None:
            if computed > override_value:
                _LOGGER.debug(
                    "Sygnal limit override dla %s/%s: %s -> %s",
                    schedule.strategy_name,
                    schedule.risk_profile,
                    computed,
                    override_value,
                )
            computed = min(computed, override_value)
            schedule.metrics["signal_limit_override"] = float(override_value)
            if override_until is not None:
                schedule.metrics["signal_limit_expires_at"] = override_until.timestamp()
            if override_reason:
                schedule.metrics["signal_limit_reason"] = 1.0
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
    "MetricWeightRule",
    "EqualWeightAllocation",
    "BlendedCapitalAllocation",
    "RiskParityAllocation",
    "VolatilityTargetAllocation",
    "SignalStrengthAllocation",
    "MetricWeightedAllocation",
    "EqualWeightAllocation",
    "RiskParityAllocation",
    "VolatilityTargetAllocation",
    "SignalStrengthAllocation",
    "SmoothedCapitalAllocationPolicy",
    "DrawdownAdaptiveAllocation",
    "FixedWeightAllocation",
    "RiskProfileBudgetAllocation",
    "TagQuotaAllocation",
]
