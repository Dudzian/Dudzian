"""Polityki alokacji kapitału wykorzystywane przez MultiStrategyScheduler."""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping, MutableMapping, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - tylko dla typowania
    from bot_core.runtime.multi_strategy_scheduler import _ScheduleContext


_LOGGER = logging.getLogger(__name__)


class CapitalAllocationPolicy(Protocol):
    """Polityka wyznaczająca wagi kapitału pomiędzy strategie."""

    name: str

    def allocate(self, schedules: Sequence["_ScheduleContext"]) -> Mapping[str, float]:
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


def normalize_weights(candidates: Mapping[str, float]) -> dict[str, float]:
    """Normalizuje wagi tak, aby ich suma wynosiła 1."""

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

    def allocate(self, schedules: Sequence["_ScheduleContext"]) -> Mapping[str, float]:
        if not schedules:
            return {}
        weight = 1.0 / len(schedules)
        return {schedule.name: weight for schedule in schedules}


class RiskParityAllocation:
    """Polityka alokacji proporcjonalna do odwrotności zmienności."""

    name = "risk_parity"

    def allocate(self, schedules: Sequence["_ScheduleContext"]) -> Mapping[str, float]:
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
        return normalize_weights(scores)


class VolatilityTargetAllocation:
    """Polityka zwiększająca udział strategii trafiających w target zmienności."""

    name = "volatility_target"

    def allocate(self, schedules: Sequence["_ScheduleContext"]) -> Mapping[str, float]:
        scores: dict[str, float] = {}
        for schedule in schedules:
            error = schedule.metrics.get("realized_vs_target_vol_pct")
            if not isinstance(error, (int, float)) or error < 0:
                error = schedule.metrics.get("allocation_error_pct")
            if not isinstance(error, (int, float)) or error < 0:
                error = 0.0
            weight = 1.0 / (1.0 + abs(float(error)))
            scores[schedule.name] = weight
        normalized = normalize_weights(scores)
        if not normalized:
            return EqualWeightAllocation().allocate(schedules)
        return normalized


class SignalStrengthAllocation:
    """Preferuje strategie generujące częściej wysokiej jakości sygnały."""

    name = "signal_strength"

    def allocate(self, schedules: Sequence["_ScheduleContext"]) -> Mapping[str, float]:
        scores: dict[str, float] = {}
        for schedule in schedules:
            signals = schedule.metrics.get("signals")
            confidence = schedule.metrics.get("avg_confidence")
            try:
                signal_rate = max(0.0, float(signals))
            except (TypeError, ValueError):
                signal_rate = 0.0
            try:
                avg_confidence = max(0.0, float(confidence))
            except (TypeError, ValueError):
                avg_confidence = 0.0
            scores[schedule.name] = signal_rate * (1.0 + avg_confidence)
        normalized = normalize_weights(scores)
        if not normalized:
            return EqualWeightAllocation().allocate(schedules)
        return normalized


class MetricWeightedAllocation:
    """Polityka bazująca na wielu metrykach z możliwością fallbacku."""

    name = "metric_weighted"

    def __init__(
        self,
        rules: Sequence[MetricWeightRule],
        *,
        fallback_policy: CapitalAllocationPolicy | None = None,
        shift_epsilon: float = 1e-6,
        default_score: float = 0.0,
        label: str | None = None,
    ) -> None:
        self._rules = tuple(rules)
        self._fallback = fallback_policy
        self._shift_epsilon = max(1e-9, float(shift_epsilon))
        self._default_score = float(default_score)
        self._last_snapshot: dict[str, dict[str, float]] = {}
        if label:
            self.name = str(label)

    @property
    def metrics(self) -> tuple[MetricWeightRule, ...]:
        return self._rules

    @property
    def fallback_policy(self) -> CapitalAllocationPolicy | None:
        return self._fallback

    def allocate(self, schedules: Sequence["_ScheduleContext"]) -> Mapping[str, float]:
        if not schedules:
            self._last_snapshot = {}
            return {}

        if not self._rules:
            return self._fallback_or_equal(
                schedules,
                diagnostics=None,
                reason="no_rules",
            )

        scores: dict[str, float] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        shift = 0.0

        for schedule in schedules:
            raw_score = self._default_score
            details: dict[str, float] = {}
            for rule in self._rules:
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
                shift = (
                    abs(min_score) + self._shift_epsilon if math.isfinite(min_score) else self._shift_epsilon
                )

        shifted: dict[str, float] = {}
        for schedule_name, raw_score in scores.items():
            shifted_score = raw_score + shift
            if shifted_score <= 0.0 or not math.isfinite(shifted_score):
                shifted_score = 0.0
            diagnostics[schedule_name]["shifted_score"] = shifted_score
            diagnostics[schedule_name]["shift"] = shift
            shifted[schedule_name] = shifted_score

        normalized = normalize_weights(shifted)
        if not normalized:
            return self._fallback_or_equal(
                schedules,
                diagnostics=diagnostics,
                reason="normalize_failed",
            )

        self._last_snapshot = {key: dict(value) for key, value in diagnostics.items()}
        return normalized

    def allocation_diagnostics(self) -> Mapping[str, Mapping[str, float]]:
        return {key: dict(value) for key, value in self._last_snapshot.items()}

    def _fallback_or_equal(
        self,
        schedules: Sequence["_ScheduleContext"],
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

    def allocate(self, schedules: Sequence["_ScheduleContext"]) -> Mapping[str, float]:
        if not schedules:
            self._last_smoothed = {}
            self._last_raw = {}
            return {}

        try:
            raw_allocation = self.inner_policy.allocate(schedules)
        except Exception:  # pragma: no cover - diagnostyka polityk zewnętrznych
            _LOGGER.exception("Błąd wewnętrznej polityki alokacji kapitału")
            raw_allocation = {}

        normalized_raw = normalize_weights(raw_allocation)
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
                if abs(delta) < self.min_delta:
                    smoothed_value = previous
                else:
                    smoothed_value = previous + alpha * delta
            smoothed[name] = max(self.floor_weight, smoothed_value)

        normalized = normalize_weights(smoothed)
        if not normalized:
            normalized = EqualWeightAllocation().allocate(schedules)

        self._last_raw = normalize_weights(raw_weights)
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

    def allocate(self, schedules: Sequence["_ScheduleContext"]) -> Mapping[str, float]:
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
            normalized_component = normalize_weights(component_allocation)
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

        normalized = normalize_weights(aggregate)
        if not normalized and self._fallback is not None:
            try:
                fallback = self._fallback.allocate(schedules)
            except Exception:  # pragma: no cover - diagnostyka fallbacku
                _LOGGER.exception(
                    "Błąd fallbackowej polityki kapitału %s",
                    getattr(self._fallback, "name", self._fallback),
                )
                fallback = {}
            normalized = normalize_weights(fallback)
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

    def allocate(self, schedules: Sequence["_ScheduleContext"]) -> Mapping[str, float]:
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

        normalized = normalize_weights(weights)
        if not normalized:
            normalized = EqualWeightAllocation().allocate(schedules)
        self._last_snapshot = snapshot
        return normalized

    def allocation_diagnostics(self) -> Mapping[str, Mapping[str, float]]:
        """Zwraca ostatnie metryki drawdownu i zastosowane kary wagowe."""

        return {key: dict(value) for key, value in self._last_snapshot.items()}

    def _extract_drawdown_pct(self, schedule: "_ScheduleContext") -> float:
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

    def _extract_drawdown_pressure(self, schedule: "_ScheduleContext") -> float:
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
        self._weights = normalize_weights(normalized_source)
        self.name = str(label or "fixed_weight")

    def allocate(self, schedules: Sequence["_ScheduleContext"]) -> Mapping[str, float]:
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
        return normalize_weights(matched)


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
        self._profile_targets = normalize_weights(normalized_profiles)
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
        normalized: Mapping[str, float],
        groups: Mapping[str, Sequence["_ScheduleContext"]],
    ) -> Mapping[str, float]:
        if not normalized:
            return {}
        floor = min(self._profile_floor, 1.0)
        keys = list(groups.keys())
        if not keys:
            return normalized
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

    def allocate(self, schedules: Sequence["_ScheduleContext"]) -> Mapping[str, float]:
        if not schedules:
            self._last_profile_weights = {}
            self._last_floor_adjustment = False
            return {}

        groups: dict[str, list["_ScheduleContext"]] = {}
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

        normalized_profiles = normalize_weights(configured_targets)
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
            normalized_inner = normalize_weights(inner_weights)
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

        return normalize_weights(schedule_weights)

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

    def _assign_tag(self, schedule: "_ScheduleContext", available_tags: Mapping[str, float]) -> str | None:
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
            self._sanitize_tag(key): int(value)
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

    def allocate(self, schedules: Sequence["_ScheduleContext"]) -> Mapping[str, float]:
        if not schedules:
            self._store_snapshots(tag_weights={}, tag_counts={}, diagnostics={}, used_fallback=False)
            return {}

        available_tags = dict(self._raw_tag_weights)
        groups: dict[str, list["_ScheduleContext"]] = {tag: [] for tag in available_tags}
        unassigned: list["_ScheduleContext"] = []

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
        normalized_tags = normalize_weights(participating)
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
                normalized_inner = normalize_weights(inner_allocation)
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

        normalized = normalize_weights(contributions)
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

