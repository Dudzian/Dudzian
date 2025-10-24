"""Workflow zarządzania presetami strategii w zależności od reżimu rynku."""
from __future__ import annotations

from collections import Counter, deque
import math
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from types import MappingProxyType
from typing import Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

from bot_core.ai import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeHistory,
    RegimeSummary,
)
from bot_core.auto_trader.schedule import ScheduleWindow
from bot_core.decision.models import DecisionCandidate
from bot_core.decision.orchestrator import DecisionOrchestrator
from bot_core.security.guards import LicenseCapabilityError, get_capability_guard
from bot_core.security.signing import build_hmac_signature, canonical_json_bytes
from bot_core.strategies.catalog import StrategyPresetWizard


_LOGGER = logging.getLogger(__name__)


def _normalize_regime(value: MarketRegime | str | None) -> MarketRegime | None:
    if value is None:
        return None
    if isinstance(value, MarketRegime):
        return value
    try:
        return MarketRegime(str(value).lower())
    except ValueError:
        raise ValueError(f"Unsupported market regime: {value!r}") from None


def _gather_strings(mapping: Iterable[Iterable[str]]) -> tuple[str, ...]:
    items: list[str] = []
    for group in mapping:
        for entry in group:
            candidate = str(entry).strip()
            if not candidate:
                continue
            if candidate not in items:
                items.append(candidate)
    return tuple(items)


@dataclass(frozen=True)
class PresetVersionInfo:
    """Metadane wersji presetu strategii."""

    hash: str
    signature: Mapping[str, str]
    issued_at: datetime
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RegimePresetActivation:
    """Wynik aktywacji presetu po ocenie reżimu rynku."""

    regime: MarketRegime
    assessment: MarketRegimeAssessment
    summary: RegimeSummary | None
    preset: Mapping[str, object]
    version: PresetVersionInfo
    decision_candidates: tuple[DecisionCandidate, ...]
    activated_at: datetime
    preset_regime: MarketRegime | None
    used_fallback: bool = False
    missing_data: tuple[str, ...] = ()
    blocked_reason: str | None = None
    recommendation: str | None = None
    license_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RegisteredPreset:
    regime: MarketRegime | None
    preset: Mapping[str, object]
    version: PresetVersionInfo
    license_tiers: tuple[str, ...]
    risk_classes: tuple[str, ...]
    required_data: tuple[str, ...]
    capabilities: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class PresetAvailability:
    """Raport gotowości presetu względem wymogów workflow."""

    regime: MarketRegime | None
    version: PresetVersionInfo
    ready: bool
    blocked_reason: str | None
    missing_data: tuple[str, ...]
    license_issues: tuple[str, ...]
    schedule_blocked: bool


@dataclass(frozen=True)
class ActivationHistoryStats:
    """Podsumowanie statystyczne historii aktywacji presetów."""

    total: int
    regime_counts: Mapping[MarketRegime, int]
    preset_regime_counts: Mapping[MarketRegime | None, int]
    fallback_count: int
    blocked_reasons: Mapping[str, int]
    missing_data: Mapping[str, int]
    license_issue_counts: Mapping[str, int]


@dataclass(frozen=True)
class ActivationTransitionStats:
    """Statystyka przejść pomiędzy kolejnymi aktywacjami presetów."""

    total: int
    regime_transitions: Mapping[tuple[MarketRegime, MarketRegime], int]
    preset_regime_transitions: Mapping[tuple[MarketRegime | None, MarketRegime | None], int]
    blocked_transitions: Mapping[tuple[str | None, str | None], int]
    fallback_transitions: int


@dataclass(frozen=True)
class ActivationCadenceStats:
    """Metryki kadencji aktywacji wyliczone z historii workflow."""

    total: int
    intervals: int
    min_interval: timedelta | None
    max_interval: timedelta | None
    mean_interval: timedelta | None
    median_interval: timedelta | None
    last_interval: timedelta | None


@dataclass(frozen=True)
class ActivationUptimeStats:
    """Zestawienie czasu spędzonego w poszczególnych aktywacjach."""

    total: int
    duration: timedelta
    regime_uptime: Mapping[MarketRegime, timedelta]
    preset_uptime: Mapping[MarketRegime | None, timedelta]
    fallback_uptime: timedelta


@dataclass(frozen=True)
class ActivationReliabilityStats:
    """Ocena niezawodności aktywacji na podstawie historii workflow."""

    total: int
    completed: int
    fallback_count: int
    blocked_count: int
    completed_ratio: float
    fallback_ratio: float
    blocked_ratio: float


@dataclass(frozen=True)
class ActivationOutcomeStats:
    """Rozkład wyników aktywacji w zależności od reżimu i presetu."""

    total: int
    completed_total: int
    fallback_total: int
    blocked_total: int
    regime_completed: Mapping[MarketRegime, int]
    regime_fallback: Mapping[MarketRegime, int]
    regime_blocked: Mapping[MarketRegime, int]
    preset_completed: Mapping[MarketRegime | None, int]
    preset_fallback: Mapping[MarketRegime | None, int]
    preset_blocked: Mapping[MarketRegime | None, int]


@dataclass(frozen=True)
class ActivationBlockerStats:
    """Analiza blokad aktywacji z perspektywy reżimów i powodów."""

    total: int
    regime_blocked: Mapping[MarketRegime, int]
    reason_counts: Mapping[str, int]
    missing_data: Mapping[str, int]
    missing_data_by_regime: Mapping[MarketRegime, Mapping[str, int]]
    license_issues: Mapping[str, int]
    license_issues_by_regime: Mapping[MarketRegime, Mapping[str, int]]


@dataclass(frozen=True)
class ActivationPresetStats:
    """Zestawienie wykorzystania presetów według nazw i wersji."""

    total: int
    preset_usage: Mapping[str, int]
    version_usage: Mapping[str, int]
    versions_by_preset: Mapping[str, Mapping[str, int]]
    regime_preset_usage: Mapping[MarketRegime, Mapping[str, int]]
    fallback_by_preset: Mapping[str, int]
    fallback_by_version: Mapping[str, int]
    fallback_by_regime: Mapping[MarketRegime, int]
    blocked_by_preset: Mapping[str, int]
    blocked_by_version: Mapping[str, int]
    blocked_by_regime: Mapping[MarketRegime, int]


@dataclass(frozen=True)
class ActivationLicenseStats:
    """Agregacja problemów licencyjnych napotkanych w historii aktywacji."""

    total: int
    activations_with_issues: int
    blocked_by_license: int
    fallback_due_to_license: int
    issue_counts: Mapping[str, int]
    issues_by_regime: Mapping[MarketRegime, Mapping[str, int]]
    issues_by_preset: Mapping[str, Mapping[str, int]]
    regimes_with_issues: Mapping[MarketRegime, int]
    presets_with_issues: Mapping[str, int]


@dataclass(frozen=True)
class ActivationAssessmentStats:
    """Podsumowanie ocen reżimu rynku z historii aktywacji."""

    total: int
    regime_counts: Mapping[MarketRegime, int]
    min_confidence: float | None
    max_confidence: float | None
    mean_confidence: float | None
    min_risk_score: float | None
    max_risk_score: float | None
    mean_risk_score: float | None
    metrics_mean: Mapping[str, float]
    metrics_min: Mapping[str, float]
    metrics_max: Mapping[str, float]


@dataclass(frozen=True)
class ActivationDataStats:
    """Podsumowanie braków danych rynkowych w historii aktywacji."""

    total: int
    activations_with_missing: int
    blocked_due_to_missing: int
    fallback_due_to_missing: int
    missing_data_counts: Mapping[str, int]
    missing_data_by_regime: Mapping[MarketRegime, Mapping[str, int]]


@dataclass(frozen=True)
class ActivationDecisionStats:
    """Podsumowanie kandydatów decyzji tworzonych podczas aktywacji."""

    total: int
    activations_with_candidates: int
    candidate_count: int
    strategy_counts: Mapping[str, int]
    strategy_counts_by_regime: Mapping[MarketRegime, Mapping[str, int]]
    mean_expected_return_bps: float | None
    mean_expected_probability: float | None
    expected_value_sum_bps: float
    expected_value_by_strategy: Mapping[str, float]
    total_notional: float
    notional_by_strategy: Mapping[str, float]


class StrategyRegimeWorkflow:
    """Orkiestruje presety strategii w zależności od reżimu rynku."""

    def __init__(
        self,
        *,
        wizard: StrategyPresetWizard | None = None,
        classifier: MarketRegimeClassifier | None = None,
        history: RegimeHistory | None = None,
        schedule_windows: Sequence[ScheduleWindow] | None = None,
        decision_engine: DecisionOrchestrator | None = None,
        logger: logging.Logger | None = None,
        activation_history_limit: int | None = 50,
    ) -> None:
        self._wizard = wizard or StrategyPresetWizard()
        self._classifier = classifier or MarketRegimeClassifier()
        thresholds_loader = getattr(self._classifier, "thresholds_loader", None)
        if history is None:
            history = RegimeHistory(thresholds_loader=thresholds_loader)
            snapshot_loader = getattr(self._classifier, "thresholds_snapshot", None)
            if callable(snapshot_loader):
                try:
                    history.reload_thresholds(thresholds=snapshot_loader())
                except Exception:  # pragma: no cover - defensywne
                    pass
        self._history = history
        if schedule_windows:
            self._schedule = tuple(schedule_windows)
        else:
            self._schedule = (
                ScheduleWindow(start=time(0, 0), end=time(0, 0), allow_trading=True),
            )
        self._decision_engine = decision_engine
        self._logger = logger or _LOGGER
        self._presets: MutableMapping[MarketRegime, _RegisteredPreset] = {}
        self._fallback: _RegisteredPreset | None = None
        self._last_activation: RegimePresetActivation | None = None
        history_limit = None
        if activation_history_limit is not None:
            try:
                parsed_limit = int(activation_history_limit)
            except (TypeError, ValueError):
                parsed_limit = 0
            if parsed_limit > 0:
                history_limit = parsed_limit
        self._activation_history: deque[RegimePresetActivation] = deque(maxlen=history_limit)

    @property
    def history(self) -> RegimeHistory:
        return self._history

    @property
    def last_activation(self) -> RegimePresetActivation | None:
        return self._last_activation

    def register_preset(
        self,
        regime: MarketRegime | str,
        *,
        name: str,
        entries: Sequence[Mapping[str, object]],
        signing_key: bytes,
        key_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PresetVersionInfo:
        registered = self._build_registered_preset(
            regime=_normalize_regime(regime),
            name=name,
            entries=entries,
            signing_key=signing_key,
            key_id=key_id,
            metadata=metadata,
        )
        assert registered.regime is not None
        self._presets[registered.regime] = registered
        return registered.version

    def register_emergency_preset(
        self,
        *,
        name: str,
        entries: Sequence[Mapping[str, object]],
        signing_key: bytes,
        key_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PresetVersionInfo:
        registered = self._build_registered_preset(
            regime=None,
            name=name,
            entries=entries,
            signing_key=signing_key,
            key_id=key_id,
            metadata=metadata,
        )
        self._fallback = registered
        return registered.version

    def activate(
        self,
        market_data: pd.DataFrame,
        *,
        available_data: Iterable[str] = (),
        symbol: str | None = None,
        now: datetime | None = None,
    ) -> RegimePresetActivation:
        if market_data is None or market_data.empty:
            raise ValueError("market_data must contain OHLCV history")
        now = now or datetime.now(timezone.utc)

        assessment = self._classifier.assess(market_data, symbol=symbol)
        self._history.update(assessment)
        summary = self._history.summarise()

        available = {
            str(item).strip().lower() for item in available_data if str(item).strip()
        }
        allowed = self._is_within_schedule(now)
        missing_data: tuple[str, ...] = ()
        blocked_reason: str | None = None
        license_issues: tuple[str, ...] = ()

        candidate = self._presets.get(assessment.regime)
        if candidate is None:
            blocked_reason = "no_preset"

        if not allowed:
            blocked_reason = "schedule_blocked"
            candidate = None

        if candidate is not None:
            missing = self._missing_data(candidate.required_data, available)
            if missing:
                missing_data = missing
                if blocked_reason is None:
                    blocked_reason = "missing_data"
                candidate = None

        if candidate is not None:
            license_issues = self._enforce_license(candidate)
            if license_issues:
                blocked_reason = "license_blocked"
                candidate = None

        used_fallback = False
        if candidate is None:
            candidate, missing_data = self._resolve_fallback(available, missing_data)
            used_fallback = True

        decision_candidates = self._build_candidates(candidate.preset, candidate.version)
        recommendation = self._fetch_recommendation(assessment.regime)

        activation = RegimePresetActivation(
            regime=assessment.regime,
            assessment=assessment,
            summary=summary,
            preset=candidate.preset,
            version=candidate.version,
            decision_candidates=decision_candidates,
            activated_at=now,
            preset_regime=candidate.regime,
            used_fallback=used_fallback,
            missing_data=missing_data,
            blocked_reason=blocked_reason,
            recommendation=recommendation,
            license_issues=license_issues,
        )
        self._last_activation = activation
        self._activation_history.append(activation)
        return activation

    def activation_history(self) -> tuple[RegimePresetActivation, ...]:
        """Zwraca historię ostatnich aktywacji w kolejności chronologicznej."""

        return tuple(self._activation_history)

    def activation_history_stats(self, *, limit: int | None = None) -> ActivationHistoryStats:
        """Podsumowuje najnowsze aktywacje z historii."""

        entries = self._history_slice(limit)

        total = len(entries)
        regime_counter: Counter[MarketRegime] = Counter()
        preset_regime_counter: Counter[MarketRegime | None] = Counter()
        blocked_counter: Counter[str] = Counter()
        missing_counter: Counter[str] = Counter()
        license_counter: Counter[str] = Counter()
        fallback_count = 0

        for activation in entries:
            regime_counter[activation.regime] += 1
            preset_regime_counter[activation.preset_regime] += 1
            if activation.used_fallback:
                fallback_count += 1
            if activation.blocked_reason:
                blocked_counter[activation.blocked_reason] += 1
            for missing in activation.missing_data:
                normalized = str(missing).strip()
                if normalized:
                    missing_counter[normalized] += 1
            for issue in activation.license_issues:
                normalized = str(issue).strip()
                if normalized:
                    license_counter[normalized] += 1

        def _freeze_regime(counter: Counter[MarketRegime]) -> Mapping[MarketRegime, int]:
            if not counter:
                return MappingProxyType({})
            return MappingProxyType(dict(counter))

        def _freeze_preset(counter: Counter[MarketRegime | None]) -> Mapping[MarketRegime | None, int]:
            if not counter:
                return MappingProxyType({})
            return MappingProxyType(dict(counter))

        def _freeze_text(counter: Counter[str]) -> Mapping[str, int]:
            if not counter:
                return MappingProxyType({})
            return MappingProxyType(dict(counter))

        return ActivationHistoryStats(
            total=total,
            regime_counts=_freeze_regime(regime_counter),
            preset_regime_counts=_freeze_preset(preset_regime_counter),
            fallback_count=fallback_count,
            blocked_reasons=_freeze_text(blocked_counter),
            missing_data=_freeze_text(missing_counter),
            license_issue_counts=_freeze_text(license_counter),
        )

    def activation_transition_stats(self, *, limit: int | None = None) -> ActivationTransitionStats:
        """Buduje macierz przejść na podstawie historii aktywacji."""

        entries = self._history_slice(limit)
        if len(entries) < 2:
            return ActivationTransitionStats(
                total=0,
                regime_transitions=MappingProxyType({}),
                preset_regime_transitions=MappingProxyType({}),
                blocked_transitions=MappingProxyType({}),
                fallback_transitions=0,
            )

        regime_counter: Counter[tuple[MarketRegime, MarketRegime]] = Counter()
        preset_regime_counter: Counter[tuple[MarketRegime | None, MarketRegime | None]] = Counter()
        blocked_counter: Counter[tuple[str | None, str | None]] = Counter()
        fallback_transitions = 0

        for previous, current in zip(entries, entries[1:]):
            regime_counter[(previous.regime, current.regime)] += 1
            preset_regime_counter[(previous.preset_regime, current.preset_regime)] += 1
            blocked_counter[(previous.blocked_reason, current.blocked_reason)] += 1
            if current.used_fallback:
                fallback_transitions += 1

        def _freeze(counter: Counter) -> Mapping:
            if not counter:
                return MappingProxyType({})
            return MappingProxyType(dict(counter))

        total = sum(regime_counter.values())
        return ActivationTransitionStats(
            total=total,
            regime_transitions=_freeze(regime_counter),
            preset_regime_transitions=_freeze(preset_regime_counter),
            blocked_transitions=_freeze(blocked_counter),
            fallback_transitions=fallback_transitions,
        )

    def activation_cadence_stats(self, *, limit: int | None = None) -> ActivationCadenceStats:
        """Agreguje odstępy czasowe pomiędzy kolejnymi aktywacjami."""

        entries = self._history_slice(limit)
        total = len(entries)
        if total < 2:
            return ActivationCadenceStats(
                total=total,
                intervals=0,
                min_interval=None,
                max_interval=None,
                mean_interval=None,
                median_interval=None,
                last_interval=None,
            )

        deltas: list[timedelta] = []
        for previous, current in zip(entries, entries[1:]):
            try:
                delta = current.activated_at - previous.activated_at
            except Exception:  # pragma: no cover - defensywne
                continue
            if not isinstance(delta, timedelta):
                continue
            deltas.append(delta)

        if not deltas:
            return ActivationCadenceStats(
                total=total,
                intervals=0,
                min_interval=None,
                max_interval=None,
                mean_interval=None,
                median_interval=None,
                last_interval=None,
            )

        intervals = len(deltas)
        min_interval = min(deltas)
        max_interval = max(deltas)
        total_delta = sum(deltas, timedelta())
        mean_interval = total_delta / intervals
        sorted_deltas = sorted(deltas)
        if intervals % 2:
            median_interval = sorted_deltas[intervals // 2]
        else:
            lower = sorted_deltas[(intervals // 2) - 1]
            upper = sorted_deltas[intervals // 2]
            median_interval = (lower + upper) / 2
        last_interval = deltas[-1]

        return ActivationCadenceStats(
            total=total,
            intervals=intervals,
            min_interval=min_interval,
            max_interval=max_interval,
            mean_interval=mean_interval,
            median_interval=median_interval,
            last_interval=last_interval,
        )

    def activation_uptime_stats(
        self,
        *,
        limit: int | None = None,
        until: datetime | None = None,
    ) -> ActivationUptimeStats:
        """Oblicza czas spędzony w poszczególnych reżimach i presetach."""

        entries = self._history_slice(limit)
        total = len(entries)
        if not entries:
            return ActivationUptimeStats(
                total=0,
                duration=timedelta(),
                regime_uptime=MappingProxyType({}),
                preset_uptime=MappingProxyType({}),
                fallback_uptime=timedelta(),
            )

        if until is None:
            until = datetime.now(timezone.utc)
        elif until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)

        regime_totals: MutableMapping[MarketRegime, timedelta] = {}
        preset_totals: MutableMapping[MarketRegime | None, timedelta] = {}
        fallback_total = timedelta()
        total_duration = timedelta()

        def _accumulate(activation: RegimePresetActivation, delta: timedelta) -> None:
            nonlocal fallback_total, total_duration
            if delta.total_seconds() <= 0:
                return
            total_duration += delta
            regime_totals[activation.regime] = (
                regime_totals.get(activation.regime, timedelta()) + delta
            )
            preset_totals[activation.preset_regime] = (
                preset_totals.get(activation.preset_regime, timedelta()) + delta
            )
            if activation.used_fallback:
                fallback_total += delta

        for previous, current in zip(entries, entries[1:]):
            try:
                delta = current.activated_at - previous.activated_at
            except Exception:  # pragma: no cover - defensywne
                continue
            if not isinstance(delta, timedelta):
                continue
            _accumulate(previous, delta)

        last = entries[-1]
        try:
            tail = until - last.activated_at
        except Exception:  # pragma: no cover - defensywne
            tail = None
        if isinstance(tail, timedelta):
            _accumulate(last, tail)

        return ActivationUptimeStats(
            total=total,
            duration=total_duration,
            regime_uptime=MappingProxyType(dict(regime_totals)),
            preset_uptime=MappingProxyType(dict(preset_totals)),
            fallback_uptime=fallback_total,
        )

    def activation_reliability_stats(
        self,
        *,
        limit: int | None = None,
    ) -> ActivationReliabilityStats:
        """Podsumowuje skuteczność aktywacji na podstawie historii."""

        entries = self._history_slice(limit)
        total = len(entries)
        if total == 0:
            return ActivationReliabilityStats(
                total=0,
                completed=0,
                fallback_count=0,
                blocked_count=0,
                completed_ratio=0.0,
                fallback_ratio=0.0,
                blocked_ratio=0.0,
            )

        completed = 0
        fallback_count = 0
        blocked_count = 0

        for activation in entries:
            if activation.used_fallback:
                fallback_count += 1
            if activation.blocked_reason:
                blocked_count += 1
            if not activation.used_fallback and activation.blocked_reason is None:
                completed += 1

        def _ratio(value: int) -> float:
            try:
                return value / total
            except Exception:  # pragma: no cover - defensywne
                return 0.0

        return ActivationReliabilityStats(
            total=total,
            completed=completed,
            fallback_count=fallback_count,
            blocked_count=blocked_count,
            completed_ratio=_ratio(completed),
            fallback_ratio=_ratio(fallback_count),
            blocked_ratio=_ratio(blocked_count),
        )

    def activation_outcome_stats(
        self,
        *,
        limit: int | None = None,
    ) -> ActivationOutcomeStats:
        """Grupuje wyniki aktywacji według reżimów i presetów."""

        entries = self._history_slice(limit)
        total = len(entries)
        if total == 0:
            empty_mapping: Mapping[MarketRegime, int] = MappingProxyType({})
            empty_preset_mapping: Mapping[MarketRegime | None, int] = MappingProxyType({})
            return ActivationOutcomeStats(
                total=0,
                completed_total=0,
                fallback_total=0,
                blocked_total=0,
                regime_completed=empty_mapping,
                regime_fallback=empty_mapping,
                regime_blocked=empty_mapping,
                preset_completed=empty_preset_mapping,
                preset_fallback=empty_preset_mapping,
                preset_blocked=empty_preset_mapping,
            )

        regime_completed: MutableMapping[MarketRegime, int] = {}
        regime_fallback: MutableMapping[MarketRegime, int] = {}
        regime_blocked: MutableMapping[MarketRegime, int] = {}
        preset_completed: MutableMapping[MarketRegime | None, int] = {}
        preset_fallback: MutableMapping[MarketRegime | None, int] = {}
        preset_blocked: MutableMapping[MarketRegime | None, int] = {}

        completed_total = 0
        fallback_total = 0
        blocked_total = 0

        for activation in entries:
            regime = activation.regime
            preset_regime = activation.preset_regime
            is_fallback = activation.used_fallback
            is_blocked = activation.blocked_reason is not None
            is_completed = not is_fallback and not is_blocked

            if is_completed:
                completed_total += 1
                regime_completed[regime] = regime_completed.get(regime, 0) + 1
                preset_completed[preset_regime] = preset_completed.get(preset_regime, 0) + 1

            if is_fallback:
                fallback_total += 1
                regime_fallback[regime] = regime_fallback.get(regime, 0) + 1
                preset_fallback[preset_regime] = preset_fallback.get(preset_regime, 0) + 1

            if is_blocked:
                blocked_total += 1
                regime_blocked[regime] = regime_blocked.get(regime, 0) + 1
                preset_blocked[preset_regime] = preset_blocked.get(preset_regime, 0) + 1

        return ActivationOutcomeStats(
            total=total,
            completed_total=completed_total,
            fallback_total=fallback_total,
            blocked_total=blocked_total,
            regime_completed=MappingProxyType(dict(regime_completed)),
            regime_fallback=MappingProxyType(dict(regime_fallback)),
            regime_blocked=MappingProxyType(dict(regime_blocked)),
            preset_completed=MappingProxyType(dict(preset_completed)),
            preset_fallback=MappingProxyType(dict(preset_fallback)),
            preset_blocked=MappingProxyType(dict(preset_blocked)),
        )

    def activation_preset_stats(
        self,
        *,
        limit: int | None = None,
    ) -> ActivationPresetStats:
        """Agreguje historię według nazw i wersji presetów."""

        entries = self._history_slice(limit)
        total = len(entries)
        if total == 0:
            empty_text: Mapping[str, int] = MappingProxyType({})
            empty_regime: Mapping[MarketRegime, int] = MappingProxyType({})
            empty_nested_str: Mapping[str, Mapping[str, int]] = MappingProxyType({})
            empty_nested_regime: Mapping[MarketRegime, Mapping[str, int]] = MappingProxyType({})
            return ActivationPresetStats(
                total=0,
                preset_usage=empty_text,
                version_usage=empty_text,
                versions_by_preset=empty_nested_str,
                regime_preset_usage=empty_nested_regime,
                fallback_by_preset=empty_text,
                fallback_by_version=empty_text,
                fallback_by_regime=empty_regime,
                blocked_by_preset=empty_text,
                blocked_by_version=empty_text,
                blocked_by_regime=empty_regime,
            )

        preset_usage: MutableMapping[str, int] = {}
        version_usage: MutableMapping[str, int] = {}
        versions_by_preset: MutableMapping[str, MutableMapping[str, int]] = {}
        regime_preset_usage: MutableMapping[MarketRegime, MutableMapping[str, int]] = {}
        fallback_by_preset: MutableMapping[str, int] = {}
        fallback_by_version: MutableMapping[str, int] = {}
        fallback_by_regime: MutableMapping[MarketRegime, int] = {}
        blocked_by_preset: MutableMapping[str, int] = {}
        blocked_by_version: MutableMapping[str, int] = {}
        blocked_by_regime: MutableMapping[MarketRegime, int] = {}

        def _preset_name(activation: RegimePresetActivation) -> str:
            candidate = activation.preset.get("name")
            if not candidate:
                candidate = activation.version.metadata.get("name")
            name = str(candidate or activation.version.hash).strip()
            return name or activation.version.hash

        for activation in entries:
            name = _preset_name(activation)
            version_hash = activation.version.hash
            regime = activation.regime

            preset_usage[name] = preset_usage.get(name, 0) + 1
            version_usage[version_hash] = version_usage.get(version_hash, 0) + 1

            preset_versions = versions_by_preset.setdefault(name, {})
            preset_versions[version_hash] = preset_versions.get(version_hash, 0) + 1

            regime_usage = regime_preset_usage.setdefault(regime, {})
            regime_usage[name] = regime_usage.get(name, 0) + 1

            if activation.used_fallback:
                fallback_by_preset[name] = fallback_by_preset.get(name, 0) + 1
                fallback_by_version[version_hash] = fallback_by_version.get(version_hash, 0) + 1
                fallback_by_regime[regime] = fallback_by_regime.get(regime, 0) + 1

            if activation.blocked_reason:
                blocked_by_preset[name] = blocked_by_preset.get(name, 0) + 1
                blocked_by_version[version_hash] = blocked_by_version.get(version_hash, 0) + 1
                blocked_by_regime[regime] = blocked_by_regime.get(regime, 0) + 1

        def _freeze_nested(
            source: MutableMapping[str, MutableMapping[str, int]]
        ) -> Mapping[str, Mapping[str, int]]:
            if not source:
                return MappingProxyType({})
            frozen: MutableMapping[str, Mapping[str, int]] = {}
            for key, value in source.items():
                if not value:
                    continue
                frozen[key] = MappingProxyType(dict(value))
            return MappingProxyType(dict(frozen)) if frozen else MappingProxyType({})

        def _freeze_regime_nested(
            source: MutableMapping[MarketRegime, MutableMapping[str, int]]
        ) -> Mapping[MarketRegime, Mapping[str, int]]:
            if not source:
                return MappingProxyType({})
            frozen: MutableMapping[MarketRegime, Mapping[str, int]] = {}
            for key, value in source.items():
                if not value:
                    continue
                frozen[key] = MappingProxyType(dict(value))
            return MappingProxyType(dict(frozen)) if frozen else MappingProxyType({})

        def _freeze_counts(source: MutableMapping[str, int]) -> Mapping[str, int]:
            return MappingProxyType(dict(source)) if source else MappingProxyType({})

        def _freeze_regime_counts(
            source: MutableMapping[MarketRegime, int]
        ) -> Mapping[MarketRegime, int]:
            return MappingProxyType(dict(source)) if source else MappingProxyType({})

        return ActivationPresetStats(
            total=total,
            preset_usage=_freeze_counts(preset_usage),
            version_usage=_freeze_counts(version_usage),
            versions_by_preset=_freeze_nested(versions_by_preset),
            regime_preset_usage=_freeze_regime_nested(regime_preset_usage),
            fallback_by_preset=_freeze_counts(fallback_by_preset),
            fallback_by_version=_freeze_counts(fallback_by_version),
            fallback_by_regime=_freeze_regime_counts(fallback_by_regime),
            blocked_by_preset=_freeze_counts(blocked_by_preset),
            blocked_by_version=_freeze_counts(blocked_by_version),
            blocked_by_regime=_freeze_regime_counts(blocked_by_regime),
        )

    def activation_license_stats(
        self,
        *,
        limit: int | None = None,
    ) -> ActivationLicenseStats:
        """Agreguje historię problemów licencyjnych."""

        entries = self._history_slice(limit)
        total = len(entries)

        issue_counts: MutableMapping[str, int] = {}
        issues_by_regime: MutableMapping[MarketRegime, MutableMapping[str, int]] = {}
        issues_by_preset: MutableMapping[str, MutableMapping[str, int]] = {}
        regimes_with_issues: MutableMapping[MarketRegime, int] = {}
        presets_with_issues: MutableMapping[str, int] = {}

        activations_with_issues = 0
        blocked_by_license = 0
        fallback_due_to_license = 0

        def _preset_name(activation: RegimePresetActivation) -> str:
            candidate = activation.preset.get("name")
            if not candidate:
                candidate = activation.version.metadata.get("name")
            name = str(candidate or activation.version.hash).strip()
            return name or activation.version.hash

        for activation in entries:
            if not activation.license_issues:
                continue

            activations_with_issues += 1
            regime = activation.regime
            name = _preset_name(activation)

            regimes_with_issues[regime] = regimes_with_issues.get(regime, 0) + 1
            presets_with_issues[name] = presets_with_issues.get(name, 0) + 1

            regime_bucket = issues_by_regime.setdefault(regime, {})
            preset_bucket = issues_by_preset.setdefault(name, {})

            for issue in activation.license_issues:
                normalized = str(issue).strip()
                if not normalized:
                    continue
                issue_counts[normalized] = issue_counts.get(normalized, 0) + 1
                regime_bucket[normalized] = regime_bucket.get(normalized, 0) + 1
                preset_bucket[normalized] = preset_bucket.get(normalized, 0) + 1

            if activation.blocked_reason == "license_blocked":
                blocked_by_license += 1
                if activation.used_fallback:
                    fallback_due_to_license += 1

        def _freeze_nested_regime(
            source: MutableMapping[MarketRegime, MutableMapping[str, int]]
        ) -> Mapping[MarketRegime, Mapping[str, int]]:
            if not source:
                return MappingProxyType({})
            frozen: MutableMapping[MarketRegime, Mapping[str, int]] = {}
            for regime, issues in source.items():
                if not issues:
                    continue
                frozen[regime] = MappingProxyType(dict(issues))
            return MappingProxyType(dict(frozen)) if frozen else MappingProxyType({})

        def _freeze_nested_preset(
            source: MutableMapping[str, MutableMapping[str, int]]
        ) -> Mapping[str, Mapping[str, int]]:
            if not source:
                return MappingProxyType({})
            frozen: MutableMapping[str, Mapping[str, int]] = {}
            for preset, issues in source.items():
                if not issues:
                    continue
                frozen[preset] = MappingProxyType(dict(issues))
            return MappingProxyType(dict(frozen)) if frozen else MappingProxyType({})

        def _freeze_counts(source: MutableMapping[str, int]) -> Mapping[str, int]:
            return MappingProxyType(dict(source)) if source else MappingProxyType({})

        def _freeze_regime_counts(
            source: MutableMapping[MarketRegime, int]
        ) -> Mapping[MarketRegime, int]:
            return MappingProxyType(dict(source)) if source else MappingProxyType({})

        return ActivationLicenseStats(
            total=total,
            activations_with_issues=activations_with_issues,
            blocked_by_license=blocked_by_license,
            fallback_due_to_license=fallback_due_to_license,
            issue_counts=_freeze_counts(issue_counts),
            issues_by_regime=_freeze_nested_regime(issues_by_regime),
            issues_by_preset=_freeze_nested_preset(issues_by_preset),
            regimes_with_issues=_freeze_regime_counts(regimes_with_issues),
            presets_with_issues=_freeze_counts(presets_with_issues),
        )

    def activation_data_stats(
        self,
        *,
        limit: int | None = None,
    ) -> ActivationDataStats:
        """Agreguje braki danych rynkowych z historii aktywacji."""

        entries = self._history_slice(limit)
        total = len(entries)

        missing_counts: MutableMapping[str, int] = {}
        missing_by_regime: MutableMapping[MarketRegime, MutableMapping[str, int]] = {}

        activations_with_missing = 0
        blocked_due_to_missing = 0
        fallback_due_to_missing = 0

        for activation in entries:
            if not activation.missing_data:
                continue

            activations_with_missing += 1
            regime = activation.regime
            regime_bucket = missing_by_regime.setdefault(regime, {})

            if activation.blocked_reason == "missing_data":
                blocked_due_to_missing += 1
                if activation.used_fallback:
                    fallback_due_to_missing += 1

            for item in activation.missing_data:
                normalized = str(item).strip()
                if not normalized:
                    continue
                missing_counts[normalized] = missing_counts.get(normalized, 0) + 1
                regime_bucket[normalized] = regime_bucket.get(normalized, 0) + 1

        def _freeze_missing(source: MutableMapping[str, int]) -> Mapping[str, int]:
            return MappingProxyType(dict(source)) if source else MappingProxyType({})

        def _freeze_regime_missing(
            source: MutableMapping[MarketRegime, MutableMapping[str, int]]
        ) -> Mapping[MarketRegime, Mapping[str, int]]:
            if not source:
                return MappingProxyType({})
            frozen: MutableMapping[MarketRegime, Mapping[str, int]] = {}
            for regime, items in source.items():
                if not items:
                    continue
                frozen[regime] = MappingProxyType(dict(items))
            return MappingProxyType(dict(frozen)) if frozen else MappingProxyType({})

        return ActivationDataStats(
            total=total,
            activations_with_missing=activations_with_missing,
            blocked_due_to_missing=blocked_due_to_missing,
            fallback_due_to_missing=fallback_due_to_missing,
            missing_data_counts=_freeze_missing(missing_counts),
            missing_data_by_regime=_freeze_regime_missing(missing_by_regime),
        )

    def activation_decision_stats(
        self,
        *,
        limit: int | None = None,
    ) -> ActivationDecisionStats:
        """Agreguje kandydatów decyzji generowanych z presetów."""

        entries = self._history_slice(limit)
        total = len(entries)
        if total == 0:
            empty_counts: Mapping[str, int] = MappingProxyType({})
            empty_float: Mapping[str, float] = MappingProxyType({})
            empty_nested: Mapping[MarketRegime, Mapping[str, int]] = MappingProxyType({})
            return ActivationDecisionStats(
                total=0,
                activations_with_candidates=0,
                candidate_count=0,
                strategy_counts=empty_counts,
                strategy_counts_by_regime=empty_nested,
                mean_expected_return_bps=None,
                mean_expected_probability=None,
                expected_value_sum_bps=0.0,
                expected_value_by_strategy=empty_float,
                total_notional=0.0,
                notional_by_strategy=empty_float,
            )

        activations_with_candidates = 0
        candidate_count = 0
        strategy_counts: MutableMapping[str, int] = {}
        strategies_by_regime: MutableMapping[MarketRegime, MutableMapping[str, int]] = {}
        expected_return_sum = 0.0
        expected_return_count = 0
        probability_sum = 0.0
        probability_count = 0
        expected_value_sum = 0.0
        expected_value_by_strategy: MutableMapping[str, float] = {}
        total_notional = 0.0
        notional_by_strategy: MutableMapping[str, float] = {}

        for activation in entries:
            candidates = activation.decision_candidates
            if not candidates:
                continue

            activations_with_candidates += 1
            regime_bucket = strategies_by_regime.setdefault(activation.regime, {})

            for candidate in candidates:
                candidate_count += 1
                strategy_name = str(candidate.strategy).strip() or "unknown"
                strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
                regime_bucket[strategy_name] = regime_bucket.get(strategy_name, 0) + 1

                expected_return = float(candidate.expected_return_bps)
                if not math.isnan(expected_return):
                    expected_return_sum += expected_return
                    expected_return_count += 1

                probability = float(candidate.expected_probability)
                if not math.isnan(probability):
                    probability_sum += probability
                    probability_count += 1

                expected_value = expected_return * probability
                if not math.isnan(expected_value):
                    expected_value_sum += expected_value
                    expected_value_by_strategy[strategy_name] = (
                        expected_value_by_strategy.get(strategy_name, 0.0) + expected_value
                    )

                notional = float(candidate.notional)
                if not math.isnan(notional):
                    notional = max(0.0, notional)
                    total_notional += notional
                    notional_by_strategy[strategy_name] = (
                        notional_by_strategy.get(strategy_name, 0.0) + notional
                    )

        def _freeze_counts(mapping: MutableMapping[str, int]) -> Mapping[str, int]:
            return MappingProxyType(dict(mapping)) if mapping else MappingProxyType({})

        def _freeze_float(mapping: MutableMapping[str, float]) -> Mapping[str, float]:
            return MappingProxyType(dict(mapping)) if mapping else MappingProxyType({})

        def _freeze_regime(
            mapping: MutableMapping[MarketRegime, MutableMapping[str, int]]
        ) -> Mapping[MarketRegime, Mapping[str, int]]:
            if not mapping:
                return MappingProxyType({})
            frozen: MutableMapping[MarketRegime, Mapping[str, int]] = {}
            for regime, items in mapping.items():
                if not items:
                    continue
                frozen[regime] = MappingProxyType(dict(items))
            return MappingProxyType(dict(frozen)) if frozen else MappingProxyType({})

        def _mean(total_sum: float, count: int) -> float | None:
            return total_sum / count if count else None

        return ActivationDecisionStats(
            total=total,
            activations_with_candidates=activations_with_candidates,
            candidate_count=candidate_count,
            strategy_counts=_freeze_counts(strategy_counts),
            strategy_counts_by_regime=_freeze_regime(strategies_by_regime),
            mean_expected_return_bps=_mean(expected_return_sum, expected_return_count),
            mean_expected_probability=_mean(probability_sum, probability_count),
            expected_value_sum_bps=expected_value_sum,
            expected_value_by_strategy=_freeze_float(expected_value_by_strategy),
            total_notional=total_notional,
            notional_by_strategy=_freeze_float(notional_by_strategy),
        )

    def activation_assessment_stats(
        self,
        *,
        limit: int | None = None,
    ) -> ActivationAssessmentStats:
        """Agreguje statystyki ocen reżimu w historii aktywacji."""

        entries = self._history_slice(limit)
        total = len(entries)
        if total == 0:
            empty_metrics: Mapping[str, float] = MappingProxyType({})
            empty_regimes: Mapping[MarketRegime, int] = MappingProxyType({})
            return ActivationAssessmentStats(
                total=0,
                regime_counts=empty_regimes,
                min_confidence=None,
                max_confidence=None,
                mean_confidence=None,
                min_risk_score=None,
                max_risk_score=None,
                mean_risk_score=None,
                metrics_mean=empty_metrics,
                metrics_min=empty_metrics,
                metrics_max=empty_metrics,
            )

        regime_counts: MutableMapping[MarketRegime, int] = {}
        confidence_sum = 0.0
        confidence_count = 0
        min_confidence: float | None = None
        max_confidence: float | None = None
        risk_sum = 0.0
        risk_count = 0
        min_risk: float | None = None
        max_risk: float | None = None
        metrics_sum: MutableMapping[str, float] = {}
        metrics_count: MutableMapping[str, int] = {}
        metrics_min: MutableMapping[str, float] = {}
        metrics_max: MutableMapping[str, float] = {}

        for activation in entries:
            regime_counts[activation.regime] = regime_counts.get(activation.regime, 0) + 1

            assessment = activation.assessment
            try:
                confidence = float(assessment.confidence)
            except (TypeError, ValueError):
                confidence = math.nan
            if not math.isnan(confidence):
                confidence_sum += confidence
                confidence_count += 1
                min_confidence = confidence if min_confidence is None else min(min_confidence, confidence)
                max_confidence = confidence if max_confidence is None else max(max_confidence, confidence)

            try:
                risk_score = float(assessment.risk_score)
            except (TypeError, ValueError):
                risk_score = math.nan
            if not math.isnan(risk_score):
                risk_sum += risk_score
                risk_count += 1
                min_risk = risk_score if min_risk is None else min(min_risk, risk_score)
                max_risk = risk_score if max_risk is None else max(max_risk, risk_score)

            metrics = assessment.metrics if isinstance(assessment.metrics, Mapping) else {}
            for name, value in metrics.items():
                key = str(name)
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(numeric):
                    continue
                metrics_sum[key] = metrics_sum.get(key, 0.0) + numeric
                metrics_count[key] = metrics_count.get(key, 0) + 1
                metrics_min[key] = numeric if key not in metrics_min else min(metrics_min[key], numeric)
                metrics_max[key] = numeric if key not in metrics_max else max(metrics_max[key], numeric)

        def _mean(value_sum: float, count: int) -> float | None:
            return value_sum / count if count else None

        metrics_mean: MutableMapping[str, float] = {}
        for key, count in metrics_count.items():
            if count:
                metrics_mean[key] = metrics_sum[key] / count

        def _freeze(values: MutableMapping[str, float]) -> Mapping[str, float]:
            return MappingProxyType(dict(values)) if values else MappingProxyType({})

        return ActivationAssessmentStats(
            total=total,
            regime_counts=MappingProxyType(dict(regime_counts)),
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            mean_confidence=_mean(confidence_sum, confidence_count),
            min_risk_score=min_risk,
            max_risk_score=max_risk,
            mean_risk_score=_mean(risk_sum, risk_count),
            metrics_mean=_freeze(metrics_mean),
            metrics_min=_freeze(metrics_min),
            metrics_max=_freeze(metrics_max),
        )

    def activation_blocker_stats(
        self,
        *,
        limit: int | None = None,
    ) -> ActivationBlockerStats:
        """Agreguje historię blokad aktywacji."""

        blocked_entries = [
            activation
            for activation in self._history_slice(limit)
            if activation.blocked_reason
        ]
        total = len(blocked_entries)

        if total == 0:
            empty_regimes: Mapping[MarketRegime, int] = MappingProxyType({})
            empty_text: Mapping[str, int] = MappingProxyType({})
            empty_nested: Mapping[MarketRegime, Mapping[str, int]] = MappingProxyType({})
            return ActivationBlockerStats(
                total=0,
                regime_blocked=empty_regimes,
                reason_counts=empty_text,
                missing_data=empty_text,
                missing_data_by_regime=empty_nested,
                license_issues=empty_text,
                license_issues_by_regime=empty_nested,
            )

        regime_counts: MutableMapping[MarketRegime, int] = {}
        reason_counts: MutableMapping[str, int] = {}
        missing_counts: MutableMapping[str, int] = {}
        license_counts: MutableMapping[str, int] = {}
        missing_by_regime: MutableMapping[MarketRegime, MutableMapping[str, int]] = {}
        license_by_regime: MutableMapping[MarketRegime, MutableMapping[str, int]] = {}

        for activation in blocked_entries:
            regime = activation.regime
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            reason = str(activation.blocked_reason)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

            if activation.missing_data:
                regime_missing = missing_by_regime.setdefault(regime, {})
                for item in activation.missing_data:
                    normalized = str(item).strip()
                    if not normalized:
                        continue
                    missing_counts[normalized] = missing_counts.get(normalized, 0) + 1
                    regime_missing[normalized] = regime_missing.get(normalized, 0) + 1

            if activation.license_issues:
                regime_issues = license_by_regime.setdefault(regime, {})
                for issue in activation.license_issues:
                    normalized = str(issue).strip()
                    if not normalized:
                        continue
                    license_counts[normalized] = license_counts.get(normalized, 0) + 1
                    regime_issues[normalized] = regime_issues.get(normalized, 0) + 1

        def _freeze_nested(
            source: MutableMapping[MarketRegime, MutableMapping[str, int]]
        ) -> Mapping[MarketRegime, Mapping[str, int]]:
            if not source:
                return MappingProxyType({})
            frozen: MutableMapping[MarketRegime, Mapping[str, int]] = {}
            for regime, items in source.items():
                if not items:
                    continue
                frozen[regime] = MappingProxyType(dict(items))
            return MappingProxyType(dict(frozen)) if frozen else MappingProxyType({})

        return ActivationBlockerStats(
            total=total,
            regime_blocked=MappingProxyType(dict(regime_counts)),
            reason_counts=MappingProxyType(dict(reason_counts)),
            missing_data=MappingProxyType(dict(missing_counts)),
            missing_data_by_regime=_freeze_nested(missing_by_regime),
            license_issues=MappingProxyType(dict(license_counts)),
            license_issues_by_regime=_freeze_nested(license_by_regime),
        )

    def activation_history_frame(self, *, limit: int | None = None) -> pd.DataFrame:
        """Zwraca historię aktywacji w formie ramki danych pandas."""

        entries = self._history_slice(limit)
        columns = [
            "activated_at",
            "regime",
            "preset_regime",
            "preset_name",
            "preset_version",
            "used_fallback",
            "blocked_reason",
            "missing_data",
            "license_issues",
            "recommendation",
        ]
        if not entries:
            return pd.DataFrame(columns=columns)

        rows: list[Mapping[str, object]] = []
        for activation in entries:
            rows.append(
                {
                    "activated_at": activation.activated_at,
                    "regime": activation.regime,
                    "preset_regime": activation.preset_regime,
                    "preset_name": activation.preset.get("name"),
                    "preset_version": activation.version.hash,
                    "used_fallback": activation.used_fallback,
                    "blocked_reason": activation.blocked_reason,
                    "missing_data": activation.missing_data,
                    "license_issues": activation.license_issues,
                    "recommendation": activation.recommendation,
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def clear_history(self) -> None:
        """Czyści zapamiętaną historię aktywacji."""

        self._activation_history.clear()

    def inspect_presets(
        self,
        available_data: Iterable[str] = (),
        *,
        now: datetime | None = None,
    ) -> tuple[PresetAvailability, ...]:
        """Zwraca raport dostępności wszystkich zarejestrowanych presetów."""

        if now is None:
            now = datetime.now(timezone.utc)
        allowed = self._is_within_schedule(now)
        available = {
            str(item).strip().lower()
            for item in available_data
            if str(item).strip()
        }

        reports: list[PresetAvailability] = []
        for preset in self._presets.values():
            reports.append(
                self._build_availability_report(
                    preset,
                    available=available,
                    schedule_allowed=allowed,
                )
            )
        if self._fallback is not None:
            reports.append(
                self._build_availability_report(
                    self._fallback,
                    available=available,
                    schedule_allowed=allowed,
                )
            )
        return tuple(reports)

    def _build_registered_preset(
        self,
        *,
        regime: MarketRegime | None,
        name: str,
        entries: Sequence[Mapping[str, object]],
        signing_key: bytes,
        key_id: str | None,
        metadata: Mapping[str, object] | None,
    ) -> _RegisteredPreset:
        preset = self._wizard.build_preset(name, entries, metadata=metadata)
        details = self._extract_metadata(preset)
        version = self._build_version_info(preset, details, signing_key=signing_key, key_id=key_id)
        return _RegisteredPreset(
            regime=regime,
            preset=MappingProxyType(preset),
            version=version,
            license_tiers=details["license_tiers"],
            risk_classes=details["risk_classes"],
            required_data=details["required_data"],
            capabilities=details["capabilities"],
            tags=details["tags"],
        )

    def _extract_metadata(self, preset: Mapping[str, object]) -> MutableMapping[str, tuple[str, ...]]:
        strategies = tuple(preset.get("strategies", []))
        strategy_keys = _gather_strings([[entry.get("engine", "")] for entry in strategies])
        strategy_names = _gather_strings([[entry.get("name", entry.get("engine", ""))] for entry in strategies])
        license_tiers = _gather_strings([[entry.get("license_tier", "")] for entry in strategies])
        risk_classes = _gather_strings([entry.get("risk_classes", []) for entry in strategies])
        required_data = _gather_strings([entry.get("required_data", []) for entry in strategies])
        capabilities = _gather_strings([[entry.get("capability", "")] for entry in strategies if entry.get("capability")])
        tags = _gather_strings([entry.get("tags", []) for entry in strategies])
        preset_metadata = preset.get("metadata")
        metadata_payload: MutableMapping[str, object]
        if isinstance(preset_metadata, Mapping):
            metadata_payload = dict(preset_metadata)
        else:
            metadata_payload = {}
        return {
            "strategy_keys": strategy_keys,
            "strategy_names": strategy_names,
            "license_tiers": license_tiers,
            "risk_classes": risk_classes,
            "required_data": required_data,
            "capabilities": capabilities,
            "tags": tags,
            "preset_metadata": tuple(sorted(metadata_payload.items())),
        }

    def _build_version_info(
        self,
        preset: Mapping[str, object],
        metadata: Mapping[str, tuple[str, ...] | tuple[tuple[str, object], ...]],
        *,
        signing_key: bytes,
        key_id: str | None,
    ) -> PresetVersionInfo:
        payload_bytes = canonical_json_bytes(preset)
        digest = hashlib.sha256(payload_bytes).hexdigest()
        signature = build_hmac_signature(preset, key=signing_key, key_id=key_id)
        issued_at = datetime.now(timezone.utc)
        meta_payload: MutableMapping[str, object] = {
            "name": preset.get("name"),
            "strategy_keys": metadata.get("strategy_keys", ()),
            "strategy_names": metadata.get("strategy_names", ()),
            "license_tiers": metadata.get("license_tiers", ()),
            "risk_classes": metadata.get("risk_classes", ()),
            "required_data": metadata.get("required_data", ()),
            "capabilities": metadata.get("capabilities", ()),
            "tags": metadata.get("tags", ()),
        }
        preset_meta = metadata.get("preset_metadata")
        if preset_meta:
            meta_payload["preset_metadata"] = dict(preset_meta)
        return PresetVersionInfo(
            hash=digest,
            signature=MappingProxyType(signature),
            issued_at=issued_at,
            metadata=MappingProxyType(meta_payload),
        )

    def _build_availability_report(
        self,
        preset: _RegisteredPreset,
        *,
        available: set[str],
        schedule_allowed: bool,
    ) -> PresetAvailability:
        missing_data = self._missing_data(preset.required_data, available)
        license_issues = self._enforce_license(preset)
        schedule_blocked = not schedule_allowed
        blocked_reason: str | None = None
        if schedule_blocked:
            blocked_reason = "schedule_blocked"
        elif missing_data:
            blocked_reason = "missing_data"
        elif license_issues:
            blocked_reason = "license_blocked"
        ready = blocked_reason is None
        return PresetAvailability(
            regime=preset.regime,
            version=preset.version,
            ready=ready,
            blocked_reason=blocked_reason,
            missing_data=missing_data,
            license_issues=license_issues,
            schedule_blocked=schedule_blocked,
        )

    def _history_slice(
        self, limit: int | None
    ) -> tuple[RegimePresetActivation, ...]:
        entries = tuple(self._activation_history)
        if limit is None:
            return entries
        try:
            parsed_limit = int(limit)
        except (TypeError, ValueError):  # pragma: no cover - defensywne
            return entries
        if parsed_limit <= 0:
            return entries
        return entries[-parsed_limit:]

    def _is_within_schedule(self, moment: datetime) -> bool:
        for window in self._schedule:
            try:
                if window.allow_trading and window.contains(moment):
                    return True
            except Exception:  # pragma: no cover - defensywne
                continue
        return False

    def _missing_data(self, required: Sequence[str], available: set[str]) -> tuple[str, ...]:
        missing = []
        for item in required:
            normalized = str(item).strip()
            if not normalized:
                continue
            if normalized.lower() not in available:
                missing.append(normalized)
        return tuple(missing)

    def _resolve_fallback(
        self,
        available: set[str],
        current_missing: tuple[str, ...],
    ) -> tuple[_RegisteredPreset, tuple[str, ...]]:
        if self._fallback is None:
            if current_missing:
                raise RuntimeError(
                    "Missing market data prevents preset activation and no fallback preset is registered"
                )
            raise RuntimeError("Decision schedule blocks activation and no fallback preset is registered")
        missing = self._missing_data(self._fallback.required_data, available)
        if missing:
            raise RuntimeError(
                "Fallback preset cannot be activated due to missing market data: "
                + ", ".join(missing)
            )
        self._enforce_license(self._fallback, raise_on_error=True)
        return self._fallback, current_missing

    def _enforce_license(
        self,
        preset: _RegisteredPreset,
        *,
        raise_on_error: bool = False,
    ) -> tuple[str, ...]:
        guard = get_capability_guard()
        if guard is None:
            return ()
        preset_name = str(preset.preset.get("name", "preset"))
        issues: list[str] = []
        last_exc: LicenseCapabilityError | None = None
        current_edition = getattr(guard.capabilities, "edition", "") if hasattr(guard, "capabilities") else ""

        for tier in preset.license_tiers:
            normalized = tier.strip()
            if not normalized:
                continue
            message = (
                f"Preset '{preset_name}' wymaga licencji '{normalized}' (obecna edycja: '{current_edition or 'unknown'}')."
            )
            try:
                guard.require_license_tier(normalized, message=message)
            except LicenseCapabilityError as exc:
                issues.append(str(exc))
                last_exc = exc

        for capability in preset.capabilities:
            normalized = capability.strip()
            if not normalized:
                continue
            message = f"Preset '{preset_name}' wymaga aktywnej strategii '{normalized}'."
            try:
                guard.require_strategy(normalized, message=message)
            except LicenseCapabilityError as exc:
                issues.append(str(exc))
                last_exc = exc

        if not issues:
            return ()

        unique_issues = tuple(dict.fromkeys(issues))
        if raise_on_error:
            capability = last_exc.capability if last_exc is not None else None
            raise LicenseCapabilityError("; ".join(unique_issues), capability=capability)
        return unique_issues

    def _build_candidates(
        self,
        preset: Mapping[str, object],
        version: PresetVersionInfo,
    ) -> tuple[DecisionCandidate, ...]:
        strategies = tuple(preset.get("strategies", []))
        candidates: list[DecisionCandidate] = []
        for entry in strategies:
            metadata = dict(entry.get("metadata", {})) if isinstance(entry, Mapping) else {}
            metadata.update(
                {
                    "preset": preset.get("name"),
                    "preset_version": version.hash,
                    "tags": tuple(entry.get("tags", [])) if isinstance(entry, Mapping) else tuple(),
                    "license_tier": entry.get("license_tier") if isinstance(entry, Mapping) else None,
                }
            )
            probability = float(metadata.get("expected_probability", 1.0))
            probability = max(0.0, min(1.0, probability))
            candidate = DecisionCandidate(
                strategy=str(entry.get("engine", "")),
                action="activate",
                risk_profile=str(entry.get("risk_profile") or metadata.get("risk_profile", "")),
                symbol=None,
                notional=float(metadata.get("notional", 0.0)),
                expected_return_bps=float(metadata.get("expected_return_bps", 0.0)),
                expected_probability=probability,
                metadata=metadata,
            )
            candidates.append(candidate)
        return tuple(candidates)

    def _fetch_recommendation(self, regime: MarketRegime) -> str | None:
        if self._decision_engine is None:
            return None
        try:
            return self._decision_engine.select_strategy(regime)
        except Exception as exc:  # pragma: no cover - defensywne
            self._logger.debug("Decision orchestrator recommendation failed: %s", exc)
            return None


__all__ = [
    "PresetVersionInfo",
    "RegimePresetActivation",
    "PresetAvailability",
    "ActivationHistoryStats",
    "ActivationTransitionStats",
    "ActivationCadenceStats",
    "ActivationUptimeStats",
    "ActivationReliabilityStats",
    "ActivationOutcomeStats",
    "ActivationBlockerStats",
    "ActivationPresetStats",
    "ActivationLicenseStats",
    "ActivationAssessmentStats",
    "ActivationDataStats",
    "ActivationDecisionStats",
    "StrategyRegimeWorkflow",
]

