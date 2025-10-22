"""Harmonogram wielostrate-giczny obsługujący wiele silników strategii."""
from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, Mapping, MutableMapping, Protocol, Sequence

from threading import RLock

if TYPE_CHECKING:
    from bot_core.runtime.portfolio_coordinator import PortfolioRuntimeCoordinator

from bot_core.runtime.capital_policies import (
    BlendedCapitalAllocation,
    CapitalAllocationPolicy,
    DrawdownAdaptiveAllocation,
    EqualWeightAllocation,
    FixedWeightAllocation,
    MetricWeightRule,
    MetricWeightedAllocation,
    RiskParityAllocation,
    RiskProfileBudgetAllocation,
    SignalStrengthAllocation,
    SmoothedCapitalAllocationPolicy,
    TagQuotaAllocation,
    VolatilityTargetAllocation,
    normalize_weights,
)
from bot_core.runtime.journal import TradingDecisionEvent, TradingDecisionJournal
from bot_core.runtime.signal_limits import SignalLimitManager, SignalLimitOverride
from bot_core.runtime.suspensions import SuspensionManager, SuspensionRecord
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
        self._signal_limit_manager = SignalLimitManager(clock=self._clock, logger=_LOGGER)
        self._suspension_manager = SuspensionManager(clock=self._clock, logger=_LOGGER)
        for strategy, profiles in (signal_limits or {}).items():
            for profile, limit in (profiles or {}).items():
                self.configure_signal_limit(strategy, profile, limit)

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
        suspension_snapshot = self._suspension_manager.snapshot()
        schedule_suspensions = dict(suspension_snapshot.get("schedules", {}))
        tag_suspensions = dict(suspension_snapshot.get("tags", {}))

        active_overrides, expired_overrides = self._signal_limit_manager.active_overrides(now=now)
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

        now = self._clock()
        active, expired = self._signal_limit_manager.active_overrides(now=now)
        if expired:
            self._handle_expired_signal_limits(expired, now)
        snapshot: dict[str, dict[str, Mapping[str, object]]] = {}
        for (strategy, profile), override in active.items():
            strategy_entry = snapshot.setdefault(strategy, {})
            strategy_entry[profile] = dict(override.to_snapshot(now))
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
        self._signal_limit_manager.configure_limit(
            strategy_name,
            risk_profile,
            limit,
            reason=reason,
            until=until,
            duration_seconds=duration_seconds,
        )

    def configure_signal_limits(
        self,
        limits: Mapping[str, Mapping[str, object]],
        *,
        reason: str | None = None,
        until: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        self._signal_limit_manager.configure_limits(
            limits,
            reason=reason,
            until=until,
            duration_seconds=duration_seconds,
        )


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
        affected = set(expired)
        for schedule in self._schedules:
            if id(schedule) in skip_ids:
                continue
            if (schedule.strategy_name, schedule.risk_profile) in affected:
                self._apply_signal_limits(schedule)

    def suspend_schedule(
        self,
        schedule_name: str,
        *,
        reason: str | None = None,
        until: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        self._suspension_manager.suspend_schedule(
            schedule_name,
            reason=reason,
            until=until,
            duration_seconds=duration_seconds,
        )

    def resume_schedule(self, schedule_name: str) -> bool:
        return self._suspension_manager.resume_schedule(schedule_name)

    def suspend_tag(
        self,
        tag: str,
        *,
        reason: str | None = None,
        until: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        self._suspension_manager.suspend_tag(
            tag,
            reason=reason,
            until=until,
            duration_seconds=duration_seconds,
        )

    def resume_tag(self, tag: str) -> bool:
        return self._suspension_manager.resume_tag(tag)

    def suspension_snapshot(self) -> Mapping[str, Mapping[str, object]]:
        """Return a snapshot of active schedule and tag suspensions.

        Besides the raw mappings exported by :class:`SuspensionManager`, the
        snapshot now exposes lightweight metadata that is useful when
        troubleshooting the runtime.  The additional keys are backwards
        compatible with previous consumers because the ``schedules`` and
        ``tags`` entries remain unchanged, while the metadata lives under new
        top-level keys such as ``counts``, ``reason_stats``, ``next_expiration``
        or the aggregated ``expiring_entries`` and ``expiration_buckets``.
        """

        snapshot = self._suspension_manager.snapshot()
        schedules = dict(snapshot.get("schedules", {})) if isinstance(snapshot, Mapping) else {}
        tags = dict(snapshot.get("tags", {})) if isinstance(snapshot, Mapping) else {}

        now = self._clock()

        metadata: dict[str, object] = {
            "schedules": schedules,
            "tags": tags,
        }

        expiring_entries: list[dict[str, object]] = []

        schedule_count = len(schedules)
        tag_count = len(tags)
        metadata["counts"] = {
            "schedules": schedule_count,
            "tags": tag_count,
            "total": schedule_count + tag_count,
        }

        schedule_reasons: dict[str, str] = {}
        tag_reasons: dict[str, str] = {}
        reason_stats: dict[str, dict[str, object]] = {}
        scope_stats: dict[str, dict[str, object]] = {
            "schedules": {
                "total": schedule_count,
                "expiring": 0,
                "indefinite": 0,
            },
            "tags": {
                "total": tag_count,
                "expiring": 0,
                "indefinite": 0,
            },
        }

        def _parse_timestamp(value: object) -> datetime | None:
            if isinstance(value, str) and value:
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    return None
            return None

        def _register_reason(
            *,
            scope: str,
            name: str,
            reason_value: object,
            remaining_seconds: float | None,
            until_value: str | None,
            applied_value: str | None,
            applied_at: datetime | None,
            age_seconds: float | None,
        ) -> None:
            if not isinstance(reason_value, str) or not reason_value:
                return

            container = schedule_reasons if scope == "schedules" else tag_reasons
            container[name] = reason_value

            stats = reason_stats.setdefault(
                reason_value,
                {
                    "schedules": 0,
                    "tags": 0,
                    "total": 0,
                    "expiring": 0,
                    "indefinite": 0,
                },
            )
            stats[scope] = int(stats.get(scope, 0)) + 1
            stats["total"] = int(stats.get("total", 0)) + 1
            entries = stats.setdefault("entries", [])

            if remaining_seconds is not None:
                stats["expiring"] = int(stats.get("expiring", 0)) + 1
                next_payload = {
                    "scope": scope[:-1],
                    "name": name,
                    "remaining_seconds": remaining_seconds,
                }
                if until_value:
                    next_payload["until"] = until_value
                if applied_value:
                    next_payload["applied_at"] = applied_value
                if age_seconds is not None:
                    next_payload["age_seconds"] = age_seconds

                existing_next = stats.get("next_expiration")
                if not isinstance(existing_next, Mapping):
                    stats["next_expiration"] = next_payload
                else:
                    existing_value = existing_next.get("remaining_seconds")
                    try:
                        existing_seconds = float(existing_value)
                    except (TypeError, ValueError):
                        existing_seconds = math.inf
                    if remaining_seconds < existing_seconds:
                        stats["next_expiration"] = next_payload

            else:
                stats["indefinite"] = int(stats.get("indefinite", 0)) + 1

            descriptor = {
                "scope": scope[:-1],
                "name": name,
            }
            if isinstance(reason_value, str) and reason_value:
                descriptor["reason"] = reason_value
            if remaining_seconds is not None:
                descriptor["remaining_seconds"] = remaining_seconds
            if until_value:
                descriptor["until"] = until_value
            if isinstance(applied_value, str) and applied_value:
                descriptor["applied_at"] = applied_value
            if age_seconds is not None:
                descriptor["age_seconds"] = age_seconds
            entries.append(descriptor)
            if remaining_seconds is not None:
                expiring_entries.append(dict(descriptor))

            scope_breakdown = stats.setdefault("scope_breakdown", {})
            bucket = scope_breakdown.setdefault(
                scope,
                {"expiring": 0, "indefinite": 0},
            )
            if remaining_seconds is not None:
                bucket["expiring"] = int(bucket.get("expiring", 0)) + 1
                candidate = {
                    "scope": scope[:-1],
                    "name": name,
                    "remaining_seconds": remaining_seconds,
                }
                if until_value:
                    candidate["until"] = until_value
                if applied_value:
                    candidate["applied_at"] = applied_value
                if age_seconds is not None:
                    candidate["age_seconds"] = age_seconds
                existing_scope_next = bucket.get("next_expiration")
                existing_scope_seconds = bucket.get("_next_seconds")
                if (
                    not isinstance(existing_scope_next, Mapping)
                    or not isinstance(existing_scope_seconds, (int, float))
                    or remaining_seconds < float(existing_scope_seconds)
                ):
                    bucket["next_expiration"] = candidate
                    bucket["_next_seconds"] = remaining_seconds
            else:
                bucket["indefinite"] = int(bucket.get("indefinite", 0)) + 1

            scopes = stats.setdefault("_scopes", set())
            scopes.add(scope)

            bucket_ages = bucket.setdefault("_ages", [])
            if age_seconds is not None:
                bucket_ages.append(age_seconds)
                stats.setdefault("_ages", []).append(age_seconds)

            if applied_at is not None and applied_value:
                oldest_marker = bucket.get("_oldest_ts")
                newest_marker = bucket.get("_newest_ts")
                if (
                    not isinstance(oldest_marker, datetime)
                    or applied_at < oldest_marker
                ):
                    bucket["oldest"] = descriptor
                    bucket["_oldest_ts"] = applied_at
                if (
                    not isinstance(newest_marker, datetime)
                    or applied_at >= newest_marker
                ):
                    bucket["newest"] = descriptor
                    bucket["_newest_ts"] = applied_at

                stats_oldest = stats.get("_oldest_ts")
                if not isinstance(stats_oldest, datetime) or applied_at < stats_oldest:
                    stats["oldest"] = descriptor
                    stats["_oldest_ts"] = applied_at
                stats_newest = stats.get("_newest_ts")
                if not isinstance(stats_newest, datetime) or applied_at >= stats_newest:
                    stats["newest"] = descriptor
                    stats["_newest_ts"] = applied_at

        def _summarize_scope(
            *, entries: Mapping[str, object], scope: str
        ) -> None:
            oldest: tuple[datetime, dict[str, object]] | None = None
            newest: tuple[datetime, dict[str, object]] | None = None
            stats = scope_stats[scope]
            ages: list[float] = []
            for name, entry in entries.items():
                if not isinstance(entry, Mapping):
                    continue

                remaining_raw = entry.get("remaining_seconds")
                remaining_seconds: float | None = None
                has_expiration = False
                if remaining_raw not in (None, ""):
                    try:
                        seconds = float(remaining_raw)
                    except (TypeError, ValueError):
                        has_expiration = False
                    else:
                        if math.isfinite(seconds) and seconds >= 0.0:
                            has_expiration = True
                            remaining_seconds = max(0.0, seconds)
                if has_expiration:
                    stats["expiring"] = int(stats.get("expiring", 0)) + 1
                else:
                    stats["indefinite"] = int(stats.get("indefinite", 0)) + 1

                applied_value = entry.get("applied_at")
                applied_at = _parse_timestamp(applied_value)
                age_seconds: float | None = None
                if applied_at is not None:
                    try:
                        age_seconds = max(0.0, (now - applied_at).total_seconds())
                    except (TypeError, OverflowError):
                        age_seconds = None
                if age_seconds is not None:
                    ages.append(age_seconds)
                until_value_obj = entry.get("until")
                until_value = (
                    str(until_value_obj)
                    if isinstance(until_value_obj, str) and until_value_obj
                    else None
                )
                reason_value = entry.get("reason")
                _register_reason(
                    scope=scope,
                    name=name,
                    reason_value=reason_value,
                    remaining_seconds=remaining_seconds,
                    until_value=until_value,
                    applied_value=applied_value if isinstance(applied_value, str) else None,
                    applied_at=applied_at,
                    age_seconds=age_seconds,
                )

                if applied_at is None or not isinstance(applied_value, str):
                    continue
                descriptor = {
                    "scope": scope[:-1],
                    "name": name,
                    "applied_at": applied_value,
                }
                if isinstance(reason_value, str) and reason_value:
                    descriptor["reason"] = reason_value
                if age_seconds is not None:
                    descriptor["age_seconds"] = age_seconds
                if oldest is None or applied_at < oldest[0]:
                    oldest = (applied_at, descriptor)
                if newest is None or applied_at >= newest[0]:
                    newest = (applied_at, descriptor)

            if oldest is not None:
                stats["oldest"] = oldest[1]
            if newest is not None:
                stats["newest"] = newest[1]
            if ages:
                stats["age_stats"] = {
                    "average": sum(ages) / len(ages),
                    "min": min(ages),
                    "max": max(ages),
                }

        _summarize_scope(entries=schedules, scope="schedules")
        _summarize_scope(entries=tags, scope="tags")

        if schedule_reasons or tag_reasons:
            unique_reasons = sorted({*schedule_reasons.values(), *tag_reasons.values()})
            for reason_value, stats in reason_stats.items():
                scope_breakdown = stats.get("scope_breakdown")
                if isinstance(scope_breakdown, Mapping):
                    for bucket in list(scope_breakdown.values()):
                        if not isinstance(bucket, dict):
                            continue
                        ages = bucket.pop("_ages", None)
                        if isinstance(ages, list) and ages:
                            bucket["age_stats"] = {
                                "average": sum(ages) / len(ages),
                                "min": min(ages),
                                "max": max(ages),
                            }
                        bucket.pop("_next_seconds", None)
                        bucket.pop("_oldest_ts", None)
                        bucket.pop("_newest_ts", None)
                        if "next_expiration" not in bucket and bucket.get("expiring") == 0:
                            bucket.pop("next_expiration", None)
                        if "oldest" not in bucket:
                            bucket.pop("oldest", None)
                        if "newest" not in bucket:
                            bucket.pop("newest", None)
                stats.pop("_oldest_ts", None)
                stats.pop("_newest_ts", None)
                scopes = stats.pop("_scopes", None)
                if scopes:
                    stats["scopes"] = sorted(set(scopes))
                ages = stats.pop("_ages", None)
                if isinstance(ages, list) and ages:
                    stats["age_stats"] = {
                        "average": sum(ages) / len(ages),
                        "min": min(ages),
                        "max": max(ages),
                    }
                entries = stats.get("entries")
                if isinstance(entries, list):
                    def _entry_sort_key(payload: Mapping[str, object]) -> tuple[str, str]:
                        applied = ""
                        applied_value = payload.get("applied_at")
                        if isinstance(applied_value, str):
                            applied = applied_value
                        return applied, str(payload.get("name", ""))

                    entries.sort(key=_entry_sort_key)

            metadata["reasons"] = {
                "schedules": schedule_reasons,
                "tags": tag_reasons,
                "unique": unique_reasons,
            }
            metadata["reason_stats"] = reason_stats
        metadata["scope_stats"] = scope_stats

        def _extract_next_expiration(
            entries: Mapping[str, object],
            scope: str,
        ) -> tuple[float, dict[str, object]] | None:
            next_candidate: tuple[float, dict[str, object]] | None = None
            for name, payload in entries.items():
                if not isinstance(payload, Mapping):
                    continue
                remaining = payload.get("remaining_seconds")
                if remaining in (None, ""):
                    continue
                try:
                    seconds = float(remaining)
                except (TypeError, ValueError):
                    continue
                seconds = max(0.0, seconds)
                candidate = {
                    "scope": scope,
                    "name": name,
                    "remaining_seconds": seconds,
                }
                until_value = payload.get("until")
                if isinstance(until_value, str) and until_value:
                    candidate["until"] = until_value
                applied_value = payload.get("applied_at")
                if isinstance(applied_value, str) and applied_value:
                    candidate["applied_at"] = applied_value
                if next_candidate is None or seconds < next_candidate[0]:
                    next_candidate = (seconds, candidate)
            return next_candidate

        upcoming: tuple[float, dict[str, object]] | None = None
        schedule_next = _extract_next_expiration(schedules, "schedule")
        if schedule_next is not None:
            scope_stats["schedules"]["next_expiration"] = schedule_next[1]
        tag_next = _extract_next_expiration(tags, "tag")
        if tag_next is not None:
            scope_stats["tags"]["next_expiration"] = tag_next[1]
        for candidate in (schedule_next, tag_next):
            if candidate is None:
                continue
            if upcoming is None or candidate[0] < upcoming[0]:
                upcoming = candidate
        if upcoming is not None:
            metadata["next_expiration"] = upcoming[1]

        if expiring_entries:
            def _expiring_sort_key(payload: Mapping[str, object]) -> tuple[float, str, str]:
                remaining = payload.get("remaining_seconds")
                try:
                    seconds = float(remaining)
                except (TypeError, ValueError):
                    seconds = math.inf
                applied_value = payload.get("applied_at")
                applied = applied_value if isinstance(applied_value, str) else ""
                return seconds, applied, str(payload.get("name", ""))

            expiring_entries.sort(key=_expiring_sort_key)
            metadata["expiring_entries"] = expiring_entries
            metadata["expiring_total"] = len(expiring_entries)

            bucket_definitions: tuple[tuple[str, float], ...] = (
                ("1m", 60.0),
                ("5m", 5 * 60.0),
                ("15m", 15 * 60.0),
                ("1h", 60.0 * 60.0),
                ("4h", 4 * 60.0 * 60.0),
                ("12h", 12 * 60.0 * 60.0),
                ("1d", 24 * 60.0 * 60.0),
                ("3d", 3 * 24 * 60.0 * 60.0),
                ("7d", 7 * 24 * 60.0 * 60.0),
            )

            expiration_buckets: dict[str, dict[str, object]] = {}
            for label, threshold in bucket_definitions:
                window: list[dict[str, object]] = [
                    entry
                    for entry in expiring_entries
                    if float(entry.get("remaining_seconds", math.inf)) <= threshold
                ]
                if not window:
                    continue
                scopes = sorted({entry.get("scope", "") for entry in window if entry.get("scope")})
                bucket_payload: dict[str, object] = {
                    "count": len(window),
                    "next": dict(window[0]),
                }
                if len(window) > 1:
                    bucket_payload["last"] = dict(window[-1])
                if scopes:
                    bucket_payload["scopes"] = scopes
                expiration_buckets[label] = bucket_payload
            if expiration_buckets:
                metadata["expiration_buckets"] = expiration_buckets

        return metadata

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
            suspension = self._suspension_manager.resolve(
                schedule.name,
                schedule.tags,
                schedule.primary_tag,
                timestamp,
            )
            schedule.last_run = timestamp
            if suspension is not None:
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
            suspension = self._suspension_manager.resolve(
                schedule.name,
                schedule.tags,
                schedule.primary_tag,
                start_time,
            )
            if suspension is not None:
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
        async with self._allocation_lock:
            schedules_snapshot = tuple(self._schedules)
            weights = policy.allocate(schedules_snapshot)
            normalized = normalize_weights(weights)
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
            normalized_profile_snapshot = normalize_weights(profile_snapshot)
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
        normalized_raw_snapshot = normalize_weights(raw_snapshot) if raw_snapshot else {}
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
        now = self._clock()
        override, expired_override = self._signal_limit_manager.resolve_override(
            schedule.strategy_name,
            schedule.risk_profile,
            now=now,
        )
        if expired_override:
            self._handle_expired_signal_limits(expired_override, now, skip=(schedule,))
        if override is not None:
            override_value = max(0, int(override.limit))
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
            if override.expires_at is not None:
                schedule.metrics["signal_limit_expires_at"] = override.expires_at.timestamp()
            if override.reason:
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
    "SmoothedCapitalAllocationPolicy",
    "DrawdownAdaptiveAllocation",
    "FixedWeightAllocation",
    "RiskProfileBudgetAllocation",
    "TagQuotaAllocation",
]
