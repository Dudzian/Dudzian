"""Harmonogram wielostrate-giczny obsługujący wiele silników strategii."""
from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
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
    from bot_core.backtest.walk_forward import WalkForwardReport
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
    license_tier: str | None = None
    risk_classes: tuple[str, ...] = ()
    required_data: tuple[str, ...] = ()
    capability: str | None = None
    last_run: datetime | None = None
    warmed_up: bool = False
    metrics: MutableMapping[str, float] = field(default_factory=dict)
    priority: int = 0
    windows: tuple[tuple[int, int], ...] = ()
    fallback_schedules: tuple[str, ...] = ()
    active_fallback: str | None = None
    failover_count: int = 0
    cost_report_summary: Mapping[str, float] | None = None
    last_window_state: bool | None = None

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


def _coerce_non_empty_str(value: object | None) -> str | None:
    if not isinstance(value, str):
        if value is None:
            return None
        try:
            candidate = str(value)
        except Exception:  # pragma: no cover - defensywne logowanie
            return None
    else:
        candidate = value
    candidate = candidate.strip()
    return candidate or None


def _normalize_metadata_sequence(value: object | None) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        candidates: Iterable[object] = (value,)
    elif isinstance(value, Mapping):
        return ()
    elif isinstance(value, Iterable):
        candidates = value
    else:
        return ()
    normalized: list[str] = []
    for item in candidates:
        if item in (None, ""):
            continue
        try:
            candidate = str(item).strip()
        except Exception:  # pragma: no cover - diagnostyka danych
            continue
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized)


def _parse_schedule_priority(metadata: Mapping[str, object] | None) -> int:
    if not isinstance(metadata, Mapping):
        return 0
    candidate = metadata.get("schedule_priority") or metadata.get("priority")
    if candidate in (None, ""):
        return 0
    try:
        return int(candidate)
    except (TypeError, ValueError):  # pragma: no cover - diagnostyka danych
        return 0


def _parse_window_entry(entry: str) -> tuple[int, int] | None:
    parts = entry.split("-", 1)
    if len(parts) != 2:
        return None
    try:
        start_hours, start_minutes = [int(value) for value in parts[0].split(":", 1)]
        end_hours, end_minutes = [int(value) for value in parts[1].split(":", 1)]
    except ValueError:
        return None
    start_total = max(0, min(23, start_hours)) * 60 + max(0, min(59, start_minutes))
    end_total = max(0, min(23, end_hours)) * 60 + max(0, min(59, end_minutes))
    return start_total % (24 * 60), end_total % (24 * 60)


def _parse_schedule_windows(metadata: Mapping[str, object] | None) -> tuple[tuple[int, int], ...]:
    if not isinstance(metadata, Mapping):
        return ()
    source = metadata.get("schedule_windows") or metadata.get("windows")
    entries: list[str] = []
    if isinstance(source, str):
        entries = [segment.strip() for segment in source.split(",") if segment.strip()]
    elif isinstance(source, Mapping):
        start = source.get("start")
        end = source.get("end")
        if isinstance(start, str) and isinstance(end, str):
            candidate = _parse_window_entry(f"{start}-{end}")
            if candidate:
                return (candidate,)
        return ()
    elif isinstance(source, Iterable):
        for item in source:
            if isinstance(item, str) and item.strip():
                entries.append(item.strip())
            elif isinstance(item, Mapping):
                start = item.get("start")
                end = item.get("end")
                if isinstance(start, str) and isinstance(end, str):
                    candidate = _parse_window_entry(f"{start}-{end}")
                    if candidate:
                        entries.append(f"{start}-{end}")
    windows: list[tuple[int, int]] = []
    for entry in entries:
        candidate = _parse_window_entry(entry)
        if candidate is not None:
            windows.append(candidate)
    return tuple(windows)


def _parse_schedule_fallbacks(metadata: Mapping[str, object] | None) -> tuple[str, ...]:
    if not isinstance(metadata, Mapping):
        return ()
    raw = metadata.get("fallback_schedules") or metadata.get("schedule_fallbacks")
    if raw in (None, ""):
        return ()
    if isinstance(raw, str):
        entries = [raw]
    elif isinstance(raw, Iterable):
        entries = list(raw)
    else:
        return ()
    normalized: list[str] = []
    for item in entries:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return tuple(dict.fromkeys(normalized))


def _format_window(window: tuple[int, int]) -> str:
    start, end = window
    def _format(minutes: int) -> str:
        minutes %= 24 * 60
        hour = minutes // 60
        minute = minutes % 60
        return f"{hour:02d}:{minute:02d}"

    return f"{_format(start)}-{_format(end)}"


def _extract_schedule_metadata(
    strategy: StrategyEngine,
) -> tuple[
    tuple[str, ...],
    str | None,
    str | None,
    tuple[str, ...],
    tuple[str, ...],
    str | None,
    int,
    tuple[tuple[int, int], ...],
    tuple[str, ...],
]:
    raw_metadata = getattr(strategy, "metadata", None)
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else None

    tags, primary_tag = _extract_tags(metadata)
    license_tier = _coerce_non_empty_str(metadata.get("license_tier")) if metadata else None
    capability = _coerce_non_empty_str(metadata.get("capability")) if metadata else None
    risk_classes = (
        _normalize_metadata_sequence(metadata.get("risk_classes")) if metadata else ()
    )
    required_data = (
        _normalize_metadata_sequence(metadata.get("required_data")) if metadata else ()
    )

    license_tier = license_tier or _coerce_non_empty_str(getattr(strategy, "license_tier", None))
    capability = capability or _coerce_non_empty_str(getattr(strategy, "capability", None))
    risk_classes = risk_classes or _normalize_metadata_sequence(
        getattr(strategy, "risk_classes", None)
    )
    required_data = required_data or _normalize_metadata_sequence(
        getattr(strategy, "required_data", None)
    )

    priority = _parse_schedule_priority(metadata)
    windows = _parse_schedule_windows(metadata)
    fallbacks = _parse_schedule_fallbacks(metadata)

    return (
        tags,
        primary_tag,
        license_tier,
        risk_classes,
        required_data,
        capability,
        priority,
        windows,
        fallbacks,
    )


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
        self._alerts: deque[Mapping[str, object]] = deque(maxlen=256)
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
            if schedule.license_tier:
                descriptor["license_tier"] = schedule.license_tier
            if schedule.capability:
                descriptor["capability"] = schedule.capability
            if schedule.risk_classes:
                descriptor["risk_classes"] = list(schedule.risk_classes)
            if schedule.required_data:
                descriptor["required_data"] = list(schedule.required_data)
            if schedule.last_run is not None:
                descriptor["last_run"] = schedule.last_run.isoformat()
            if schedule.priority:
                descriptor["priority"] = int(schedule.priority)
            if schedule.windows:
                descriptor["windows"] = [_format_window(window) for window in schedule.windows]
            if schedule.fallback_schedules:
                descriptor["fallbacks"] = list(schedule.fallback_schedules)
            if schedule.active_fallback:
                descriptor["active_fallback"] = schedule.active_fallback
            descriptor["failover_count"] = int(schedule.failover_count)
            if schedule.cost_report_summary:
                descriptor["cost_report"] = dict(schedule.cost_report_summary)
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
        (
            tags,
            primary_tag,
            license_tier,
            risk_classes,
            required_data,
            capability,
            priority,
            windows,
            fallbacks,
        ) = _extract_schedule_metadata(strategy)

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
            license_tier=license_tier,
            risk_classes=risk_classes,
            required_data=required_data,
            capability=capability,
            priority=priority,
            windows=windows,
            fallback_schedules=fallbacks,
        )
        self._schedules.append(context)
        self._schedules.sort(key=lambda item: (-item.priority, item.name))
        _LOGGER.debug("Zarejestrowano harmonogram %s dla strategii %s", name, strategy_name)

    def attach_cost_report(self, schedule_name: str, report: "WalkForwardReport") -> None:
        """Podpina raport kosztowy walk-forward do harmonogramu."""

        try:
            schedule = self._find_schedule(schedule_name)
        except KeyError:
            raise KeyError(f"Harmonogram '{schedule_name}' nie jest zarejestrowany") from None

        summary = {
            "total_return_pct": float(report.total_return_pct),
            "total_trades": float(report.cost_summary.total_trades),
            "total_fees": float(report.cost_summary.total_fees),
            "total_slippage": float(report.cost_summary.total_slippage),
            "total_notional": float(report.cost_summary.total_notional),
            "segments": float(len(report.segments)),
        }
        schedule.cost_report_summary = summary

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
        return self._suspension_manager.snapshot()

    def alerts_snapshot(self, *, clear: bool = False) -> tuple[Mapping[str, object], ...]:
        """Zwraca kolejkę alertów zarejestrowanych przez scheduler."""

        payload = tuple(self._alerts)
        if clear:
            self._alerts.clear()
        return payload

    def _record_alert(
        self,
        code: str,
        *,
        message: str,
        schedule: str | None = None,
        severity: str = "warning",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "code": code,
            "message": message,
            "severity": severity,
            "timestamp": self._clock().isoformat(),
        }
        if schedule:
            payload["schedule"] = schedule
        if metadata:
            payload["metadata"] = dict(metadata)
        self._alerts.append(payload)

    def _find_schedule(self, name: str) -> _ScheduleContext:
        for schedule in self._schedules:
            if schedule.name == name:
                return schedule
        raise KeyError(name)

    def _apply_cost_summary(self, schedule: _ScheduleContext) -> None:
        if not schedule.cost_report_summary:
            return
        for key, value in schedule.cost_report_summary.items():
            try:
                schedule.metrics[f"cost_{key}"] = float(value)
            except (TypeError, ValueError):  # pragma: no cover - diagnostyka danych
                continue

    def _update_window_state(
        self, schedule: _ScheduleContext, *, active: bool, timestamp: datetime
    ) -> None:
        previous = schedule.last_window_state
        schedule.last_window_state = active
        if previous is None:
            if not active:
                self._record_alert(
                    "schedule-window-closed",
                    message=f"Harmonogram {schedule.name} poza oknem czasowym",
                    schedule=schedule.name,
                    severity="info",
                    metadata={"timestamp": timestamp.isoformat()},
                )
            return
        if previous == active:
            return
        code = "schedule-window-open" if active else "schedule-window-closed"
        severity = "info" if active else "warning"
        self._record_alert(
            code,
            message=(
                f"Harmonogram {schedule.name} wznowił działanie"
                if active
                else f"Harmonogram {schedule.name} został wstrzymany poza oknem czasowym"
            ),
            schedule=schedule.name,
            severity=severity,
            metadata={"timestamp": timestamp.isoformat()},
        )

    @staticmethod
    def _is_window_open(schedule: _ScheduleContext, timestamp: datetime) -> bool:
        if not schedule.windows:
            return True
        minutes = timestamp.hour * 60 + timestamp.minute
        for start, end in schedule.windows:
            if start == end:
                return True
            if start < end and start <= minutes < end:
                return True
            if start > end and (minutes >= start or minutes < end):
                return True
        return False

    def _handle_schedule_failure(
        self,
        schedule: _ScheduleContext,
        reason: str,
        exc: Exception | None = None,
    ) -> None:
        metadata: dict[str, object] = {"reason": reason}
        if exc is not None:
            metadata["error"] = str(exc)
        schedule.metrics["failed"] = 1.0
        self._record_alert(
            "strategy-failure",
            message=f"Harmonogram {schedule.name} zgłosił błąd: {reason}",
            schedule=schedule.name,
            metadata=metadata,
        )
        self._trigger_failover(schedule, reason, metadata)

    def _trigger_failover(
        self,
        schedule: _ScheduleContext,
        reason: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        schedule.failover_count += 1
        schedule.metrics["failover_count"] = float(schedule.failover_count)
        for candidate in schedule.fallback_schedules:
            if candidate == schedule.name:
                continue
            try:
                fallback = self._find_schedule(candidate)
            except KeyError:
                continue
            schedule.active_fallback = fallback.name
            self._record_alert(
                "strategy-failover",
                message=f"Aktywowano fallback {fallback.name} dla harmonogramu {schedule.name}",
                schedule=schedule.name,
                metadata={"reason": reason, "fallback": fallback.name, **(metadata or {})},
            )
            self.resume_schedule(fallback.name)
            self.suspend_schedule(schedule.name, reason=f"failover:{reason}", duration_seconds=900.0)
            return

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
            schedule.metrics["governor_signal_factor"] = float(schedule.governor_signal_factor)
            schedule.metrics["governor_max_signals"] = float(
                schedule.base_max_signals * max(0.0, schedule.governor_signal_factor)
            )
            self._apply_signal_limits(schedule)
            self._apply_cost_summary(schedule)
            schedule.metrics["priority"] = float(schedule.priority)
            schedule.metrics["failover_count"] = float(schedule.failover_count)
            schedule.metrics["active_fallback_indicator"] = 1.0 if schedule.active_fallback else 0.0

            window_active = self._is_window_open(schedule, timestamp)
            schedule.metrics["window_active"] = 1.0 if window_active else 0.0
            self._update_window_state(schedule, active=window_active, timestamp=timestamp)
            if not window_active:
                return

            try:
                snapshots = schedule.feed.fetch_latest(schedule.strategy_name)
            except Exception as exc:  # pragma: no cover - diagnostyka feedu
                self._handle_schedule_failure(schedule, "feed", exc)
                return

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
                try:
                    raw_signals = list(schedule.strategy.on_data(snapshot))
                except Exception as exc:  # pragma: no cover - diagnostyka strategii
                    self._handle_schedule_failure(schedule, "strategy", exc)
                    return
                if not raw_signals:
                    continue
                bounded_signals = self._bounded_signals(raw_signals, schedule.active_max_signals)
                if not bounded_signals:
                    continue
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
                try:
                    schedule.sink.submit(
                        strategy_name=schedule.strategy_name,
                        schedule_name=schedule.name,
                        risk_profile=schedule.risk_profile,
                        timestamp=timestamp,
                        signals=bounded_signals,
                    )
                except Exception as exc:  # pragma: no cover - diagnostyka sinka
                    self._handle_schedule_failure(schedule, "sink", exc)
                    return
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
            schedule.metrics["governor_signal_factor"] = float(schedule.governor_signal_factor)
            schedule.metrics["governor_max_signals"] = float(
                schedule.base_max_signals * max(0.0, schedule.governor_signal_factor)
            )
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
        governor_factor = max(0.0, schedule.governor_signal_factor)
        allocator_factor = max(0.0, schedule.allocator_signal_factor)
        schedule.metrics["governor_signal_factor"] = float(governor_factor)
        schedule.metrics["allocator_signal_factor"] = float(allocator_factor)
        schedule.metrics["governor_max_signals"] = float(
            schedule.base_max_signals * governor_factor
        )
        if allocator_factor <= 0.0:
            computed = 0
        else:
            computed = int(round(schedule.base_max_signals * governor_factor))
            if schedule.base_max_signals > 0:
                computed = max(floor if governor_factor > 0 else 0, computed)
            if allocator_factor > 1.0 and computed > 0:
                computed = int(round(computed * allocator_factor))
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
