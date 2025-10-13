"""Harmonogram wielostrate-giczny obsługujący wiele silników strategii."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping, MutableMapping, Protocol, Sequence

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
    max_signals: int
    last_run: datetime | None = None
    warmed_up: bool = False
    metrics: MutableMapping[str, float] = field(default_factory=dict)


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
    ) -> None:
        self._environment = environment
        self._portfolio = portfolio
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._telemetry = telemetry_emitter
        self._decision_journal = decision_journal
        self._schedules: list[_ScheduleContext] = []
        self._stop_event: asyncio.Event | None = None
        self._tasks: list[asyncio.Task[None]] = []

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
            max_signals=max(1, max_signals),
        )
        self._schedules.append(context)
        _LOGGER.debug("Zarejestrowano harmonogram %s dla strategii %s", name, strategy_name)

    async def run_forever(self) -> None:
        if self._tasks:
            raise RuntimeError("Scheduler został już uruchomiony")

        self._stop_event = asyncio.Event()
        self._tasks = [
            asyncio.create_task(self._run_schedule(schedule), name=f"strategy:{schedule.name}")
            for schedule in self._schedules
        ]

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

    async def _execute_schedule(self, schedule: _ScheduleContext, timestamp: datetime) -> None:
        try:
            if not schedule.warmed_up and schedule.warmup_bars > 0:
                history = schedule.feed.load_history(schedule.strategy_name, schedule.warmup_bars)
                if history:
                    schedule.strategy.warm_up(history)
                schedule.warmed_up = True

            schedule.metrics.clear()
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
                bounded_signals = self._bounded_signals(signals, schedule.max_signals)
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
                schedule.metrics["last_latency_ms"] = max(
                    0.0, (self._clock() - timestamp).total_seconds() * 1000
                )
                self._record_decisions(schedule, bounded_signals, timestamp, snapshot.symbol)
                schedule.sink.submit(
                    strategy_name=schedule.strategy_name,
                    schedule_name=schedule.name,
                    risk_profile=schedule.risk_profile,
                    timestamp=timestamp,
                    signals=bounded_signals,
                )
            schedule.metrics["signals"] = float(total_signals)
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
        except Exception:  # pragma: no cover - chronimy scheduler przed przerwaniem
            _LOGGER.exception("Błąd podczas wykonywania harmonogramu %s", schedule.name)

    def _bounded_signals(
        self, signals: Sequence[StrategySignal], max_signals: int
    ) -> Sequence[StrategySignal]:
        if len(signals) <= max_signals:
            return signals
        ordered = sorted(signals, key=lambda signal: signal.confidence, reverse=True)
        return tuple(ordered[:max_signals])

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
]
