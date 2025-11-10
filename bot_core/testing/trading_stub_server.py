"""Stub serwera gRPC dla kontraktu `botcore.trading.v1` na potrzeby powłoki Qt/QML."""

from __future__ import annotations

import itertools
import threading
import time
from collections import defaultdict
from concurrent import futures
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, MutableMapping, Mapping

import grpc
import yaml
from google.protobuf import timestamp_pb2

from bot_core.runtime._grpc_support import GrpcServerLifecycleMixin

__all__ = [
    "InMemoryTradingDataset",
    "TradingStubServer",
    "build_default_dataset",
    "load_dataset_from_yaml",
    "merge_datasets",
]


InstrumentKey = Tuple[str, str, str]


def _ensure_stubs_loaded():
    generated_dir = Path(__file__).resolve().parent.parent / "generated"
    if str(generated_dir) not in sys.path:
        sys.path.insert(0, str(generated_dir))
    try:
        from bot_core.generated import trading_pb2, trading_pb2_grpc  # type: ignore
    except ImportError as exc:  # pragma: no cover - instrukcja dla developera
        raise RuntimeError(
            "Brak wygenerowanych stubów trading_pb2/trading_pb2_grpc. Uruchom "
            "'python scripts/generate_trading_stubs.py --skip-cpp' przed startem stubu."
        ) from exc
    return trading_pb2, trading_pb2_grpc


@dataclass
class InMemoryTradingDataset:
    """Deterministyczne dane wykorzystywane przez stub serwera."""

    history: Dict[InstrumentKey, List[Any]] = field(default_factory=dict)
    stream_snapshots: Dict[InstrumentKey, List[Any]] = field(default_factory=dict)
    stream_increments: Dict[InstrumentKey, List[Any]] = field(default_factory=dict)
    risk_states: Dict[Optional[InstrumentKey], List[Any]] = field(default_factory=dict)
    metrics: List[Any] = field(default_factory=list)
    health: Any | None = None
    performance_guard: Dict[str, Any] = field(default_factory=dict)
    tradable_instruments: Dict[str, List[Any]] = field(default_factory=dict)
    decision_snapshot: List[Any] = field(default_factory=list)
    decision_increments: List[Any] = field(default_factory=list)
    decision_cycle_metrics: Dict[str, float] = field(default_factory=dict)

    def add_history(self, instrument: Any, granularity: Any, candles: Sequence[Any]) -> None:
        self.history[_instrument_key(instrument, granularity)] = list(candles)

    def set_stream_data(
        self,
        instrument: Any,
        granularity: Any,
        snapshot_candles: Sequence[Any] | None = None,
        increments: Sequence[Any] | None = None,
    ) -> None:
        key = _instrument_key(instrument, granularity)
        if snapshot_candles is not None:
            self.stream_snapshots[key] = list(snapshot_candles)
        if increments is not None:
            self.stream_increments[key] = list(increments)

    def add_risk_states(self, instrument: Any | None, states: Sequence[Any]) -> None:
        key = _instrument_key(instrument, None) if instrument is not None else None
        self.risk_states[key] = list(states)

    def set_metrics(self, snapshots: Sequence[Any]) -> None:
        self.metrics = list(snapshots)

    def set_tradable_instruments(self, exchange: str, instruments: Sequence[Any]) -> None:
        normalized = (exchange or "*").upper()
        self.tradable_instruments[normalized] = [_clone_message(item) for item in instruments]

    def list_tradable_instruments(self, exchange: str) -> List[Any]:
        normalized = (exchange or "*").upper()
        items = self.tradable_instruments.get(normalized)
        if items is None and normalized != "*":
            items = self.tradable_instruments.get("*")
        if not items:
            return []
        return [_clone_message(item) for item in items]

    def set_decision_stream(
        self,
        snapshot: Sequence[Any] | None = None,
        increments: Sequence[Any] | None = None,
    ) -> None:
        if snapshot is not None:
            self.decision_snapshot = [_clone_message(item) for item in snapshot]
        if increments is not None:
            self.decision_increments = [_clone_message(item) for item in increments]

    def set_decision_cycle_metrics(self, metrics: Mapping[str, float]) -> None:
        normalized: Dict[str, float] = {}
        for key, value in metrics.items():
            try:
                normalized[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        self.decision_cycle_metrics = normalized


def merge_datasets(
    base: InMemoryTradingDataset,
    overlay: InMemoryTradingDataset,
) -> InMemoryTradingDataset:
    """Łączy dwa zbiory danych, nadpisując wpisy bazowe danymi z nakładki.

    Funkcja pozwala rozszerzać domyślny dataset o dodatkowe instrumenty oraz
    aktualizacje z plików YAML. Dla metryk i risk state'ów łączone są listy
    (nakładka dodawana jest na końcu), a zdrowie (`health`) zostaje nadpisane
    jeśli w nakładce zdefiniowano nową wartość. Parametry performance guard
    są scalane jako słownik — nakładka może nadpisać pojedyncze klucze.
    """

    base.history.update(overlay.history)
    base.stream_snapshots.update(overlay.stream_snapshots)
    base.stream_increments.update(overlay.stream_increments)
    base.risk_states.update(overlay.risk_states)

    if overlay.metrics:
        if base.metrics:
            base.metrics.extend(overlay.metrics)
        else:
            base.metrics = list(overlay.metrics)

    if overlay.health is not None:
        base.health = overlay.health

    if overlay.performance_guard:
        base.performance_guard.update(overlay.performance_guard)

    if overlay.tradable_instruments:
        for exchange, items in overlay.tradable_instruments.items():
            base.tradable_instruments[exchange] = [
                _clone_message(item) for item in items
            ]

    return base


def _instrument_key(instrument: Any | None, granularity: Any | None) -> InstrumentKey:
    exchange = instrument.exchange if instrument is not None else "*"
    symbol = instrument.symbol if instrument is not None else "*"
    duration = getattr(granularity, "iso8601_duration", "*") if granularity else "*"
    return (exchange, symbol, duration)


def _match_history(dataset: InMemoryTradingDataset, request: Any) -> List[Any]:
    key = _instrument_key(request.instrument, request.granularity)
    return list(dataset.history.get(key, []))


def _filter_by_time(
    candles: Sequence[Any],
    start: Optional[datetime],
    end: Optional[datetime],
) -> List[Any]:
    result: List[Any] = []
    for candle in candles:
        candle_dt = candle.open_time.ToDatetime().replace(tzinfo=None)
        if start and candle_dt < start:
            continue
        if end and candle_dt >= end:
            continue
        result.append(candle)
    return result


def _timestamp_to_datetime(ts: Optional[timestamp_pb2.Timestamp]) -> Optional[datetime]:
    if ts is None:
        return None
    if ts.seconds == 0 and ts.nanos == 0:
        return None
    return ts.ToDatetime().replace(tzinfo=None)


def _snapshot_for(dataset: InMemoryTradingDataset, request: Any) -> List[Any]:
    key = _instrument_key(request.instrument, request.granularity)
    return list(dataset.stream_snapshots.get(key, []))


def _increments_for(dataset: InMemoryTradingDataset, request: Any) -> List[Any]:
    key = _instrument_key(request.instrument, request.granularity)
    return list(dataset.stream_increments.get(key, []))


def _decision_snapshot(dataset: InMemoryTradingDataset, request: Any) -> List[Any]:
    snapshot = list(dataset.decision_snapshot)
    limit = getattr(request, "limit", 0) or 0
    if limit > 0 and len(snapshot) > limit:
        return snapshot[-limit:]
    return snapshot


def _decision_increments(dataset: InMemoryTradingDataset) -> List[Any]:
    return list(dataset.decision_increments)


def _apply_cycle_metrics(message: Any, metrics: Mapping[str, float]) -> None:
    if not isinstance(metrics, Mapping):
        return
    container = getattr(getattr(message, "cycle_metrics", None), "values", None)
    if container is None:
        return
    try:
        container.clear()
    except AttributeError:
        pass
    for key, value in metrics.items():
        try:
            container[str(key)] = float(value)
        except (TypeError, ValueError):
            continue


def _parse_journal_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _extract_decision_filters(filters) -> Dict[str, Any]:
    if filters is None:
        return {}

    extracted: Dict[str, Any] = {}

    def _collect(name: str) -> List[str]:
        values = getattr(filters, name, [])
        return [str(value).strip() for value in values if str(value).strip()]

    def _assign(keys: Sequence[str], values: List[str]) -> None:
        if not values:
            return
        normalized = {value.lower() for value in values}
        if not normalized:
            return
        for key in keys:
            extracted[key] = normalized

    _assign(("event", "events"), _collect("events"))
    _assign(("strategy", "strategies"), _collect("strategies"))
    _assign(("symbol", "symbols"), _collect("symbols"))
    _assign(("status", "statuses"), _collect("statuses"))
    _assign(("side", "sides"), _collect("sides"))
    _assign(("environment",), _collect("environments"))
    _assign(("portfolio",), _collect("portfolios"))
    _assign(("risk_profile", "riskProfile"), _collect("risk_profiles"))

    if hasattr(filters, "HasField") and filters.HasField("since"):
        extracted["since"] = filters.since.ToDatetime().replace(tzinfo=timezone.utc)
    if hasattr(filters, "HasField") and filters.HasField("until"):
        extracted["until"] = filters.until.ToDatetime().replace(tzinfo=timezone.utc)

    return extracted


def _filter_decision_entries(records: Sequence[Any], filters: Dict[str, Any]) -> List[Any]:
    if not filters:
        return list(records)

    def _match(entry_value: Optional[str], expected: Optional[Sequence[str]]) -> bool:
        if not expected:
            return True
        if entry_value is None:
            return False
        return entry_value.strip().lower() in expected

    since = filters.get("since")
    until = filters.get("until")
    expected_event = filters.get("event")
    expected_strategy = filters.get("strategy")
    expected_symbol = filters.get("symbol")
    expected_status = filters.get("status")
    expected_side = filters.get("side")
    expected_environment = filters.get("environment")
    expected_portfolio = filters.get("portfolio")
    expected_risk = filters.get("risk_profile")

    result: List[Any] = []
    for entry in records:
        fields = getattr(entry, "fields", {})
        timestamp_value = fields.get("timestamp") if isinstance(fields, MutableMapping) else None
        parsed_timestamp = _parse_journal_timestamp(timestamp_value)
        if since and (parsed_timestamp is None or parsed_timestamp < since):
            continue
        if until and (parsed_timestamp is None or parsed_timestamp >= until):
            continue

        if not _match(fields.get("event") if isinstance(fields, MutableMapping) else None, expected_event):
            continue
        if not _match(fields.get("strategy") if isinstance(fields, MutableMapping) else None, expected_strategy):
            continue
        if not _match(fields.get("symbol") if isinstance(fields, MutableMapping) else None, expected_symbol):
            continue
        if not _match(fields.get("status") if isinstance(fields, MutableMapping) else None, expected_status):
            continue
        if not _match(fields.get("side") if isinstance(fields, MutableMapping) else None, expected_side):
            continue
        if not _match(fields.get("environment") if isinstance(fields, MutableMapping) else None, expected_environment):
            continue
        if not _match(fields.get("portfolio") if isinstance(fields, MutableMapping) else None, expected_portfolio):
            continue
        if not _match(fields.get("risk_profile") if isinstance(fields, MutableMapping) else None, expected_risk):
            continue

        result.append(entry)

    return result


def _clone_message(message: Any) -> Any:
    clone = type(message)()
    clone.CopyFrom(message)
    return clone


def _context_is_active(context) -> bool:
    if context is None:
        return True
    is_active = getattr(context, "is_active", None)
    if callable(is_active):
        try:
            return bool(is_active())
        except Exception:  # pragma: no cover - defensywne
            return True
    return True


def _sleep_with_context(duration: float, context) -> None:
    if duration <= 0:
        return
    end_time = time.monotonic() + duration
    while True:
        if not _context_is_active(context):
            return
        remaining = end_time - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.1))


class _MarketDataService:
    def __init__(
        self,
        dataset: InMemoryTradingDataset,
        *,
        repeat_streams: bool = False,
        stream_interval: float = 0.0,
    ) -> None:
        trading_pb2, _ = _ensure_stubs_loaded()
        self._dataset = dataset
        self._history_response_cls = trading_pb2.GetOhlcvHistoryResponse
        self._snapshot_cls = trading_pb2.StreamOhlcvSnapshot
        self._increment_cls = trading_pb2.StreamOhlcvIncrement
        self._update_cls = trading_pb2.StreamOhlcvUpdate
        self._list_response_cls = trading_pb2.ListTradableInstrumentsResponse
        self._repeat_streams = repeat_streams
        self._stream_interval = max(0.0, stream_interval)

    def GetOhlcvHistory(self, request, context):  # noqa: N802
        candles = _match_history(self._dataset, request)
        start = _timestamp_to_datetime(request.start_time if request.HasField("start_time") else None)
        end = _timestamp_to_datetime(request.end_time if request.HasField("end_time") else None)
        filtered = _filter_by_time(candles, start, end)

        limit = request.limit or 0
        has_more = False
        next_start = None
        if limit and len(filtered) > limit:
            has_more = True
            next_start = filtered[limit].open_time
            filtered = filtered[:limit]
        elif filtered:
            next_start = filtered[-1].open_time

        return self._history_response_cls(
            candles=filtered,
            has_more=has_more,
            next_start_time=next_start,
        )

    def StreamOhlcv(self, request, context):  # noqa: N802
        if request.deliver_snapshots:
            snapshot = _snapshot_for(self._dataset, request)
            if snapshot:
                cloned_snapshot = [_clone_message(candle) for candle in snapshot]
                yield self._update_cls(
                    snapshot=self._snapshot_cls(candles=cloned_snapshot)
                )

        increments = _increments_for(self._dataset, request)
        if not increments:
            return

        for candle in increments:
            yield self._build_increment_update(candle)

        if not self._repeat_streams:
            return

        while _context_is_active(context):
            for candle in increments:
                if self._stream_interval:
                    _sleep_with_context(self._stream_interval, context)
                    if not _context_is_active(context):
                        return
                yield self._build_increment_update(candle)
            if not self._stream_interval:
                # brak interwału – natychmiast powtarzaj kolejne cykle
                continue
            if not _context_is_active(context):
                return

    def ListTradableInstruments(self, request, context):  # noqa: N802
        exchange = (request.exchange or "").strip().upper()
        entries = self._dataset.list_tradable_instruments(exchange)
        response = self._list_response_cls()
        response.instruments.extend(entries)
        return response

    def _build_increment_update(self, candle):
        clone = _clone_message(candle)
        increment = self._increment_cls()
        increment.candle.CopyFrom(clone)
        return self._update_cls(increment=increment)


class _OrderService:
    def __init__(self) -> None:
        trading_pb2, _ = _ensure_stubs_loaded()
        self._response_cls = trading_pb2.SubmitOrderResponse
        self._cancel_cls = trading_pb2.CancelOrderResponse
        self._status = trading_pb2.OrderStatus
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._orders: Dict[str, Any] = {}

    def SubmitOrder(self, request, context):  # noqa: N802
        with self._lock:
            order_id = f"SIM-{next(self._counter):06d}"
            self._orders[order_id] = request
        return self._response_cls(
            order_id=order_id,
            external_order_id=f"{order_id}-EXT",
            status=self._status.ORDER_STATUS_ACCEPTED,
            violations=[],
        )

    def CancelOrder(self, request, context):  # noqa: N802
        with self._lock:
            if request.order_id and request.order_id in self._orders:
                self._orders.pop(request.order_id, None)
                return self._cancel_cls(
                    status=self._status.ORDER_STATUS_ACCEPTED,
                    message="Order cancelled",
                )
        return self._cancel_cls(
            status=self._status.ORDER_STATUS_REJECTED,
            message="Order not found",
        )


class _RiskService:
    def __init__(self, dataset: InMemoryTradingDataset) -> None:
        trading_pb2, _ = _ensure_stubs_loaded()
        self._dataset = dataset
        self._risk_cls = trading_pb2.RiskState

    def _states_for(self, request) -> List[Any]:
        instrument = request.instrument if request.HasField("instrument") else None
        key = _instrument_key(instrument, None)
        if key in self._dataset.risk_states:
            return self._dataset.risk_states[key]
        return self._dataset.risk_states.get(None, [])

    def GetRiskState(self, request, context):  # noqa: N802
        states = self._states_for(request)
        if not states:
            return self._risk_cls()
        return states[-1]

    def StreamRiskState(self, request, context):  # noqa: N802
        for state in self._states_for(request):
            yield state


class _MetricsService:
    def __init__(self, dataset: InMemoryTradingDataset) -> None:
        trading_pb2, _ = _ensure_stubs_loaded()
        self._dataset = dataset
        self._ack_cls = trading_pb2.MetricsAck

    def StreamMetrics(self, request, context):  # noqa: N802
        for snapshot in self._dataset.metrics:
            yield snapshot

    def PushMetrics(self, request, context):  # noqa: N802
        self._dataset.metrics.append(request)
        return self._ack_cls(accepted=True)


class _RuntimeService:
    def __init__(
        self,
        dataset: InMemoryTradingDataset,
        *,
        repeat_streams: bool = False,
        stream_interval: float = 0.0,
    ) -> None:
        trading_pb2, _ = _ensure_stubs_loaded()
        self._dataset = dataset
        self._snapshot_cls = trading_pb2.StreamDecisionsSnapshot
        self._increment_cls = trading_pb2.StreamDecisionsIncrement
        self._update_cls = trading_pb2.StreamDecisionsUpdate
        self._repeat_streams = repeat_streams
        self._stream_interval = max(0.0, stream_interval)

    def StreamDecisions(self, request, context):  # noqa: N802
        if not request.skip_snapshot:
            snapshot = _decision_snapshot(self._dataset, request)
            if snapshot:
                cloned = [_clone_message(entry) for entry in snapshot]
                update = self._update_cls(
                    snapshot=self._snapshot_cls(records=cloned)
                )
                _apply_cycle_metrics(update, self._dataset.decision_cycle_metrics)
                yield update

        increments = _decision_increments(self._dataset)
        if increments:
            for entry in increments:
                increment = self._increment_cls(record=_clone_message(entry))
                update = self._update_cls(increment=increment)
                _apply_cycle_metrics(update, self._dataset.decision_cycle_metrics)
                yield update
        if not self._repeat_streams or not increments:
            return

        while _context_is_active(context):
            for entry in increments:
                if self._stream_interval:
                    _sleep_with_context(self._stream_interval, context)
                    if not _context_is_active(context):
                        return
                increment = self._increment_cls(record=_clone_message(entry))
                update = self._update_cls(increment=increment)
                _apply_cycle_metrics(update, self._dataset.decision_cycle_metrics)
                yield update

    def ListDecisions(self, request, context):  # noqa: N802
        del context
        trading_pb2, _ = _ensure_stubs_loaded()
        limit = int(getattr(request, "limit", 0) or 0)
        cursor = int(getattr(request, "cursor", 0) or 0)
        if cursor < 0:
            cursor = 0

        records = list(self._dataset.decision_snapshot)
        records.extend(self._dataset.decision_increments)

        filters = _extract_decision_filters(getattr(request, "filters", None))
        filtered = _filter_decision_entries(records, filters)

        total = len(filtered)
        start_index = min(cursor, total)
        if limit and limit > 0:
            end_index = min(total, start_index + limit)
        else:
            end_index = total

        response = trading_pb2.ListDecisionsResponse()
        response.cursor = end_index
        response.total = total
        response.has_more = end_index < total
        for entry in filtered[start_index:end_index]:
            response.records.append(_clone_message(entry))
        _apply_cycle_metrics(response, self._dataset.decision_cycle_metrics)
        return response


class _HealthService:
    def __init__(self, dataset: InMemoryTradingDataset) -> None:
        trading_pb2, _ = _ensure_stubs_loaded()
        self._dataset = dataset
        self._health_cls = trading_pb2.HealthCheckResponse

    def Check(self, request, context):  # noqa: N802
        if self._dataset.health is not None:
            return self._dataset.health
        ts = timestamp_pb2.Timestamp()
        ts.FromDatetime(datetime.utcnow())
        return self._health_cls(version="stub", git_commit="dev", started_at=ts)


class TradingStubServer(GrpcServerLifecycleMixin):
    """Lekki serwer gRPC zgodny z kontraktem tradingowym."""

    def __init__(
        self,
        dataset: InMemoryTradingDataset | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        max_workers: int = 8,
        *,
        stream_repeat: bool = False,
        stream_interval: float = 0.0,
    ) -> None:
        trading_pb2, trading_pb2_grpc = _ensure_stubs_loaded()
        self._dataset = dataset or build_default_dataset()
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        trading_pb2_grpc.add_MarketDataServiceServicer_to_server(
            _MarketDataService(
                self._dataset,
                repeat_streams=stream_repeat,
                stream_interval=stream_interval,
            ),
            self._server,
        )
        trading_pb2_grpc.add_OrderServiceServicer_to_server(_OrderService(), self._server)
        trading_pb2_grpc.add_RiskServiceServicer_to_server(
            _RiskService(self._dataset), self._server
        )
        trading_pb2_grpc.add_MetricsServiceServicer_to_server(
            _MetricsService(self._dataset), self._server
        )
        trading_pb2_grpc.add_RuntimeServiceServicer_to_server(
            _RuntimeService(
                self._dataset,
                repeat_streams=stream_repeat,
                stream_interval=stream_interval,
            ),
            self._server,
        )
        trading_pb2_grpc.add_HealthServiceServicer_to_server(
            _HealthService(self._dataset), self._server
        )
        self._address = f"{host}:{port}"
        bound_port = self._server.add_insecure_port(self._address)
        if bound_port == 0:
            raise RuntimeError("Nie udało się otworzyć portu dla stub serwera")
        if port == 0:
            self._address = f"{host}:{bound_port}"
        self.stream_repeat = stream_repeat
        self.stream_interval = max(0.0, stream_interval)

    @property
    def address(self) -> str:
        return self._address

    def start(self) -> None:
        self._server.start()

    @property
    def performance_guard(self) -> Dict[str, Any]:
        """Zwraca aktualną konfigurację performance guard."""

        return dict(self._dataset.performance_guard)

    def __enter__(self) -> "TradingStubServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def build_default_dataset() -> InMemoryTradingDataset:
    trading_pb2, _ = _ensure_stubs_loaded()
    dataset = InMemoryTradingDataset()

    instrument = trading_pb2.Instrument(
        exchange="BINANCE",
        symbol="BTC/USDT",
        venue_symbol="BTCUSDT",
        quote_currency="USDT",
        base_currency="BTC",
    )
    granularity = trading_pb2.CandleGranularity(iso8601_duration="PT1M")
    base_ts = _make_timestamp("2024-01-01T00:00:00Z")
    candles = [
        trading_pb2.OhlcvCandle(
            instrument=instrument,
            open_time=_offset_timestamp(base_ts, minutes=i),
            open=42000.0 + i * 20,
            high=42100.0 + i * 20,
            low=41950.0 + i * 20,
            close=42080.0 + i * 20,
            volume=10.0 + i,
            closed=True,
            granularity=granularity,
            sequence=i + 1,
        )
        for i in range(3)
    ]
    dataset.add_history(instrument, granularity, candles)
    dataset.set_stream_data(
        instrument,
        granularity,
        snapshot_candles=candles[:2],
        increments=[candles[-1]],
    )

    dataset.add_risk_states(
        None,
        [
            trading_pb2.RiskState(
                profile=trading_pb2.RiskProfile.RISK_PROFILE_BALANCED,
                portfolio_value=1_000_000.0,
                current_drawdown=0.07,
                max_daily_loss=0.05,
                used_leverage=1.2,
                limits=[
                    trading_pb2.ExposureLimit(
                        code="MAX_POSITION",
                        max_value=100_000.0,
                        current_value=20_000.0,
                        threshold_value=80_000.0,
                    ),
                    trading_pb2.ExposureLimit(
                        code="DAILY_LOSS",
                        max_value=90_000.0,
                        current_value=70_000.0,
                        threshold_value=80_000.0,
                    ),
                    trading_pb2.ExposureLimit(
                        code="MARGIN_USAGE",
                        max_value=120_000.0,
                        current_value=95_000.0,
                        threshold_value=90_000.0,
                    )
                ],
                generated_at=_make_timestamp("2024-01-01T00:05:00Z"),
            )
        ],
    )

    dataset.set_metrics(
        [
            trading_pb2.MetricsSnapshot(
                generated_at=_make_timestamp("2024-01-01T00:05:00Z"),
                event_to_frame_p95_ms=110.0,
                fps=60.0,
                cpu_utilization=12.0,
                gpu_utilization=20.0,
                ram_megabytes=180.0,
                dropped_frames=0,
                processed_messages_per_second=2500,
                notes="default dataset",
            )
        ]
    )

    dataset.set_decision_stream(
        snapshot=[
            trading_pb2.DecisionRecordEntry(
                fields={
                    "event": "order_submitted",
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "environment": "stub",
                    "portfolio": "alpha",
                    "risk_profile": "balanced",
                    "strategy": "trend_follow",
                    "schedule": "auto",
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "status": "submitted",
                    "quantity": "0.10",
                    "price": "42000.0",
                    "decision_state": "trade",
                    "decision_should_trade": "true",
                    "decision_confidence": "0.82",
                    "ai_model": "stub-transformer",
                    "ai_version": "1.0",
                    "market_regime_state": "bull",
                    "market_regime_risk_level": "moderate",
                    "latency_ms": "45.0",
                }
            ),
            trading_pb2.DecisionRecordEntry(
                fields={
                    "event": "order_filled",
                    "timestamp": "2024-01-01T00:00:02+00:00",
                    "environment": "stub",
                    "portfolio": "alpha",
                    "risk_profile": "balanced",
                    "strategy": "trend_follow",
                    "schedule": "auto",
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "status": "filled",
                    "quantity": "0.10",
                    "price": "42001.5",
                    "decision_state": "trade",
                    "decision_should_trade": "true",
                    "decision_signal_strength": "0.74",
                    "ai_model": "stub-transformer",
                    "ai_version": "1.0",
                    "market_regime_state": "bull",
                    "market_regime_risk_level": "moderate",
                    "latency_ms": "38.0",
                }
            ),
        ],
        increments=[
            trading_pb2.DecisionRecordEntry(
                fields={
                    "event": "risk_update",
                    "timestamp": "2024-01-01T00:05:00+00:00",
                    "environment": "stub",
                    "portfolio": "alpha",
                    "risk_profile": "balanced",
                    "strategy": "trend_follow",
                    "schedule": "auto",
                    "status": "ok",
                    "decision_state": "monitor",
                    "decision_should_trade": "false",
                    "risk_flags": "latency_watch",
                    "latency_ms": "52.0",
                }
            )
        ],
    )

    dataset.set_decision_cycle_metrics(
        {
            "cycles_total": 128.0,
            "strategy_switch_total": 7.0,
            "guardrail_blocks_total": 2.0,
        }
    )

    dataset.health = trading_pb2.HealthCheckResponse(
        version="stub-1.0",
        git_commit="0000000",
        started_at=_make_timestamp("2024-01-01T00:00:00Z"),
    )

    dataset.performance_guard.update(
        {
            "fps_target": 60,
            "reduce_motion_after_seconds": 1.0,
            "jank_threshold_ms": 18.0,
            "max_overlay_count": 3,
        }
    )

    dataset.set_tradable_instruments(
        "BINANCE",
        [
            trading_pb2.TradableInstrumentMetadata(
                instrument=_clone_message(instrument),
                price_step=0.1,
                amount_step=0.001,
                min_notional=10.0,
                min_amount=0.0005,
                max_amount=5.0,
                min_price=1.0,
                max_price=1_000_000.0,
            ),
            trading_pb2.TradableInstrumentMetadata(
                instrument=trading_pb2.Instrument(
                    exchange="BINANCE",
                    symbol="ETH/USDT",
                    venue_symbol="ETHUSDT",
                    quote_currency="USDT",
                    base_currency="ETH",
                ),
                price_step=0.01,
                amount_step=0.0001,
                min_notional=5.0,
                min_amount=0.001,
                max_amount=250.0,
                min_price=0.5,
                max_price=500_000.0,
            ),
        ],
    )

    return dataset


def load_dataset_from_yaml(path: str | Path) -> InMemoryTradingDataset:
    data = yaml.safe_load(Path(path).read_text()) or {}
    trading_pb2, _ = _ensure_stubs_loaded()
    dataset = InMemoryTradingDataset()

    for item in data.get("market_data", []):
        instrument = trading_pb2.Instrument(**item["instrument"])
        granularity = trading_pb2.CandleGranularity(
            iso8601_duration=item.get("granularity", "PT1M")
        )
        candles = [
            trading_pb2.OhlcvCandle(
                instrument=instrument,
                open_time=_make_timestamp(candle["open_time"]),
                open=candle["open"],
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                volume=candle.get("volume", 0.0),
                closed=candle.get("closed", True),
                granularity=granularity,
                sequence=candle.get("sequence", idx + 1),
            )
            for idx, candle in enumerate(item.get("candles", []))
        ]
        dataset.add_history(instrument, granularity, candles)
        if "stream" in item:
            stream_cfg = item["stream"]
            snapshot = [
                trading_pb2.OhlcvCandle(
                    instrument=instrument,
                    open_time=_make_timestamp(candle["open_time"]),
                    open=candle["open"],
                    high=candle["high"],
                    low=candle["low"],
                    close=candle["close"],
                    volume=candle.get("volume", 0.0),
                    closed=candle.get("closed", True),
                    granularity=granularity,
                    sequence=candle.get("sequence", idx + 1),
                )
                for idx, candle in enumerate(stream_cfg.get("snapshot", []))
            ]
            increments = [
                trading_pb2.OhlcvCandle(
                    instrument=instrument,
                    open_time=_make_timestamp(candle["open_time"]),
                    open=candle["open"],
                    high=candle["high"],
                    low=candle["low"],
                    close=candle["close"],
                    volume=candle.get("volume", 0.0),
                    closed=candle.get("closed", False),
                    granularity=granularity,
                    sequence=candle.get("sequence", idx + 1),
                )
                for idx, candle in enumerate(stream_cfg.get("increments", []))
            ]
            dataset.set_stream_data(instrument, granularity, snapshot, increments)

    for risk_item in data.get("risk_states", []):
        instrument_cfg = risk_item.get("instrument")
        instrument = (
            trading_pb2.Instrument(**instrument_cfg) if instrument_cfg is not None else None
        )
        states = [
            trading_pb2.RiskState(
                profile=getattr(
                    trading_pb2.RiskProfile,
                    state.get("profile", "RISK_PROFILE_BALANCED"),
                ),
                portfolio_value=state.get("portfolio_value", 0.0),
                current_drawdown=state.get("current_drawdown", 0.0),
                max_daily_loss=state.get("max_daily_loss", 0.0),
                used_leverage=state.get("used_leverage", 0.0),
                limits=[
                    trading_pb2.ExposureLimit(
                        code=limit["code"],
                        max_value=limit.get("max_value", 0.0),
                        current_value=limit.get("current_value", 0.0),
                        threshold_value=limit.get("threshold_value", 0.0),
                    )
                    for limit in state.get("limits", [])
                ],
                generated_at=_make_timestamp(state.get("generated_at", "1970-01-01T00:00:00Z")),
            )
            for state in risk_item.get("states", [])
        ]
        dataset.add_risk_states(instrument, states)

    if "metrics" in data:
        snapshots = [
            trading_pb2.MetricsSnapshot(
                generated_at=_make_timestamp(item.get("generated_at", "1970-01-01T00:00:00Z")),
                event_to_frame_p95_ms=item.get("event_to_frame_p95_ms", 0.0),
                fps=item.get("fps", 0.0),
                cpu_utilization=item.get("cpu_utilization", 0.0),
                gpu_utilization=item.get("gpu_utilization", 0.0),
                ram_megabytes=item.get("ram_megabytes", 0.0),
                dropped_frames=item.get("dropped_frames", 0),
                processed_messages_per_second=item.get("processed_messages_per_second", 0),
                notes=item.get("notes", ""),
            )
            for item in data.get("metrics", [])
        ]
        dataset.set_metrics(snapshots)

    if "health" in data:
        health_cfg = data["health"]
        dataset.health = trading_pb2.HealthCheckResponse(
            version=health_cfg.get("version", "stub"),
            git_commit=health_cfg.get("git_commit", "dev"),
            started_at=_make_timestamp(health_cfg.get("started_at", "1970-01-01T00:00:00Z")),
        )

    if "performance_guard" in data:
        guard_cfg = data["performance_guard"]
        if isinstance(guard_cfg, dict):
            dataset.performance_guard.update(guard_cfg)

    listings: Dict[str, list[Any]] = defaultdict(list)
    for item in data.get("tradable_instruments", []):
        exchange = (item.get("exchange") or "*").upper()
        instrument_cfg = item.get("instrument")
        if not isinstance(instrument_cfg, dict):
            continue
        try:
            instrument = trading_pb2.Instrument(**instrument_cfg)
        except TypeError:
            continue
        metadata = trading_pb2.TradableInstrumentMetadata(
            instrument=instrument,
            price_step=float(item.get("price_step", 0.0) or 0.0),
            amount_step=float(item.get("amount_step", 0.0) or 0.0),
            min_notional=float(item.get("min_notional", 0.0) or 0.0),
            min_amount=float(item.get("min_amount", 0.0) or 0.0),
            max_amount=float(item.get("max_amount", 0.0) or 0.0),
            min_price=float(item.get("min_price", 0.0) or 0.0),
            max_price=float(item.get("max_price", 0.0) or 0.0),
        )
        listings[exchange].append(metadata)

    for exchange, instruments in listings.items():
        dataset.set_tradable_instruments(exchange, instruments)

    return dataset


def _make_timestamp(value) -> timestamp_pb2.Timestamp:
    ts = timestamp_pb2.Timestamp()
    if isinstance(value, timestamp_pb2.Timestamp):
        ts.CopyFrom(value)
    elif isinstance(value, datetime):
        ts.FromDatetime(value)
    else:
        ts.FromJsonString(str(value))
    return ts


def _offset_timestamp(base: timestamp_pb2.Timestamp, minutes: int) -> timestamp_pb2.Timestamp:
    dt = base.ToDatetime() + timedelta(minutes=minutes)
    ts = timestamp_pb2.Timestamp()
    ts.FromDatetime(dt)
    return ts
