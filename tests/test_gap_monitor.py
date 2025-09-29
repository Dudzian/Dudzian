from __future__ import annotations
from datetime import datetime, timedelta, timezone

from bot_core.alerts import AlertMessage
from bot_core.data.ohlcv.audit import GapAuditRecord
from bot_core.data.ohlcv.backfill import BackfillSummary
from bot_core.data.ohlcv.gap_monitor import DataGapIncidentTracker, GapAlertPolicy


class DummyRouter:
    def __init__(self) -> None:
        self.messages: list[AlertMessage] = []

    def dispatch(self, message: AlertMessage) -> None:
        self.messages.append(message)


class DummyAuditLogger:
    def __init__(self) -> None:
        self.records: list[GapAuditRecord] = []

    def log(self, record: GapAuditRecord) -> None:
        self.records.append(record)


def _summary(symbol: str, interval: str, end: int) -> BackfillSummary:
    return BackfillSummary(
        symbol=symbol,
        interval=interval,
        requested_start=end - 600_000,
        requested_end=end,
        fetched_candles=0,
        skipped_candles=0,
    )


def test_gap_tracker_sends_warning_on_threshold_exceeded() -> None:
    router = DummyRouter()
    audit = DummyAuditLogger()
    now_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    metadata = {
        "last_timestamp::BTCUSDT::1h": str(now_ms - 90 * 60_000),
        "row_count::BTCUSDT::1h": "1200",
    }
    policy = GapAlertPolicy(warning_gap_minutes={"1h": 60})
    tracker = DataGapIncidentTracker(
        router=router,
        metadata_provider=lambda: metadata,
        policy=policy,
        environment_name="test-env",
        exchange="binance_spot",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
        audit_logger=audit,
    )

    tracker.handle_summaries(interval="1h", summaries=[_summary("BTCUSDT", "1h", now_ms)], as_of_ms=now_ms)

    assert len(router.messages) == 1
    message = router.messages[0]
    assert message.severity == "warning"
    assert message.context["symbol"] == "BTCUSDT"
    assert message.context["interval"] == "1h"

    assert len(audit.records) == 1
    record = audit.records[0]
    assert record.status == "warning"
    assert record.symbol == "BTCUSDT"
    assert record.interval == "1h"


def test_gap_tracker_throttles_repeated_warnings() -> None:
    router = DummyRouter()
    audit = DummyAuditLogger()
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    clock_times = [base_time, base_time + timedelta(minutes=2)]

    def clock() -> datetime:
        return clock_times.pop(0)

    now_ms = int((base_time + timedelta(hours=2)).timestamp() * 1000)
    metadata = {
        "last_timestamp::ETHUSDT::1h": str(now_ms - 180 * 60_000),
        "row_count::ETHUSDT::1h": "900",
    }
    policy = GapAlertPolicy(warning_gap_minutes={"1h": 60}, warning_throttle_minutes=5)
    tracker = DataGapIncidentTracker(
        router=router,
        metadata_provider=lambda: metadata,
        policy=policy,
        environment_name="paper",
        exchange="binance_spot",
        clock=clock,
        audit_logger=audit,
    )

    tracker.handle_summaries(interval="1h", summaries=[_summary("ETHUSDT", "1h", now_ms)], as_of_ms=now_ms)
    tracker.handle_summaries(interval="1h", summaries=[_summary("ETHUSDT", "1h", now_ms)], as_of_ms=now_ms)

    assert len(router.messages) == 1
    assert router.messages[0].severity == "warning"
    assert [record.status for record in audit.records] == ["warning", "warning_suppressed"]


def test_gap_tracker_opens_incident_after_repeated_warnings() -> None:
    router = DummyRouter()
    audit = DummyAuditLogger()
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [base_time + timedelta(minutes=idx) for idx in range(3)]

    def clock() -> datetime:
        return times.pop(0)

    now_ms = int((base_time + timedelta(minutes=30)).timestamp() * 1000)
    metadata = {
        "last_timestamp::ETHUSDT::1h": str(now_ms - 180 * 60_000),
        "row_count::ETHUSDT::1h": "600",
    }
    policy = GapAlertPolicy(
        warning_gap_minutes={"1h": 60},
        incident_threshold_count=3,
        incident_window_minutes=10,
        sms_escalation_minutes=15,
    )
    tracker = DataGapIncidentTracker(
        router=router,
        metadata_provider=lambda: metadata,
        policy=policy,
        environment_name="test-env",
        exchange="binance_spot",
        clock=clock,
        audit_logger=audit,
    )

    for _ in range(3):
        tracker.handle_summaries(
            interval="1h",
            summaries=[_summary("ETHUSDT", "1h", now_ms)],
            as_of_ms=now_ms,
        )

    assert [message.severity for message in router.messages] == ["warning", "critical"]
    assert "INCIDENT" in router.messages[-1].title

    statuses = [record.status for record in audit.records]
    assert statuses.count("warning") == 1
    assert statuses.count("warning_suppressed") == 1
    assert statuses[-1] == "incident"


def test_gap_tracker_escalates_sms_and_recovers() -> None:
    router = DummyRouter()
    audit = DummyAuditLogger()
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    clock_times = [
        base_time,                              # warn #1
        base_time + timedelta(minutes=2),       # warn #2 (wciąż w throttlingu)
        base_time + timedelta(minutes=4),       # warn #3 -> incident open
        base_time + timedelta(minutes=20),      # SMS escalate (>15m)
        base_time + timedelta(minutes=40),      # recovery
    ]

    def clock() -> datetime:
        return clock_times.pop(0)

    now_ms = int((base_time + timedelta(minutes=30)).timestamp() * 1000)
    metadata = {
        "last_timestamp::SOLUSDT::15m": str(now_ms - 45 * 60_000),
        "row_count::SOLUSDT::15m": "350",
    }
    policy = GapAlertPolicy(
        warning_gap_minutes={"15m": 10},
        incident_threshold_count=3,
        incident_window_minutes=10,
        sms_escalation_minutes=15,
    )
    tracker = DataGapIncidentTracker(
        router=router,
        metadata_provider=lambda: metadata,
        policy=policy,
        environment_name="test-env",
        exchange="binance_spot",
        clock=clock,
        audit_logger=audit,
    )

    # 3 ostrzeżenia -> incydent
    for _ in range(3):
        tracker.handle_summaries(
            interval="15m",
            summaries=[_summary("SOLUSDT", "15m", now_ms)],
            as_of_ms=now_ms,
        )

    # Upływ czasu aż do eskalacji SMS
    tracker.handle_summaries(
        interval="15m",
        summaries=[_summary("SOLUSDT", "15m", now_ms)],
        as_of_ms=now_ms,
    )

    # Odzyskanie – zaktualizowany last_timestamp
    metadata["last_timestamp::SOLUSDT::15m"] = str(now_ms)
    tracker.handle_summaries(
        interval="15m",
        summaries=[_summary("SOLUSDT", "15m", now_ms)],
        as_of_ms=now_ms,
    )

    assert any("Eskalacja SMS" in msg.title for msg in router.messages)
    assert router.messages[-1].severity == "info"
    assert "Incydent zamknięty" in router.messages[-1].title

    statuses = [record.status for record in audit.records]
    assert "sms_escalated" in statuses
    assert statuses[-1] == "ok"


def test_gap_tracker_audit_records_missing_metadata() -> None:
    router = DummyRouter()
    audit = DummyAuditLogger()
    now_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    metadata: dict[str, str] = {"row_count::BTCPLN::1h": "42"}
    policy = GapAlertPolicy(warning_gap_minutes={})
    tracker = DataGapIncidentTracker(
        router=router,
        metadata_provider=lambda: metadata,
        policy=policy,
        environment_name="paper",
        exchange="zonda_spot",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
        audit_logger=audit,
    )

    tracker.handle_summaries(interval="1h", summaries=[_summary("BTCPLN", "1h", now_ms)], as_of_ms=now_ms)

    assert router.messages[-1].severity == "critical"
    assert audit.records[-1].status == "missing_metadata"
